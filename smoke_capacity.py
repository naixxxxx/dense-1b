import os
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT


parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, default=24)
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=1)
args = parser.parse_args()
if args.batch_size < 1:
    parser.error("--batch-size must be at least 1")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

dist.init_process_group(backend="nccl")


def memory(label):
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    free, total = torch.cuda.mem_get_info(device)

    print(
        f"{label}: "
        f"allocated={allocated:.2f} GiB, "
        f"reserved={reserved:.2f} GiB, "
        f"peak={peak:.2f} GiB, "
        f"GPU used={(total - free) / 1024**3:.2f} GiB",
        flush=True,
    )


stage = "initialization"

try:
    torch.cuda.reset_peak_memory_stats(device)

    model = GPT(
        vocab_size=151936,
        d_model=1280,
        d_ff=3840,
        num_layers=args.layers,
        num_heads=16,
        num_kv_heads=4,
        head_dim=128,
    ).to(device)

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Layers: {args.layers}", flush=True)
    print(f"Batch size per GPU: {args.batch_size}", flush=True)
    print(f"Tokens per GPU per step: {args.batch_size * 4096:,}", flush=True)
    print(f"Parameters: {parameters:,}", flush=True)
    print(f"Parameters (B): {parameters / 1e9:.3f}B", flush=True)
    memory("Model initialized")

    # Match train.py parameter groups.
    muon_params = []
    adamw_decay_params = []
    adamw_no_decay_params = []

    for name, param in model.named_parameters():
        if name.startswith("embedding.") or name.startswith("lm_head."):
            adamw_decay_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        elif param.ndim == 3:
            adamw_decay_params.append(param)
        else:
            adamw_no_decay_params.append(param)

    muon_optimizer = torch.optim.Muon(
        muon_params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.1,
        nesterov=True,
    )

    adamw_optimizer = torch.optim.AdamW(
        [
            {"params": adamw_decay_params, "weight_decay": 0.1},
            {"params": adamw_no_decay_params, "weight_decay": 0.0},
        ],
        lr=3e-4,
        betas=(0.9, 0.95),
    )

    ddp_model = DDP(model, device_ids=[local_rank])
    memory("DDP initialized")

    for step in range(1, args.steps + 1):
        print(f"\n========== STEP {step}/{args.steps} ==========", flush=True)

        stage = f"step {step}: zero_grad"
        muon_optimizer.zero_grad(set_to_none=True)
        adamw_optimizer.zero_grad(set_to_none=True)

        stage = f"step {step}: prepare batch"
        x = torch.randint(0, 151936, (args.batch_size, 4096), device=device)
        y = torch.randint(0, 151936, (args.batch_size, 4096), device=device)

        # Reset the peak only, not the cache or optimizer state.
        torch.cuda.reset_peak_memory_stats(device)
        memory("Before forward")

        stage = f"step {step}: forward"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ddp_model(x, y)
        memory("Forward completed")

        stage = f"step {step}: backward"
        loss.backward()
        memory("Backward completed")

        stage = f"step {step}: gradient clipping"
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        stage = f"step {step}: Muon optimizer"
        muon_optimizer.step()
        memory("Muon step completed")

        stage = f"step {step}: AdamW optimizer"
        adamw_optimizer.step()
        memory("AdamW step completed")

        print(
            f"Step {step} done | "
            f"Loss: {loss.item():.4f} | "
            f"Grad norm: {grad_norm.item():.4f} | "
            f"Peak allocated: "
            f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB",
            flush=True,
        )

    print(f"\nSMOKE TEST: PASS ({args.steps} steps)", flush=True)

except torch.cuda.OutOfMemoryError:
    print(f"\nOUT OF MEMORY during: {stage}", flush=True)
    memory("OOM")
    raise

finally:
    dist.destroy_process_group()
