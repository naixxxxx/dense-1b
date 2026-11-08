import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
import get_batch as data # data.TOTAL_TOKENS data.tokens_seen
from model import GPT
from scheduler import get_wsd_lr_ratio


local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)  
device = torch.device("cuda", local_rank)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()


model = GPT(
   vocab_size=151936,
    d_model=1280,
    d_ff=3840,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,
    head_dim=128,
).to(device)


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


muon_base_lr = 0.02
adamw_base_lr = 3e-4

muon_optimizer = torch.optim.Muon(
    muon_params,
    lr=muon_base_lr,
    momentum=0.95,
    weight_decay=0.1,
    nesterov=True
)

adamw_optimizer = torch.optim.AdamW(
    [
        {
            "params": adamw_decay_params,
            "weight_decay": 0.1,
        },
        {
            "params": adamw_no_decay_params,
            "weight_decay": 0.0,
        },
    ],
    lr=adamw_base_lr,
    betas=(0.9, 0.95),
)


ddp_model = DDP(model, device_ids=[local_rank])

if rank == 0:
    run = wandb.init(
        project="pretrain-dense-1b",
        name="4090-no-vpn",
    )

    run.define_metric("tokens")
    run.define_metric("loss", step_metric="tokens")
    run.define_metric("grad_norm", step_metric="tokens")
    run.define_metric("muon_lr", step_metric="tokens")
    run.define_metric("adamw_lr", step_metric="tokens")

try:
    while True:
        lr_ratio = get_wsd_lr_ratio(data.tokens_seen, data.TOTAL_TOKENS)
        batch = data.get_batch(rank, world_size)
        if batch is None:
            break


        for group in muon_optimizer.param_groups:
            group["lr"] = muon_base_lr * lr_ratio

        for group in adamw_optimizer.param_groups:
            group["lr"] = adamw_base_lr * lr_ratio

        x, y = batch
        x = x.unsqueeze(0).to(device)  # [4096] → [1, 4096]
        y = y.unsqueeze(0).to(device)

        muon_optimizer.zero_grad()
        adamw_optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ddp_model(x, y)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        muon_optimizer.step()
        adamw_optimizer.step()

        loss_mean = loss.detach().clone()
        dist.all_reduce(loss_mean, op=dist.ReduceOp.SUM)
        loss_mean /= world_size

        if rank == 0:
            run.log({
                "tokens": data.tokens_seen,
                "loss": loss_mean.item(),
                "grad_norm": grad_norm.item(),
                "muon_lr": muon_optimizer.param_groups[0]["lr"],
                "adamw_lr": adamw_optimizer.param_groups[0]["lr"],
            })
except KeyboardInterrupt:
    if rank == 0:
        print("Test interrupted; no checkpoint was saved.", flush=True)
finally:
    if rank == 0:
        run.finish()
    dist.destroy_process_group()
