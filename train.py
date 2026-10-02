import torch
import torch.nn.functional as F

from model import GPT
import get_batch as data # data.TOTAL_TOKENS data.tokens_seen

device = "cuda"

model = GPT(
    vocab_size=151936,
    d_model=128,
    d_ff=384,
    num_layers=8,
    num_heads=4,
    num_kv_heads=1,
    head_dim=32,
).to(device)


muon_params = []
adamw_decay_params = []
adamw_no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    if name.startswith("embedding.") or name.startswith("lm_head."):
        adamw_decay_params.append(param)

    elif param.ndim == 2:
        muon_params.append(param)

    elif param.ndim >= 2:
        adamw_decay_params.append(param)

    else:
        adamw_no_decay_params.append(param)


muon_optimizer = torch.optim.Muon(
    muon_params,
    lr=0.02,
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
    lr=3e-4,
    betas=(0.9, 0.95),
)

while True:
    batch = data.get_batch()
    if batch is None:
        break

    x, y = batch
    x = x.unsqueeze(0).to(device)  # [4096] → [1, 4096]
    y = y.unsqueeze(0).to(device)

    muon_optimizer.zero_grad()
    adamw_optimizer.zero_grad()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)  # [1, 4096, vocab_size]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    muon_optimizer.step()
    adamw_optimizer.step()