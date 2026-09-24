import torch
import torch.nn.functional as F

from model import GPT
from get_batch import get_batch


device = "cuda"


model = GPT(
    vocab_size=151936,
    d_model=64,
    d_ff=128,
    num_layers=4,
    num_heads=1,
    num_kv_heads=1,
    head_dim=64,
).to(device)

model.train()


adamw_decay = []
adamw_no_decay = []
muon_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    if name.startswith("embedding."):
        adamw_no_decay.append(param)

    elif name.startswith("lm_head."):
        adamw_decay.append(param)

    elif param.ndim == 2:
        muon_params.append(param)

    else:
        adamw_no_decay.append(param)


optimizer_muon = torch.optim.Muon(
    muon_params,
    lr=1e-3,
    momentum=0.95,
    weight_decay=0.1,
)

optimizer_adamw = torch.optim.AdamW(
    [
        {
            "params": adamw_decay,
            "weight_decay": 0.1,
        },
        {
            "params": adamw_no_decay,
            "weight_decay": 0.0,
        },
    ],
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
)

for step in range(100):
    batch = get_batch()

    if batch is None:
        break

    x, y = batch

    # [4096] -> [1, 4096]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    optimizer.zero_grad(set_to_none=True)

    # [1, 4096] -> [1, 4096, vocab_size]
    logits = model(x)

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
    )

    loss.backward()

    optimizer.step()

    print(f"step {step}: loss = {loss.item():.4f}")