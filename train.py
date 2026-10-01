import torch
import torch.nn.functional as F

from model import GPT
from get_batch import get_batch

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
    weight_decay=0.01,
)

adamw_optimizer = torch.optim.AdamW(
    [
        {
            "params": adamw_decay_params,
            "weight_decay": 0.01,
        },
        {
            "params": adamw_no_decay_params,
            "weight_decay": 0.0,
        },
    ],
    lr=3e-4,
    betas=(0.9, 0.95),
)