import torch
import torch.nn.functional as F

from model import GPT
from get_batch import get_batch


device = "cuda"


model = GPT(
    vocab_size=151936,
    d_model=1280,
    d_ff=3840,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,
    head_dim=128,
).to(device)



# Muon
lr = 2e-2
momentum = 0.95
nesterov = True

# AdamW
lr = 3e-4
betas = (0.9, 0.95)
eps = 1e-8

# WSD
warmup = 0.02
stable = 0.88
decay = 0.1
min_lr_ratio = 0.00


muon_params = []
adamw_decay_params = []
adamw_no_decay_params = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue

    if (
        p.ndim == 2
        and min(p.shape) > 1
        and not name.startswith("embedding.")
        and not name.startswith("lm_head.")
    ):
        muon_params.append(p)

    elif (
        name.startswith("embedding.")
        or getattr(p, "_no_weight_decay", False)
    ):
        adamw_no_decay_params.append(p)

    else:
        adamw_decay_params.append(p)