import torch
from fla.layers.kda import KimiDeltaAttention

layer = KimiDeltaAttention(
    hidden_size=512,
    num_heads=8,
    head_dim=64,
).cuda().bfloat16()

x = torch.randn(
    2, 128, 512,
    device="cuda",
    dtype=torch.bfloat16,
)

y, _, _ = layer(x)

print(y.shape)