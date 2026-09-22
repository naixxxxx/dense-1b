import torch
import torch.nn.functional as F

from model import GPT


device = "cuda"
dtype = torch.bfloat16

vocab_size = 10000
d_model = 512
d_ff = 1408
num_layers = 4
num_heads = 8
num_kv_heads = 2
head_dim = 64

batch_size = 2
seq_len = 128


model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    d_ff=d_ff,
    num_layers=num_layers,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
).to(device=device, dtype=dtype)


input_ids = torch.randint(
    0,
    vocab_size,
    (batch_size, seq_len),
    device=device,
)

labels = torch.randint(
    0,
    vocab_size,
    (batch_size, seq_len),
    device=device,
)


# --------------------------------------------------
# Model information
# --------------------------------------------------

num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("parameters:", f"{num_params:,}")
print("trainable:", f"{num_trainable:,}")


# --------------------------------------------------
# Forward
# --------------------------------------------------

logits = model(input_ids)

print("\nforward:")
print("logits shape:", logits.shape)
print("logits dtype:", logits.dtype)
print("logits device:", logits.device)

print("logits has nan:", torch.isnan(logits).any().item())
print("logits has inf:", torch.isinf(logits).any().item())

print("logits mean:", logits.float().mean().item())
print("logits std:", logits.float().std().item())


# --------------------------------------------------
# Loss
# --------------------------------------------------

loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    labels.reshape(-1),
)

print("\nloss:")
print("loss:", loss.item())
print("loss finite:", torch.isfinite(loss).item())


# --------------------------------------------------
# Backward
# --------------------------------------------------

loss.backward()

print("\nbackward passed")


# --------------------------------------------------
# Gradient check
# --------------------------------------------------

missing_grad = []
nan_grad = []
inf_grad = []

total_grad_norm_sq = 0.0

print("\ngradients:")

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    if param.grad is None:
        missing_grad.append(name)
        continue

    grad = param.grad

    if torch.isnan(grad).any():
        nan_grad.append(name)

    if torch.isinf(grad).any():
        inf_grad.append(name)

    grad_norm = grad.float().norm().item()
    total_grad_norm_sq += grad_norm ** 2

    print(f"{name:70s} grad_norm={grad_norm:.6f}")


total_grad_norm = total_grad_norm_sq ** 0.5


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nsummary:")
print("total grad norm:", total_grad_norm)

print("missing gradients:", len(missing_grad))
for name in missing_grad:
    print("  ", name)

print("nan gradients:", len(nan_grad))
for name in nan_grad:
    print("  ", name)

print("inf gradients:", len(inf_grad))
for name in inf_grad:
    print("  ", name)


if (
    torch.isfinite(loss)
    and not torch.isnan(logits).any()
    and not torch.isinf(logits).any()
    and len(nan_grad) == 0
    and len(inf_grad) == 0
):
    print("\nSMOKE TEST PASSED")
else:
    print("\nSMOKE TEST FAILED")