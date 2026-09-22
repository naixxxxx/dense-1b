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


logits = model(input_ids)

print("logits:", logits.shape)
print("dtype:", logits.dtype)
print("device:", logits.device)


loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
)

print("loss:", loss.item())


loss.backward()

print("backward passed")