import torch
import torch.nn.functional as F

from model import GPT
from get_batch import get_batch


device = "cuda"


model = GPT(
    vocab_size=151936,
    d_model=64,
    d_ff=128,
    num_layers=1,
    num_heads=1,
    num_kv_heads=1,
    head_dim=64,
).to(device)

model.train()


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-3,
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