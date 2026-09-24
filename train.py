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
