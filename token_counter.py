import numpy as np

tokens = np.memmap(
    "data/train_data.bin",
    dtype=np.uint32,
    mode="r",
)

TOTAL_TOKENS = len(tokens)

print(f"Total tokens: {TOTAL_TOKENS:,}")
print(f"Total tokens (B): {TOTAL_TOKENS / 1e9:.3f}B")