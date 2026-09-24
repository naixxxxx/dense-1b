import os
import numpy as np

path = "data/en_000.bin"

file_size = os.path.getsize(path)
bytes_per_token = np.dtype(np.uint32).itemsize

assert file_size % bytes_per_token == 0

num_tokens = file_size // bytes_per_token

print(f"file size: {file_size / 1024**3:.3f} GiB")
print(f"tokens: {num_tokens:,}")
print(f"tokens: {num_tokens / 1e9:.3f}B")