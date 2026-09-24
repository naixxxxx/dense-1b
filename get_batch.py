import numpy as np
import torch

SEQ_LEN = 4096

tokens = np.memmap(
    "data/en_000.bin",
    dtype=np.uint32,
    mode="r",
)

pos = 0


def get_batch():
    global pos

    if pos + SEQ_LEN >= len(tokens):
        return None

    x = np.asarray(
        tokens[pos : pos + SEQ_LEN],
        dtype=np.int64,
    )

    y = np.asarray(
        tokens[pos + 1 : pos + SEQ_LEN + 1],
        dtype=np.int64,
    )

    pos += SEQ_LEN

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x, y