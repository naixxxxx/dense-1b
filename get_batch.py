import numpy as np
import torch

SEQ_LEN = 4096

tokens = np.memmap(
    "data/en_000.bin",
    dtype=np.uint32,
    mode="r",
)

TOTAL_TOKENS = len(tokens)

tokens_seen = 0


def get_batch():
    global tokens_seen

    if tokens_seen + SEQ_LEN >= len(tokens):
        return None

    x = np.asarray(
        tokens[tokens_seen : tokens_seen + SEQ_LEN],
        dtype=np.int64,
    )

    y = np.asarray(
        tokens[tokens_seen + 1 : tokens_seen + SEQ_LEN + 1],
        dtype=np.int64,
    )

    tokens_seen += SEQ_LEN

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x, y