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


def get_batch(rank, world_size):
    global tokens_seen

    step_tokens = SEQ_LEN * world_size

    if tokens_seen + step_tokens >= len(tokens):
        return None

    start = tokens_seen + rank * SEQ_LEN

    x = np.asarray(
        tokens[start : start + SEQ_LEN],
        dtype=np.int64,
    )

    y = np.asarray(
        tokens[start + 1 : start + SEQ_LEN + 1],
        dtype=np.int64,
    )

    tokens_seen += step_tokens

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x, y