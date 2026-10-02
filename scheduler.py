import math

def get_wsd_lr_ratio(
    tokens_seen,
    total_tokens,
    warmup_ratio=0.02,
    stable_ratio=0.88,
):
    progress = tokens_seen / total_tokens

    warmup_end = warmup_ratio
    stable_end = warmup_ratio + stable_ratio

    if progress < warmup_end:
        return progress / warmup_end

    if progress < stable_end:
        return 1.0

    decay_progress = (
        (progress - stable_end)
        / (1.0 - stable_end)
    )

    return 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )