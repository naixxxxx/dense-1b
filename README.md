"""
    #
    parameters=1,002,931,488
    vocab_size=151936,
    d_model=1280,
    d_ff=3840,
    num_layers=24,
    num_heads=16,
    num_kv_heads=4,
    head_dim=128,
    
    # Muon
    lr = 2e-2
    momentum = 0.95
    nesterov = True

    # AdamW
    lr = 3e-4
    betas = (0.9, 0.95)
    eps = 1e-8

    # WSD
    warmup = 0.02
    stable = 0.88
    decay = 0.1
    min_lr_ratio = 0.00

    Muon:
    - hidden 2D matrix


    AdamW + decay:
    - embedding.weight
    - lm_head.weight
    - KDA  q/k/v_conv1d.weight

    AdamW + no_decay:
    - RMSNorm.weight
    - bias
    - A_log
    - dt_bias
    - etc. 1D / scalar parameters
"""