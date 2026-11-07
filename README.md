## Model

parameters    = 840,599,104
vocab_size    = 151936
d_model       = 1280
d_ff          = 3840
num_layers    = 24
num_heads     = 16
num_kv_heads  = 4
head_dim      = 128

tokens        =  19,682,992,109
seq_len       = 4096

## Tokenizer

Qwen/Qwen3-0.6B-Base


## Optimizer

Muon
  lr          = 2e-2
  momentum    = 0.95
  nesterov    = True

AdamW
  lr          = 3e-4
  betas       = (0.9, 0.95)

## WSD

warmup        = 0.02
stable        = 0.88
decay         = 0.10
min_lr_ratio  = 0.00

## Parameter Groups

**Muon**
- Hidden 2D Linear weights

**AdamW + decay**
- Embedding, LMHead
- Gated DeltaNet Conv1d (3D)

**AdamW + no decay**
- RMSNorm, bias
- A_log, dt_bias
- Other 1D / scalar parameters

uv run torchrun --standalone --nproc_per_node=8 train.py > train.log 2>&1 &
