import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.layers.kda import KimiDeltaAttention

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        nn.init.trunc_normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
            a=-0.06,
            b=0.06,
        )

    def forward(self, x):
        return self.embedding(x)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        for layer in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.trunc_normal_(
            layer.weight,
            mean=0.0,
            std=0.02,
            a=-0.06,
            b=0.06,
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class KDA(nn.Module):
    def __init__(self, d_model ,num_heads, head_dim):
        super().__init__()

        self.kda = KimiDeltaAttention(
            hidden_size = d_model,
            num_heads = num_heads,
            head_dim = head_dim,
        )

    def forward(self, x):
        output, _, _ = self.kda(x)
        return output


class KDABlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_heads,
        head_dim,
    ):
        super().__init__()

        self.attn_norm = nn.RMSNorm(
            d_model,
            eps=1e-6,
        )

        self.kda = KDA(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        self.ffn_norm = nn.RMSNorm(
            d_model,
            eps=1e-6,
        )

        self.ffn = FFN(
            d_model=d_model,
            d_ff=d_ff,
        )

    def forward(self, x):
        residual = x

        x = self.attn_norm(x)
        x = self.kda(x)
        x = residual + x

        residual = x

        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x
