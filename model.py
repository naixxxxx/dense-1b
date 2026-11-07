import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.layers.gated_deltanet import GatedDeltaNet
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
 

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        d_ff,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim
    ):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if (i + 1) % 4 == 0:
                block = GatedAttentionBlock(d_model, d_ff, num_heads, num_kv_heads, head_dim)
            else:
                block = GatedDeltaNetBlock(d_model, d_ff, num_heads, head_dim)

            self.layers.append(block)

        self.norm = nn.RMSNorm(d_model, eps=1e-6)
        self.lm_head = LMHead(d_model, vocab_size)
        self.lm_head.proj.weight = self.embedding.embedding.weight
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        if labels is not None:
            return self.loss_fn(
                self.lm_head.proj.weight,
                x.reshape(-1, x.size(-1)),
                labels.reshape(-1),
            )

        return self.lm_head(x)




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


class LMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.proj = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.trunc_normal_(
            self.proj.weight,
            mean=0.0,
            std=0.02,
            a=-0.06,
            b=0.06,
        )

    def forward(self, x):
        logits = self.proj(x)
        return logits


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


class GatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model, num_heads, head_dim):
        super().__init__()

        self.gated_deltanet = GatedDeltaNet(
            hidden_size=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            expand_v=1.0,
            norm_eps=1e-6,
        )

        for weight in [
            self.gated_deltanet.q_proj.weight,
            self.gated_deltanet.k_proj.weight,
            self.gated_deltanet.v_proj.weight,
            self.gated_deltanet.a_proj.weight,
            self.gated_deltanet.b_proj.weight,
            self.gated_deltanet.g_proj.weight,
            self.gated_deltanet.o_proj.weight,
            self.gated_deltanet.q_conv1d.weight,
            self.gated_deltanet.k_conv1d.weight,
            self.gated_deltanet.v_conv1d.weight,
        ]:
            nn.init.trunc_normal_(
                weight,
                mean=0.0,
                std=0.02,
                a=-0.06,
                b=0.06,
            )

    def forward(self, x):
        output, _, _ = self.gated_deltanet(x)
        return output

class GatedAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, head_dim):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

        for layer in [self.q_proj, self.k_proj, self.v_proj, self.gate_proj, self.o_proj]:
            nn.init.trunc_normal_(
                layer.weight,
                mean=0.0,
                std=0.02,
                a=-0.06,
                b=0.06,
            )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        gate = self.gate_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        num_groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(num_groups, dim=2)
        v = v.repeat_interleave(num_groups, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        gate = torch.sigmoid(gate)
        attn_output = attn_output * gate

        output = self.o_proj(attn_output)
        return output

class GatedDeltaNetBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, head_dim):
        super().__init__()

        self.attn_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.gated_deltanet = GatedDeltaNetLayer(d_model, num_heads, head_dim)

        self.ffn_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        residual = x
        x = self.attn_norm(x)
        x = self.gated_deltanet(x)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class GatedAttentionBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_kv_heads, head_dim):
        super().__init__()

        self.attn_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.attn = GatedAttention(d_model, num_heads, num_kv_heads, head_dim)

        self.ffn_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x

