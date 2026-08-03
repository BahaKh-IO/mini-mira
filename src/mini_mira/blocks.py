from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


@dataclass
class BlockConfig:
    width: int = 256
    num_heads: int = 8
    mlp_dim_multiplier: int = 4
    causal: bool = True
    eps: float = 1e-6
    layerscale_init: float = 1e-4


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class SwiGLU(nn.Module):
    def __init__(self, dim: int, dim_multiplier: int = 4, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * dim_multiplier * dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.swish_linear = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_linear = nn.Linear(dim, hidden_dim, bias=False)
        self.output_linear = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.output_linear(F.silu(self.swish_linear(x)) * self.gate_linear(x))


def apply_rotary_emb(x: Tensor, freqs: tuple[Tensor, Tensor]) -> Tensor:
    # x: (b, n, n_head, head_dim). cos, sin: (n, head_dim).
    cos, sin = freqs
    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)  # each (b, n, n_head, head_dim // 2)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)  # (b, n, n_head, head_dim)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, causal: bool, freqs: tuple[Tensor, Tensor] | None = None):
        b, n, c = x.shape
        qkv = self.wqkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if freqs is not None:
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.wo(out)


class SpaceTimeBlock(nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        dim = config.width

        self.space_norm = nn.LayerNorm(dim, eps=config.eps)
        self.space_attn = SelfAttention(dim, config.num_heads)
        self.space_ls = LayerScale(dim, config.layerscale_init)

        self.time_norm = nn.LayerNorm(dim, eps=config.eps)
        self.time_attn = SelfAttention(dim, config.num_heads)
        self.time_ls = LayerScale(dim, config.layerscale_init)
        self.causal = config.causal

        self.mlp_norm = nn.LayerNorm(dim, eps=config.eps)
        self.mlp = SwiGLU(dim, dim_multiplier=config.mlp_dim_multiplier)
        self.mlp_ls = LayerScale(dim, config.layerscale_init)

    def forward(
        self,
        x,
        rope_spatial: tuple[Tensor, Tensor] | None = None,
        rope_temporal: tuple[Tensor, Tensor] | None = None,
    ):
        b, t, h, w, _ = x.shape

        xs = rearrange(x, "b t h w c -> (b t) (h w) c")
        xs = xs + self.space_ls(self.space_attn(self.space_norm(xs), causal=False, freqs=rope_spatial))

        xt = rearrange(xs, "(b t) (h w) c -> (b h w) t c", b=b, t=t, h=h, w=w)
        xt = xt + self.time_ls(self.time_attn(self.time_norm(xt), causal=self.causal, freqs=rope_temporal))

        x = rearrange(xt, "(b h w) t c -> b t h w c", b=b, h=h, w=w)
        x = x + self.mlp_ls(self.mlp(self.mlp_norm(x)))
        return x
