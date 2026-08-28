from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class BlockConfig:
    width: int = 256
    num_heads: int = 8
    mlp_dim_multiplier: int = 4
    causal: bool = True
    eps: float = 1e-6
    layerscale_init: float = 1e-4
    qk_norm: str = "layernorm"  # mira's shipped codec config overrides its own "rmsnorm" default to this


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


class QKRMSNorm(nn.Module):
    """Per-head RMSNorm for q/k, matching mira/src/mira/ml/attention.py's QKRMSNorm exactly.
    Always computed in fp32 regardless of the ambient autocast dtype: without a fixed scale,
    q/k magnitudes can grow unboundedly over training, and under fp16 specifically that growth
    is much more likely to overflow or lose precision than under fp32/bf16.
    """

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.qk_scale = nn.Parameter(torch.ones(num_heads, head_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.qk_scale.to(torch.float32) * x
        return x.to(input_dtype)


class QKLayerNorm(nn.Module):
    """Per-head LayerNorm for q/k, matching mira/src/mira/ml/attention.py's QKLayerNorm exactly --
    same as QKRMSNorm but mean-centered first. Same qk_scale shape as QKRMSNorm (no bias term
    either), so the two are interchangeable at the parameter level, only the forward math differs.
    """

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.qk_scale = nn.Parameter(torch.ones(num_heads, head_dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        x = self.qk_scale.to(torch.float32) * x
        return x.to(input_dtype)


# SpaceTimeBlock runs attention at two shapes that are about as different as two attention
# shapes get -- many short sequences over time, few long sequences over space -- and the fastest
# SDPA backend is not the same one for both. PyTorch's default priority order picks
# FlashAttention for both and loses on each. Measured on an H100, bf16, forward+backward, at the
# decoder's real 448x768x40 shapes:
#
#   temporal, 2688 sequences of length 20:   flash 7.5ms   mem-efficient 2.8ms   cuDNN 3.8ms
#   spatial,     40 sequences of length 1344: flash 6.9ms   cuDNN         4.1ms   mem-efficient 13.2ms
#
# Flash's per-sequence tiling has nowhere near enough work to amortize its own setup across
# 20-token sequences; on the long spatial sequences cuDNN's fused kernel is simply faster. Both
# lists keep every other backend as a fallback (set_priority orders them, it does not disable
# them), so an unsupported shape still lands somewhere valid rather than raising.
_SHORT_SEQUENCE_LEN = 128
_SHORT_SEQUENCE_BACKENDS = [
    SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION, SDPBackend.MATH,
]
_LONG_SEQUENCE_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
]


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: str = "layernorm"):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        qk_norm_cls = QKRMSNorm if qk_norm == "rmsnorm" else QKLayerNorm
        self.q_ln = qk_norm_cls(num_heads, self.head_dim)
        self.k_ln = qk_norm_cls(num_heads, self.head_dim)

    def forward(self, x, causal: bool, freqs: tuple[Tensor, Tensor] | None = None):
        b, n, c = x.shape
        qkv = self.wqkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_ln(q)
        k = self.k_ln(k)
        # Normalize before rotating, not after: normalizing rescales magnitude, rotating
        # changes direction -- reversed, the norm would partly undo what RoPE set up.
        if freqs is not None:
            q = apply_rotary_emb(q, freqs)
            k = apply_rotary_emb(k, freqs)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        backends = _SHORT_SEQUENCE_BACKENDS if n <= _SHORT_SEQUENCE_LEN else _LONG_SEQUENCE_BACKENDS
        with sdpa_kernel(backends, set_priority=True):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.wo(out)


class SpaceTimeBlock(nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        dim = config.width

        self.space_norm = nn.LayerNorm(dim, eps=config.eps)
        self.space_attn = SelfAttention(dim, config.num_heads, config.qk_norm)
        self.space_ls = LayerScale(dim, config.layerscale_init)

        self.time_norm = nn.LayerNorm(dim, eps=config.eps)
        self.time_attn = SelfAttention(dim, config.num_heads, config.qk_norm)
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


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, embed_dim: int, cond_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.gamma_beta = nn.Linear(cond_dim, 2 * embed_dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        normalized_x = self.layer_norm(x)
        gamma_beta = self.gamma_beta(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1 + gamma) * normalized_x + beta


class AdaSTBlock(nn.Module):
    """Conditioned sibling of SpaceTimeBlock, for the world model only.

    Matches mira's AdaSTBlock in three ways: forward takes (x, cond); all three norms
    (space, time, mlp) are AdaptiveLayerNorm; there is no LayerScale anywhere.

    Deliberately NOT implemented, unlike mira's version: the time_attention flag (temporal
    attention is always on here), kv_cache, and return_kv (forward returns just x, not a
    (x, to_cache) tuple). rope_spatial/rope_temporal are kept (mira's AdaSTBlock has these
    too, as temporal_rotary_emb/spatial_rotary_emb) so the RoPE wiring already in the world
    model keeps working.
    """

    def __init__(self, dim: int, num_heads: int, mlp_dim_multiplier: int, causal: bool, cond_dim: int):
        super().__init__()
        self.space_attn_ln = AdaptiveLayerNorm(dim, cond_dim)
        self.space_attn = SelfAttention(dim, num_heads)

        self.time_attn_ln = AdaptiveLayerNorm(dim, cond_dim)
        self.time_attn = SelfAttention(dim, num_heads)
        self.causal = causal

        self.mlp_ln = AdaptiveLayerNorm(dim, cond_dim)
        self.mlp = SwiGLU(dim, dim_multiplier=mlp_dim_multiplier)

    def forward(
        self,
        x,
        cond: Tensor,
        rope_spatial: tuple[Tensor, Tensor] | None = None,
        rope_temporal: tuple[Tensor, Tensor] | None = None,
    ):
        b, t, h, w, _ = x.shape

        xs = rearrange(x, "b t h w c -> (b t) (h w) c")
        cond_s = rearrange(cond, "b t h w c -> (b t) (h w) c")
        xs = xs + self.space_attn(self.space_attn_ln(xs, cond_s), causal=False, freqs=rope_spatial)

        xt = rearrange(xs, "(b t) (h w) c -> (b h w) t c", b=b, t=t, h=h, w=w)
        cond_t = rearrange(cond, "b t h w c -> (b h w) t c")
        xt = xt + self.time_attn(self.time_attn_ln(xt, cond_t), causal=self.causal, freqs=rope_temporal)

        x = rearrange(xt, "(b h w) t c -> b t h w c", b=b, h=h, w=w)
        x = x + self.mlp(self.mlp_ln(x, cond))
        return x
