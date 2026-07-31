from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class DecoderConfig:
    latent_dim: int = 32
    stride: int = 2  
    width: int = 256  
    depth: int = 4
    num_heads: int = 8
    mlp_dim_multiplier: int = 4
    causal: bool = True  
    out_channels: int = 3
    patch_size: int = 16
    patch_size_t: int = 2
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


class SimpleSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, causal: bool):
        b, n, c = x.shape
        qkv = self.wqkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.wo(out)


class SpaceTimeBlock(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        dim = config.width

        self.space_norm = nn.LayerNorm(dim, eps=config.eps)
        self.space_attn = SimpleSelfAttention(dim, config.num_heads)
        self.space_ls = LayerScale(dim, config.layerscale_init)

        self.time_norm = nn.LayerNorm(dim, eps=config.eps)
        self.time_attn = SimpleSelfAttention(dim, config.num_heads)
        self.time_ls = LayerScale(dim, config.layerscale_init)
        self.causal = config.causal

        self.mlp_norm = nn.LayerNorm(dim, eps=config.eps)
        self.mlp = SwiGLU(dim, dim_multiplier=config.mlp_dim_multiplier)
        self.mlp_ls = LayerScale(dim, config.layerscale_init)

    def forward(self, x):
        b, t, h, w, _ = x.shape

        xs = rearrange(x, "b t h w c -> (b t) (h w) c")
        xs = xs + self.space_ls(self.space_attn(self.space_norm(xs), causal=False))

        xt = rearrange(xs, "(b t) (h w) c -> (b h w) t c", b=b, t=t, h=h, w=w)
        xt = xt + self.time_ls(self.time_attn(self.time_norm(xt), causal=self.causal))

        x = rearrange(xt, "(b h w) t c -> b t h w c", b=b, h=h, w=w)
        x = x + self.mlp_ls(self.mlp(self.mlp_norm(x)))
        return x


class PatchUnembed(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size
        self.patch_size_t = config.patch_size_t
        self.proj = nn.Linear(
            config.width,
            config.out_channels * config.patch_size_t * config.patch_size * config.patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        return rearrange(
            x,
            "b t h w (c pt p1 p2) -> b (t pt) c (h p1) (w p2)",
            c=self.out_channels,
            pt=self.patch_size_t,
            p1=self.patch_size,
            p2=self.patch_size,
        )


class MyDecoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        self.from_latent = nn.ConvTranspose2d(
            config.latent_dim,
            config.width,
            kernel_size=config.stride,
            stride=config.stride,
            bias=True,
        )
        
        self.blocks = nn.ModuleList([SpaceTimeBlock(config) for _ in range(config.depth)])
        self.norm_out = nn.LayerNorm(config.width, eps=config.eps)
        self.patch_unembed = PatchUnembed(config)

    def forward(self, z):
        b, t = z.shape[:2]
        x = rearrange(z, "b t c h w -> (b t) c h w")
        x = self.from_latent(x)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=b, t=t)

        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)
        return torch.tanh(self.patch_unembed(x))
