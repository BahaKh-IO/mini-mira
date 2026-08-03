from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from mini_mira.blocks import BlockConfig, SpaceTimeBlock
from mini_mira.rope import spatial_rope, temporal_rope


@dataclass
class ViTDecoderConfig:
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
    rope_theta_spatial: float = 100.0
    rope_theta_temporal: float = 64.0


class PatchUnembed(nn.Module):
    def __init__(self, config: ViTDecoderConfig):
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


class ViTVideoDecoder(nn.Module):
    def __init__(self, config: ViTDecoderConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.width // config.num_heads

        self.from_latent = nn.ConvTranspose2d(
            config.latent_dim,
            config.width,
            kernel_size=config.stride,
            stride=config.stride,
            bias=True,
        )
        
        block_config = BlockConfig(
            width=config.width,
            num_heads=config.num_heads,
            mlp_dim_multiplier=config.mlp_dim_multiplier,
            causal=config.causal,
            eps=config.eps,
            layerscale_init=config.layerscale_init,
        )
        self.blocks = nn.ModuleList([SpaceTimeBlock(block_config) for _ in range(config.depth)])
        self.norm_out = nn.LayerNorm(config.width, eps=config.eps)
        self.patch_unembed = PatchUnembed(config)

    def forward(self, z):
        b, t = z.shape[:2]
        x = rearrange(z, "b t c h w -> (b t) c h w")
        x = self.from_latent(x)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=b, t=t)

        _, t, h, w, _ = x.shape
        rope_spatial = spatial_rope(h, w, self.head_dim, self.config.rope_theta_spatial, x.device)
        rope_temporal = temporal_rope(t, self.head_dim, self.config.rope_theta_temporal, x.device)

        for block in self.blocks:
            x = block(x, rope_spatial, rope_temporal)

        x = self.norm_out(x)
        return torch.tanh(self.patch_unembed(x))
