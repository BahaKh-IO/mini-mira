from dataclasses import dataclass

import torch.nn as nn
from einops import rearrange

from mini_mira.blocks import AdaSTBlock
from mini_mira.rope import spatial_rope, temporal_rope
from mini_mira.timestep_encoder import DiffusionTimeEmbedding


@dataclass
class LatentWorldModelConfig:
    latent_dim: int = 32
    hidden_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_dim_multiplier: int = 4
    eps: float = 1e-6
    layerscale_init: float = 1e-4
    rope_theta_spatial: float = 100.0
    rope_theta_temporal: float = 64.0


class DiffusionTransformer(nn.Module):
    def __init__(self, config: LatentWorldModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_dim // config.num_heads
        self.in_proj = nn.Linear(config.latent_dim, config.hidden_dim)
        self.past_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                AdaSTBlock(
                    dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    mlp_dim_multiplier=config.mlp_dim_multiplier,
                    causal=True,  # temporal attention causal, matching the real repo
                    cond_dim=config.hidden_dim,
                )
                for _ in range(config.depth)
            ]
        )
        self.diffusion_time_embedding = DiffusionTimeEmbedding(dim=config.hidden_dim)
        self.norm_out = nn.LayerNorm(config.hidden_dim, eps=config.eps)
        self.out_proj = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, z_t, a=None, tau=None, clean_past=None):
        b, t, c, h, w = z_t.shape
        x = rearrange(z_t, "b t c h w -> b t h w c")
        x = self.in_proj(x)

        # The past frames are always clean (un-noised) latents -- added to the noisy
        # projection, not concatenated, so the sequence length never changes.
        clean_past = rearrange(clean_past, "b t c h w -> b t h w c")
        x = x + self.past_proj(clean_past)

        rope_spatial = spatial_rope(h, w, self.head_dim, self.config.rope_theta_spatial, x.device)
        rope_temporal = temporal_rope(t, self.head_dim, self.config.rope_theta_temporal, x.device)

        tau_emb = self.diffusion_time_embedding(tau)  # (b, t, 1, 1, hidden_dim)
        a_broadcast = rearrange(a, "b t c -> b t 1 1 c").repeat(1, 1, h, w, 1)  # (b, t, h, w, hidden_dim)
        cond = a_broadcast + tau_emb.repeat(1, 1, h, w, 1)  # (b, t, h, w, hidden_dim)

        for block in self.blocks:
            x = block(x, cond, rope_spatial, rope_temporal)

        x = self.norm_out(x)
        x = self.out_proj(x)
        return rearrange(x, "b t h w c -> b t c h w")
