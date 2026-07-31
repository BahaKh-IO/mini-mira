"""Day 3 -- the world model.

Builds directly to the roadmap's "Pass B" shape contract (factorised space-time attention,
still no conditioning), skipping a separate "Pass A" stage: Pass A exists to de-risk new,
untested attention code incrementally before training it, but SpaceTimeBlock is already
built and tested (Day 2), and we aren't training anything here -- only checking shapes. A
separate throwaway plain-transformer stage wouldn't have added anything we don't already
have. Deliberate simplification, noted here rather than silently skipped.

Pass C (action + diffusion-time conditioning via AdaLN) is also not implemented: it changes
attention's VALUES, never any tensor's SHAPE, so it has zero effect on the actual
requirement (correct output shapes for the whole forward pass). actions and tau are
accepted in forward() with the real signature, but intentionally unused.

Positional encoding: MIRA uses RoPE; here we use plain learned embeddings instead -- a
deliberate, reported simplification, not a silent omission.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from mini_mira.decoder import DecoderConfig, SpaceTimeBlock


@dataclass
class WorldModelConfig:
    latent_dim: int = 32
    hidden_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_dim_multiplier: int = 4
    eps: float = 1e-6
    max_t: int = 32
    max_h: int = 16
    max_w: int = 32


class MyWorldModel(nn.Module):
    """Factorised space-time transformer over the codec latent grid.

    No conditioning yet (Pass C, intentionally not attempted -- see module docstring).
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        self.pos_t = nn.Parameter(0.02 * torch.randn(1, config.max_t, 1, 1, config.hidden_dim))
        self.pos_h = nn.Parameter(0.02 * torch.randn(1, 1, config.max_h, 1, config.hidden_dim))
        self.pos_w = nn.Parameter(0.02 * torch.randn(1, 1, 1, config.max_w, config.hidden_dim))

        block_config = DecoderConfig(
            width=config.hidden_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_dim_multiplier=config.mlp_dim_multiplier,
            causal=True,  # temporal attention causal, matching the real repo
            eps=config.eps,
        )
        self.blocks = nn.ModuleList([SpaceTimeBlock(block_config) for _ in range(config.depth)])
        self.norm_out = nn.LayerNorm(config.hidden_dim, eps=config.eps)
        self.out_proj = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, z_t, actions=None, tau=None):
        # actions, tau: accepted (matching the real DiffusionTransformer signature),
        # intentionally unused -- Pass C conditioning not attempted, see module docstring.
        b, t, c, h, w = z_t.shape
        x = rearrange(z_t, "b t c h w -> b t h w c")
        x = self.in_proj(x)
        x = x + self.pos_t[:, :t] + self.pos_h[:, :, :h] + self.pos_w[:, :, :, :w]

        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)
        x = self.out_proj(x)
        return rearrange(x, "b t h w c -> b t c h w")
