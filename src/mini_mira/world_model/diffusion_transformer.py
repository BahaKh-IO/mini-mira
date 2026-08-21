from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from mini_mira.ml.blocks import AdaSTBlock
from mini_mira.ml.init import init_weights
from mini_mira.ml.rope import spatial_rope, temporal_rope
from mini_mira.world_model.timestep_encoder import DiffusionTimeEmbedding


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
    # PSD self-distillation (mira's real world_model/config.py fields, same names/defaults --
    # both off by default, matching mira's own shipped single-player config). See
    # LatentWorldModel._compute_psdm_loss (mini_mira.world_model.latent_world_model) for the loss
    # that actually uses these; DiffusionTransformer only needs to know whether to build the extra
    # tau_delta embedding below.
    psd_loss_prob: float = 0.0
    psd_weight: float = 0.0

    @property
    def psd_enabled(self) -> bool:
        return self.psd_loss_prob > 0 or self.psd_weight > 0


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
        # Second time embedding for PSD self-distillation's integration-step size (tau_delta),
        # matching real mira's diffusion_transformer.py exactly: only built when PSD is actually
        # enabled, so a default-config model has no extra parameters and no tau_delta pathway at
        # all -- forward() below is fully backward-compatible when this stays None (see
        # verify_conditioning.py's tau_delta backward-compatibility check).
        self.diffusion_time_embedding_delta = None
        if config.psd_enabled:
            self.diffusion_time_embedding_delta = DiffusionTimeEmbedding(dim=config.hidden_dim)
        self.norm_out = nn.LayerNorm(config.hidden_dim, eps=config.eps)
        self.out_proj = nn.Linear(config.hidden_dim, config.latent_dim)

        self.apply(init_weights)

        # RoPE tables only depend on (h, w) / t / head_dim / theta / device -- all constant across
        # an entire training run, and across most calls within one rollout too. Single-slot cache
        # (not a dict) since in practice only one (h, w) and one t are ever in play per process;
        # avoids recomputing the same table on literally every forward call. Tables have no grad
        # (plain cos/sin, see ml/rope.py), so caching across train/eval calls is safe.
        self._rope_spatial_cache: tuple | None = None
        self._rope_temporal_cache: tuple | None = None

    def forward(self, z_t, a=None, tau=None, tau_delta=None, clean_past=None):
        b, t, c, h, w = z_t.shape
        x = rearrange(z_t, "b t c h w -> b t h w c")
        x = self.in_proj(x)

        # The past frames are always clean (un-noised) latents -- added to the noisy
        # projection, not concatenated, so the sequence length never changes.
        clean_past = rearrange(clean_past, "b t c h w -> b t h w c")
        x = x + self.past_proj(clean_past)

        spatial_key = (h, w, x.device)
        if self._rope_spatial_cache is None or self._rope_spatial_cache[0] != spatial_key:
            self._rope_spatial_cache = (
                spatial_key, spatial_rope(h, w, self.head_dim, self.config.rope_theta_spatial, x.device)
            )
        rope_spatial = self._rope_spatial_cache[1]

        temporal_key = (t, x.device)
        if self._rope_temporal_cache is None or self._rope_temporal_cache[0] != temporal_key:
            self._rope_temporal_cache = (
                temporal_key, temporal_rope(t, self.head_dim, self.config.rope_theta_temporal, x.device)
            )
        rope_temporal = self._rope_temporal_cache[1]

        tau_emb = self.diffusion_time_embedding(tau)  # (b, t, 1, 1, hidden_dim)
        if self.diffusion_time_embedding_delta is not None:
            # PSD-trained models take the integration-step size as input; None means "no delta"
            # (a diagonal-loss forward pass through a PSD-enabled model), matching mira's own
            # convention in latent_world_model.py's non-PSD call sites.
            if tau_delta is None:
                tau_delta = torch.zeros_like(tau)
            tau_emb = tau_emb + self.diffusion_time_embedding_delta(tau_delta)
        a_broadcast = rearrange(a, "b t c -> b t 1 1 c").repeat(1, 1, h, w, 1)  # (b, t, h, w, hidden_dim)
        cond = a_broadcast + tau_emb.repeat(1, 1, h, w, 1)  # (b, t, h, w, hidden_dim)

        for block in self.blocks:
            x = block(x, cond, rope_spatial, rope_temporal)

        x = self.norm_out(x)
        x = self.out_proj(x)
        return rearrange(x, "b t h w c -> b t c h w")
