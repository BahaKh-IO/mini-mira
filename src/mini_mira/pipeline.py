"""The pipeline: codec encode -> latent diffusion (multi-step) -> codec decode.

Design decisions:
  - Plain multi-step loop, no streaming KV-cache -- every step re-runs world_model on the whole
    z_t. Slower than the real denoise_streaming, but shape-equivalent and much simpler.
  - A few evenly-spaced diffusion steps (n_diffusion_steps), constant delta_t -- the real
    schedule affects sample quality, not shape.
  - z_t starts from pure random noise, matching real generation (you never have the true clean
    latent at generation time).
  - bos and ActionEncoder live here (on MyPipeline), not on DiffusionTransformer: in the real
    repo both belong to LatentWorldModel, the wrapper that owns the codec's encoder output --
    MyPipeline is mini_mira's equivalent of that wrapper.
  - bos is a single per-channel vector broadcast over h/w, not a full spatial grid -- mini_mira's
    configs never fix a spatial resolution ahead of time.

Known limitation: actions are keys-only (no mouse), and ActionEncoder is simplified relative to
mira's (no dropout, mean-pooling instead of a learned temporal pool) -- see notes/deviations.md.

use_real_dino (default False): off keeps forward's first argument as pre-encoded DINO-shaped
features (existing fast tests keep working, no gated weights needed). On builds a real frozen
DinoModel and forward's first argument becomes raw video instead, with DINO running internally
before the bottleneck -- what a real training loop uses.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from mini_mira.world_model.action_encoder import ActionEncoder
from mini_mira.codec.bottleneck import StridedConvBottleneckConfig, MyBottleneck
from mini_mira.codec.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.codec.dino import DinoModel
from mini_mira.world_model.diffusion_transformer import DiffusionTransformer, LatentWorldModelConfig


@dataclass
class PipelineConfig:
    bottleneck: StridedConvBottleneckConfig = field(default_factory=StridedConvBottleneckConfig)
    world_model: LatentWorldModelConfig = field(default_factory=LatentWorldModelConfig)
    decoder: ViTDecoderConfig = field(default_factory=ViTDecoderConfig)
    n_diffusion_steps: int = 4
    num_keys: int = 9  # the real Rocket League release vocabulary size (W,A,S,D,Q,E,Space,LShift,LCtrl)
    use_real_dino: bool = False  # see module docstring


class MyPipeline(nn.Module):
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self.bottleneck = MyBottleneck(config.bottleneck)
        self.world_model = DiffusionTransformer(config.world_model)
        self.decoder = ViTVideoDecoder(config.decoder)
        self.bos = nn.Parameter(0.02 * torch.randn(config.bottleneck.latent_dim))
        self.action_encoder = ActionEncoder(
            num_keys=config.num_keys,
            hidden_dim=config.world_model.hidden_dim,
            temporal_downsampling=config.bottleneck.temporal_stride,
        )
        # Only built when use_real_dino is on, so a pipeline built without it stays a strict
        # subset rather than carrying an unused frozen backbone.
        self.dino = DinoModel() if config.use_real_dino else None

    def forward(self, x, actions):
        # x: (B, T, dino_dim, H, W) pre-encoded DINO features if use_real_dino is False;
        # (B, T, 3, H, W) raw video in [0, 1] if True. no_grad here (encoder side): nothing
        # upstream of raw pixels needs a gradient, so no graph is needed through frozen DINO.
        if self.config.use_real_dino:
            with torch.no_grad():
                dino_features = self.dino.dino_forward(x)
        else:
            dino_features = x
        z = self.bottleneck(dino_features)
        b, t, c, h, w = z.shape

        # clean_past[t] = real content of frame t-1; clean_past[0] = bos. Built once, reused
        # unchanged across every diffusion step -- the past is always clean, never the noisy z_t.
        bos = self.bos.view(1, 1, c, 1, 1).expand(b, 1, c, h, w)
        clean_past = torch.cat([bos, z[:, :-1]], dim=1)

        a = self.action_encoder(actions)  # encoded once, same as clean_past

        z_t = torch.randn_like(z)  # generation starts from pure noise, not the true latent

        n_steps = self.config.n_diffusion_steps
        timesteps = torch.linspace(0, 1, n_steps + 1, device=z_t.device)
        delta_t = 1.0 / n_steps

        for i in range(n_steps):
            # Same tau for every frame this step; DiffusionTimeEmbedding needs (b, t, 1, 1, 1).
            tau = timesteps[i] * torch.ones(b, t, 1, 1, 1, device=z_t.device)
            pred_v = self.world_model(z_t, a, tau, clean_past=clean_past)
            z_t = z_t + delta_t * pred_v

        return self.decoder(z_t)
