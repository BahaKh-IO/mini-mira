"""The pipeline: codec encode -> latent diffusion (multi-step) -> codec decode.

Design decisions (asked and confirmed rather than assumed):
  - Plain multi-step loop, no streaming KV-cache. The real denoise_streaming only
    re-denoises the last frame per call, caching earlier frames for efficiency during long
    autoregressive rollouts. That requires kv_cache/return_kv support in SelfAttention
    and SpaceTimeBlock, which don't have it. Here, every step just calls world_model on the
    WHOLE z_t and re-runs the whole stack -- slower, but shape-equivalent and much simpler.
  - A few steps (n_diffusion_steps), evenly spaced tau in [0, 1], constant delta_t. The real
    schedule ("linear_quadratic", via build_inference_schedule) affects sample QUALITY, not
    shape -- same reasoning already used for Pass C and RoPE.
  - z_t starts from pure random noise (torch.randn_like(z)), matching how
    LatentWorldModel.inference actually starts a real generation -- you never have the true
    clean latent at generation time. The bottleneck-encoded z is used only to get the right
    shape/dtype for that noise, then otherwise unused.

Known limitation, stated rather than hidden: DiffusionTransformer still ignores actions/tau. So this loop is mechanically real (it really
does call the network multiple times and integrate the predicted velocity), but the
network's predictions don't actually improve as tau advances, since it can't see tau. The
loop is correct machinery around a network that doesn't yet use the signal it's given.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from mini_mira.bottleneck import StridedConvBottleneckConfig, MyBottleneck
from mini_mira.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.world_model import DiffusionTransformer, LatentWorldModelConfig


@dataclass
class PipelineConfig:
    bottleneck: StridedConvBottleneckConfig = field(default_factory=StridedConvBottleneckConfig)
    world_model: LatentWorldModelConfig = field(default_factory=LatentWorldModelConfig)
    decoder: ViTDecoderConfig = field(default_factory=ViTDecoderConfig)
    n_diffusion_steps: int = 4


class MyPipeline(nn.Module):
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self.bottleneck = MyBottleneck(config.bottleneck)
        self.world_model = DiffusionTransformer(config.world_model)
        self.decoder = ViTVideoDecoder(config.decoder)

    def forward(self, dino_features, actions=None):
        z = self.bottleneck(dino_features)  # codec encode -- used here only for shape/dtype
        z_t = torch.randn_like(z)  # real generation starts from pure noise, not the true latent

        n_steps = self.config.n_diffusion_steps
        timesteps = torch.linspace(0, 1, n_steps + 1, device=z_t.device)
        delta_t = 1.0 / n_steps

        for i in range(n_steps):
            tau = timesteps[i]
            pred_v = self.world_model(z_t, actions, tau)
            z_t = z_t + delta_t * pred_v

        return self.decoder(z_t)
