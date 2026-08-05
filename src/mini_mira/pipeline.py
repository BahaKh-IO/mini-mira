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
    clean latent at generation time.
  - bos lives here (on MyPipeline), not on DiffusionTransformer: in the real repo it belongs
    to LatentWorldModel, the wrapper that owns the codec's encoder output. MyPipeline is
    mini_mira's equivalent of that wrapper (it's the one that calls self.bottleneck and holds
    z), so bos and the shift-by-one-frame that builds clean_past live here, matching that
    ownership rather than DiffusionTransformer's.
  - bos is a single per-channel vector (latent_dim,), broadcast over whatever h/w the input
    happens to have -- not a full (latent_height, latent_width, latent_dim) grid like the real
    repo's. mini_mira's bottleneck/world model never fix a spatial resolution in their configs
    (h, w are only known from the actual input tensor at forward time), so a fixed-grid bos
    would need a new resolution config field that doesn't otherwise exist here.
  - ActionEncoder lives here too, for the same reason as bos: mira's ActionEncoder belongs to
    LatentWorldModel, the wrapper that owns the codec's encoder output, not to
    DiffusionTransformer. Raw actions are encoded into a conditioning tensor here, once, then
    passed into the world model already-encoded -- DiffusionTransformer never sees raw
    key-press data, only the embedding.

Known limitation, stated rather than hidden: actions are keys-only (no mouse -- see
action_encoder.py's docstring), and simplified relative to mira's ActionEncoder (no dropout,
mean-pooling instead of a learned temporal pool, a plain Linear instead of mira's power-of-2
per-key split -- see notes/deviations.md 1.10).
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from mini_mira.action_encoder import ActionEncoder
from mini_mira.bottleneck import StridedConvBottleneckConfig, MyBottleneck
from mini_mira.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.world_model import DiffusionTransformer, LatentWorldModelConfig


@dataclass
class PipelineConfig:
    bottleneck: StridedConvBottleneckConfig = field(default_factory=StridedConvBottleneckConfig)
    world_model: LatentWorldModelConfig = field(default_factory=LatentWorldModelConfig)
    decoder: ViTDecoderConfig = field(default_factory=ViTDecoderConfig)
    n_diffusion_steps: int = 4
    num_keys: int = 9  # the real Rocket League release vocabulary size (W,A,S,D,Q,E,Space,LShift,LCtrl)


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

    def forward(self, dino_features, actions):
        z = self.bottleneck(dino_features)  # codec encode -- now actually used, not discarded
        b, t, c, h, w = z.shape

        # clean_past[t] = the real, clean content of frame t-1; clean_past[0] = bos (there is
        # no real frame before the first one in this clip). Built once from the real z, then
        # reused unchanged across every diffusion step below -- the past is always clean,
        # never the evolving noisy z_t.
        bos = self.bos.view(1, 1, c, 1, 1).expand(b, 1, c, h, w)
        clean_past = torch.cat([bos, z[:, :-1]], dim=1)

        # Encoded once, like clean_past -- the actions that produced this clip don't change
        # across diffusion steps, only the noise level (tau) does.
        a = self.action_encoder(actions)

        z_t = torch.randn_like(z)  # real generation starts from pure noise, not the true latent

        n_steps = self.config.n_diffusion_steps
        timesteps = torch.linspace(0, 1, n_steps + 1, device=z_t.device)
        delta_t = 1.0 / n_steps

        for i in range(n_steps):
            # Same tau for every frame at this step (this loop denoises the whole clip
            # together, not per-frame) -- but DiffusionTimeEmbedding needs a real
            # (b, t, 1, 1, 1) tensor, not the bare scalar timesteps[i] used to be.
            tau = timesteps[i] * torch.ones(b, t, 1, 1, 1, device=z_t.device)
            pred_v = self.world_model(z_t, a, tau, clean_past=clean_past)
            z_t = z_t + delta_t * pred_v

        return self.decoder(z_t)
