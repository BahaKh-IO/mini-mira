from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mini_mira.ml.blocks import BlockConfig, LayerScale, SpaceTimeBlock
from mini_mira.ml.init import init_weights
from mini_mira.ml.rope import spatial_rope, temporal_rope


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
    # PixelRefinementHead, applied after PatchUnembed's per-token linear readout, before tanh --
    # see that class's docstring. Off by default: a deliberate opt-in, not a change to any
    # existing config's behavior (see notes/vjepa_codec_quality_research.md).
    use_refinement_head: bool = False
    refinement_channels: int = 64
    refinement_num_layers: int = 3


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


class PixelRefinementHead(nn.Module):
    """Small conv stack applied to the decoder's raw pixel output, right before tanh.

    PatchUnembed turns each token into its own 16x16(x2) pixel block independently -- nothing
    afterward looks across patch boundaries in pixel space, only the transformer blocks' global
    attention does, before readout. This gives the decoder a second, local, weight-shared way to
    blend across those seams that doesn't have to be learned through attention alone -- see
    notes/blockiness_investigation.md and notes/vjepa_codec_quality_research.md for the diagnosed
    failure mode this targets.

    A residual correction, not a replacement: forward returns `pixels + layerscale(conv_stack
    (pixels))`. Uses the same LayerScale this codebase's own transformer blocks already use
    (mini_mira.ml.blocks.LayerScale, initialized to `layerscale_init`, e.g. 1e-4) to keep the
    correction small and safe at the start -- NOT the exact-zero-init this class used previously.

    That's a deliberate, evidenced fix, not a style choice: an earlier exact-zero version of this
    head (real weight, real bias, both hard-zeroed) trained fine at first but caused a real `NaN`
    crash partway through a real overfit run -- an unbounded correction with nothing capping its
    growth. LayerScale's own paper (Touvron et al., ICCV 2021) is explicit that the right init is
    "a small epsilon different from zero", not exactly zero, for precisely this reason: every
    conv layer here gets real, nonzero gradient from step one (nothing is exactly zero anymore),
    while the overall correction still starts small and grows gradually rather than being capped
    off entirely until one single weight happens to move first. See
    scripts/verify_refinement_head.py for this proven directly.
    """

    def __init__(self, out_channels: int, channels: int, num_layers: int, eps: float, layerscale_init: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = out_channels
        for i in range(num_layers):
            is_last = i == num_layers - 1
            layer_out = out_channels if is_last else channels
            layers.append(nn.Conv2d(in_channels, layer_out, kernel_size=3, padding=1))
            if not is_last:
                layers.append(nn.GroupNorm(num_groups=min(32, channels), num_channels=channels, eps=eps))
                layers.append(nn.GELU())
            in_channels = layer_out
        self.layers = nn.ModuleList(layers)
        self.layerscale = LayerScale(out_channels, init=layerscale_init)

    @property
    def last_conv(self) -> nn.Conv2d:
        return self.layers[-1]

    def forward(self, x: Tensor) -> Tensor:
        """x: (b, t, c, h, w) raw pixels, pre-tanh."""
        b, t = x.shape[:2]
        y = rearrange(x, "b t c h w -> (b t) c h w")
        for layer in self.layers:
            y = layer(y)
        y = rearrange(y, "(b t) c h w -> b t c h w", b=b, t=t)
        # LayerScale.gamma has shape (out_channels,) and normally multiplies a channels-LAST
        # tensor (its usual (..., c) use in SpaceTimeBlock) -- here channels is dim 2, not last,
        # so gamma is reshaped to broadcast correctly instead of calling layerscale(y) directly.
        return x + y * self.layerscale.gamma.view(1, 1, -1, 1, 1)


class ViTVideoDecoder(nn.Module):
    def __init__(self, config: ViTDecoderConfig, use_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.use_checkpointing = use_checkpointing
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
        self.refinement_head: PixelRefinementHead | None = None
        if config.use_refinement_head:
            self.refinement_head = PixelRefinementHead(
                out_channels=config.out_channels,
                channels=config.refinement_channels,
                num_layers=config.refinement_num_layers,
                eps=config.eps,
                layerscale_init=config.layerscale_init,
            )

        self.apply(init_weights)
        # No manual zero-init needed here anymore -- PixelRefinementHead's own LayerScale
        # (initialized to config.layerscale_init) is what keeps its contribution small and safe
        # at construction. init_weights doesn't match LayerScale's type, so it leaves gamma
        # untouched -- no post-apply fixup step required, unlike the earlier zero-init version.

    @property
    def last_layer_weight(self):
        """The decoder's final projection weight -- used by CodecLoss's auto_weight to measure
        how hard each loss term pushes on the actual output, matching mira's vit_decoder.py
        exactly (same PatchUnembed.proj role in both) when there's no refinement head. With one,
        the refinement head's own LayerScale gamma IS the true final operation instead -- pointing
        there keeps auto_weight measuring the real last step, not an now-outdated stand-in."""
        if self.refinement_head is not None:
            return self.refinement_head.layerscale.gamma
        return self.patch_unembed.proj.weight

    def forward(self, z):
        b, t = z.shape[:2]
        x = rearrange(z, "b t c h w -> (b t) c h w")
        if self.use_checkpointing and self.training:
            x = checkpoint(self.from_latent, x, use_reentrant=False)
        else:
            x = self.from_latent(x)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=b, t=t)

        _, t, h, w, _ = x.shape
        rope_spatial = spatial_rope(h, w, self.head_dim, self.config.rope_theta_spatial, x.device)
        rope_temporal = temporal_rope(t, self.head_dim, self.config.rope_theta_temporal, x.device)

        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, rope_spatial, rope_temporal, use_reentrant=False)
            else:
                x = block(x, rope_spatial, rope_temporal)

        def produce_pixels(hidden):
            pixels = self.patch_unembed(self.norm_out(hidden))
            if self.refinement_head is not None:
                pixels = self.refinement_head(pixels)
            return torch.tanh(pixels)

        if self.use_checkpointing and self.training:
            return checkpoint(produce_pixels, x, use_reentrant=False)
        return produce_pixels(x)
