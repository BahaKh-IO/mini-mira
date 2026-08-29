from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from mini_mira.ml.init import init_weights

@dataclass
class StridedConvBottleneckConfig :
    dino_dim : int = 768  # dinov3_vitb16's feature width -- the real variant this project loads
    latent_dim : int = 32
    stride : int = 2
    temporal_stride : int = 2
    # Off by default. When on, the SHALLOWEST layer in a multi-layer input is kept as a separate
    # "texture" channel group instead of being blended into the mean below -- see bottleneck.py's
    # forward() and notes/vjepa_codec_quality_research.md (DINO-Tok, arXiv:2511.20565).
    use_shallow_texture_branch: bool = False
    shallow_texture_channels: int = 128


class MyBottleneck(nn.Module):
    def __init__(self,config:StridedConvBottleneckConfig, use_checkpointing: bool = False):
        super().__init__()
        self.config=config
        self.use_checkpointing = use_checkpointing

        in_channels = config.dino_dim
        self.texture_proj: nn.Conv2d | None = None
        if config.use_shallow_texture_branch:
            self.texture_proj = nn.Conv2d(config.dino_dim, config.shallow_texture_channels, kernel_size=1)
            in_channels += config.shallow_texture_channels

        if config.temporal_stride > 1:
            self.projection = nn.Conv3d(
                in_channels,
                config.latent_dim,
                kernel_size=(config.temporal_stride,config.stride, config.stride),
                stride = (config.temporal_stride,config.stride, config.stride),
                bias=True
            )
        else :
            self.projection = nn.Conv2d(
                in_channels,
                config.latent_dim,
                kernel_size=(config.stride, config.stride),
                stride = (config.stride, config.stride),
                bias=True
            )

        self.apply(init_weights)

    def _project(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint(self.projection, x, use_reentrant=False)
        return self.projection(x)

    def _project_texture(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint(self.texture_proj, x, use_reentrant=False)
        return self.texture_proj(x)

    def forward(self,x):
        # Multi-layer DINO input (a list of (b,t,c,h,w) tensors): blend into one tensor the same
        # way real mira's RAEEncoder.forward does before its own projection conv -- mean across
        # layers, plus the last layer again (extra weight toward the most semantic/late features).
        # A single tensor (the old single-layer case) passes through unchanged.
        if isinstance(x, list):
            deep = torch.stack(x, dim=0).mean(dim=0) + x[-1]
            if self.texture_proj is not None:
                # x[0] is the SHALLOWEST layer (least semantic, most texture-rich) -- projected
                # and concatenated as its own channel group, not blended into the mean above, so
                # the decoder gets undiluted access to it instead of it being averaged away.
                b, t = x[0].shape[:2]
                shallow = rearrange(x[0], "b t c h w -> (b t) c h w")
                shallow = self._project_texture(shallow)
                shallow = rearrange(shallow, "(b t) c h w -> b t c h w", b=b, t=t)
                x = torch.cat([deep, shallow], dim=2)
            else:
                x = deep

        if self.config.temporal_stride>1:
            x=rearrange(x,"b t c h w -> b c t h w")
            z=self._project(x)
            z=rearrange(z,"b c t h w -> b t c h w")
        else :
            b, t = x.shape[:2]
            x=rearrange(x,"b t c h w -> (b t) c h w")
            z=self._project(x)
            z=rearrange(z,"(b t) c h w -> b t c h w", b=b,t=t)

        return (z)
