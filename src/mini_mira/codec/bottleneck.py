from dataclasses import dataclass

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


class MyBottleneck(nn.Module):
    def __init__(self,config:StridedConvBottleneckConfig, use_checkpointing: bool = False):
        super().__init__()
        self.config=config
        self.use_checkpointing = use_checkpointing

        if config.temporal_stride > 1:
            self.projection = nn.Conv3d(
                config.dino_dim,
                config.latent_dim,
                kernel_size=(config.temporal_stride,config.stride, config.stride),
                stride = (config.temporal_stride,config.stride, config.stride),
                bias=True
            )
        else :
            self.projection = nn.Conv2d(
                config.dino_dim,
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

    def forward(self,x):
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
