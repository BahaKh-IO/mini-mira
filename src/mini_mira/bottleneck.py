from dataclasses import dataclass

import torch.nn as nn
from einops import rearrange

@dataclass
class BottleneckConfig :
    dino_dim : int = 1024
    latent_dim : int = 32
    stride : int = 2
    temporal_stride : int = 2


class MyBottleneck(nn.Module):
    def __init__(self,config:BottleneckConfig):
        super().__init__()
        self.config=config

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

    def forward(self,x):
        if self.config.temporal_stride>1:
            x=rearrange(x,"b t c h w -> b c t h w")
            z=self.projection(x)
            z=rearrange(z,"b c t h w -> b t c h w")
        else :
            b, t = x.shape[:2]
            x=rearrange(x,"b t c h w -> (b t) c h w")
            z=self.projection(x)
            z=rearrange(z,"(b t) c h w -> b t c h w", b=b,t=t)

        return (z)