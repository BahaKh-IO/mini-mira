"""Weight initialization, matching mira/src/mira/ml/init.py: Linear/Conv weights from
N(0, 0.02) with zero bias, norm layers to unit weight/zero bias, AdaptiveLayerNorm's
conditioning projection zeroed. QKRMSNorm/LayerScale already set their own init inline, so
they're not in the dispatch below.
"""

import torch.nn as nn

from mini_mira.ml.blocks import AdaptiveLayerNorm


def init_weights(module: nn.Module) -> None:
    """Pass to nn.Module.apply to run this over every submodule."""
    if isinstance(module, AdaptiveLayerNorm):
        nn.init.zeros_(module.gamma_beta.weight)
        nn.init.zeros_(module.gamma_beta.bias)
    elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.RMSNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
