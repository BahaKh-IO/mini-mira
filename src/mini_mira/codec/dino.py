"""Frozen DINOv3 backbone loading, matching mira's DinoModel.

Loads the backbone by importing dinov3.hub.backbones directly instead of going through
torch.hub.load's hubconf.py (which eagerly pulls in segmentation/detection/depth/text-alignment
code and their dependencies this project never touches). See notes/deviations.md for the full
writeup.

Pretrained weights are gated by Meta and must be requested/downloaded manually:
https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/. Point RS_DINO_WEIGHTS_DIR
at the local directory containing the .pth file (same env var convention as real mira).
"""

import os
import sys
from pathlib import Path

import torch
import torch.hub
import torch.nn as nn
from einops import rearrange
from torch import Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16

# Real mira also lists vitl16 (1024-dim); mini_mira only loads vitb16, the variant actually
# downloaded for this project, but both stay here since this is copied from mira's own DINO_DIM.
# vits16 isn't in mira's own tables at all (mira never uses it) -- added here for the smaller
# perceptual-loss-only DINO (see loss.py's bind_perceptual_dino). 384 is the standard DINOv3
# ViT-S width; worth confirming against the loaded model on the GPU box the first time it's used.
DINO_DIM = {
    "dinov3_vitl16": 1024,
    "dinov3_vitb16": 768,
    "dinov3_vits16": 384,
}

DINO_WEIGHT_FILENAMES = {
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
}

# mira's own multi-layer default for the DINO-consistency loss (last_layer_only=False). Not in
# mira's own table for vits16 (mira never uses that variant) -- inferred, not confirmed: DINOv2/v3's
# S and B variants share the same 12-layer depth, differing only in width, so vitb16's depth-indexed
# choice should carry over unchanged.
DEFAULT_DINO_LAYERS = {
    "dinov3_vitl16": (2, 6, 10, 14, 18, 22),
    "dinov3_vitb16": (2, 5, 8, 11),
    "dinov3_vits16": (2, 5, 8, 11),
}


def resolve_dino_weights(dino_model: str) -> Path | None:
    """Local DINOv3 weights file for `dino_model`, or None if RS_DINO_WEIGHTS_DIR isn't set."""
    weights_dir = os.environ.get("RS_DINO_WEIGHTS_DIR")
    if not weights_dir:
        return None
    candidate = Path(weights_dir) / DINO_WEIGHT_FILENAMES[dino_model]
    return candidate if candidate.exists() else None


def _load_dinov3_backbone_fn(dino_model: str):
    """Real facebookresearch/dinov3 backbone constructor (e.g. dinov3_vitb16), without executing
    the repo's hubconf.py -- uses the same private cache-fetch helper torch.hub.load calls
    internally, then imports only the backbone submodule instead of the whole hub entrypoint.

    Relies on a private torch.hub API, so it isn't guaranteed stable across torch versions (it
    already needed a fix once, for a `calling_fn` argument added in a newer torch). If it breaks
    again, the fallback is what torch.hub.load does under the hood: locate/download
    <torch.hub.get_dir()>/facebookresearch_dinov3_main and import dinov3.hub.backbones directly.
    """
    repo_dir = torch.hub._get_cache_or_reload(
        "facebookresearch/dinov3",
        force_reload=False,
        trust_repo=True,
        calling_fn="load",  # matches torch.hub.load's own internal call
        verbose=False,
        skip_validation=True,
    )
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import dinov3.hub.backbones as backbones

    return getattr(backbones, dino_model)


class DinoModel(nn.Module):
    """Frozen DINOv3 feature extractor, matching mira's DinoModel. Single-layer by default (what
    the encoder path needs); pass last_layer_only=False for the multi-layer mode mira's own
    DinoPerceptualLoss uses, e.g. via bind_perceptual_dino. No torch.compile, and no internal
    torch.no_grad() -- gradient flow is left to the caller (see dino_forward below).
    """

    def __init__(
        self,
        dino_model: str = "dinov3_vitb16",
        require_pretrained: bool = True,
        last_layer_only: bool = True,
        layer_indices: tuple[int, ...] | None = None,
    ):
        super().__init__()
        assert dino_model in DINO_DIM, f"unsupported dino_model {dino_model!r}"
        if last_layer_only and layer_indices is not None:
            raise ValueError("DinoModel: pass either last_layer_only=True OR layer_indices=(...), not both.")
        self.dino_model_name = dino_model
        self.dino_dim = DINO_DIM[dino_model]
        self.patch_size = PATCH_SIZE
        self.layers = layer_indices if layer_indices is not None else (
            1 if last_layer_only else DEFAULT_DINO_LAYERS[dino_model]
        )

        backbone_fn = _load_dinov3_backbone_fn(dino_model)
        weights_path = resolve_dino_weights(dino_model)
        if weights_path is not None:
            self.dino_model = backbone_fn(pretrained=True, weights=str(weights_path))
        elif require_pretrained:
            raise FileNotFoundError(
                f"DINOv3 pretrained weights for {dino_model} not found. Set RS_DINO_WEIGHTS_DIR "
                f"to a directory containing {DINO_WEIGHT_FILENAMES[dino_model]} (request access "
                f"at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), or "
                f"pass require_pretrained=False to build with random weights instead."
            )
        else:
            self.dino_model = backbone_fn(pretrained=False)

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float)[None, :, None, None], persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD, dtype=torch.float)[None, :, None, None], persistent=False
        )

        self.requires_grad_(False)
        self.eval()

    def image_normalization(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def dino_forward(self, x: Tensor) -> Tensor | list[Tensor]:
        """x: (b, t, 3, h, w), values in [0, 1]. Returns (b, t, dino_dim, h', w') for the default
        single-layer case, or a list of that shape (one per entry in self.layers) when built with
        last_layer_only=False.

        No internal no_grad(): call it under torch.no_grad() yourself when the input doesn't
        need a gradient (the encoder side); leave it unwrapped when it does (CodecLoss's
        DINO-consistency term backprops through this call onto the reconstruction).
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            b, t, _, h, w = x.shape
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.image_normalization(x)
            new_h = self.patch_size * (h // self.patch_size)
            new_w = self.patch_size * (w // self.patch_size)
            x = torch.nn.functional.interpolate(x, (new_h, new_w), mode="bilinear", antialias=True)
            features = self.dino_model.get_intermediate_layers(x, n=self.layers, norm=True, reshape=True)
            features = [rearrange(f, "(b t) c h w -> b t c h w", b=b, t=t) for f in features]
            return features[0] if isinstance(self.layers, int) else features
