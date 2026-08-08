"""Frozen DINOv3 backbone loading, matching mira's DinoModel.

Real mira loads the backbone via `torch.hub.load(repo_or_dir="facebookresearch/dinov3", ...)`.
That call executes the repo's `hubconf.py` before it does anything else -- and `hubconf.py`
eagerly imports segmentation, detection, depth-estimation, and text-alignment code mini_mira
never touches, each pulling in its own extra dependency (`torchmetrics`, `termcolor`, and
likely more after that). mini_mira gets the exact same code and the exact same real pretrained
weights by importing the backbone constructor function directly instead
(`dinov3.hub.backbones.dinov3_vitb16`), skipping `hubconf.py` entirely. See notes/deviations.md
for the full writeup of this and the other simplifications below.

Pretrained weights are gated by Meta and must be requested and downloaded manually (see
https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) -- this file cannot
fetch them. Point RS_DINO_WEIGHTS_DIR at the local directory containing the .pth file,
matching mira's own environment variable convention exactly.
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

# Real mira also lists dinov3_vitl16 (1024-dim) here. mini_mira only ever loads vitb16 -- the
# variant actually requested and downloaded for this project -- so only that one is wired up
# below, but both stay in this dict since it's copied straight from mira's own DINO_DIM.
DINO_DIM = {
    "dinov3_vitl16": 1024,
    "dinov3_vitb16": 768,
}

DINO_WEIGHT_FILENAMES = {
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}


def resolve_dino_weights(dino_model: str) -> Path | None:
    """Return a local DINOv3 weights file for `dino_model`, or None if none is configured.

    Set RS_DINO_WEIGHTS_DIR to a directory holding the pretrained .pth file (named as in
    DINO_WEIGHT_FILENAMES) to load real weights; otherwise DinoModel either raises (if
    require_pretrained=True, the default) or builds with random weights.
    """
    weights_dir = os.environ.get("RS_DINO_WEIGHTS_DIR")
    if not weights_dir:
        return None
    candidate = Path(weights_dir) / DINO_WEIGHT_FILENAMES[dino_model]
    return candidate if candidate.exists() else None


def _load_dinov3_backbone_fn(dino_model: str):
    """Return the real facebookresearch/dinov3 backbone constructor (e.g. dinov3_vitb16),
    without executing the repo's hubconf.py.

    torch.hub._get_cache_or_reload is the same private helper torch.hub.load itself calls
    first, to make sure the repo is downloaded to the local hub cache (cloning it from GitHub
    the first time, reusing the cache after that). Calling it directly gets the exact same
    repo, at the exact same commit, as torch.hub.load would -- the only difference is what
    happens next: torch.hub.load would go on to import hubconf.py (the whole point being
    avoided here); this instead imports only the one submodule that actually defines the
    backbone functions.

    Relies on a private (underscore-prefixed) torch.hub function, so it isn't a guaranteed
    stable API -- it could break on a future torch version. If it ever does, the fix is the
    same thing torch.hub.load does under the hood: locate (or download)
    <torch.hub.get_dir()>/facebookresearch_dinov3_main and import dinov3.hub.backbones from
    there directly.
    """
    repo_dir = torch.hub._get_cache_or_reload(
        "facebookresearch/dinov3",
        force_reload=False,
        trust_repo=True,
        verbose=False,
        skip_validation=True,
    )
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import dinov3.hub.backbones as backbones

    return getattr(backbones, dino_model)


class DinoModel(nn.Module):
    """Frozen DINOv3 feature extractor. Matches mira's DinoModel, trimmed to what mini_mira
    actually uses:
      - Single-layer features only, matching mira's own last_layer_only=True default --
        dino_forward below returns one Tensor, not the List[Tensor] mira's version returns
        (mira's list only ever has one element here too, unless last_layer_only=False, which
        mini_mira never sets). Multi-layer features exist in mira only to feed
        DinoPerceptualLoss's latent-consistency term, which needs several layers to average
        over -- see the next point.
      - No DinoPerceptualLoss class (mira's own perceptual/latent-consistency loss module) --
        mini_mira's equivalent lives directly in mini_mira.codec.loss.CodecLoss instead, calling
        dino_forward the same way mira's RAEEncoder.forward does.
      - No torch.compile. A speed optimization for repeated real training-scale calls, not a
        correctness requirement -- and not obviously worth it for a CPU-only educational
        project making a handful of forward passes.
      - dino_forward does NOT wrap itself in torch.no_grad() (an earlier version of this class
        did -- see notes/deviations.md). That blocked gradients from flowing back through it,
        which breaks CodecLoss's DINO-latent-consistency term: it needs to backprop from DINO's
        features on the *reconstruction* through the decoder and bottleneck. This module's own
        weights still never get gradients (requires_grad_(False) in __init__, set once and never
        undone) -- only the *input's* gradient path is left open, matching mira's own convention
        of leaving no_grad to the caller (RAEEncoder.forward doesn't wrap its DINO call either).
    """

    def __init__(self, dino_model: str = "dinov3_vitb16", require_pretrained: bool = True):
        super().__init__()
        assert dino_model in DINO_DIM, f"unsupported dino_model {dino_model!r}"
        self.dino_model_name = dino_model
        self.dino_dim = DINO_DIM[dino_model]
        self.patch_size = PATCH_SIZE

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

    def dino_forward(self, x: Tensor) -> Tensor:
        """x: (b, t, 3, h, w) video, values in [0, 1]. Returns (b, t, dino_dim, h', w').

        No internal no_grad(): the caller decides. Call it under torch.no_grad() yourself (as the
        encoder side of a training step should, since the *target* features never need a gradient)
        when the input doesn't need one; leave it unwrapped when it does (the DINO-latent-
        consistency loss backprops through this call onto the reconstruction).
        """
        b, t, _, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.image_normalization(x)
        new_h = self.patch_size * (h // self.patch_size)
        new_w = self.patch_size * (w // self.patch_size)
        x = torch.nn.functional.interpolate(x, (new_h, new_w), mode="bilinear", antialias=True)
        features = self.dino_model.get_intermediate_layers(x, n=1, norm=True, reshape=True)[0]
        return rearrange(features, "(b t) c h w -> b t c h w", b=b, t=t)
