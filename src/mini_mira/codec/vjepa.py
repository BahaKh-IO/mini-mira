"""Frozen V-JEPA 2.1 (ViT-B/16) feature extractor. Matches DinoModel's contract (dino_forward +
.dino_dim) so it drops into LatentWorldModel's `dino: nn.Module | None` seam unchanged.

Encoder only -- vjepa2_1_vit_base_384 returns (encoder, predictor); the predictor is discarded.

Loaded via a plain `git clone` of facebookresearch/vjepa2 (needs `git` on PATH), not
transformers.AutoModel (no 2.1 support yet, upstream PR unmerged) and not torch.hub's private
_get_cache_or_reload (broken on this dev machine, same fragility dino.py's loader already flags).
Cached at torch.hub's own directory convention so it isn't re-cloned every run. Sparse-checks out
only src/ and app/ -- also sidesteps a real Windows case-collision warning in configs/eval_2_1/
(vitG-384/ vs vitg-384/), a tree we never need; falls back to a plain full clone if the sparse
flags ever misbehave.

Known upstream bug (Meta's, not ours): src/hub/backbones.py ships VJEPA_BASE_URL pointed at
localhost:8300 ("for testing"), with the real CDN URL commented out above it. Patched below.

V-JEPA halves the frame count internally (tubelet_size=2) -- unlike DinoModel, dino_forward here
returns t // 2 frames, not t. Matters once this gets wired into a real bottleneck later. A t < 2
input (e.g. CodecLoss's random frame-subset sampling can hand this a single-frame chunk, fine for
DinoModel, not for a model that always needs frame pairs) is padded up to 2 by repeating the last
frame, not rejected -- see dino_forward.
"""

import contextlib
import subprocess
import sys
from pathlib import Path

import torch
import torch.hub
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VJEPA_REPO_URL = "https://github.com/facebookresearch/vjepa2.git"
VJEPA_BASE_URL_FIX = "https://dl.fbaipublicfiles.com/vjepa2"
VJEPA_DIM_EXPECTED = 768  # sanity check only -- source of truth is encoder.embed_dim
VJEPA_TUBELET_SIZE_EXPECTED = 2  # sanity check only -- source of truth is encoder.tubelet_size

# ViT-B's own hierarchical_layers (app/vjepa_2_1/models/vision_transformer.py) -- the only valid
# out_layers indices for this model. Same indices as DinoModel's own DEFAULT_DINO_LAYERS["dinov3_vitb16"].
DEFAULT_VJEPA_LAYERS = (2, 5, 8, 11)


# cuDNN's fused attention is measurably the right backend for this encoder's shape -- at
# 448x768x40 (a 26880-token joint space-time sequence, head_dim 64, bf16) it runs the attention
# in 12.5ms against FlashAttention-2's 20.1ms on an H100, and 1.7x on the single largest kernel
# in the step is worth taking. Listed first with set_priority so the rest stay as fallbacks
# rather than being disabled: any shape cuDNN can't serve still lands on flash exactly as before.
_ATTENTION_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
]


@contextlib.contextmanager
def _preferred_attention_backend():
    """Prefer cuDNN attention for the duration of an encoder forward.

    Upstream's RoPEAttention wraps its own SDPA call in `torch.backends.cuda.sdp_kernel()` -- the
    deprecated no-argument form, which re-enables every backend and so silently discards any
    preference an outer context established. Neutralizing that one call for the duration of the
    forward is what lets the preference below actually reach the attention; it changes nothing
    else about upstream's behavior (the call it replaces enables all backends, which is the state
    the surrounding sdpa_kernel already leaves them in).
    """
    original = torch.backends.cuda.sdp_kernel
    torch.backends.cuda.sdp_kernel = lambda *args, **kwargs: contextlib.nullcontext()
    try:
        with sdpa_kernel(_ATTENTION_BACKENDS, set_priority=True):
            yield
    finally:
        torch.backends.cuda.sdp_kernel = original


def _vjepa_repo_dir() -> Path:
    repo_dir = Path(torch.hub.get_dir()) / "facebookresearch_vjepa2_main"
    marker = repo_dir / "src" / "hub" / "backbones.py"
    if marker.exists():
        return repo_dir
    if repo_dir.exists():
        raise FileNotFoundError(f"{repo_dir} exists but looks incomplete (missing {marker})")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", VJEPA_REPO_URL, str(repo_dir)],
            check=True,
        )
        subprocess.run(["git", "sparse-checkout", "set", "src", "app"], cwd=str(repo_dir), check=True)
    except subprocess.CalledProcessError:
        # Sparse checkout unsupported/failed -- fall back to a plain full clone (proven to work,
        # just noisier: harmless case-collision warnings in configs/eval_2_1/ on Windows).
        subprocess.run(["git", "clone", VJEPA_REPO_URL, str(repo_dir)], check=True)
    return repo_dir


def _load_vjepa_encoder_fn():
    repo_dir = _vjepa_repo_dir()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    import src.hub.backbones as backbones  # noqa: PLC0415

    backbones.VJEPA_BASE_URL = VJEPA_BASE_URL_FIX  # real upstream bug fix, see module docstring
    return backbones.vjepa2_1_vit_base_384


class VjepaModel(nn.Module):
    """Frozen V-JEPA 2.1 ViT-B feature extractor, encoder only. One variant, so no dino_model
    param. last_layer_only/layer_indices mirror DinoModel's own interface -- unlike DINO, V-JEPA
    picks layers at construction time (out_layers=...), not per forward call. V-JEPA 2.1 isn't
    gated, so require_pretrained just toggles the real download.
    """

    def __init__(
        self,
        require_pretrained: bool = True,
        last_layer_only: bool = True,
        layer_indices: tuple[int, ...] | None = None,
    ):
        super().__init__()
        if last_layer_only and layer_indices is not None:
            raise ValueError("VjepaModel: pass either last_layer_only=True OR layer_indices=(...), not both.")
        self.layers = layer_indices if layer_indices is not None else (
            None if last_layer_only else DEFAULT_VJEPA_LAYERS
        )

        encoder_fn = _load_vjepa_encoder_fn()
        self.encoder, _predictor = encoder_fn(pretrained=require_pretrained, out_layers=self.layers)
        del _predictor

        self.dino_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size
        self.tubelet_size = self.encoder.tubelet_size
        assert self.dino_dim == VJEPA_DIM_EXPECTED, (
            f"vjepa2_1_vit_base_384 embed_dim changed upstream: {self.dino_dim} != {VJEPA_DIM_EXPECTED}"
        )
        assert self.tubelet_size == VJEPA_TUBELET_SIZE_EXPECTED, (
            f"vjepa2_1_vit_base_384 tubelet_size changed upstream: "
            f"{self.tubelet_size} != {VJEPA_TUBELET_SIZE_EXPECTED} -- code elsewhere (e.g. "
            f"train_world_model_vjepa.py's pre-flight assertions) hardcodes this expectation."
        )

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
        """x: (b, t, 3, h, w) in [0, 1]. Returns (b, t // tubelet_size, dino_dim, h', w') for the
        default single-layer case, or a list of that shape (one per entry in self.layers) when
        built with last_layer_only=False. t < tubelet_size is padded up to tubelet_size by
        repeating the last frame (so it returns exactly one output frame) instead of raising --
        see module docstring for why this matters.
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda), \
                _preferred_attention_backend():
            b, t, _, h, w = x.shape
            assert t >= 1, f"t={t} must be at least 1"
            if t < self.tubelet_size:
                x = torch.cat([x, x[:, -1:].repeat(1, self.tubelet_size - t, 1, 1, 1)], dim=1)
                t = self.tubelet_size
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.image_normalization(x)
            new_h = self.patch_size * (h // self.patch_size)
            new_w = self.patch_size * (w // self.patch_size)
            # Skipped when the input is already patch-aligned, which is the case for every real
            # training resolution here (448x768 is 28x48 whole patches). Resizing something to the
            # size it already is costs a full antialiased-bilinear pass over the whole clip for a
            # result identical to its input.
            if (new_h, new_w) != (h, w):
                x = torch.nn.functional.interpolate(x, (new_h, new_w), mode="bilinear", antialias=True)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)

            tokens = self.encoder(x)  # Tensor if self.layers is None, else list[Tensor] -- one
            # per requested layer, each (b, t' * h' * w', dino_dim), T-major flat order (confirmed
            # via PatchEmbed3D.forward: proj(x).flatten(2).transpose(1, 2) on a (B,C,T',H',W') tensor)

            t_prime = t // self.tubelet_size
            h_prime = new_h // self.patch_size
            w_prime = new_w // self.patch_size

            def _to_dino_shape(tok: Tensor) -> Tensor:
                features = tok.reshape(b, t_prime, h_prime, w_prime, self.dino_dim)
                return rearrange(features, "b t h w c -> b t c h w")

            if isinstance(tokens, list):
                return [_to_dino_shape(tok) for tok in tokens]
            return _to_dino_shape(tokens)