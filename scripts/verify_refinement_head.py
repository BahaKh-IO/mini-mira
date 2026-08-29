"""CPU-only checks for PixelRefinementHead (mini_mira.codec.decoder) and its checkpoint-resume
path (mini_mira.codec.checkpoint) -- no GPU, no real training script invocation. Covers the
LayerScale-based version (a real fix, not a rewrite for style: an earlier exact-zero-init version
of this head caused a real NaN crash on a real overfit run -- see decoder.py's own docstring).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from mini_mira.codec.checkpoint import load_checkpoint, save_checkpoint
from mini_mira.codec.decoder import ViTDecoderConfig, ViTVideoDecoder

torch.manual_seed(0)

BASE_CONFIG = dict(
    latent_dim=4, stride=2, width=32, depth=2, num_heads=4, mlp_dim_multiplier=2,
    causal=True, out_channels=3, patch_size=4, patch_size_t=2, eps=1e-6,
)


def _decoder(**overrides) -> ViTVideoDecoder:
    return ViTVideoDecoder(ViTDecoderConfig(**{**BASE_CONFIG, **overrides}))


def _fake_latent(decoder: ViTVideoDecoder) -> torch.Tensor:
    return torch.randn(2, 3, decoder.config.latent_dim, 5, 5)


LAYERSCALE_INIT = 1e-4

# --- Check 1: LayerScale gamma starts at layerscale_init (small, NOT zero) ---
head_decoder = _decoder(
    use_refinement_head=True, refinement_channels=8, refinement_num_layers=3, layerscale_init=LAYERSCALE_INIT,
)
gamma = head_decoder.refinement_head.layerscale.gamma
assert torch.allclose(gamma, torch.full_like(gamma, LAYERSCALE_INIT)), "gamma must start at layerscale_init"
assert torch.all(gamma != 0), "gamma must NOT be zero -- that's the exact bug this version fixes"
print("[PASS] PixelRefinementHead's LayerScale starts at layerscale_init, deliberately not zero")

# --- Check 2: with the head on, output is CLOSE to (not identical to) the head being off ---
# Deliberately NOT comparing two separately-constructed decoders here: ViTVideoDecoder.__init__
# calls self.apply(init_weights) over its WHOLE module tree, using the global RNG, in tree-
# traversal order -- adding the refinement_head submodule changes that traversal, which (via the
# shared RNG stream) can shift which random draws land on the OTHER, shared-path layers even under
# the same manual_seed. That's a real RNG-ordering fact, not a bug in the head itself, but it means
# "same seed -> identical shared-path weights" doesn't hold once the tree shape differs. Instead,
# compare ONE instance's real forward() against manually replaying its own submodules up to (but
# not through) the refinement head -- isolates exactly what the head's presence changes.
head_decoder2 = _decoder(
    use_refinement_head=True, refinement_channels=8, refinement_num_layers=3, layerscale_init=LAYERSCALE_INIT,
)
head_decoder2.eval()
z = _fake_latent(head_decoder2)
with torch.no_grad():
    out_with_head = head_decoder2(z)

    b, t = z.shape[:2]
    from einops import rearrange as _rearrange

    x = _rearrange(z, "b t c h w -> (b t) c h w")
    x = head_decoder2.from_latent(x)
    x = _rearrange(x, "(b t) c h w -> b t h w c", b=b, t=t)
    from mini_mira.ml.rope import spatial_rope, temporal_rope

    _, t2, h, w, _ = x.shape
    rope_spatial = spatial_rope(h, w, head_decoder2.head_dim, head_decoder2.config.rope_theta_spatial, x.device)
    rope_temporal = temporal_rope(t2, head_decoder2.head_dim, head_decoder2.config.rope_theta_temporal, x.device)
    for block in head_decoder2.blocks:
        x = block(x, rope_spatial, rope_temporal)
    out_without_head = torch.tanh(head_decoder2.patch_unembed(head_decoder2.norm_out(x)))

assert not torch.equal(out_with_head, out_without_head), (
    "the head must contribute something, however small -- LayerScale is near-zero, not exactly zero"
)
assert torch.allclose(out_with_head, out_without_head, atol=0.05), (
    "the head's initial contribution should still be small at layerscale_init=1e-4, not large"
)
print("[PASS] Enabling the refinement head makes a small, real, nonzero difference -- not a true "
      "no-op, but not a large jump either")

# --- Check 3: last_layer_weight points at the true final layer, with and without the head ---
plain_decoder = _decoder(use_refinement_head=False)
assert plain_decoder.last_layer_weight is plain_decoder.patch_unembed.proj.weight
assert head_decoder2.last_layer_weight is head_decoder2.refinement_head.layerscale.gamma
print("[PASS] last_layer_weight tracks the real final layer in both configurations")

# --- Check 4: EVERY layer in the head gets real, nonzero gradient from step one ---
# Different from the exact-zero version this replaces: nothing here is hard-zeroed anymore, so
# there's no "only the last layer can move first" dead-branch phase -- confirm that directly.
head_decoder3 = _decoder(
    use_refinement_head=True, refinement_channels=8, refinement_num_layers=3, layerscale_init=LAYERSCALE_INIT,
)
z2 = _fake_latent(head_decoder3).requires_grad_(True)
out = head_decoder3(z2)
out.sum().backward()
first_conv = head_decoder3.refinement_head.layers[0]
last_conv3 = head_decoder3.refinement_head.last_conv
gamma3 = head_decoder3.refinement_head.layerscale.gamma
for name, param in [("first conv", first_conv.weight), ("last conv", last_conv3.weight), ("layerscale gamma", gamma3)]:
    assert param.grad is not None and param.grad.abs().sum() > 0, f"{name} must receive a real, nonzero gradient"
print("[PASS] Every layer in the head (first conv through LayerScale gamma) gets real gradient "
      "from step one -- no dead-branch warmup phase like the exact-zero version had")

# --- Check 5: resuming an OLD checkpoint (no refinement head) into a NEW decoder (with one) ---
old_bottleneck = torch.nn.Identity()  # save_checkpoint only calls .state_dict()/.load_state_dict()
old_decoder = _decoder(use_refinement_head=False)
old_optimizer = torch.optim.AdamW(old_decoder.parameters(), lr=1e-4)
old_scheduler = torch.optim.lr_scheduler.ConstantLR(old_optimizer)
tmp_ckpt = Path(tempfile.mkdtemp()) / "checkpoint.pth"
save_checkpoint(tmp_ckpt, step=41, bottleneck=old_bottleneck, decoder=old_decoder,
                optimizer=old_optimizer, lr_scheduler=old_scheduler)

new_bottleneck = torch.nn.Identity()
new_decoder = _decoder(
    use_refinement_head=True, refinement_channels=8, refinement_num_layers=3, layerscale_init=LAYERSCALE_INIT,
)
new_params = list(new_decoder.parameters())
new_optimizer = torch.optim.AdamW(new_params, lr=1e-4)
new_scheduler = torch.optim.lr_scheduler.ConstantLR(new_optimizer)
resumed_step, _ = load_checkpoint(
    tmp_ckpt, new_bottleneck, new_decoder, new_optimizer, new_scheduler,
    reset_optimizer_state=True, decoder_new_submodule_prefix="refinement_head.",
)
assert resumed_step == 42, f"expected to resume at step 42, got {resumed_step}"
# The refinement head's params were never in the old checkpoint -- load_state_dict(strict=False)
# must have left them exactly as constructed (LayerScale at layerscale_init), not touched them.
new_gamma = new_decoder.refinement_head.layerscale.gamma
assert torch.allclose(new_gamma, torch.full_like(new_gamma, LAYERSCALE_INIT))
print("[PASS] Resuming an old (no-head) checkpoint into a new (with-head) decoder works, and "
      "leaves the new head's own LayerScale init untouched")

# --- Check 6: a REAL mismatch (missing key that ISN'T the new submodule) still raises ---
bad_state = dict(old_decoder.state_dict())
del bad_state["patch_unembed.proj.weight"]  # a genuine, unrelated missing key
another_new_decoder = _decoder(use_refinement_head=True, refinement_channels=8, refinement_num_layers=3)
try:
    from mini_mira.codec.checkpoint import _load_decoder_flexible

    _load_decoder_flexible(another_new_decoder, bad_state, "refinement_head.")
    raise AssertionError("expected RuntimeError on a real (non-refinement-head) missing key")
except RuntimeError:
    pass
print("[PASS] A real missing key (not the new submodule) still raises -- the tolerance is scoped, not a blanket swallow")

print("\nAll refinement-head checks passed.")
