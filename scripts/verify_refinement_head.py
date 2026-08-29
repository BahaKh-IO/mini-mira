"""CPU-only checks for PixelRefinementHead (mini_mira.codec.decoder) and its checkpoint-resume
path (mini_mira.codec.checkpoint) -- no GPU, no real training script invocation. Written before
touching the live H100 run: proves the zero-init-so-it's-a-no-op claim and the resume-with-a-new-
submodule path both actually hold, rather than trusting them by construction.
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


# --- Check 1: refinement head's last conv is zero right after construction ---
head_decoder = _decoder(use_refinement_head=True, refinement_channels=8, refinement_num_layers=3)
last_conv = head_decoder.refinement_head.last_conv
assert torch.all(last_conv.weight == 0), "last conv weight must be zero-initialized"
assert torch.all(last_conv.bias == 0), "last conv bias must be zero-initialized"
print("[PASS] PixelRefinementHead's last conv is zero-initialized after ViTVideoDecoder.__init__")

# --- Check 2: with the head on (zero-init), output is bit-identical to the head being off ---
# Deliberately NOT comparing two separately-constructed decoders here: ViTVideoDecoder.__init__
# calls self.apply(init_weights) over its WHOLE module tree, using the global RNG, in tree-
# traversal order -- adding the refinement_head submodule changes that traversal, which (via the
# shared RNG stream) can shift which random draws land on the OTHER, shared-path layers even under
# the same manual_seed. That's a real RNG-ordering fact, not a bug in the head itself, but it means
# "same seed -> identical shared-path weights" doesn't hold once the tree shape differs. Instead,
# compare ONE instance's real forward() against manually replaying its own submodules up to (but
# not through) the refinement head -- isolates exactly what the head's presence changes.
head_decoder2 = _decoder(use_refinement_head=True, refinement_channels=8, refinement_num_layers=3)
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

assert torch.equal(out_with_head, out_without_head), "zero-initialized refinement head must be a true no-op"
print("[PASS] Enabling the refinement head (zero-init) does not change decoder output at all")

# --- Check 3: last_layer_weight points at the true final layer, with and without the head ---
plain_decoder = _decoder(use_refinement_head=False)
assert plain_decoder.last_layer_weight is plain_decoder.patch_unembed.proj.weight
assert head_decoder2.last_layer_weight is head_decoder2.refinement_head.last_conv.weight
print("[PASS] last_layer_weight tracks the real final layer in both configurations")

# --- Check 4: the head's own last conv gets a real gradient (it can start moving off zero) ---
# NOT checking the EARLIER layers here -- with the last conv's weight at exactly zero, backprop
# through it multiplies the incoming gradient by that same zero weight matrix, so d(loss)/d(input
# to the last conv) is exactly zero on this very first call. That's expected, not a dead branch:
# it's the standard zero-init-residual dynamic (also used by e.g. LayerScale/adaLN-zero designs)
# -- only the last layer's OWN weight (grad = outer(grad_output, its input), no zero-weight
# multiply involved) can move on step one, and once it's nonzero, gradient starts reaching the
# earlier layers too on later steps. Documented on PixelRefinementHead itself.
head_decoder3 = _decoder(use_refinement_head=True, refinement_channels=8, refinement_num_layers=3)
z2 = _fake_latent(head_decoder3).requires_grad_(True)
out = head_decoder3(z2)
out.sum().backward()
last_conv3 = head_decoder3.refinement_head.last_conv
assert last_conv3.weight.grad is not None and last_conv3.weight.grad.abs().sum() > 0, (
    "refinement head's own last conv must receive a real, nonzero gradient (its weight is zero, "
    "not its gradient -- this is what lets it start moving away from zero)"
)
first_conv = head_decoder3.refinement_head.layers[0]
assert first_conv.weight.grad is None or first_conv.weight.grad.abs().sum() == 0, (
    "earlier layers should see exactly zero gradient on this first call -- confirms the zero-init "
    "residual dynamic is behaving as documented, not a coincidence"
)
print("[PASS] The head's last conv receives a real gradient (can move off zero); earlier layers "
      "correctly see none yet, exactly as the zero-init-residual design predicts")

# --- Check 5: resuming an OLD checkpoint (no refinement head) into a NEW decoder (with one) ---
old_bottleneck = torch.nn.Identity()  # save_checkpoint only calls .state_dict()/.load_state_dict()
old_decoder = _decoder(use_refinement_head=False)
old_optimizer = torch.optim.AdamW(old_decoder.parameters(), lr=1e-4)
old_scheduler = torch.optim.lr_scheduler.ConstantLR(old_optimizer)
tmp_ckpt = Path(tempfile.mkdtemp()) / "checkpoint.pth"
save_checkpoint(tmp_ckpt, step=41, bottleneck=old_bottleneck, decoder=old_decoder,
                optimizer=old_optimizer, lr_scheduler=old_scheduler)

new_bottleneck = torch.nn.Identity()
new_decoder = _decoder(use_refinement_head=True, refinement_channels=8, refinement_num_layers=3)
new_params = list(new_decoder.parameters())
new_optimizer = torch.optim.AdamW(new_params, lr=1e-4)
new_scheduler = torch.optim.lr_scheduler.ConstantLR(new_optimizer)
resumed_step, _ = load_checkpoint(
    tmp_ckpt, new_bottleneck, new_decoder, new_optimizer, new_scheduler,
    reset_optimizer_state=True, decoder_new_submodule_prefix="refinement_head.",
)
assert resumed_step == 42, f"expected to resume at step 42, got {resumed_step}"
# The refinement head's params were never in the old checkpoint -- load_state_dict(strict=False)
# must have left them exactly as constructed (zero-initialized), not touched them at all.
assert torch.all(new_decoder.refinement_head.last_conv.weight == 0)
print("[PASS] Resuming an old (no-head) checkpoint into a new (with-head) decoder works, and "
      "leaves the new head's own zero-init untouched")

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
