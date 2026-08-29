"""CPU-only checks for MyBottleneck's use_shallow_texture_branch (mini_mira.codec.bottleneck) --
no GPU needed. Confirms the flag-off path is unchanged, the flag-on path actually uses the
shallow layer and gets real gradient, and a decoder checkpoint is unaffected by this change
(it only ever sees the bottleneck's output, not its internal input width).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from einops import rearrange

from mini_mira.codec.bottleneck import MyBottleneck, StridedConvBottleneckConfig
from mini_mira.codec.decoder import ViTDecoderConfig, ViTVideoDecoder

torch.manual_seed(0)

BASE = dict(dino_dim=16, latent_dim=4, stride=2, temporal_stride=1)


def _fake_layers(n_layers=4, b=2, t=4, h=8, w=8, c=16) -> list[torch.Tensor]:
    return [torch.randn(b, t, c, h, w) for _ in range(n_layers)]


# --- Check 1: flag off -- in_channels/texture_proj unchanged, output matches manual recompute ---
off = MyBottleneck(StridedConvBottleneckConfig(**BASE, use_shallow_texture_branch=False))
assert off.texture_proj is None
assert off.projection.in_channels == BASE["dino_dim"]
layers = _fake_layers()
with torch.no_grad():
    out = off(layers)
    # Manually recompute today's own blend + projection, bypassing forward()'s branching --
    # proves the flag-off path really is the same math, not just "no crash".
    deep = torch.stack(layers, dim=0).mean(dim=0) + layers[-1]
    b, t = deep.shape[:2]
    expected = rearrange(off._project(rearrange(deep, "b t c h w -> (b t) c h w")), "(b t) c h w -> b t c h w", b=b, t=t)
assert torch.equal(out, expected), "flag-off output must exactly match the pre-existing blend+project math"
print("[PASS] use_shallow_texture_branch=False: no texture_proj, output matches today's own math exactly")

# --- Check 2: flag on -- wider projection, real gradient reaches both branches ---
on = MyBottleneck(StridedConvBottleneckConfig(**BASE, use_shallow_texture_branch=True, shallow_texture_channels=8))
assert on.texture_proj is not None
assert on.projection.in_channels == BASE["dino_dim"] + 8
layers2 = [t.clone().requires_grad_(True) for t in _fake_layers()]
out2 = on(layers2)
out2.sum().backward()
assert on.texture_proj.weight.grad is not None and on.texture_proj.weight.grad.abs().sum() > 0, (
    "texture_proj must receive a real, nonzero gradient"
)
assert on.projection.weight.grad is not None and on.projection.weight.grad.abs().sum() > 0, (
    "the main projection must still receive a real, nonzero gradient"
)
assert layers2[0].grad is not None and layers2[0].grad.abs().sum() > 0, (
    "gradient must reach the shallow layer input, not just the deep blend"
)
print("[PASS] use_shallow_texture_branch=True: wider projection, real gradient reaches both branches")

# --- Check 3: output shape (what the decoder sees) is IDENTICAL whether the flag is on or off ---
assert out.shape == out2.shape, "bottleneck output shape must not depend on this flag -- decoder must be unaffected"
print("[PASS] Bottleneck output shape unchanged by the flag -- confirms the decoder doesn't need to change")

# --- Check 4: a decoder checkpoint built independent of this flag loads and runs against either ---
decoder_config = ViTDecoderConfig(
    latent_dim=BASE["latent_dim"], stride=BASE["stride"], width=16, depth=1, num_heads=2,
    mlp_dim_multiplier=2, out_channels=3, patch_size=2, patch_size_t=1,
)
decoder = ViTVideoDecoder(decoder_config)
decoder.eval()
with torch.no_grad():
    recon_off = decoder(off(layers))
    recon_on = decoder(on(layers2))
assert recon_off.shape == recon_on.shape, "the SAME decoder must produce the same-shaped output from either bottleneck"
print("[PASS] The same decoder (no changes, no retraining) runs cleanly against both bottleneck configs")

print("\nAll bottleneck texture-branch checks passed.")
