"""Behavioral checks for VjepaModel (mini_mira.codec.vjepa), mirroring verify_dino.py's role.

Uses require_pretrained=False as the primary (and only) tier -- V-JEPA 2.1 isn't gated like
DINOv3, so this is a real parametrized forward pass, not a fake, just with random-init weights
(same convention as verify_codec_training.py's DinoModel(require_pretrained=False)). No
_FakeVjepa (that pattern belongs to scripts exercising LatentWorldModel's dino= seam, not this
one) and no MyPipeline end-to-end section (no use_real_vjepa-equivalent wiring exists yet).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from mini_mira.codec.vjepa import VjepaModel

with torch.no_grad():
    vjepa = VjepaModel(require_pretrained=False)
    vjepa.eval()

    # --- frozen ---
    n_trainable = sum(p.requires_grad for p in vjepa.parameters())
    assert n_trainable == 0, f"VjepaModel has {n_trainable} trainable parameters -- should be frozen"
    print("[PASS] frozen: 0 trainable parameters")

    # --- shape at mini_mira's actual pipeline resolution ---
    # 288x512 at patch_size=16 -> 18x32; 8 frames at tubelet_size=2 -> 4 temporal tokens.
    video = torch.rand(1, 8, 3, 288, 512)
    out = vjepa.dino_forward(video)
    expected = (1, 4, vjepa.dino_dim, 18, 32)
    assert out.shape == expected, f"got {out.shape}, expected {expected}"
    print(f"[PASS] shape at pipeline resolution: {tuple(out.shape)}")

    # --- non-patch-aligned resolution + odd frame count in one check ---
    # 300x500 -> rounds down to 288x496 (18x31 patches). t=3 (odd) also exercises the tubelet
    # floor-drop (3 // 2 = 1), something DINO has no equivalent for.
    odd_video = torch.rand(1, 3, 3, 300, 500)
    odd_out = vjepa.dino_forward(odd_video)
    odd_expected = (1, 1, vjepa.dino_dim, 18, 31)
    assert odd_out.shape == odd_expected, f"got {odd_out.shape}, expected {odd_expected}"
    print(f"[PASS] non-aligned resolution (300x500) + odd frame count (t=3): {tuple(odd_out.shape)}")

    # --- not degenerate output ---
    assert torch.isfinite(out).all(), "V-JEPA features contain NaN/Inf"
    assert out.std().item() > 0.0, f"V-JEPA features look degenerate (std={out.std().item():.6f})"
    print(f"[PASS] output is finite and non-degenerate: mean={out.mean().item():.4f}, std={out.std().item():.4f}")

    # --- multi-layer mode (last_layer_only=False) ---
    vjepa_multi = VjepaModel(require_pretrained=False, last_layer_only=False)
    multi_out = vjepa_multi.dino_forward(video)
    assert isinstance(multi_out, list), f"expected list, got {type(multi_out)}"
    assert len(multi_out) == 4, f"expected 4 layers (2,5,8,11), got {len(multi_out)}"
    for f in multi_out:
        assert f.shape == expected, f"layer got {f.shape}, expected {expected}"
    print(f"[PASS] multi-layer mode: {len(multi_out)} layers, each {tuple(multi_out[0].shape)}")

print("\nAll V-JEPA 2.1 checks passed.")