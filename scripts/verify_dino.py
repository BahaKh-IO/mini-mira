"""Behavioral checks for the real DINOv3 backbone (mini_mira.dino.DinoModel).

Kept separate from test_shapes.py deliberately: this script needs the real, gated pretrained
weights on disk (RS_DINO_WEIGHTS_DIR must be set) and runs an actual ViT-B/16 forward pass,
both much heavier requirements than the fake-tensor shape checks everywhere else in this
project. Nothing here should block a contributor who doesn't have DINOv3 weights downloaded.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from mini_mira.codec.bottleneck import StridedConvBottleneckConfig
from mini_mira.codec.dino import DinoModel
from mini_mira.pipeline import PipelineConfig, MyPipeline

if not os.environ.get("RS_DINO_WEIGHTS_DIR"):
    raise SystemExit(
        "RS_DINO_WEIGHTS_DIR is not set. Point it at the local directory containing "
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth before running this script."
    )

with torch.no_grad():
    dino = DinoModel()
    dino.eval()

    # --- frozen ---
    n_trainable = sum(p.requires_grad for p in dino.parameters())
    assert n_trainable == 0, f"DinoModel has {n_trainable} trainable parameters -- should be frozen"
    print("[PASS] frozen: 0 trainable parameters")

    # --- shape at mini_mira's actual pipeline resolution ---
    # 288x512 at patch_size=16 -> 18x32, matching StridedConvBottleneckConfig.dino_dim=768 and
    # the (2, 40, 768, 18, 32) DINO-shaped tensor test_shapes.py feeds the bottleneck.
    bottleneck_config = StridedConvBottleneckConfig()
    video = torch.rand(1, 3, 3, 288, 512)  # (b, t, c, h, w), values in [0, 1]
    out = dino.dino_forward(video)
    expected = (1, 3, bottleneck_config.dino_dim, 18, 32)
    assert out.shape == expected, f"got {out.shape}, expected {expected}"
    print(f"[PASS] shape at pipeline resolution: {tuple(out.shape)}")

    # --- resize-to-patch-multiple, for a resolution that ISN'T a multiple of 16 ---
    # 300x500 -> rounds down to 288x496 (18x31 patches) inside dino_forward. If this silently
    # crashed or mis-rounded, every video whose resolution doesn't tile evenly would break.
    odd_video = torch.rand(1, 2, 3, 300, 500)
    odd_out = dino.dino_forward(odd_video)
    odd_expected = (1, 2, bottleneck_config.dino_dim, 18, 31)
    assert odd_out.shape == odd_expected, f"got {odd_out.shape}, expected {odd_expected}"
    print(f"[PASS] non-patch-aligned input (300x500) resizes correctly: {tuple(odd_out.shape)}")

    # --- not degenerate output ---
    # A real trained backbone should produce features with real variance, not all-zero or NaN
    # (which is what a silently-broken weight load, e.g. a strict=False key mismatch, would
    # tend to produce instead of a clean crash).
    assert torch.isfinite(out).all(), "DINO features contain NaN/Inf"
    assert out.std().item() > 0.01, f"DINO features look degenerate (std={out.std().item():.6f})"
    print(f"[PASS] output is finite and non-degenerate: mean={out.mean().item():.4f}, std={out.std().item():.4f}")

    # --- end-to-end: MyPipeline with use_real_dino=True, raw video in, decoded video out ---
    # This exercises the actual wiring (pipeline.py's forward branches on the flag and calls
    # self.dino.dino_forward internally), not just DinoModel alone -- the exact thing a real
    # training loop would call. Small resolution/depths (PipelineConfig()'s own small defaults)
    # to keep this fast despite the real backbone.
    real_dino_config = PipelineConfig(use_real_dino=True)
    real_dino_pipeline = MyPipeline(real_dino_config)
    real_dino_pipeline.eval()

    raw_video = torch.rand(1, 4, 3, 32, 32)  # (b, t, c, h, w) in [0, 1] -- 32x32 is patch-aligned
    raw_actions = torch.randint(0, 2, (1, 2, real_dino_config.num_keys))
    decoded = real_dino_pipeline(raw_video, raw_actions)
    print(f"[PASS] end-to-end use_real_dino=True: raw video {tuple(raw_video.shape)} -> decoded {tuple(decoded.shape)}")

print("\nAll DINOv3 checks passed.")
