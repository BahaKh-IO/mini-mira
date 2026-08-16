"""CPU-friendly mechanism proof for the full world-model eval suite (Frechet DINO/Inception
Distance, PSNR, LPIPS, SSIM, rollout video visualization), mirroring verify_world_model_training.py's
role: no real weights, no network, no GPU, no pytorch_fid/lpips installed (neither is on this dev
machine -- fake stand-ins exercise the mechanism exactly like _FakeDino already does for DINO).

Seven checks in one script:
  1. OnlineGaussian / frechet_distance / SlicedFrechetMetric on synthetic tensors.
  2. compute_psnr / compute_ssim directly.
  3. compute_lpips's per-video loop/flatten logic, fake lpips_fn.
  4. FullEvalMetrics end-to-end, fake inception/lpips_fn.
  5. compute_drift_metrics's new (model, z, z_t, n_context_latents) signature + compute_full_eval_
     metrics on the SAME shared rollout -- also regression-tests that removing the internal
     model.rollout() call from compute_drift_metrics didn't break anything.
  6. render_rollout_sample / draw_key_grid_video / overlay_video / add_prediction_border /
     video_to_uint8 -- pure tensor/PIL ops.
  7. write_video_ffmpeg / log_rollout_videos -- genuinely end-to-end here (not mechanism-only):
     this dev machine has imageio-ffmpeg's bundled binary, so a real tiny mp4 actually gets written.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from einops import rearrange
from mira.data.batch import VideoActionBatch
from mira.world_model.actions_config import ActionConfig, ActionTensors

from mini_mira.codec.bottleneck import StridedConvBottleneckConfig
from mini_mira.codec.decoder import ViTDecoderConfig
from mini_mira.world_model.diffusion_transformer import LatentWorldModelConfig
from mini_mira.world_model.eval_metrics import compute_drift_metrics
from mini_mira.world_model.full_eval_metrics import (
    FullEvalMetrics,
    OnlineGaussian,
    SlicedFrechetMetric,
    compute_full_eval_metrics,
    compute_lpips,
    compute_psnr,
    compute_ssim,
    frechet_distance,
)
from mini_mira.world_model.latent_world_model import LatentWorldModel
from mini_mira.world_model.rollout_visualization import log_rollout_videos, render_rollout_sample, write_video_ffmpeg

torch.manual_seed(0)


class _FakeDino(torch.nn.Module):
    """Same stand-in verify_world_model_training.py already uses -- avoids the real torch.hub-
    dependent DinoModel, currently broken on this dev machine (pre-existing, see notes/
    deviations.md 1.14/1.18), unrelated to this script."""

    def __init__(self, dino_dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.dino_dim = dino_dim
        self.conv = torch.nn.Conv2d(3, dino_dim, kernel_size=patch_size, stride=patch_size)

    def dino_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.conv(x)
        return rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)


class _FakeInception(torch.nn.Module):
    """Stand-in for pytorch_fid.inception.InceptionV3 -- real (if tiny) Conv2d, matches the real
    class's contract: one input tensor, returns a list of feature maps (one per requested block)."""

    def __init__(self, dim: int = 2048):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, dim, kernel_size=4, stride=4)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feat = self.conv(x).mean(dim=(-1, -2), keepdim=True)
        return [feat]


class _FakeLpipsFn(torch.nn.Module):
    """Stand-in for lpips.LPIPS(net='alex') -- matches the real callable's contract: two [-1,1]
    image tensors in, one distance-per-image tensor out."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.conv(x - y).mean(dim=(1, 2, 3), keepdim=True)


# --- Check 1: OnlineGaussian / frechet_distance / SlicedFrechetMetric ---
data = torch.randn(200, 8)
g1 = OnlineGaussian(dim=8)
g1.update(data[:100])
g1.update(data[100:])
g2 = OnlineGaussian(dim=8)
g2.update(data)
mean1, cov1 = g1.compute()
mean2, cov2 = g2.compute()
d_same = frechet_distance(mean1, cov1, mean2, cov2)
assert d_same < 1e-6, f"same distribution should give ~0 Frechet distance, got {d_same}"

g3 = OnlineGaussian(dim=8)
g3.update(data + 5.0)
mean3, cov3 = g3.compute()
d_diff = frechet_distance(mean1, cov1, mean3, cov3)
assert d_diff > 1.0, f"shifted distribution should give a large Frechet distance, got {d_diff}"
print(f"[PASS] OnlineGaussian/frechet_distance: same={d_same:.8f} shifted={d_diff:.4f}")

sliced = SlicedFrechetMetric(dim=8, num_slices=3)
for s in range(3):
    real_slice = torch.randn(20, 8)
    pred_slice = real_slice + 0.1 * torch.randn(20, 8)
    sliced.update(s, real_slice, pred_slice)
aggregate, per_slice = sliced.compute()
assert len(per_slice) == 3 and all(v >= 0 for v in per_slice) and aggregate >= 0
print(f"[PASS] SlicedFrechetMetric: aggregate={aggregate:.4f} per_slice={[f'{v:.4f}' for v in per_slice]}")

# --- Check 2: compute_psnr / compute_ssim ---
pred_img = torch.rand(2, 4, 3, 32, 32)
target_img = torch.rand(2, 4, 3, 32, 32)
psnr = compute_psnr(pred_img, target_img)
assert psnr.shape == (2, 4) and torch.isfinite(psnr).all()
print(f"[PASS] compute_psnr: shape={tuple(psnr.shape)} mean={psnr.mean().item():.2f}dB")

ssim = compute_ssim(pred_img, target_img)
assert ssim.shape == (2,) and torch.isfinite(ssim).all()
print(f"[PASS] compute_ssim: shape={tuple(ssim.shape)} mean={ssim.mean().item():.4f}")

# --- Check 3: compute_lpips with a fake lpips_fn ---
lpips_scores = compute_lpips(_FakeLpipsFn(), pred_img, target_img)
assert lpips_scores.numel() == 2 * 4 and torch.isfinite(lpips_scores).all()
print(f"[PASS] compute_lpips: shape={tuple(lpips_scores.shape)}")

# --- Check 4: FullEvalMetrics end-to-end ---
fake_dino_dim = 16
full_metrics_standalone = FullEvalMetrics(
    dino_dim=fake_dino_dim, fdd_slice_frames=2, num_slices=3, device="cpu",
    inception=_FakeInception(), lpips_fn=_FakeLpipsFn(),
)
b, t_gen = 2, 6
full_metrics_standalone.update(
    torch.rand(b, t_gen, 3, 16, 16), torch.rand(b, t_gen, 3, 16, 16),
    torch.randn(b, t_gen, fake_dino_dim, 4, 4), torch.randn(b, t_gen, fake_dino_dim, 4, 4),
)
scalars, curves = full_metrics_standalone.compute_and_reset()
assert set(scalars.keys()) == {"psnr", "lpips", "ssim", "frechet_dino_distance", "frechet_inception_distance"}
assert all(torch.isfinite(torch.tensor(v)) for v in scalars.values())
assert len(curves["fdd"]) == 3 and len(curves["fid"]) == 3
print(f"[PASS] FullEvalMetrics end-to-end: {scalars}")

# --- Check 5: compute_drift_metrics's new signature + compute_full_eval_metrics on ONE shared rollout ---
NUM_KEYS = 9
HEIGHT = WIDTH = 64
RAW_FRAMES = 12  # -> 6 latent frames at temporal_stride=2

action_config = ActionConfig(valid_keys=[f"k{i}" for i in range(NUM_KEYS)])
actions = ActionTensors(config=action_config, batch_size=1)
actions.key_presses = torch.randint(0, 2, (1, RAW_FRAMES, NUM_KEYS), dtype=torch.int32)
actions.mouse_movements = torch.zeros((1, RAW_FRAMES, 2), dtype=torch.float32)
video = torch.randint(0, 256, (1, RAW_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.uint8)
batch = VideoActionBatch(video=video, actions=actions)

bottleneck_config = StridedConvBottleneckConfig()
decoder_config = ViTDecoderConfig()
world_model_config = LatentWorldModelConfig(hidden_dim=32, depth=2, num_heads=2, mlp_dim_multiplier=2)

model = LatentWorldModel(
    world_model_config, bottleneck_config, decoder_config, num_keys=NUM_KEYS, codec_checkpoint=None,
    dino=_FakeDino(dino_dim=bottleneck_config.dino_dim),
)
model.eval()

N_CONTEXT_LATENTS = 3
z, z_t = model.rollout(batch, n_context_latents=N_CONTEXT_LATENTS, n_diffusion_steps=2, schedule_type="linear")

drift = compute_drift_metrics(model, z, z_t, N_CONTEXT_LATENTS)
assert set(drift.keys()) == {"dino_cos_drift", "dino_l2_drift", "latent_drift"}
assert all(torch.isfinite(v).all() for v in drift.values())
print(f"[PASS] compute_drift_metrics (new signature): shapes={ {k: tuple(v.shape) for k, v in drift.items()} }")

generated_video_frames = (6 - N_CONTEXT_LATENTS) * bottleneck_config.temporal_stride  # (6-3)*2=6
fdd_slice_frames = 3  # 2 slices of 3 frames
full_metrics_shared = FullEvalMetrics(
    dino_dim=bottleneck_config.dino_dim, fdd_slice_frames=fdd_slice_frames,
    num_slices=generated_video_frames // fdd_slice_frames, device="cpu",
    inception=_FakeInception(), lpips_fn=_FakeLpipsFn(),
)
compute_full_eval_metrics(model, z, z_t, N_CONTEXT_LATENTS, full_metrics_shared)
shared_scalars, _shared_curves = full_metrics_shared.compute_and_reset()
assert all(torch.isfinite(torch.tensor(v)) for v in shared_scalars.values())
print(f"[PASS] compute_full_eval_metrics on the same shared rollout: {shared_scalars}")

# --- Check 6: render_rollout_sample / draw_key_grid_video / overlay_video / add_prediction_border / video_to_uint8 ---
sample_video = torch.rand(RAW_FRAMES, 3, HEIGHT, WIDTH)
key_presses_sample = actions.key_presses[0]
rendered = render_rollout_sample(sample_video, key_presses_sample, n_context_frames=6)
assert rendered.shape == (RAW_FRAMES, 3, HEIGHT, WIDTH) and rendered.dtype == torch.uint8
print(f"[PASS] render_rollout_sample: shape={tuple(rendered.shape)} dtype={rendered.dtype}")

# --- Check 7: write_video_ffmpeg / log_rollout_videos -- genuinely end-to-end here ---
tmp_video_path = Path(tempfile.mkdtemp()) / "test_rollout.mp4"
write_video_ffmpeg(str(tmp_video_path), rendered, fps=10)
assert tmp_video_path.exists() and tmp_video_path.stat().st_size > 0
print(f"[PASS] write_video_ffmpeg: wrote {tmp_video_path.stat().st_size} bytes to {tmp_video_path}")

log_dir = Path(tempfile.mkdtemp())
log_rollout_videos([rendered, rendered], fps=10, step=0, wandb_enabled=False, output_dir=log_dir)
saved = list((log_dir / "rollout_previews").glob("*.mp4"))
assert len(saved) == 2 and all(p.stat().st_size > 0 for p in saved)
print(f"[PASS] log_rollout_videos: saved {len(saved)} files to {log_dir / 'rollout_previews'}")

print("\nAll full eval metrics checks passed.")
