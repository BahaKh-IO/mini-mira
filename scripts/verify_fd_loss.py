"""CPU-friendly mechanism proof for FD-loss (arXiv:2604.28190v1), the additive fine-tuning term
added to LatentWorldModel.diffusion_loss -- no real weights, no network, no GPU. See
notes on mini_mira.world_model.fd_loss for the mechanism itself.

Doesn't re-run verify_world_model_training_vjepa.py's own checks (temporal_downsampling, the
base diffusion/PSD/scheduled-sampling loss, checkpoint round-trip) -- those are unaffected by this
change and already covered there. This script is scoped to what's actually new.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from mira.data.batch import VideoActionBatch
from mira.world_model.actions_config import ActionConfig, ActionTensors

from mini_mira.codec.bottleneck import StridedConvBottleneckConfig
from mini_mira.codec.decoder import ViTDecoderConfig
from mini_mira.world_model.diffusion_transformer import LatentWorldModelConfig
from mini_mira.world_model.fd_loss import FDLossEMAState, RealFrechetStats, differentiable_frechet_distance
from mini_mira.world_model.latent_world_model import LatentWorldModel

torch.manual_seed(0)

DINO_DIM = 16  # small on purpose -- eigh on the real 768-dim case is fine, this just keeps CPU checks fast


class _FakeVjepaLike(torch.nn.Module):
    """Minimal frozen-encoder stand-in -- a real (if tiny) Conv2d, not a random tensor generator,
    so gradients genuinely flow through it to the decoder's output, same spirit as
    verify_world_model_training_vjepa.py's own _FakeVjepaLike."""

    TUBELET_SIZE = 2

    def __init__(self, dino_dim: int = DINO_DIM, patch_size: int = 16):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, dino_dim, kernel_size=patch_size, stride=patch_size)
        self.dino_dim = dino_dim
        self.tubelet_size = self.TUBELET_SIZE
        self.requires_grad_(False)
        self.eval()

    def dino_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        t_prime = t // self.tubelet_size
        x = x[:, : t_prime * self.tubelet_size]
        x = x.unflatten(dim=1, sizes=(t_prime, self.tubelet_size)).mean(dim=2)
        x = x.flatten(0, 1)
        x = self.conv(x)
        return x.unflatten(0, (b, t_prime))


def _trainable_params(model: LatentWorldModel) -> list[torch.nn.Parameter]:
    return list(model.world_model.parameters()) + list(model.action_encoder.parameters()) + [model.bos]


NUM_KEYS = 9
HEIGHT = WIDTH = 64
RAW_FRAMES = 8
BATCH_SIZE = 2

action_config = ActionConfig(valid_keys=[f"k{i}" for i in range(NUM_KEYS)])
actions = ActionTensors(config=action_config, batch_size=BATCH_SIZE)
actions.key_presses = torch.randint(0, 2, (BATCH_SIZE, RAW_FRAMES, NUM_KEYS), dtype=torch.int32)
actions.mouse_movements = torch.zeros((BATCH_SIZE, RAW_FRAMES, 2), dtype=torch.float32)
video = torch.randint(0, 256, (BATCH_SIZE, RAW_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.uint8)
batch = VideoActionBatch(video=video, actions=actions)

bottleneck_config = StridedConvBottleneckConfig(temporal_stride=1, dino_dim=DINO_DIM)
decoder_config = ViTDecoderConfig(latent_dim=bottleneck_config.latent_dim)


def _build_model(config: LatentWorldModelConfig, **kwargs) -> LatentWorldModel:
    return LatentWorldModel(
        config, bottleneck_config, decoder_config, num_keys=NUM_KEYS, codec_checkpoint=None,
        dino=_FakeVjepaLike(dino_dim=DINO_DIM), **kwargs,
    )


# --- Check 1: differentiable_frechet_distance -- ~0 for identical distributions, real positive
# finite value for different ones, gradients flow into both mean_g and cov_g. ---
mean_r = torch.randn(DINO_DIM, dtype=torch.double)
A = torch.randn(DINO_DIM, DINO_DIM, dtype=torch.double)
cov_r = A @ A.T + 0.1 * torch.eye(DINO_DIM, dtype=torch.double)
real_stats = RealFrechetStats.from_mean_cov(mean_r, cov_r)

fd_same = differentiable_frechet_distance(real_stats, mean_r.clone(), cov_r.clone())
assert fd_same.item() < 1e-4, f"FD between identical distributions should be ~0, got {fd_same.item()}"

mean_g_diff = mean_r + 5.0
fd_diff = differentiable_frechet_distance(real_stats, mean_g_diff, cov_r.clone())
assert torch.isfinite(fd_diff), "FD between different distributions is not finite"
assert fd_diff.item() > 1.0, f"FD between clearly-different distributions should be well above 0, got {fd_diff.item()}"

mean_g_grad = (mean_r + 1.0).clone().requires_grad_(True)
cov_g_grad = cov_r.clone().requires_grad_(True)
fd_grad = differentiable_frechet_distance(real_stats, mean_g_grad, cov_g_grad)
fd_grad.backward()
assert mean_g_grad.grad is not None and torch.isfinite(mean_g_grad.grad).all(), "no/non-finite gradient into mean_g"
assert cov_g_grad.grad is not None and torch.isfinite(cov_g_grad.grad).all(), "no/non-finite gradient into cov_g"
print(
    f"[PASS] differentiable_frechet_distance: {fd_same.item():.2e} for identical distributions, "
    f"{fd_diff.item():.2f} (finite, positive) for different ones, gradients flow into mean_g/cov_g"
)

# --- Check 2: FDLossEMAState's update rule -- first call seeds directly (no blend against an
# arbitrary zero/identity init), later calls follow the paper's EMA formula exactly. ---
ema = FDLossEMAState(dim=4, beta=0.9)
batch1 = torch.randn(5, 4)
mu1, _cov1 = ema.update(batch1)
expected_mu1 = batch1.double().mean(dim=0)
assert torch.allclose(mu1, expected_mu1, atol=1e-6), "first update should seed the EMA directly from the batch"

batch2 = torch.randn(5, 4)
mu2, _cov2 = ema.update(batch2)
expected_mu2 = 0.9 * mu1.detach() + 0.1 * batch2.double().mean(dim=0)
assert torch.allclose(mu2, expected_mu2, atol=1e-6), "second update should blend via beta*ema + (1-beta)*batch exactly"
print("[PASS] FDLossEMAState: first call seeds directly, later calls follow the paper's EMA blend exactly")

# --- Check 3: fd_loss_weight=0 (the default) is a provable no-op -- no fd_ema/real-stats buffers
# ever get built, no loss_fd key appears, same discipline already used for use_shallow_texture_branch. ---
model_off = _build_model(LatentWorldModelConfig(hidden_dim=32, depth=2, num_heads=2, mlp_dim_multiplier=2))
assert model_off.fd_ema is None and not model_off.fd_loss_enabled, "fd_loss_weight=0 must not build any FD-loss state"
assert not hasattr(model_off, "fd_real_mean"), "fd_loss_weight=0 must not register real-stats buffers at all"
out_off = model_off(batch)
assert "loss_fd" not in out_off, "fd_loss_weight=0 must not add a loss_fd term"
print(f"[PASS] fd_loss_weight=0 no-op: loss_total={out_off['loss_total'].item():.4f}, no loss_fd key, no FD state built")

# --- Check 4: fd_loss_weight>0 end-to-end -- finite loss_fd, gradients reach every trainable
# param, using the exact reused z_t/pred_v/tau path (no extra forward pass for generation). ---
config_on = LatentWorldModelConfig(
    hidden_dim=32, depth=2, num_heads=2, mlp_dim_multiplier=2, fd_loss_weight=1.0, fd_loss_ema_decay=0.9,
)
model_on = _build_model(config_on, real_frechet_mean=mean_r.clone(), real_frechet_cov=cov_r.clone())
model_on.train()
model_on.warm_start_fd_loss(batch)
assert bool(model_on.fd_ema.initialized), "warm_start_fd_loss should mark the EMA state initialized"

optimizer_on = torch.optim.Adam(_trainable_params(model_on), lr=2e-3)
out_on = model_on(batch)
assert torch.isfinite(out_on["loss_total"]), "loss_total not finite with fd_loss_weight>0"
assert torch.isfinite(out_on["loss_fd"]), "loss_fd not finite"
out_on["loss_total"].backward()
for p in _trainable_params(model_on):
    assert p.grad is not None, "a trainable parameter has no gradient with fd_loss_weight>0"
    assert torch.isfinite(p.grad).all(), "a trainable parameter has a non-finite gradient"
optimizer_on.step()
print(
    f"[PASS] fd_loss_weight>0 end-to-end: loss_total={out_on['loss_total'].item():.4f}, "
    f"loss_fd={out_on['loss_fd'].item():.4f}, gradients reach every trainable param"
)

print("\nAll FD-loss checks passed.")
