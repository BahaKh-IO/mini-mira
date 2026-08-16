"""CPU-friendly mechanism proof that the world model trains, mirroring verify_codec_training.py's
role for the codec: no real weights, no network, no GPU. Four checks in one script:
  1. Overfit a fixed batch; loss_total must drop substantially.
  2. Frozen-vs-trainable gradient isolation after one backward pass.
  3. PSD self-distillation mechanism (both psd_weight and psd_loss_prob variants).
  4. Checkpoint save/load round-trip.

Uses codec_checkpoint=None (random-init frozen bottleneck/decoder, same convention as
LatentWorldModel's own docstring) and a local _FakeDino stand-in instead of the real DinoModel --
DinoModel's loader depends on a private torch.hub API (notes/deviations.md 1.14) that is
currently broken on this specific dev machine (confirmed pre-existing: it also blocks the
already-shipped verify_codec_training.py, unrelated to this script). _FakeDino is a real (if
tiny) Conv2d, not a random generator, so it genuinely responds to its input -- same spirit as
deviations.md 1.18's own FakeDino precedent.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Real mira, cloned as a sibling repo -- only for the two lightweight batch/action dataclasses
# and the LR scheduler; no real mira weights or network access needed for any of it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from einops import rearrange
from mira.data.batch import VideoActionBatch
from mira.training.lr_schedule import WarmupConstantCosineDecayLR
from mira.world_model.actions_config import ActionConfig, ActionTensors

from mini_mira.codec.bottleneck import StridedConvBottleneckConfig
from mini_mira.codec.decoder import ViTDecoderConfig
from mini_mira.world_model.action_encoder import ActionEncoder
from mini_mira.world_model.checkpoint import load_checkpoint, save_checkpoint
from mini_mira.world_model.diffusion_transformer import DiffusionTransformer, LatentWorldModelConfig
from mini_mira.world_model.latent_world_model import LatentWorldModel

torch.manual_seed(0)


class _FakeDino(torch.nn.Module):
    """Local stand-in for DinoModel -- see module docstring for why. A real (if tiny) Conv2d, not
    a random tensor generator, so it genuinely responds to its input."""

    def __init__(self, dino_dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, dino_dim, kernel_size=patch_size, stride=patch_size)

    def dino_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.conv(x)
        return rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)


def _trainable_params(model: LatentWorldModel) -> list[torch.nn.Parameter]:
    return list(model.world_model.parameters()) + list(model.action_encoder.parameters()) + [model.bos]


# --- Fixed synthetic batch, reused by every check below ---
NUM_KEYS = 9
HEIGHT = WIDTH = 64
RAW_FRAMES = 8
BATCH_SIZE = 1

action_config = ActionConfig(valid_keys=[f"k{i}" for i in range(NUM_KEYS)])
actions = ActionTensors(config=action_config, batch_size=BATCH_SIZE)
actions.key_presses = torch.randint(0, 2, (BATCH_SIZE, RAW_FRAMES, NUM_KEYS), dtype=torch.int32)
actions.mouse_movements = torch.zeros((BATCH_SIZE, RAW_FRAMES, 2), dtype=torch.float32)
video = torch.randint(0, 256, (BATCH_SIZE, RAW_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.uint8)
batch = VideoActionBatch(video=video, actions=actions)

bottleneck_config = StridedConvBottleneckConfig()
decoder_config = ViTDecoderConfig()
world_model_config = LatentWorldModelConfig(hidden_dim=32, depth=2, num_heads=2, mlp_dim_multiplier=2)


def _build_model(config: LatentWorldModelConfig) -> LatentWorldModel:
    return LatentWorldModel(
        config, bottleneck_config, decoder_config, num_keys=NUM_KEYS, codec_checkpoint=None,
        dino=_FakeDino(dino_dim=bottleneck_config.dino_dim),
    )


# --- Check 1: overfit ---
model = _build_model(world_model_config)
model.train()

optimizer = torch.optim.Adam(_trainable_params(model), lr=2e-3)
lr_scheduler = WarmupConstantCosineDecayLR(optimizer, warmup_steps=10, constant_steps=0, decay_steps=290, min_lr=2e-4)

STEPS = 300
losses: list[float] = []
for step in range(STEPS):
    optimizer.zero_grad()
    out = model(batch)
    out["loss_total"].backward()
    optimizer.step()
    lr_scheduler.step()
    losses.append(out["loss_total"].item())

# Flow matching resamples noise/tau every step, so a single-step-to-single-step comparison is
# noisy -- average the first/last 20 steps instead of comparing the raw endpoints.
early = sum(losses[:20]) / 20
late = sum(losses[-20:]) / 20
drop = (early - late) / early
assert drop > 0.5, f"loss only dropped {drop:.1%} (early={early:.4f}, late={late:.4f}), expected >50%"
print(f"[PASS] overfit: loss {early:.4f} -> {late:.4f} ({drop:.1%} drop over {STEPS} steps)")

# --- Check 2: frozen-vs-trainable gradient isolation (reuses the trained model above) ---
optimizer.zero_grad()
out = model(batch)
out["loss_total"].backward()

for frozen_name, frozen_module in (("bottleneck", model.bottleneck), ("decoder", model.decoder), ("dino", model.dino)):
    for name, p in frozen_module.named_parameters():
        assert p.grad is None, f"{frozen_name}.{name} unexpectedly has a gradient (should be frozen)"

n_trainable = 0
for p in _trainable_params(model):
    assert p.grad is not None, "a trainable parameter has no gradient"
    n_trainable += 1
print(f"[PASS] frozen/trainable gradient isolation: {n_trainable} trainable tensors have grads, "
      f"frozen codec (bottleneck/decoder/dino) has none")

# --- Check 3: PSD self-distillation mechanism, both variants ---
psd_weight_config = LatentWorldModelConfig(hidden_dim=32, depth=2, num_heads=2, mlp_dim_multiplier=2, psd_weight=0.1)
model_psd_weight = _build_model(psd_weight_config)
out_psd_weight = model_psd_weight(batch)
assert "loss_psd" in out_psd_weight, "psd_weight>0 must produce a loss_psd entry"
assert torch.isfinite(out_psd_weight["loss_psd"]), "loss_psd is not finite"
assert out_psd_weight["loss_total"].item() != out_psd_weight["loss_diffusion"].item(), (
    "psd_weight>0 must change loss_total relative to loss_diffusion"
)
print(
    # loss_psd prints small at model init -- zero-initialized AdaLN gates (mini_mira.ml.init)
    # make every fresh model's velocity predictions near-zero, so student/teacher start out
    # near-identical (see verify_conditioning.py's own _unzero_adaln_gates for the same root
    # cause). The assert above already confirms it's genuinely nonzero, not broken.
    f"[PASS] PSD (psd_weight=0.1): loss_diffusion={out_psd_weight['loss_diffusion'].item():.4f} "
    f"loss_psd={out_psd_weight['loss_psd'].item():.6e} loss_total={out_psd_weight['loss_total'].item():.6f}"
)

psd_prob_config = LatentWorldModelConfig(hidden_dim=32, depth=2, num_heads=2, mlp_dim_multiplier=2, psd_loss_prob=1.0)
model_psd_prob = _build_model(psd_prob_config)
out_psd_prob = model_psd_prob(batch)  # psd_loss_prob=1.0 -- always taken, not stochastic here
assert "loss_psd" in out_psd_prob, "psd_loss_prob>0 must produce a loss_psd entry"
assert torch.isfinite(out_psd_prob["loss_psd"]), "loss_psd is not finite"
assert out_psd_prob["loss_total"].item() != out_psd_prob["loss_diffusion"].item(), (
    "psd_loss_prob=1.0 must change loss_total relative to loss_diffusion"
)
print(
    f"[PASS] PSD (psd_loss_prob=1.0): loss_diffusion={out_psd_prob['loss_diffusion'].item():.4f} "
    f"loss_psd={out_psd_prob['loss_psd'].item():.6e} loss_total={out_psd_prob['loss_total'].item():.6f}"
)

# --- Check 4: checkpoint save/load round-trip (reuses the trained model from checks 1-2) ---
torch.manual_seed(123)
with torch.no_grad():
    loss_before = model(batch)["loss_total"].item()

tmp_path = Path(tempfile.mkdtemp()) / "wm_checkpoint_test.pth"
save_checkpoint(
    tmp_path, STEPS - 1, model.world_model, model.action_encoder, model.bos, optimizer, lr_scheduler,
    wandb_run_id="fake-run-id-for-test",
)

# Mutate: replace the trainable components with fresh random-init ones on the SAME model instance
# (same frozen dino/bottleneck/decoder throughout) -- isolates this check to exactly what the
# checkpoint covers, rather than confounding it with a second model's independently-random codec.
model.world_model = DiffusionTransformer(world_model_config)
model.action_encoder = ActionEncoder(
    num_keys=NUM_KEYS, hidden_dim=world_model_config.hidden_dim, temporal_downsampling=model.temporal_downsampling
)
with torch.no_grad():
    model.bos.copy_(torch.randn_like(model.bos))

optimizer2 = torch.optim.Adam(_trainable_params(model), lr=2e-3)
lr_scheduler2 = WarmupConstantCosineDecayLR(
    optimizer2, warmup_steps=10, constant_steps=0, decay_steps=290, min_lr=2e-4
)
resume_step, resume_wandb_run_id = load_checkpoint(
    tmp_path, model.world_model, model.action_encoder, model.bos, optimizer2, lr_scheduler2
)
assert resume_step == STEPS, f"expected resume step {STEPS}, got {resume_step}"
assert resume_wandb_run_id == "fake-run-id-for-test", f"wandb_run_id did not round-trip: {resume_wandb_run_id!r}"

torch.manual_seed(123)  # same z_0/tau draws as loss_before, isolating the comparison to weights
with torch.no_grad():
    loss_after = model(batch)["loss_total"].item()

assert abs(loss_before - loss_after) < 1e-6, f"checkpoint round-trip mismatch: {loss_before} vs {loss_after}"
print(
    f"[PASS] checkpoint round-trip: loss identical before/after save+load ({loss_before:.6f}), "
    f"wandb_run_id round-tripped correctly"
)

print("\nAll world-model training checks passed.")
