"""CPU-friendly mechanism proof for the V-JEPA-track world model, mirroring
verify_world_model_training.py's own role for DINO -- no real weights, no network, no GPU.

NOT a straight fork of that script's checks 1-5: those already exercise the shared, backbone-
agnostic machinery (diffusion loss, PSD, scheduled sampling, checkpoint round-trip) and would pass
identically here regardless of which encoder is injected -- rerunning them adds no new coverage.
See scripts/verify_world_model_training.py for that coverage; run it directly (it already passes
for both tracks, since LatentWorldModel doesn't know or care which encoder produced its input).

What THIS script checks instead, and why it needs its own stand-in:
verify_world_model_training.py's `_FakeDino` does not touch the time dimension at all (its
dino_forward returns the same t it was given), so it never exercises V-JEPA's one real structural
divergence from DINO: VjepaModel.dino_forward halves time internally (tubelet_size=2, see
mini_mira.codec.vjepa) before the bottleneck ever sees it. LatentWorldModel.__init__ accounts for
this by reading self.dino.tubelet_size (via getattr, defaulting to 1) to compute the TRUE
raw-frames-per-latent-frame ratio used for action/latent alignment -- see
mini_mira/world_model/latent_world_model.py's self.temporal_downsampling computation and its
surrounding comment for the full writeup of why this matters (notes/deviations.md 1.21). This
script's _FakeVjepaLike stand-in halves time the same way the real VjepaModel does, specifically to
prove that wiring is correct end-to-end: the computed ratio, the action-alignment slicing in
_encode(), and a full forward/backward pass all still work once an encoder actually reduces time.
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
from mini_mira.world_model.eval_metrics import compute_drift_metrics, decode_and_dino
from mini_mira.world_model.diffusion_transformer import DiffusionTransformer, LatentWorldModelConfig
from mini_mira.world_model.latent_world_model import LatentWorldModel

torch.manual_seed(0)


class _FakeVjepaLike(torch.nn.Module):
    """Local stand-in for VjepaModel -- see module docstring for why. A real (if tiny) Conv2d, not
    a random tensor generator (same spirit as verify_world_model_training.py's _FakeDino), but
    additionally halves the time dimension and exposes .tubelet_size, matching VjepaModel's own
    real dino_forward contract (vjepa.py) instead of DinoModel's (which never touches time)."""

    TUBELET_SIZE = 2

    def __init__(self, dino_dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, dino_dim, kernel_size=patch_size, stride=patch_size)
        self.tubelet_size = self.TUBELET_SIZE

    def dino_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        # Real VjepaModel pads t<tubelet_size by repeating the last frame (vjepa.py) rather than
        # raising -- mirrored here for the same reason, even though this script's own fixed
        # RAW_FRAMES below is already a multiple of TUBELET_SIZE so the pad path never triggers.
        if t < self.tubelet_size:
            x = torch.cat([x, x[:, -1:].repeat(1, self.tubelet_size - t, 1, 1, 1)], dim=1)
            t = self.tubelet_size
        # Average-pool adjacent frame pairs before the per-frame conv, mimicking V-JEPA's own
        # tubelet embedding collapsing pairs of raw frames into one token in time -- simpler than
        # a real 3D conv, but genuinely halves t and genuinely responds to both frames in a pair,
        # not just a reshape/drop that would ignore half the input.
        t_prime = t // self.tubelet_size
        x = x[:, : t_prime * self.tubelet_size]
        x = x.unflatten(dim=1, sizes=(t_prime, self.tubelet_size)).mean(dim=2)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.conv(x)
        return rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t_prime)


def _trainable_params(model: LatentWorldModel) -> list[torch.nn.Parameter]:
    return list(model.world_model.parameters()) + list(model.action_encoder.parameters()) + [model.bos]


# --- Fixed synthetic batch, reused by every check below ---
NUM_KEYS = 9
HEIGHT = WIDTH = 64
RAW_FRAMES = 8  # multiple of _FakeVjepaLike.TUBELET_SIZE, matching the real 40-frame/tubelet=2 case
BATCH_SIZE = 1

action_config = ActionConfig(valid_keys=[f"k{i}" for i in range(NUM_KEYS)])
actions = ActionTensors(config=action_config, batch_size=BATCH_SIZE)
actions.key_presses = torch.randint(0, 2, (BATCH_SIZE, RAW_FRAMES, NUM_KEYS), dtype=torch.int32)
actions.mouse_movements = torch.zeros((BATCH_SIZE, RAW_FRAMES, 2), dtype=torch.float32)
video = torch.randint(0, 256, (BATCH_SIZE, RAW_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.uint8)
batch = VideoActionBatch(video=video, actions=actions)

# temporal_stride=1, matching configs/scaled_300m_vjepa.yaml's real choice: the encoder's own
# tubelet halving is the ONLY temporal reduction, the bottleneck contributes none on top of it.
bottleneck_config = StridedConvBottleneckConfig(temporal_stride=1)
decoder_config = ViTDecoderConfig()
world_model_config = LatentWorldModelConfig(hidden_dim=32, depth=2, num_heads=2, mlp_dim_multiplier=2)


def _build_model(config: LatentWorldModelConfig) -> LatentWorldModel:
    return LatentWorldModel(
        config, bottleneck_config, decoder_config, num_keys=NUM_KEYS, codec_checkpoint=None,
        dino=_FakeVjepaLike(dino_dim=bottleneck_config.dino_dim),
    )


# --- Check 1: temporal_downsampling accounts for the encoder's own reduction, not just the
# bottleneck's -- the direct regression check for the latent_world_model.py fix. ---
model = _build_model(world_model_config)
expected = bottleneck_config.temporal_stride * _FakeVjepaLike.TUBELET_SIZE
assert model.temporal_downsampling == expected, (
    f"model.temporal_downsampling={model.temporal_downsampling}, expected {expected} "
    f"(bottleneck_config.temporal_stride={bottleneck_config.temporal_stride} * "
    f"tubelet_size={_FakeVjepaLike.TUBELET_SIZE}) -- the encoder's own temporal reduction isn't "
    f"being accounted for, exactly the notes/deviations.md 1.21 risk this script exists to catch."
)
print(
    f"[PASS] temporal_downsampling: {model.temporal_downsampling} "
    f"(= bottleneck_stride {bottleneck_config.temporal_stride} * tubelet_size {_FakeVjepaLike.TUBELET_SIZE}, "
    f"not just the bottleneck's own stride)"
)

# --- Check 2: a full forward/backward pass produces a finite loss with the halving stand-in wired
# in -- proves _encode()'s action-alignment slicing and ActionEncoder's own reshape both still work
# once the downsampling factor actually doubles, not just that the scalar above computes right. ---
model.train()
optimizer = torch.optim.Adam(_trainable_params(model), lr=2e-3)
out = model(batch)
assert torch.isfinite(out["loss_total"]), "loss_total is not finite with a time-halving encoder wired in"
out["loss_total"].backward()
for p in _trainable_params(model):
    assert p.grad is not None, "a trainable parameter has no gradient"
optimizer.step()
print(f"[PASS] forward/backward with time-halving encoder: loss_total={out['loss_total'].item():.4f}, gradients flow")

# --- Check 3: overfit -- same shape as verify_world_model_training.py's check 1, confirms the
# alignment fix doesn't just avoid crashing but produces a genuinely learnable training signal. ---
model2 = _build_model(world_model_config)
model2.train()
optimizer2 = torch.optim.Adam(_trainable_params(model2), lr=2e-3)
lr_scheduler2 = WarmupConstantCosineDecayLR(optimizer2, warmup_steps=10, constant_steps=0, decay_steps=290, min_lr=2e-4)

STEPS = 300
losses: list[float] = []
for step in range(STEPS):
    optimizer2.zero_grad()
    out2 = model2(batch)
    out2["loss_total"].backward()
    optimizer2.step()
    lr_scheduler2.step()
    losses.append(out2["loss_total"].item())

early = sum(losses[:20]) / 20
late = sum(losses[-20:]) / 20
drop = (early - late) / early
assert drop > 0.5, f"loss only dropped {drop:.1%} (early={early:.4f}, late={late:.4f}), expected >50%"
print(f"[PASS] overfit: loss {early:.4f} -> {late:.4f} ({drop:.1%} drop over {STEPS} steps)")

# --- Check 4: checkpoint save/load round-trip -- same pattern as verify_world_model_training.py's
# check 5, cheap to include, no reason a time-halving encoder would behave differently here. ---
torch.manual_seed(123)
with torch.no_grad():
    loss_before = model2(batch)["loss_total"].item()

tmp_path = Path(tempfile.mkdtemp()) / "wm_vjepa_checkpoint_test.pth"
save_checkpoint(
    tmp_path, STEPS - 1, model2.world_model, model2.action_encoder, model2.bos, optimizer2, lr_scheduler2,
    wandb_run_id="fake-run-id-for-test",
)

model2.world_model = DiffusionTransformer(world_model_config)
model2.action_encoder = ActionEncoder(
    num_keys=NUM_KEYS, hidden_dim=world_model_config.hidden_dim, temporal_downsampling=model2.temporal_downsampling
)
with torch.no_grad():
    model2.bos.copy_(torch.randn_like(model2.bos))

optimizer3 = torch.optim.Adam(_trainable_params(model2), lr=2e-3)
lr_scheduler3 = WarmupConstantCosineDecayLR(optimizer3, warmup_steps=10, constant_steps=0, decay_steps=290, min_lr=2e-4)
resume_step, resume_wandb_run_id, _resume_provenance = load_checkpoint(
    tmp_path, model2.world_model, model2.action_encoder, model2.bos, optimizer3, lr_scheduler3
)
assert resume_step == STEPS, f"expected resume step {STEPS}, got {resume_step}"
assert resume_wandb_run_id == "fake-run-id-for-test", f"wandb_run_id did not round-trip: {resume_wandb_run_id!r}"

torch.manual_seed(123)
with torch.no_grad():
    loss_after = model2(batch)["loss_total"].item()

assert abs(loss_before - loss_after) < 1e-6, f"checkpoint round-trip mismatch: {loss_before} vs {loss_after}"
print(
    f"[PASS] checkpoint round-trip: loss identical before/after save+load ({loss_before:.6f}), "
    f"wandb_run_id round-tripped correctly"
)

# --- Check 5: dino_temporal_scale correctly accounts for the encoder's own reduction when
# re-encoding the decoded video (Finding 3's fix, world_model/eval_metrics.py) ---
# Reuses `model` from checks 1-2 -- weights don't matter here, this is a shape/slicing check, not
# a numerical-correctness one. n_context_latents=1 chosen so the correct vs. old-formula slice
# lengths are unambiguously different (3 vs 2) at this file's own RAW_FRAMES=8 (z's t=4).
z5, _a5 = model._encode(batch)
_real_video5, _pred_video5, real_dino5, pred_dino5 = decode_and_dino(model, z5, z5)
n_context_latents5 = 1
expected_gen_len = z5.shape[1] - n_context_latents5  # 4 - 1 = 3

# real_dino5 comes from re-encoding the DECODED video through _FakeVjepaLike, which halves time
# again -- lands back at T=z5.shape[1] (latent-frame units), not T=z5.shape[1]*temporal_downsampling
# (video-frame units, what the OLD formula assumed). Confirmed directly, not just asserted below.
assert real_dino5.shape[1] == z5.shape[1], (
    f"sanity check on this test's own setup: expected the fake time-halving encoder to land "
    f"real_dino back at T={z5.shape[1]} (latent-frame units), got T={real_dino5.shape[1]}"
)

correct_scale = model.temporal_downsampling // getattr(model.dino, "tubelet_size", 1)
drift5 = compute_drift_metrics(
    z5, z5, n_context_latents5, real_dino5, pred_dino5, model.temporal_downsampling,
    dino_temporal_scale=correct_scale,
)
assert drift5["dino_cos_drift"].shape[1] == expected_gen_len, (
    f"correct dino_temporal_scale={correct_scale} produced generated-region length "
    f"{drift5['dino_cos_drift'].shape[1]}, expected {expected_gen_len}"
)

# Contrast: the OLD formula (dino_temporal_scale defaulting to temporal_downsampling itself, i.e.
# not passed at all) produces a DIFFERENT, wrong length here -- proving this test actually
# distinguishes correct from broken, not just that some slice happens to succeed.
drift5_old_formula = compute_drift_metrics(
    z5, z5, n_context_latents5, real_dino5, pred_dino5, model.temporal_downsampling,
)
old_len = drift5_old_formula["dino_cos_drift"].shape[1]
assert old_len != expected_gen_len, (
    f"expected the old (pre-fix) formula to produce a WRONG length here (this test's whole "
    f"point) -- got {old_len}, same as the correct {expected_gen_len}, meaning this test's own "
    f"setup no longer distinguishes the two"
)
print(
    f"[PASS] dino_temporal_scale: correct slice length {expected_gen_len} "
    f"(old formula would silently give {old_len})"
)

print("\nAll V-JEPA-track world-model temporal-alignment checks passed.")
