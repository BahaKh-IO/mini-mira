"""CPU-friendly proof that generate_next_frame (scripts/serve_interactive_vjepa.py) produces
numerically IDENTICAL latents to LatentWorldModel.rollout() -- the one-shot-length-known-upfront
mechanism generate_next_frame is a single-frame-at-a-time restructuring of. Same RNG seed reset
right before each path, same fixed inputs, model in .eval() (no dropout stochasticity) -- if both
paths consume randomness in the same order (one torch.randn(b,c,h,w) per new frame, confirmed by
reading rollout()'s own loop body), the results must match exactly if the math is equivalent.
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
from mini_mira.world_model.latent_world_model import LatentWorldModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_interactive_vjepa import generate_next_frame  # noqa: E402 -- see sys.path.insert above

torch.manual_seed(0)


class _FakeVjepaLike(torch.nn.Module):
    """Same minimal stand-in as verify_fd_loss.py -- a real (if tiny) Conv2d, halves time like
    the real VjepaModel's own tubelet reduction."""

    TUBELET_SIZE = 2

    def __init__(self, dino_dim: int, patch_size: int = 16):
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


DINO_DIM = 16
NUM_KEYS = 9
HEIGHT = WIDTH = 64
RAW_FRAMES = 16  # -> 8 latent frames at temporal_downsampling=2
BATCH_SIZE = 1
N_CONTEXT_LATENTS = 3
N_DIFFUSION_STEPS = 3
SEED = 42

action_config = ActionConfig(valid_keys=[f"k{i}" for i in range(NUM_KEYS)])
actions = ActionTensors(config=action_config, batch_size=BATCH_SIZE)
actions.key_presses = torch.randint(0, 2, (BATCH_SIZE, RAW_FRAMES, NUM_KEYS), dtype=torch.int32)
actions.mouse_movements = torch.zeros((BATCH_SIZE, RAW_FRAMES, 2), dtype=torch.float32)
video = torch.randint(0, 256, (BATCH_SIZE, RAW_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.uint8)
batch = VideoActionBatch(video=video, actions=actions)

bottleneck_config = StridedConvBottleneckConfig(temporal_stride=1, dino_dim=DINO_DIM)
decoder_config = ViTDecoderConfig(latent_dim=bottleneck_config.latent_dim)
world_model_config = LatentWorldModelConfig(hidden_dim=32, depth=2, num_heads=2, mlp_dim_multiplier=2)

model = LatentWorldModel(
    world_model_config, bottleneck_config, decoder_config, num_keys=NUM_KEYS, codec_checkpoint=None,
    dino=_FakeVjepaLike(dino_dim=DINO_DIM),
)
model.eval()

# --- Path 1: rollout() in one shot ---
torch.manual_seed(SEED)
_z, z_t_rollout = model.rollout(batch, N_CONTEXT_LATENTS, N_DIFFUSION_STEPS, "linear")

# --- Path 2: generate_next_frame, one call per new frame ---
z_context, _a = model._encode(batch)  # noqa: SLF001 -- established pattern, see serve_interactive_vjepa.py
z_context = z_context[:, :N_CONTEXT_LATENTS]
t_total = _z.shape[1]

torch.manual_seed(SEED)
z_incremental = z_context.clone()
for _k in range(N_CONTEXT_LATENTS, t_total):
    z_incremental = generate_next_frame(
        model, z_incremental, batch.actions.key_presses, n_diffusion_steps=N_DIFFUSION_STEPS, schedule_type="linear",
    )

assert z_incremental.shape == z_t_rollout.shape, f"{z_incremental.shape} != {z_t_rollout.shape}"
assert torch.allclose(z_incremental, z_t_rollout, atol=1e-5), (
    f"generate_next_frame diverges from rollout(): max abs diff = "
    f"{(z_incremental - z_t_rollout).abs().max().item()}"
)
# Context region must also be untouched/identical (both paths start from the same real encode).
assert torch.allclose(z_incremental[:, :N_CONTEXT_LATENTS], _z[:, :N_CONTEXT_LATENTS], atol=1e-5)
print(
    f"[PASS] generate_next_frame matches rollout() exactly across {t_total - N_CONTEXT_LATENTS} "
    f"incrementally-generated frames (max abs diff vs rollout: "
    f"{(z_incremental - z_t_rollout).abs().max().item():.2e})"
)

print("\nAll interactive-rollout checks passed.")
