"""Computes the world model's latent_mean/latent_std (global scalars, matching real mira's
LatentWorldModelConfig.latent_mean_std: [mean, std]) in ONE direct pass over real data through an
already-trained frozen codec (DinoModel + MyBottleneck only -- no decoder, no backward pass).
Writes {"latent_mean": float, "latent_std": float, "n_elements": int} as JSON to --output.

Deliberately not an online EMA retrofitted into train_codec.py (see notes/deviations.md) -- a
direct pass over a couple hundred batches is simpler, needs no changes to the already-run codec
training, and works immediately on any already-trained checkpoint.

Two data modes, same convention as train_codec.py:
  - --index-path <dir>: streams real clips via mira's own create_loader.
  - Default: one fixed synthetic video reused every batch -- a mechanism check, not a real
    statistic (matches train_codec.py's own framing for its no-index-path mode).

--codec-checkpoint is optional (default None -> random-init bottleneck) so the mechanism/wiring
can be checked without a real checkpoint on hand; pass a real one for a real statistic.

Deliberately CPU-friendly, unlike train_codec.py (no backward pass, no GradScaler needed) --
auto-detects CUDA if available and falls back to CPU otherwise, so this is independently
verifiable on a machine with no GPU.
"""

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Real mira, cloned as a sibling repo -- reused for the dataloader.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch

from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.dino import DEFAULT_ENCODER_AGGREGATION_LAYERS, DinoModel
from mini_mira.codec.video_prep import resize_to_canonical
from mini_mira.ml.config_loading import load_pipeline_config

_CONVOLUTION_TYPES = (
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
    torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
)


def _keep_convolutions_in_bf16(module: torch.nn.Module) -> None:
    """Nest BF16 autocast around convolutions while the surrounding model uses FP16/FP32 elsewhere
    -- same helper as scripts/train_codec.py and mini_mira.world_model.latent_world_model
    (duplicated, not imported/shared, to avoid touching either while they may be in active use --
    see notes/gpu_amp_investigation.md for why this is needed on the real GPU). A no-op on CPU
    tensors either way, so this stays safe to apply unconditionally."""
    for convolution in (m for m in module.modules() if isinstance(m, _CONVOLUTION_TYPES)):
        if getattr(convolution, "_mini_mira_bf16_forward", False):
            continue
        original_forward = convolution.forward

        def bf16_forward(_self, *args, _forward=original_forward, **kwargs):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return _forward(*args, **kwargs)

        convolution.forward = types.MethodType(bf16_forward, convolution)
        convolution._mini_mira_bf16_forward = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--codec-checkpoint", default=None, help="Real checkpoint for a real statistic")
    parser.add_argument("--index-path", default=None, help="Real dataset dir from download_shards.py")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--require-pretrained-dino", action="store_true")
    parser.add_argument("--output", default="latent_stats.json")
    return parser.parse_args()


def build_next_video(args: argparse.Namespace):
    """Real streamed clips if --index-path is set, else one fixed synthetic video every call --
    same two-mode shape as train_codec.py's build_next_video."""
    if args.index_path:
        from mira.data.training_loader import create_loader  # noqa: PLC0415 -- only needed here

        loader = create_loader(
            index_path=args.index_path, clip_len=args.frames, target_fps=args.target_fps,
            n_players=1, batch_size=args.batch_size, frame_size=None,
        )
        data_iter = iter(loader)

        def next_video() -> torch.Tensor:
            batch, _metadata = next(data_iter)  # only latent stats needed, not actions
            video = batch.video.float() / 255.0
            return resize_to_canonical(video, args.height, args.width)
    else:
        fixed_video = torch.rand(1, args.frames, 3, args.height, args.width)

        def next_video() -> torch.Tensor:
            return fixed_video

    return next_video


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    encoder_dino_model = "dinov3_vitb16"
    dino = DinoModel(
        dino_model=encoder_dino_model, require_pretrained=args.require_pretrained_dino,
        last_layer_only=False, layer_indices=DEFAULT_ENCODER_AGGREGATION_LAYERS[encoder_dino_model],
    ).to(device)
    bottleneck = MyBottleneck(config.bottleneck).to(device)
    if args.codec_checkpoint is not None:
        ckpt: dict[str, Any] = torch.load(args.codec_checkpoint, map_location="cpu", weights_only=False)
        bottleneck.load_state_dict(ckpt["bottleneck"])
    # codec_checkpoint=None keeps bottleneck at random init -- a wiring/shape check, not a real
    # statistic, matching this script's own no-index-path mode.
    bottleneck.eval()
    _keep_convolutions_in_bf16(bottleneck)

    next_video = build_next_video(args)

    total_sum = torch.zeros((), dtype=torch.float64)
    total_sum_sq = torch.zeros((), dtype=torch.float64)
    total_n = 0

    with torch.no_grad():
        for i in range(args.num_batches):
            video = next_video().to(device)
            dino_features = dino.dino_forward(video)
            z = bottleneck(dino_features)
            z64 = z.double()
            total_sum += z64.sum().cpu()
            total_sum_sq += (z64 * z64).sum().cpu()
            total_n += z.numel()
            if (i + 1) % max(1, args.num_batches // 10) == 0:
                print(f"batch {i + 1}/{args.num_batches}")

    mean = (total_sum / total_n).item()
    variance = (total_sum_sq / total_n) - mean**2
    std = variance.clamp(min=0.0).sqrt().item()

    result = {"latent_mean": mean, "latent_std": std, "n_elements": total_n}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.output}: {result}")


if __name__ == "__main__":
    main()
