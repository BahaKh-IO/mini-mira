"""Computes real, held-out V-JEPA feature-space Fréchet statistics (mean + full covariance) for
FD-loss (arXiv:2604.28190v1) -- see mini_mira.world_model.fd_loss. Writes {"mean": Tensor(768),
"cov": Tensor(768,768)} via torch.save to --output (not JSON -- a 768x768 covariance is the wrong
shape for JSON text).

Structural template: compute_latent_stats_vjepa.py (loader/device/batching pattern reused as-is).
Key difference: extracts features BEFORE the bottleneck -- vjepa.dino_forward(video) output, not
the compressed latent -- since FD-loss compares distributions in V-JEPA's own 768-dim feature
space (matching full_eval_metrics.py's existing frechet_dino_distance convention), not latent
space. No bottleneck/codec-checkpoint needed at all as a result.

Accumulation: full_eval_metrics.OnlineGaussian, reused directly (already exactly the "accumulate
sufficient stats over many batches, .compute() once at the end" pattern this needs -- no
differentiability required here, this is a pure offline precompute).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch

from mini_mira.codec.vjepa import DEFAULT_VJEPA_LAYERS, VjepaModel
from mini_mira.codec.video_prep import resize_to_canonical
from mini_mira.world_model.full_eval_metrics import OnlineGaussian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-path", default=None, help="Real held-out dataset dir (the test split)")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--require-pretrained-vjepa", action="store_true")
    parser.add_argument("--output", default="real_frechet_stats_vjepa.pt")
    return parser.parse_args()


def build_next_video(args: argparse.Namespace):
    """Same two-mode shape as compute_latent_stats_vjepa.py's build_next_video."""
    if args.index_path:
        from mira.data.training_loader import create_loader  # noqa: PLC0415 -- only needed here

        loader = create_loader(
            index_path=args.index_path, clip_len=args.frames, target_fps=args.target_fps,
            n_players=1, batch_size=args.batch_size, frame_size=None,
        )
        data_iter = iter(loader)

        def next_video() -> torch.Tensor:
            batch, _metadata = next(data_iter)  # only real stats needed, not actions
            video = batch.video.float() / 255.0
            return resize_to_canonical(video, args.height, args.width)
    else:
        fixed_video = torch.rand(1, args.frames, 3, args.height, args.width)

        def next_video() -> torch.Tensor:
            return fixed_video

    return next_video


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    vjepa = VjepaModel(
        require_pretrained=args.require_pretrained_vjepa,
        last_layer_only=False, layer_indices=DEFAULT_VJEPA_LAYERS,
    ).to(device)

    next_video = build_next_video(args)
    gaussian = OnlineGaussian(dim=vjepa.dino_dim).to(device)

    with torch.no_grad():
        for i in range(args.num_batches):
            video = next_video().to(device)
            dino_features = vjepa.dino_forward(video)
            if isinstance(dino_features, list):
                dino_features = dino_features[-1]
            # Same per-frame, spatially-pooled convention full_eval_metrics.py's Frechet
            # accumulation already uses -- one sample per (batch, frame), not per clip.
            pooled = dino_features.mean(dim=(-1, -2)).flatten(0, 1)
            gaussian.update(pooled)
            if (i + 1) % max(1, args.num_batches // 10) == 0:
                print(f"batch {i + 1}/{args.num_batches}")

    mean, cov = gaussian.compute()
    torch.save({"mean": mean.cpu(), "cov": cov.cpu()}, args.output)
    print(f"Wrote {args.output}: mean shape={tuple(mean.shape)}, cov shape={tuple(cov.shape)}, n={int(gaussian.n)}")


if __name__ == "__main__":
    main()
