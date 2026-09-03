"""Standalone rollout sampling + eval for an already-trained V-JEPA world-model checkpoint -- no
training loop, no optimizer, just: load the real checkpoint, roll out on real held-out clips with
a small context (a genuinely LONG generated region), save the rollout videos, compute the same
Frechet DINO/Inception Distance + PSNR/LPIPS/SSIM + drift metrics
train_world_model_vjepa.py's own periodic full-eval computes, print + save the numbers to JSON.

Built for a genuinely comparable DINO-vs-V-JEPA "same long rollout" benchmark -- see
scripts/sample_rollout.py for the DINO-track sibling (full duplicate, same reasoning as every
other DINO/V-JEPA script pair in this project). Pass the SAME --seed on both tracks' invocations
to pull the same real held-out clips (before each track's own resize) for a genuinely apples-to-
apples comparison, not just "both evaluated on some held-out data."
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from mira.data.batch import VideoActionBatch
from mira.data.training_loader import create_loader

from mini_mira.codec.video_prep import resize_to_canonical
from mini_mira.codec.vjepa import DEFAULT_VJEPA_LAYERS, VjepaModel
from mini_mira.ml.config_loading import load_pipeline_config
from mini_mira.world_model.eval_metrics import RunningMean, compute_drift_metrics, decode_and_dino
from mini_mira.world_model.full_eval_metrics import FullEvalMetrics, compute_full_eval_metrics
from mini_mira.world_model.latent_world_model import LatentWorldModel
from mini_mira.world_model.rollout_visualization import log_rollout_videos, render_rollout_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/scaled_300m_vjepa.yaml")
    parser.add_argument("--codec-checkpoint", required=True)
    parser.add_argument("--wm-checkpoint", required=True, help="Trained world-model checkpoint (checkpoint_wm_vjepa.pth)")
    parser.add_argument("--latent-stats", required=True)
    parser.add_argument("--index-path", required=True, help="Real held-out dataset dir (the test split)")
    parser.add_argument("--require-pretrained-vjepa", action="store_true")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument(
        "--context-latents", type=int, default=2,
        help="Small context -> long generated rollout. Default 2 matches this project's own real "
        "world-model eval config (fdd_slice_frames divides evenly at this value).",
    )
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--schedule-type", choices=["linear", "linear_quadratic"], default="linear")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=20, help="Real held-out clips to average metrics over")
    parser.add_argument("--fdd-slice-frames", type=int, default=6)
    parser.add_argument("--viz-n-samples", type=int, default=10, help="How many rollouts to also save as videos")
    parser.add_argument("--precision", choices=["fp16-hybrid", "bf16"], default="bf16")
    parser.add_argument(
        "--seed", type=int, default=37,
        help="Matches the training scripts' own test-loader seed. Pass the SAME value on both "
        "tracks' invocations to pull the same real clips before each one's own resize.",
    )
    parser.add_argument("--output-dir", default="sample_rollout_vjepa_out")
    return parser.parse_args()


def _autocast(precision: str) -> torch.autocast:
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _resize_batch(batch: VideoActionBatch, height: int, width: int) -> VideoActionBatch:
    return VideoActionBatch(video=resize_to_canonical(batch.video, height, width), actions=batch.actions)


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)

    required_divisor = config.decoder.patch_size * config.bottleneck.stride
    assert args.height % required_divisor == 0 and args.width % required_divisor == 0, (
        f"--height {args.height} / --width {args.width} must both be divisible by {required_divisor}"
    )

    latent_stats = json.loads(Path(args.latent_stats).read_text())
    vjepa = VjepaModel(
        require_pretrained=args.require_pretrained_vjepa,
        last_layer_only=False, layer_indices=DEFAULT_VJEPA_LAYERS,
    ).cuda()
    model = LatentWorldModel(
        config.world_model, config.bottleneck, config.decoder, num_keys=config.num_keys,
        codec_checkpoint=args.codec_checkpoint, latent_mean=latent_stats["latent_mean"],
        latent_std=latent_stats["latent_std"], dino=vjepa,
    ).cuda()

    # Plain state-dict load, not checkpoint.load_checkpoint -- that also restores optimizer/
    # lr_scheduler/RNG state, all irrelevant for pure inference. Checkpoints are always saved
    # already-unwrapped from any --compile OptimizedModule prefix (checkpoint.py's own
    # _unwrap_compiled), so a plain load_state_dict works regardless of whether training used
    # --compile.
    ckpt = torch.load(args.wm_checkpoint, map_location="cpu", weights_only=False)
    model.world_model.load_state_dict(ckpt["world_model"])
    model.action_encoder.load_state_dict(ckpt["action_encoder"])
    with torch.no_grad():
        model.bos.copy_(ckpt["bos"].to(model.bos.device))
    print(f"Loaded world-model checkpoint from step {ckpt['step']}")
    model.eval()

    n_latent_frames = args.frames // model.temporal_downsampling
    assert 0 < args.context_latents < n_latent_frames, (
        f"--context-latents ({args.context_latents}) must leave at least one latent frame to "
        f"generate (of {n_latent_frames} total at --frames {args.frames})"
    )
    n_generated_video_frames = (n_latent_frames - args.context_latents) * model.temporal_downsampling
    assert n_generated_video_frames % args.fdd_slice_frames == 0, (
        f"generated region ({n_generated_video_frames} video frames) must be a multiple of "
        f"--fdd-slice-frames ({args.fdd_slice_frames})"
    )
    print(
        f"Rollout: {args.context_latents} context latent frames -> {n_generated_video_frames} "
        f"generated video frames (of {args.frames} total)"
    )

    loader = create_loader(
        index_path=args.index_path, clip_len=args.frames, target_fps=args.target_fps,
        n_players=1, batch_size=args.batch_size, frame_size=None, seed=args.seed,
    )
    eval_iter = iter(loader)

    full_eval_metrics = FullEvalMetrics(
        dino_dim=vjepa.dino_dim, fdd_slice_frames=args.fdd_slice_frames,
        num_slices=n_generated_video_frames // args.fdd_slice_frames, device="cuda",
    )
    drift_trackers = {"dino_cos_drift": RunningMean(), "dino_l2_drift": RunningMean(), "latent_drift": RunningMean()}
    viz_samples: list[torch.Tensor] = []
    viz_key_presses: list[torch.Tensor] = []

    n_batches = max(1, args.num_samples // args.batch_size)
    with torch.no_grad():
        for i in range(n_batches):
            batch, _metadata = next(eval_iter)
            batch = _resize_batch(batch, args.height, args.width).to("cuda", non_blocking=True)
            with _autocast(args.precision):
                z, z_t = model.rollout(batch, args.context_latents, args.diffusion_steps, args.schedule_type)
                real_video, pred_video, real_dino, pred_dino = decode_and_dino(model, z, z_t)
                dino_temporal_scale = model.temporal_downsampling // getattr(model.dino, "tubelet_size", 1)
                drift = compute_drift_metrics(
                    z, z_t, args.context_latents, real_dino, pred_dino, model.temporal_downsampling,
                    dino_temporal_scale=dino_temporal_scale,
                )
                compute_full_eval_metrics(
                    real_video, pred_video, real_dino, pred_dino,
                    args.context_latents, model.temporal_downsampling, full_eval_metrics,
                    dino_temporal_scale=dino_temporal_scale,
                )
                if len(viz_samples) < args.viz_n_samples:
                    for j in range(min(args.viz_n_samples - len(viz_samples), pred_video.shape[0])):
                        viz_samples.append(pred_video[j])
                        viz_key_presses.append(batch.actions.key_presses[j])
            for name, tracker in drift_trackers.items():
                tracker.update(drift[name])
            print(f"batch {i + 1}/{n_batches} done")

    metrics = {f"drift_{name}": tracker.compute() for name, tracker in drift_trackers.items()}
    full_scalars, full_curves = full_eval_metrics.compute_and_reset()
    metrics.update(full_scalars)
    for curve_name, values in full_curves.items():
        for i, value in enumerate(values):
            metrics[f"{curve_name}_at_{(i + 1) * args.fdd_slice_frames}"] = value

    print("\nSampling results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"wm_step": ckpt["step"], "context_latents": args.context_latents,
               "generated_video_frames": n_generated_video_frames, **metrics}
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))

    n_context_frames = args.context_latents * model.temporal_downsampling
    rendered = [
        render_rollout_sample(pred_video, key_presses, n_context_frames)
        for pred_video, key_presses in zip(viz_samples, viz_key_presses)
    ]
    log_rollout_videos(rendered, args.target_fps, ckpt["step"], wandb_enabled=False, output_dir=output_dir)
    print(f"\nSaved {len(rendered)} rollout videos + results.json to {output_dir}/")


if __name__ == "__main__":
    main()
