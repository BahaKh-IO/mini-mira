"""Real, quantitative held-out evaluation for a TRAINED V-JEPA-track codec checkpoint -- full
fork of evaluate_codec.py for the V-JEPA track (VjepaModel in place of DinoModel), same reasoning
as train_codec_vjepa.py: kept as a separate script on purpose, not a shared core loop, so each
track's logic reads on its own. Loads the real bottleneck/decoder weights, runs them on clips,
reports:
  - loss_mae, loss_lpips_perceptual, loss_dino_latent_consistency, loss_total -- CAUTION: only
    loss_mae is directly comparable to train_codec_vjepa.py's own charts. The other two are
    auto_weight OFF here (this script has no gradients to compute that rescaling from, running
    entirely under torch.no_grad()), while training's charts show them auto_weight-rescaled --
    often a 10-50x difference in magnitude, not a real quality gap. Fine to compare across two
    eval runs of this script; not fine to compare against a training chart.
  - psnr/ssim/lpips_standardized (AlexNet-based, matching full_eval_metrics.py's world-model
    convention) -- never computed during training at all, so no such caveat applies to these
plus a batch of side-by-side preview videos (reusing codec/logging_utils.py's log_preview).

Point --index-path at data the codec genuinely never trained on, e.g. the dataset's own "test"
split (scripts/download_shards.py --split test) rather than the "train" split it streams from --
build_holdout_split.py is also available for isolating newly-downloaded train shards if a
dedicated split isn't an option for some other dataset later.

No backward pass anywhere, so memory is far lower than training -- no --activation-checkpointing
flag needed, and --batch-size can safely be larger than what training used.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch

from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.decoder import ViTVideoDecoder
from mini_mira.codec.logging_utils import init_wandb, log_preview
from mini_mira.codec.loss import CodecLoss, CodecLossWeights, CodecOutputs, denormalize_for_dino, normalize_video
from mini_mira.codec.vjepa import DEFAULT_VJEPA_LAYERS, VjepaModel
from mini_mira.codec.video_prep import resize_to_canonical
from mini_mira.ml.config_loading import load_pipeline_config
from mini_mira.world_model.full_eval_metrics import compute_lpips, compute_psnr, compute_ssim
from mini_mira.world_model.latent_world_model import _keep_convolutions_in_bf16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/scaled_300m_vjepa.yaml")
    parser.add_argument("--codec-checkpoint", required=True, help="Trained V-JEPA-track checkpoint to evaluate (checkpoint_vjepa.pth)")
    parser.add_argument("--index-path", required=True, help="Real dataset dir from download_shards.py")
    parser.add_argument(
        "--seed", type=int, default=97,
        help="Passed to create_loader -- different from training's own default (2025) so this "
        "samples a different draw from the shard pool. See module docstring: not a real held-out "
        "split, since no disjoint one exists yet.",
    )
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--require-pretrained-vjepa", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4, help="No backward pass, so this can be larger than training's")
    parser.add_argument("--num-samples", type=int, default=50, help="Held-out clips to average metrics over")
    parser.add_argument("--num-preview-videos", type=int, default=8, help="How many of those to also save as preview videos")
    parser.add_argument("--precision", choices=["fp16-hybrid", "bf16"], default="bf16")
    parser.add_argument("--output-dir", default="eval_previews_vjepa")
    parser.add_argument("--wandb-project", default=None)
    args = parser.parse_args()
    if args.num_preview_videos > args.num_samples:
        parser.error("--num-preview-videos can't exceed --num-samples")
    return args


def _autocast(precision: str) -> torch.autocast:
    # Same convention as train_codec_vjepa.py's own _autocast -- bf16 needs no GradScaler-equivalent
    # since there's no backward pass here at all, only forward.
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    # Real, previously-hit-for-real requirement (notes/vjepa_next_session.md): the decoder's own
    # temporal/spatial upsample only exactly reconstructs --height/--width if both are divisible
    # by decoder.patch_size * bottleneck.stride -- otherwise it silently reconstructs a slightly
    # different size (e.g. 720 -> 704), surfacing later as a confusing loss.py shape-mismatch
    # crash instead of a clear error here. Purely additive: every already-proven-valid real
    # launch already satisfies this, so this never fires for a config that already works.
    required_divisor = config.decoder.patch_size * config.bottleneck.stride
    assert args.height % required_divisor == 0 and args.width % required_divisor == 0, (
        f"--height {args.height} / --width {args.width} must both be divisible by "
        f"{required_divisor} (decoder.patch_size {config.decoder.patch_size} * bottleneck.stride "
        f"{config.bottleneck.stride})"
    )

    # V-JEPA 2.1 has only one real variant (ViT-B), so no per-variant model-name/layer-dict lookup
    # like DinoModel's -- DEFAULT_VJEPA_LAYERS is already the right constant, no subscript needed.
    vjepa = VjepaModel(
        require_pretrained=args.require_pretrained_vjepa, last_layer_only=False, layer_indices=DEFAULT_VJEPA_LAYERS,
    ).cuda()
    bottleneck = MyBottleneck(config.bottleneck).cuda()
    decoder = ViTVideoDecoder(config.decoder).cuda()

    # Same lightweight load as compute_latent_stats.py -- no optimizer/scheduler/GradScaler
    # needed for eval, so codec/checkpoint.py's full load_checkpoint (which requires all of those
    # as arguments) would be more machinery than this script needs.
    ckpt: dict[str, Any] = torch.load(args.codec_checkpoint, map_location="cpu", weights_only=False)
    bottleneck.load_state_dict(ckpt["bottleneck"])
    decoder.load_state_dict(ckpt["decoder"])
    print(f"Loaded checkpoint from step {ckpt.get('step')}")

    vjepa.eval()
    bottleneck.eval()
    decoder.eval()

    # auto_weight left at its dataclass default (False): CodecLoss's adaptive-weight branch is
    # already guarded behind torch.is_grad_enabled() (see loss.py), so running the whole script
    # under torch.no_grad() below makes this a safe no-op either way -- explicit here for clarity
    # anyway, since there's no training loop reason to want it on.
    loss_fn = CodecLoss(CodecLossWeights()).cuda()
    loss_fn.bind_encoder_dino(vjepa)

    # Same conditional as train_codec_vjepa.py: only needed under fp16-hybrid (bf16's ambient
    # autocast already covers these convs; vjepa is never included here either, for the same
    # reason -- VjepaModel.dino_forward wraps its own body in bf16 autocast internally regardless).
    if args.precision == "fp16-hybrid":
        for module in (bottleneck, decoder, loss_fn):
            _keep_convolutions_in_bf16(module)

    # Second frozen conv-heavy net (LPIPS's AlexNet backbone) -- same bf16-conv-patch insurance
    # already applied to bottleneck/decoder/loss_fn elsewhere in this project, against the V100
    # cuDNN crash class documented in notes/gpu_amp_investigation.md.
    import lpips as lpips_lib  # noqa: PLC0415 -- optional dep, only needed here

    lpips_fn = lpips_lib.LPIPS(net="alex")
    lpips_fn.requires_grad_(False)
    _keep_convolutions_in_bf16(lpips_fn)  # same order as full_eval_metrics.py: patch, then .to(device)
    lpips_fn = lpips_fn.cuda()

    from mira.data.training_loader import create_loader  # noqa: PLC0415 -- only needed here

    loader = create_loader(
        index_path=args.index_path, clip_len=args.frames, target_fps=args.target_fps,
        n_players=1, batch_size=args.batch_size, frame_size=None, seed=args.seed,
    )
    data_iter = iter(loader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_enabled = init_wandb(args.wandb_project, vars(args))

    n_batches = max(1, args.num_samples // args.batch_size)
    totals: dict[str, float] = {}
    n_seen = 0
    n_previews_saved = 0

    with torch.no_grad():
        for batch_idx in range(n_batches):
            batch, _metadata = next(data_iter)
            video = resize_to_canonical(batch.video.float() / 255.0, args.height, args.width).cuda()

            with _autocast(args.precision):
                dino_features = vjepa.dino_forward(video)
                z = bottleneck(dino_features)
                reconstructed = decoder(z)
                outputs = CodecOutputs(
                    input_video=normalize_video(video), output_video=reconstructed, dino_features=dino_features
                )
                losses = loss_fn(outputs)

            # .float() before scoring, matching CodecLoss.forward's own convention (predicted =
            # outputs.output_video.float()) -- `reconstructed` was produced inside the autocast
            # block above and likely still carries bf16/fp16 dtype even after it exits, while
            # `video` was never autocast at all (stays fp32); comparing the two directly would
            # silently mix precisions in the metric computation instead of scoring at full fp32.
            video_01 = video.float().clamp(0, 1)
            recon_01 = denormalize_for_dino(reconstructed.float()).clamp(0, 1)
            psnr = compute_psnr(recon_01, video_01).mean().item()
            ssim = compute_ssim(recon_01, video_01).mean().item()
            lpips_score = compute_lpips(lpips_fn, recon_01, video_01).mean().item()

            batch_metrics = {
                **{k: v.item() for k, v in losses.items() if not k.endswith("_auto_w")},
                "psnr": psnr, "ssim": ssim, "lpips_standardized": lpips_score,
            }
            bs = video.shape[0]
            for k, v in batch_metrics.items():
                totals[k] = totals.get(k, 0.0) + v * bs
            n_seen += bs

            if n_previews_saved < args.num_preview_videos:
                # log_preview's own "first sample" convention (see codec/logging_utils.py) means
                # one call = one preview -- loop over individual samples in this batch, not the
                # batch as a whole, to get --num-preview-videos separate files rather than one.
                for i in range(min(bs, args.num_preview_videos - n_previews_saved)):
                    log_preview(
                        enabled=wandb_enabled, step=n_previews_saved,
                        original=video[i : i + 1], reconstructed=reconstructed[i : i + 1],
                        fps=args.target_fps, output_dir=output_dir,
                    )
                    n_previews_saved += 1

            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"batch {batch_idx + 1}/{n_batches} ({n_seen} clips seen), cuda_peak_allocated={peak:.2f}GiB")

    results = {k: v / n_seen for k, v in totals.items()}
    results["n_clips"] = n_seen
    print("\nHeld-out evaluation results:")
    print("  (only loss_mae is comparable to train_codec_vjepa.py's charts -- see module docstring)")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}" if k != "n_clips" else f"  {k}: {v}")

    results_path = output_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {results_path}")
    print(f"Saved {n_previews_saved} preview videos to {output_dir}/")

    if wandb_enabled:
        import wandb  # noqa: PLC0415

        wandb.log(results)
        wandb.finish()


if __name__ == "__main__":
    main()
