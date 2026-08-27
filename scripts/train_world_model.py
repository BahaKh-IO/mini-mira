"""Config-driven world-model training: a frozen codec checkpoint (DINO + bottleneck + decoder)
plus a trainable DiffusionTransformer + ActionEncoder, trained on real mira's diagonal
flow-matching loss (+ optional PSD self-distillation) -- see mini_mira.world_model.
latent_world_model.LatentWorldModel. Matches scripts/train_codec.py's own shape: plain argparse,
no Hydra, real data via mira's create_loader (now actually consuming batch.actions, unlike the
codec).

Which codec checkpoint to use is left to you -- point --codec-checkpoint at whichever one you've
settled on, and --latent-stats at that checkpoint's scripts/compute_latent_stats.py output.

Assumes a CUDA GPU (no CPU fallback) -- this script is for real training runs; see
scripts/verify_world_model_training.py for the CPU-friendly mechanism check.

Precision: --precision (default bf16) picks the autocast dtype for the trainable world
model/action encoder, same flag/choices as train_codec.py. bf16 needs no GradScaler (disabled as a
documented no-op) and is what real training on this project's current GPU (native bf16 tensor
cores) actually uses -- fp16-hybrid (+ GradScaler) is kept only for parity with train_codec.py's
own V100-era fallback, not the expected choice here. The frozen bottleneck/decoder are always
bf16-conv-patched inside LatentWorldModel's own __init__ regardless of --precision (see
notes/gpu_amp_investigation.md for why) -- harmless overlap under --precision bf16, load-bearing
under fp16-hybrid.

Eval: periodic validation loss (cheap, same forward as training, no backward) + a merged full
eval -- lightweight DINO/latent "drift" metrics, the full Frechet DINO/Inception Distance +
PSNR/LPIPS/SSIM suite (mini_mira.world_model.full_eval_metrics), and a handful of rendered rollout
videos for visual inspection (mini_mira.world_model.rollout_visualization) -- all sharing ONE
model.rollout(...) call per eval batch rather than rolling out separately for each. This suite was
originally scoped down to drift metrics only for compute/dependency-cost reasons
(notes/deviations.md entry 1.19); that decision has since been reversed given eval only fires a
handful of times across a run.
"""

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Real mira, cloned as a sibling repo -- reused for the dataloader, LR schedule, and batch type.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from mira.data.batch import VideoActionBatch
from mira.data.training_loader import create_loader
from mira.training.lr_schedule import WarmupConstantCosineDecayLR

from mini_mira.codec.logging_utils import get_wandb_run_id, init_wandb, log_step
from mini_mira.codec.video_prep import resize_to_canonical
from mini_mira.ml.config_loading import apply_run_config, load_pipeline_config, load_run_config
from mini_mira.ml.run_config import WorldModelRunConfig
from mini_mira.world_model.checkpoint import load_checkpoint, save_checkpoint
from mini_mira.world_model.eval_metrics import RunningMean, compute_drift_metrics, decode_and_dino
from mini_mira.world_model.full_eval_metrics import FullEvalMetrics, compute_full_eval_metrics
from mini_mira.world_model.latent_world_model import LatentWorldModel
from mini_mira.world_model.rollout_visualization import log_rollout_videos, render_rollout_sample


def _autocast(precision: str) -> torch.autocast:
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


# Set by _request_shutdown, checked once per completed optimizer step. `timeout` (used to
# time-box long unattended runs, see README) sends SIGTERM with no warning -- without this, the
# process just dies wherever it happens to be, losing up to --checkpoint-every steps of progress
# and leaving wandb with no clean sign-off (shows "Crashed" even when nothing actually broke).
# A signal handler must stay minimal (no CUDA/file I/O directly inside it), so this only sets a
# flag; the actual save happens in the main loop once it's safe to do so.
_shutdown_requested = False


def _request_shutdown(signum, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument(
        "--run-config", default=None,
        help="Optional YAML of hyperparameters (WorldModelRunConfig, see configs/runs/). Any "
        "flag below, explicitly passed, always overrides it. Omit for today's hardcoded defaults.",
    )
    parser.add_argument("--index-path", required=True, help="Real training dataset dir from download_shards.py")
    parser.add_argument("--test-index-path", required=True, help="Held-out dir for validation + drift eval")
    parser.add_argument("--codec-checkpoint", required=True, help="Frozen codec checkpoint (bottleneck+decoder)")
    parser.add_argument("--latent-stats", required=True, help="scripts/compute_latent_stats.py JSON output")
    parser.add_argument("--require-pretrained-dino", action="store_true")
    parser.add_argument(
        "--precision", choices=["fp16-hybrid", "bf16"], default=None,
        help="Default bf16: plain bfloat16 autocast, GradScaler disabled (a documented no-op) "
        "-- the actual setup real training uses on this project's current GPU. fp16-hybrid: "
        "float16 autocast + GradScaler, kept only for parity with train_codec.py's own V100-era "
        "fallback -- see this script's module docstring before picking it.",
    )

    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None, help="Raw clip length in frames. Default 40")
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-forward micro-batch size. Default 4")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="Micro-batches accumulated per optimizer step (effective batch = batch-size * this). Default 2",
    )

    parser.add_argument("--steps", type=int, default=None, help="Default 2000")
    parser.add_argument("--lr", type=float, default=None)  # matches mira's world-model optimizer, default 1e-4
    parser.add_argument("--lr-warmup-steps", type=int, default=None, help="Default: steps // 20")
    parser.add_argument(
        "--lr-decay-steps", type=int, default=None,
        help="Default 0 -- matches mira's own shipped single-player config (warmup then constant, "
        "no cosine decay). NOT proportional to --steps like train_codec.py's default -- see this "
        "script's module docstring / notes/deviations.md before changing this.",
    )
    parser.add_argument(
        "--lr-min", type=float, default=None, help="Default 1e-6. Matches mira; inert while --lr-decay-steps=0"
    )

    parser.add_argument(
        "--psd-weight", type=float, default=None, help="Deterministic PSD. Default 0.0 (mira default)"
    )
    parser.add_argument(
        "--psd-loss-prob", type=float, default=None, help="Stochastic PSD. Default 0.0 (mira default)"
    )
    parser.add_argument(
        "--scheduled-sampling-prob", type=float, default=None,
        help="Probability of training on a self-generated (not real) clean_past -- not a mira "
        "flag, added here to address the rollout-depth quality drift found in real training. "
        "Default 0.0 (off). See LatentWorldModel._fake_shifted_z.",
    )

    parser.add_argument("--eval-batch-size", type=int, default=None, help="Default: --batch-size")
    parser.add_argument("--val-every", type=int, default=None, help="Default: steps // 50")
    parser.add_argument("--val-n-samples", type=int, default=None, help="Default 64")
    parser.add_argument("--drift-eval-every", type=int, default=None, help="Default: steps // 10")
    parser.add_argument("--drift-eval-n-samples", type=int, default=None, help="Default 8")
    parser.add_argument("--drift-eval-context-latents", type=int, default=None, help="Default 6")
    parser.add_argument("--drift-eval-diffusion-steps", type=int, default=None, help="Default 4")
    parser.add_argument(
        "--drift-eval-schedule", default=None, choices=["linear", "linear_quadratic"], help="Default linear"
    )
    parser.add_argument(
        "--fdd-slice-frames", type=int, default=None,
        help="Frechet-distance slice window, in GENERATED video frames -- must evenly divide "
        "(--frames - drift_eval_context_latents*temporal_stride). Default 7 gives 4 slices at "
        "this script's own defaults (28 generated video frames). Real mira's own default is 20 "
        "(6 slices over 120 unrolled frames) -- scaled down proportionally.",
    )
    parser.add_argument(
        "--viz-n-samples", type=int, default=None,
        help="Rollout videos rendered for visual inspection per full eval run -- fixed and small, "
        "logged as separate wandb.Video entries, not a gridded batch. Default 2",
    )

    parser.add_argument("--checkpoint-dir", default="checkpoints_wm")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Default: steps // 10")
    parser.add_argument("--console-log-every", type=int, default=None, help="Default: steps // 100")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hf-backup-repo", default=None, help="Optional HF Hub repo to back up checkpoints to")
    parser.add_argument("--wandb-project", default=None)

    return parser.parse_args()


def resize_batch(batch: VideoActionBatch, height: int, width: int) -> VideoActionBatch:
    """resize_to_canonical preserves dtype/range, so this works directly on the loader's raw
    uint8 video -- same pad+resize convention as train_codec.py's build_next_video."""
    return VideoActionBatch(video=resize_to_canonical(batch.video, height, width), actions=batch.actions)


def run_validation(
    model: LatentWorldModel, loader, step: int, wandb_enabled: bool, n_samples: int, batch_size: int,
    height: int, width: int, lr: float, precision: str,
) -> None:
    """Averages the same forward loss training uses (no backward) over a fixed-seed held-out
    subsample. Fresh iterator each call so the same subsample is scored every time, matching real
    mira's own convention."""
    model.eval()
    n_batches = max(1, n_samples // batch_size)
    val_iter = iter(loader)
    totals: dict[str, float] = {}
    with torch.no_grad():
        for _ in range(n_batches):
            batch, _metadata = next(val_iter)
            batch = resize_batch(batch, height, width).to("cuda")
            with _autocast(precision):
                losses = model(batch)
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item() / n_batches
    metrics = {f"val_{k}": v for k, v in totals.items()}
    print(f"Validation at step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    log_step(wandb_enabled, step, metrics, lr)
    model.train()


def run_full_eval(
    model: LatentWorldModel, loader, step: int, wandb_enabled: bool, n_samples: int, batch_size: int,
    context_latents: int, diffusion_steps: int, schedule_type: str, height: int, width: int, lr: float,
    full_eval_metrics: FullEvalMetrics, viz_n_samples: int, target_fps: int, checkpoint_dir: Path,
    precision: str,
) -> None:
    """Merged eval: lightweight drift metrics + the full Frechet/PSNR/LPIPS/SSIM suite + a handful
    of rendered rollout videos -- all sharing ONE model.rollout(...) call per eval batch rather
    than rolling out separately for each (the autoregressive rollout is the expensive part)."""
    model.eval()
    n_batches = max(1, n_samples // batch_size)
    eval_iter = iter(loader)
    drift_trackers = {"dino_cos_drift": RunningMean(), "dino_l2_drift": RunningMean(), "latent_drift": RunningMean()}
    viz_samples: list[torch.Tensor] = []
    viz_key_presses: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_batches):
            batch, _metadata = next(eval_iter)
            batch = resize_batch(batch, height, width).to("cuda")
            with _autocast(precision):
                z, z_t = model.rollout(batch, context_latents, diffusion_steps, schedule_type)
                # One shared decode + DINO pass, reused by all three eval tiers below instead of
                # each redoing it independently (used to be 2-3x redundant).
                real_video, pred_video, real_dino, pred_dino = decode_and_dino(model, z, z_t)
                drift = compute_drift_metrics(
                    z, z_t, context_latents, real_dino, pred_dino, model.temporal_downsampling
                )
                compute_full_eval_metrics(
                    real_video, pred_video, real_dino, pred_dino,
                    context_latents, model.temporal_downsampling, full_eval_metrics,
                )

                # Render a few samples for visual inspection, drawn from whichever batches come
                # first -- reuses pred_video from the shared decode above.
                if len(viz_samples) < viz_n_samples:
                    for i in range(min(viz_n_samples - len(viz_samples), pred_video.shape[0])):
                        viz_samples.append(pred_video[i])
                        viz_key_presses.append(batch.actions.key_presses[i])

            for name, tracker in drift_trackers.items():
                tracker.update(drift[name])

    metrics = {f"drift_{name}": tracker.compute() for name, tracker in drift_trackers.items()}
    full_scalars, full_curves = full_eval_metrics.compute_and_reset()
    metrics.update(full_scalars)
    for curve_name, values in full_curves.items():
        for i, value in enumerate(values):
            metrics[f"{curve_name}_at_{(i + 1) * full_eval_metrics.fdd_slice_frames}"] = value

    print(f"Full eval at step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    log_step(wandb_enabled, step, metrics, lr)

    # key_presses is already 1:1 aligned with pred_video's raw frame count -- action_fps always
    # equals video fps in this project (see LatentWorldModel._encode), and the raw (unsliced)
    # key_presses tensor covers the whole clip, context and generated region alike.
    n_context_frames = context_latents * model.temporal_downsampling
    rendered = [
        render_rollout_sample(pred_video, key_presses, n_context_frames)
        for pred_video, key_presses in zip(viz_samples, viz_key_presses)
    ]
    log_rollout_videos(rendered, target_fps, step, wandb_enabled, checkpoint_dir)

    model.train()


def main() -> None:
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    args = parse_args()
    run_config = load_run_config(args.run_config, WorldModelRunConfig) if args.run_config else WorldModelRunConfig()
    apply_run_config(args, run_config)
    if args.psd_weight > 0 and args.psd_loss_prob > 0:
        raise SystemExit(
            "Set at most one of --psd-weight and --psd-loss-prob (matches mira's own "
            "LatentWorldModelConfig constraint: both being positive at once is undefined)"
        )

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
    # CLI/--run-config (resolved above) is authoritative over the YAML preset's own
    # psd_weight/psd_loss_prob (both default 0.0 there too, matching mira's own shipped
    # single-player config).
    config.world_model.psd_weight = args.psd_weight
    config.world_model.psd_loss_prob = args.psd_loss_prob
    config.world_model.scheduled_sampling_prob = args.scheduled_sampling_prob

    temporal_stride = config.bottleneck.temporal_stride
    assert args.frames % temporal_stride == 0, (
        f"--frames ({args.frames}) must be a multiple of bottleneck.temporal_stride ({temporal_stride})"
    )
    n_latent_frames = args.frames // temporal_stride
    assert 0 < args.drift_eval_context_latents < n_latent_frames, (
        f"--drift-eval-context-latents ({args.drift_eval_context_latents}) must leave at least one "
        f"latent frame to generate (of {n_latent_frames} total at --frames {args.frames})"
    )
    n_context_frames = args.drift_eval_context_latents * temporal_stride
    generated_video_frames = args.frames - n_context_frames
    assert generated_video_frames % args.fdd_slice_frames == 0, (
        f"generated region ({generated_video_frames} video frames) must be a multiple of "
        f"--fdd-slice-frames ({args.fdd_slice_frames})"
    )
    fdd_num_slices = generated_video_frames // args.fdd_slice_frames

    torch.manual_seed(0)
    latent_stats = json.loads(Path(args.latent_stats).read_text())
    model = LatentWorldModel(
        config.world_model, config.bottleneck, config.decoder, num_keys=config.num_keys,
        codec_checkpoint=args.codec_checkpoint, latent_mean=latent_stats["latent_mean"],
        latent_std=latent_stats["latent_std"], require_pretrained_dino=args.require_pretrained_dino,
    ).cuda()
    model.train()

    params = list(model.world_model.parameters()) + list(model.action_encoder.parameters()) + [model.bos]
    # Matches mira's own world-model optimizer betas (0.9, 0.99) -- note this differs from
    # train_codec.py's own (0.9, 0.95) for the codec.
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)
    # GradScaler(enabled=False) is a documented no-op under --precision bf16 -- same "no branching
    # to support both precisions" convention as train_codec.py; bf16 doesn't need loss scaling.
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16-hybrid"))

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    val_every = args.val_every if args.val_every is not None else max(1, args.steps // 50)
    drift_eval_every = args.drift_eval_every if args.drift_eval_every is not None else max(1, args.steps // 10)
    checkpoint_every = args.checkpoint_every if args.checkpoint_every is not None else max(1, args.steps // 10)
    console_log_every = args.console_log_every if args.console_log_every is not None else max(1, args.steps // 100)

    warmup_steps = args.lr_warmup_steps if args.lr_warmup_steps is not None else max(1, args.steps // 20)
    decay_steps = args.lr_decay_steps
    min_lr = args.lr_min
    lr_scheduler = WarmupConstantCosineDecayLR(
        optimizer, warmup_steps=warmup_steps, constant_steps=0, decay_steps=decay_steps, min_lr=min_lr,
    )

    # No seed passed here -- create_loader's own default (2025) applies every launch, and
    # num_workers isn't passed either (default 0, single-process). Both matter: dataloader-
    # position fast-forwarding below (skip N already-consumed batches on --resume) is only correct
    # because the un-seeded, single-process stream is deterministic across relaunches. Passing an
    # explicit seed or num_workers here later would silently break that assumption.
    train_loader = create_loader(
        index_path=args.index_path, clip_len=args.frames, target_fps=args.target_fps,
        n_players=1, batch_size=args.batch_size, frame_size=None,
    )
    train_iter = iter(train_loader)
    test_loader = create_loader(
        index_path=args.test_index_path, clip_len=args.frames, target_fps=args.target_fps,
        n_players=1, batch_size=eval_batch_size, frame_size=None, seed=37,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "checkpoint.pth"

    start_step = 0
    wandb_run_id = None
    batches_consumed = 0
    if args.resume and ckpt_path.exists():
        start_step, wandb_run_id, provenance = load_checkpoint(
            ckpt_path, model.world_model, model.action_encoder, model.bos, optimizer, lr_scheduler, grad_scaler
        )
        # Real risk this checks for: nothing else stops a --resume from silently pairing against a
        # different codec checkpoint or latent-stats file than the run was actually trained
        # against (the codec is loaded fresh from --codec-checkpoint every launch, never saved
        # into this checkpoint). A mismatch here doesn't crash -- it just trains on latents that
        # mean something different than what the model already learned. Warn loudly, don't block:
        # a moved/renamed-but-identical checkpoint file would otherwise trip this unnecessarily.
        prev_codec = provenance["codec_checkpoint"]
        if prev_codec is not None and prev_codec != str(Path(args.codec_checkpoint)):
            print(
                f"WARNING: this checkpoint was trained against codec checkpoint {prev_codec!r}, "
                f"but --codec-checkpoint is {args.codec_checkpoint!r} -- if these aren't the same "
                f"weights, training will silently corrupt on mismatched latents."
            )
        prev_mean, prev_std = provenance["latent_mean"], provenance["latent_std"]
        if prev_mean is not None and (prev_mean != latent_stats["latent_mean"] or prev_std != latent_stats["latent_std"]):
            print(
                f"WARNING: this checkpoint was trained with latent_mean={prev_mean}, "
                f"latent_std={prev_std}, but --latent-stats gives "
                f"{latent_stats['latent_mean']}/{latent_stats['latent_std']} -- mismatched "
                f"normalization, training will silently corrupt."
            )
        batches_consumed = provenance["dataloader_batches_consumed"]
        if batches_consumed > 0:
            print(f"Fast-forwarding dataloader past {batches_consumed} already-consumed batches...")
            for _ in range(batches_consumed):
                next(train_iter)
            print("Dataloader caught up.")
        # Same real, verified bug fix as train_codec.py: load_state_dict never pushes a
        # recomputed lr into optimizer.param_groups, and it restores the checkpoint's OWN
        # warmup/decay/min_lr shape wholesale (stale if --steps changed since the checkpoint was
        # made). Re-apply this run's own shape, then recompute+apply the real lr for the resumed
        # step -- see train_codec.py for the full writeup of why this matters for every --resume.
        #
        # load_state_dict above also restored base_lrs from the checkpoint (whatever --lr that
        # earlier run used) -- unlike train_codec.py this script has no --reset-lr-schedule flag,
        # but the same silent-override risk exists: a --resume with a DIFFERENT --lr than the
        # checkpoint's original would otherwise keep training at the OLD lr with no error or
        # warning. This script's own --lr is always meant to be authoritative (same "CLI overrides
        # loaded state" convention used elsewhere), so always re-derive it here -- a no-op when
        # --lr matches the checkpoint's original, a real fix when it doesn't.
        lr_scheduler.base_lrs = [args.lr for _ in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["initial_lr"] = args.lr
        lr_scheduler.warmup_steps = warmup_steps
        lr_scheduler.decay_steps = decay_steps
        lr_scheduler.min_lr = min_lr
        for group, lr in zip(optimizer.param_groups, lr_scheduler.get_lr()):
            group["lr"] = lr
        print(f"Resumed from {ckpt_path} at step {start_step}")

    # wandb_run_id: None on a fresh run (wandb.init mints a new one, captured below) or the id
    # loaded from the checkpoint above -- passing it back in continues that SAME wandb run
    # instead of --resume silently fragmenting the loss/lr history into a new, disconnected one.
    wandb_enabled = init_wandb(args.wandb_project, vars(args), run_id=wandb_run_id)
    wandb_run_id = get_wandb_run_id(wandb_enabled)
    torch.cuda.reset_peak_memory_stats()

    # Built lazily on the first full eval: constructing it loads InceptionV3 + LPIPS's AlexNet
    # backbone. If steps is small enough that eval never fires, that cost is never paid at all --
    # matches real mira's own WorldModelMetrics lazy-construction pattern.
    full_eval_metrics: FullEvalMetrics | None = None

    def _save_checkpoint_now(step: int) -> None:
        save_checkpoint(
            ckpt_path, step, model.world_model, model.action_encoder, model.bos, optimizer, lr_scheduler,
            grad_scaler, wandb_run_id,
            codec_checkpoint=args.codec_checkpoint,
            latent_mean=latent_stats["latent_mean"], latent_std=latent_stats["latent_std"],
            dataloader_batches_consumed=batches_consumed,
        )
        if args.hf_backup_repo:
            from huggingface_hub import HfApi  # noqa: PLC0415 -- optional dep, only used here

            HfApi().upload_file(
                path_or_fileobj=str(ckpt_path), path_in_repo="checkpoint.pth",
                repo_id=args.hf_backup_repo, repo_type="model",
            )

    for step in range(start_step, args.steps):
        optimizer.zero_grad()
        # Held as GPU tensors through the micro-step loop, not Python floats -- v.item() forces a
        # full CUDA sync (blocks the CPU until every queued kernel finishes), and doing that once
        # per loss term per micro-step (this loop's original behavior) serializes what should be
        # an async pipeline. Identical root cause, identical fix already proven for real in
        # train_codec_vjepa.py this project (found via nvidia-smi dmon showing the GPU
        # alternating 0%/100%), later confirmed live on train_world_model_vjepa.py too -- backbone-
        # agnostic bug, ported here so a future DINO run isn't artificially slower than V-JEPA's
        # for reasons unrelated to real architectural cost. Converted to floats once, after the
        # loop, instead -- same final numbers, far fewer sync points (was num_loss_terms *
        # grad_accum_steps per step, now num_loss_terms once).
        accumulated: dict[str, torch.Tensor] = {}
        for _ in range(args.grad_accum_steps):
            batch, _metadata = next(train_iter)
            batches_consumed += 1
            batch = resize_batch(batch, args.height, args.width).to("cuda")
            with _autocast(args.precision):
                losses = model(batch)
            grad_scaler.scale(losses["loss_total"] / args.grad_accum_steps).backward()
            for k, v in losses.items():
                term = v.detach() / args.grad_accum_steps
                accumulated[k] = term if k not in accumulated else accumulated[k] + term
        # Single sync point per loss term for the whole step, not one per term per micro-step.
        accumulated_floats: dict[str, float] = {k: v.item() for k, v in accumulated.items()}
        grad_scaler.step(optimizer)
        grad_scaler.update()
        lr_scheduler.step()

        # Checked here, right after a step fully lands and before any of this step's own
        # (potentially slow) eval work -- prioritizes actually getting the save done inside
        # `timeout`'s kill-after grace period over finishing this step's logging/eval first. Not
        # airtight: a signal arriving mid-eval (the slowest single thing this loop does) could
        # still race the grace period, but that's a narrow window against 11 hours of runtime.
        if _shutdown_requested:
            print(f"Shutdown signal received at step {step} -- saving and exiting cleanly.")
            _save_checkpoint_now(step)
            if wandb_enabled:
                import wandb  # noqa: PLC0415

                wandb.finish()
            return

        current_lr = optimizer.param_groups[0]["lr"]
        if (step + 1) % console_log_every == 0 or step == start_step:
            term_str = ", ".join(f"{k}={v:.4f}" for k, v in accumulated_floats.items() if k != "loss_total")
            print(f"step {step}: lr={current_lr:.2e} loss_total={accumulated_floats['loss_total']:.4f} ({term_str})")
            print(
                f"cuda_peak_allocated={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
                f"cuda_peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB"
            )
        log_step(wandb_enabled, step, accumulated_floats, current_lr)

        is_last = step == args.steps - 1
        if (step + 1) % checkpoint_every == 0 or is_last:
            _save_checkpoint_now(step)

        if (step + 1) % val_every == 0 or is_last:
            run_validation(
                model, test_loader, step, wandb_enabled, args.val_n_samples, eval_batch_size,
                args.height, args.width, current_lr, args.precision,
            )

        if (step + 1) % drift_eval_every == 0 or is_last:
            if full_eval_metrics is None:
                full_eval_metrics = FullEvalMetrics(
                    dino_dim=model.dino.dino_dim, fdd_slice_frames=args.fdd_slice_frames,
                    num_slices=fdd_num_slices, device="cuda",
                )
            run_full_eval(
                model, test_loader, step, wandb_enabled, args.drift_eval_n_samples, eval_batch_size,
                args.drift_eval_context_latents, args.drift_eval_diffusion_steps, args.drift_eval_schedule,
                args.height, args.width, current_lr, full_eval_metrics, args.viz_n_samples, args.target_fps,
                checkpoint_dir, args.precision,
            )


if __name__ == "__main__":
    main()
