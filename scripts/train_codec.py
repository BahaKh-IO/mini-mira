"""Config-driven codec training: DinoModel -> MyBottleneck -> ViTVideoDecoder, trained on the
real three-term CodecLoss (L1 + LPIPS + DINO latent-consistency, with auto_weight balancing --
see mini_mira.codec.loss), matching mira's own CodecLoss and its shipped config.

Two data modes:
  - Default (no --index-path): trains on one fixed synthetic video -- a mechanism check, not
    real training (see verify_codec_training.py for the CPU-friendly version of this).
  - --index-path <dir>: streams real clips via mira's own create_loader. Point it at the path
    scripts/download_shards.py prints after downloading.

Assumes a CUDA GPU (no CPU fallback) -- this script is for real training runs.

Precision: float16 autocast for the trainable codec, with its convolution-family modules and
the complete frozen DINO backbones kept in bfloat16. This avoids this V100/cuDNN stack's FP16
convolution engine failures and DINO's FP16 numerical instability.

Effective batch size is --batch-size (the real, per-forward micro-batch) times
--grad-accum-steps -- see mira.training.lr_schedule.WarmupConstantCosineDecayLR for the LR
schedule shape, and --activation-checkpointing to trade compute for the memory a larger
micro-batch would need.
"""

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Real mira, cloned as a sibling repo -- reused for the dataloader.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from mira.data.training_loader import create_loader
from mira.training.lr_schedule import WarmupConstantCosineDecayLR

from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.checkpoint import load_checkpoint, save_checkpoint
from mini_mira.codec.decoder import ViTVideoDecoder
from mini_mira.codec.dino import DEFAULT_ENCODER_AGGREGATION_LAYERS, DinoModel
from mini_mira.codec.logging_utils import init_wandb, log_preview, log_step
from mini_mira.codec.loss import CodecLoss, CodecLossWeights, CodecOutputs, normalize_video
from mini_mira.codec.video_prep import resize_to_canonical
from mini_mira.ml.config_loading import load_pipeline_config


def _autocast() -> torch.autocast:
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


_CONVOLUTION_TYPES = (
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
    torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
)


def _keep_convolutions_in_bf16(module: torch.nn.Module) -> None:
    """Nest BF16 autocast around convolutions while the surrounding model uses FP16 AMP."""
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
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)  # matches mira's train_codec.yaml
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--frames", type=int, default=4, help="Clip length in frames (both data modes)")
    parser.add_argument("--require-pretrained-dino", action="store_true")
    parser.add_argument("--index-path", default=None, help="Real dataset dir from download_shards.py")
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-forward micro-batch size")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="Micro-batches accumulated per optimizer step (effective batch = batch-size * this)",
    )
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument(
        "--perceptual-chunk-size", type=int, default=0,
        help="Frames per LPIPS/DINO loss forward (0 processes the selected frames together)",
    )
    parser.add_argument(
        "--perceptual-dino-model", default=None,
        help="If set, score the DINO-consistency loss in a separate (typically smaller) DINO "
        "variant's feature space instead of the encoder's own, e.g. dinov3_vits16",
    )
    parser.add_argument(
        "--perceptual-dino-multilayer", action="store_true",
        help="Aggregate multiple DINO layers for the consistency loss (matches mira's "
        "DinoPerceptualLoss) instead of just the last one. Requires --perceptual-dino-model.",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=None, help="Default: steps // 20")
    parser.add_argument("--lr-decay-steps", type=int, default=None, help="Default: steps - warmup_steps")
    parser.add_argument("--lr-min", type=float, default=None, help="Default: --lr * 0.01")
    parser.add_argument("--loss-mae-weight", type=float, default=1.0, help="CodecLossWeights.loss_mae")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--preview-every", type=int, default=100, help="W&B image/video preview interval")
    parser.add_argument("--console-log-every", type=int, default=10, help="Loss/GPU-memory print interval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hf-backup-repo", default=None, help="Optional HF Hub repo to back up checkpoints to")
    parser.add_argument("--wandb-project", default=None)
    return parser.parse_args()


def build_next_video(args: argparse.Namespace):
    """Real streamed clips if --index-path is set, else one fixed synthetic video every step."""
    if args.index_path:
        loader = create_loader(
            index_path=args.index_path,
            clip_len=args.frames,
            target_fps=args.target_fps,
            n_players=1,  # codec training
            batch_size=args.batch_size,
            frame_size=None,  # native decode -- resize_to_canonical below does the pad+resize
        )
        data_iter = iter(loader)

        def next_video() -> torch.Tensor:
            batch, _metadata = next(data_iter)  # codec ignores actions
            video = batch.video.float() / 255.0  # uint8 (B,T,C,H,W) -> float [0,1]
            return resize_to_canonical(video, args.height, args.width)
    else:
        fixed_video = torch.rand(1, args.frames, 3, args.height, args.width)

        def next_video() -> torch.Tensor:
            return fixed_video

    return next_video


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)

    torch.manual_seed(0)
    # dinov3_vitb16 is the only variant with weights downloaded for this project (real mira's
    # encoder always uses vitl16) -- matches mira's RAEEncoder in every other respect: multi-layer
    # aggregation on, layer_indices from DEFAULT_ENCODER_AGGREGATION_LAYERS.
    encoder_dino_model = "dinov3_vitb16"
    dino = DinoModel(
        dino_model=encoder_dino_model, require_pretrained=args.require_pretrained_dino,
        last_layer_only=False, layer_indices=DEFAULT_ENCODER_AGGREGATION_LAYERS[encoder_dino_model],
    ).cuda()
    dino.eval()
    bottleneck = MyBottleneck(config.bottleneck, use_checkpointing=args.activation_checkpointing).cuda()
    decoder = ViTVideoDecoder(config.decoder, use_checkpointing=args.activation_checkpointing).cuda()

    loss_fn = CodecLoss(
        CodecLossWeights(auto_weight=True, loss_mae=args.loss_mae_weight),
        use_checkpointing=args.activation_checkpointing,
        perceptual_chunk_size=args.perceptual_chunk_size,
    ).cuda()
    loss_fn.bind_encoder_dino(dino)
    if args.perceptual_dino_model:
        perceptual_dino = DinoModel(
            dino_model=args.perceptual_dino_model, require_pretrained=args.require_pretrained_dino,
            last_layer_only=not args.perceptual_dino_multilayer,
        ).cuda()
        perceptual_dino.eval()
        loss_fn.bind_perceptual_dino(perceptual_dino)
    loss_fn.bind_last_layer(decoder.last_layer_weight)

    for module in (dino, bottleneck, decoder, loss_fn):
        _keep_convolutions_in_bf16(module)

    params = list(bottleneck.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    grad_scaler = torch.amp.GradScaler("cuda")

    # mira's own schedule shape: warmup ~5% of args.steps, then cosine decay for the rest, no
    # constant plateau -- proportional to args.steps rather than mira's literal 1000/249000
    # (those were sized for its 250,001-step run).
    warmup_steps = args.lr_warmup_steps if args.lr_warmup_steps is not None else max(1, args.steps // 20)
    decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else max(1, args.steps - warmup_steps)
    min_lr = args.lr_min if args.lr_min is not None else args.lr * 0.01
    lr_scheduler = WarmupConstantCosineDecayLR(
        optimizer, warmup_steps=warmup_steps, constant_steps=0, decay_steps=decay_steps, min_lr=min_lr,
    )

    next_video = build_next_video(args)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "checkpoint.pth"

    start_step = 0
    if args.resume and ckpt_path.exists():
        start_step = load_checkpoint(ckpt_path, bottleneck, decoder, optimizer, lr_scheduler)
        print(f"Resumed from {ckpt_path} at step {start_step}")

    wandb_enabled = init_wandb(args.wandb_project, vars(args))
    torch.cuda.reset_peak_memory_stats()

    for step in range(start_step, args.steps):
        optimizer.zero_grad()
        accumulated: dict[str, float] = {}
        for _ in range(args.grad_accum_steps):
            video = next_video().cuda()
            with _autocast():
                with torch.no_grad():  # encoder side: no grad needed here
                    dino_features = dino.dino_forward(video)
                z = bottleneck(dino_features)
                reconstructed = decoder(z)
                outputs = CodecOutputs(
                    input_video=normalize_video(video), output_video=reconstructed, dino_features=dino_features
                )
                losses = loss_fn(outputs)
            grad_scaler.scale(losses["loss_total"] / args.grad_accum_steps).backward()
            for k, v in losses.items():
                accumulated[k] = accumulated.get(k, 0.0) + v.item() / args.grad_accum_steps
        grad_scaler.step(optimizer)
        grad_scaler.update()
        lr_scheduler.step()

        # Last micro-step's real per-term gradient norms (see CodecLoss._hook_clone) -- same
        # "last micro-step only" convention already used for the preview video below.
        grad_norms = {f"grad_norm_{k}": v.item() for k, v in loss_fn.backward_metrics.items()}

        current_lr = optimizer.param_groups[0]["lr"]
        term_str = ", ".join(
            f"{k}={v:.4f}" for k, v in accumulated.items() if k != "loss_total" and not k.endswith("_auto_w")
        )
        if (step + 1) % args.console_log_every == 0 or step == start_step:
            print(f"step {step}: lr={current_lr:.2e} loss_total={accumulated['loss_total']:.4f} ({term_str})")
            print(
                f"cuda_peak_allocated={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
                f"cuda_peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB"
            )
            print(", ".join(f"{k}={v:.4f}" for k, v in grad_norms.items()))
        log_step(wandb_enabled, step, {**accumulated, **grad_norms}, current_lr)

        is_last = step == args.steps - 1
        if (step + 1) % args.checkpoint_every == 0 or is_last:
            save_checkpoint(ckpt_path, step, bottleneck, decoder, optimizer, lr_scheduler)
            if args.hf_backup_repo:
                from huggingface_hub import HfApi  # optional dep, only used here

                HfApi().upload_file(
                    path_or_fileobj=str(ckpt_path), path_in_repo="checkpoint.pth",
                    repo_id=args.hf_backup_repo, repo_type="model",
                )
        # Always persist the first completed step so a new run proves its output path before
        # committing hours of compute. Subsequent samples follow the configured interval.
        if step == start_step or (step + 1) % args.preview_every == 0 or is_last:
            log_preview(
                wandb_enabled, step, video, reconstructed, fps=args.target_fps,
                output_dir=checkpoint_dir / "previews",
            )


if __name__ == "__main__":
    main()
