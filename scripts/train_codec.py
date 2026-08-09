"""Config-driven codec training: DinoModel -> MyBottleneck -> ViTVideoDecoder, trained on the
real three-term CodecLoss (L1 + LPIPS + DINO latent-consistency, with auto_weight balancing --
see mini_mira.codec.loss), matching mira's own CodecLoss and its shipped config.

Two data modes:
  - Default (no --index-path): trains on one fixed synthetic video -- a mechanism check, not
    real training (see verify_codec_training.py for the CPU-friendly version of this).
  - --index-path <dir>: streams real clips via mira's own create_loader. Point it at the path
    scripts/download_shards.py prints after downloading.

Assumes a CUDA GPU (no CPU fallback) -- this script is for real training runs.

Precision: bfloat16. float16 and float32 both hit a cuDNN "unable to find an engine" error on
this project's V100 for DINOv3's patch_embed conv specifically; bfloat16 is the one that works.
See notes/gpu_amp_investigation.md for the full investigation.

Effective batch size is --batch-size (the real, per-forward micro-batch) times
--grad-accum-steps -- see mini_mira.codec.lr_schedule for the LR schedule shape, and
--activation-checkpointing to trade compute for the memory a larger micro-batch would need.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Real mira, cloned as a sibling repo -- reused for the dataloader.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from mira.data.training_loader import create_loader

from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.checkpoint import load_checkpoint, save_checkpoint
from mini_mira.codec.decoder import ViTVideoDecoder
from mini_mira.codec.dino import DinoModel
from mini_mira.codec.logging_utils import init_wandb, log_preview, log_step
from mini_mira.codec.loss import CodecLoss, CodecLossWeights, CodecOutputs, normalize_video
from mini_mira.codec.lr_schedule import WarmupConstantLinearDecayLR
from mini_mira.codec.video_prep import resize_to_canonical
from mini_mira.ml.config_loading import load_pipeline_config


def _autocast() -> torch.autocast:
    """bfloat16 -- the only precision that works for DINOv3's patch_embed conv on this GPU."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


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
        "--perceptual-dino-model", default=None,
        help="If set, score the DINO-consistency loss in a separate (typically smaller) DINO "
        "variant's feature space instead of the encoder's own, e.g. dinov3_vits16",
    )
    parser.add_argument("--lr-warmup-start", type=float, default=1e-7)
    parser.add_argument("--lr-warmup-steps", type=int, default=None, help="Default: steps // 50")
    parser.add_argument("--lr-decay-steps", type=int, default=None, help="Default: steps // 10")
    parser.add_argument("--lr-min", type=float, default=None, help="Default: --lr * 0.01")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=100)
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
    dino = DinoModel(require_pretrained=args.require_pretrained_dino).cuda()
    dino.eval()
    bottleneck = MyBottleneck(config.bottleneck).cuda()
    decoder = ViTVideoDecoder(config.decoder, use_checkpointing=args.activation_checkpointing).cuda()

    loss_fn = CodecLoss(
        CodecLossWeights(auto_weight=True), use_checkpointing=args.activation_checkpointing
    ).cuda()
    loss_fn.bind_encoder_dino(dino)
    if args.perceptual_dino_model:
        perceptual_dino = DinoModel(
            dino_model=args.perceptual_dino_model, require_pretrained=args.require_pretrained_dino
        ).cuda()
        perceptual_dino.eval()
        loss_fn.bind_perceptual_dino(perceptual_dino)
    loss_fn.bind_last_layer(decoder.last_layer_weight)

    params = list(bottleneck.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Long constant plateau in the middle -- proportional defaults (warmup ~2%, decay ~10% of
    # args.steps) rather than mira's literal 1000/249000 (sized for its 250,001-step run).
    warmup_steps = args.lr_warmup_steps if args.lr_warmup_steps is not None else max(1, args.steps // 50)
    decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else max(1, args.steps // 10)
    min_lr = args.lr_min if args.lr_min is not None else args.lr * 0.01
    lr_scheduler = WarmupConstantLinearDecayLR(
        optimizer, warmup_steps=warmup_steps, constant_steps=max(0, args.steps - warmup_steps - decay_steps),
        decay_steps=decay_steps, min_lr=min_lr, warmup_start_lr=args.lr_warmup_start,
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
            (losses["loss_total"] / args.grad_accum_steps).backward()
            for k, v in losses.items():
                accumulated[k] = accumulated.get(k, 0.0) + v.item() / args.grad_accum_steps
        optimizer.step()
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        term_str = ", ".join(
            f"{k}={v:.4f}" for k, v in accumulated.items() if k != "loss_total" and not k.endswith("_auto_w")
        )
        print(f"step {step}: lr={current_lr:.2e} loss_total={accumulated['loss_total']:.4f} ({term_str})")
        log_step(wandb_enabled, step, accumulated, current_lr)

        is_last = step == args.steps - 1
        if (step + 1) % args.checkpoint_every == 0 or is_last:
            save_checkpoint(ckpt_path, step, bottleneck, decoder, optimizer, lr_scheduler)
            log_preview(wandb_enabled, step, video, reconstructed)
            if args.hf_backup_repo:
                from huggingface_hub import HfApi  # optional dep, only used here

                HfApi().upload_file(
                    path_or_fileobj=str(ckpt_path), path_in_repo="checkpoint.pth",
                    repo_id=args.hf_backup_repo, repo_type="model",
                )


if __name__ == "__main__":
    main()
