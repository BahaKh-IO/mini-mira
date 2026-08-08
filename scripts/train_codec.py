"""Config-driven codec training: DinoModel -> MyBottleneck -> ViTVideoDecoder, trained on the
real three-term CodecLoss (L1 + LPIPS + DINO latent-consistency, with auto_weight adaptive
balancing -- see mini_mira.codec.loss), matching mira's own CodecLoss and its shipped config.

Two data modes:
  - Default (no --index-path): trains on ONE fixed synthetic video, same "overfit one example"
    approach as verify_codec_training.py -- a fast, no-dependencies mechanism check, not real
    training. Unchanged from before this script grew real-data support.
  - --index-path <dir>: streams real clips via mira's own create_loader (mira.data.training_loader)
    -- the output of scripts/download_shards.py (it prints the exact path to pass here). This is
    the real training path; everything below (checkpointing, wandb, AMP, the LR schedule) exists
    for this mode.

Real mira's actual training resolution/clip length is 288x512, 40 frames @ 20fps -- pass
--height 288 --width 512 --frames 40 for a real run; the defaults here stay small (64x64, 4
frames) so the synthetic no-args path keeps running in seconds, matching this script's original
behavior. Measured directly: even the small config at real resolution takes ~48.9s/step on this
CPU with the current (full) loss -- DINO + LPIPS both add real cost -- so real speed depends on
actually running this on a GPU; see notes/deviations.md and the README's Scale section for the
CPU numbers this estimate is grounded in.
"""

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Real mira, cloned as a sibling repo next to mini_mira -- reused directly for the dataloader and
# LR schedule (generic training infrastructure, not the architecture this project exists to
# teach -- same reasoning already applied to RocketScienceDataset in download_shards.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.checkpoint import load_checkpoint, save_checkpoint
from mini_mira.codec.decoder import ViTVideoDecoder
from mini_mira.codec.dino import DinoModel
from mini_mira.codec.logging_utils import init_wandb, log_preview, log_step
from mini_mira.codec.loss import CodecLoss, CodecLossWeights, CodecOutputs, normalize_video
from mini_mira.codec.video_prep import resize_to_canonical
from mini_mira.ml.config_loading import load_pipeline_config


def _autocast(device: torch.device):
    """bfloat16 autocast on CUDA, a no-op elsewhere (so this script runs unchanged on CPU) --
    copied from mira's exact pattern (mira/scripts/train_codec.py)."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", default="configs/small.yaml",
        help="Path to a PipelineConfig-shaped YAML (configs/small.yaml or configs/scaled_300m.yaml) "
             "-- only its bottleneck/decoder sections are used, world_model/actions are ignored.",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)  # matches real mira's configs/train_codec.yaml
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--frames", type=int, default=4, help="Clip length in frames (both data modes)")
    parser.add_argument(
        "--require-pretrained-dino", action="store_true",
        help="Use real gated DINOv3 weights (needs RS_DINO_WEIGHTS_DIR set) instead of random-init.",
    )

    # Real-data mode (Part 1). Omit --index-path to keep training on the synthetic video.
    parser.add_argument(
        "--index-path", default=None,
        help="Local dataset directory (or its index.json) from scripts/download_shards.py's "
             "printed path. Switches from the synthetic video to real streamed clips.",
    )
    parser.add_argument("--target-fps", type=int, default=20, help="Matches mira's codec (encoder.video.fps)")
    parser.add_argument("--batch-size", type=int, default=4, help="Matches mira's train_codec.yaml")

    # Checkpointing (Part 6).
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint-dir if present")
    parser.add_argument(
        "--hf-backup-repo", default=None,
        help="Optional HF Hub model repo (e.g. 'username/mini-mira-checkpoints') to upload each "
             "checkpoint to -- for a rented GPU box whose disk may not survive between rentals.",
    )

    # wandb (Part 7). Off unless a project name is given.
    parser.add_argument("--wandb-project", default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_pipeline_config(args.config)

    torch.manual_seed(0)
    dino = DinoModel(require_pretrained=args.require_pretrained_dino).to(device)
    dino.eval()
    bottleneck = MyBottleneck(config.bottleneck).to(device)
    decoder = ViTVideoDecoder(config.decoder).to(device)

    # auto_weight=True matches mira's own shipped train config (configs/model/raev2_codec_tdown.yaml)
    # -- see mini_mira.codec.loss for why this was worth building now instead of staying deferred.
    loss_fn = CodecLoss(CodecLossWeights(auto_weight=True)).to(device)
    loss_fn.bind_encoder_dino(dino)  # share the same frozen backbone, not a second copy
    loss_fn.bind_last_layer(decoder.last_layer_weight)

    params = list(bottleneck.parameters()) + list(decoder.parameters())
    # betas/weight_decay match real mira's configs/train_codec.yaml.
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Warmup ~5% of steps, then cosine decay the rest -- proportional, not mira's literal
    # 1000/249000 (those were sized for a 250,001-step run; this preserves the same *shape* of
    # schedule at whatever length this run actually is). See mira/src/mira/training/lr_schedule.py.
    from mira.training.lr_schedule import WarmupConstantCosineDecayLR  # noqa: E402

    warmup_steps = max(1, args.steps // 20)
    lr_scheduler = WarmupConstantCosineDecayLR(
        optimizer, warmup_steps=warmup_steps, constant_steps=0,
        decay_steps=max(1, args.steps - warmup_steps), min_lr=args.lr * 0.01,
    )

    # --- data: real streamed clips if --index-path is set, else the synthetic stand-in (Part 1) ---
    if args.index_path:
        from mira.data.training_loader import create_loader  # noqa: E402

        loader = create_loader(
            index_path=args.index_path,
            clip_len=args.frames,
            target_fps=args.target_fps,
            n_players=1,  # codec training; matches mira's configs/dataset/rocket_league.yaml
            batch_size=args.batch_size,
            frame_size=None,  # native decode -- resize_to_canonical below does the pad+resize,
                               # matching mira's own split between the loader and preprocess_batch
        )
        data_iter = iter(loader)

        def next_video() -> torch.Tensor:
            batch, _metadata = next(data_iter)  # (VideoActionBatch, list[ClipMeta]); codec ignores actions
            video = batch.video.float() / 255.0  # uint8 (B,T,C,H,W) -> float [0,1]
            return resize_to_canonical(video, args.height, args.width)
    else:
        fixed_video = torch.rand(1, args.frames, 3, args.height, args.width)

        def next_video() -> torch.Tensor:
            return fixed_video

    # --- checkpointing + wandb setup ---
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "checkpoint.pth"

    start_step = 0
    if args.resume and ckpt_path.exists():
        start_step = load_checkpoint(ckpt_path, bottleneck, decoder, optimizer, lr_scheduler)
        print(f"Resumed from {ckpt_path} at step {start_step}")

    wandb_enabled = init_wandb(args.wandb_project, vars(args))

    for step in range(start_step, args.steps):
        video = next_video().to(device)
        optimizer.zero_grad()
        with _autocast(device):
            with torch.no_grad():  # encoder side: target features never need a gradient (see dino.py)
                dino_features = dino.dino_forward(video)
            z = bottleneck(dino_features)
            reconstructed = decoder(z)
            outputs = CodecOutputs(
                input_video=normalize_video(video), output_video=reconstructed, dino_features=dino_features
            )
            losses = loss_fn(outputs)
        losses["loss_total"].backward()
        optimizer.step()
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        term_str = ", ".join(
            f"{k}={v.item():.4f}" for k, v in losses.items() if k != "loss_total" and not k.endswith("_auto_w")
        )
        print(f"step {step}: lr={current_lr:.2e} loss_total={losses['loss_total'].item():.4f} ({term_str})")
        log_step(wandb_enabled, step, losses, current_lr)

        is_last = step == args.steps - 1
        if (step + 1) % args.checkpoint_every == 0 or is_last:
            save_checkpoint(ckpt_path, step, bottleneck, decoder, optimizer, lr_scheduler)
            log_preview(wandb_enabled, step, video, reconstructed)
            if args.hf_backup_repo:
                from huggingface_hub import HfApi  # noqa: PLC0415 -- optional dep, only used here

                HfApi().upload_file(
                    path_or_fileobj=str(ckpt_path), path_in_repo="checkpoint.pth",
                    repo_id=args.hf_backup_repo, repo_type="model",
                )


if __name__ == "__main__":
    main()
