"""Real-data, real-GPU diagnostic: pull ONE real clip and overfit the V-JEPA codec on just that
clip, repeated every step -- same "can it even memorize one example" logic as
verify_codec_training.py, but with a real Rocket League clip instead of synthetic noise, on real
GPU hardware, with real pretrained V-JEPA weights. Nothing to generalize here, only one example to
memorize -- if reconstruction quality still looks bad after enough steps, that's a real,
fundamental problem, not a "needs more training" question.

No checkpoint saving -- this is a pure diagnostic, not meant to produce a usable checkpoint.
Progress and preview images go to wandb only (no local preview files either).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from mira.data.training_loader import create_loader

from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.decoder import ViTVideoDecoder
from mini_mira.codec.logging_utils import init_wandb, log_preview, log_step
from mini_mira.codec.loss import CodecLoss, CodecLossWeights, CodecOutputs, normalize_video
from mini_mira.codec.vjepa import DEFAULT_VJEPA_LAYERS, VjepaModel
from mini_mira.codec.video_prep import resize_to_canonical
from mini_mira.ml.config_loading import load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/scaled_300m_vjepa.yaml")
    parser.add_argument("--index-path", required=True, help="Real dataset dir from download_shards.py")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300, help="One real example to memorize -- should show real progress well before this")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", choices=["fp16-hybrid", "bf16"], default="bf16")
    parser.add_argument("--activation-checkpointing", action="store_true", help="Trade compute for memory -- needed at larger --height/--width")
    parser.add_argument("--preview-every", type=int, default=25)
    parser.add_argument("--console-log-every", type=int, default=10)
    parser.add_argument("--wandb-project", default=None)
    return parser.parse_args()


def _autocast(precision: str) -> torch.autocast:
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)

    torch.manual_seed(0)
    # require_pretrained=True unconditionally -- there's no sensible reason to run this specific
    # diagnostic against random-init weights, the whole point is checking the real ones.
    vjepa = VjepaModel(
        require_pretrained=True, last_layer_only=False, layer_indices=DEFAULT_VJEPA_LAYERS,
    ).cuda()
    vjepa.eval()
    bottleneck = MyBottleneck(config.bottleneck, use_checkpointing=args.activation_checkpointing).cuda()
    decoder = ViTVideoDecoder(config.decoder, use_checkpointing=args.activation_checkpointing).cuda()
    loss_fn = CodecLoss(CodecLossWeights(auto_weight=True), use_checkpointing=args.activation_checkpointing).cuda()
    loss_fn.bind_encoder_dino(vjepa)
    loss_fn.bind_last_layer(decoder.last_layer_weight)

    params = list(bottleneck.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16-hybrid"))
    loss_fn.bind_grad_scaler(grad_scaler)

    # One real batch, pulled once, reused every step -- the actual overfit target. batch_size=1:
    # exactly one clip, matching "feed the decoder one single clip" literally.
    loader = create_loader(
        index_path=args.index_path, clip_len=args.frames, target_fps=args.target_fps,
        n_players=1, batch_size=1, frame_size=None,
    )
    batch, _metadata = next(iter(loader))
    fixed_video = resize_to_canonical(batch.video.float() / 255.0, args.height, args.width).cuda()
    print(f"Overfitting one real clip: {tuple(fixed_video.shape)}")

    wandb_enabled = init_wandb(args.wandb_project, vars(args))

    for step in range(args.steps):
        optimizer.zero_grad()
        with _autocast(args.precision):
            with torch.no_grad():
                dino_features = vjepa.dino_forward(fixed_video)
            z = bottleneck(dino_features)
            reconstructed = decoder(z)
            outputs = CodecOutputs(
                input_video=normalize_video(fixed_video), output_video=reconstructed, dino_features=dino_features
            )
            losses = loss_fn(outputs)
        grad_scaler.scale(losses["loss_total"]).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        if (step + 1) % args.console_log_every == 0 or step == 0:
            term_str = ", ".join(
                f"{k}={v:.4f}" for k, v in losses.items() if k != "loss_total" and not k.endswith("_auto_w")
            )
            print(f"step {step}: loss_total={losses['loss_total'].item():.4f} ({term_str})")
        log_step(wandb_enabled, step, {k: v.item() for k, v in losses.items()}, args.lr)

        # output_dir=None deliberately -- wandb only, no local files left behind on the shared box.
        if step == 0 or (step + 1) % args.preview_every == 0 or step == args.steps - 1:
            log_preview(wandb_enabled, step, fixed_video, reconstructed, fps=args.target_fps, output_dir=None)

    print("Done. Check wandb for the loss curve and preview images -- if reconstruction still")
    print("looks bad by the final preview, that's a real, fundamental problem, not 'needs more steps'.")


if __name__ == "__main__":
    main()
