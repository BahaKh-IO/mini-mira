"""Config-driven codec training: DinoModel -> MyBottleneck -> ViTVideoDecoder, trained on a
reconstruction loss (real mira's loss_mae; LPIPS/DINO-consistency terms not implemented -- see
notes/deviations.md 2.2, still an open question for the supervisor).

No real dataset exists yet (see README's Scope) -- this trains on ONE fixed synthetic video,
same "overfit one example" approach as verify_codec_training.py, just config-driven and meant to
actually be pointed at real GPU compute once that's available, not a mechanism sanity check.

Defaults are deliberately a toy size (small config, 64x64, 4 frames) so this runs in seconds
here. Measured directly: even the small config at real mira's actual training resolution
(288x512, 40 frames) takes ~22.7s/step on this CPU -- DINO's own forward cost dominates
regardless of decoder size -- and configs/scaled_300m.yaml's decoder is much larger again (see
README's Scale section for real per-step timing at that config). Override --config/--height/
--width/--frames for the real run once there's a GPU.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F
from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.decoder import ViTVideoDecoder
from mini_mira.codec.dino import DinoModel
from mini_mira.ml.config_loading import load_pipeline_config


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
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--require-pretrained-dino", action="store_true",
        help="Use real gated DINOv3 weights (needs RS_DINO_WEIGHTS_DIR set) instead of random-init.",
    )
    args = parser.parse_args()

    config = load_pipeline_config(args.config)

    torch.manual_seed(0)
    dino = DinoModel(require_pretrained=args.require_pretrained_dino)
    dino.eval()
    bottleneck = MyBottleneck(config.bottleneck)
    decoder = ViTVideoDecoder(config.decoder)

    params = list(bottleneck.parameters()) + list(decoder.parameters())
    # betas/weight_decay match real mira's configs/train_codec.yaml; no warmup/cosine schedule
    # yet (see notes/deviations.md) -- plain AdamW is enough to prove the mechanism.
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # No real dataloader exists yet -- one fixed synthetic video stands in. Swap this block for
    # a real dataloader once the standalone-vs-import decision (notes/deviations.md 2.2) is made.
    video = torch.rand(1, args.frames, 3, args.height, args.width)

    for step in range(args.steps):
        optimizer.zero_grad()
        dino_features = dino.dino_forward(video)
        z = bottleneck(dino_features)
        reconstructed = decoder(z)
        loss = F.l1_loss(reconstructed, video)
        loss.backward()
        optimizer.step()
        print(f"step {step}: loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
