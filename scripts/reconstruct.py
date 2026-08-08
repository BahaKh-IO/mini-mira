"""Run a real image or a real video clip through the codec (DinoModel -> MyBottleneck ->
ViTVideoDecoder) once and save a side-by-side comparison grid: top row = original frames in
time order, bottom row = reconstructed frames, aligned column by column.

Every weight here is randomly initialized -- train_codec.py has no trained checkpoint yet, and
this script doesn't train anything either. This does NOT prove the codec produces a good
reconstruction; it only proves the real-data path (load real frames -> resize -> normalize ->
encode -> decode -> save) works mechanically, end to end, with no shape errors or crashes.
Expect the bottom row to look like structured noise, not the top row, until real training
happens (see scripts/train_codec.py).

Images are treated as a single frame repeated across --frames identical frames (no real motion
to show). Video files (.mp4/.mov/.avi/.mkv/.webm) are decoded for real via imageio -- actual
distinct frames, actual motion.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.decoder import ViTVideoDecoder
from mini_mira.codec.dino import DinoModel
from mini_mira.codec.loss import denormalize_for_dino
from mini_mira.ml.config_loading import load_pipeline_config

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _resize_frame(frame: np.ndarray, height: int, width: int) -> torch.Tensor:
    """A single (H, W, 3) uint8 array -> a resized (3, height, width) tensor in [0, 1]."""
    image = Image.fromarray(frame).convert("RGB").resize((width, height))
    tensor = torch.from_numpy(np.array(image)).float() / 255.0  # (h, w, 3)
    return tensor.permute(2, 0, 1)  # (3, h, w)


def load_image_as_video(path: str, height: int, width: int, num_frames: int) -> torch.Tensor:
    """Load a single image, resize it, and repeat it across num_frames to form a static "video"
    tensor: (1, num_frames, 3, height, width), values in [0, 1] (what DinoModel expects)."""
    frame = _resize_frame(np.array(Image.open(path)), height, width)
    video = frame.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # (T, 3, H, W)
    return video.unsqueeze(0)  # (1, T, 3, H, W)


def load_video(path: str, height: int, width: int, num_frames: int) -> torch.Tensor:
    """Decode a real video file's first num_frames frames, resize them, and return
    (1, num_frames, 3, height, width), values in [0, 1]. Real, distinct frames -- real motion."""
    frames = []
    for frame in iio.imiter(path):
        frames.append(_resize_frame(frame, height, width))
        if len(frames) >= num_frames:
            break
    if len(frames) < num_frames:
        raise ValueError(f"{path} only has {len(frames)} frames, need at least {num_frames}")
    video = torch.stack(frames, dim=0)  # (T, 3, H, W)
    return video.unsqueeze(0)  # (1, T, 3, H, W)


def save_comparison_grid(original: torch.Tensor, reconstructed: torch.Tensor, path: str) -> None:
    """One image file: top row = original frames concatenated left to right in time order,
    bottom row = reconstructed frames, same column = same timestep.

    The decoder's raw output is [-1, 1] (its last layer is tanh) -- denormalize_for_dino's
    [-1,1]->[0,1] conversion is reused here purely for correct display, nothing to do with DINO.
    Clamping the raw [-1,1] output straight to [0,1] instead (an earlier bug here) would crush
    every negative pixel to solid black rather than rescaling it.
    """
    t = original.shape[1]
    top_row = torch.cat([original[0, i].clamp(0, 1) for i in range(t)], dim=2)  # cat along width
    bottom_row = torch.cat(
        [denormalize_for_dino(reconstructed[0, i]).clamp(0, 1) for i in range(t)], dim=2
    )
    grid = torch.cat([top_row, bottom_row], dim=1)  # cat along height: top row above bottom row
    grid = grid.permute(1, 2, 0).detach().numpy()
    Image.fromarray((grid * 255).astype("uint8")).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("media", help="Path to a real image (jpg/png/etc.) or video (.mp4/.mov/.avi/.mkv/.webm)")
    parser.add_argument("--config", default="configs/small.yaml", help="PipelineConfig-shaped YAML; only bottleneck/decoder sections are used")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--frames", type=int, default=4, help="Must be even (bottleneck temporal_stride=2)")
    parser.add_argument("--require-pretrained-dino", action="store_true", help="Use real gated DINOv3 weights instead of random-init")
    parser.add_argument("--out", default="comparison.png")
    args = parser.parse_args()

    if args.frames % 2 != 0:
        raise SystemExit(f"--frames must be even (bottleneck temporal_stride=2), got {args.frames}")

    config = load_pipeline_config(args.config)
    dino = DinoModel(require_pretrained=args.require_pretrained_dino)
    dino.eval()
    bottleneck = MyBottleneck(config.bottleneck)
    decoder = ViTVideoDecoder(config.decoder)
    bottleneck.eval()
    decoder.eval()

    is_video = Path(args.media).suffix.lower() in VIDEO_EXTENSIONS
    if is_video:
        video = load_video(args.media, args.height, args.width, args.frames)
    else:
        video = load_image_as_video(args.media, args.height, args.width, args.frames)

    with torch.no_grad():
        dino_features = dino.dino_forward(video)
        z = bottleneck(dino_features)
        reconstructed = decoder(z)

    save_comparison_grid(video, reconstructed, args.out)

    kind = "video clip" if is_video else "image (repeated, no real motion)"
    print(f"Loaded a real {kind}: {args.media}")
    print(f"Comparison grid (top=original, bottom=reconstruction) saved to: {args.out}")
    print("Every weight is randomly initialized -- this proves the real-data path works")
    print("mechanically, not that the codec reconstructs anything meaningful yet.")


if __name__ == "__main__":
    main()
