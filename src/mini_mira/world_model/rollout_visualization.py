"""Rollout video visualization for world-model eval: renders a small, fixed number of generated
rollouts as watchable video clips (keyboard-press overlay + prediction-border marking) for human
inspection -- separate from the numeric metrics in full_eval_metrics.py, though both are computed
from the same model.rollout(...) call (see train_world_model.run_full_eval).

Simplified from real mira's mira/src/mira/training/visualization.py -- see draw_key_grid_video's
docstring for why the keyboard HUD specifically can't be a direct port, and log_rollout_videos's
for why there's no separate videos_to_grid/videos_for_wandb split (mini_mira only ever logs a
small, fixed number of individually-keyed samples, never a large gridded batch).
"""

import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw
from torch import Tensor


def video_to_uint8(video: Tensor) -> Tensor:
    """Convert a floating-point video in [0, 1] to uint8 in [0, 255] (no-op if already uint8).
    Ported verbatim from real mira's visualization.py."""
    if video.dtype != torch.uint8:
        if not torch.is_floating_point(video):
            raise ValueError(f"Expected uint8 or floating point video tensor, got dtype {video.dtype}")
        # Cast to float32 first so rounding in reduced-precision dtypes doesn't push values
        # outside [0, 255].
        video = video.float()
        video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
    return video


def overlay_video(
    main_video: Tensor,
    overlay: Tensor,
    xpad: int = 0,
    ypad: int = 0,
    corner: str = "top-left",
    opacity: float = 0.5,
) -> Tensor:
    """Alpha-blend a smaller (T,C,h,w) overlay onto (T,C,H,W) at a chosen corner. Ported verbatim
    from real mira's visualization.py."""
    res = main_video.clone()
    _, _, video_h, video_w = main_video.shape
    _, _, overlay_h, overlay_w = overlay.shape

    if corner == "top-left":
        start_x, start_y = xpad, ypad
    elif corner == "top-right":
        start_x, start_y = video_w - overlay_w - xpad, ypad
    elif corner == "bottom-left":
        start_x, start_y = xpad, video_h - overlay_h - ypad
    else:  # bottom-right
        start_x, start_y = video_w - overlay_w - xpad, video_h - overlay_h - ypad

    region = res[:, :, start_y : start_y + overlay_h, start_x : start_x + overlay_w]
    if opacity >= 1.0:
        region[:] = overlay
    else:
        blended = (1.0 - opacity) * region.float() + opacity * overlay.float()
        res[:, :, start_y : start_y + overlay_h, start_x : start_x + overlay_w] = blended.to(res.dtype)
    return res


def add_prediction_border(
    video: Tensor, context: int, color: tuple = (128, 0, 32), border: int = 4, clone: bool = True
) -> Tensor:
    """Draw a coloured border around frames at index `context` onward (marks predicted, not
    given-as-context, frames). video: (T,C,H,W) -- a single sample, not batched. Real mira's
    version operates on batched (B,T,C,H,W) video; this module always renders one rollout sample
    at a time (see render_rollout_sample), so the indexing below is one dim shorter -- adapted,
    not a verbatim port, unlike video_to_uint8/overlay_video above."""
    video = video_to_uint8(video)
    video_output = video.clone() if clone else video

    c = rearrange(torch.tensor(color, device=video_output.device), "c -> c 1 1")
    video_output[context:, :, :border, :] = c
    video_output[context:, :, -border:, :] = c
    video_output[context:, :, :, :border] = c
    video_output[context:, :, :, -border:] = c
    return video_output


def draw_key_grid_video(
    key_presses: Tensor,
    width: int = 200,
    height: int = 40,
    background_color: tuple = (0, 0, 0),
    key_color: tuple = (50, 50, 50),
    pressed_color: tuple = (100, 200, 255),
) -> Tensor:
    """Simplified keyboard HUD: num_keys numbered boxes in a single row that light up when
    pressed. Real mira's draw_keyboard_video renders a real keyboard graphic with named,
    positioned keys (its LAYOUT table: "Q","W","E"... at real QWERTY positions) -- mini_mira's
    ActionEncoder has no key-name-to-index mapping at all (numeric indices 0..num_keys-1 only,
    see notes/deviations.md 1.10), so there is nothing to position on a keyboard graphic. This
    draws the same information (which key indices are active per frame) as a row of numbered
    boxes instead. No fade-out effect (real mira's fade_frames) -- dropped as an unrequested
    nicety; instantaneous on/off is enough for visual inspection.

    key_presses: (T, num_keys) 0/1. Returns (T, 3, height, width) uint8.
    """
    n_steps, num_keys = key_presses.shape
    key_presses_np = key_presses.detach().cpu().numpy()

    padding = 2
    gap = 2
    box_width = (width - 2 * padding - gap * (num_keys - 1)) / num_keys
    box_height = height - 2 * padding

    frames = []
    for t in range(n_steps):
        img = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(img)
        for k in range(num_keys):
            x = padding + k * (box_width + gap)
            color = pressed_color if key_presses_np[t, k] > 0 else key_color
            draw.rectangle([x, padding, x + box_width, padding + box_height], fill=color)
            draw.text((x + box_width / 2 - 3, padding + box_height / 2 - 6), str(k), fill=(255, 255, 255))
        frames.append(torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1))
    return torch.stack(frames, dim=0)


def render_rollout_sample(pred_video: Tensor, key_presses: Tensor, n_context_frames: int) -> Tensor:
    """One (T,3,H,W) sample -> annotated uint8 (T,3,H,W): keyboard HUD overlay top-right
    (overlay_height = H // 4, overlay_width = 2.5x that, opacity 0.5 -- matches real mira's
    visualize_sample sizing), then the prediction border drawn last so it's always on top.

    key_presses: (T, num_keys), already aligned 1:1 with pred_video's T frames (this project's
    action_fps always equals video fps -- see LatentWorldModel._encode) -- caller's job to slice
    to the matching window, this function does no alignment of its own.
    """
    video = video_to_uint8(pred_video)
    overlay_height = video.shape[2] // 4
    overlay_width = int(overlay_height * 2.5)
    key_overlay = draw_key_grid_video(key_presses, width=overlay_width, height=overlay_height)
    video = overlay_video(video, key_overlay, corner="top-right", xpad=10, ypad=10, opacity=0.5)
    return add_prediction_border(video, context=n_context_frames)


def write_video_ffmpeg(
    filename: str, video_tensor: Tensor, fps: float = 10, video_codec: str = "libx264", crf: int = 23,
    preset: str = "medium",
) -> None:
    """Write a (T,C,H,W) video to `filename` via ffmpeg (video-only, no audio). Ported from real
    mira's write_video_ffmpeg with one deviation: invokes imageio_ffmpeg's bundled binary
    (imageio-ffmpeg is already a mini_mira dependency -- mini_mira.codec.logging_utils.log_preview
    uses it via imageio) instead of a bare "ffmpeg" needing system PATH. This dev machine has no
    system ffmpeg on PATH, so the bare-"ffmpeg" version would be untestable here and is less
    portable generally; the GPU box's own system ffmpeg (already confirmed present, needed by
    torchcodec itself) works fine either way since imageio_ffmpeg only supplies its own binary
    when nothing better is configured.
    """
    import imageio_ffmpeg  # noqa: PLC0415 -- only needed here

    video_np = video_to_uint8(video_tensor).permute(0, 2, 3, 1).detach().cpu().numpy()
    _, height, width, _ = video_np.shape

    v_args = [
        "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{width}x{height}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
    ]
    out_args = ["-vcodec", video_codec, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", *v_args, *out_args, str(filename)]
    result = subprocess.run(cmd, input=video_np.tobytes(), capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed ({result.returncode}):\n{result.stderr.decode()}")


def log_rollout_videos(
    viz_samples: list[Tensor], fps: int, step: int, wandb_enabled: bool, output_dir: str | Path | None = None
) -> None:
    """Encodes each sample to mp4 -- permanently under output_dir/rollout_previews/ if given,
    else a throwaway temp file used only for the wandb upload -- and, if wandb_enabled, logs each
    as "rollout_video/sample_N". Always encodes somewhere before deciding whether to log, so
    wandb logging works correctly even when output_dir is None (an earlier draft of this function
    only ever encoded when output_dir was set, which would have silently skipped the wandb upload
    whenever previews weren't also being saved to disk -- fixed before this ever shipped).

    Matches log_preview's non-fatal-on-failure convention: a rollout video encoding/logging
    problem must never look like, or cause, a training failure.
    """
    media_paths: dict[str, str] = {}
    tmp_paths: list[str] = []
    try:
        for i, sample in enumerate(viz_samples):
            if output_dir is not None:
                preview_dir = Path(output_dir) / "rollout_previews"
                preview_dir.mkdir(parents=True, exist_ok=True)
                path = preview_dir / f"step_{step:06d}_sample_{i}.mp4"
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp.close()
                tmp_paths.append(tmp.name)
                path = Path(tmp.name)

            try:
                write_video_ffmpeg(str(path), sample.cpu(), fps=fps)
                if output_dir is not None:
                    print(f"Saved rollout preview to {path}")
            except Exception as exc:  # media encoding must never discard a completed eval
                warnings.warn(f"Rollout video encoding failed at step {step}, sample {i}: {exc}", stacklevel=2)
                continue

            if wandb_enabled:
                media_paths[f"rollout_video/sample_{i}"] = str(path)

        if wandb_enabled and media_paths:
            try:
                import wandb  # noqa: PLC0415 -- optional dep, only imported when actually used

                wandb.log({k: wandb.Video(v, format="mp4") for k, v in media_paths.items()}, step=step)
            except Exception as exc:  # logging must never discard a completed eval
                warnings.warn(f"W&B rollout video logging failed at step {step}: {exc}", stacklevel=2)
    finally:
        for tmp_path in tmp_paths:
            Path(tmp_path).unlink(missing_ok=True)
