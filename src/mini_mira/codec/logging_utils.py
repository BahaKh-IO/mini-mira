"""Optional wandb logging for train_codec.py.

Every function here is a no-op when disabled, and `wandb` is imported lazily inside each function
(never at module import time) -- so a run without --wandb-project never needs wandb installed at
all, matching how mira's own scripts scope the `wandb`/`lpips` imports to only where they're used.
"""

from pathlib import Path
from typing import Any
import warnings

import torch
from torch import Tensor


def init_wandb(project: str | None, config: dict[str, Any], run_id: str | None = None) -> bool:
    """Starts a wandb run if `project` is set. Returns whether logging is enabled.

    run_id: pass a previously-saved run id (see get_wandb_run_id) to CONTINUE that run instead of
    starting a fresh one. Without this, every --resume silently opened a brand-new wandb run,
    fragmenting one training run's loss/lr history across several disconnected entries instead of
    one continuous line."""
    if not project:
        return False
    import wandb  # noqa: PLC0415 -- optional dep, only imported when actually used

    wandb.init(project=project, config=config, id=run_id, resume="allow" if run_id else None)
    return True


def get_wandb_run_id(enabled: bool) -> str | None:
    """The current run's id, to save into a checkpoint so a future --resume can pass it back to
    init_wandb's run_id param. None when wandb logging is disabled."""
    if not enabled:
        return None
    import wandb  # noqa: PLC0415

    return wandb.run.id if wandb.run is not None else None


def log_step(enabled: bool, step: int, losses: dict[str, float], lr: float) -> None:
    """Logs one step's loss terms (already scalar -- see train_codec.py's grad-accum averaging)
    plus the current learning rate."""
    if not enabled:
        return
    import wandb  # noqa: PLC0415

    wandb.log({**losses, "lr": lr}, step=step)


def log_preview(
    enabled: bool,
    step: int,
    original: Tensor,
    reconstructed: Tensor,
    fps: int = 20,
    output_dir: str | Path | None = None,
) -> Path | None:
    """Saves and optionally logs an original/reconstruction preview for the first sample.

    The MP4 is encoded before it is handed to W&B. Passing a filename to ``wandb.Video`` avoids
    W&B's optional moviepy dependency, and retaining the file also makes early validation easy.
    A W&B/media failure is intentionally non-fatal: observability must not stop training.

    original: (b, t, 3, h, w) in [0, 1]. reconstructed: (b, t, 3, h, w) in [-1, 1] (tanh output).
    """
    from mini_mira.codec.loss import denormalize_for_dino

    original_display = original[0].clamp(0, 1)
    recon_display = denormalize_for_dino(reconstructed[0]).clamp(0, 1)
    comparison = torch.cat([original_display, recon_display], dim=3)  # side by side along width
    video = (comparison.detach().cpu() * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()

    video_path = None
    if output_dir is not None:
        try:
            import imageio.v3 as iio  # noqa: PLC0415

            preview_dir = Path(output_dir)
            preview_dir.mkdir(parents=True, exist_ok=True)
            video_path = preview_dir / f"step_{step:06d}.mp4"
            iio.imwrite(video_path, video, fps=fps, codec="libx264", pixelformat="yuv420p")
            print(f"Saved preview to {video_path}")
        except Exception as exc:  # media encoding must never discard a completed optimizer step
            video_path = None
            warnings.warn(f"Preview encoding failed at step {step}: {exc}", stacklevel=2)

    if enabled:
        try:
            import wandb  # noqa: PLC0415

            media: dict[str, Any] = {
                "preview": wandb.Image(video[0]),
            }
            if video_path is not None:
                media["preview_video"] = wandb.Video(str(video_path), fps=fps, format="mp4")
            wandb.log(media, step=step)
        except Exception as exc:  # logging must never discard a completed optimizer step
            warnings.warn(f"W&B preview logging failed at step {step}: {exc}", stacklevel=2)

    return video_path
