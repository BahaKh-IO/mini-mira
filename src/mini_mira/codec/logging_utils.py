"""Optional wandb logging for train_codec.py.

Every function here is a no-op when disabled, and `wandb` is imported lazily inside each function
(never at module import time) -- so a run without --wandb-project never needs wandb installed at
all, matching how mira's own scripts scope the `wandb`/`lpips` imports to only where they're used.
"""

from typing import Any

import torch
from torch import Tensor


def init_wandb(project: str | None, config: dict[str, Any]) -> bool:
    """Starts a wandb run if `project` is set. Returns whether logging is enabled."""
    if not project:
        return False
    import wandb  # noqa: PLC0415 -- optional dep, only imported when actually used

    wandb.init(project=project, config=config)
    return True


def log_step(enabled: bool, step: int, losses: dict[str, float], lr: float) -> None:
    """Logs one step's loss terms (already scalar -- see train_codec.py's grad-accum averaging)
    plus the current learning rate."""
    if not enabled:
        return
    import wandb  # noqa: PLC0415

    wandb.log({**losses, "lr": lr}, step=step)


def log_preview(
    enabled: bool, step: int, original: Tensor, reconstructed: Tensor, fps: int = 20
) -> None:
    """Logs side-by-side original/reconstruction image and video previews for the first sample.
    original: (b, t, 3, h, w) in [0, 1]. reconstructed: (b, t, 3, h, w) in [-1, 1] (tanh output).
    """
    if not enabled:
        return
    import wandb  # noqa: PLC0415

    from mini_mira.codec.loss import denormalize_for_dino

    original_display = original[0].clamp(0, 1)
    recon_display = denormalize_for_dino(reconstructed[0]).clamp(0, 1)
    comparison = torch.cat([original_display, recon_display], dim=3)  # side by side along width
    video = (comparison.detach().cpu() * 255).round().to(torch.uint8).numpy()
    wandb.log(
        {
            "preview": wandb.Image(comparison[0].permute(1, 2, 0).detach().cpu().numpy()),
            "preview_video": wandb.Video(video, fps=fps, format="mp4"),
        },
        step=step,
    )
