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


def log_step(enabled: bool, step: int, losses: dict[str, Tensor], lr: float) -> None:
    """Logs one step's loss terms (including loss_total) plus the current learning rate."""
    if not enabled:
        return
    import wandb  # noqa: PLC0415

    wandb.log({**{k: v.item() for k, v in losses.items()}, "lr": lr}, step=step)


def log_preview(enabled: bool, step: int, original: Tensor, reconstructed: Tensor) -> None:
    """Logs one side-by-side preview image: original vs. reconstructed, first frame of the batch.

    original: (b, t, 3, h, w) in [0, 1] (raw video). reconstructed: (b, t, 3, h, w) in [-1, 1]
    (the decoder's raw tanh output) -- same denormalize-before-display convention as
    reconstruct.py's save_comparison_grid, not a copy of it (that one builds a full multi-frame
    grid to a PNG file; this logs a single lightweight frame to wandb every checkpoint interval).
    """
    if not enabled:
        return
    import wandb  # noqa: PLC0415

    from mini_mira.codec.loss import denormalize_for_dino

    recon_display = denormalize_for_dino(reconstructed[0, 0]).clamp(0, 1)
    grid = torch.cat([original[0, 0].clamp(0, 1), recon_display], dim=2)  # side by side (width)
    wandb.log({"preview": wandb.Image(grid.permute(1, 2, 0).detach().cpu().numpy())}, step=step)
