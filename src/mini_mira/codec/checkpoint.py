"""Minimal save/resume for codec training: bottleneck + decoder + optimizer + scheduler + step
count, saved together as one file. Not a port of mira's CheckpointManager (built for a
multi-day, distributed recipe with retention tiers and EMA) -- this is a single short run with
two known modules, so a simpler pair of functions covers it.
"""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    step: int,
    bottleneck: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    """Atomic write: save to a temp file, then rename -- a crash mid-save never leaves `path`
    pointing at a half-written file (the standard reason CheckpointManager does the same thing)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "bottleneck": bottleneck.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    bottleneck: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    """Loads every component in place. Returns the step to resume from (saved_step + 1)."""
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    bottleneck.load_state_dict(ckpt["bottleneck"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    return ckpt["step"] + 1
