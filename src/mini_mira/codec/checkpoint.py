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
    grad_scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Atomic write: save to a temp file, then rename -- a crash mid-save never leaves `path`
    pointing at a half-written file (the standard reason CheckpointManager does the same thing).

    grad_scaler is optional (fp16 runs only) -- without it, --resume restarts the scaler at its
    default init_scale, discarding whatever it had actually converged to. Harmless, just wastes
    a few steps re-calibrating after each resume; see notes/deviations.md."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "bottleneck": bottleneck.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "grad_scaler": grad_scaler.state_dict() if grad_scaler is not None else None,
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
    grad_scaler: torch.amp.GradScaler | None = None,
) -> int:
    """Loads every component in place. Returns the step to resume from (saved_step + 1).

    Checkpoints saved before grad_scaler support existed simply have no "grad_scaler" key --
    .get(...) treats that the same as grad_scaler=None was passed at save time, so older
    checkpoints keep loading fine, just without restoring scaler state."""
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    bottleneck.load_state_dict(ckpt["bottleneck"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if grad_scaler is not None and ckpt.get("grad_scaler") is not None:
        grad_scaler.load_state_dict(ckpt["grad_scaler"])
    return ckpt["step"] + 1
