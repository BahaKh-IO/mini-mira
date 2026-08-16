"""Minimal save/resume for world-model training: world_model + action_encoder + bos + optimizer +
scheduler + step count, saved together as one file. Structural sibling of
mini_mira.codec.checkpoint (same atomic-write pattern, same "simple pair of functions, not a port
of mira's CheckpointManager" reasoning -- see that module's docstring), just with this training
run's own set of components: the frozen codec (dino/bottleneck/decoder) is never saved here since
nothing about it changes during world-model training.
"""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    step: int,
    world_model: torch.nn.Module,
    action_encoder: torch.nn.Module,
    bos: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_scaler: torch.amp.GradScaler | None = None,
    wandb_run_id: str | None = None,
) -> None:
    """Atomic write: save to a temp file, then rename -- a crash mid-save never leaves `path`
    pointing at a half-written file.

    grad_scaler is optional (fp16 runs only, matching train_codec.py's precision setup -- see
    notes/gpu_box_notes_backup.md for why this project uses fp16+GradScaler rather than bf16 for
    trainable ops) -- without it, --resume restarts the scaler at its default init_scale,
    discarding whatever it had actually converged to. Harmless, just wastes a few steps
    re-calibrating after each resume.

    wandb_run_id: the current wandb run's id (mini_mira.codec.logging_utils.get_wandb_run_id), so
    a future --resume can pass it back into init_wandb and continue the same run instead of
    opening a new, disconnected one."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "world_model": world_model.state_dict(),
            "action_encoder": action_encoder.state_dict(),
            "bos": bos.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "grad_scaler": grad_scaler.state_dict() if grad_scaler is not None else None,
            "wandb_run_id": wandb_run_id,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    world_model: torch.nn.Module,
    action_encoder: torch.nn.Module,
    bos: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, str | None]:
    """Loads every component in place. Returns (step to resume from (saved_step + 1), the saved
    wandb run id or None).

    Checkpoints saved before grad_scaler/wandb_run_id support existed simply have no matching
    key -- .get(...) treats that the same as grad_scaler=None / wandb_run_id=None was passed at
    save time, so older checkpoints keep loading fine, just without restoring that state."""
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    world_model.load_state_dict(ckpt["world_model"])
    action_encoder.load_state_dict(ckpt["action_encoder"])
    with torch.no_grad():
        bos.copy_(ckpt["bos"].to(bos.device))
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if grad_scaler is not None and ckpt.get("grad_scaler") is not None:
        grad_scaler.load_state_dict(ckpt["grad_scaler"])
    return ckpt["step"] + 1, ckpt.get("wandb_run_id")
