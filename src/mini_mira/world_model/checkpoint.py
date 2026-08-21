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
    codec_checkpoint: str | Path | None = None,
    latent_mean: float | None = None,
    latent_std: float | None = None,
    dataloader_batches_consumed: int = 0,
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
    opening a new, disconnected one.

    codec_checkpoint/latent_mean/latent_std: which frozen codec + normalization stats this run was
    trained against. Recorded for load_checkpoint to compare against a future --resume's own
    --codec-checkpoint/--latent-stats -- neither is restored into the model (the codec is loaded
    fresh every launch, same as always), this is purely a mismatch check.

    dataloader_batches_consumed: total micro-batches pulled from the train loader across this
    whole run (including any earlier --resume's), so a future --resume can fast-forward the fresh
    loader past already-seen data instead of silently restarting from the beginning. Only correct
    if the loader is rebuilt with the same seed/num_workers every launch -- see train_world_model.py.

    Also snapshots torch's CPU + CUDA RNG state, so a future --resume continues the same random
    stream (noise/tau draws) instead of restarting it -- see train_world_model.py's unconditional
    torch.manual_seed(0), which this is restored after."""
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
            "codec_checkpoint": str(codec_checkpoint) if codec_checkpoint is not None else None,
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "dataloader_batches_consumed": dataloader_batches_consumed,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
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
) -> tuple[int, str | None, dict[str, Any]]:
    """Loads every component in place, including CPU/CUDA RNG state (so the resumed run continues
    the same random stream rather than restarting it). Returns (step to resume from (saved_step +
    1), the saved wandb run id or None, a dict with this run's own codec_checkpoint/latent_mean/
    latent_std -- caller compares these against its current --codec-checkpoint/--latent-stats and
    warns on mismatch -- plus dataloader_batches_consumed, for the caller to fast-forward a fresh
    loader past already-seen data. See train_world_model.py for both.

    Checkpoints saved before grad_scaler/wandb_run_id/provenance/rng support existed simply have
    no matching key -- .get(...) treats that the same as those args weren't passed at save time,
    so older checkpoints keep loading fine, just without restoring/checking that state."""
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    world_model.load_state_dict(ckpt["world_model"])
    action_encoder.load_state_dict(ckpt["action_encoder"])
    with torch.no_grad():
        bos.copy_(ckpt["bos"].to(bos.device))
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if grad_scaler is not None and ckpt.get("grad_scaler") is not None:
        grad_scaler.load_state_dict(ckpt["grad_scaler"])
    if ckpt.get("rng_state") is not None:
        torch.set_rng_state(ckpt["rng_state"])
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
    provenance = {
        "codec_checkpoint": ckpt.get("codec_checkpoint"),
        "latent_mean": ckpt.get("latent_mean"),
        "latent_std": ckpt.get("latent_std"),
        "dataloader_batches_consumed": ckpt.get("dataloader_batches_consumed", 0),
    }
    return ckpt["step"] + 1, ckpt.get("wandb_run_id"), provenance
