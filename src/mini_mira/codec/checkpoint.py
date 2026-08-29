"""Minimal save/resume for codec training: bottleneck + decoder + optimizer + scheduler + step
count, saved together as one file. Not a port of mira's CheckpointManager (built for a
multi-day, distributed recipe with retention tiers and EMA) -- this is a single short run with
two known modules, so a simpler pair of functions covers it.
"""

from pathlib import Path
from typing import Any

import torch


def _unwrap_compiled(module: torch.nn.Module) -> torch.nn.Module:
    """torch.compile() wraps a module in OptimizedModule, whose state_dict() keys carry a real
    "_orig_mod." prefix (confirmed empirically: real cross-load test, torch==2.8.0) -- so a
    checkpoint saved under --compile fails to load into a plain module (evaluate_codec_vjepa.py,
    or the same script without --compile), and vice versa. Unwrapping to the real underlying
    module here, on both save and load, keeps every checkpoint in one plain, portable format
    regardless of whether --compile produced or is loading it. A no-op (returns module unchanged)
    whenever compile was never used -- covers train_codec.py, which has no --compile at all.
    """
    return getattr(module, "_orig_mod", module)


def save_checkpoint(
    path: str | Path,
    step: int,
    bottleneck: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_scaler: torch.amp.GradScaler | None = None,
    wandb_run_id: str | None = None,
) -> None:
    """Atomic write: save to a temp file, then rename -- a crash mid-save never leaves `path`
    pointing at a half-written file (the standard reason CheckpointManager does the same thing).

    grad_scaler is optional (fp16 runs only) -- without it, --resume restarts the scaler at its
    default init_scale, discarding whatever it had actually converged to. Harmless, just wastes
    a few steps re-calibrating after each resume; see notes/deviations.md.

    wandb_run_id: the current wandb run's id (mini_mira.codec.logging_utils.get_wandb_run_id), so
    a future --resume can pass it back into init_wandb and continue the same run instead of
    opening a new, disconnected one."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "bottleneck": _unwrap_compiled(bottleneck).state_dict(),
            "decoder": _unwrap_compiled(decoder).state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "grad_scaler": grad_scaler.state_dict() if grad_scaler is not None else None,
            "wandb_run_id": wandb_run_id,
        },
        tmp,
    )
    tmp.replace(path)


def _load_decoder_flexible(
    decoder: torch.nn.Module, state_dict: dict[str, Any], new_submodule_prefix: str | None
) -> None:
    """decoder.load_state_dict, tolerant of missing keys IF every one of them belongs to
    new_submodule_prefix -- a submodule (e.g. "refinement_head.") added to the architecture after
    this checkpoint was saved, with no counterpart in the saved file. Any OTHER missing key, or
    any unexpected key at all, still raises: this is meant to let one specific, known kind of
    architecture growth resume cleanly, not to blanket-swallow every mismatch.

    new_submodule_prefix=None (the default) skips all of this and loads strictly, unchanged from
    before this function existed -- the common case, and every existing caller's behavior."""
    if new_submodule_prefix is None:
        decoder.load_state_dict(state_dict)
        return
    result = decoder.load_state_dict(state_dict, strict=False)
    unexpected_missing = [k for k in result.missing_keys if not k.startswith(new_submodule_prefix)]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Decoder checkpoint mismatch beyond the expected new {new_submodule_prefix!r} "
            f"submodule -- missing: {unexpected_missing}, unexpected: {result.unexpected_keys}"
        )
    if result.missing_keys:
        print(f"load_checkpoint: decoder missing {new_submodule_prefix!r} keys (new submodule, "
              f"expected, starts fresh): {result.missing_keys}")


def load_checkpoint(
    path: str | Path,
    bottleneck: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_scaler: torch.amp.GradScaler | None = None,
    reset_optimizer_state: bool = False,
    decoder_new_submodule_prefix: str | None = None,
) -> tuple[int, str | None]:
    """Loads every component in place. Returns (step to resume from (saved_step + 1), the saved
    wandb run id or None).

    Checkpoints saved before grad_scaler/wandb_run_id support existed simply have no matching
    key -- .get(...) treats that the same as grad_scaler=None / wandb_run_id=None was passed at
    save time, so older checkpoints keep loading fine, just without restoring that state.

    reset_optimizer_state: skip loading the checkpoint's saved optimizer state entirely, so the
    caller's own (already-constructed) optimizer keeps whatever state it started with. Needed,
    not just preferred, whenever the caller's optimizer covers a different set of parameters than
    the checkpoint was saved with (e.g. decoder_new_submodule_prefix added new ones) --
    optimizer.load_state_dict raises on a param-count mismatch, so loading first and discarding
    after (the old pattern) doesn't work once the counts actually differ. The lr_scheduler is
    NOT affected by this flag: its own state (warmup/decay shape, current step) is independent of
    AdamW's momentum/variance and still loads either way, matching how --reset-optimizer-state and
    --reset-lr-schedule are already independent, separately-opted-into things at the CLI level.

    decoder_new_submodule_prefix: see _load_decoder_flexible. None (default) loads the decoder
    strictly, unchanged from before this parameter existed.
    """
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    _unwrap_compiled(bottleneck).load_state_dict(ckpt["bottleneck"])
    _load_decoder_flexible(_unwrap_compiled(decoder), ckpt["decoder"], decoder_new_submodule_prefix)
    if not reset_optimizer_state:
        optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if grad_scaler is not None and ckpt.get("grad_scaler") is not None:
        grad_scaler.load_state_dict(ckpt["grad_scaler"])
    return ckpt["step"] + 1, ckpt.get("wandb_run_id")
