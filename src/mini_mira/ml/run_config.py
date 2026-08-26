"""Run-level (hyperparameter) config dataclasses, loaded via ml.config_loading.load_run_config --
a second axis alongside PipelineConfig's architecture axis. See notes/deviations.md 2.1.

Field names/defaults mirror each script's own argparse flags exactly (lossless extraction, not a
redesign). Paths, --resume, and other per-invocation flags stay CLI-only, not here.
"""

from dataclasses import dataclass


@dataclass
class WorldModelRunConfig:
    """Mirrors scripts/train_world_model.py's hyperparameter flags."""

    precision: str = "bf16"
    height: int = 288
    width: int = 512
    frames: int = 40
    target_fps: int = 20
    batch_size: int = 4
    grad_accum_steps: int = 2

    steps: int = 2000
    lr: float = 1e-4
    lr_warmup_steps: int | None = None
    lr_decay_steps: int = 0
    lr_min: float = 1e-6

    psd_weight: float = 0.0
    psd_loss_prob: float = 0.0
    scheduled_sampling_prob: float = 0.0

    eval_batch_size: int | None = None
    val_every: int | None = None
    val_n_samples: int = 64
    drift_eval_every: int | None = None
    drift_eval_n_samples: int = 8
    drift_eval_context_latents: int = 6
    drift_eval_diffusion_steps: int = 4
    drift_eval_schedule: str = "linear"
    fdd_slice_frames: int = 7
    viz_n_samples: int = 2

    checkpoint_every: int | None = None
    console_log_every: int | None = None


@dataclass
class CodecRunConfig:
    """Mirrors scripts/train_codec.py's hyperparameter flags."""

    steps: int = 30
    lr: float = 1e-4
    height: int = 64
    width: int = 64
    frames: int = 4
    target_fps: int = 20
    batch_size: int = 4
    grad_accum_steps: int = 1

    activation_checkpointing: bool = False
    perceptual_chunk_size: int = 0
    perceptual_dino_model: str | None = None
    perceptual_dino_multilayer: bool = False

    lr_warmup_steps: int | None = None
    lr_decay_steps: int | None = None
    lr_min: float | None = None

    loss_mae_weight: float = 1.0
    log_activation_grad_norms: bool = False
    log_per_term_grad_norm: bool = False

    checkpoint_every: int = 100
    hf_backup_every: int | None = None
    preview_every: int = 100
    console_log_every: int = 10
    precision: str = "fp16-hybrid"
