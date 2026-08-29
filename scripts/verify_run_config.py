"""CPU-only checks for the run-config mechanism (ml.run_config, ml.config_loading's
load_run_config/apply_run_config) -- no GPU, no real training script invocation.
"""

import argparse
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from mini_mira.ml.config_loading import apply_run_config, load_run_config
from mini_mira.ml.run_config import CodecRunConfig, WorldModelRunConfig

# --- Check 1: unrecognized key raises TypeError ---
tmp = Path(tempfile.mkdtemp()) / "bad.yaml"
tmp.write_text(yaml.dump({"batch_size": 4, "not_a_real_field": 1}))
try:
    load_run_config(tmp, WorldModelRunConfig)
    raise AssertionError("expected TypeError on an unrecognized key")
except TypeError:
    pass
print("[PASS] load_run_config raises TypeError on an unrecognized key")

# --- Check 2: apply_run_config fills None, leaves explicit values untouched ---
tmp2 = Path(tempfile.mkdtemp()) / "good.yaml"
tmp2.write_text(yaml.dump({"batch_size": 8, "steps": 5000}))
run_config = load_run_config(tmp2, WorldModelRunConfig)
args = argparse.Namespace(batch_size=2, steps=None, precision=None)
apply_run_config(args, run_config)
assert args.batch_size == 2, "explicit CLI value must not be overwritten"
assert args.steps == 5000, "None value must be filled from run_config"
assert args.precision == "bf16", "None value must be filled from run_config's own default"
print("[PASS] apply_run_config: explicit values kept, None values filled from run_config")

# --- Check 3: dataclass defaults match today's real argparse defaults, field by field ---
# The actual regression test that this stays a lossless extraction as the scripts evolve.
EXPECTED_WM_DEFAULTS = {
    "precision": "bf16", "height": 288, "width": 512, "frames": 40, "target_fps": 20,
    "batch_size": 4, "grad_accum_steps": 2, "steps": 2000, "lr": 1e-4, "lr_warmup_steps": None,
    "lr_decay_steps": 0, "lr_min": 1e-6, "psd_weight": 0.0, "psd_loss_prob": 0.0,
    "scheduled_sampling_prob": 0.0, "eval_batch_size": None, "val_every": None,
    "val_n_samples": 64, "drift_eval_every": None, "drift_eval_n_samples": 8,
    "drift_eval_context_latents": 6, "drift_eval_diffusion_steps": 4,
    "drift_eval_schedule": "linear", "fdd_slice_frames": 7, "viz_n_samples": 2,
    "checkpoint_every": None, "console_log_every": None, "num_workers": 0, "compile": False,
}
wm_defaults = {f.name: getattr(WorldModelRunConfig(), f.name) for f in fields(WorldModelRunConfig)}
assert wm_defaults == EXPECTED_WM_DEFAULTS, f"WorldModelRunConfig defaults drifted: {wm_defaults}"
print(f"[PASS] WorldModelRunConfig: all {len(wm_defaults)} defaults match train_world_model.py's own")

EXPECTED_CODEC_DEFAULTS = {
    "steps": 30, "lr": 1e-4, "height": 64, "width": 64, "frames": 4, "target_fps": 20,
    "batch_size": 4, "grad_accum_steps": 1, "activation_checkpointing": False, "compile": False,
    "perceptual_chunk_size": 0, "auto_weight_every": 1,
    "perceptual_dino_model": None, "perceptual_dino_multilayer": False,
    "lr_warmup_steps": None, "lr_decay_steps": None, "lr_min": None, "loss_mae_weight": 1.0,
    "reconstruction_loss": "l1", "perceptual_warmup_steps": 0,
    "log_activation_grad_norms": False, "log_per_term_grad_norm": False, "checkpoint_every": 100,
    "hf_backup_every": None, "preview_every": 100, "console_log_every": 10,
    "precision": "fp16-hybrid",
}
codec_defaults = {f.name: getattr(CodecRunConfig(), f.name) for f in fields(CodecRunConfig)}
assert codec_defaults == EXPECTED_CODEC_DEFAULTS, f"CodecRunConfig defaults drifted: {codec_defaults}"
print(f"[PASS] CodecRunConfig: all {len(codec_defaults)} defaults match train_codec.py's own")

print("\nAll run-config checks passed.")
