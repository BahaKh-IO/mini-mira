# mini_mira

A from-scratch reimplementation of the core architecture behind **MIRA** — an action-conditioned
latent world model for Rocket League, built on a representation-autoencoder (RAEv2) codec and a
flow-matching diffusion transformer. Reimplements [`mira-wm/mira`](https://github.com/mira-wm/mira)
(Apache 2.0) end to end at a smaller scale, verified against the real source rather than assumed.

**Current status**: the codec is frozen at step 3,999; the world model has completed two real
training runs, through step 5,500, on top of it. A second, parallel track has also started —
swapping the frozen DINOv3 backbone for V-JEPA 2.1, as a controlled benchmark against the DINO
track rather than a replacement for it. See [Status](#status) below.

## What this is

One full forward pass — **codec encode → latent diffusion → codec decode** — matching the real
MIRA architecture. Built by tracing the real repo, verifying shape contracts, and checking that
each mechanism actually does what it claims (see [Verification](#verification)). Simplifications
are disclosed decisions, not gaps found later — see [Scope](#scope).

## Architecture

```
DINO-shaped features  (B, T, dino_dim, H, W)
        │
        ▼
   StridedConvBottleneckConfig / MyBottleneck      strided-conv projection to a latent grid
        │
        ▼
   z  (B, T', latent_dim, H', W')
        │
        ▼
   LatentWorldModelConfig / DiffusionTransformer   AdaLN-conditioned space-time diffusion
        │        (multi-step Euler integration,     transformer; predicts flow-matching velocity
        │         flow matching: noise → data)
        ▼
   ViTDecoderConfig / ViTVideoDecoder               space-time ViT decoder, RoPE-based attention
        │
        ▼
   video  (B, T, 3, H_out, W_out)
```

Both the decoder and world model factorize attention into **spatial** (bidirectional, within a
frame) and **temporal** (causal, across frames) sublayers, with RoPE for position instead of
learned embeddings. The world model additionally conditions every block, via AdaLN, on the
diffusion timestep `tau` and the player's key-press actions, plus a separate additive conditioning
on the previous frame's clean latent content (`clean_past`).

### Scale

Target ~300M params; `configs/scaled_300m.yaml` is the real attempt, measured from an instantiated
model:

| Component | Parameters |
|---|---|
| Bottleneck | 196,640 |
| Decoder (width=1024, depth=6, 16 heads) | 104,000,000 |
| World model (hidden_dim=1024, depth=8, 16 heads) | 188,110,880 |
| Action encoder (9 keys) | 149,792 |
| `bos` | 32 |
| **Total** | **292,457,344** (−2.51% vs. the 300M target) |
| `DinoModel` (frozen, real pretrained weights, not in the total) | 85,669,632 |

Tuned so the world_model/decoder parameter ratio (1.81) roughly matches real mira's own shipped
ratio (1.82), not picked by feel. Fast verification scripts instead use `configs/small.yaml`
(~11.3M params, mirrors class defaults) for millisecond runs.

## Status

**Codec**: step 3,999 of training, full resolution (288×512, 40-frame clips), real Rocket League
data, real mira's cosine LR schedule and three-term loss (L1 + LPIPS + DINO latent-consistency).
Real held-out evaluation on 20 clips it never trained on (`scripts/evaluate_codec.py`):
`PSNR 19.56dB, SSIM 0.552, LPIPS 0.486` — reads as "poor" by standard external benchmarks, though
an earlier read based on training curves and preview videos alone (not these numbers) judged it
"genuinely decent." Real mira's own recipe for this component runs 250,001 steps — we're at
roughly 1.6% of that, which is the most likely explanation for the quality gap.

![Decoder reconstruction preview](assets/decoder_preview.gif)

**World model**: `scripts/train_world_model.py` fully implements real mira's diagonal
flow-matching loss (+ optional PSD self-distillation), real action conditioning from real streamed
key-press data, checkpointing, and a full eval suite (drift metrics, Frechet DINO/Inception
Distance, PSNR, LPIPS, SSIM, rendered rollout videos). **First real, multi-hour training run
completed**: ~2,900 steps over ~11 hours on real data, real measured improvement (SSIM 0.48→0.65,
LPIPS 0.59→0.40, Frechet DINO/Inception Distance both dropped by more than half). Also surfaced a
real, concrete finding: quality degrades the deeper into a self-generated rollout it goes (e.g.
Frechet DINO Distance 2.6 right after context vs. 26.6 by 28 frames deep — confirmed visually too,
not just numerically) — the expected consequence of `clean_past` only ever being real during
training but partly self-generated during rollout. Two things added in response: a `timeout`-safe
`SIGTERM` handler (graceful checkpoint save + wandb sign-off instead of an abrupt kill), and
opt-in **scheduled sampling** (`--scheduled-sampling-prob`, default off) — occasionally trains on
a self-generated `clean_past` instead of always-real, directly targeting that gap. Verified via
`verify_world_model_training.py`'s CPU mechanism check, then run for real: a second training run,
resuming the first run's checkpoint with `--scheduled-sampling-prob 0.3`, reached **step 5,500**.
Real improvement across every headline metric versus the first run's own final numbers (SSIM
0.65→0.80, LPIPS 0.40→0.30, Frechet DINO Distance 10.3→3.86, Frechet Inception Distance
112.9→92.66, PSNR broke its earlier flat/noisy pattern to reach 20.17dB), and the rollout-depth
degradation itself narrowed from a roughly 10x shallow-to-deep blowup to roughly 2.6x, now
plateauing instead of continuing to climb — encouraging, though not an isolated before/after (more
training steps and a changed eval window happened in the same run). A real, separate obstacle
surfaced getting that run started: `--resume` replays every already-consumed training batch before
continuing (by design, so a resumed run doesn't silently re-see old data) — at ~2,900 steps' worth,
that meant ~11,600 batches to replay, a real measured 9-46 hour wait for zero new training. Fixed
by patching the one relevant checkpoint field (`dataloader_batches_consumed`) directly, nothing
else touched.
Confirmed working end-to-end on real GPU hardware before that, including multi-session `--resume`
— RNG state, dataloader position, and codec/latent-stats provenance all persist and were verified
in a real two-phase smoke test (fresh run → resume). Two real bugs were found and fixed this way
(a list-vs-tensor crash in drift-metric eval, a device-mismatch crash in rollout video rendering)
— the kind reading-only verification
can't catch. Real (full-scale) training config is confirmed and proven across two real runs —
`--batch-size 4 --grad-accum-steps 4` (effective batch 16), full resolution, `--precision bf16` —
each time-boxed rather than left running indefinitely, given the shared, rotating GPU access this
project runs on. One open architectural divergence from real mira remains (an extra, unconditioned
final `LayerNorm`) — low-risk, not urgent to fix.

**V-JEPA track**: a second, parallel pipeline, benchmarking V-JEPA 2.1 against the DINOv3 backbone
above — not a replacement, and the DINO-track checkpoints stay untouched as the control. So far:
`VjepaModel` (`src/mini_mira/codec/vjepa.py`), a frozen V-JEPA 2.1 ViT-B encoder shaped as a
drop-in sibling to `DinoModel` (same `dino_forward`/`.dino_dim` contract), verified via
`scripts/verify_vjepa.py`. Real facts confirmed live before writing it: V-JEPA 2.1 is ungated
(unlike DINOv3), `embed_dim=768` (matches DINOv3-B exactly), and it halves the frame count
internally (`tubelet_size=2` — `dino_forward` here returns `t // 2` frames, a genuine, documented
deviation from `DinoModel`'s own contract). Multi-layer feature aggregation is supported too,
matching `DinoModel`'s own interface and layer indices.

**Next goal**: two design decisions for this track are now made, deliberately kept simple rather
than general — this is a benchmark run by two interns, not infrastructure meant to outlive it.
**Scripts**: no runtime backbone-selection flag; V-JEPA gets its own full copies of the training
scripts (`train_codec_vjepa.py`, `train_world_model_vjepa.py`, mirroring `train_codec.py`/
`train_world_model.py`) rather than a shared core loop — the DINO scripts stay untouched, and each
track's logic is fully readable on its own, at the accepted cost that a future bug fix has to be
applied to both copies by hand. **Checkpoints**: separate directories (`checkpoints_vjepa/`,
`checkpoints_wm_vjepa/`) and a `_vjepa`-tagged filename, no embedded backbone-metadata field —
DINO's own checkpoint dirs and defaults are untouched. **Config preset**: done —
`configs/scaled_300m_vjepa.yaml`, identical to `scaled_300m.yaml` except `bottleneck.temporal_stride`
2→1 (V-JEPA's own tubelet already halves frame count before the bottleneck sees it). Turned out no
library code needs to change for any of this: `MyBottleneck` is already backbone-agnostic (works
from `dino_dim`/`temporal_stride` alone, holds no reference to which encoder produced its input),
and `LatentWorldModel` already accepts an injected `dino` module — both confirmed by reading the
code, not assumed. What's still ahead: writing the two new training scripts themselves (instantiate
`VjepaModel` where the DINO ones instantiate `DinoModel`, point at the new checkpoint dirs/config),
then a full codec retrain from scratch under V-JEPA's feature space (the existing checkpoint can't
be reused — the whole representation changes), then a world-model retrain on top. Three real
methodology questions still need a decision before any final numbers count as comparable: whether
both tracks get scored by the same fixed judge rather than each by its own backbone, what step
budget each track gets, and whether hyperparameters stay identical across both.

Full bug-by-bug history and the evidence trail behind every claim above: `notes/deviations.md` and
`notes/session_handoff.md` (both git-ignored, local only).

## Project layout

| File | Contents |
|---|---|
| `src/mini_mira/codec/bottleneck.py` | Encoder-side strided-conv projection into the latent |
| `src/mini_mira/codec/decoder.py` | Space-time ViT decoder |
| `src/mini_mira/world_model/diffusion_transformer.py` | AdaLN-conditioned diffusion transformer |
| `src/mini_mira/ml/blocks.py` | Shared attention/MLP/AdaLN blocks (decoder + world model) |
| `src/mini_mira/ml/init.py` | Mira-matching weight initialization |
| `src/mini_mira/ml/rope.py` | Shared RoPE implementation (temporal + spatial) |
| `src/mini_mira/world_model/timestep_encoder.py` | Sinusoidal embedding of diffusion timestep `tau` |
| `src/mini_mira/world_model/action_encoder.py` | Encodes key-press actions into conditioning vectors |
| `src/mini_mira/pipeline.py` | Architecture-demo pipeline, no real checkpoint loading — see `LatentWorldModel` for the real trainer |
| `src/mini_mira/world_model/latent_world_model.py` | Real training wrapper: frozen codec + trainable world model, real flow-matching + PSD loss |
| `src/mini_mira/world_model/checkpoint.py` | Save/resume for world-model training |
| `src/mini_mira/world_model/eval_metrics.py` | Cheap, always-on drift-metric eval |
| `src/mini_mira/world_model/full_eval_metrics.py` | Frechet DINO/Inception Distance, PSNR, LPIPS, SSIM |
| `src/mini_mira/world_model/rollout_visualization.py` | Renders rollout videos with an action HUD overlay |
| `src/mini_mira/codec/dino.py` | Real, frozen DINOv3 backbone |
| `src/mini_mira/codec/vjepa.py` | Real, frozen V-JEPA 2.1 backbone — `DinoModel`-shaped sibling |
| `src/mini_mira/codec/loss.py` | Codec training loss: L1 + LPIPS + DINO latent-consistency |
| `src/mini_mira/codec/video_prep.py` | Resizes/pads real clips to canonical shape |
| `src/mini_mira/codec/checkpoint.py` | Save/resume for codec training |
| `src/mini_mira/codec/logging_utils.py` | Optional wandb logging for `train_codec.py` |
| `scripts/test_shapes.py` | Shape-correctness checks |
| `scripts/verify_rope.py` | Behavioral checks for RoPE |
| `scripts/verify_conditioning.py` | Behavioral checks for AdaLN/clean-past/actions |
| `scripts/verify_dino.py` | Behavioral checks for the real DINOv3 backbone (needs gated weights) |
| `scripts/verify_vjepa.py` | Behavioral checks for the real V-JEPA 2.1 backbone (ungated, no weights needed to set up first) |
| `scripts/test_dino.py` | Raw DINOv3 sanity check, bypassing `DinoModel` |
| `src/mini_mira/ml/config_loading.py` | Builds a `PipelineConfig` (architecture) or a run-config (hyperparameters) from YAML |
| `src/mini_mira/ml/run_config.py` | `WorldModelRunConfig`/`CodecRunConfig` — the hyperparameter axis, loaded via `--run-config` |
| `configs/small.yaml` | Fast-test preset, mirrors class defaults |
| `configs/scaled_300m.yaml` | ~300M-param target preset |
| `configs/runs/` | Real `--run-config` examples (hyperparameters — batch size, steps, eval cadence, ...) |
| `scripts/verify_codec_training.py` | Mechanism proof the codec trains (synthetic data, no GPU needed) |
| `scripts/download_shards.py` | Downloads real Rocket League shards from `kyutai/rocket-science` |
| `scripts/train_codec.py` | Real GPU codec training |
| `scripts/reconstruct.py` | Mechanism smoke test: runs a video through the codec (random-init weights) |
| `scripts/evaluate_codec.py` | Real quantitative eval of a trained codec checkpoint on held-out data |
| `scripts/compute_latent_stats.py` | One-shot latent mean/std computation, feeds `train_world_model.py` |
| `scripts/train_world_model.py` | Real GPU world-model training |
| `scripts/verify_world_model_training.py` | CPU mechanism proof for `train_world_model.py` |
| `scripts/verify_full_eval_metrics.py` | CPU mechanism proof for the full eval suite |
| `scripts/verify_run_config.py` | CPU mechanism proof for the `--run-config` system |

## Scope

**Implemented**: strided-conv bottleneck + ViT space-time decoder matching the real codec's shape
contract; RoPE (temporal + axial spatial); QK-norm and mira-matching weight init; AdaLN
conditioning on `tau`, clean-past, and actions; flow-matching sampling; the real, frozen DINOv3
backbone with real pretrained weights; **real codec training** on real data with the real loss,
adaptive loss balancing, and checkpoint save/resume; **real world-model training mechanism** with
real flow-matching + PSD loss, real action-conditioned data, a full eval suite, and opt-in
scheduled sampling for rollout-depth robustness.

**Deliberately simplified / not yet implemented** (disclosed decisions, not gaps found later):
- Actions are keyboard keys only, no mouse — matches the real released data, which has no real
  mouse signal either. Also simplified vs. mira's `ActionEncoder`: no dropout, mean-pooling instead
  of a learned temporal pool, plain `Linear` instead of mira's per-key dimension split.
- `clean_past` is real encoded input by default (never the model's own previous output) — matches
  mira's own default too. `--scheduled-sampling-prob` opts into training on a self-generated
  estimate instead, some fraction of the time; off unless explicitly set.
- No streaming inference / KV-cache — every diffusion step recomputes the whole sequence.
- No grouped-query attention — always as many KV heads as query heads.
- One shared implementation where the real repo has two separate ones (codec vs. world model) for
  identical logic — consolidated into `blocks.py`/`rope.py` instead.

Full audit trail for every intentional or since-corrected difference from real mira:
`notes/deviations.md`.

## Verification

Shape correctness alone doesn't prove a mechanism works, so each is checked behaviorally too:

- **`scripts/test_shapes.py`** — every stage's output shape against the real codec's config.
- **`scripts/verify_rope.py`** — RoPE causality and position sensitivity.
- **`scripts/verify_conditioning.py`** — `tau`/`clean_past`/action sensitivity and determinism,
  plus end-to-end regression checks that output actually depends on input and on actions.
- **`scripts/verify_dino.py`** — the real DINOv3 backbone is frozen, correctly shaped, handles
  non-multiple-of-16 resolutions, and produces non-degenerate output.
- **`scripts/verify_codec_training.py`** — overfitting one synthetic video with real optimizer
  steps must substantially reduce the loss (catches dead gradients, detached graphs, wrong losses).

## Getting started

Requirements: Python ≥ 3.10, plus `requirements.txt`. `wandb`/`huggingface_hub` are optional,
lazily-imported (`--wandb-project`/`--hf-backup-repo` only); `torchcodec` is needed only for real
streamed data (`--index-path`). No packaging config — scripts add `src/` to `sys.path` directly.

```
pip install -r requirements.txt
python scripts/test_shapes.py
python scripts/verify_rope.py
python scripts/verify_conditioning.py
python scripts/verify_codec_training.py
```

The first run of anything using `CodecLoss` downloads pretrained VGG16 weights for LPIPS (~528MB,
one-time). `scripts/verify_dino.py` additionally needs real, gated DINOv3 weights on disk (request
access at [ai.meta.com/resources/models-and-libraries/dinov3-downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)),
pointed to via `RS_DINO_WEIGHTS_DIR`.

**Real training** needs a CUDA GPU and real data:

```
python scripts/download_shards.py --shards 50   # prints the local index path to pass below
python scripts/train_codec.py --config configs/scaled_300m.yaml \
  --index-path <path printed above> --require-pretrained-dino \
  --height 288 --width 512 --frames 40 --batch-size 4 --grad-accum-steps 8 \
  --activation-checkpointing --steps 1600
```

```
python scripts/compute_latent_stats.py --codec-checkpoint <checkpoint.pth> --index-path <path> \
  --output latent_stats.json
python scripts/train_world_model.py --config configs/scaled_300m.yaml \
  --codec-checkpoint <checkpoint.pth> --latent-stats latent_stats.json \
  --index-path <train data> --test-index-path <held-out data> --require-pretrained-dino
```

`--help` on either script lists the full flag set (LR schedule, precision, PSD weights, eval
cadence, checkpoint/resume, wandb/HF Hub backup) — or skip retyping it every launch with
`--run-config <path>` (see `configs/runs/` for real examples): any flag passed explicitly still
overrides the file, and omitting `--run-config` entirely reproduces the exact defaults above.

**Resume gotchas** (each has bitten this project for real at least once — see
`notes/session_handoff.md` for the full writeup of each):
- The resumed step count is `checkpoint's saved step + 1`, not the saved step itself.
- `--resume` alone continues the checkpoint's original LR curve; add `--reset-lr-schedule` to start
  a fresh warmup/decay shaped for a new phase, and pass this run's own `--lr`/`--lr-min`/
  `--lr-warmup-steps`/`--lr-decay-steps` explicitly either way.
- **Continuing the same phase across multiple sessions** needs `--lr-warmup-steps`/
  `--lr-decay-steps` passed identically on every session, sized to the *whole* multi-session arc —
  the script recomputes them from `--steps` unconditionally, so omitting them silently corrupts the
  curve instead of erroring.
- `train_codec.py`: resuming under a *different* `--precision` than the checkpoint was saved with
  crashes (a known, unfixed gap) — keep `--precision` consistent across a checkpoint's resumes.

## Configs

Named presets for `PipelineConfig` live in `configs/` at the repo root (data, not package code) —
one YAML file mirrors `PipelineConfig`'s whole nested shape, no Hydra/config-group system.

| File | Purpose |
|---|---|
| `configs/scaled_300m.yaml` | The intended architecture scale (~300M params) |
| `configs/scaled_300m_vjepa.yaml` | Same, `temporal_stride` 2→1 for the V-JEPA track |
| `configs/small.yaml` | Mirrors class defaults — what fast verification scripts use |

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mini_mira.ml.config_loading import load_pipeline_config
from mini_mira.pipeline import MyPipeline

config = load_pipeline_config("configs/scaled_300m.yaml")
pipeline = MyPipeline(config)
```

## Relationship to mira and attribution

Built by tracing the official [`mira-wm/mira`](https://github.com/mira-wm/mira) release (Apache
License 2.0). Class/config names match the real repository wherever there's a genuine one-to-one
correspondence (`ViTVideoDecoder`, `SelfAttention`, `DiffusionTransformer`); `MyBottleneck` and
`MyPipeline` keep their own names since they have no direct equivalent in the real repo. A few
small, self-contained pieces (`AdaptiveLayerNorm`, the sinusoidal timestep embedding, the RoPE
frequency computation) are adapted directly from the original source under its license terms.
