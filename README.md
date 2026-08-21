# mini_mira

A from-scratch reimplementation of the core architecture behind **MIRA** — an
action-conditioned latent world model for Rocket League built on a representation-autoencoder
(RAEv2) codec and a flow-matching diffusion transformer. This project reimplements the codec
and world-model architecture end to end and verifies it with shape and behavioral tests. The
codec is trained for real, on real Rocket League data (`scripts/train_codec.py`); the world
model has a real training mechanism too (`scripts/train_world_model.py`) — flow-matching
diffusion loss, optional PSD self-distillation, checkpointing, and a full eval suite — verified
by reading and CPU-mechanism tests, and now confirmed to execute on real hardware as well (its
first real-GPU smoke tests caught and fixed two genuine runtime bugs the reading-only
verification couldn't have found; see [Status](#status)).

## What this is

mini_mira implements one full forward pass — **codec encode → latent diffusion → codec
decode** — matching the architecture of the official MIRA code release
([`mira-wm/mira`](https://github.com/mira-wm/mira), Apache License 2.0) at a smaller scale.
It was built to develop a first-principles understanding of the real architecture: every
component here was implemented after tracing the equivalent code in the real repository,
verifying the shape contract, and checking that the specific mechanism it implements actually
does what it claims (see [Verification](#verification) below).

It deliberately does not attempt to match every capability of the real system. Where a
simplification was made, it was a disclosed decision, not an oversight — see
[Scope](#scope) below.

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

Both the decoder and the world model factorize attention into **spatial** (bidirectional,
within a frame) and **temporal** (causal, across frames) sublayers, followed by a SwiGLU MLP —
each a pre-norm residual block. Position is encoded with rotary embeddings (RoPE) rather than
learned positional embeddings; the world model additionally conditions every block, via adaptive
LayerNorm (AdaLN), on the diffusion timestep `tau` and on the player's key-press actions
(encoded by `ActionEncoder`) — and separately conditions on the clean latent content of the
previous frame (`clean_past`) via an additive projection.

### Scale

The target is ~300M parameters. `configs/scaled_300m.yaml` is the real attempt at
that target — this is the architecture's intended scale, measured from a real instantiated
`MyPipeline`, not estimated:

| Component | Parameters |
|---|---|
| Bottleneck | 196,640 |
| Decoder (width=1024, depth=6, 16 heads) | 104,000,000 |
| World model (hidden_dim=1024, depth=8, 16 heads) | 188,110,880 |
| Action encoder (9 keys) | 149,792 |
| `bos` | 32 |
| **Total** | **292,457,344** (−2.51% vs. the 300M target) |
| `DinoModel` (dinov3_vitb16, frozen, real pretrained weights, not part of the total above) | 85,669,632 |

The decoder/world-model split isn't arbitrary: it's tuned so the **world_model/decoder parameter
ratio (1.81) roughly matches real mira's own shipped-config ratio (1.82)** — measured the same
way from real mira's actual `ViTVideoDecoder`/`DiffusionTransformer` classes at their shipped
hyperparameters — rather than picking a split by feel. An earlier draft instead pinned the
decoder to match the frozen-DINOv3-plus-bottleneck "encoder" size exactly, but that made the
ratio 2.43 — notably *more* lopsided than real mira's own architecture. See `configs/scaled_300m.yaml`'s comments for the full numbers, and [Configs](#configs) below for how to load it.

**Every fast verification script in this project (see [Verification](#verification)) instead
uses a much smaller configuration** — `PipelineConfig()`'s own Python defaults, matching
`configs/small.yaml`, ~11.3M parameters — purely so shape and behavior checks run in
milliseconds without needing gated weights loaded. This mirrors real mira's own relationship
between its class-level config defaults and its actual shipped configs: the small numbers are a
convenience for fast iteration, not the intended final scale.

## Project layout

| File | Contents |
|---|---|
| `src/mini_mira/codec/bottleneck.py` | `StridedConvBottleneckConfig`, `MyBottleneck` — the encoder-side strided-conv projection into the latent |
| `src/mini_mira/codec/decoder.py` | `ViTDecoderConfig`, `ViTVideoDecoder`, `PatchUnembed` — the space-time ViT decoder |
| `src/mini_mira/world_model/diffusion_transformer.py` | `LatentWorldModelConfig`, `DiffusionTransformer` — the AdaLN-conditioned diffusion transformer |
| `src/mini_mira/ml/blocks.py` | Shared building blocks: `LayerScale`, `SwiGLU`, `SelfAttention` (with `QKRMSNorm`/`QKLayerNorm`, selected per-block via `BlockConfig.qk_norm`), `SpaceTimeBlock` (decoder), `AdaptiveLayerNorm`, `AdaSTBlock` (world model) |
| `src/mini_mira/ml/init.py` | `init_weights` — mira-matching initialization (`N(0, 0.02)` for Linear/Conv, unit weight/zero bias for norms, zeroed AdaLN conditioning projection), applied by the bottleneck and decoder |
| `src/mini_mira/ml/rope.py` | Rotary position embeddings — one shared implementation (temporal + axial spatial), used by both the decoder and the world model |
| `src/mini_mira/world_model/timestep_encoder.py` | `DiffusionTimeEmbedding` — sinusoidal embedding of the diffusion timestep `tau` |
| `src/mini_mira/world_model/action_encoder.py` | `ActionEncoder` — encodes raw multi-hot key-press actions into per-latent-frame conditioning vectors |
| `src/mini_mira/pipeline.py` | `PipelineConfig`, `MyPipeline` — wires bottleneck → multi-step diffusion loop → decoder into one forward pass; owns `bos`/`clean_past` and `ActionEncoder`. Architecture demonstration only — never loads a real trained codec checkpoint (see `LatentWorldModel` below for the class that actually trains) |
| `src/mini_mira/world_model/latent_world_model.py` | `LatentWorldModel` — the real training-time wrapper: a frozen, real-checkpoint-loaded codec (DINO + bottleneck + decoder) plus the trainable `DiffusionTransformer`/`ActionEncoder`/`bos`. Real mira's diagonal flow-matching loss + optional PSD self-distillation, ported 1:1; `rollout()` for autoregressive eval. See `notes/naming.md` for why this is a separate class from `MyPipeline` rather than a rename of it |
| `src/mini_mira/world_model/checkpoint.py` | `save_checkpoint`/`load_checkpoint` for world-model training — structural sibling of `codec/checkpoint.py` |
| `src/mini_mira/world_model/eval_metrics.py` | Cheap, always-on drift-metric eval tier: DINO cosine/L2 drift and latent drift between a real rollout and ground truth |
| `src/mini_mira/world_model/full_eval_metrics.py` | Heavier eval tier: sliced Frechet DINO/Inception Distance, PSNR, LPIPS, SSIM — shares one `rollout()` call with the drift tier rather than rolling out twice |
| `src/mini_mira/world_model/rollout_visualization.py` | Renders a small, fixed number of rollout samples as watchable video (keyboard-press HUD overlay + prediction-region border) for human inspection alongside the numeric eval metrics |
| `src/mini_mira/codec/dino.py` | `DinoModel` — the real, frozen DINOv3 backbone (ViT-S/B/L/16, single- or multi-layer); loads real pretrained weights, wired into `MyPipeline` behind `PipelineConfig.use_real_dino` (see Scope). `DEFAULT_DINO_LAYERS`/`DEFAULT_ENCODER_AGGREGATION_LAYERS` — mira's real per-variant layer-selection tables for the perceptual loss and the encoder's own feature aggregation, respectively (two different selections, for two different purposes) |
| `src/mini_mira/codec/loss.py` | `CodecLoss`, `CodecLossWeights`, `CodecOutputs` — the codec's real training loss (L1 + LPIPS + DINO latent-consistency, with `auto_weight` adaptive balancing and opt-in per-term activation-gradient instrumentation, matching mira's `CodecLoss`); the LPIPS term is `.reset()` after every `forward()` call — `torchmetrics`' wrapper otherwise accumulates internal state forever, an unbounded leak with no equivalent in mira's own stateless `lpips.LPIPS`; `normalize_video`/`denormalize_for_dino` — the `[0,1]`↔`[-1,1]` pixel-range conversions the codec and DINO each expect |
| `src/mini_mira/codec/video_prep.py` | `resize_to_canonical` — pads to the target aspect ratio then bilinear-resizes, matching mira's `VideoCodec.preprocess_batch`, so real clips of any native resolution land at the shape the model expects |
| `src/mini_mira/codec/checkpoint.py` | `save_checkpoint`/`load_checkpoint` — minimal save/resume for codec training (bottleneck + decoder + optimizer + scheduler + `GradScaler` + step count in one file), not a port of mira's distributed `CheckpointManager` |
| `src/mini_mira/codec/logging_utils.py` | Optional wandb logging for `train_codec.py` — `init_wandb`/`log_step`/`log_preview`; a no-op throughout if `--wandb-project` is omitted, so wandb is never a hard dependency |
| `scripts/test_shapes.py` | Shape-correctness checks for every stage of the pipeline |
| `scripts/verify_rope.py` | Behavioral checks for RoPE: causal masking and position sensitivity |
| `scripts/verify_conditioning.py` | Behavioral checks for AdaLN, clean-past, and actions: `tau`/`clean_past`/action sensitivity, determinism, and end-to-end pipeline checks |
| `scripts/verify_dino.py` | Behavioral checks for the real DINOv3 backbone: frozen parameters, correct output shape, patch-alignment resizing, non-degenerate features. Requires real gated weights on disk (see Getting started) |
| `scripts/test_dino.py` | Raw sanity check for the dinov3_vitb16 backbone bypassing `DinoModel` entirely — a diagnostic to tell apart "mini_mira's wrapper is broken" from "the underlying library/weights are broken" if `verify_dino.py` ever fails |
| `src/mini_mira/ml/config_loading.py` | `load_pipeline_config` — builds a `PipelineConfig` from a YAML preset under `configs/` |
| `configs/small.yaml` | Named preset mirroring today's dataclass defaults exactly (regression-safe baseline) |
| `configs/scaled_300m.yaml` | Named preset scaling `decoder`/`world_model` width and depth toward a ~300M parameter target |
| `scripts/verify_codec_training.py` | Mechanical proof the codec's training mechanism is wired correctly: overfits one fixed synthetic video on the real `CodecLoss` and asserts the total loss drops substantially. Small config, random-init DINOv3 — no gated weights, runs in seconds |
| `scripts/download_shards.py` | Downloads N real Rocket League match shards via mira's own `RocketScienceDataset.from_hub`, straight from the gated `kyutai/rocket-science` HuggingFace dataset — the same mechanism for a quick correctness smoke test (`--shards 3`, the default) and for pulling the real training set on the GPU box (larger `--shards`) |
| `scripts/train_codec.py` | Real GPU codec training: real streamed data (`--index-path`, from `download_shards.py`) or one fixed synthetic video (default, for fast local checks); mira's real `WarmupConstantCosineDecayLR` schedule; `--precision {fp16-hybrid,bf16}` (fp16-hybrid is the proven default; bf16 is opt-in, for hardware with native bf16 tensor cores); gradient accumulation and (optional — `--activation-checkpointing`) activation checkpointing for full-resolution batches; `auto_weight` on; always-on real diagnostics (`grad_norm_params_total`, correctly precision-unscaled; `weight_drift_l2_from_run_start` at checkpoint cadence) plus opt-in ones (`--log-per-term-grad-norm` for real per-loss-term parameter gradients, ~5% extra compute; `--log-activation-grad-norms` for the older, less-reliable activation-gradient probes, off by default); checkpoint save/resume (`--checkpoint-dir`/`--resume`/`--reset-lr-schedule`/`--reset-optimizer-state`/`--wandb-new-run`, optional HF Hub backup via `--hf-backup-repo`/`--hf-backup-every` — decoupled so a slow upload doesn't have to happen on every local save); optional wandb logging. See `--help` for the full flag list |
| `scripts/reconstruct.py` | Runs a real image or a real video clip (`.mp4`/`.mov`/`.avi`/`.mkv`/`.webm`, decoded frame-by-frame via `imageio`) through the codec once and saves a side-by-side comparison grid — top row original frames, bottom row reconstructed, aligned by timestep. Random-init weights only, no `--codec-checkpoint` — a mechanism smoke test, not an evaluation of a trained checkpoint |
| `scripts/evaluate_codec.py` | Real, quantitative evaluation of a **trained** codec checkpoint: the same three training-loss terms plus PSNR/SSIM/LPIPS (reusing `full_eval_metrics.py`'s functions), plus a batch of saved preview videos (`log_preview`, works locally, optional `--wandb-project`). Point `--index-path` at genuinely unseen data via `kyutai/rocket-science`'s own dedicated `test` split (`scripts/download_shards.py --split test`) rather than the `train` split the codec streams from — `scripts/build_holdout_split.py` is also available for isolating extra `train` shards if a dataset elsewhere has no dedicated split. Only `loss_mae` in its output is comparable to `train_codec.py`'s own charts — see the script's own docstring for why the other two loss terms aren't |
| `scripts/compute_latent_stats.py` | One-shot latent mean/std computation from a frozen codec checkpoint — its JSON output is `train_world_model.py`'s `--latent-stats` input |
| `scripts/train_world_model.py` | Real GPU world-model training: a frozen codec checkpoint (`--codec-checkpoint`) plus a trainable `DiffusionTransformer`/`ActionEncoder`, real streamed data with real actions (unlike the codec, which ignores them), mira's real diagonal flow-matching loss with optional PSD self-distillation (`--psd-weight`/`--psd-loss-prob`), periodic validation loss plus the full drift/Frechet/PSNR/LPIPS/SSIM eval suite and rendered rollout videos, checkpoint save/resume, optional wandb/HF Hub backup. Its first real-GPU run confirmed the core training step, validation loop, and PSD loss all work correctly; it crashed one layer deeper, in the drift-metric eval (`compute_drift_metrics` assumed a tensor where multi-layer DINO aggregation returns a list) — found and fixed, not yet re-run to confirm past that point, see [Status](#status) |
| `scripts/verify_world_model_training.py` | CPU-friendly mechanism proof for `train_world_model.py`: overfit convergence, frozen/trainable gradient isolation, the PSD mechanism, checkpoint round-trip — a fake DINO stand-in, no gated weights or GPU needed |
| `scripts/verify_full_eval_metrics.py` | CPU-friendly mechanism proof for the full eval suite (Frechet/PSNR/LPIPS/SSIM/rollout video) using fake DINO/Inception/LPIPS stand-ins, including genuine ffmpeg-encoded video output |

See [Scale](#scale) above for parameter counts at both the intended ~300M scale
(`configs/scaled_300m.yaml`) and the small default used for fast testing (`configs/small.yaml`).

## Scope

**Implemented:**
- Strided-conv bottleneck (encoder) and ViT space-time decoder, matching the real codec's shape
  contract and space/time attention factorization.
- Rotary position embeddings (temporal + axial 2D spatial).
- QK-norm (`QKRMSNorm`/`QKLayerNorm`, matching mira's shipped codec config's choice of
  `layernorm`) and mira-matching weight initialization (`src/mini_mira/ml/init.py`). QK-norm
  reaches both the decoder and the world model automatically, since both share the same
  `SelfAttention` class; weight init is applied explicitly to the bottleneck, decoder, and
  world model each.
- AdaLN conditioning of the world model on the diffusion timestep `tau`, via a sinusoidal
  timestep embedding matched to the real implementation.
- Clean-past conditioning: each denoising step additively conditions on the real, clean latent
  content of the previous frame (`clean_past`), with a learned `bos` parameter standing in for
  the frame before the first one in a clip. This is what makes the pipeline's output actually
  depend on its input, rather than only on noise and `tau`. Note this is teacher forcing, not
  generation: `clean_past` here comes from the real, encoded input (ground truth), not from
  previously-generated frames — true autoregressive rollout is not implemented.
- Action conditioning: raw multi-hot key-press actions are encoded by `ActionEncoder` (per-key
  learned embeddings, pooled from video frame rate to latent frame rate, with a learned
  `initial_action_token` standing in for the action before the first frame — the same role
  `bos` plays for `clean_past`) and added into the same AdaLN `cond` signal as `tau`.
- Flow-matching sampling: a multi-step Euler integration loop from pure noise to a decoded video.
- The frozen visual encoder (`DinoModel`, `src/mini_mira/codec/dino.py`): the real DINOv3 ViT-B/16
  backbone, loaded with real pretrained weights (gated by Meta — must be downloaded manually
  and pointed to via `RS_DINO_WEIGHTS_DIR`, matching mira's own environment variable
  convention). Loaded by importing the backbone constructor directly rather than through
  `torch.hub.load`'s `hubconf.py`, which otherwise pulls in unrelated segmentation/detection/
  depth/text-alignment dependencies this project never uses — see `notes/deviations.md`.
  Wired into `MyPipeline` behind an opt-in flag: `PipelineConfig.use_real_dino` (default
  `False`) keeps every existing fast test unchanged, taking pre-encoded `dino_features`
  directly; setting it `True` builds the real frozen backbone in `__init__` and switches
  `forward`'s first argument to raw video instead, verified end to end in
  `scripts/verify_dino.py`.
- **Real codec training**, on a rented GPU, on real Rocket League data:
  `scripts/train_codec.py` streams real clips (`mira.data.training_loader.create_loader`, real
  shards from `scripts/download_shards.py`) through the real three-term loss
  (`src/mini_mira/codec/loss.py`'s `CodecLoss`: L1 + LPIPS + DINO latent-consistency, matching
  mira's own `CodecLoss` — mira's codec has no ELBO or KL anywhere; it isn't a VAE) with
  `auto_weight` adaptive loss balancing on, mira's real cosine LR schedule, hybrid fp16/bf16
  precision with `GradScaler`, gradient accumulation and activation checkpointing for
  full-resolution batches, per-term gradient-norm instrumentation, and checkpoint save/resume.
  `scripts/verify_codec_training.py` remains the fast, GPU-free mechanism check (one synthetic
  video, random-init DINOv3, asserts loss drops); `train_codec.py` is the real thing.
- **World-model training mechanism**: `scripts/train_world_model.py` — a frozen, real-checkpoint
  codec plus a trainable `DiffusionTransformer`/`ActionEncoder`, real mira's diagonal
  flow-matching loss with optional PSD self-distillation, real streamed data now actually
  consuming its action channel (unlike the codec, which ignores actions entirely), checkpoint
  save/resume, and a full eval suite (drift metrics, Frechet DINO/Inception Distance, PSNR,
  LPIPS, SSIM, rendered rollout videos). Verified by reading and by CPU-mechanism tests
  (`scripts/verify_world_model_training.py`, `scripts/verify_full_eval_metrics.py`), and its
  **first real-GPU smoke tests** confirmed the training step, validation loop, PSD loss, drift
  metrics, and the full Frechet/PSNR/LPIPS/SSIM suite all work — see [Status](#status) for the
  two bugs those runs caught, and rollout video rendering, the one piece still unconfirmed.

**Deliberately simplified or not yet implemented**, each a disclosed decision rather than a gap
found later:
- **Actions are keys only, no mouse.** The released Rocket League data never carries real mouse
  signal either (mouse movement is always zero, sensitivity always unknown), so this matches
  mira's actual data more than it simplifies away from it. Also simplified relative to mira's
  `ActionEncoder`: no dropout (training-only), mean-pooling instead of a learned temporal pool,
  and a plain `Linear` projection instead of mira's power-of-2 per-key dimension split (which,
  at mira's real numbers, wastes 44% of its keyboard channels as zero-padding — see
  `notes/deviations.md`).
- **Autoregressive rollout.** `clean_past` is always built from the real encoded input, never
  from the model's own previous output — see the note above.
- **Streaming inference.** No KV-cache; every diffusion step recomputes the whole sequence.
- **Grouped-query attention.** Attention here always uses as many KV heads as query heads.
- **World-model training fully exercised on real hardware.** `train_world_model.py` (see
  Implemented above) has now run on a real GPU once, but only far enough to confirm the training
  step, validation loop, and PSD loss — it crashed in the drift-metric eval (now fixed, see
  [Status](#status)) before reaching the heavier Frechet/PSNR/LPIPS/SSIM suite or rollout video
  rendering, which remain unconfirmed on real hardware.
- **Quantitative held-out evaluation for the codec — built, run for real, on genuinely unseen
  data.** `train_codec.py` itself still has no `--test-index-path`/validation loss
  (`scripts/reconstruct.py` doesn't fill this either — no `--codec-checkpoint` flag at all,
  random-init weights only). `scripts/evaluate_codec.py` loads a real trained checkpoint, reuses
  the exact PSNR/SSIM/LPIPS functions already proven for the world-model eval suite, and saves a
  batch of side-by-side preview videos. Evaluated against `kyutai/rocket-science`'s own dedicated
  `test` split (found while building this — a cleaner solution than isolating extra `train`
  shards). First real numbers, on the step-3999 checkpoint: `psnr=19.56dB, ssim=0.552,
  lpips_standardized=0.486` — by standard external benchmarks, still solidly "poor" quality,
  consistent with (and now a quantitative confirmation of) the qualitative preview-frame
  assessment throughout this document.
- **One shared implementation where the real repo has two.** The real codec and world model
  never share code with each other, even for identical logic (e.g. two separately-named but
  functionally identical MLP classes, two separate RoPE implementations). mini_mira
  consolidates the shared pieces into `blocks.py` and `rope.py` instead.

## Verification

Shape correctness alone doesn't prove a mechanism works — a broken RoPE implementation or a
zeroed-out conditioning signal can still produce the right-shaped output. This project checks
both:

- **`scripts/test_shapes.py`** — asserts every stage's output shape against the real codec's
  actual configuration (`(2, 40, 768, 18, 32)` DINO-shaped input through to `(2, 40, 3, 288,
  512)` decoded video).
- **`scripts/verify_rope.py`** — targeted behavioral checks for RoPE, each aimed at a specific
  silent failure mode:
  - *Causality*: perturbing only the last frame of a clip must not change any earlier frame's
    output.
  - *Position sensitivity*: swapping two input frames must not simply swap the output back — a
    degenerate (all-zero or all-identical) set of rotary frequencies would do exactly that.
- **`scripts/verify_conditioning.py`** — targeted behavioral checks for AdaLN, clean-past, and
  actions:
  - *`tau` sensitivity*: the same latent with two different `tau` values must produce different
    predicted velocities.
  - *`clean_past` sensitivity*: the same latent and `tau` with two different `clean_past` values
    must produce different predicted velocities — and the reverse, identical `clean_past` must
    produce identical output, confirming the model is deterministic.
  - *Actions sensitivity*: the same latent, `tau`, and `clean_past` with two different encoded
    action tensors must produce different predicted velocities.
  - *End-to-end (clean_past)*: the same noise seed with two completely different input videos
    must now produce different decoded output — the direct regression test for the bug where
    the pipeline's output used to be independent of its input entirely.
  - *End-to-end (actions)*: the same input video and noise seed with two different raw key-press
    sequences must produce different decoded output — this exercises `ActionEncoder` itself
    (per-key embeddings, temporal pooling, `initial_action_token`), not just the AdaLN pathway.
- **`scripts/verify_dino.py`** — targeted behavioral checks for the real DINOv3 backbone:
  - *Frozen*: every parameter has `requires_grad=False`.
  - *Shape at pipeline resolution*: a `288×512` input produces `(dino_dim, 18, 32)` features,
    matching what `MyBottleneck` expects.
  - *Patch-alignment resizing*: a resolution that isn't a multiple of 16 (e.g. `300×500`) must
    still resize and produce a correctly-rounded shape, not crash or silently misalign.
  - *Non-degenerate output*: real features must be finite and have real variance — a silent
    weight-loading failure would tend to produce all-zero or NaN output instead of a clean crash.
- **`scripts/verify_codec_training.py`** — checks the codec's training mechanism itself, not
  just its architecture: overfitting one fixed synthetic video with real optimizer steps must
  substantially reduce the reconstruction loss. A wrong loss, a detached graph, or params that
  silently aren't receiving gradients would all show up here as a loss that never moves.

## Getting started

Requirements: Python ≥ 3.10, plus `requirements.txt` (`torch`, `einops`, `pyyaml`, `pillow`,
`imageio`/`imageio-ffmpeg`, `numpy`, `torchmetrics`, `torchvision` — each verified against the
actual code path that uses it, not against everything installed while debugging DINOv3 loading; see
`notes/deviations.md` 1.14 for why `termcolor` specifically is NOT listed). `wandb` and
`huggingface_hub` are optional, lazily-imported dependencies of `train_codec.py`'s
`--wandb-project`/`--hf-backup-repo` flags only; `torchcodec` is needed only for `--index-path`
(real streamed data) — see `requirements.txt`'s own comments for version-pinning details and a
known Windows/FFmpeg issue with it. There is no packaging configuration — the scripts add `src/`
to `sys.path` directly.

```
pip install -r requirements.txt
python scripts/test_shapes.py
python scripts/verify_rope.py
python scripts/verify_conditioning.py
python scripts/verify_codec_training.py
```

The first run of anything that builds a `CodecLoss` (`verify_codec_training.py`, `train_codec.py`)
downloads pretrained VGG16 ImageNet weights for the LPIPS term (~528MB, one-time, needs internet).

`scripts/verify_dino.py` additionally requires real, gated DINOv3 pretrained weights on disk
(request access at
[ai.meta.com/resources/models-and-libraries/dinov3-downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)),
pointed to via an environment variable — matching mira's own convention exactly:

```
RS_DINO_WEIGHTS_DIR=/path/to/dir/containing/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth python scripts/verify_dino.py
```

**Real training** additionally needs a CUDA GPU and real data:

```
python scripts/download_shards.py --shards 50   # prints the local index path to pass below
python scripts/train_codec.py --config configs/scaled_300m.yaml \
  --index-path <path printed above> --require-pretrained-dino \
  --height 288 --width 512 --frames 40 --batch-size 4 --grad-accum-steps 8 \
  --activation-checkpointing --steps 1600
```

`train_codec.py --help` lists the full set of flags (LR schedule, `--loss-mae-weight`, chunked
LPIPS/DINO scoring, checkpoint/resume, wandb). Without `--index-path`, it trains on one fixed
synthetic video instead — no GPU or real data required, useful as a fast local mechanism check
(same role as `scripts/verify_codec_training.py`, just via the real training script's own path).

**Resuming into a new fine-tune phase** (different `--lr`/resolution/step budget than the
checkpoint was originally trained under, e.g. moving to full resolution for the run's final
steps) needs a few more flags together:

```
python scripts/train_codec.py --config configs/scaled_300m.yaml \
  --index-path <path> --require-pretrained-dino \
  --height 288 --width 512 --frames 40 --batch-size 1 --grad-accum-steps 32 \
  --activation-checkpointing --resume --steps <checkpoint's resumed step + N> \
  --lr 5e-5 --lr-min 1e-5 --lr-warmup-steps 0 --reset-lr-schedule \
  --precision bf16 --wandb-new-run
```

- `--resume` alone restores weights *and* optimizer momentum, but leaves the LR schedule
  continuing the checkpoint's original curve — add `--reset-lr-schedule` to instead start a
  fresh warmup/decay shape sized for this new phase (using this run's own `--lr`/`--lr-min`/
  `--lr-warmup-steps`/`--lr-decay-steps`, not the checkpoint's original ones).
- `--reset-optimizer-state` is a separate, independent reset dimension: it discards AdamW's
  momentum/variance instead of carrying it over, for when the checkpoint's momentum was
  calibrated under a regime (resolution/precision) that may not transfer cleanly to the new one.
  Most restarts of this kind want both `--reset-lr-schedule` and `--reset-optimizer-state`
  together, but they're independent flags — one can be set without the other.
- **The resumed step count is `checkpoint's saved step + 1`, not the saved step itself** — e.g.
  a checkpoint saved at step 3099 resumes training at step 3100. Off-by-one here silently runs
  one fewer step than intended; there's no error, since `range(start_step, --steps)` just
  produces a shorter-than-expected range.
- `--precision bf16` is opt-in (default `fp16-hybrid`) — see the `Precision` note in
  `train_codec.py`'s own module docstring for when each is the better choice. Resuming a
  checkpoint under a *different* `--precision` than it was saved with crashes (`GradScaler`
  state saved under a disabled scaler is an empty dict; loading it into an enabled one raises) —
  a known, currently-unfixed gap, harmless as long as a checkpoint's resumes all agree on
  `--precision`.
- `--wandb-new-run` opens a fresh chart instead of continuing the checkpoint's original run —
  useful precisely when the phase itself is new (different resolution/LR/step budget), not a
  continuation of an interrupted one.

**Continuing the same phase across multiple sessions** (e.g. GPU access is time-boxed and one
long run has to be split into several shorter launches) needs plain `--resume` — no reset flags,
since this is explicitly *not* a new phase — but with one easy-to-miss requirement:

```
python scripts/train_codec.py --config configs/scaled_300m.yaml \
  --index-path <path> --require-pretrained-dino \
  --height 288 --width 512 --frames 40 --batch-size 1 --grad-accum-steps 32 \
  --resume --steps <this session's target> \
  --lr 5e-5 --lr-min 1e-5 --lr-warmup-steps 25 --lr-decay-steps 475 \
  --precision bf16
```

`--lr`/`--lr-min`/`--lr-warmup-steps`/`--lr-decay-steps` must be **passed again, identically,
on every session** in the arc — even though nothing about the schedule's *shape* is changing.
This is because the script recomputes `warmup_steps`/`decay_steps` from whatever `--steps` is
passed on the command line **unconditionally, regardless of `--reset-lr-schedule`** — so a later
session that omits these flags doesn't "just continue," it silently recomputes them from that
session's own absolute step target instead of the original arc's length, corrupting the LR curve
with no error message. Passing the same values every time is a safe no-op (they match what's
already in the checkpoint); omitting them is the actual failure mode. Size `--lr-warmup-steps`/
`--lr-decay-steps` once, to the *full* multi-session arc's total step count, not any individual
session's slice of it.

**World-model training** (run once as a real-hardware smoke test so far, not yet a real training
run — see [Status](#status)) needs a frozen codec checkpoint and its latent statistics, on top of
everything the codec needs:

```
python scripts/compute_latent_stats.py --codec-checkpoint <checkpoint.pth> --index-path <path> \
  > latent_stats.json
python scripts/train_world_model.py --config configs/scaled_300m.yaml \
  --codec-checkpoint <checkpoint.pth> --latent-stats latent_stats.json \
  --index-path <train data> --test-index-path <held-out data> --require-pretrained-dino
```

`train_world_model.py --help` lists the full flag set (PSD self-distillation weights, eval
cadence/scope, checkpoint/resume, wandb/HF Hub backup) — same shape as `train_codec.py`'s own
flags wherever the concept is shared.

## Configs

Named, comparable presets for `PipelineConfig` live in `configs/` at the repo root — data, not
installable package code, so it sits alongside `src/`, `scripts/`, `notes/`, and `traces/` rather
than under `src/mini_mira/` — matching the real mira release's own top-level `mira/configs/`
convention. This is a deliberately lightweight system: one YAML file is one full preset mirroring
`PipelineConfig`'s whole nested shape (`bottleneck` / `world_model` / `decoder` sections, plus the
top-level `n_diffusion_steps` / `num_keys` fields) — not mira's own per-component-file composition
system (no Hydra, no config groups, no CLI overrides).

| File | Purpose |
|---|---|
| `configs/scaled_300m.yaml` | **The intended architecture scale** — scales `decoder`/`world_model` width and depth toward a ~300M parameter target, tuned to also roughly match real mira's own world_model/decoder size ratio (1.82) rather than pinning the decoder to the frozen-encoder size exactly (measured: 292,457,344 parameters, ratio 1.81). See [Scale](#scale) above. |
| `configs/small.yaml` | Mirrors `PipelineConfig()`'s own Python defaults exactly — the size every fast verification script uses (11,320,960 parameters), not the intended final scale |

Load a preset with `load_pipeline_config` (`src/mini_mira/ml/config_loading.py`):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mini_mira.ml.config_loading import load_pipeline_config
from mini_mira.pipeline import MyPipeline

config = load_pipeline_config("configs/scaled_300m.yaml")
pipeline = MyPipeline(config)
```

Each section is keyword-unpacked straight into its dataclass's constructor, so an unrecognized
key raises `TypeError` immediately — the same typo safety as constructing the dataclass directly
in Python.

## Relationship to mira and attribution

mini_mira was built by tracing the official MIRA code release,
[`mira-wm/mira`](https://github.com/mira-wm/mira) (Apache License 2.0), and reimplementing its
architecture from that understanding. Class and config names match the real repository wherever
there is a genuine one-to-one correspondence — for example `ViTVideoDecoder`, `SelfAttention`,
and `DiffusionTransformer` all name the same component they do in the original. Two components
deliberately keep their own names instead: mini_mira's `MyBottleneck` and `MyPipeline` have no
equivalent class in the real repository (the real bottleneck is an unnamed `nn.Conv3d` attribute
inside a larger class, and the closest match to `MyPipeline` owns a full frozen codec rather than
a bare decoder), so renaming them would claim an equivalence that isn't there. A few small,
self-contained pieces of math with no reason to be reinvented — `AdaptiveLayerNorm`, the
sinusoidal timestep embedding, and the RoPE frequency computation — are adapted directly from
the original source under its Apache License 2.0 terms.

## Status

Architecture-correctness work is complete for the codec: shared block/RoPE infrastructure
decoupled from the decoder and world model, RoPE wired into both, AdaLN conditioning on `tau`,
clean-past conditioning (with a learned `bos` for the first frame of a clip), action
conditioning (with a learned `initial_action_token` playing the same role for the first frame's
missing action), and the real, frozen DINOv3 backbone (loaded with real pretrained weights,
wired into `MyPipeline` behind the opt-in `use_real_dino` flag so existing fast tests stay
unaffected). The forward pass now conditions on every signal mira's real architecture conditions
on.

The codec is now in real training on a rented GPU, on real Rocket League data — see the Training
bullet under [Scope](#scope) for what that covers. Getting there surfaced gaps that shape/behavior
checks alone couldn't: QK-norm and weight initialization were both missing entirely until real
training exposed why they matter (unbounded attention logits and a bad initial loss landscape,
respectively, neither visible without an actual optimizer loop), and the encoder was silently
feeding the bottleneck only its last DINO layer rather than mira's real multi-layer aggregation —
a genuine information bottleneck, not just a training-stability issue. All are fixed and verified
against mira's real source, not assumed. `notes/deviations.md` has the full, evidence-based audit
trail for these and every other intentional or since-corrected difference from real mira.

The world model now has a real training mechanism too (`scripts/train_world_model.py`): mira's
diagonal flow-matching loss, optional PSD self-distillation, real action conditioning from real
streamed data, checkpointing, and a full eval suite. Every tensor shape along the loss's full
computation path — encode → shift/bos → flow-matching targets → `DiffusionTransformer` (AdaLN
conditioning, timestep embedding) → loss — has been checked against real mira's own source where
a direct claim was checkable, not assumed from comments alone. Its first real-GPU smoke test
confirmed the training step, validation loop, and PSD loss all execute correctly, then crashed
one layer deeper: `eval_metrics.py`'s `compute_drift_metrics` assumed `model.dino.dino_forward()`
always returns a tensor, but it returns a **list** under multi-layer DINO aggregation (the mode
this project actually uses) — exactly the kind of bug reading-only verification can't catch.
Fixed to match the same last-layer convention `full_eval_metrics.py`'s sibling function already
used correctly for the identical situation. Re-run after that fix, the heavier Frechet DINO/
Inception Distance, PSNR, LPIPS, and SSIM suite (`full_eval_metrics.py`) now works correctly on
real hardware too, producing real numbers for the first time. Rollout video rendering
(`rollout_visualization.py`) hit a second real bug one step further: `render_rollout_sample`
blends a GPU-resident video with a keyboard-HUD overlay that's inherently built on CPU
(`draw_key_grid_video`, via PIL/numpy) without ever moving it to the video's device first — a
device-mismatch crash no CPU-only fake-component test could have caught, since everything would
already be on the same (CPU) device there. Fixed by moving the overlay to the video's device at
the call site, not inside the ported-verbatim `overlay_video` itself. Not yet re-run to confirm.
Three further points of comparison against real mira's `DiffusionTransformer` were checked, and
only one is a genuine open divergence: an extra, unconditioned `LayerNorm` before the output
projection with no equivalent in real mira. The other two — AdaLN conditioning hardcoded onto the
attention sublayers, and clean-past conditioning unconditionally on — were flagged earlier as
divergences because real mira's *Python class defaults* are off for both. But real mira's actual
*shipped, trained-with* config overrides both to on (`ada_attn_ln: true`, `use_clean_past: true`
in its `configs/model/latent_world_model.yaml`) — so mini_mira hardcoding them on actually matches
what real mira trains with in practice; the class defaults were never the recipe. Only the extra
final `LayerNorm` remains open, and it's low-risk either way — not urgent to fix.

Resuming a codec checkpoint into a *new* fine-tune phase (different resolution/LR/step budget
than the checkpoint was originally trained under) surfaced a real, general bug class in this
project's resume logic: `torch.optim.lr_scheduler.LRScheduler.load_state_dict` restores its
*entire* internal state, including `base_lrs`, not just the step counter — so any resume that
changes the LR schedule's shape needs to explicitly re-derive everything `get_lr()` depends on,
or it silently keeps computing off the checkpoint's *original* values with no error. Found and
fixed in `train_codec.py`'s `--reset-lr-schedule` path first (confirmed directly from a benchmark
run's printed LR staying wrong despite a different `--lr`), then found to affect the *plain*
`--resume` path too (`base_lrs` was only being re-derived inside the reset branch, while
`--lr-min` was already unconditional — a self-contradictory decay curve on any plain resume with
a changed `--lr`) and fixed there as well; `train_world_model.py` already handled this
unconditionally from the start.

A real, since-fixed correctness bug and a real, since-fixed memory leak were also found this way:
`compute_drift_metrics`'s list-vs-tensor crash above, and `CodecLoss`'s LPIPS term — a
`torchmetrics.Metric` that accumulates internal state on every `forward()` call unless explicitly
`.reset()`, unbounded over a long run, with no equivalent leak in real mira's own stateless
`lpips.LPIPS`. Both fixed and confirmed (the LPIPS fix via an isolated mechanism test, given no
local GPU to run the codec training loop itself on).

Remaining gaps are the ones listed in Scope above: no autoregressive rollout, no streaming
inference, and the one remaining world-model architectural divergence from real mira noted above
(the extra final `LayerNorm`). Quantitative held-out codec evaluation is no longer a gap — see
`scripts/evaluate_codec.py` above.

Several safety-net and efficiency fixes landed in `train_world_model.py` ahead of a real
(non-smoke-test) run:
- **Checkpoint provenance**: a checkpoint now records which codec checkpoint and latent-stats
  values it was trained against, and `--resume` warns (doesn't block) if a future run pairs it with
  different ones — closes the same class of silent-mismatch risk that caused the codec's
  LR-schedule continuity bug.
- **RNG state and dataloader position are now both saved/restored across `--resume`**: a resumed
  run continues the same random stream (noise/tau draws) and fast-forwards a fresh dataloader past
  already-consumed batches, instead of silently restarting both from scratch every relaunch. The
  dataloader fast-forward relies on `create_loader`'s un-seeded default (fixed seed, single
  process) staying un-overridden — documented at the call site.
- **Eval no longer redoes the same decode+DINO pass 2-3 times per batch**: `compute_drift_metrics`,
  `compute_full_eval_metrics`, and rollout-video rendering now all share one
  `eval_metrics.decode_and_dino(...)` call instead of each decoding/DINO-ing independently.
- **RoPE tables are now cached** on the `DiffusionTransformer` instance instead of rebuilt on every
  forward call — they only depend on (height, width, frame count), which are constant for the
  whole life of a model instance.
- **`--precision` flag added** (`fp16-hybrid` / `bf16`, default `bf16`), matching `train_codec.py`'s
  own flag. Previously hardcoded to `fp16-hybrid` + GradScaler unconditionally, a leftover of the
  project's old V100-only workaround — the codec script itself moved to a plain-`bf16`-by-default
  setup once the project moved to a GPU with native bf16 tensor cores, but the world-model script
  never got the same update until now.

An `--activation-checkpointing` flag (no equivalent exists for the world model at all yet — no
KV-cache, full temporal attention every layer) was scoped but deliberately held back until the
supervisor confirms a real batch-size/resolution target for this run, since that's what would
determine whether it's actually needed.
