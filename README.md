# mini_mira

A from-scratch reimplementation of the core architecture behind **MIRA** — an
action-conditioned latent world model for Rocket League built on a representation-autoencoder
(RAEv2) codec and a flow-matching diffusion transformer. This project reimplements the codec
and world-model architecture end to end and verifies it with shape and behavioral tests; it does
not train a model.

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

The supervisor's target is ~300M parameters. `configs/scaled_300m.yaml` is the real attempt at
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
| `src/mini_mira/ml/blocks.py` | Shared building blocks: `LayerScale`, `SwiGLU`, `SelfAttention`, `SpaceTimeBlock` (decoder), `AdaptiveLayerNorm`, `AdaSTBlock` (world model) |
| `src/mini_mira/ml/rope.py` | Rotary position embeddings — one shared implementation (temporal + axial spatial), used by both the decoder and the world model |
| `src/mini_mira/world_model/timestep_encoder.py` | `DiffusionTimeEmbedding` — sinusoidal embedding of the diffusion timestep `tau` |
| `src/mini_mira/world_model/action_encoder.py` | `ActionEncoder` — encodes raw multi-hot key-press actions into per-latent-frame conditioning vectors |
| `src/mini_mira/pipeline.py` | `PipelineConfig`, `MyPipeline` — wires bottleneck → multi-step diffusion loop → decoder into one forward pass; owns `bos`/`clean_past` and `ActionEncoder` |
| `src/mini_mira/codec/dino.py` | `DinoModel` — the real, frozen DINOv3 (ViT-B/16) backbone; loads real pretrained weights, wired into `MyPipeline` behind `PipelineConfig.use_real_dino` (see Scope) |
| `src/mini_mira/codec/loss.py` | `CodecLoss`, `CodecLossWeights`, `CodecOutputs` — the codec's real training loss (L1 + LPIPS + DINO latent-consistency, matching mira's `CodecLoss` minus `auto_weight`); `normalize_video`/`denormalize_for_dino` — the `[0,1]`↔`[-1,1]` pixel-range conversions the codec and DINO each expect |
| `scripts/test_shapes.py` | Shape-correctness checks for every stage of the pipeline |
| `scripts/verify_rope.py` | Behavioral checks for RoPE: causal masking and position sensitivity |
| `scripts/verify_conditioning.py` | Behavioral checks for AdaLN, clean-past, and actions: `tau`/`clean_past`/action sensitivity, determinism, and end-to-end pipeline checks |
| `scripts/verify_dino.py` | Behavioral checks for the real DINOv3 backbone: frozen parameters, correct output shape, patch-alignment resizing, non-degenerate features. Requires real gated weights on disk (see Getting started) |
| `scripts/test_dino.py` | Raw sanity check for the dinov3_vitb16 backbone bypassing `DinoModel` entirely — a diagnostic to tell apart "mini_mira's wrapper is broken" from "the underlying library/weights are broken" if `verify_dino.py` ever fails |
| `src/mini_mira/ml/config_loading.py` | `load_pipeline_config` — builds a `PipelineConfig` from a YAML preset under `configs/` |
| `configs/small.yaml` | Named preset mirroring today's dataclass defaults exactly (regression-safe baseline) |
| `configs/scaled_300m.yaml` | Named preset scaling `decoder`/`world_model` width and depth toward the supervisor's ~300M parameter target |
| `scripts/verify_codec_training.py` | Mechanical proof the codec's training mechanism is wired correctly: overfits one fixed synthetic video on the real `CodecLoss` and asserts the total loss drops substantially. Small config, random-init DINOv3 — no gated weights, runs in seconds |
| `scripts/train_codec.py` | Config-driven codec training on the real `CodecLoss` (L1 + LPIPS + DINO latent-consistency; no `auto_weight` yet — see `notes/deviations.md`). Defaults to a fast toy size; `--config`/`--height`/`--width`/`--frames` point it at the real target scale once there's a GPU. No real dataset exists yet — trains on one fixed synthetic video |
| `scripts/reconstruct.py` | Runs a real image or a real video clip (`.mp4`/`.mov`/`.avi`/`.mkv`/`.webm`, decoded frame-by-frame via `imageio`) through the codec once (untrained weights) and saves a side-by-side comparison grid — top row original frames, bottom row reconstructed, aligned by timestep. Proves the real-data path works mechanically — not that the reconstruction looks like anything yet |

See [Scale](#scale) above for parameter counts at both the intended ~300M scale
(`configs/scaled_300m.yaml`) and the small default used for fast testing (`configs/small.yaml`).

## Scope

**Implemented:**
- Strided-conv bottleneck (encoder) and ViT space-time decoder, matching the real codec's shape
  contract and space/time attention factorization.
- Rotary position embeddings (temporal + axial 2D spatial).
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
- **Training.** The codec (`DinoModel` → `MyBottleneck` → `ViTVideoDecoder`) trains on the real
  three-term reconstruction loss (`src/mini_mira/codec/loss.py`'s `CodecLoss`: L1 + LPIPS + DINO
  latent-consistency, matching mira's own `CodecLoss` — mira's codec has no ELBO or KL anywhere;
  it isn't a VAE), verified to actually reduce loss (`scripts/verify_codec_training.py`) — but
  only on one fixed synthetic video; no real dataset, no `auto_weight` adaptive loss balancing
  (needs the decoder's last-layer weight exposed plus gradient-norm bookkeeping — deferred, flat
  fixed weights used instead), no LR schedule. The world model has no training mechanism at all
  yet (only the inference-time sampling loop in `MyPipeline.forward` exists) — deliberately
  deferred to a later task. Every weight starts randomly initialized either way; this project
  mostly verifies architecture and shapes, not learned behavior.
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
`imageio`/`imageio-ffmpeg`, `torchmetrics`, `torchvision` — each verified against the actual code
path that uses it, not against everything installed while debugging DINOv3 loading; see
`notes/deviations.md` 1.14 for why `termcolor` specifically is NOT listed). There is no packaging
configuration — the scripts add `src/` to `sys.path` directly.

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
| `configs/scaled_300m.yaml` | **The intended architecture scale** — scales `decoder`/`world_model` width and depth toward the supervisor's ~300M parameter target, tuned to also roughly match real mira's own world_model/decoder size ratio (1.82) rather than pinning the decoder to the frozen-encoder size exactly (measured: 292,457,344 parameters, ratio 1.81). See [Scale](#scale) above. |
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

Architecture-correctness work in progress. Completed so far: shared block/RoPE infrastructure
decoupled from the decoder and world model, RoPE wired into both, AdaLN conditioning on `tau`,
clean-past conditioning (with a learned `bos` for the first frame of a clip), action
conditioning (with a learned `initial_action_token` playing the same role for the first frame's
missing action), and the real, frozen DINOv3 backbone (loaded with real pretrained weights,
wired into `MyPipeline` behind the opt-in `use_real_dino` flag so existing fast tests stay
unaffected). The forward pass now conditions on every signal mira's real architecture conditions
on. Remaining gaps are the ones listed in Scope above: no autoregressive rollout, no streaming
inference, and no training.
