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
| `VjepaModel` (frozen, real pretrained weights, V-JEPA track only, not in the total) | 86,833,152 |

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
code, not assumed.

`scripts/train_codec_vjepa.py` is done — a full fork of `train_codec.py`, `VjepaModel` in place of
`DinoModel` throughout, `--checkpoint-dir` defaulting to `checkpoints_vjepa/` and its local/HF
checkpoint filename tagged `checkpoint_vjepa.pth` so it can never collide with DINO's own upload to
the same HF repo. One flag pair has no V-JEPA equivalent and was dropped rather than faked:
`--perceptual-dino-model`/`--perceptual-dino-multilayer` swap in a second, differently-sized DINO
variant for the consistency-loss term, and V-JEPA 2.1 only has the one variant.

**Three real bugs found and fixed**, only surfaced by actually running the training mechanism
end to end (`scripts/verify_codec_training_vjepa.py`, new — same overfit-one-video proof
`verify_codec_training.py` uses for the DINO track), not by reading the code:
- `VjepaModel.dino_forward` crashed on fewer than `tubelet_size` (2) frames — `CodecLoss`'s
  consistency term scores a random frame subset, sometimes chunked down to a single frame, which
  `DinoModel` always tolerated (no minimum) but V-JEPA's frame-pairing can't. Fixed by padding a
  too-short input up to 2 frames (repeating the last one) instead of rejecting it — contained
  entirely inside `vjepa.py`, no shared code touched.
- `CodecLoss` assumed a pixel-space frame index always equals a feature-space frame index — true
  for `DinoModel` (frame count in = frame count out) but not V-JEPA (halves it). Fixed in `loss.py`:
  target-feature lookup now remaps indices by the encoder's own reduction ratio (inferred from
  tensor shapes, not a hardcoded per-backbone constant — a no-op for DinoModel, ratio 1 always).
- The same term flattened batch and selected-frame together before chunking, so a chunk could
  straddle two *different videos* in the batch — invisible for `DinoModel` (no cross-frame
  interaction at all) but silently wrong for V-JEPA, which would pair the last selected frame of
  one video with the first frame of a completely different one and treat it as real motion. Silent,
  no crash — dormant at today's real 40-frame default settings by coincidence (the random selection
  size happens to be even), live the moment `--frames` or `--perceptual-chunk-size` changes to
  almost any other value. Fixed in `loss.py` by chunking within each video's own selected frames
  only, never across videos — confirmed directly: instrumented the real encoder call and proved,
  under the exact conditions that reproduced the bug (odd selection size, small chunk size), zero
  calls mix two videos, across 6 real calls.

Verified for real: a 100-step overfit run (batch=2, 16 frames — large enough to exercise all three
bugs' trigger conditions, unlike a smaller test) drops `loss_total` 57.7%, matching the same >50%
bar `verify_codec_training.py` itself uses. `loss_dino_latent_consistency` itself barely moves
(expected — matches the DINO track's own already-documented finding that MAE dominates this term's
gradient 4-8x; the auto-balancing mechanism visibly compensating, its weight climbing 36.2→47.0, is
this working as designed, not a new problem).

**Supervisor directive: train V-JEPA at native resolution (720×1280), not the 288×512 downscale —
V-JEPA track only, DINO untouched.** No code change needed: `resize_to_canonical` (shared with
DINO, in `video_prep.py`) already no-ops once `--height`/`--width` match a clip's real shape, so
passing `--height 720 --width 1280` on the V-JEPA launch is the entire change. `MyBottleneck`
(strided-conv, no absolute-position assumption) and the decoder (RoPE) are both already
resolution-agnostic, confirmed by reading the code, and V-JEPA 2.1's own sincos position
embeddings interpolate to arbitrary input shapes internally.

**Native resolution was tried for real on the rented box and abandoned.** `--batch-size 2` OOM'd
outright; `--batch-size 1` OOM'd too and barely moved the memory number (44.21GB → 44.10GB,
proving the cost is per-frame token count, not batch size); 720 also turns out to violate a real
architectural requirement (height/width must divide evenly by `patch_size(16) ×
bottleneck_stride(2) = 32` — `720/32=22.5`, so the decoder silently reconstructed the wrong shape
and crashed downstream instead of failing at startup). Cropping to 704 (`22×32`) plus
`--activation-checkpointing` got it to fit — but at only ~1.4GB of headroom, judged too risky for
an unattended multi-day run. **Supervisor pivoted the target resolution twice more**: 512×896
(architecturally clean, but OOM'd even with checkpointing) then **settled at 448×768** (the
supervisor's requested 448×784 also violates the same 32-divisibility rule — `784/32=24.5` — 768
was chosen as the nearest valid crop). Confirmed working with real margin: `cuda_peak_reserved`
= 34.11GB (35.44GB with `--compile`, see below), out of a ~44.42GB usable pool.

**Real, unrelated performance bug found and fixed while investigating why the GPU was reportedly
only ~57% utilized**: `nvidia-smi dmon` confirmed the pattern directly (`sm%` alternating 0/100
almost every second, ~62% average) and ruled out the dataloader first, with real evidence, before
touching any training code — `top` showed the CPU ~60% idle and individual dataloader workers
lightly loaded even at `--num-workers 8` (all of `ubuntu-gpu`'s real cores). Root cause:
`train_codec_vjepa.py`'s per-micro-step loss accumulation called `.item()` on every loss term,
every micro-step — each call forces a full CUDA sync, serializing what should be an async
pipeline. Fixed by accumulating losses as GPU tensors through the micro-step loop and converting
to Python floats once per step instead of once per term per micro-step (same final numbers, far
fewer sync points). Confirmed via `nvidia-smi dmon` again after the fix: **`sm%` sustained at
100% across every sample.** V-JEPA-track only, by decision — `train_codec.py` (DINO) has the
identical pattern but was left untouched.

**Supervisor separately asked to compile the model before the real training run** — `--compile`
added to `train_codec_vjepa.py`, wrapping just the trainable `bottleneck`/`decoder` in
`torch.compile()` (the frozen V-JEPA encoder is deliberately excluded — external, git-cloned
`facebookresearch/vjepa2` code, real risk of graph breaks on unfamiliar ops). Confirmed working
combined with `--activation-checkpointing` (a real, version-sensitive PyTorch interaction that
had never been tested together before) — no crash, no `torch._dynamo` errors, across several real
steps on GPU.

`train_codec_vjepa.py` now has the same `SIGTERM`/`SIGINT` handling `train_world_model.py` already
had (graceful checkpoint save + forced HF upload + wandb sign-off instead of an abrupt kill) —
`train_codec.py`, its un-forked DINO-track sibling, still lacks this, a pre-existing gap left as-is.
Added ahead of the real training run on the newly rented GPU box, where an unattended `timeout`
kill was judged worth guarding against rather than accepting the same risk DINO's own codec run
already ran with successfully.

**Real bug found and fixed in `codec/checkpoint.py`** (shared, so it also protects `train_codec.py`
even though DINO never triggers it): `train_codec_vjepa.py`'s new `--compile` flag wraps
`bottleneck`/`decoder` in `torch.compile()`, whose `OptimizedModule` wrapper adds a real
`"_orig_mod."` prefix to every `state_dict()` key (confirmed both locally, with a CPU
reproduction, and on real GPU: saved a checkpoint under `--compile`, then failed to load it into
`evaluate_codec_vjepa.py`'s plain, uncompiled modules — `Missing key(s) ... Unexpected key(s):
"_orig_mod.projection.weight"`). Fixed by unwrapping to the real underlying module (via
`OptimizedModule`'s own `._orig_mod` attribute) on both save and load, so every checkpoint is
always stored in one plain, portable format regardless of whether `--compile` produced or is
loading it — a no-op whenever `--compile` was never used, which is every existing DINO-track and
pre-fix V-JEPA-track checkpoint.

`scripts/evaluate_codec_vjepa.py` is done — a full fork of `evaluate_codec.py`, `VjepaModel` in
place of `DinoModel` throughout, `--config` defaulting to `configs/scaled_300m_vjepa.yaml`. Built
ahead of there being a real trained V-JEPA codec checkpoint to point it at (the real 4,000-step run
hasn't launched yet) — syntax/import-checked, not yet run against real output, since there's
nothing real to evaluate until that checkpoint exists. Also served as the real, independent proof
that the `checkpoint.py` fix above actually works: saved a checkpoint under `--compile` on GPU,
loaded it through this (non-compiled) script, got real eval numbers back instead of the
`Missing/Unexpected key(s)` crash.

**Real per-step timing measured** at the settled config (448×768, `--compile
--activation-checkpointing`, batch=2/accum=16): a clean, post-compile-warmup read via the real
filesystem timestamps of two saved preview videos, 4 steps apart — **≈94.6 sec/step, so ≈105
hours (≈4.4 days) for the full 4,000-step run.** This is the number the GPU rental decision below
is built on.

**GPU rental decision in progress**: real requirements (≥40GB VRAM with margin, bf16/Ampere-or-
newer, CUDA-only — rules out non-NVIDIA accelerators regardless of specs) plus the timing number
above were used to evaluate a real cloud GPU price list against the current A40 rental. vCPU
count and system RAM both turned out not to be differentiators (confirmed: `--num-workers 8` left
the CPU ~60% idle once the sync-stall bug above was fixed; every viable VRAM-qualifying option
already ships far more RAM than the ~18GB actually observed in use). Recommendation: A100 80GB —
cheaper per hour *and* faster *and* more VRAM than the A40 already tested, beating it on every
axis with no tradeoff. Decision handed to the supervisor; not yet acted on.

What's still ahead: a deferred overfit-one-clip convergence check at 448×768 specifically (the
existing ~83%-loss-drop proof was at 288×512, a different resolution) — queued for just before
the real launch, not done yet. Then, once a GPU is chosen and the real 4,000-step run produces a
checkpoint: `compute_latent_stats.py` needs the same V-JEPA fork (not before — nothing to compute
stats from yet); then `train_world_model_vjepa.py`, forked the same way; then a full codec retrain
from scratch under V-JEPA's feature space (the existing checkpoint can't be reused — the whole
representation changes), then a world-model retrain on top. Three real methodology questions still
need a decision before any final numbers count as comparable: whether both tracks get scored by
the same fixed judge rather than each by its own backbone, what step budget each track gets, and
whether hyperparameters stay identical across both.

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
| `src/mini_mira/codec/logging_utils.py` | Optional wandb logging, shared by both codec tracks — preview videos encode at `crf=18` explicitly (found and fixed for real: no explicit value meant falling back to ffmpeg's own default, visibly compressed — confirmed by comparing a lossless raw frame against the same frame through this encoder) |
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
| `scripts/train_codec_vjepa.py` | Same, V-JEPA track — full fork, `VjepaModel` in place of `DinoModel` |
| `scripts/verify_codec_training_vjepa.py` | Mechanism proof the V-JEPA-track codec trains (synthetic data, no GPU needed) |
| `scripts/overfit_one_clip_vjepa.py` | Real-data, real-GPU diagnostic: overfit one real clip, wandb only, no checkpoints |
| `scripts/reconstruct.py` | Mechanism smoke test: runs a video through the codec (random-init weights) |
| `scripts/evaluate_codec.py` | Real quantitative eval of a trained codec checkpoint on held-out data |
| `scripts/evaluate_codec_vjepa.py` | Same, V-JEPA track — full fork, `VjepaModel` in place of `DinoModel` |
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
- `train_world_model.py`'s `--resume` fast-forwards past already-consumed batches by calling
  `next()` on a freshly-built loader — only correct because that loader is single-process,
  unseeded (deterministic by default). The codec scripts (`train_codec.py`/`train_codec_vjepa.py`)
  have no such mechanism, so raising `--num-workers` there is safe (`--num-workers 6` is now the
  default, matching this project's real GPU box's 8 real CPU cores — recheck against `nproc` on
  different hardware); doing the same for `train_world_model.py` would need that resume mechanism
  addressed first, not done unprompted.

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
