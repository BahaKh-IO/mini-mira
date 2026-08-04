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
learned positional embeddings; the world model additionally conditions every block on the
diffusion timestep `tau` via adaptive LayerNorm (AdaLN).

## Project layout

| File | Contents |
|---|---|
| `src/mini_mira/bottleneck.py` | `StridedConvBottleneckConfig`, `MyBottleneck` — the encoder-side strided-conv projection into the latent |
| `src/mini_mira/decoder.py` | `ViTDecoderConfig`, `ViTVideoDecoder`, `PatchUnembed` — the space-time ViT decoder |
| `src/mini_mira/world_model.py` | `LatentWorldModelConfig`, `DiffusionTransformer` — the AdaLN-conditioned diffusion transformer |
| `src/mini_mira/blocks.py` | Shared building blocks: `LayerScale`, `SwiGLU`, `SelfAttention`, `SpaceTimeBlock` (decoder), `AdaptiveLayerNorm`, `AdaSTBlock` (world model) |
| `src/mini_mira/rope.py` | Rotary position embeddings — one shared implementation (temporal + axial spatial), used by both the decoder and the world model |
| `src/mini_mira/timestep_encoder.py` | `DiffusionTimeEmbedding` — sinusoidal embedding of the diffusion timestep `tau` |
| `src/mini_mira/pipeline.py` | `PipelineConfig`, `MyPipeline` — wires bottleneck → multi-step diffusion loop → decoder into one forward pass |
| `scripts/test_shapes.py` | Shape-correctness checks for every stage of the pipeline |
| `scripts/verify_rope.py` | Behavioral checks: causal masking, position sensitivity, and `tau` sensitivity |

At the default configuration, the components have the following parameter counts:

| Component | Parameters |
|---|---|
| Bottleneck | 262,176 |
| Decoder | 4,893,952 |
| World model | 6,184,224 |
| **Total** | **11,340,352** |

## Scope

**Implemented:**
- Strided-conv bottleneck (encoder) and ViT space-time decoder, matching the real codec's shape
  contract and space/time attention factorization.
- Rotary position embeddings (temporal + axial 2D spatial).
- AdaLN conditioning of the world model on the diffusion timestep `tau`, via a sinusoidal
  timestep embedding matched to the real implementation.
- Flow-matching sampling: a multi-step Euler integration loop from pure noise to a decoded video.

**Deliberately simplified or not yet implemented**, each a disclosed decision rather than a gap
found later:
- **Action conditioning.** `actions` is threaded through every relevant signature but not yet
  encoded or used — the model cannot currently distinguish different action sequences.
- **Clean-past / diffusion-forcing conditioning.** No mechanism yet lets a denoising step
  condition on the clean latents of previous frames.
- **Streaming inference.** No KV-cache; every diffusion step recomputes the whole sequence.
- **The frozen visual encoder.** The pipeline accepts DINOv3-shaped feature tensors as input; it
  does not load or run a real DINOv3 backbone.
- **Grouped-query attention.** Attention here always uses as many KV heads as query heads.
- **Training.** No loss functions, optimizer, or data loading exist. Every weight is randomly
  initialized; this project verifies architecture and shapes, not learned behavior.
- **One shared implementation where the real repo has two.** The real codec and world model
  never share code with each other, even for identical logic (e.g. two separately-named but
  functionally identical MLP classes, two separate RoPE implementations). mini_mira
  consolidates the shared pieces into `blocks.py` and `rope.py` instead.

## Verification

Shape correctness alone doesn't prove a mechanism works — a broken RoPE implementation or a
zeroed-out conditioning signal can still produce the right-shaped output. This project checks
both:

- **`scripts/test_shapes.py`** — asserts every stage's output shape against the real codec's
  actual configuration (`(2, 40, 1024, 18, 32)` DINO-shaped input through to `(2, 40, 3, 288,
  512)` decoded video).
- **`scripts/verify_rope.py`** — targeted behavioral checks, each aimed at a specific silent
  failure mode:
  - *Causality*: perturbing only the last frame of a clip must not change any earlier frame's
    output.
  - *Position sensitivity*: swapping two input frames must not simply swap the output back — a
    degenerate (all-zero or all-identical) set of rotary frequencies would do exactly that.
  - *`tau` sensitivity*: the same latent with two different `tau` values must produce different
    predicted velocities; the same latent with identical `tau` values must produce identical
    output.

## Getting started

Requirements: Python ≥ 3.10, `torch`, `einops`. There is no packaging configuration — the
scripts add `src/` to `sys.path` directly.

```
python scripts/test_shapes.py
python scripts/verify_rope.py
```

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
decoupled from the decoder and world model, RoPE wired into both, and AdaLN conditioning on
`tau` wired into the world model. Action conditioning and clean-past conditioning are the next
pieces of the forward pass to implement.
