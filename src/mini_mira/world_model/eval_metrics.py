"""Lightweight autoregressive-rollout "drift" metrics for world-model training: how fast the
rollout's DINO features and raw latents diverge from ground truth, frame by frame. This is the
cheap, always-on tier -- mini_mira.world_model.full_eval_metrics (Frechet DINO/Inception Distance,
PSNR, LPIPS, SSIM) and mini_mira.world_model.rollout_visualization (rendered rollout videos) are
the heavier tier, both consuming the SAME model.rollout(...) call AND the same decode_and_dino(...)
call this module provides -- see scripts/train_world_model.py's run_full_eval, which calls
rollout() and decode_and_dino() exactly once per eval batch and feeds their output to all three,
rather than each of the three decoding/DINO-ing independently. (notes/deviations.md entry 1.19
previously described the full suite as deliberately unbuilt for cost reasons; that decision has
since been reversed.)

RunningMean mirrors real mira's DistributedMetric contract (update(values)/compute()) minus its
torch.distributed.all_reduce() -- mini_mira never runs distributed, so that machinery is omitted
entirely rather than ported-then-stripped.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


class RunningMean:
    """Plain single-process running mean over however many values update() has seen so far."""

    def __init__(self) -> None:
        self._sum = torch.zeros((), dtype=torch.float64)
        self._n = 0

    def update(self, values: Tensor) -> None:
        self._sum += values.detach().double().sum().cpu()
        self._n += values.numel()

    def compute(self) -> float:
        if self._n == 0:
            return 0.0
        return (self._sum / self._n).item()


def decode_and_dino(model, z: Tensor, z_t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The one decode + DINO pass every eval tier needs (compute_drift_metrics,
    full_eval_metrics.compute_full_eval_metrics, and rollout-video rendering) -- factored out so
    the caller (train_world_model.py's run_full_eval) runs it exactly once per eval batch and
    shares the result, instead of each of those three redoing it independently.

    Returns (real_video, pred_video, real_dino, pred_dino), the last two already reduced to the
    last DINO layer (model.dino is multi-layer, last_layer_only=False, for the bottleneck's own
    encoding needs -- same last-layer convention loss.py's training-time _layer_averaged_mse
    deliberately does NOT use, a different purpose)."""
    with torch.no_grad():
        real_video = model.decode_to_video(z)
        pred_video = model.decode_to_video(z_t)
        real_dino = model.dino.dino_forward(real_video)
        pred_dino = model.dino.dino_forward(pred_video)
        if isinstance(real_dino, list):
            real_dino = real_dino[-1]
        if isinstance(pred_dino, list):
            pred_dino = pred_dino[-1]
    return real_video, pred_video, real_dino, pred_dino


def compute_drift_metrics(
    z: Tensor, z_t: Tensor, n_context_latents: int, real_dino: Tensor, pred_dino: Tensor, temporal_downsampling: int,
    dino_temporal_scale: int | None = None,
) -> dict[str, Tensor]:
    """Given one rollout's (z, z_t) and the (real_dino, pred_dino) already produced by
    decode_and_dino -- computed once by the caller and shared with compute_full_eval_metrics and
    rollout-video rendering, not redone here -- returns three RAW per-element tensors (not
    pre-reduced) -- so RunningMean.update() weights correctly across batches of different sizes,
    matching real mira's DistributedMetric.update(values) contract:
      dino_cos_drift = 1 - cosine_similarity(pred_dino, real_dino, over the channel dim)
      dino_l2_drift  = mse(pred_dino, real_dino), averaged over the channel dim
      latent_drift   = 1 - cosine_similarity(rolled-out latents, real latents, over the channel dim)
    Only the GENERATED region (from n_context_latents onward) is scored -- the context region is
    real ground truth on both sides by construction and would trivially read as zero drift.

    dino_temporal_scale: how many entries of real_dino/pred_dino one latent frame's worth of
    context actually spans -- NOT always the same as temporal_downsampling. decode_and_dino
    re-runs model.dino.dino_forward on the DECODED video, and for an encoder with its own
    temporal reduction (e.g. VjepaModel, which halves time via its tubelet) that re-encoding
    undoes some of the round-trip: real_dino/pred_dino land back in latent-frame units (dino_
    temporal_scale=1), not video-frame units (temporal_downsampling, DinoModel's case, since
    DinoModel never touches time at all). Defaults to None -> falls back to temporal_downsampling,
    byte-identical to this function's own original behavior -- a real caller must pass the
    correct value explicitly for any encoder with its own reduction (see train_world_model_vjepa.
    py's run_full_eval: temporal_downsampling // getattr(model.dino, "tubelet_size", 1)).
    """
    if dino_temporal_scale is None:
        dino_temporal_scale = temporal_downsampling
    # n_context_latents is in LATENT-frame units; real_dino/pred_dino are scaled by
    # dino_temporal_scale relative to that (see this function's own docstring above for why that
    # can differ from temporal_downsampling) -- scale the offset before slicing either.
    n_context_frames = n_context_latents * dino_temporal_scale
    real_dino_gen = real_dino[:, n_context_frames:]
    pred_dino_gen = pred_dino[:, n_context_frames:]

    dino_cos_drift = 1 - F.cosine_similarity(pred_dino_gen, real_dino_gen, dim=2)
    dino_l2_drift = F.mse_loss(pred_dino_gen, real_dino_gen, reduction="none").mean(dim=2)
    latent_drift = 1 - F.cosine_similarity(
        z_t[:, n_context_latents:], z[:, n_context_latents:], dim=2
    )

    return {
        "dino_cos_drift": dino_cos_drift,
        "dino_l2_drift": dino_l2_drift,
        "latent_drift": latent_drift,
    }
