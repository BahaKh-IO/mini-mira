"""The full world-model eval suite: Frechet DINO/Inception Distance, PSNR, LPIPS, SSIM.

Ported near-verbatim from real mira's mira/src/mira/training/metrics/{image_metrics,frechet}.py --
same formulas, same OnlineGaussian sufficient-statistics accumulation, same scipy.linalg.sqrtm-based
Frechet distance -- minus torch.distributed.all_reduce() (mini_mira never runs distributed) and
minus a second DINOv3 instance (see FullEvalMetrics below).

This is the heavier tier on top of eval_metrics.py's always-on drift metrics -- see that module's
docstring and notes/deviations.md entry 1.19 for the scope-cut history and its reversal.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.functional.image import structural_similarity_index_measure

from mini_mira.world_model.eval_metrics import RunningMean
from mini_mira.world_model.latent_world_model import _keep_convolutions_in_bf16

INCEPTION_FID_DIM = 2048


def compute_psnr(pred: Tensor, target: Tensor) -> Tensor:
    """pred, target: (b,t,c,h,w) in [0,1]. Returns (b,t) raw per-frame PSNR (matches real mira's
    PSNRMetric.update exactly)."""
    mse = torch.nn.functional.mse_loss(pred, target, reduction="none").mean(dim=(2, 3, 4))
    return -10 * torch.log10(mse)


def compute_lpips(lpips_fn, pred: Tensor, target: Tensor) -> Tensor:
    """pred, target: (b,t,c,h,w) in [0,1]. Returns (b*t,) raw per-frame LPIPS scores. Per-video
    loop (not one big batched call) to avoid OOM on large inputs, matching real mira's
    DistributedLPIPS.update exactly; lpips.LPIPS expects [-1,1] inputs."""
    batch = pred.shape[0]
    scores = []
    for b in range(batch):
        scores.append(lpips_fn(pred[b] * 2 - 1, target[b] * 2 - 1).flatten())
    return torch.cat(scores)


def compute_ssim(pred: Tensor, target: Tensor) -> Tensor:
    """pred, target: (b,t,c,h,w) in [0,1]. Returns (b,) raw per-video SSIM -- torchmetrics'
    structural_similarity_index_measure already averages over t frames internally, so this is a
    per-video mean, not per-frame; processed per-video (not one big (b*t) batch) to stay within
    CUDA's 32-bit index limits at high resolution, matching real mira's DistributedSSIM exactly."""
    batch = pred.shape[0]
    scores = []
    for b in range(batch):
        ssim_mean = structural_similarity_index_measure(pred[b], target[b], data_range=1.0)
        assert isinstance(ssim_mean, Tensor)
        scores.append(ssim_mean)
    return torch.stack(scores)


class OnlineGaussian(nn.Module):
    """Online estimator of a Gaussian's mean and covariance from sufficient statistics. Ported
    verbatim from real mira's image_metrics.py, minus all_reduce() (single-process only)."""

    sum_x: Tensor
    sum_xxT: Tensor
    n: Tensor

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("sum_x", torch.zeros((dim,), dtype=torch.double))
        self.register_buffer("sum_xxT", torch.zeros((dim, dim), dtype=torch.double))
        self.register_buffer("n", torch.zeros((), dtype=torch.long))

    def reset(self) -> None:
        self.sum_x.zero_()
        self.sum_xxT.zero_()
        self.n.zero_()

    def update(self, x: Tensor) -> None:
        """x: (b, dim)."""
        x = x.to(dtype=torch.double)
        self.n += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_xxT += x.T @ x

    def compute(self, unbiased: bool = True, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        assert self.n > 1, "Need at least 2 samples to compute statistics."
        n = self.n.to(dtype=torch.double)
        mean = self.sum_x / n
        cov_num = self.sum_xxT - n * torch.outer(mean, mean)
        denom = (n - 1.0) if unbiased else n
        cov = cov_num / denom
        if eps > 0:
            cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        return mean, cov


def frechet_distance(mean1: Tensor, cov1: Tensor, mean2: Tensor, cov2: Tensor) -> float:
    """Frechet distance between two Gaussians given their means and covariances. Ported verbatim
    from real mira's image_metrics.py (scipy.linalg.sqrtm-based -- no easy Torch equivalent)."""
    from scipy.linalg import sqrtm  # noqa: PLC0415 -- optional dep, only imported when used

    mean1_np, mean2_np = mean1.cpu().numpy(), mean2.cpu().numpy()
    cov1_np, cov2_np = cov1.cpu().numpy(), cov2.cpu().numpy()

    diff = mean1_np - mean2_np
    cov_prod = sqrtm(cov1_np.dot(cov2_np))
    cov_prod = np.asarray(cov_prod)
    if np.iscomplexobj(cov_prod):
        cov_prod = np.real(cov_prod)

    distance = diff.dot(diff) + np.trace(cov1_np + cov2_np - 2 * cov_prod)
    return float(distance)


def _pool(gaussians: list[OnlineGaussian]) -> OnlineGaussian:
    """Sum per-slice sufficient statistics into one Gaussian over all slices' frames."""
    pooled = OnlineGaussian(dim=gaussians[0].sum_x.shape[0]).to(gaussians[0].sum_x.device)
    for g in gaussians:
        pooled.sum_x += g.sum_x
        pooled.sum_xxT += g.sum_xxT
        pooled.n += g.n
    return pooled


def _frechet(target: OnlineGaussian, pred: OnlineGaussian) -> float:
    t_mean, t_cov = target.compute()
    p_mean, p_cov = pred.compute()
    return frechet_distance(t_mean, t_cov, p_mean, p_cov)


class SlicedFrechetMetric(nn.Module):
    """Frechet distance between a target and predicted feature distribution, split into per-slice
    windows of the unrolled prediction plus the aggregate over all slices. Ported verbatim from
    real mira's frechet.py, minus all_reduce()."""

    def __init__(self, dim: int, num_slices: int):
        super().__init__()
        self.target = nn.ModuleList(OnlineGaussian(dim) for _ in range(num_slices))
        self.pred = nn.ModuleList(OnlineGaussian(dim) for _ in range(num_slices))

    def reset(self) -> None:
        for g in list(self.target) + list(self.pred):
            g.reset()  # type: ignore[union-attr]

    def update(self, slice_idx: int, target_feats: Tensor, pred_feats: Tensor) -> None:
        self.target[slice_idx].update(target_feats)  # type: ignore[union-attr]
        self.pred[slice_idx].update(pred_feats)  # type: ignore[union-attr]

    def compute(self) -> tuple[float, list[float]]:
        """Returns (aggregate over all slices, per-slice curve). Pools raw sufficient statistics
        across slices BEFORE calling .compute() on the pooled Gaussian -- pooling the already-
        computed per-slice means/covariances instead would double-count."""
        targets = list(self.target)
        preds = list(self.pred)
        aggregate = _frechet(_pool(targets), _pool(preds))  # type: ignore[arg-type]
        per_slice = [_frechet(t, p) for t, p in zip(targets, preds)]  # type: ignore[arg-type]
        return aggregate, per_slice


class FullEvalMetrics(nn.Module):
    """Bundles the Frechet DINO/Inception Distance + PSNR/LPIPS/SSIM trackers for one eval run.

    Reuses the LatentWorldModel's OWN dino instance for DINO Frechet (see compute_full_eval_metrics
    below) rather than loading a second DINOv3 backbone the way real mira's WorldModelMetrics does
    (its DinoForMetrics is separate because real mira's LatentWorldModel/codec split is different) --
    saves real VRAM, matters given the eval-time OOM concern that shaped this whole feature.
    """

    def __init__(
        self,
        dino_dim: int,
        fdd_slice_frames: int,
        num_slices: int,
        device: str | torch.device,
        inception: nn.Module | None = None,
        lpips_fn: nn.Module | None = None,
        dino_fdd_slice_frames: int | None = None,
    ):
        super().__init__()
        self.fdd_slice_frames = fdd_slice_frames
        self.num_slices = num_slices
        # real_dino_gen/pred_dino_gen (compute_full_eval_metrics's dino_temporal_scale-corrected
        # slice) is the SAME total length as real_video_gen/pred_video_gen only when
        # dino_temporal_scale == temporal_downsampling (true for DinoModel, tubelet_size=1) --
        # for a time-halving encoder (VjepaModel), it's proportionally SHORTER (e.g. 14 vs 28 at
        # this project's real V-JEPA config: dino_temporal_scale=1, temporal_downsampling=2). Using
        # the same fdd_slice_frames-wide window for both tensors runs the later slices past the end
        # of the shorter one, leaving them with zero samples -- OnlineGaussian.compute()'s own
        # `assert self.n > 1` then crashes the whole eval (confirmed for real: this exact crash,
        # on a real V-JEPA run, is what this parameter was added to fix). Defaults to None ->
        # falls back to fdd_slice_frames, byte-identical to this class's original behavior for any
        # caller that doesn't pass it (i.e. train_world_model.py needs zero changes).
        self.dino_fdd_slice_frames = dino_fdd_slice_frames if dino_fdd_slice_frames is not None else fdd_slice_frames

        # inception/lpips_fn: injection seam for testing, same precedent as LatentWorldModel's
        # own `dino: nn.Module | None = None` -- default None builds the real frozen nets.
        if inception is None:
            from pytorch_fid.inception import InceptionV3  # noqa: PLC0415 -- optional dep

            inception_block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[INCEPTION_FID_DIM]
            inception = InceptionV3([inception_block_idx])
            inception.requires_grad_(False)
            inception.eval()
        if lpips_fn is None:
            import lpips as lpips_lib  # noqa: PLC0415 -- optional dep

            lpips_fn = lpips_lib.LPIPS(net="alex")
            lpips_fn.requires_grad_(False)

        # Same treatment already proven for bottleneck/decoder (notes/gpu_amp_investigation.md):
        # these are new conv-heavy frozen nets with no protection against the V100 cuDNN crash
        # this project already hit once (fp16 AND fp32 failed for a specific conv shape, only
        # bf16 worked). Harmless if the eval GPU never hits this -- bf16-forcing a conv that
        # already works under fp16 is just a redundant, inert precision choice.
        _keep_convolutions_in_bf16(inception)
        _keep_convolutions_in_bf16(lpips_fn)
        self.inception = inception.to(device)
        self.lpips_fn = lpips_fn.to(device)

        self.dino_frechet = SlicedFrechetMetric(dino_dim, num_slices).to(device)
        self.inception_frechet = SlicedFrechetMetric(INCEPTION_FID_DIM, num_slices).to(device)
        self.psnr = RunningMean()
        self.lpips = RunningMean()
        self.ssim = RunningMean()

    @torch.no_grad()
    def update(
        self, real_video_gen: Tensor, pred_video_gen: Tensor, real_dino_gen: Tensor, pred_dino_gen: Tensor
    ) -> None:
        """real/pred_video_gen: (b,t,3,h,w) in [0,1]. real/pred_dino_gen: (b,t,dino_dim,h',w').
        All four already sliced to the GENERATED region only (see compute_full_eval_metrics)."""
        self.psnr.update(compute_psnr(pred_video_gen, real_video_gen))
        self.lpips.update(compute_lpips(self.lpips_fn, pred_video_gen, real_video_gen))
        self.ssim.update(compute_ssim(pred_video_gen, real_video_gen))

        for s in range(self.num_slices):
            dino_fs = slice(s * self.dino_fdd_slice_frames, (s + 1) * self.dino_fdd_slice_frames)
            self.dino_frechet.update(
                s,
                real_dino_gen[:, dino_fs].mean(dim=(-1, -2)).flatten(0, 1),
                pred_dino_gen[:, dino_fs].mean(dim=(-1, -2)).flatten(0, 1),
            )

            fs = slice(s * self.fdd_slice_frames, (s + 1) * self.fdd_slice_frames)
            real_slice, pred_slice = real_video_gen[:, fs], pred_video_gen[:, fs]
            bs, ts = real_slice.shape[:2]
            real_flat = real_slice.reshape(bs * ts, *real_slice.shape[2:])
            pred_flat = pred_slice.reshape(bs * ts, *pred_slice.shape[2:])
            both = self.inception(torch.cat([pred_flat, real_flat], dim=0))[0]
            both = both.squeeze(-1).squeeze(-1)
            pred_inception, real_inception = both.chunk(2, dim=0)
            self.inception_frechet.update(s, real_inception, pred_inception)

    def compute_and_reset(self) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Returns (scalars, curves) -- scalars: psnr/lpips/ssim/frechet_dino_distance/
        frechet_inception_distance aggregates. curves: {"fdd": [...], "fid": [...]} per-slice
        values, one entry per Frechet slice -- log these as plain scalars keyed by unroll length
        (e.g. fdd_at_{(i+1)*fdd_slice_frames}) at the call site, no plotly."""
        dino_agg, dino_curve = self.dino_frechet.compute()
        inception_agg, inception_curve = self.inception_frechet.compute()
        scalars: dict[str, float] = {
            "psnr": self.psnr.compute(),
            "lpips": self.lpips.compute(),
            "ssim": self.ssim.compute(),
            "frechet_dino_distance": dino_agg,
            "frechet_inception_distance": inception_agg,
        }
        curves = {"fdd": dino_curve, "fid": inception_curve}

        self.dino_frechet.reset()
        self.inception_frechet.reset()
        self.psnr = RunningMean()
        self.lpips = RunningMean()
        self.ssim = RunningMean()

        return scalars, curves


def compute_full_eval_metrics(
    real_video: Tensor, pred_video: Tensor, real_dino: Tensor, pred_dino: Tensor,
    n_context_latents: int, temporal_downsampling: int, metrics: FullEvalMetrics,
    dino_temporal_scale: int | None = None,
) -> None:
    """Given the (real_video, pred_video, real_dino, pred_dino) already produced by
    eval_metrics.decode_and_dino -- computed once by the caller and shared with
    compute_drift_metrics and rollout-video rendering, not redone here -- slices to the generated
    region and updates `metrics`'s running accumulators in place.

    dino_temporal_scale: same meaning as compute_drift_metrics's own parameter of the same name --
    see that function's docstring. Only applies to the real_dino/pred_dino slice: real_video/
    pred_video genuinely are in video-frame units for every encoder (the decoder's own upsample
    always reconstructs the real raw frame count), so temporal_downsampling stays correct for
    those regardless. Defaults to None -> falls back to temporal_downsampling, byte-identical to
    this function's own original behavior.
    """
    if dino_temporal_scale is None:
        dino_temporal_scale = temporal_downsampling
    n_context_frames = n_context_latents * temporal_downsampling
    n_context_dino_frames = n_context_latents * dino_temporal_scale
    metrics.update(
        real_video[:, n_context_frames:],
        pred_video[:, n_context_frames:],
        real_dino[:, n_context_dino_frames:],
        pred_dino[:, n_context_dino_frames:],
    )
