"""The codec training loss: L1 + LPIPS + DINO latent-consistency.

Matches real mira's mira.codec.loss.CodecLoss (same three terms, same LPIPS net, same
frame-subsampling trick for cost control), simplified: mini_mira's DinoModel returns a single
Tensor (one layer), not a list, so there's one consistency term, not several averaged together --
and there is no auto_weight adaptive balancing yet (see notes/deviations.md: real mira's version
rescales LPIPS/consistency by the ratio of their gradient norms against the L1 anchor, which needs
the decoder's last layer exposed plus torch.autograd.grad bookkeeping -- deferred deliberately;
flat fixed weights, matching mira's own raw weight values, are used instead for now).

Also: mira's codec has no ELBO and no KL divergence anywhere. Its bottleneck is a deterministic
strided conv, not a VAE with reparameterization -- confirmed directly in
mira/src/mira/codec/rae_encoder.py. The only stochastic element is noise_tau, a plain train-time
Gaussian noise regularizer on the latent (disabled, 0.0, in mira's own shipped config), unrelated
to any KL term. These three reconstruction losses are the whole of it.

Pixel-range convention, matching real mira exactly (mira/src/mira/codec/codec_model.py):
raw video is [0,1]; normalize_video maps it to [-1,1] -- the space the decoder's tanh output and
the reconstruction loss both live in. denormalize_for_dino maps back to [0,1] whenever DINO needs
to look at an image (DINO's own preprocessing expects [0,1]).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from mini_mira.codec.dino import DinoModel


def normalize_video(x: Tensor) -> Tensor:
    """[0, 1] -> [-1, 1]. The space the decoder's tanh output and the loss target both live in."""
    return (x - 0.5) / 0.5


def denormalize_for_dino(x: Tensor) -> Tensor:
    """[-1, 1] -> [0, 1]. DINO's own preprocessing expects [0, 1]."""
    return (x + 1) / 2


@dataclass
class CodecOutputs:
    """What CodecLoss needs from one training step.

    input_video/output_video are both [-1, 1]. dino_features are the encoder's own (target-side)
    features, already computed on the raw [0,1] video during encoding -- reused here so the
    consistency loss doesn't run DINO on the target a second time, matching mira's
    bind_encoder_dino sharing trick.
    """

    input_video: Tensor  # (b, t, 3, h, w), [-1, 1] -- normalize_video(raw video)
    output_video: Tensor  # (b, t, 3, h, w), [-1, 1] -- decoder's raw tanh output
    dino_features: Tensor  # (b, t, dino_dim, h', w') -- encoder-side, from the raw [0,1] video


@dataclass
class CodecLossWeights:
    """A term is active iff its weight is > 0 (matches mira's own convention)."""

    loss_mae: float = 1.0
    loss_lpips_perceptual: float = 1.0
    lpips_perceptual_frame_frac: float = 0.25
    loss_dino_latent_consistency: float = 1.0
    dino_latent_consistency_frame_frac: float = 0.25


class CodecLoss(nn.Module):
    """Computes the codec's reconstruction loss terms and their weighted total.

    The DINO-consistency term needs the encoder's frozen DINO backbone to look at the
    reconstruction; bind it after construction via bind_encoder_dino (mirrors mira's pattern of
    sharing the encoder's already-loaded DINO rather than loading a second copy).
    """

    def __init__(self, weights: CodecLossWeights):
        super().__init__()
        self.weights = weights

        self.lpips_perceptual_loss: nn.Module | None = None
        if weights.loss_lpips_perceptual > 0:
            # net_type="vgg" matches mira's own choice (mira/src/mira/codec/loss.py uses
            # lpips.LPIPS(net="vgg")). torchmetrics' version loads VGG16 via torchvision.models,
            # not the separate `lpips` pip package mira uses directly -- confirmed by reading
            # torchmetrics' own source (torchmetrics/functional/image/lpips.py).
            self.lpips_perceptual_loss = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
            self.lpips_perceptual_loss.eval()
            for p in self.lpips_perceptual_loss.parameters():
                p.requires_grad = False

        self.dino: DinoModel | None = None  # bound post-init, see bind_encoder_dino

    def bind_encoder_dino(self, dino: DinoModel) -> None:
        """Share the encoder's already-loaded, frozen DINO backbone (no second copy loaded)."""
        self.dino = dino

    def forward(self, outputs: CodecOutputs) -> dict[str, Tensor]:
        predicted = outputs.output_video.float()  # [-1, 1]
        target = outputs.input_video.float()  # [-1, 1]

        loss: dict[str, Tensor] = {}

        if self.weights.loss_mae > 0:
            loss["loss_mae"] = F.l1_loss(predicted, target)

        t_total = predicted.shape[1]

        if self.lpips_perceptual_loss is not None:
            # LPIPS expects (N, 3, H, W) in [-1, 1] -- matches this module's native range already.
            # Only a random frame subset is scored per step (mira's own cost-control trick, not
            # ours): running a full VGG forward+backward on every frame every step is expensive,
            # and averaging over different random subsets across steps still covers all frames
            # over the course of training.
            k = max(1, round(t_total * self.weights.lpips_perceptual_frame_frac))
            t_idx = torch.randperm(t_total, device=predicted.device)[:k].sort().values
            pred_2d = rearrange(predicted[:, t_idx], "b t c h w -> (b t) c h w")
            tgt_2d = rearrange(target[:, t_idx], "b t c h w -> (b t) c h w")
            loss["loss_lpips_perceptual"] = self.lpips_perceptual_loss(pred_2d, tgt_2d)

        if self.weights.loss_dino_latent_consistency > 0:
            assert self.dino is not None, "call bind_encoder_dino before forward"
            k = max(1, round(t_total * self.weights.dino_latent_consistency_frame_frac))
            t_idx = torch.randperm(t_total, device=predicted.device)[:k].sort().values
            # No no_grad here: this call needs to backprop from DINO's features on the
            # reconstruction, through the decoder and bottleneck (see dino.py's dino_forward
            # docstring). The target side is already detached -- it was computed on the real
            # video by the encoder, before the decoder ever ran.
            pred_features = self.dino.dino_forward(denormalize_for_dino(predicted[:, t_idx]))
            target_features = outputs.dino_features[:, t_idx].detach()
            # L2-normalize along the channel dim before MSE: compares feature *direction*, not
            # magnitude -- matches mira's DinoPerceptualLoss(normalize=True) latent-consistency
            # variant (real mira's dim=2 is also the channel dim in its (B,T,C,H,W) layout).
            p = F.normalize(pred_features, dim=2, eps=1e-6)
            t = F.normalize(target_features, dim=2, eps=1e-6)
            loss["loss_dino_latent_consistency"] = F.mse_loss(p, t)

        weighted = [getattr(self.weights, name) * value for name, value in loss.items()]
        loss["loss_total"] = torch.stack(weighted).sum()
        return loss
