"""The codec training loss: L1 + LPIPS + DINO latent-consistency, with auto_weight adaptive
balancing. Matches real mira's CodecLoss: same three terms, same LPIPS net, same frame-subsampling
trick, same auto_weight math, and the same multi-layer DINO consistency term when the bound
DinoModel(s) are configured for it (see dino.py's last_layer_only/layer_indices).

The codec is a deterministic autoencoder, not a VAE -- no ELBO, no KL divergence anywhere.

Pixel-range convention (matches mira/src/mira/codec/codec_model.py): raw video is [0,1];
normalize_video maps it to [-1,1] -- the space the decoder's tanh output and the reconstruction
loss both live in. denormalize_for_dino maps back to [0,1] whenever DINO needs to look at an image.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from mini_mira.codec.dino import DinoModel


def normalize_video(x: Tensor) -> Tensor:
    """[0, 1] -> [-1, 1]. The space the decoder's tanh output and the loss target both live in."""
    return (x - 0.5) / 0.5


def denormalize_for_dino(x: Tensor) -> Tensor:
    """[-1, 1] -> [0, 1]. DINO's own preprocessing expects [0, 1]."""
    return (x + 1) / 2


def _layer_averaged_mse(pred_features: Tensor | list[Tensor], target_features: Tensor | list[Tensor]) -> Tensor:
    """Normalize (direction, not magnitude) then MSE per DINO layer, averaged across layers --
    matches mira's DinoPerceptualLoss.forward exactly. Single-tensor inputs (the default,
    last-layer-only case) reduce to one term, so this is a strict superset of the old behavior.
    """
    if not isinstance(pred_features, list):
        pred_features, target_features = [pred_features], [target_features]
    layer_terms = [
        F.mse_loss(F.normalize(p, dim=2, eps=1e-6), F.normalize(t, dim=2, eps=1e-6))
        for p, t in zip(pred_features, target_features)
    ]
    return torch.stack(layer_terms).mean()


def _select_target_features(
    features: Tensor | list[Tensor], t_idx: Tensor, start: int, stop: int
) -> Tensor | list[Tensor]:
    """Select/reshape/chunk/detach the encoder's own DINO features for the self-consistency
    target, per layer -- matches mira's `real_lc = tuple(f[:, t_lc].detach() for f in
    model_outputs.dino_features)`. `features` is a single Tensor in the old single-layer case.
    """

    def _one(f: Tensor) -> Tensor:
        selected = rearrange(f[:, t_idx], "b t c h w -> (b t) c h w")
        return selected[start:stop].unsqueeze(0).detach()

    if isinstance(features, list):
        return [_one(f) for f in features]
    return _one(features)


def calculate_adaptive_weight(
    anchor_loss: Tensor, other_loss: Tensor, last_layer: Tensor, max_weight: float = 1e4
) -> Tensor:
    """VQ-GAN-style adaptive weight (Esser et al., arXiv:2012.09841 S3.3), ported from mira's
    calculate_adaptive_weight. Rescales other_loss so its gradient at `last_layer` matches
    anchor_loss's gradient there in size, so it neither dominates nor gets drowned out. Detached:
    scales the loss but carries no gradient of its own.
    """
    # These probes happen before the training loop's GradScaler sees loss_total. Scale both
    # equally under FP16 autocast to prevent their small last-layer gradients from underflowing;
    # the common factor cancels in the norm ratio.
    probe_scale = (
        65536.0
        if torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.float16
        else 1.0
    )
    anchor_grads = torch.autograd.grad(anchor_loss * probe_scale, last_layer, retain_graph=True)[0]
    other_grads = torch.autograd.grad(other_loss * probe_scale, last_layer, retain_graph=True)[0]
    return (anchor_grads.norm() / (other_grads.norm() + 1e-6)).clamp(0.0, max_weight).detach()


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
    # (b, t, dino_dim, h', w') -- encoder-side, from the raw [0,1] video. A list when the
    # encoder's DinoModel is multi-layer (one tensor per aggregated layer), matching mira's own
    # RAEEncoderOutputs.dino_features.
    dino_features: Tensor | list[Tensor]


@dataclass
class CodecLossWeights:
    """A term is active iff its weight is > 0 (matches mira's own convention)."""

    loss_mae: float = 1.0
    loss_lpips_perceptual: float = 1.0
    lpips_perceptual_frame_frac: float = 0.25
    loss_dino_latent_consistency: float = 1.0
    dino_latent_consistency_frame_frac: float = 0.25

    # Adaptive rescaling of the two perceptual terms against loss_mae -- see
    # calculate_adaptive_weight. Off by default (matches mira's own field default); the actual
    # training call sites turn it on, matching mira's shipped config.
    auto_weight: bool = False
    max_auto_weight: float = 1e4


class CodecLoss(nn.Module):
    """Computes the codec's reconstruction loss terms and their weighted total.

    The DINO-consistency term needs the encoder's frozen DINO backbone to look at the
    reconstruction; bind it after construction via bind_encoder_dino (mirrors mira's pattern of
    sharing the encoder's already-loaded DINO rather than loading a second copy).
    """

    def __init__(
        self, weights: CodecLossWeights, use_checkpointing: bool = False,
        perceptual_chunk_size: int = 0,
    ):
        super().__init__()
        self.weights = weights
        # Only the DINO call below needs it: it's the one that backprops through the whole
        # backbone (onto the reconstruction), unlike the encoder's own no_grad call.
        self.use_checkpointing = use_checkpointing
        self.perceptual_chunk_size = perceptual_chunk_size

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
        self.perceptual_dino: DinoModel | None = None  # optional, see bind_perceptual_dino
        self._last_layer: Tensor | None = None  # bound post-init, see bind_last_layer

        # Per-term real gradient norms, populated during backward() -- see _hook_clone. Read
        # after backward(), not before: empty until then. Matches mira's own backward_metrics.
        self.backward_metrics: dict[str, Tensor] = {}

    def _hook_clone(self, tensor: Tensor, loss_name: str) -> Tensor:
        """Clone `tensor` and register a backward hook recording its gradient norm under
        `loss_name` in backward_metrics -- lets training watch each term's real gradient
        magnitude directly, not just auto_weight's derived ratio. Matches mira's own
        CodecLoss._hook_clone exactly, including its dim=-1 norm convention."""
        if not tensor.requires_grad:
            return tensor
        clone = tensor.clone()
        clone.register_hook(
            lambda grad: self.backward_metrics.update({loss_name: grad.data.norm(p=2, dim=-1).mean()})
        )
        return clone

    def bind_encoder_dino(self, dino: DinoModel) -> None:
        """Share the encoder's already-loaded, frozen DINO backbone (no second copy loaded)."""
        self.dino = dino

    def bind_perceptual_dino(self, dino: DinoModel) -> None:
        """Optional: score the consistency term in a different (typically smaller) DINO's feature
        space instead of the encoder's own -- e.g. ViT-S instead of ViT-B, since this is the one
        DINO call each step that backprops (see forward below), so a smaller model here cuts real
        compute/memory without touching the encoder or the bottleneck's input width at all.

        Not bound by default: forward() falls back to the encoder's own DINO and its
        already-computed target features (outputs.dino_features), matching mira's shipped config.
        """
        self.perceptual_dino = dino

    def bind_last_layer(self, param: Tensor) -> None:
        """Share the decoder's last-layer weight, needed only when weights.auto_weight is True."""
        self._last_layer = param

    def forward(self, outputs: CodecOutputs) -> dict[str, Tensor]:
        self.backward_metrics.clear()
        predicted = outputs.output_video.float()  # [-1, 1]
        # Hooked once here too, matching mira: the three per-term hooks below all clone from this
        # already-hooked tensor, so this one captures the combined gradient from all three loss
        # terms together, not just one of them.
        predicted = self._hook_clone(predicted, "loss_total_video")
        target = outputs.input_video.float()  # [-1, 1]

        loss: dict[str, Tensor] = {}

        if self.weights.loss_mae > 0:
            loss["loss_mae"] = F.l1_loss(self._hook_clone(predicted, "loss_mae"), target)

        t_total = predicted.shape[1]

        if self.lpips_perceptual_loss is not None:
            # Only a random frame subset scored per step (mira's cost-control trick): cheaper
            # than every frame, and different random subsets across steps still cover everything.
            k = max(1, round(t_total * self.weights.lpips_perceptual_frame_frac))
            t_idx = torch.randperm(t_total, device=predicted.device)[:k].sort().values
            predicted_lpips = self._hook_clone(predicted, "loss_lpips_perceptual")
            pred_2d = rearrange(predicted_lpips[:, t_idx], "b t c h w -> (b t) c h w")
            tgt_2d = rearrange(target[:, t_idx], "b t c h w -> (b t) c h w")
            chunk_size = self.perceptual_chunk_size or pred_2d.shape[0]
            lpips_terms = []
            for start in range(0, pred_2d.shape[0], chunk_size):
                stop = min(start + chunk_size, pred_2d.shape[0])
                weight = (stop - start) / pred_2d.shape[0]
                lpips_terms.append(weight * self.lpips_perceptual_loss(pred_2d[start:stop], tgt_2d[start:stop]))
            loss["loss_lpips_perceptual"] = torch.stack(lpips_terms).sum()

        if self.weights.loss_dino_latent_consistency > 0:
            assert self.dino is not None, "call bind_encoder_dino before forward"
            consistency_dino = self.perceptual_dino if self.perceptual_dino is not None else self.dino
            k = max(1, round(t_total * self.weights.dino_latent_consistency_frame_frac))
            t_idx = torch.randperm(t_total, device=predicted.device)[:k].sort().values
            # No no_grad here: needs to backprop from DINO's features on the reconstruction,
            # through the decoder and bottleneck. Target side is already detached.
            predicted_dino = self._hook_clone(predicted, "loss_dino_latent_consistency")
            pred_frames = rearrange(predicted_dino[:, t_idx], "b t c h w -> (b t) c h w")
            target_frames = rearrange(target[:, t_idx], "b t c h w -> (b t) c h w")
            chunk_size = self.perceptual_chunk_size or pred_frames.shape[0]
            dino_terms = []
            for start in range(0, pred_frames.shape[0], chunk_size):
                stop = min(start + chunk_size, pred_frames.shape[0])
                dino_input = denormalize_for_dino(pred_frames[start:stop]).unsqueeze(0)
                if self.use_checkpointing:
                    pred_features = checkpoint(consistency_dino.dino_forward, dino_input, use_reentrant=False)
                else:
                    pred_features = consistency_dino.dino_forward(dino_input)

                if self.perceptual_dino is not None:
                    target_input = denormalize_for_dino(target_frames[start:stop]).unsqueeze(0)
                    with torch.no_grad():
                        target_features = consistency_dino.dino_forward(target_input)
                else:
                    target_features = _select_target_features(outputs.dino_features, t_idx, start, stop)

                weight = (stop - start) / pred_frames.shape[0]
                dino_terms.append(weight * _layer_averaged_mse(pred_features, target_features))
            loss["loss_dino_latent_consistency"] = torch.stack(dino_terms).sum()

        if (
            self.weights.auto_weight
            and self._last_layer is not None
            and "loss_mae" in loss
            and loss["loss_mae"].requires_grad
            and torch.is_grad_enabled()
        ):
            for name in ("loss_lpips_perceptual", "loss_dino_latent_consistency"):
                if name not in loss or not loss[name].requires_grad:
                    continue
                factor = calculate_adaptive_weight(
                    loss["loss_mae"], loss[name], self._last_layer, self.weights.max_auto_weight
                )
                loss[name] = factor * loss[name]
                loss[f"{name}_auto_w"] = factor  # logged, not summed (not a real dict key below)

        weighted = [
            getattr(self.weights, name) * value
            for name, value in loss.items()
            if hasattr(self.weights, name)  # excludes the *_auto_w diagnostic entries above
        ]
        # weighted is non-empty for any sensible config (loss_mae is on by default); guard the
        # degenerate all-zero-weights case so it returns a finite zero rather than raising on
        # stack, matching mira's own guard.
        loss["loss_total"] = torch.stack(weighted).sum() if weighted else predicted.new_zeros(())
        return loss
