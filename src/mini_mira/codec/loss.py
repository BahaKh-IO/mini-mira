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
from torchmetrics.functional.image.lpips import _normalize_tensor, _spatial_average
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


def _lpips_layer_scores(features_pred: list[Tensor], features_target: list[Tensor], lins) -> Tensor:
    """torchmetrics' own per-scale LPIPS comparison, as one function inductor can fuse.

    Same three steps, in the same order, as `_LPIPS.forward`: unit-normalize both sides of each
    VGG scale, square the difference, then 1x1-convolve that down to one channel and average it
    spatially. _normalize_tensor and _spatial_average are imported from torchmetrics rather than
    re-derived, so the math here is literally theirs (their eps goes inside the sqrt, which is
    not what F.normalize does -- exactly the kind of detail worth not retyping).
    """
    scores = [
        _spatial_average(lin((_normalize_tensor(p) - _normalize_tensor(t)) ** 2), keep_dim=True)
        for p, t, lin in zip(features_pred, features_target, lins)
    ]
    return torch.stack(scores).sum(dim=0).squeeze().mean()


class FusedLpips(nn.Module):
    """LPIPS with the per-scale comparison compiled, and torchmetrics' per-call overhead removed.

    Borrows torchmetrics' already-loaded modules (the VGG16 trunk, the five 1x1 `lins`, the
    scaling layer) rather than defining or downloading anything of its own, so the network and its
    weights are unchanged and so is the value it returns. What changes is the machinery around it:

      - the five-scale comparison runs as fused kernels instead of ~10 eager passes over the
        ~840M feature elements a 20-frame 448x768 batch produces. Measured on an H100: 109ms ->
        10ms forward+backward, against 49ms for the two VGG passes themselves.
      - torchmetrics' input validation is gone. `_valid_img` evaluates `img.min() >= -1` and
        converts it straight to a Python bool -- a full reduction plus a hard CUDA sync, twice per
        call, stalling a pipeline that is otherwise entirely asynchronous. Both inputs here are in
        [-1, 1] by construction (a tanh decoder output, and normalize_video's own output), so the
        check could only ever pass.
      - torchmetrics' Metric.forward accumulates every call into module state that then has to be
        .reset() afterward or it grows without bound. This is a plain function of its arguments.
    """

    def __init__(self, lpips: LearnedPerceptualImagePatchSimilarity, compile_scoring: bool = True):
        super().__init__()
        inner = lpips.net  # torchmetrics' _LPIPS: scaling_layer + VGG16 trunk + the 1x1 lins
        self.scaling_layer = inner.scaling_layer
        self.features = inner.net
        self.lins = inner.lins
        self._score = torch.compile(_lpips_layer_scores) if compile_scoring else _lpips_layer_scores

    def _vgg_scales(self, image: Tensor) -> list[Tensor]:
        """VGG's five feature maps as a plain list.

        The list() is load-bearing, not tidiness. Vgg16.forward declares its output NamedTuple
        class INSIDE its own function body, so every call returns an instance of a brand-new type;
        dynamo guards a compiled function on its arguments' type ids, so handing it that tuple
        recompiles on every single call until the recompile limit trips and the whole thing
        silently falls back to eager. That cost the entire speedup this class exists for until the
        list() went in.

        The VGG trunk itself is deliberately left in eager. Compiling it is worth about 3% of a
        training step, and costs bitwise reproducibility: inductor's convolution backward
        accumulates with atomics, so two identical runs stop agreeing to the last bit. Not a trade
        worth making for 3%.
        """
        return list(self.features(self.scaling_layer(image)))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """pred/target: (n, 3, h, w) in [-1, 1]. Returns the batch-mean LPIPS distance."""
        return self._score(self._vgg_scales(pred), self._vgg_scales(target), self.lins)


def _with_gradient(source: Tensor, gradient: Tensor, value: Tensor) -> Tensor:
    """A scalar that reports `value` but differentiates as if it were `gradient . source`.

    `value` here is the real weighted total loss -- what gets logged and watched -- while
    `gradient` is the already-combined d(total)/d(source) that
    CodecLoss._apply_adaptive_weights_reusing_gradients assembled from the per-term gradients it
    had to compute anyway. Calling .backward() on the result seeds `source` with exactly
    `gradient` and continues from there, so the decoder sees the gradient it would have seen from
    value.backward() without any of the terms being backpropagated a second time. The
    subtract-its-own-detached-self is the usual way to keep a value and a gradient from two
    different expressions: the two `surrogate` terms cancel numerically and only the left one
    carries a graph.
    """
    surrogate = (source * gradient.detach()).sum()
    return surrogate - surrogate.detach() + value.detach()


def _feature_indices(t_idx: Tensor, t_pixel_total: int, feature_frames: int) -> Tensor:
    """Map pixel-space frame indices to the encoder's own (possibly temporally reduced) feature
    frames, one representative per tubelet group. Shared by both target-selection paths below --
    see _select_target_features for the full reasoning about why the dedup is needed.
    """
    reduction = max(1, t_pixel_total // feature_frames)
    if reduction <= 1:
        return t_idx
    return (t_idx // reduction).clamp(max=feature_frames - 1)[::reduction]


def _select_batched_target_features(
    features: Tensor | list[Tensor], t_idx: Tensor, t_pixel_total: int
) -> Tensor | list[Tensor]:
    """The whole batch's consistency targets in one gather, batch dim kept -- the counterpart to
    the batched prediction path in forward(). Equivalent to stacking what _select_target_features
    returns for each item when the chunk covers all selected frames, without the per-item
    slice/flatten/re-add-batch-dim round trip.
    """

    def _one(f: Tensor) -> Tensor:
        return f[:, _feature_indices(t_idx, t_pixel_total, f.shape[1])].detach()

    if isinstance(features, list):
        return [_one(f) for f in features]
    return _one(features)


def _select_target_features(
    features: Tensor | list[Tensor], t_idx: Tensor, t_pixel_total: int, start: int, stop: int
) -> Tensor | list[Tensor]:
    """Select/reshape/chunk/detach the encoder's own DINO features for the self-consistency
    target, per layer -- matches mira's `real_lc = tuple(f[:, t_lc].detach() for f in
    model_outputs.dino_features)`. `features` is a single Tensor in the old single-layer case.

    t_idx indexes pixel-space frames (0..t_pixel_total-1); `features`' own time dim can be
    shorter (VjepaModel halves frame count via its tubelet, DinoModel doesn't) -- remapped to
    feature-space via the ratio between the two, inferred from shapes rather than a
    per-backbone constant. A no-op for DinoModel (ratio 1, same indices either way).

    When reduction>1, t_idx is guaranteed (see _sample_frame_indices) to be laid out as sorted,
    reduction-sized runs of real adjacent pixel frames -- every `reduction` consecutive
    feature_idx entries are therefore identical (both members of one real tubelet pair map to the
    same feature). Deduplicated via a strided [::reduction] pick to one representative per group,
    matching the prediction side's own post-dino_forward group count exactly, rather than keeping
    raw duplicates and relying on _align_time_dim's blunt tail-truncation to paper over the length
    mismatch (which would silently misalign which groups survive). start/stop are themselves
    already reduction-aligned by construction (see the chunk_size computation in forward()), so
    plain floor division converts them to group-space exactly.
    """

    def _one(f: Tensor) -> Tensor:
        reduction = max(1, t_pixel_total // f.shape[1])
        selected = rearrange(f[:, _feature_indices(t_idx, t_pixel_total, f.shape[1])],
                             "b t c h w -> (b t) c h w")
        group_start, group_stop = start // reduction, stop // reduction
        return selected[group_start:group_stop].unsqueeze(0).detach()

    if isinstance(features, list):
        return [_one(f) for f in features]
    return _one(features)


def _sample_frame_indices(t_total: int, k: int, reduction: int, device: torch.device) -> Tensor:
    """Random frame-subset for the DINO-consistency term's cost-control trick (mira's own "only
    score a fraction of frames per step"). reduction=1 (DinoModel, no temporal coupling between
    frames): independent scattered frames, exactly the original behavior --
    torch.randperm(t_total)[:k].sort().values.

    reduction>1 (a tubelet-pairing encoder like VjepaModel): sampling k independent scattered
    frames and feeding them to dino_forward as a fake contiguous clip is wrong -- the encoder
    pairs CONSECUTIVE POSITIONS in whatever it's given, so two arbitrarily-spaced selected frames
    would get paired as if they were real temporal neighbors, fabricating a cross-frame pair with
    no correspondence to the real target features (see notes/deviations.md and this module's own
    forward() comment). Instead samples whole reduction-sized ADJACENT groups: real temporal
    neighbors every time, so every dino_forward call only ever sees genuine tubelet-sized chunks.
    Returns up to k indices, rounded down to a whole number of groups, sorted ascending.
    """
    if reduction <= 1:
        return torch.randperm(t_total, device=device)[:k].sort().values
    num_valid_groups = t_total // reduction
    num_groups = max(1, min(k // reduction, num_valid_groups))
    group_starts = torch.randperm(num_valid_groups, device=device)[:num_groups].sort().values * reduction
    offsets = torch.arange(reduction, device=device)
    return (group_starts.unsqueeze(1) + offsets.unsqueeze(0)).flatten().sort().values


def _item_slice(features: Tensor | list[Tensor], item: int) -> Tensor | list[Tensor]:
    """features sliced to one batch item, batch dim kept (size 1) -- same Tensor | list[Tensor]
    convention as _select_target_features/_align_time_dim."""
    if isinstance(features, list):
        return [f[item : item + 1] for f in features]
    return features[item : item + 1]


def _align_time_dim(
    pred_features: Tensor | list[Tensor], target_features: Tensor | list[Tensor]
) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor]]:
    """Trim the longer side's time dim down to the shorter, when they don't already match.
    Defensive fallback only, not the primary alignment mechanism: _sample_frame_indices +
    _select_target_features's own deduplication (see both) already make pred_features and
    target_features come out the same length by construction for a tubelet-pairing encoder
    (VjepaModel) -- kept as insurance against any remaining edge case, and still a no-op for
    DinoModel (already always equal).
    """
    pred_list = pred_features if isinstance(pred_features, list) else [pred_features]
    target_list = target_features if isinstance(target_features, list) else [target_features]
    n = min(pred_list[0].shape[1], target_list[0].shape[1])
    if pred_list[0].shape[1] == n and target_list[0].shape[1] == n:
        return pred_features, target_features
    trimmed_pred = [f[:, :n] for f in pred_list]
    trimmed_target = [f[:, :n] for f in target_list]
    if not isinstance(pred_features, list):
        return trimmed_pred[0], trimmed_target[0]
    return trimmed_pred, trimmed_target


def calculate_adaptive_weight(
    anchor_loss: Tensor, other_loss: Tensor, last_layer: Tensor, max_weight: float = 1e4,
    probe_scale: float | None = None,
) -> Tensor:
    """VQ-GAN-style adaptive weight (Esser et al., arXiv:2012.09841 S3.3), ported from mira's
    calculate_adaptive_weight. Rescales other_loss so its gradient at `last_layer` matches
    anchor_loss's gradient there in size, so it neither dominates nor gets drowned out. Detached:
    scales the loss but carries no gradient of its own.

    probe_scale: pass the training loop's actual GradScaler.get_scale() here under FP16 -- these
    probes happen before GradScaler sees loss_total, so their own small last-layer gradients can
    underflow without it; the common factor cancels in the norm ratio either way, so this doesn't
    need to be exact, just present. GradScaler's scale drifts up/down during real training, so a
    hardcoded constant (this used to just assume 65536, its default init_scale) silently drifts
    out of sync with what the scaler is actually doing. Left as None (no scaler bound to pass a
    real value), falls back to the same fp16-autocast-detection heuristic as before -- still a
    real, if less precise, safety net for any caller that isn't threading a live scaler through.
    """
    if probe_scale is None:
        probe_scale = (
            65536.0
            if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
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
    # Applies ONLY to CodecLoss's probing fallback (_apply_adaptive_weights_by_probing), where
    # each factor costs a real extra backward pass -- through all of VGG for the LPIPS term,
    # through the whole frozen backbone for the DINO one. Re-probing every N micro-steps and
    # reusing the last value in between trades a little exactness for skipping most of that.
    # The default path does not probe at all (see _apply_adaptive_weights_reusing_gradients) and
    # ignores this: there, exact factors are already essentially free.
    auto_weight_every: int = 1


class CodecLoss(nn.Module):
    """Computes the codec's reconstruction loss terms and their weighted total.

    The DINO-consistency term needs the encoder's frozen DINO backbone to look at the
    reconstruction; bind it after construction via bind_encoder_dino (mirrors mira's pattern of
    sharing the encoder's already-loaded DINO rather than loading a second copy).
    """

    def __init__(
        self, weights: CodecLossWeights, use_checkpointing: bool = False,
        perceptual_chunk_size: int = 0, log_activation_grad_norms: bool = False,
    ):
        super().__init__()
        self.weights = weights
        # Only the DINO call below needs it: it's the one that backprops through the whole
        # backbone (onto the reconstruction), unlike the encoder's own no_grad call.
        self.use_checkpointing = use_checkpointing
        self.perceptual_chunk_size = perceptual_chunk_size
        # Off by default: these are activation gradients (see _hook_clone), not parameter
        # gradients, and notes/grad_norm_investigation.md found them GradScaler-scale-confounded
        # under --precision fp16-hybrid and generally less directly interpretable than
        # train_codec.py's grad_norm_params_total. Costs a real (if modest) extra tensor clone per
        # term per micro-step for a signal that's now mostly superseded -- opt-in only.
        self.log_activation_grad_norms = log_activation_grad_norms

        self.lpips_perceptual_loss: nn.Module | None = None
        if weights.loss_lpips_perceptual > 0:
            # net_type="vgg" matches mira's own choice (mira/src/mira/codec/loss.py uses
            # lpips.LPIPS(net="vgg")). torchmetrics' version loads VGG16 via torchvision.models,
            # not the separate `lpips` pip package mira uses directly -- confirmed by reading
            # torchmetrics' own source (torchmetrics/functional/image/lpips.py).
            torchmetrics_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
            torchmetrics_lpips.eval()
            for p in torchmetrics_lpips.parameters():
                p.requires_grad = False
            self.lpips_perceptual_loss = FusedLpips(torchmetrics_lpips)

        self.dino: DinoModel | None = None  # bound post-init, see bind_encoder_dino
        self.perceptual_dino: DinoModel | None = None  # optional, see bind_perceptual_dino
        self._last_layer: Tensor | None = None  # bound post-init, see bind_last_layer
        self._grad_scaler: torch.amp.GradScaler | None = None  # optional, see bind_grad_scaler
        self._perceptual_channels_last = False  # see use_channels_last_perceptual
        # Keep each loss term's graph alive after its gradient has been taken. Only needed by a
        # caller that differentiates an individual term again afterwards -- see
        # train_codec_vjepa.py's --log-per-term-grad-norm, which is the only one that does.
        self.retain_term_graphs = False

        # Per-term real gradient norms, populated during backward() -- see _hook_clone. Read
        # after backward(), not before: empty until then. Matches mira's own backward_metrics.
        self.backward_metrics: dict[str, Tensor] = {}

        # Last computed adaptive weights, reused between probes when auto_weight_every > 1.
        self._auto_weights: dict[str, Tensor] = {}
        self._auto_weight_calls = 0

    def _hook_clone(self, tensor: Tensor, loss_name: str) -> Tensor:
        """Clone `tensor` and register a backward hook recording its gradient norm under
        `loss_name` in backward_metrics -- lets training watch each term's real gradient
        magnitude directly, not just auto_weight's derived ratio. Matches mira's own
        CodecLoss._hook_clone exactly, including its dim=-1 norm convention.
        A no-op (returns tensor unchanged, no clone/hook) unless log_activation_grad_norms is on."""
        if not tensor.requires_grad or not self.log_activation_grad_norms:
            return tensor
        clone = tensor.clone()
        clone.register_hook(
            lambda grad: self.backward_metrics.update({loss_name: grad.data.norm(p=2, dim=-1).mean()})
        )
        return clone

    def use_channels_last_perceptual(self) -> None:
        """Put the LPIPS VGG stack in channels-last, the layout cuDNN's tensor-core convolution
        kernels actually want. Purely a memory-layout change (same weights, same math); measured
        ~1.17x on the LPIPS term's forward+backward at 20 frames of 448x768. The inputs are
        converted to match in forward()."""
        if self.lpips_perceptual_loss is not None:
            self.lpips_perceptual_loss.to(memory_format=torch.channels_last)
            self._perceptual_channels_last = True

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

    def bind_grad_scaler(self, grad_scaler: torch.amp.GradScaler) -> None:
        """Optional: share the training loop's real GradScaler, so calculate_adaptive_weight's
        underflow-guard probe_scale tracks its actual current scale instead of guessing at one.
        Not bound by default: forward() falls back to a heuristic guess when this isn't set."""
        self._grad_scaler = grad_scaler

    def _dino_features_of(self, dino, video: Tensor):
        """The consistency term's own backbone pass over the RECONSTRUCTION -- the one DINO call
        each step that is not under no_grad, and so the one worth checkpointing."""
        if self.use_checkpointing:
            return checkpoint(dino.dino_forward, video, use_reentrant=False)
        return dino.dino_forward(video)

    def _dino_consistency_batched(self, dino, pred_selected, target_selected, encoder_features,
                                  t_idx: Tensor, t_total: int) -> Tensor:
        """Unchunked path (the default): the whole batch through the backbone in ONE call.

        dino_forward pairs frames within each batch element, so the batch dimension by itself
        gives the "never pair frames across two different videos" property that the chunked path
        below needs its per-item loop for. The value is identical to that loop's too: its weights
        are all k_actual / (b * k_actual) = 1/b, and an equally weighted mean of equal-sized
        per-item MSEs is exactly the MSE over the whole batch.
        """
        pred_features = self._dino_features_of(dino, denormalize_for_dino(pred_selected))
        if self.perceptual_dino is not None:
            with torch.no_grad():
                target_features = dino.dino_forward(denormalize_for_dino(target_selected))
        else:
            target_features = _select_batched_target_features(encoder_features, t_idx, t_total)
        return _layer_averaged_mse(*_align_time_dim(pred_features, target_features))

    def _dino_consistency_chunked(self, dino, pred_selected, target_selected, encoder_features,
                                  t_idx: Tensor, t_total: int, chunk_size: int) -> Tensor:
        """--perceptual-chunk-size path: one backbone call per (item, chunk), summed by weight.

        Chunks WITHIN each video's own selected frames, never across videos -- flattening batch
        and frame together and chunking blindly (an earlier approach) let a chunk straddle two
        different videos, so a tubelet-pairing encoder (VjepaModel) would pair the last selected
        frame of one video with the first of a totally different one. DinoModel never cared about
        frame adjacency, so this was invisible until V-JEPA existed; confirmed live with two
        distinguishable synthetic videos before this fix.
        """
        b, k_actual = pred_selected.shape[:2]
        total_selected = b * k_actual
        terms = []
        for item in range(b):
            item_target_features = _item_slice(encoder_features, item)
            for start in range(0, k_actual, chunk_size):
                stop = min(start + chunk_size, k_actual)
                dino_input = denormalize_for_dino(pred_selected[item, start:stop]).unsqueeze(0)
                pred_features = self._dino_features_of(dino, dino_input)
                if self.perceptual_dino is not None:
                    target_input = denormalize_for_dino(target_selected[item, start:stop]).unsqueeze(0)
                    with torch.no_grad():
                        target_features = dino.dino_forward(target_input)
                else:
                    target_features = _select_target_features(item_target_features, t_idx, t_total, start, stop)
                weight = (stop - start) / total_selected
                terms.append(weight * _layer_averaged_mse(*_align_time_dim(pred_features, target_features)))
        return torch.stack(terms).sum()

    def _can_reuse_probe_gradients(self) -> bool:
        """Whether the single-backward adaptive-weight path below can be used.

        It computes each term's gradient at `predicted` once and hands the combination to
        autograd, which means the individual terms are never separately backpropagated again --
        and that is exactly what --log-activation-grad-norms' per-term backward hooks
        (_hook_clone) exist to observe. With those on, fall back to the probing path so the
        metric keeps measuring what it says it measures.
        """
        return not self.log_activation_grad_norms

    def _apply_adaptive_weights_reusing_gradients(self, loss: dict[str, Tensor], predicted: Tensor) -> Tensor:
        """Adaptive weights, computed exactly, without any extra backward pass.

        The probing version below asks autograd for d(term)/d(last_layer) once per term, and then
        the training loop's own .backward() computes all of that a second time. Those probe passes
        are not cheap: the LPIPS one runs backward through the whole VGG stack and the DINO one
        through the whole frozen V-JEPA backbone. Measured at 448x768x40 on an H100 they cost
        ~575ms per micro-step, more than the entire rest of the micro-step put together.

        Every term here is a function of exactly one non-detached tensor -- `predicted` -- so all
        of that work can be done once instead of twice:

          1. take g_term = d(term)/d(predicted) for each term (the expensive part, once);
          2. push each g_term the short remaining distance to the last layer to get the gradient
             norms the adaptive weights are defined by (tanh + one matmul -- negligible);
          3. hand autograd the already-weighted combination of the g_terms, so the real backward
             starts from `predicted` and never re-enters VGG or V-JEPA at all.

        The weights that come out are the same numbers calculate_adaptive_weight produces, and
        the gradient the decoder receives is the same one it received before -- see
        scripts/verify_adaptive_weight_fusion.py, which checks both against the probing path.
        """
        probe_scale = self._grad_scaler.get_scale() if self._grad_scaler is not None else 1.0
        # Each term's own subgraph is finished with as soon as its gradient at `predicted` is in
        # hand -- the real backward below starts FROM `predicted` and never re-enters VGG or
        # V-JEPA -- so releasing it here rather than at the end of the micro-step is what keeps
        # the peak from carrying both perceptual graphs through the decoder's backward. Only the
        # part strictly above `predicted` is released: `predicted` is the traversal's endpoint, so
        # its own grad_fn and the whole decoder below it are never visited. retain_term_graphs
        # turns this off for the one caller that does differentiate a term a second time.
        retain = self.retain_term_graphs
        anchor_gradient = torch.autograd.grad(
            loss["loss_mae"] * probe_scale, predicted, retain_graph=True
        )[0]
        anchor_norm = self._last_layer_gradient(predicted, anchor_gradient).norm()

        combined = self.weights.loss_mae * anchor_gradient
        for name in ("loss_lpips_perceptual", "loss_dino_latent_consistency"):
            if name not in loss or not loss[name].requires_grad:
                continue
            gradient = torch.autograd.grad(loss[name] * probe_scale, predicted, retain_graph=retain)[0]
            other_norm = self._last_layer_gradient(predicted, gradient).norm()
            factor = (anchor_norm / (other_norm + 1e-6)).clamp(0.0, self.weights.max_auto_weight).detach()
            loss[name] = factor * loss[name]
            loss[f"{name}_auto_w"] = factor  # logged, not summed
            combined = combined + (getattr(self.weights, name) * factor) * gradient
        # probe_scale is a pure fp16 underflow guard (it cancels in every weight ratio above); the
        # gradient handed back to autograd has to be at its true scale, so divide it back out.
        return combined / probe_scale

    def _last_layer_gradient(self, predicted: Tensor, gradient_at_predicted: Tensor) -> Tensor:
        """d(term)/d(last_layer), given d(term)/d(predicted) -- a vector-Jacobian product over
        just the decoder's output tail (tanh, then the final projection), not the whole graph."""
        return torch.autograd.grad(
            predicted, self._last_layer, grad_outputs=gradient_at_predicted, retain_graph=True
        )[0]

    def _apply_adaptive_weights_by_probing(self, loss: dict[str, Tensor]) -> None:
        """The original adaptive-weight path: one extra backward pass per term, per micro-step.

        Kept for --log-activation-grad-norms (see _can_reuse_probe_gradients) and honoured
        auto_weight_every, which trades exactness for skipping most of those passes -- unnecessary
        on the path above, which is both exact and free.
        """
        every = max(1, self.weights.auto_weight_every)
        refresh = self._auto_weight_calls % every == 0
        self._auto_weight_calls += 1
        for name in ("loss_lpips_perceptual", "loss_dino_latent_consistency"):
            if name not in loss or not loss[name].requires_grad:
                continue
            if refresh or name not in self._auto_weights:
                probe_scale = self._grad_scaler.get_scale() if self._grad_scaler is not None else None
                self._auto_weights[name] = calculate_adaptive_weight(
                    loss["loss_mae"], loss[name], self._last_layer, self.weights.max_auto_weight,
                    probe_scale,
                )
            factor = self._auto_weights[name]
            loss[name] = factor * loss[name]
            loss[f"{name}_auto_w"] = factor  # logged, not summed

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
            if self._perceptual_channels_last:
                pred_2d = pred_2d.contiguous(memory_format=torch.channels_last)
                tgt_2d = tgt_2d.contiguous(memory_format=torch.channels_last)
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
            b = predicted.shape[0]
            k = max(1, round(t_total * self.weights.dino_latent_consistency_frame_frac))
            # Encoders with their own temporal reduction (tubelet-pairing, e.g. VjepaModel) need
            # every dino_forward call to see REAL temporally-adjacent frames -- see
            # _sample_frame_indices for why sampling independent scattered frames (the old
            # behavior) fabricates nonsense cross-frame pairs with no correspondence to the real
            # target features. getattr(...,1) is a no-op for DinoModel.
            reduction = getattr(consistency_dino, "tubelet_size", 1)
            t_idx = _sample_frame_indices(t_total, k, reduction, predicted.device)
            k_actual = t_idx.shape[0]
            # No no_grad here: needs to backprop from DINO's features on the reconstruction,
            # through the decoder and bottleneck. Target side is already detached.
            predicted_dino = self._hook_clone(predicted, "loss_dino_latent_consistency")
            pred_selected = predicted_dino[:, t_idx]  # (b, k_actual, c, h, w)
            target_selected = target[:, t_idx]  # (b, k_actual, c, h, w)
            # Chunk WITHIN each video's own k_actual selected frames, never across videos --
            # flattening batch+frame together and chunking blindly (the old approach) let a chunk
            # straddle two different videos, so a tubelet-pairing encoder (VjepaModel) would pair
            # the last selected frame of one video with the first of a totally different one.
            # DinoModel never cared about frame adjacency, so this was invisible until V-JEPA
            # existed; confirmed live with two distinguishable synthetic videos before this fix.
            # Also rounded down to a reduction-aligned boundary (a no-op for DinoModel,
            # reduction=1) -- an explicit --perceptual-chunk-size that isn't itself a multiple of
            # reduction could otherwise split a real tubelet pair across two separate
            # dino_forward calls, reintroducing the same fabricated-pairing problem at the chunk
            # boundary that _sample_frame_indices already fixed at the selection stage.
            raw_chunk_size = min(self.perceptual_chunk_size or k_actual, k_actual)
            chunk_size = max(reduction, (raw_chunk_size // reduction) * reduction)
            total_selected = b * k_actual
            if chunk_size >= k_actual:
                loss["loss_dino_latent_consistency"] = self._dino_consistency_batched(
                    consistency_dino, pred_selected, target_selected, outputs.dino_features, t_idx, t_total
                )
            else:
                loss["loss_dino_latent_consistency"] = self._dino_consistency_chunked(
                    consistency_dino, pred_selected, target_selected, outputs.dino_features,
                    t_idx, t_total, chunk_size,
                )

        auto_weighting = (
            self.weights.auto_weight
            and self._last_layer is not None
            and "loss_mae" in loss
            and loss["loss_mae"].requires_grad
            and torch.is_grad_enabled()
        )
        combined_gradient = None
        if auto_weighting and self._can_reuse_probe_gradients():
            combined_gradient = self._apply_adaptive_weights_reusing_gradients(loss, predicted)
        elif auto_weighting:
            self._apply_adaptive_weights_by_probing(loss)

        weighted = [
            getattr(self.weights, name) * value
            for name, value in loss.items()
            if hasattr(self.weights, name)  # excludes the *_auto_w diagnostic entries above
        ]
        # weighted is non-empty for any sensible config (loss_mae is on by default); guard the
        # degenerate all-zero-weights case so it returns a finite zero rather than raising on
        # stack, matching mira's own guard.
        total = torch.stack(weighted).sum() if weighted else predicted.new_zeros(())
        if combined_gradient is not None:
            total = _with_gradient(predicted, combined_gradient, total)
        loss["loss_total"] = total
        return loss
