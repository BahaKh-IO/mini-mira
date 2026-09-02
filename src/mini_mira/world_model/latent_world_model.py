"""The training-time world-model wrapper: a frozen codec (DINO + bottleneck + decoder, loaded
from an already-trained checkpoint) plus a trainable DiffusionTransformer + ActionEncoder + a
learned bos parameter. Named LatentWorldModel to match real mira's own class
(mira/src/mira/world_model/latent_world_model.py) directly -- see notes/naming.md for why this is
a closer structural match than MyPipeline ever was (MyPipeline never loads/freezes a real codec
checkpoint; this class exists specifically to do that).

Ports real mira's diagonal flow-matching loss + PSD self-distillation 1:1 (same tau sampling, same
upper-triangle PSD sampling, same torch.no_grad() teacher passes), minus the
torch.distributed.broadcast (mini_mira never runs distributed).
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from mini_mira.codec.bottleneck import MyBottleneck, StridedConvBottleneckConfig
from mini_mira.codec.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.codec.dino import DEFAULT_ENCODER_AGGREGATION_LAYERS, DinoModel
from mini_mira.codec.loss import denormalize_for_dino
from mini_mira.world_model.action_encoder import ActionEncoder
from mini_mira.world_model.diffusion_transformer import DiffusionTransformer, LatentWorldModelConfig
from mini_mira.world_model.fd_loss import FDLossEMAState, RealFrechetStats, differentiable_frechet_distance

_CONVOLUTION_TYPES = (
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
    torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
)


def _keep_convolutions_in_bf16(module: torch.nn.Module) -> None:
    """Nest BF16 autocast around convolutions while the surrounding model uses FP16 AMP.

    Duplicated from scripts/train_codec.py rather than imported, deliberately -- see this
    project's plan for the world-model script: importing from train_codec.py would risk touching
    that already-running training script. Same reasoning it exists there: DINOv3's
    patch-embedding conv (and other conv-family ops) crash under fp16/fp32 autocast on this
    project's V100 with a cuDNN "unable to find an engine" error; bf16 is the one precision that
    works for those ops specifically (notes/gpu_amp_investigation.md), while the *trainable*
    layers (pure Linear/attention here) use fp16+GradScaler, the tested/stable setup
    (notes/gpu_box_notes_backup.md). Applied below to `bottleneck`/`decoder` (frozen, but their
    conv forward passes still hit the same crash regardless of grad tracking) -- NOT to `dino`,
    which already wraps its own body in bf16 autocast internally (DinoModel.dino_forward).
    """
    import types  # noqa: PLC0415 -- only needed inside this monkey-patch helper

    for convolution in (m for m in module.modules() if isinstance(m, _CONVOLUTION_TYPES)):
        if getattr(convolution, "_mini_mira_bf16_forward", False):
            continue
        original_forward = convolution.forward

        def bf16_forward(_self, *args, _forward=original_forward, **kwargs):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return _forward(*args, **kwargs)

        convolution.forward = types.MethodType(bf16_forward, convolution)
        convolution._mini_mira_bf16_forward = True


def _load_codec_weights(path: str | Path, bottleneck: MyBottleneck, decoder: ViTVideoDecoder) -> None:
    """Loads just the frozen bottleneck+decoder weights from a codec checkpoint.pth (produced by
    scripts/train_codec.py / mini_mira.codec.checkpoint.save_checkpoint). Not that module's own
    load_checkpoint -- this has no optimizer/lr_scheduler to restore into, and the codec is frozen
    here, never resumed."""
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    bottleneck.load_state_dict(ckpt["bottleneck"])
    decoder.load_state_dict(ckpt["decoder"])


class LatentWorldModel(nn.Module):
    def __init__(
        self,
        world_model_config: LatentWorldModelConfig,
        bottleneck_config: StridedConvBottleneckConfig,
        decoder_config: ViTDecoderConfig,
        num_keys: int,
        codec_checkpoint: str | Path | None,
        latent_mean: float = 0.0,
        latent_std: float = 1.0,
        real_frechet_mean: torch.Tensor | None = None,
        real_frechet_cov: torch.Tensor | None = None,
        require_pretrained_dino: bool = True,
        encoder_dino_model: str = "dinov3_vitb16",
        dino: nn.Module | None = None,
    ):
        super().__init__()
        self.world_model_config = world_model_config

        # --- Frozen codec: DINO + bottleneck + decoder ---
        # Multi-layer aggregation, matching train_codec.py's own encoder-side DinoModel
        # construction exactly -- the bottleneck was trained expecting this same aggregated
        # multi-layer input, not a single last-layer tensor.
        # `dino` is an injection seam for testing only (default None -> build the real backbone):
        # DinoModel's loader depends on a private, version-fragile torch.hub API
        # (notes/deviations.md 1.14) that is currently broken on this dev machine specifically
        # (confirmed pre-existing: it also blocks the already-working verify_codec_training.py,
        # unrelated to this class) -- passing a lightweight stand-in with the same dino_forward
        # contract lets verify_world_model_training.py exercise everything else without it.
        self.dino = dino if dino is not None else DinoModel(
            dino_model=encoder_dino_model,
            require_pretrained=require_pretrained_dino,
            last_layer_only=False,
            layer_indices=DEFAULT_ENCODER_AGGREGATION_LAYERS[encoder_dino_model],
        )
        # True raw-video-frames-per-latent-frame ratio -- NOT just the bottleneck's own stride.
        # DinoModel.dino_forward never touches time, so bottleneck_config.temporal_stride alone is
        # already the full ratio for it (getattr below falls back to 1, byte-identical to the old
        # formula). VjepaModel.dino_forward halves time internally BEFORE the bottleneck ever sees
        # it (tubelet_size=2, vjepa.py) -- configs/scaled_300m_vjepa.yaml's temporal_stride=1 only
        # fixes the LATENT TENSOR's shape (so both tracks land on T'=20 from 40 raw frames), it
        # doesn't account for this separately-tracked ratio. Left uncorrected, every latent frame's
        # action window (_encode below, ActionEncoder's own reshape) would silently be computed as
        # covering 1 raw frame instead of the 2 it actually represents -- a real, silent
        # action/latent misalignment specific to any encoder with its own temporal reduction.
        # Exactly the risk notes/deviations.md 1.21 already flagged as open, unresolved by the
        # config alone. Must be computed here, after self.dino is known, not by the caller after
        # construction -- ActionEncoder below is built from this value and keeps its own copy.
        self.temporal_downsampling = bottleneck_config.temporal_stride * getattr(self.dino, "tubelet_size", 1)
        self.bottleneck = MyBottleneck(bottleneck_config)
        # use_checkpointing only when FD-loss is active -- see _fd_pooled_features, the only place
        # this ever needs a backward-compatible activation footprint through the decoder. Gated
        # here (not just at the call site) because it's the module's OWN self.training that decides
        # whether checkpointing fires each forward, so building it False keeps every non-FD-loss
        # path byte-identical to before this existed.
        self.decoder = ViTVideoDecoder(decoder_config, use_checkpointing=world_model_config.fd_loss_weight > 0)
        if codec_checkpoint is not None:
            _load_codec_weights(codec_checkpoint, self.bottleneck, self.decoder)
        # codec_checkpoint=None keeps bottleneck/decoder at random init -- verify_world_model_
        # training.py's mechanism-only mode, no real weights, no network.
        self.bottleneck.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.bottleneck.eval()
        self.decoder.eval()
        _keep_convolutions_in_bf16(self.bottleneck)
        _keep_convolutions_in_bf16(self.decoder)
        # DinoModel freezes and .eval()s itself in its own __init__ -- nothing more needed here.

        self.register_buffer("latent_mean", torch.tensor(latent_mean), persistent=False)
        self.register_buffer("latent_std", torch.tensor(latent_std), persistent=False)

        # --- Trainable: world model + action encoder + bos ---
        self.world_model = DiffusionTransformer(world_model_config)
        self.action_encoder = ActionEncoder(
            num_keys=num_keys,
            hidden_dim=world_model_config.hidden_dim,
            temporal_downsampling=self.temporal_downsampling,
        )
        # Single vector broadcast over h/w, matching MyPipeline's existing bos exactly (not real
        # mira's grid-shaped version -- already-documented deviation, notes/deviations.md 1.7).
        self.bos = nn.Parameter(0.02 * torch.randn(bottleneck_config.latent_dim))

        # FD-loss (arXiv:2604.28190v1) -- opt-in via world_model_config.fd_loss_weight, matches
        # the psd_weight convention. real_frechet_mean/cov are the real-data stats, precomputed
        # offline (scripts/compute_real_frechet_stats_vjepa.py); registered as buffers (not kept
        # as a plain dataclass) so they move with the model's own .to(device)/.cuda() calls, same
        # reasoning as latent_mean/latent_std above. cov_sqrt is computed once here (eigh is the
        # expensive part), not every training step -- see fd_loss.RealFrechetStats.
        self.fd_loss_enabled = world_model_config.fd_loss_weight > 0
        self.fd_ema: FDLossEMAState | None = None
        if self.fd_loss_enabled:
            assert real_frechet_mean is not None and real_frechet_cov is not None, (
                "fd_loss_weight > 0 needs real_frechet_mean/real_frechet_cov "
                "(scripts/compute_real_frechet_stats_vjepa.py)"
            )
            real_stats = RealFrechetStats.from_mean_cov(real_frechet_mean, real_frechet_cov)
            self.register_buffer("fd_real_mean", real_stats.mean, persistent=False)
            self.register_buffer("fd_real_cov_sqrt", real_stats.cov_sqrt, persistent=False)
            self.register_buffer("fd_real_trace_cov", real_stats.trace_cov, persistent=False)
            self.fd_ema = FDLossEMAState(dim=self.dino.dino_dim, beta=world_model_config.fd_loss_ema_decay)

    def train(self, mode: bool = True) -> "LatentWorldModel":
        """Ports real mira's override verbatim: keeps the frozen codec in .eval() even when the
        wrapper itself is set to train()."""
        super().train(mode)
        self.dino.eval()
        self.bottleneck.eval()
        self.decoder.eval()
        return self

    def normalize_tokens(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean) / self.latent_std

    def unnormalize_tokens(self, z: torch.Tensor) -> torch.Tensor:
        return self.latent_std * z + self.latent_mean

    def decode_to_video(self, z: torch.Tensor) -> torch.Tensor:
        """z: (b, t, c, h, w) normalized latents. Returns [0, 1] video, (b, t*patch_size_t, 3, H, W)."""
        z = self.unnormalize_tokens(z)
        video = self.decoder(z)  # [-1, 1] tanh output
        return denormalize_for_dino(video)  # -> [0, 1]

    def _fd_pooled_features(self, z1_estimate: torch.Tensor) -> torch.Tensor:
        """z1_estimate: (b,t,c,h,w) normalized latents. Decodes and extracts per-frame V-JEPA
        features, spatially pooled -- same convention full_eval_metrics.py's Frechet accumulation
        already uses (real_dino_gen[:, s].mean(dim=(-1,-2)).flatten(0,1)), just made
        differentiable: no no_grad here, self.decoder/self.dino are frozen-but-differentiable
        (requires_grad_(False) blocks gradient into their own weights only, not through them to
        the input -- same proven pattern as CodecLoss's DINO-consistency term, src/mini_mira/
        codec/loss.py).

        Both the decode and the V-JEPA forward are gradient-checkpointed -- real, hit-for-real
        necessity, not precautionary: this is the only place LatentWorldModel ever backprops
        through the full decoder+encoder at training resolution (eval/rollout always run under
        no_grad; neither needed a backward-compatible activation footprint before FD-loss
        existed), and a real 448x768/batch=4 run OOM'd even with a first attempt at this (a single
        whole-function checkpoint around decode_to_video) -- that only recomputes the ENTIRE
        decoder in one shot during backward, no cheaper peak than no checkpointing at all.
        ViTVideoDecoder has its own real per-block checkpointing already (same lever that got the
        codec itself from OOM to comfortably fitting at this exact resolution) -- gated on
        self.training, permanently False here since train() above always .eval()s the decoder.
        Toggled on just for this call; safe (no BatchNorm/Dropout in decoder.py, confirmed by
        direct read -- LayerNorm is train/eval-identical). V-JEPA has no equivalent internal
        option (third-party code), so it keeps the coarser external checkpoint(), same
        use_reentrant=False pattern already proven in CodecLoss (loss.py)."""
        self.decoder.train()
        video = self.decode_to_video(z1_estimate)
        self.decoder.eval()
        dino_features = checkpoint(self.dino.dino_forward, video, use_reentrant=False)
        if isinstance(dino_features, list):
            dino_features = dino_features[-1]
        return dino_features.mean(dim=(-1, -2)).flatten(0, 1)

    def warm_start_fd_loss(self, batch) -> None:
        """One no-grad EMA update from a real batch -- call this a few times before real
        fine-tuning starts, so the first real gradient steps aren't optimizing against an
        undefined EMA. Small-scale analog of the paper's own 50k-sample warm-start."""
        assert self.fd_loss_enabled and self.fd_ema is not None, "fd_loss_weight must be > 0"
        with torch.no_grad():
            z, a = self._encode(batch)
            shifted_z = self._shifted_z(z)
            z_t, _v, tau = self.prepare_training_inputs(z)
            pred_v = self.world_model(z_t, a=a, tau=tau, clean_past=shifted_z)
            z1_estimate = z_t + (1 - tau) * pred_v
            features = self._fd_pooled_features(z1_estimate)
            self.fd_ema.update(features)

    def _encode(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """batch.video (uint8, [0,255]) -> normalized latents; batch.actions.key_presses ->
        encoded conditioning. Shared by forward() and rollout() so the action/latent alignment
        offset (below) lives in exactly one place."""
        video = batch.video.float() / 255.0
        with torch.no_grad():
            dino_features = self.dino.dino_forward(video)
            z = self.normalize_tokens(self.bottleneck(dino_features))

        b, t = z.shape[:2]
        td = self.temporal_downsampling
        # Real mira's exact action/latent alignment (latent_world_model.py's forward/
        # n_action_steps): the window of raw action steps that produced latent frame k starts one
        # action before frame k's own raw-frame index -- getting this offset wrong silently
        # misaligns actions and frames by one step.
        off = td - 1
        n_action_steps = (t - 1) * td
        # int32 (real mira's ActionTensors dtype) -> int64: nn.Embedding requires long indices.
        key_presses = batch.actions.key_presses[:, off : off + n_action_steps].long()
        a = self.action_encoder(key_presses)
        return z, a

    def forward(self, batch) -> dict[str, torch.Tensor]:
        z, a = self._encode(batch)
        return self.diffusion_loss(z, a)

    def _shifted_z(self, z: torch.Tensor) -> torch.Tensor:
        b, _, c, h, w = z.shape
        bos = self.bos.view(1, 1, c, 1, 1).expand(b, 1, c, h, w)
        return torch.cat([bos, z[:, :-1]], dim=1)

    def _fake_shifted_z(self, z: torch.Tensor, a: torch.Tensor, shifted_z: torch.Tensor) -> torch.Tensor:
        """Scheduled sampling: a cheap, single-step self-generated estimate of clean_past, so
        training occasionally sees the same kind of imperfect reference a real rollout actually
        produces, instead of always-real. One extra no-grad forward pass, not a full recursive
        rollout -- this auxiliary pass still conditions on the real shifted_z; only its output
        becomes this step's clean_past. Same one-step flow-matching estimate rollout's own
        integration uses: z_1 ~= z_t + (1 - tau) * predicted_velocity."""
        with torch.no_grad():
            tau_aux = self.sample_training_tau(z)
            z_t_aux = tau_aux * z + (1 - tau_aux) * torch.randn_like(z)
            pred_v_aux = self.world_model(z_t_aux, a=a, tau=tau_aux, clean_past=shifted_z)
            z_1_estimate = z_t_aux + (1 - tau_aux) * pred_v_aux
        return self._shifted_z(z_1_estimate)

    def diffusion_loss(self, z: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
        shifted_z = self._shifted_z(z)
        if self.world_model_config.scheduled_sampling_prob > 0:
            use_fake_past = torch.rand((), device=z.device) < self.world_model_config.scheduled_sampling_prob
            if use_fake_past:
                shifted_z = self._fake_shifted_z(z, a, shifted_z)

        z_t, v, tau = self.prepare_training_inputs(z)
        pred_v = self.world_model(z_t, a=a, tau=tau, clean_past=shifted_z)

        loss_diffusion = F.mse_loss(pred_v.float(), v.float())
        outputs: dict[str, torch.Tensor] = {"loss_total": loss_diffusion, "loss_diffusion": loss_diffusion}

        if self.world_model_config.psd_weight > 0:
            loss_psd = self._compute_psdm_loss(z, a, shifted_z)
            outputs["loss_total"] = loss_diffusion + self.world_model_config.psd_weight * loss_psd
            outputs["loss_psd"] = loss_psd.detach()
        elif self.world_model_config.psd_loss_prob > 0:
            # Stochastic PSD: only a psd_loss_prob fraction of steps pay the three-extra-forward-
            # pass cost. Importance-weighted logging (loss_psd_logged) so its expectation over
            # steps still equals the true PSD loss -- matches real mira exactly, minus the
            # torch.distributed.broadcast (single-process only, no rank-divergence risk here).
            loss_psd_logged = torch.zeros((), device=z.device)
            do_psd = torch.rand((), device=z.device) < self.world_model_config.psd_loss_prob
            if do_psd:
                loss_psd = self._compute_psdm_loss(z, a, shifted_z)
                outputs["loss_total"] = loss_diffusion + loss_psd
                loss_psd_logged = loss_psd.detach() / self.world_model_config.psd_loss_prob
            outputs["loss_psd"] = loss_psd_logged

        if self.fd_loss_enabled:
            assert self.fd_ema is not None
            # Reuses this step's own z_t/pred_v/tau -- no extra forward pass for the generation
            # side itself, the same one-step estimate _fake_shifted_z already uses for scheduled
            # sampling: z1 ~= z_t + (1-tau)*pred_v.
            z1_estimate = z_t + (1 - tau) * pred_v
            features = self._fd_pooled_features(z1_estimate)
            mean_g, cov_g = self.fd_ema.update(features)
            real_stats = RealFrechetStats(self.fd_real_mean, self.fd_real_cov_sqrt, self.fd_real_trace_cov)
            loss_fd = differentiable_frechet_distance(real_stats, mean_g, cov_g)
            outputs["loss_total"] = outputs["loss_total"] + self.world_model_config.fd_loss_weight * loss_fd.float()
            outputs["loss_fd"] = loss_fd.detach()

        return outputs

    def _compute_psdm_loss(self, z: torch.Tensor, a: torch.Tensor, shifted_z: torch.Tensor) -> torch.Tensor:
        """PSD-M self-distillation: the student's long-range velocity (s -> t) regressed onto the
        midpoint two-hop teacher target (s -> u -> t, stop-grad). Three extra forward passes."""
        z_s, tau_s, tau_t = self.prepare_psdm_inputs(z)
        tau_u = (tau_s + tau_t) / 2

        pred_v_st = self.world_model(z_s, a=a, tau=tau_s, tau_delta=tau_t - tau_s, clean_past=shifted_z)

        with torch.no_grad():
            pred_v_su = self.world_model(z_s, a=a, tau=tau_s, tau_delta=tau_u - tau_s, clean_past=shifted_z)
            x_su = z_s + (tau_u - tau_s) * pred_v_su
            pred_v_ut = self.world_model(x_su, a=a, tau=tau_u, tau_delta=tau_t - tau_u, clean_past=shifted_z)
            v_target_st = 0.5 * pred_v_su + 0.5 * pred_v_ut

        return F.mse_loss(pred_v_st.float(), v_target_st.float())

    def prepare_training_inputs(self, z_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_0 = torch.randn_like(z_1)
        v = z_1 - z_0
        tau = self.sample_training_tau(z_1)
        z_t = tau * z_1 + (1 - tau) * z_0
        return z_t, v, tau

    def sample_training_tau(self, z_1: torch.Tensor) -> torch.Tensor:
        batch_size, n_latents = z_1.shape[:2]
        tau = torch.rand(batch_size, n_latents, device=z_1.device)
        return rearrange(tau, "b t -> b t 1 1 1")

    def prepare_psdm_inputs(self, z_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_0 = torch.randn_like(z_1)
        tau_s, tau_t = self.sample_upper_triangle_tau(z_1)
        z_s = tau_s * z_1 + (1 - tau_s) * z_0
        return z_s, tau_s, tau_t

    def sample_upper_triangle_tau(self, z_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (s, t) uniformly from the upper triangle 0 <= s < t <= 1."""
        batch_size, n_latents = z_1.shape[:2]
        device = z_1.device
        draw_a = torch.rand(batch_size, n_latents, device=device)
        draw_b = torch.rand(batch_size, n_latents, device=device)
        s = rearrange(torch.min(draw_a, draw_b), "b t -> b t 1 1 1")
        t = rearrange(torch.max(draw_a, draw_b), "b t -> b t 1 1 1")
        return s, t

    def rollout(
        self, batch, n_context_latents: int, n_diffusion_steps: int = 4, schedule_type: str = "linear"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lightweight autoregressive rollout for the drift eval only
        (mini_mira.world_model.eval_metrics) -- no streaming kv-cache (DiffusionTransformer has
        none, matching notes/deviations.md 1.3), so every diffusion step re-runs the sequence
        built so far. Correct because temporal attention is causal: frame k's own prediction
        never depends on positions > k, so a working sequence truncated to [0, k] is exactly
        equivalent to running the full-length sequence and ignoring positions > k.

        The first n_context_latents latent frames are real encoded ground truth (teacher-forced
        context, same "clean_past is real history" caveat already documented for MyPipeline --
        see notes/deviations.md 6.1); frames from n_context_latents onward are generated, each one
        becoming clean context for the next.

        Returns (z, z_t): the real encoded (normalized) latents and the rolled-out (normalized)
        latents, both (b, t, c, h, w), same shape, for the caller to diff.
        """
        from mira.world_model.schedule import build_inference_schedule  # noqa: PLC0415 -- see module docstring

        z, a = self._encode(batch)
        b, t, c, h, w = z.shape
        assert 0 < n_context_latents < t, "n_context_latents must leave at least one frame to generate"

        z_t = z.clone()
        timesteps = build_inference_schedule(n_diffusion_steps, z.device, schedule_type)
        delta_ts = timesteps[1:] - timesteps[:-1]
        bos = self.bos.view(1, 1, c, 1, 1).expand(b, 1, c, h, w)

        with torch.no_grad():
            for k in range(n_context_latents, t):
                z_t[:, k] = torch.randn(b, c, h, w, device=z.device, dtype=z.dtype)
                # Working sequence [0, k]: everything before k is already clean (real context or
                # a previously generated frame) -- context_tau=1 for those, matching real mira's
                # own denoise_streaming convention -- only position k's tau actually moves.
                clean_past = torch.cat([bos, z_t[:, :k]], dim=1)
                tau = torch.ones(b, k + 1, 1, 1, 1, device=z.device, dtype=z.dtype)

                for timestep, delta_t in zip(timesteps[:-1], delta_ts):
                    tau[:, k] = timestep
                    pred_v = self.world_model(z_t[:, : k + 1], a=a[:, : k + 1], tau=tau, clean_past=clean_past)
                    z_t[:, k] = z_t[:, k] + delta_t * pred_v[:, k]

        return z, z_t
