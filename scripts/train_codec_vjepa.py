"""Config-driven codec training: VjepaModel -> MyBottleneck -> ViTVideoDecoder, trained on the
real three-term CodecLoss (L1 + LPIPS + DINO-latent-consistency, with auto_weight balancing --
see mini_mira.codec.loss), matching mira's own CodecLoss and its shipped config. Full fork of
train_codec.py for the V-JEPA track -- kept as a separate script on purpose (not a shared core
loop), so each track's logic reads on its own; see notes/session_handoff.md for why.

Two data modes:
  - Default (no --index-path): trains on one fixed synthetic video -- a mechanism check, not
    real training (see verify_codec_training.py for the CPU-friendly version of this).
  - --index-path <dir>: streams real clips via mira's own create_loader. Point it at the path
    scripts/download_shards.py prints after downloading.

Assumes a CUDA GPU (no CPU fallback) -- this script is for real training runs.

Precision: float16 autocast for the trainable codec, with its convolution-family modules and
the complete frozen V-JEPA backbone kept in bfloat16. This avoids this V100/cuDNN stack's FP16
convolution engine failures and V-JEPA's FP16 numerical instability.

Effective batch size is --batch-size (the real, per-forward micro-batch) times
--grad-accum-steps -- see mira.training.lr_schedule.WarmupConstantCosineDecayLR for the LR
schedule shape, and --activation-checkpointing to trade compute for the memory a larger
micro-batch would need.
"""

import argparse
import signal
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Real mira, cloned as a sibling repo -- reused for the dataloader.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from torch import Tensor
from mira.data.training_loader import create_loader
from mira.training.lr_schedule import WarmupConstantCosineDecayLR

from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.checkpoint import load_checkpoint, save_checkpoint
from mini_mira.codec.decoder import ViTVideoDecoder
from mini_mira.codec.logging_utils import get_wandb_run_id, init_wandb, log_preview, log_step
from mini_mira.codec.loss import (
    CodecLoss, CodecLossSchedule, CodecLossWeights, CodecOutputs, normalize_video,
)
from mini_mira.codec.data_pipeline import PrefetchingVideoStream
from mini_mira.codec.profiling import StepTimer, profile_window
from mini_mira.codec.vjepa import DEFAULT_VJEPA_LAYERS, VjepaModel
from mini_mira.ml.config_loading import apply_run_config, load_pipeline_config, load_run_config
from mini_mira.ml.run_config import CodecRunConfig


def _autocast(precision: str) -> torch.autocast:
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


# Set by _request_shutdown, checked once per completed optimizer step. `timeout` (used to
# time-box long unattended runs) sends SIGTERM with no warning -- without this, the process just
# dies wherever it happens to be, losing up to --checkpoint-every steps of progress and leaving
# wandb with no clean sign-off (shows "Crashed" even when nothing actually broke). Ports
# train_world_model.py's own handling, which train_codec.py (this script's un-forked sibling)
# still lacks. A signal handler must stay minimal (no CUDA/file I/O directly inside it), so this
# only sets a flag; the actual save happens in the main loop once it's safe to do so.
_shutdown_requested = False


def _request_shutdown(signum, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True


_CONVOLUTION_TYPES = (
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
    torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
)


def _keep_convolutions_in_bf16(module: torch.nn.Module) -> None:
    """Nest BF16 autocast around convolutions while the surrounding model uses FP16 AMP."""
    for convolution in (m for m in module.modules() if isinstance(m, _CONVOLUTION_TYPES)):
        if getattr(convolution, "_mini_mira_bf16_forward", False):
            continue
        original_forward = convolution.forward

        def bf16_forward(_self, *args, _forward=original_forward, **kwargs):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return _forward(*args, **kwargs)

        convolution.forward = types.MethodType(bf16_forward, convolution)
        convolution._mini_mira_bf16_forward = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument(
        "--run-config", default=None,
        help="Optional YAML of hyperparameters (CodecRunConfig, see configs/runs/). Any flag "
        "below, explicitly passed, always overrides it. Omit for today's hardcoded defaults.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Default 30")
    parser.add_argument("--lr", type=float, default=None)  # matches mira's train_codec.yaml, default 1e-4
    parser.add_argument("--height", type=int, default=None, help="Default 64")
    parser.add_argument("--width", type=int, default=None, help="Default 64")
    parser.add_argument(
        "--frames", type=int, default=None, help="Clip length in frames (both data modes). Default 4"
    )
    parser.add_argument("--require-pretrained-vjepa", action="store_true")
    parser.add_argument("--index-path", default=None, help="Real dataset dir from download_shards.py")
    parser.add_argument(
        "--num-workers", type=int, default=6,
        help="Parallel dataloader processes. Default 6 (this box has 8 real CPU cores -- nproc -- "
        "leaves 2 free for the main process/OS; matches real mira's own choice). Safe to raise for "
        "the codec here specifically: unlike train_world_model.py, --resume never tries to skip "
        "exactly the batches already consumed, so there's no determinism requirement this could "
        "silently break. Re-check against nproc if this ever runs on different hardware.",
    )
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-forward micro-batch size. Default 4")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="Micro-batches accumulated per optimizer step (effective batch = batch-size * this). Default 1",
    )
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=None,
        help="torch.compile the trainable bottleneck+decoder. The decoder is where this pays: its "
        "blocks are dominated by memory-bound elementwise work (RoPE, per-head qk-norm, the "
        "LayerScale/residual chain) that inductor fuses away -- measured 2.5x on decoder "
        "forward+backward at 448x768x40. Untested combined with --activation-checkpointing -- "
        "verify separately before assuming both together work. See --compile-encoder for the "
        "frozen V-JEPA backbone, which is compiled separately because it is external code.",
    )
    parser.add_argument(
        "--compile-encoder", action=argparse.BooleanOptionalAction, default=None,
        help="Also torch.compile the frozen V-JEPA encoder (default: follow --compile). Worth "
        "~1.3x on the encoder pass on top of the cuDNN attention backend, but it is external, "
        "git-cloned facebookresearch/vjepa2 code, so it gets its own flag to turn off without "
        "giving up the decoder's compile if a future upstream change starts breaking graphs.",
    )
    parser.add_argument(
        "--auto-weight-every", type=int, default=None,
        help="Only relevant under --log-activation-grad-norms, which forces CodecLoss onto its "
        "probing fallback for the adaptive loss weights. There each factor costs a real extra "
        "backward pass (through all of VGG, and through the whole frozen V-JEPA backbone), "
        "measured at ~575ms per micro-step at 448x768x40 -- more than the entire rest of the "
        "micro-step. This re-probes every N micro-steps instead of every one, reusing the last "
        "factor in between. Default 1 (exact). Without that flag the factors are computed exactly "
        "and for ~3ms, and this does nothing.",
    )
    parser.add_argument(
        "--perceptual-chunk-size", type=int, default=None,
        help="Frames per LPIPS/DINO loss forward (0 processes the selected frames together). Default 0",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=None, help="Default: steps // 20")
    parser.add_argument("--lr-decay-steps", type=int, default=None, help="Default: steps - warmup_steps")
    parser.add_argument("--lr-min", type=float, default=None, help="Default: --lr * 0.01")
    parser.add_argument(
        "--loss-mae-weight", type=float, default=None, help="CodecLossWeights.loss_mae. Default 1.0"
    )
    parser.add_argument(
        "--reconstruction-loss", choices=["l1", "l2"], default=None,
        help="Pixel-space distance for the loss_mae term. Default l1 (mira's own recipe). l2 won "
        "PSNR by ~2dB over l1 in a real 1000-step single-clip sweep on this architecture, at some "
        "cost in LPIPS -- see CodecLossSchedule's docstring for the full numbers.",
    )
    parser.add_argument(
        "--perceptual-warmup-steps", type=int, default=None,
        help="Ramp the LPIPS and DINO-consistency weights linearly from 0 to their full value "
        "over this many steps, training on the pixel term alone before that (see "
        "mini_mira.codec.loss.CodecLossSchedule). Default 0 -- disabled, weights full from step "
        "0, exactly today's behavior. Steps during the ramp are also cheaper: a zero-weight term "
        "is skipped, not computed and multiplied by zero.",
    )
    parser.add_argument(
        "--log-activation-grad-norms", action=argparse.BooleanOptionalAction, default=None,
        help="Log grad_norm_loss_mae/lpips_perceptual/dino_latent_consistency/total_video (the "
        "ORIGINAL per-term activation-gradient hooks, CodecLoss._hook_clone) -- off by default. "
        "notes/grad_norm_investigation.md found these GradScaler-scale-confounded under "
        "--precision fp16-hybrid and generally less directly interpretable than "
        "grad_norm_params_total (always on) or --log-per-term-grad-norm's real parameter "
        "gradients. Costs a real extra tensor clone per term per micro-step -- opt-in.",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints_vjepa")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Default 100")
    parser.add_argument(
        "--hf-backup-every", type=int, default=None,
        help="How often (in steps) to upload to --hf-backup-repo, independent of --checkpoint-"
        "every's local-save cadence. Must be a multiple of --checkpoint-every to behave "
        "predictably -- the upload check only runs on steps where a local save already happened, "
        "so e.g. hf-backup-every=100 with checkpoint-every=50 uploads every 2nd local save; a "
        "non-multiple would silently skip some intended upload steps. Default: same as "
        "--checkpoint-every (upload on every local save, the original behavior). Useful when the "
        "upload itself is slow -- frequent cheap local safety saves without paying the upload cost "
        "every time.",
    )
    parser.add_argument("--preview-every", type=int, default=None, help="W&B image/video preview interval. Default 100")
    parser.add_argument("--console-log-every", type=int, default=None, help="Loss/GPU-memory print interval. Default 10")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reset-lr-schedule", action="store_true",
        help="With --resume, restart the LR scheduler's own step clock at 0 instead of continuing "
        "from the checkpoint's step count -- use when --resume starts a new fine-tune phase "
        "(different --lr/--lr-min/--steps) rather than continuing an interrupted run.",
    )
    parser.add_argument(
        "--log-per-term-grad-norm", action=argparse.BooleanOptionalAction, default=None,
        help="Log each loss term's OWN real parameter-gradient norm separately (grad_norm_params_"
        "loss_mae/loss_lpips_perceptual/loss_dino_latent_consistency), not just their combined "
        "total (grad_norm_params_total) -- shows which term is actually driving the weight update "
        "most. Costs ~3 extra backward-equivalent passes on the last micro-step of every optimizer "
        "step (same 'last micro-step only' convention as grad_norm_loss_*), so opt-in rather than "
        "always-on: skip this for real long runs where every bit of step time matters, use it for "
        "runs where understanding per-term contribution is the actual point.",
    )
    parser.add_argument(
        "--reset-optimizer-state", action="store_true",
        help="With --resume, load weights only and start AdamW with fresh (zero) momentum/variance "
        "instead of the checkpoint's saved optimizer state -- use when the checkpoint's momentum "
        "was calibrated under a different regime (resolution/precision) that may not transfer "
        "cleanly. Independent of --reset-lr-schedule (that resets the LR curve's shape, this "
        "resets AdamW's own state); most restarts of this kind want both.",
    )
    parser.add_argument(
        "--wandb-new-run", action="store_true",
        help="With --resume, start a fresh W&B run instead of continuing the checkpoint's saved one.",
    )
    parser.add_argument(
        "--precision", choices=["fp16-hybrid", "bf16"], default=None,
        help="Default fp16-hybrid (unchanged): float16 autocast + GradScaler, convolutions "
        "force-patched to bf16 -- the proven V100 setup (notes/gpu_amp_investigation.md). bf16: "
        "plain bfloat16 autocast everywhere, GradScaler disabled (a documented no-op), no conv "
        "patch needed. Opt-in only -- test against fp16-hybrid before switching a run over.",
    )
    parser.add_argument(
        "--profile-steps", type=int, default=0,
        help="Run torch.profiler over this many steps (after --profile-wait warmup steps) and "
        "dump a Chrome trace + kernel table under --checkpoint-dir/profile. 0 (default) leaves "
        "the profiler off; per-step wall-clock timing is always on either way.",
    )
    parser.add_argument("--profile-wait", type=int, default=5, help="Warmup steps before --profile-steps starts")
    parser.add_argument("--hf-backup-repo", default=None, help="Optional HF Hub repo to back up checkpoints to")
    parser.add_argument("--wandb-project", default=None)
    return parser.parse_args()


def build_next_video(args: argparse.Namespace):
    """Real streamed clips if --index-path is set, else one fixed synthetic video every step.

    Either way this returns clips already on the GPU, already at (--height, --width) and in
    [0, 1] -- the training loop never touches a CPU tensor. For the real-data path the
    uint8->float conversion and the pad+resize happen on the GPU, and the host->device copy runs
    a few batches ahead on its own stream; see mini_mira.codec.data_pipeline for why.
    """
    if args.index_path:
        loader = create_loader(
            index_path=args.index_path,
            clip_len=args.frames,
            target_fps=args.target_fps,
            n_players=1,  # codec training
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            frame_size=None,  # native decode -- resize_to_canonical does the pad+resize on GPU
        )
        data_iter = iter(loader)

        def fetch_uint8() -> torch.Tensor:
            batch, _metadata = next(data_iter)  # codec ignores actions
            return batch.video  # uint8 (B,T,C,H,W), native decode resolution

        stream = PrefetchingVideoStream(fetch_uint8, args.height, args.width)
        return lambda: next(stream)

    fixed_video = torch.rand(1, args.frames, 3, args.height, args.width, device="cuda")
    return lambda: fixed_video


def main() -> None:
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    args = parse_args()
    run_config = load_run_config(args.run_config, CodecRunConfig) if args.run_config else CodecRunConfig()
    apply_run_config(args, run_config)

    config = load_pipeline_config(args.config)
    # Real, previously-hit-for-real requirement (notes/vjepa_next_session.md): the decoder's own
    # temporal/spatial upsample only exactly reconstructs --height/--width if both are divisible
    # by decoder.patch_size * bottleneck.stride -- otherwise it silently reconstructs a slightly
    # different size (e.g. 720 -> 704), surfacing later as a confusing loss.py shape-mismatch
    # crash instead of a clear error here. Purely additive: every already-proven-valid real
    # launch already satisfies this, so this never fires for a config that already works.
    required_divisor = config.decoder.patch_size * config.bottleneck.stride
    assert args.height % required_divisor == 0 and args.width % required_divisor == 0, (
        f"--height {args.height} / --width {args.width} must both be divisible by "
        f"{required_divisor} (decoder.patch_size {config.decoder.patch_size} * bottleneck.stride "
        f"{config.bottleneck.stride})"
    )

    torch.manual_seed(0)
    # cuDNN autotunes each convolution's algorithm on first sight of a shape. Every shape here is
    # fixed for the whole run (one clip geometry, one batch size), so the one-off autotune cost is
    # paid once and the better algorithm is then used for every remaining step.
    torch.backends.cudnn.benchmark = True
    # Only affects the few genuinely-fp32 matmuls left under bf16 autocast (the qk-norm path, the
    # auto_weight probes); everything on the hot path is already bf16.
    torch.set_float32_matmul_precision("high")

    # V-JEPA 2.1 has only one real variant (ViT-B), so no per-variant model-name/layer-dict lookup
    # like DinoModel's -- DEFAULT_VJEPA_LAYERS is already the right constant, no subscript needed.
    vjepa = VjepaModel(
        require_pretrained=args.require_pretrained_vjepa,
        last_layer_only=False, layer_indices=DEFAULT_VJEPA_LAYERS,
    ).cuda()
    vjepa.eval()
    bottleneck = MyBottleneck(config.bottleneck, use_checkpointing=args.activation_checkpointing).cuda()
    decoder = ViTVideoDecoder(config.decoder, use_checkpointing=args.activation_checkpointing).cuda()

    # Built with the perceptual terms at their FULL target weights, always -- CodecLoss only
    # constructs its LPIPS module when that weight is already > 0, so the schedule below has to
    # ramp down from the target rather than up from zero (see CodecLossSchedule's docstring).
    loss_fn = CodecLoss(
        CodecLossWeights(auto_weight=True, loss_mae=args.loss_mae_weight,
                         auto_weight_every=args.auto_weight_every,
                         reconstruction_loss=args.reconstruction_loss),
        use_checkpointing=args.activation_checkpointing,
        perceptual_chunk_size=args.perceptual_chunk_size,
        log_activation_grad_norms=args.log_activation_grad_norms,
    ).cuda()
    loss_schedule = CodecLossSchedule(
        perceptual_warmup_steps=args.perceptual_warmup_steps,
        lpips_target=loss_fn.weights.loss_lpips_perceptual,
        dino_target=loss_fn.weights.loss_dino_latent_consistency,
    )
    loss_fn.bind_encoder_dino(vjepa)
    loss_fn.bind_last_layer(decoder.last_layer_weight)
    loss_fn.use_channels_last_perceptual()
    # This flag differentiates individual loss terms again, after CodecLoss has already taken
    # their gradients once, so their graphs have to survive the call.
    loss_fn.retain_term_graphs = bool(args.log_per_term_grad_norm)

    # vjepa isn't included here: VjepaModel.dino_forward already wraps its whole body in its own
    # bf16 autocast (vjepa.py), so every conv inside it is already covered -- same reasoning as
    # DinoModel's own loader. bottleneck/decoder/loss_fn have no autocast of their own, so they
    # still need the monkey-patch. Only needed under fp16-hybrid: with --precision bf16 the
    # ambient autocast is already bfloat16, so forcing a nested bf16 override here would be a
    # no-op, not a fix.
    if args.precision == "fp16-hybrid":
        for module in (bottleneck, decoder, loss_fn):
            _keep_convolutions_in_bf16(module)

    # Compilation happens AFTER the fp16-hybrid monkey-patch above, not before -- torch.compile
    # needs to see each module's real final forward, not have it swapped out from under it
    # afterward.
    #
    # Everything here uses Module.compile() (in-place) rather than the module = torch.compile(...)
    # rebinding form, for two reasons:
    #
    #   - torch.compile() returns an OptimizedModule WRAPPER, which prefixes every state_dict key
    #     with "_orig_mod." -- the long-standing checkpoint-compatibility hazard this script's
    #     comments used to flag as an unverified risk of --compile. Compiling in place sidesteps
    #     it entirely: the module keeps its identity and its key names, so a checkpoint saved
    #     under --compile loads without it and vice versa.
    #   - it lets compilation be applied at the granularity that is actually wanted (below).
    if args.compile:
        # Inductor's "donated buffer" optimization assumes a compiled backward is entered exactly
        # once, and hard-errors on the second entry ("compiled with non-empty donated buffers
        # which requires ... retain_graph=False"). Two opt-in paths here do differentiate the same
        # graph more than once -- CodecLoss's probing fallback under --log-activation-grad-norms,
        # and --log-per-term-grad-norm -- so it stays off. Leaving it on is what made --compile a
        # net LOSS before this: the probes crashed out of, or fell off, the compiled path.
        torch._functorch.config.donated_buffer = False

        # Compile the decoder's transformer BLOCKS, not the decoder as a whole. An
        # inductor-compiled module is one opaque autograd node, and CodecLoss's adaptive weights
        # need the gradient of each loss term at the decoder's LAST LAYER -- so with the whole
        # decoder as a single graph, reading a weight that sits at its very end means running the
        # entire decoder backward, three times per micro-step, for a quantity that depends only on
        # the output head. Compiling per block leaves the head (norm_out + patch_unembed + tanh)
        # in eager autograd, where that gradient is one matmul, and gives up essentially none of
        # the speedup: what inductor wins here is the fusion of each block's own memory-bound
        # elementwise chain (RoPE, per-head qk-norm, LayerScale, the residual adds), all of which
        # is within-block. Measured at 448x768x40: compiling the whole decoder made the step 1.6x
        # SLOWER than compiling the blocks.
        bottleneck.compile()
        for block in decoder.blocks:
            block.compile()
    compile_encoder = bool(args.compile) if args.compile_encoder is None else args.compile_encoder
    if compile_encoder:
        # Only the transformer body, not VjepaModel.dino_forward itself: dino_forward's own
        # normalize/interpolate/rearrange wrapper is cheap, and leaving it out of the graph keeps
        # the compiled region to the one shape-stable piece that actually benefits.
        vjepa.encoder.compile()

    params = list(bottleneck.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    # GradScaler(enabled=False) is a documented no-op: .scale() returns the loss unchanged,
    # .step() just calls optimizer.step(), .update() no-ops -- so the training loop below needs
    # no branching to support both precisions. bf16 doesn't need loss scaling at all.
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16-hybrid"))
    loss_fn.bind_grad_scaler(grad_scaler)

    # mira's own schedule shape: warmup ~5% of args.steps, then cosine decay for the rest, no
    # constant plateau -- proportional to args.steps rather than mira's literal 1000/249000
    # (those were sized for its 250,001-step run).
    warmup_steps = args.lr_warmup_steps if args.lr_warmup_steps is not None else max(1, args.steps // 20)
    decay_steps = args.lr_decay_steps if args.lr_decay_steps is not None else max(1, args.steps - warmup_steps)
    min_lr = args.lr_min if args.lr_min is not None else args.lr * 0.01
    lr_scheduler = WarmupConstantCosineDecayLR(
        optimizer, warmup_steps=warmup_steps, constant_steps=0, decay_steps=decay_steps, min_lr=min_lr,
    )

    next_video = build_next_video(args)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "checkpoint_vjepa.pth"
    hf_backup_every = args.hf_backup_every if args.hf_backup_every is not None else args.checkpoint_every

    start_step = 0
    wandb_run_id = None
    if args.resume and ckpt_path.exists():
        start_step, wandb_run_id = load_checkpoint(ckpt_path, bottleneck, decoder, optimizer, lr_scheduler, grad_scaler)
        if args.wandb_new_run:
            wandb_run_id = None
        if args.reset_optimizer_state:
            # Fresh AdamW state (no momentum/variance carried over) bound to the SAME just-loaded
            # weights -- load_checkpoint above already loaded the OLD optimizer state into the
            # existing `optimizer` object; replacing it here discards that state entirely rather
            # than editing it in place. Only the optimizer's own state resets -- loaded model
            # weights and start_step are untouched. lr_scheduler was constructed bound to the old
            # optimizer object (torch.optim.lr_scheduler stores this as self.optimizer), so it
            # needs to be pointed at the new one or every later lr_scheduler.step()/get_lr() call
            # would silently keep operating on the discarded optimizer instead.
            optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
            lr_scheduler.optimizer = optimizer
        # Two real bugs, verified directly, not just reasoned about:
        # 1. load_state_dict only updates the scheduler's own bookkeeping -- it never pushes a
        #    recomputed lr into optimizer.param_groups. Every --resume was silently leaving lr at
        #    whatever the freshly-constructed scheduler set at its OWN init (near-zero, deep in
        #    warmup), not the correct value for the step actually being resumed from.
        # 2. load_state_dict also restores the checkpoint's own warmup_steps/decay_steps/min_lr
        #    wholesale, not just its step counter -- if --steps grew (extending a finished short
        #    run, not resuming an interrupted one), that stale shape thinks decay already
        #    finished, pinning lr at min_lr for the entire extension.
        # Re-apply this run's own shape, then recompute+apply the real lr for the resumed step --
        # a no-op for #2 when --steps didn't change, but always fixes #1 either way.
        # load_state_dict above restores base_lrs from whatever --lr the CHECKPOINT's own run
        # used, and get_lr() reads self.base_lrs directly for every phase (warmup/constant/decay
        # peak). Re-derive it from THIS run's own --lr unconditionally, not just under
        # --reset-lr-schedule -- previously this was only done inside that branch, while --lr-min
        # (below, via min_lr) was ALREADY being re-derived from the current args unconditionally,
        # so a plain --resume with a different --lr than the checkpoint's original silently kept
        # training at the OLD peak lr against a NEW floor -- a self-contradictory decay curve with
        # no error. Matches train_world_model.py's own resume block, which does this
        # unconditionally for the same reason ("this script's own --lr is always meant to be
        # authoritative"). A no-op when --lr matches the checkpoint's original either way.
        lr_scheduler.base_lrs = [args.lr for _ in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["initial_lr"] = args.lr
        if args.reset_lr_schedule:
            # --resume's load_state_dict above restored last_epoch to the checkpoint's own step
            # (e.g. ~3100). Even after warmup_steps/decay_steps/min_lr are re-applied below,
            # get_lr() evaluates them against THIS stale clock -- for a schedule sized for a short
            # new phase, that lands past the end, pinned at min_lr, not a fresh curve. Use 0, not
            # -1: get_lr() is called directly below (not via .step()), and -1 would produce a
            # negative warmup-phase lr (base_lr * -1 / warmup_steps). Only the schedule's own
            # timing reference resets -- start_step (outer loop / checkpoint bookkeeping) and the
            # loaded weights/optimizer momentum are untouched.
            lr_scheduler.last_epoch = 0
            # warmup_steps/decay_steps above were sized off args.steps directly (e.g. 3299, the
            # RESUMED run's absolute final step) -- correct for a fresh run, wrong here: last_epoch
            # just reset to 0, so the scheduler's own clock only ever advances through THIS phase's
            # actual length (args.steps - start_step, e.g. 200), never anywhere near 3299. Sizing
            # decay_steps off the absolute count spreads the intended curve over ~16x too many
            # steps -- confirmed directly: a 5-step console sample showed lr flat at the same
            # printed value the whole time, consistent with a decay curve sized for a much longer
            # run. Recompute both relative to the phase actually being run, but only when the
            # caller didn't already explicitly pin one (an explicit --lr-warmup-steps/
            # --lr-decay-steps is a deliberate choice, not something to override).
            phase_steps = args.steps - start_step
            if args.lr_warmup_steps is None:
                warmup_steps = max(1, phase_steps // 20)
            if args.lr_decay_steps is None:
                decay_steps = max(1, phase_steps - warmup_steps)
        # Only reapply warmup_steps/decay_steps/min_lr when there's an actual reason to: either
        # --reset-lr-schedule (a genuinely new phase, sized above) or the caller explicitly passed
        # the corresponding flag (a deliberate override mid-phase). A PLAIN resume with neither --
        # e.g. splitting one long phase across multiple --steps invocations because the GPU is only
        # available in windows -- must NOT touch these: load_checkpoint above already restored the
        # checkpoint's own values, and warmup_steps/decay_steps computed earlier in this function
        # are sized off THIS invocation's own (possibly much larger or smaller) absolute --steps,
        # not the original phase length. Overwriting them unconditionally (the previous behavior)
        # silently changed the decay curve's shape mid-phase -- last_epoch keeps advancing from
        # where the checkpoint left off, but decay_steps would jump to a value sized for a whole
        # different phase, moving the LR far from where the smooth curve actually was.
        if args.reset_lr_schedule or args.lr_warmup_steps is not None:
            lr_scheduler.warmup_steps = warmup_steps
        if args.reset_lr_schedule or args.lr_decay_steps is not None:
            lr_scheduler.decay_steps = decay_steps
        if args.reset_lr_schedule or args.lr_min is not None:
            lr_scheduler.min_lr = min_lr
        for group, lr in zip(optimizer.param_groups, lr_scheduler.get_lr()):
            group["lr"] = lr
        print(f"Resumed from {ckpt_path} at step {start_step}")

    if args.steps <= start_step:
        raise SystemExit(
            f"--steps ({args.steps}) must be greater than the resumed step ({start_step}) -- "
            f"otherwise range(start_step, args.steps) runs zero iterations and the script exits "
            f"looking successful having trained nothing. Did you mean --steps {start_step + 200}?"
        )

    # wandb_run_id: None on a fresh run (wandb.init mints a new one, captured below), the id
    # loaded from the checkpoint above (continues that SAME wandb run instead of --resume
    # silently fragmenting the loss/lr history into a new, disconnected one), or forced back to
    # None by --wandb-new-run above when a fresh chart is wanted despite resuming training state.
    wandb_enabled = init_wandb(args.wandb_project, vars(args), run_id=wandb_run_id)
    wandb_run_id = get_wandb_run_id(wandb_enabled)
    torch.cuda.reset_peak_memory_stats()

    # Baseline for weight-drift logging below -- the actual starting point of THIS run (post-resume
    # if --resume, post-init otherwise), moved to CPU so it doesn't compete with training for GPU
    # memory. See notes/grad_norm_investigation.md: the existing grad_norm_loss_* metrics are
    # activation gradients (CodecLoss._hook_clone), not parameter gradients, and can't answer "are
    # the weights actually moving" on their own -- this does, directly.
    initial_params = torch.cat([p.detach().flatten() for p in params]).cpu()

    def _save_checkpoint_now(step: int, *, force_hf_upload: bool = False) -> None:
        # L2 distance from this run's own starting weights (initial_params above) -- an
        # unambiguous "did the model change at all" signal, independent of any
        # gradient-interpretation question. Only at checkpoint cadence, not every step: needs
        # a full param-vector CPU round-trip, not cheap enough for the hot loop.
        current_params = torch.cat([p.detach().flatten() for p in params]).cpu()
        weight_drift_l2 = (current_params - initial_params).norm().item()
        print(f"step {step}: weight_drift_l2_from_run_start={weight_drift_l2:.6e}")
        log_step(
            wandb_enabled, step, {"weight_drift_l2_from_run_start": weight_drift_l2},
            optimizer.param_groups[0]["lr"],
        )
        del current_params
        save_checkpoint(ckpt_path, step, bottleneck, decoder, optimizer, lr_scheduler, grad_scaler, wandb_run_id)
        # Decoupled from the local save above: local saves are cheap and want to be frequent
        # for crash safety, but the HF upload itself can be slow (network-bound), so it runs on
        # its own, sparser cadence -- always still fires on the last step or force_hf_upload
        # (set only by the SIGTERM/SIGINT shutdown path below, this run's last chance to upload)
        # so the final state is never skipped regardless of where hf_backup_every's modulo lands.
        is_last_step = step == args.steps - 1
        if args.hf_backup_repo and (force_hf_upload or (step + 1) % hf_backup_every == 0 or is_last_step):
            from huggingface_hub import HfApi  # optional dep, only used here

            HfApi().upload_file(
                path_or_fileobj=str(ckpt_path), path_in_repo="checkpoint_vjepa.pth",
                repo_id=args.hf_backup_repo, repo_type="model",
            )

    timer = StepTimer()
    profile_dir = checkpoint_dir / "profile"
    with profile_window(profile_dir, args.profile_steps, args.profile_wait) as profiler_step:
        for step in range(start_step, args.steps):
            # Before the micro-step loop, so every micro-step in one optimizer step is scored
            # under the same weights. Keyed off the absolute step, so --resume lands exactly
            # where the ramp left off rather than restarting it.
            loss_schedule.apply(loss_fn.weights, step)
            optimizer.zero_grad()
            # Held as GPU tensors through the micro-step loop, not Python floats -- v.item() forces a
            # full CUDA sync (blocks the CPU until every queued kernel finishes), and doing that once
            # per loss term per micro-step (found for real: nvidia-smi dmon showed the GPU alternating
            # 0%/100% every second, ~60% average, matching a real-world 57% finding from an earlier
            # session) serializes what should be an async pipeline. Converted to floats once, after
            # the loop, instead -- same final numbers, far fewer sync points (was
            # num_loss_terms * grad_accum_steps per step, now num_loss_terms once).
            accumulated: dict[str, Tensor] = {}
            per_term_grad_norms: dict[str, float] = {}
            for micro_step in range(args.grad_accum_steps):
                with timer.phase("data"):
                    video = next_video()
                with timer.phase("forward"), _autocast(args.precision):
                    with torch.no_grad():  # encoder side: no grad needed here
                        dino_features = vjepa.dino_forward(video)
                    z = bottleneck(dino_features)
                    reconstructed = decoder(z)
                    outputs = CodecOutputs(
                        input_video=normalize_video(video), output_video=reconstructed, dino_features=dino_features
                    )
                    losses = loss_fn(outputs)
                if args.log_per_term_grad_norm and micro_step == args.grad_accum_steps - 1:
                    # Each term's OWN real parameter-gradient norm, computed independently via
                    # autograd.grad (doesn't touch .grad, doesn't affect the real training step below).
                    # Must run BEFORE the real .backward() call: that call frees the graph by default,
                    # and every one of these needs it still intact -- retain_graph=True on each so the
                    # next one (and the real backward() after) can still use it. allow_unused=True
                    # since a term with a zero weight (loss_mae/lpips/dino_latent_consistency can each
                    # be individually disabled via config) has no graph to differentiate through at
                    # all. Unlike grad_norm_params_total, these need no unscale_() step: they operate
                    # directly on the raw, not-yet-GradScaler-multiplied loss values (scaling only
                    # happens at the .backward() call below), so they're precision-comparable by
                    # construction -- see notes/grad_norm_investigation.md for why that matters here.
                    for term_name in ("loss_mae", "loss_lpips_perceptual", "loss_dino_latent_consistency"):
                        if term_name not in losses:
                            continue
                        term_grads = torch.autograd.grad(losses[term_name], params, retain_graph=True, allow_unused=True)
                        term_grads = [g for g in term_grads if g is not None]
                        norm = torch.norm(torch.stack([g.norm() for g in term_grads])).item() if term_grads else 0.0
                        per_term_grad_norms[f"grad_norm_params_{term_name}"] = norm
                with timer.phase("backward"):
                    grad_scaler.scale(losses["loss_total"] / args.grad_accum_steps).backward()
                for k, v in losses.items():
                    term = v.detach() / args.grad_accum_steps
                    accumulated[k] = term if k not in accumulated else accumulated[k] + term
            # Single sync point per loss term for the whole step, not one per term per micro-step --
            # see the comment on `accumulated`'s declaration above for why that matters.
            accumulated_floats: dict[str, float] = {k: v.item() for k, v in accumulated.items()}
            # Real total gradient norm across every trainable parameter (bottleneck + decoder) --
            # unlike grad_norm_loss_* below, this is an actual parameter gradient, the quantity
            # "is AdamW's update small" genuinely depends on (notes/grad_norm_investigation.md).
            # unscale_ first: under --precision fp16-hybrid, .grad is still GradScaler-scaled at this
            # point (unscaling normally only happens inside grad_scaler.step()) -- without this the
            # logged norm would repeat the exact scale-factor confusion already found and corrected for
            # the activation-gradient probes. A documented no-op under bf16 (scaler disabled). clip
            # with max_norm=inf so this only ever measures the norm, never actually clips gradients --
            # clipping isn't something this project has decided to do.
            with timer.phase("optimizer"):
                grad_scaler.unscale_(optimizer)
                grad_norm_params_total = torch.nn.utils.clip_grad_norm_(params, float("inf")).item()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                lr_scheduler.step()

            # Checked here, right after a step fully lands and before any of this step's own
            # (potentially slow) logging/eval work -- prioritizes actually getting the save done
            # inside `timeout`'s kill-after grace period over finishing this step's own work first.
            if _shutdown_requested:
                print(f"Shutdown signal received at step {step} -- saving and exiting cleanly.")
                _save_checkpoint_now(step, force_hf_upload=True)
                if wandb_enabled:
                    import wandb  # noqa: PLC0415

                    wandb.finish()
                return

            # Last micro-step's real per-term gradient norms (see CodecLoss._hook_clone) -- same
            # "last micro-step only" convention already used for the preview video below.
            grad_norms = {f"grad_norm_{k}": v.item() for k, v in loss_fn.backward_metrics.items()}
            grad_norms["grad_norm_params_total"] = grad_norm_params_total
            grad_norms.update(per_term_grad_norms)  # empty dict unless --log-per-term-grad-norm
            # The schedule's current weights, so the ramp is visible as a curve rather than only
            # inferable from the loss terms it silently switches on. Constant when
            # --perceptual-warmup-steps is 0 (the default).
            grad_norms["weight_lpips_perceptual"] = loss_fn.weights.loss_lpips_perceptual
            grad_norms["weight_dino_latent_consistency"] = loss_fn.weights.loss_dino_latent_consistency

            timer.end_step()
            profiler_step()

            current_lr = optimizer.param_groups[0]["lr"]
            term_str = ", ".join(
                f"{k}={v:.4f}" for k, v in accumulated_floats.items() if k != "loss_total" and not k.endswith("_auto_w")
            )
            if (step + 1) % args.console_log_every == 0 or step == start_step:
                print(f"step {step}: lr={current_lr:.2e} loss_total={accumulated_floats['loss_total']:.4f} ({term_str})")
                print(
                    f"cuda_peak_allocated={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
                    f"cuda_peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB"
                )
                print(timer.report(args.batch_size * args.grad_accum_steps, args.frames))
                # :.6e, not :.4f -- a genuinely small-but-nonzero grad norm (plausible this deep into
                # training, on an already-converged checkpoint) rounds to a misleading "0.0000" at 4
                # decimal places, same illusion this project already hit once with the PSD loss print.
                print(", ".join(f"{k}={v:.6e}" for k, v in grad_norms.items()))
            log_step(wandb_enabled, step, {**accumulated_floats, **grad_norms}, current_lr)

            is_last = step == args.steps - 1
            if (step + 1) % args.checkpoint_every == 0 or is_last:
                _save_checkpoint_now(step)
            # Always persist the first completed step so a new run proves its output path before
            # committing hours of compute. Subsequent samples follow the configured interval.
            if step == start_step or (step + 1) % args.preview_every == 0 or is_last:
                log_preview(
                    wandb_enabled, step, video, reconstructed, fps=args.target_fps,
                    output_dir=checkpoint_dir / "previews",
                )


if __name__ == "__main__":
    main()
