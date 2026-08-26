"""Config-driven codec training: DinoModel -> MyBottleneck -> ViTVideoDecoder, trained on the
real three-term CodecLoss (L1 + LPIPS + DINO latent-consistency, with auto_weight balancing --
see mini_mira.codec.loss), matching mira's own CodecLoss and its shipped config.

Two data modes:
  - Default (no --index-path): trains on one fixed synthetic video -- a mechanism check, not
    real training (see verify_codec_training.py for the CPU-friendly version of this).
  - --index-path <dir>: streams real clips via mira's own create_loader. Point it at the path
    scripts/download_shards.py prints after downloading.

Assumes a CUDA GPU (no CPU fallback) -- this script is for real training runs.

Precision: float16 autocast for the trainable codec, with its convolution-family modules and
the complete frozen DINO backbones kept in bfloat16. This avoids this V100/cuDNN stack's FP16
convolution engine failures and DINO's FP16 numerical instability.

Effective batch size is --batch-size (the real, per-forward micro-batch) times
--grad-accum-steps -- see mira.training.lr_schedule.WarmupConstantCosineDecayLR for the LR
schedule shape, and --activation-checkpointing to trade compute for the memory a larger
micro-batch would need.
"""

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Real mira, cloned as a sibling repo -- reused for the dataloader.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import torch
from mira.data.training_loader import create_loader
from mira.training.lr_schedule import WarmupConstantCosineDecayLR

from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.checkpoint import load_checkpoint, save_checkpoint
from mini_mira.codec.decoder import ViTVideoDecoder
from mini_mira.codec.dino import DEFAULT_ENCODER_AGGREGATION_LAYERS, DinoModel
from mini_mira.codec.logging_utils import get_wandb_run_id, init_wandb, log_preview, log_step
from mini_mira.codec.loss import CodecLoss, CodecLossWeights, CodecOutputs, normalize_video
from mini_mira.codec.video_prep import resize_to_canonical
from mini_mira.ml.config_loading import apply_run_config, load_pipeline_config, load_run_config
from mini_mira.ml.run_config import CodecRunConfig


def _autocast(precision: str) -> torch.autocast:
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


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
    parser.add_argument("--require-pretrained-dino", action="store_true")
    parser.add_argument("--index-path", default=None, help="Real dataset dir from download_shards.py")
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-forward micro-batch size. Default 4")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="Micro-batches accumulated per optimizer step (effective batch = batch-size * this). Default 1",
    )
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--perceptual-chunk-size", type=int, default=None,
        help="Frames per LPIPS/DINO loss forward (0 processes the selected frames together). Default 0",
    )
    parser.add_argument(
        "--perceptual-dino-model", default=None,
        help="If set, score the DINO-consistency loss in a separate (typically smaller) DINO "
        "variant's feature space instead of the encoder's own, e.g. dinov3_vits16",
    )
    parser.add_argument(
        "--perceptual-dino-multilayer", action=argparse.BooleanOptionalAction, default=None,
        help="Aggregate multiple DINO layers for the consistency loss (matches mira's "
        "DinoPerceptualLoss) instead of just the last one. Requires --perceptual-dino-model.",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=None, help="Default: steps // 20")
    parser.add_argument("--lr-decay-steps", type=int, default=None, help="Default: steps - warmup_steps")
    parser.add_argument("--lr-min", type=float, default=None, help="Default: --lr * 0.01")
    parser.add_argument(
        "--loss-mae-weight", type=float, default=None, help="CodecLossWeights.loss_mae. Default 1.0"
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
    parser.add_argument("--checkpoint-dir", default="checkpoints")
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
    parser.add_argument("--hf-backup-repo", default=None, help="Optional HF Hub repo to back up checkpoints to")
    parser.add_argument("--wandb-project", default=None)
    return parser.parse_args()


def build_next_video(args: argparse.Namespace):
    """Real streamed clips if --index-path is set, else one fixed synthetic video every step."""
    if args.index_path:
        loader = create_loader(
            index_path=args.index_path,
            clip_len=args.frames,
            target_fps=args.target_fps,
            n_players=1,  # codec training
            batch_size=args.batch_size,
            frame_size=None,  # native decode -- resize_to_canonical below does the pad+resize
        )
        data_iter = iter(loader)

        def next_video() -> torch.Tensor:
            batch, _metadata = next(data_iter)  # codec ignores actions
            video = batch.video.float() / 255.0  # uint8 (B,T,C,H,W) -> float [0,1]
            return resize_to_canonical(video, args.height, args.width)
    else:
        fixed_video = torch.rand(1, args.frames, 3, args.height, args.width)

        def next_video() -> torch.Tensor:
            return fixed_video

    return next_video


def main() -> None:
    args = parse_args()
    run_config = load_run_config(args.run_config, CodecRunConfig) if args.run_config else CodecRunConfig()
    apply_run_config(args, run_config)

    config = load_pipeline_config(args.config)

    torch.manual_seed(0)
    # dinov3_vitb16 is the only variant with weights downloaded for this project (real mira's
    # encoder always uses vitl16) -- matches mira's RAEEncoder in every other respect: multi-layer
    # aggregation on, layer_indices from DEFAULT_ENCODER_AGGREGATION_LAYERS.
    encoder_dino_model = "dinov3_vitb16"
    dino = DinoModel(
        dino_model=encoder_dino_model, require_pretrained=args.require_pretrained_dino,
        last_layer_only=False, layer_indices=DEFAULT_ENCODER_AGGREGATION_LAYERS[encoder_dino_model],
    ).cuda()
    dino.eval()
    bottleneck = MyBottleneck(config.bottleneck, use_checkpointing=args.activation_checkpointing).cuda()
    decoder = ViTVideoDecoder(config.decoder, use_checkpointing=args.activation_checkpointing).cuda()

    loss_fn = CodecLoss(
        CodecLossWeights(auto_weight=True, loss_mae=args.loss_mae_weight),
        use_checkpointing=args.activation_checkpointing,
        perceptual_chunk_size=args.perceptual_chunk_size,
        log_activation_grad_norms=args.log_activation_grad_norms,
    ).cuda()
    loss_fn.bind_encoder_dino(dino)
    if args.perceptual_dino_model:
        perceptual_dino = DinoModel(
            dino_model=args.perceptual_dino_model, require_pretrained=args.require_pretrained_dino,
            last_layer_only=not args.perceptual_dino_multilayer,
        ).cuda()
        perceptual_dino.eval()
        loss_fn.bind_perceptual_dino(perceptual_dino)
    loss_fn.bind_last_layer(decoder.last_layer_weight)

    # dino isn't included here: DinoModel.dino_forward already wraps its whole body in its own
    # bf16 autocast (dino.py), so every conv inside it is already covered -- same reason
    # perceptual_dino (also a DinoModel, bound above when --perceptual-dino-model is set) was
    # never in this list either. bottleneck/decoder/loss_fn have no autocast of their own, so
    # they still need the monkey-patch. Two mechanisms covering the same convs was harmless
    # (nested same-dtype autocasts don't conflict) but redundant enough to be worth removing
    # rather than leaving as a trap for whoever touches this next.
    # Only needed under fp16-hybrid: with --precision bf16 the ambient autocast is already
    # bfloat16, so forcing a nested bf16 override here would be a no-op, not a fix.
    if args.precision == "fp16-hybrid":
        for module in (bottleneck, decoder, loss_fn):
            _keep_convolutions_in_bf16(module)

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
    ckpt_path = checkpoint_dir / "checkpoint.pth"
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

    for step in range(start_step, args.steps):
        optimizer.zero_grad()
        accumulated: dict[str, float] = {}
        per_term_grad_norms: dict[str, float] = {}
        for micro_step in range(args.grad_accum_steps):
            video = next_video().cuda()
            with _autocast(args.precision):
                with torch.no_grad():  # encoder side: no grad needed here
                    dino_features = dino.dino_forward(video)
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
            grad_scaler.scale(losses["loss_total"] / args.grad_accum_steps).backward()
            for k, v in losses.items():
                accumulated[k] = accumulated.get(k, 0.0) + v.item() / args.grad_accum_steps
        # Real total gradient norm across every trainable parameter (bottleneck + decoder) --
        # unlike grad_norm_loss_* below, this is an actual parameter gradient, the quantity
        # "is AdamW's update small" genuinely depends on (notes/grad_norm_investigation.md).
        # unscale_ first: under --precision fp16-hybrid, .grad is still GradScaler-scaled at this
        # point (unscaling normally only happens inside grad_scaler.step()) -- without this the
        # logged norm would repeat the exact scale-factor confusion already found and corrected for
        # the activation-gradient probes. A documented no-op under bf16 (scaler disabled). clip
        # with max_norm=inf so this only ever measures the norm, never actually clips gradients --
        # clipping isn't something this project has decided to do.
        grad_scaler.unscale_(optimizer)
        grad_norm_params_total = torch.nn.utils.clip_grad_norm_(params, float("inf")).item()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        lr_scheduler.step()

        # Last micro-step's real per-term gradient norms (see CodecLoss._hook_clone) -- same
        # "last micro-step only" convention already used for the preview video below.
        grad_norms = {f"grad_norm_{k}": v.item() for k, v in loss_fn.backward_metrics.items()}
        grad_norms["grad_norm_params_total"] = grad_norm_params_total
        grad_norms.update(per_term_grad_norms)  # empty dict unless --log-per-term-grad-norm

        current_lr = optimizer.param_groups[0]["lr"]
        term_str = ", ".join(
            f"{k}={v:.4f}" for k, v in accumulated.items() if k != "loss_total" and not k.endswith("_auto_w")
        )
        if (step + 1) % args.console_log_every == 0 or step == start_step:
            print(f"step {step}: lr={current_lr:.2e} loss_total={accumulated['loss_total']:.4f} ({term_str})")
            print(
                f"cuda_peak_allocated={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB "
                f"cuda_peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GiB"
            )
            # :.6e, not :.4f -- a genuinely small-but-nonzero grad norm (plausible this deep into
            # training, on an already-converged checkpoint) rounds to a misleading "0.0000" at 4
            # decimal places, same illusion this project already hit once with the PSD loss print.
            print(", ".join(f"{k}={v:.6e}" for k, v in grad_norms.items()))
        log_step(wandb_enabled, step, {**accumulated, **grad_norms}, current_lr)

        is_last = step == args.steps - 1
        if (step + 1) % args.checkpoint_every == 0 or is_last:
            # L2 distance from this run's own starting weights (initial_params above) -- an
            # unambiguous "did the model change at all" signal, independent of any
            # gradient-interpretation question. Only at checkpoint cadence, not every step: needs
            # a full param-vector CPU round-trip, not cheap enough for the hot loop.
            current_params = torch.cat([p.detach().flatten() for p in params]).cpu()
            weight_drift_l2 = (current_params - initial_params).norm().item()
            print(f"step {step}: weight_drift_l2_from_run_start={weight_drift_l2:.6e}")
            log_step(wandb_enabled, step, {"weight_drift_l2_from_run_start": weight_drift_l2}, current_lr)
            del current_params
            save_checkpoint(ckpt_path, step, bottleneck, decoder, optimizer, lr_scheduler, grad_scaler, wandb_run_id)
            # Decoupled from the local save above: local saves are cheap and want to be frequent
            # for crash safety, but the HF upload itself can be slow (network-bound), so it runs on
            # its own, sparser cadence -- always still fires on is_last so the FINAL state is never
            # skipped regardless of where hf_backup_every's modulo lands.
            if args.hf_backup_repo and ((step + 1) % hf_backup_every == 0 or is_last):
                from huggingface_hub import HfApi  # optional dep, only used here

                HfApi().upload_file(
                    path_or_fileobj=str(ckpt_path), path_in_repo="checkpoint.pth",
                    repo_id=args.hf_backup_repo, repo_type="model",
                )
        # Always persist the first completed step so a new run proves its output path before
        # committing hours of compute. Subsequent samples follow the configured interval.
        if step == start_step or (step + 1) % args.preview_every == 0 or is_last:
            log_preview(
                wandb_enabled, step, video, reconstructed, fps=args.target_fps,
                output_dir=checkpoint_dir / "previews",
            )


if __name__ == "__main__":
    main()
