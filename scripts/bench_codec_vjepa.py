"""Component-level cost breakdown for the codec training step, so optimization targets come from
measurement rather than guesswork.

Times each stage of one micro-step in isolation (V-JEPA encode, bottleneck, decoder, each loss
term, the auto_weight probes, the real backward) with proper CUDA synchronization, at whatever
--height/--width/--frames/--batch-size a real run uses. Run it before and after a change.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from mini_mira.codec.bottleneck import MyBottleneck
from mini_mira.codec.decoder import ViTVideoDecoder
from mini_mira.codec.loss import CodecLoss, CodecLossWeights, CodecOutputs, normalize_video
from mini_mira.codec.vjepa import DEFAULT_VJEPA_LAYERS, VjepaModel
from mini_mira.ml.config_loading import load_pipeline_config


def timeit(name: str, fn, iters: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1e3
    print(f"{name:<44s} {ms:9.1f} ms   peak={torch.cuda.max_memory_allocated() / 2**30:6.2f} GiB")
    torch.cuda.reset_peak_memory_stats()
    return ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scaled_300m_vjepa.yaml")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch._functorch.config.donated_buffer = False

    vjepa = VjepaModel(require_pretrained=True, last_layer_only=False, layer_indices=DEFAULT_VJEPA_LAYERS).cuda().eval()
    bottleneck = MyBottleneck(config.bottleneck).cuda()
    decoder = ViTVideoDecoder(config.decoder).cuda()
    loss_fn = CodecLoss(CodecLossWeights(auto_weight=True)).cuda()
    loss_fn.bind_encoder_dino(vjepa)
    loss_fn.bind_last_layer(decoder.last_layer_weight)
    loss_fn.use_channels_last_perceptual()
    if args.compile:
        # Same granularity as scripts/train_codec_vjepa.py -- blocks, not whole modules -- or this
        # measures a configuration nothing actually runs. See that script's --compile handling.
        bottleneck.compile()
        for block in decoder.blocks:
            block.compile()
        vjepa.encoder.compile()

    video = torch.rand(args.batch_size, args.frames, 3, args.height, args.width, device="cuda")
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731

    with autocast(), torch.no_grad():
        features = vjepa.dino_forward(video)
    with autocast():
        z = bottleneck(features)
        reconstructed = decoder(z)

    print(f"\nshapes: video={tuple(video.shape)} dino={tuple(features[0].shape)} "
          f"latent={tuple(z.shape)} recon={tuple(reconstructed.shape)}\n")
    torch.cuda.reset_peak_memory_stats()

    def encode():
        with autocast(), torch.no_grad():
            vjepa.dino_forward(video)

    features = [f.detach() for f in features]

    def codec_forward():
        with autocast():
            decoder(bottleneck(features))

    def term(name: str, **weights):
        # A fresh CodecLoss per term: whether LPIPS runs at all is decided at CONSTRUCTION time
        # (weights.loss_lpips_perceptual > 0), so lowering the weight afterwards would leave it
        # running and quietly fold its cost into every other term's number.
        off = {"loss_mae": 0.0, "loss_lpips_perceptual": 0.0, "loss_dino_latent_consistency": 0.0}
        term_loss = CodecLoss(CodecLossWeights(auto_weight=False, **{**off, **weights})).cuda()
        term_loss.bind_encoder_dino(vjepa)
        term_loss.use_channels_last_perceptual()
        detached = reconstructed.detach().requires_grad_(True)

        def run():
            with autocast():
                term_loss(CodecOutputs(normalize_video(video), detached, features))

        return timeit(name, run, args.iters)

    def full_step(auto_weight: bool):
        loss_fn.weights = CodecLossWeights(auto_weight=auto_weight)

        def run():
            with autocast():
                with torch.no_grad():
                    f = vjepa.dino_forward(video)
                recon = decoder(bottleneck(f))
                losses = loss_fn(CodecOutputs(normalize_video(video), recon, f))
            losses["loss_total"].backward()
            bottleneck.zero_grad(set_to_none=True)
            decoder.zero_grad(set_to_none=True)

        return run

    timeit("vjepa encode (no_grad, full clip)", encode, args.iters)
    timeit("bottleneck + decoder forward", codec_forward, args.iters)
    term("loss fwd: L1 only", loss_mae=1.0)
    term("loss fwd: LPIPS only", loss_lpips_perceptual=1.0)
    term("loss fwd: DINO-consistency only", loss_dino_latent_consistency=1.0)

    # Free the retained graphs from the isolated phases above before the full-step timings --
    # those hold on to enough activations to OOM an 80 GiB card on their own.
    del z, reconstructed, features
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    on = timeit("FULL micro-step (auto_weight ON)", full_step(True), args.iters)
    off = timeit("FULL micro-step (auto_weight OFF)", full_step(False), args.iters)
    print(f"{'  -> adaptive weighting alone':<44s} {on - off:9.1f} ms")


if __name__ == "__main__":
    main()
