"""Check CodecLoss's single-backward adaptive-weight path against the original probing path.

The fast path (_apply_adaptive_weights_reusing_gradients) takes each loss term's gradient at the
reconstruction once, derives the adaptive weights from those, and hands autograd the combined
gradient -- so no term is ever backpropagated twice. It is meant to be exactly equivalent to the
probing path (_apply_adaptive_weights_by_probing followed by a plain .backward()).

Loss values and the adaptive weights themselves come out identical to the last bit. The parameter
gradients agree to well within bf16's own precision, and the two places they do not agree bitwise
are both understood and both accounted for below:

  1. L1 + DINO-consistency, adaptive factor pinned to 1: BITWISE identical. This is the strict
     check -- it pins down that reusing the DINO term's gradient instead of recomputing it inside
     the real backward changes nothing at all.
  2. L1 + DINO-consistency at the real factor (~37x here): ~5e-3 relative. The probing path
     multiplies the term by its factor BEFORE backpropagating it, so the entire bf16 backward
     through V-JEPA runs at 37x scale; the fused path applies the same factor once, in fp32, at
     the end. Same value, different rounding -- and check 1 is what proves that is all it is,
     since pinning the factor makes it vanish completely. If anything the fused path is the
     better-conditioned of the two.
  3. Anything involving LPIPS: ~3.5e-4 relative, independent of the factor. FusedLpips' scorer is
     torch.compile'd, and inductor's backward for it is reached differently by autograd.grad than
     by .backward(). Two orders of magnitude below bf16's own ~4e-3 element precision.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from mini_mira.codec.bottleneck import MyBottleneck, StridedConvBottleneckConfig
from mini_mira.codec.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.codec.loss import CodecLoss, CodecLossWeights, CodecOutputs, normalize_video
from mini_mira.codec.vjepa import DEFAULT_VJEPA_LAYERS, VjepaModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(0)
    vjepa = VjepaModel(require_pretrained=False, last_layer_only=False, layer_indices=DEFAULT_VJEPA_LAYERS).cuda()
    bottleneck = MyBottleneck(StridedConvBottleneckConfig(temporal_stride=1)).cuda()
    decoder = ViTVideoDecoder(ViTDecoderConfig(width=256, depth=2, num_heads=8)).cuda()
    parameters = list(bottleneck.parameters()) + list(decoder.parameters())
    video = torch.rand(2, args.frames, 3, args.size, args.size, device="cuda")

    def run(fused: bool, max_auto_weight: float, **term_weights):
        loss_fn = CodecLoss(
            CodecLossWeights(auto_weight=True, max_auto_weight=max_auto_weight, **term_weights)
        ).cuda()
        loss_fn.bind_encoder_dino(vjepa)
        loss_fn.bind_last_layer(decoder.last_layer_weight)
        # Select the path directly rather than via --log-activation-grad-norms, which would also
        # insert _hook_clone's clones and stop this being a clean A/B of just the two paths.
        loss_fn._can_reuse_probe_gradients = lambda: fused
        # Same random frame subsets on both sides: the loss samples which frames to score, and a
        # different draw would swamp everything this is trying to measure.
        torch.manual_seed(123)
        for p in parameters:
            p.grad = None
        with torch.no_grad():
            features = vjepa.dino_forward(video)
        losses = loss_fn(CodecOutputs(normalize_video(video), decoder(bottleneck(features)), features))
        losses["loss_total"].backward()
        gradient = torch.cat([p.grad.flatten() for p in parameters])
        return {k: v.item() for k, v in losses.items()}, gradient

    def compare(label: str, max_auto_weight: float, **term_weights) -> float:
        probed_values, probed_gradient = run(False, max_auto_weight, **term_weights)
        fused_values, fused_gradient = run(True, max_auto_weight, **term_weights)
        for key, probed in probed_values.items():
            assert probed == fused_values[key], f"{key}: {probed} != {fused_values[key]}"
        difference = ((fused_gradient - probed_gradient).norm() / probed_gradient.norm()).item()
        weights = " ".join(
            f"{k.removesuffix('_auto_w')}={v:.4g}" for k, v in probed_values.items() if k.endswith("_auto_w")
        )
        print(f"{label:<28s} loss+weights identical ({weights}), gradient rel. difference {difference:.3e}")
        return difference

    print(f"clip: 2 x {args.frames} frames at {args.size}x{args.size}\n")
    exact = compare("L1+DINO, factor pinned", 1.0, loss_lpips_perceptual=0.0)
    assert exact == 0.0, f"pinned-factor gradients must be bitwise identical, got {exact}"

    rescaled = compare("L1+DINO, real factor", 1e4, loss_lpips_perceptual=0.0)
    assert rescaled < 1e-2, f"gradient difference {rescaled} exceeds what bf16 rescaling explains"

    everything = compare("all three terms", 1e4)
    assert everything < 1e-2, f"gradient difference {everything} exceeds what bf16 explains"
    print("\n[PASS] fused adaptive weights: identical losses and factors throughout, and bitwise "
          "identical gradients once the bf16 rescaling difference is removed.")


if __name__ == "__main__":
    main()
