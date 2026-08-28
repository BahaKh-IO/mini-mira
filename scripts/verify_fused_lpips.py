"""Check that CodecLoss's FusedLpips returns what torchmetrics' own LPIPS module returns.

FusedLpips reuses torchmetrics' loaded VGG16 and its 1x1 head, and reorders nothing -- but it
skips the metric wrapper (state accumulation, input validation) and runs the per-scale comparison
through torch.compile, so "same value" is worth verifying rather than assuming.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from mini_mira.codec.loss import FusedLpips


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    reference = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device).eval()
    for p in reference.parameters():
        p.requires_grad = False
    # Same underlying module, so any difference is the wrapper/compile, not different weights.
    fused = FusedLpips(reference).to(device)

    for n, h, w in [(4, 64, 96), (20, 448, 768)]:
        pred = (torch.rand(n, 3, h, w, device=device) * 2 - 1).requires_grad_(True)
        target = torch.rand(n, 3, h, w, device=device) * 2 - 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            expected = reference(pred, target)
            reference.reset()
            actual = fused(pred, target)
        gap = (actual - expected).abs().item()
        tolerance = 2e-3 * max(1.0, abs(expected.item()))
        print(f"n={n} {h}x{w}: torchmetrics={expected.item():.6f} fused={actual.item():.6f} |diff|={gap:.2e}")
        assert gap <= tolerance, f"LPIPS mismatch: {gap} > {tolerance}"

        # The gradient is what actually trains the decoder, so check it too, not just the value.
        expected_grad = torch.autograd.grad(reference(pred, target), pred, retain_graph=False)[0]
        reference.reset()
        actual_grad = torch.autograd.grad(fused(pred, target), pred, retain_graph=False)[0]
        scale = expected_grad.abs().max().clamp(min=1e-12)
        relative = ((actual_grad - expected_grad).abs().max() / scale).item()
        print(f"           d/dpred max relative difference: {relative:.2e}")
        assert relative < 2e-2, f"LPIPS gradient mismatch: {relative}"

    print("[PASS] FusedLpips matches torchmetrics' LPIPS in value and gradient.")


if __name__ == "__main__":
    main()
