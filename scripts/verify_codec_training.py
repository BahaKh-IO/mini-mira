"""Mechanical proof that the codec (DinoModel -> MyBottleneck -> ViTVideoDecoder) can actually
be trained: overfit ONE fixed synthetic video and confirm the reconstruction loss drops
substantially.

Deliberately overfits a single fixed example, not fresh random data every step: this is the
standard way to sanity-check a training loop is wired correctly. A loop with a real bug (wrong
loss, detached graph, frozen params that shouldn't be, optimizer never stepping) would plateau
at a noise floor either way -- reusing one example makes "did this actually learn anything"
unambiguous, since a working loop should be able to drive a single example's loss most of the
way to zero.

Uses require_pretrained=False (random-init DINOv3) and the small default bottleneck/decoder
config -- no gated weights, no real data, runs in seconds. This is a correctness check on the
training MECHANISM, not a real training run; see train_codec.py for that.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F
from mini_mira.codec.bottleneck import StridedConvBottleneckConfig, MyBottleneck
from mini_mira.codec.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.codec.dino import DinoModel

torch.manual_seed(0)

dino = DinoModel(require_pretrained=False)  # random-init backbone -- frozen either way, no
                                             # gated weights needed for this mechanical check
dino.eval()
bottleneck = MyBottleneck(StridedConvBottleneckConfig())
decoder = ViTVideoDecoder(ViTDecoderConfig())

params = list(bottleneck.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

# One fixed video, reused every step -- 64x64 keeps this fast (dino patch_size=16 -> 4x4 dino
# grid -> bottleneck stride=2 -> 2x2 latent -> decoder upconv/patch-unembed back to 64x64).
video = torch.rand(1, 4, 3, 64, 64)

n_steps = 30
losses = []
for step in range(n_steps):
    optimizer.zero_grad()
    dino_features = dino.dino_forward(video)  # frozen; dino_forward already wraps itself in no_grad
    z = bottleneck(dino_features)
    reconstructed = decoder(z)
    loss = F.l1_loss(reconstructed, video)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"loss: step 0 = {losses[0]:.4f}, step {n_steps - 1} = {losses[-1]:.4f}")
assert losses[-1] < losses[0] * 0.5, (
    f"loss did not drop enough to trust the training mechanism: {losses[0]:.4f} -> {losses[-1]:.4f}"
)
print(f"[PASS] codec training mechanism works: loss dropped by {(1 - losses[-1] / losses[0]) * 100:.1f}%")
