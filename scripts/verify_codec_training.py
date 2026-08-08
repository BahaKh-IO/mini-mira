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
config -- no gated weights, no real data, runs in roughly a minute (was "seconds" back when this
trained on bare L1 alone; the fuller loss below needs more steps -- see the n_steps comment).
This is a correctness check on the training MECHANISM, not a real training run; see
train_codec.py for that.

Trains on the real three-term CodecLoss (L1 + LPIPS + DINO latent-consistency), with auto_weight
adaptive balancing on -- matches mira's own CodecLoss and its shipped config now (see
mini_mira.codec.loss for the auto_weight math).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from mini_mira.codec.bottleneck import StridedConvBottleneckConfig, MyBottleneck
from mini_mira.codec.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.codec.dino import DinoModel
from mini_mira.codec.loss import CodecLoss, CodecLossWeights, CodecOutputs, normalize_video

torch.manual_seed(0)

dino = DinoModel(require_pretrained=False)  # random-init backbone -- frozen either way, no
                                             # gated weights needed for this mechanical check
dino.eval()
bottleneck = MyBottleneck(StridedConvBottleneckConfig())
decoder = ViTVideoDecoder(ViTDecoderConfig())
loss_fn = CodecLoss(CodecLossWeights(auto_weight=True))
loss_fn.bind_encoder_dino(dino)  # share the same frozen backbone, not a second copy
loss_fn.bind_last_layer(decoder.last_layer_weight)

params = list(bottleneck.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

# One fixed video, reused every step -- 64x64 keeps this fast (dino patch_size=16 -> 4x4 dino
# grid -> bottleneck stride=2 -> 2x2 latent -> decoder upconv/patch-unembed back to 64x64).
video = torch.rand(1, 4, 3, 64, 64)

n_steps = 100  # 30 was enough for pure L1 (66.5% drop), but the combined L1+LPIPS+DINO-consistency
               # loss (no auto_weight -- see mini_mira.codec.loss) converges slower: the three
               # terms don't all point the same direction, so progress on any one is diluted.
               # Measured directly: 30 steps only reaches 33% (loss_mae itself barely moves, 19%,
               # even though loss_lpips_perceptual alone drops 57%); the loss keeps falling
               # steadily with no plateau -- 150 steps reaches 83.5% -- so this is "needs more
               # time," not "stuck." 100 steps clears the 50% bar with real margin.
losses = []  # each entry: the full {name: value} dict from CodecLoss, not just the total
for step in range(n_steps):
    optimizer.zero_grad()
    with torch.no_grad():  # encoder side: target features never need a gradient (see dino.py)
        dino_features = dino.dino_forward(video)
    z = bottleneck(dino_features)
    reconstructed = decoder(z)
    outputs = CodecOutputs(
        input_video=normalize_video(video), output_video=reconstructed, dino_features=dino_features
    )
    step_losses = loss_fn(outputs)
    step_losses["loss_total"].backward()
    optimizer.step()
    losses.append({k: v.item() for k, v in step_losses.items()})

# Per-term breakdown, not just the total -- with three unweighted terms added together (no
# auto_weight, see mini_mira.codec.loss), the total can be dragged down by whichever term is
# hardest to move even if the others are learning fine. Printed always, not just on failure, so
# this is diagnostic every run, not just when something goes wrong.
print("per-term loss, step 0 -> step", n_steps - 1, ":")
for name in losses[0]:
    print(f"  {name}: {losses[0][name]:.4f} -> {losses[-1][name]:.4f}")

total0, totalN = losses[0]["loss_total"], losses[-1]["loss_total"]
print(f"loss_total: step 0 = {total0:.4f}, step {n_steps - 1} = {totalN:.4f}")
assert totalN < total0 * 0.5, (
    f"loss did not drop enough to trust the training mechanism: {total0:.4f} -> {totalN:.4f}"
)
print(f"[PASS] codec training mechanism works: loss dropped by {(1 - totalN / total0) * 100:.1f}%")
