"""Mechanical proof that the V-JEPA-track codec (VjepaModel -> MyBottleneck -> ViTVideoDecoder)
can actually be trained: overfit fixed synthetic video and confirm the reconstruction loss drops
substantially. V-JEPA fork of verify_codec_training.py -- see that file for the full rationale
on why overfitting one example is the right mechanism check.

batch=2, 16 frames (not verify_codec_training.py's 1x4): large enough to exercise k>1 in
CodecLoss's random frame-subset sampling for the DINO-consistency term, which is exactly where
three real bugs were found and fixed -- VjepaModel.dino_forward couldn't accept fewer than
tubelet_size(=2) frames (a chunk can legitimately be handed just 1); CodecLoss's target-feature
lookup assumed pixel-frame index == feature-frame index, true for DinoModel (no temporal
reduction) but not V-JEPA (halves frame count via its tubelet); and CodecLoss's frame-selection
flattened batch+frame together before chunking, so a chunk could straddle two different videos in
the batch -- harmless for DinoModel (no cross-frame interaction at all) but silently wrong for a
tubelet-pairing encoder, which would then pair the last selected frame of one video with the
first of a completely different one. A 1x4 video never exercises any of these (k=1 always,
chunk_size=1 always, batch=1 so no second video to ever mix with) -- this size is chosen
specifically so a regression here doesn't silently pass again. The third bug also needed its own
dedicated check below (batch>=2 with an odd k), since the main overfit run's own even k=4 doesn't
trigger it either.

temporal_stride=1 on the bottleneck (not StridedConvBottleneckConfig's default of 2): V-JEPA
already halves frame count internally, matching configs/scaled_300m_vjepa.yaml's real setting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from mini_mira.codec.bottleneck import StridedConvBottleneckConfig, MyBottleneck
from mini_mira.codec.decoder import ViTDecoderConfig, ViTVideoDecoder
from mini_mira.codec.vjepa import DEFAULT_VJEPA_LAYERS, VjepaModel
from mini_mira.codec.loss import CodecLoss, CodecLossWeights, CodecOutputs, normalize_video

torch.manual_seed(0)

vjepa = VjepaModel(require_pretrained=False, last_layer_only=False, layer_indices=DEFAULT_VJEPA_LAYERS)
vjepa.eval()
bottleneck = MyBottleneck(StridedConvBottleneckConfig(temporal_stride=1))
decoder = ViTVideoDecoder(ViTDecoderConfig())
loss_fn = CodecLoss(CodecLossWeights(auto_weight=True))
loss_fn.bind_encoder_dino(vjepa)  # bind_encoder_dino takes any dino_forward/.dino_dim-shaped module
loss_fn.bind_last_layer(decoder.last_layer_weight)

params = list(bottleneck.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

video = torch.rand(2, 16, 3, 64, 64)

with torch.no_grad():
    probe_features = vjepa.dino_forward(video)
    probe_z = bottleneck(probe_features)
    probe_recon = decoder(probe_z)
    assert probe_recon.shape == video.shape, (
        f"shape round-trip broken: input {tuple(video.shape)} != reconstructed {tuple(probe_recon.shape)}"
    )
    print(f"[PASS] shape round-trips input video -> vjepa -> bottleneck -> decoder: {tuple(video.shape)}")

n_steps = 100
losses = []
for step in range(n_steps):
    optimizer.zero_grad()
    with torch.no_grad():
        vjepa_features = vjepa.dino_forward(video)
    z = bottleneck(vjepa_features)
    reconstructed = decoder(z)
    outputs = CodecOutputs(
        input_video=normalize_video(video), output_video=reconstructed, dino_features=vjepa_features
    )
    step_losses = loss_fn(outputs)
    step_losses["loss_total"].backward()
    optimizer.step()
    losses.append({k: v.item() for k, v in step_losses.items()})

print(f"\nper-term loss, step 0 -> step {n_steps - 1}:")
for name in losses[0]:
    print(f"  {name}: {losses[0][name]:.4f} -> {losses[-1][name]:.4f}")

total0, totalN = losses[0]["loss_total"], losses[-1]["loss_total"]
print(f"loss_total: step 0 = {total0:.4f}, step {n_steps - 1} = {totalN:.4f}")
assert totalN < total0 * 0.5, (
    f"loss did not drop enough to trust the training mechanism: {total0:.4f} -> {totalN:.4f}"
)
print(f"[PASS] V-JEPA codec training mechanism works: loss dropped by {(1 - totalN / total0) * 100:.1f}%")

# --- no cross-video frame mixing in the DINO-consistency term ---
# Deliberately: batch=2 (a second video to possibly mix with), 36 frames (k=9, ODD -- the exact
# repro condition), --perceptual-chunk-size=3 (doesn't divide 9 evenly either). predicted ==
# target == a hand-crafted video (bypasses the decoder entirely -- it's untrained at this point in
# the script and its output doesn't preserve input distinctness, which would make this check
# meaningless): item 0 constant 0.05, item 1 constant 0.95, so any dino_forward call mixing frames
# from both is immediately visible in that call's own input values.
mix_check_loss_fn = CodecLoss(CodecLossWeights(auto_weight=False), perceptual_chunk_size=3)
mix_check_loss_fn.bind_encoder_dino(vjepa)

mix_video = torch.empty(2, 36, 3, 32, 32)
mix_video[0], mix_video[1] = 0.05, 0.95
with torch.no_grad():
    mix_dino_features = vjepa.dino_forward(mix_video)

calls = []
original_forward = vjepa.dino_forward


def spying_forward(x):
    calls.append(x.detach().clone())
    return original_forward(x)


vjepa.dino_forward = spying_forward

mix_outputs = CodecOutputs(
    input_video=normalize_video(mix_video),
    output_video=normalize_video(mix_video).clone().requires_grad_(True),
    dino_features=mix_dino_features,
)
mix_check_loss_fn(mix_outputs)["loss_dino_latent_consistency"].backward()
vjepa.dino_forward = original_forward

assert len(calls) > 0, "no calls recorded -- this check isn't exercising the code path it means to"
mixed_calls = [
    i for i, c in enumerate(calls)
    if (c.mean(dim=(0, 2, 3, 4)) < 0.5).any() and (c.mean(dim=(0, 2, 3, 4)) >= 0.5).any()
]
assert not mixed_calls, f"call(s) {mixed_calls} mixed frames from two different videos"
print(f"[PASS] no cross-video frame mixing across {len(calls)} dino_forward calls (odd k=9, chunk_size=3)")
