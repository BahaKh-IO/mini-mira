"""Resize a batch of decoded video to one canonical shape for training, matching real mira's
VideoCodec.preprocess_batch exactly (mira/src/mira/codec/codec_model.py) -- pad to the target
aspect ratio first (black padding, right or bottom), then bilinear-resize, so real clips of
whatever native resolution/aspect they were recorded at all land at the same shape the model
expects, without distorting their content.

Kept as a free function, not a class method: mini_mira has no combined "VideoCodec" class the way
mira does (bottleneck/decoder/dino are wired together directly in the training scripts), so there
is no natural home to attach this to as a method.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def resize_to_canonical(video: Tensor, height: int, width: int) -> Tensor:
    """video: (b, t, c, h, w), any dtype. Returns the same dtype/range, resized to (height, width).

    Padding first (rather than a plain resize) preserves the original aspect ratio -- a plain
    resize to a different aspect ratio would squash/stretch the content, which a straight video
    codec has no way to "undo" and would just be training on distorted frames.
    """
    is_float = video.is_floating_point()
    x = video.float() if not is_float else video
    b, t, _, image_h, image_w = x.shape

    # Pad with black on the right or bottom to match the target aspect ratio, matching mira's own
    # preprocess_batch exactly -- whichever side is "too narrow" for the target ratio gets padded.
    if image_w * height < width * image_h:
        right_pad = round(width / height * image_h) - image_w
        x = F.pad(x, (0, right_pad))
    elif image_w * height > width * image_h:
        bottom_pad = round(height / width * image_w) - image_h
        x = F.pad(x, (0, 0, 0, bottom_pad))

    if x.shape[-2:] != (height, width):
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = F.interpolate(x, size=(height, width), mode="bilinear", antialias=True)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)

    # Antialiased bilinear interpolation runs in float and can ring slightly past the valid range;
    # round and clamp back before restoring the original dtype (matching decode.py's own _resize).
    return x if is_float else x.round().clamp_(0, 255).to(video.dtype)
