"""Resize a batch of decoded video to one canonical shape for training, matching mira's
VideoCodec.preprocess_batch: pad to the target aspect ratio first (black padding, right or
bottom), then bilinear-resize -- so clips of any native resolution/aspect land at the model's
expected shape without distorting their content.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def resize_to_canonical(video: Tensor, height: int, width: int) -> Tensor:
    """video: (b, t, c, h, w), any dtype. Returns the same dtype/range, resized to (height, width).
    Pads first so the aspect ratio is preserved instead of squashing the content.
    """
    is_float = video.is_floating_point()
    x = video.float() if not is_float else video
    b, t, _, image_h, image_w = x.shape

    # Black-pad the narrower side to match the target aspect ratio.
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

    # Antialiasing can ring slightly past the valid range; round/clamp before restoring dtype.
    return x if is_float else x.round().clamp_(0, 255).to(video.dtype)
