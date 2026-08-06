import torch
from torch import Tensor


def _rope_cos_sin(positions: Tensor, dim: int, theta: float) -> tuple[Tensor, Tensor]:
    assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
    k = torch.arange(dim // 2, dtype=torch.float32, device=positions.device)
    inv_freq = theta ** (-2.0 * k / dim)  # (dim // 2,)
    freqs = positions.float()[:, None] * inv_freq[None, :]  # (N, dim // 2)
    cos = freqs.cos().repeat_interleave(2, dim=-1)  # (N, dim)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def temporal_rope(num_frames: int, head_dim: int, theta: float, device: torch.device):
    """Causal 1D RoPE over the time axis. Shape (T, head_dim)."""
    positions = torch.arange(num_frames, device=device)
    return _rope_cos_sin(positions, head_dim, theta)


def spatial_rope(height: int, width: int, head_dim: int, theta: float, device: torch.device):
    """Axial 2D RoPE over a height x width grid. Shape (height * width, head_dim).

    Half of head_dim encodes the row position, the other half the column position.
    """
    assert head_dim % 4 == 0, f"2D RoPE needs head_dim divisible by 4, got {head_dim}"
    half = head_dim // 2
    rows = torch.arange(height, device=device).repeat_interleave(width)  # (h*w,)
    cols = torch.arange(width, device=device).repeat(height)  # (h*w,)
    cos_h, sin_h = _rope_cos_sin(rows, half, theta)
    cos_w, sin_w = _rope_cos_sin(cols, half, theta)
    cos = torch.cat([cos_h, cos_w], dim=-1)  # (h*w, head_dim)
    sin = torch.cat([sin_h, sin_w], dim=-1)
    return cos, sin
