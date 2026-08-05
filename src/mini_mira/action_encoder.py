import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    """Encodes raw per-frame key-press actions into per-latent-frame conditioning vectors.

    Keys only, no mouse: the released Rocket League data never carries real mouse signal
    anyway (mouse_movements is always zero, game_mouse_sensitivity is always NaN), so there
    is nothing to reconstruct there. Simplified relative to mira's ActionEncoder: no dropout
    (training-only), mean-pooling instead of a learned temporal pool, and a plain Linear
    projection instead of mira's power-of-2 per-key split (see notes/deviations.md 1.10).
    """

    def __init__(self, num_keys: int, hidden_dim: int, temporal_downsampling: int, key_dim: int = 16):
        super().__init__()
        self.temporal_downsampling = temporal_downsampling
        self.key_embeddings = nn.ModuleList([nn.Embedding(2, key_dim) for _ in range(num_keys)])
        self.proj = nn.Linear(num_keys * key_dim, hidden_dim)
        # Learned "action before frame 0" -- same role as bos, for actions instead of frames.
        self.initial_action_token = nn.Parameter(0.02 * torch.randn(1, 1, hidden_dim))

    def forward(self, key_presses: torch.Tensor) -> torch.Tensor:
        # key_presses: (b, raw_t, num_keys), int/long, multi-hot (0 or 1).
        b, raw_t, num_keys = key_presses.shape
        embedded = [self.key_embeddings[k](key_presses[:, :, k]) for k in range(num_keys)]
        x = torch.cat(embedded, dim=-1)  # (b, raw_t, num_keys * key_dim)
        x = self.proj(x)  # (b, raw_t, hidden_dim)

        # Pool from raw (video) frame rate down to latent frame rate.
        x = x.unflatten(dim=1, sizes=(-1, self.temporal_downsampling))  # (b, raw_t/td, td, hidden_dim)
        x = x.mean(dim=2)  # (b, raw_t/td, hidden_dim)

        initial_token = self.initial_action_token.expand(b, -1, -1)
        x = torch.cat([initial_token, x], dim=1)  # (b, raw_t/td + 1, hidden_dim)
        return x
