"""LR schedule for the real training run: linear warmup from a small floor, a long constant
plateau, then linear decay to min_lr. mira's own WarmupConstantCosineDecayLR (mira/src/mira/
training/lr_schedule.py) doesn't support either knob -- warmup starts at exactly 0, and decay is
cosine-only -- so this is mini_mira-native code, same shape as that class, not a subclass of it.
"""

import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupConstantLinearDecayLR(LRScheduler):
    """Linear warmup from warmup_start_lr, then a constant plateau, then linear decay to min_lr.

    Set decay_steps=0 to disable the decay phase (lr stays at base_lr after warmup + constant).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        constant_steps: int,
        decay_steps: int,
        min_lr: float,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr

        assert warmup_steps >= 0
        assert constant_steps >= 0
        assert decay_steps >= 0
        assert min_lr >= 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore[override]
        step = self.last_epoch

        if step < self.warmup_steps:
            frac = step / max(1, self.warmup_steps)
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * frac for base_lr in self.base_lrs]
        elif step < self.warmup_steps + self.constant_steps:
            return list(self.base_lrs)
        elif step < self.warmup_steps + self.constant_steps + self.decay_steps:
            decay_step = step - self.warmup_steps - self.constant_steps
            frac = decay_step / self.decay_steps
            return [base_lr + (self.min_lr - base_lr) * frac for base_lr in self.base_lrs]
        elif self.decay_steps > 0:
            return [self.min_lr for _ in self.base_lrs]
        else:
            return list(self.base_lrs)
