"""Step timing and an opt-in torch.profiler window for the codec training loops.

Two levels, both cheap enough to leave permanently wired into the loop:

  - StepTimer: wall-clock per optimizer step (and a breakdown of the named phases inside it),
    printed as a rolling median so one slow warmup step doesn't hide the steady-state number.
    Always on -- a training run that can't say how long a step takes can't be optimized.
  - profile_window: a torch.profiler schedule around a few steps in the middle of the run,
    dumping a Chrome trace plus the top CUDA kernels. Opt-in (--profile-steps), because the
    profiler itself distorts step time while it's active.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from pathlib import Path

import torch


class StepTimer:
    """Per-step wall clock plus per-phase breakdown, reported as a rolling median.

    Phases are recorded with `with timer.phase("data"): ...`. They are wall-clock, not CUDA
    events: the loop is async, so a phase's number only means "how long the CPU spent here",
    which is exactly the question that matters for finding host-side stalls (data loading,
    Python overhead, syncs). End-to-end step time is the ground truth for everything else.
    """

    def __init__(self, window: int = 20, warmup: int = 3):
        self.window = window
        self.warmup = warmup
        self.step_times: list[float] = []
        self.phase_times: dict[str, list[float]] = {}
        self._step_start = time.perf_counter()
        self._count = 0

    @contextlib.contextmanager
    def phase(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.phase_times.setdefault(name, []).append(time.perf_counter() - start)

    def end_step(self) -> float:
        now = time.perf_counter()
        elapsed = now - self._step_start
        self._step_start = now
        self._count += 1
        if self._count > self.warmup:
            self.step_times.append(elapsed)
            self.step_times[:] = self.step_times[-self.window :]
            for values in self.phase_times.values():
                values[:] = values[-self.window * 64 :]
        else:
            self.phase_times.clear()
        return elapsed

    @property
    def median_step(self) -> float:
        return statistics.median(self.step_times) if self.step_times else float("nan")

    def report(self, clips_per_step: int, frames_per_clip: int) -> str:
        step = self.median_step
        if step != step:  # NaN -- still inside warmup
            return "step_time=warmup"
        phases = " ".join(
            f"{name}={sum(values) / len(self.step_times) * 1e3:.0f}ms"
            for name, values in self.phase_times.items()
            if values
        )
        return (
            f"step_time={step * 1e3:.0f}ms ({1.0 / step:.3f} steps/s, "
            f"{clips_per_step / step:.1f} clips/s, {clips_per_step * frames_per_clip / step:.0f} frames/s) "
            f"[{phases}]"
        )


@contextlib.contextmanager
def profile_window(output_dir: Path, active_steps: int, wait_steps: int):
    """torch.profiler over `active_steps` steps after `wait_steps` warmup steps, or a no-op
    context yielding None when active_steps <= 0. Yields a step-callback to call once per step.
    """
    if active_steps <= 0:
        yield lambda: None
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = torch.profiler.schedule(wait=wait_steps, warmup=1, active=active_steps, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        yield prof.step
    trace_path = output_dir / "trace.json.gz"
    prof.export_chrome_trace(str(trace_path))
    print(f"\n--- profiler: top CUDA kernels (trace: {trace_path}) ---")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25, max_name_column_width=70))
    print("--- profiler: top ops by CUDA time (grouped by input shape) ---")
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=20, max_name_column_width=55
        )
    )
