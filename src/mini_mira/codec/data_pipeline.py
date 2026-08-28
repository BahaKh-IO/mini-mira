"""Host->device video streaming for the training loops.

The straightforward version of this -- `next(loader)` then `.float()/255` then resize then
`.cuda()`, all inline in the training loop -- costs the main process real time on every
micro-step, and none of it overlaps with the GPU:

  - the uint8->float32 conversion and the pad+antialiased-bilinear resize run on ONE CPU core
    at native decode resolution (a 40-frame 720p clip is ~350 MB once it is float32),
  - the resulting float32 tensor is then copied to the GPU, 4x more bytes than the uint8 it
    came from, out of unpinned memory, so the copy is synchronous and the CPU waits for it.

This module moves both halves where they belong. The conversion and resize happen on the GPU
(same math, same `resize_to_canonical`, just a different device); the host->device copy moves
the uint8 tensor instead of the float32 one, out of pinned memory, on its own CUDA stream, from
a background thread that runs a batch or two ahead of the training loop. The training loop then
just takes a tensor that is already on the GPU.

The stream/event handshake is what makes the overlap safe: the copy is recorded on `copy_stream`
and the consuming (default) stream waits on that event before touching the tensor, and the
tensor is kept alive by `record_stream` until the copy stream is actually done with it.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Iterator

import torch
from torch import Tensor

from mini_mira.codec.video_prep import resize_to_canonical


def to_canonical_on_gpu(video_uint8: Tensor, height: int, width: int) -> Tensor:
    """uint8 (b, t, c, h, w) on GPU -> float32 [0, 1] at (height, width).

    Same two steps, in the same order, as the inline CPU version this replaces: scale into
    [0, 1] first, then pad-to-aspect + resize in float space (`resize_to_canonical`), so the
    antialiased resize sees the same values it always did.
    """
    return resize_to_canonical(video_uint8.float().div_(255.0), height, width)


class PrefetchingVideoStream:
    """Iterator of GPU-resident, canonical-shape clip batches, filled a few batches ahead.

    `fetch` is called from a background thread and must return a CPU uint8 (b, t, c, h, w)
    tensor -- i.e. the dataloader's own output, with no preprocessing applied.
    """

    def __init__(self, fetch: Callable[[], Tensor], height: int, width: int, depth: int = 3):
        self.fetch = fetch
        self.height = height
        self.width = width
        self.copy_stream = torch.cuda.Stream()
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                video = self.fetch()
                # Pinned staging + a dedicated stream is what lets this copy actually overlap
                # with training compute instead of serializing against it.
                video = video.pin_memory()
                with torch.cuda.stream(self.copy_stream):
                    on_gpu = video.to("cuda", non_blocking=True)
                    ready = torch.cuda.Event()
                    ready.record(self.copy_stream)
                self._queue.put((on_gpu, ready))
        except BaseException as exc:  # surfaced on the consumer side, see __next__
            self._error = exc
            self._queue.put(None)

    def __iter__(self) -> Iterator[Tensor]:
        return self

    def __next__(self) -> Tensor:
        item = self._queue.get()
        if item is None:
            raise self._error if self._error is not None else StopIteration
        on_gpu, ready = item
        torch.cuda.current_stream().wait_event(ready)
        # The producer thread's reference dies at the end of this call; record_stream tells the
        # caching allocator not to hand those bytes to anyone else until the copy stream retires.
        on_gpu.record_stream(torch.cuda.current_stream())
        return to_canonical_on_gpu(on_gpu, self.height, self.width)
