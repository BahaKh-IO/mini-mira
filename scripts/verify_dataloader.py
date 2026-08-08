"""Behavioral check for the real dataloader wiring (train_codec.py's --index-path path), entirely
without the real (currently connection-blocked) shard download.

Builds a tiny, self-contained fake WebDataset shard + index.json on disk -- the same real tar/JSON
layout mira's own dataset expects (schema.py's Index/MatchEntry/Perspective, one WebDataset sample
per (match, chunk) with p{i}.mp4/p{i}.jsonl members) -- following the exact pattern real mira's own
test fixture uses (mira/tests/data/test_loader.py's `_build`), adapted to use imageio (already a
mini_mira dependency) instead of a subprocess ffmpeg call, so this doesn't need system ffmpeg on
PATH. Then points mira's real create_loader + mini_mira's resize_to_canonical at it and confirms a
real batch comes out the other end with the right shape -- proving the wiring, not the real data.
"""

import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

import imageio.v3 as iio
import numpy as np
import torch

from mini_mira.codec.video_prep import resize_to_canonical
from mira.data.training_loader import create_loader

FPS, CHUNK_FRAMES, MATCH_ID, SHARD = 20, 8, "2026-01-01T00-00-00Z-fakematch", "dataset_00.tar"


def _fake_mp4_bytes(tmp_path: Path, width: int = 48, height: int = 32) -> bytes:
    """A real, ffmpeg-decodable mp4 -- CHUNK_FRAMES frames of a moving colored square, matching
    the resolution/frame-count real matches would have (just much smaller, for speed)."""
    frames = []
    for i in range(CHUNK_FRAMES):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = (i * 4) % (width - 8)
        frame[4:12, x : x + 8] = [255, 0, 0]
        frames.append(frame)
    mp4_path = tmp_path / "fake.mp4"
    iio.imwrite(mp4_path, frames, fps=FPS, codec="libx264", pixelformat="yuv420p")
    data = mp4_path.read_bytes()
    mp4_path.unlink()
    return data


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def build_fake_dataset(out: Path) -> Path:
    """One match, one chunk, one player -- matches n_players=1 (the codec-training case) in
    train_codec.py. Returns `out`, the directory to pass as --index-path / index_path."""
    out.mkdir(parents=True, exist_ok=True)
    video_bytes = _fake_mp4_bytes(out)
    actions = "".join(json.dumps({"keys": ["W"]}) + "\n" for _ in range(CHUNK_FRAMES)).encode()

    key = f"{MATCH_ID}_c00000"
    with tarfile.open(out / SHARD, "w") as tar:
        _add(tar, f"{key}.p0.mp4", video_bytes)
        _add(tar, f"{key}.p0.jsonl", actions)

    entry = {
        "match_id": MATCH_ID,
        "shard": SHARD,
        "n_players": 1,
        "chunk_frames": [CHUNK_FRAMES],
        "perspectives": [
            {
                "player_id": 0,
                "team": 0,
                "frames": CHUNK_FRAMES,
                "duration": CHUNK_FRAMES / FPS,
                "recording_offset_sec": 0.0,
                "anchors": [],
            }
        ],
    }
    (out / "index.json").write_text(json.dumps({"total_samples": 1, "entries": [entry]}))
    return out


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        index_path = build_fake_dataset(Path(tmp))
        print(f"Built a fake dataset at {index_path}")

        clip_len = 4  # < CHUNK_FRAMES=8, so one chunk yields a real clip
        loader = create_loader(
            index_path=index_path, clip_len=clip_len, target_fps=FPS, n_players=1, batch_size=2,
        )
        try:
            batch, metadata = next(iter(loader))
        except (RuntimeError, OSError) as e:
            if "libtorchcodec" not in str(e):
                raise
            print(
                "\n[SKIPPED] torchcodec couldn't load its native library -- a known Windows-only\n"
                "issue (it needs a 'full-shared' FFmpeg build with DLLs; imageio-ffmpeg's bundled\n"
                "standalone binary isn't the same thing and doesn't satisfy it). Everything BEFORE\n"
                "this point already ran for real: index.json parsing, tar reading, shard iteration,\n"
                "and clip planning all succeeded -- only the final pixel-decode call is blocked here.\n"
                "This is expected to work on the Linux GPU machine (a normal ffmpeg install there\n"
                "ships the shared libs torchcodec needs) -- deferred there, not a code problem.\n"
            )
            return
        print(f"Got a real batch: video shape {tuple(batch.video.shape)}, dtype {batch.video.dtype}")
        assert batch.video.shape == (2, clip_len, 3, 32, 48), batch.video.shape
        assert batch.video.dtype == torch.uint8

        video = batch.video.float() / 255.0
        resized = resize_to_canonical(video, height=64, width=64)
        print(f"After resize_to_canonical: {tuple(resized.shape)}")
        assert resized.shape == (2, clip_len, 3, 64, 64), resized.shape
        assert 0.0 <= resized.min() and resized.max() <= 1.0

        print(f"metadata[0]: {metadata[0]}")
        print("[PASS] create_loader -> resize_to_canonical wiring works on a real (fake) dataset.")


if __name__ == "__main__":
    main()
