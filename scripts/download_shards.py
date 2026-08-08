"""Download a small number of real Rocket League match shards, for a correctness smoke test
of our precompute pipeline -- NOT the real training data pull.

Reuses real mira's own RocketScienceDataset.from_hub(shards=N) directly (see notes/deviations.md
for why: data loading is orthogonal to the architecture we're actually here to learn, and mira's
own class already does exactly this "download only the first N shards" thing) instead of
reimplementing WebDataset tar parsing from scratch.

Requirements to run this yourself:
  - A real HuggingFace account token with access granted to the gated kyutai/rocket-science
    dataset, set as HF_TOKEN in the terminal you run this from (must be your own terminal --
    a token set in one process isn't visible to a different process).
  - `pip install huggingface_hub` (already a dependency of real mira's dataset.py).

Downloads land in the default HuggingFace cache (~/.cache/huggingface/hub) on THIS machine --
correct for this small check. The real training pull (many more shards) should be run directly
on the GPU machine instead, not routed through here first (see README's Scope / notes).
"""

import argparse
import sys
from pathlib import Path

# mini_mira's own source (unused directly here, kept for consistency with the other scripts).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Real mira, cloned as a sibling repo next to mini_mira -- reused directly for this one step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))

from mira.data.dataset import RocketScienceDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="kyutai/rocket-science")
    parser.add_argument("--split", default="train")
    parser.add_argument("--shards", type=int, default=3, help="Number of tar shards to fetch (~2.65GB each)")
    args = parser.parse_args()

    print(f"Requesting {args.shards} shard(s) from {args.repo_id}/{args.split} "
          f"(~{args.shards * 2.65:.1f}GB total) ...")
    dataset = RocketScienceDataset.from_hub(args.repo_id, split=args.split, shards=args.shards)
    print(f"Done. {len(dataset.match_ids())} match(es) available across {args.shards} shard(s).")
    print(f"Local index path (pass this to train_codec.py's --index-path): {dataset.root}")


if __name__ == "__main__":
    main()
