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

from huggingface_hub import snapshot_download  # noqa: E402
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

    # from_hub restricts dataset.index to only the shards THIS call requested, but only in memory
    # -- the index.json FILE it leaves on disk is still whatever was fetched, not necessarily
    # everything actually downloaded here. Left unfixed, a later create_loader() reading this
    # directory has no way to know which shards are actually present, and can reach for one
    # that isn't -> crash. The earlier fix for that (blindly overwriting index.json with just
    # this call's restricted set) had a real bug: a smaller --shards N on a later run would
    # silently shrink an already-larger local dataset instead of growing it -- confirmed
    # directly, not hypothetical (see notes/deviations.md). --shards N means "make sure at
    # least N are downloaded", not "make it exactly N".
    #
    # The .tar shard files themselves are never deleted by any of this -- only ever added --
    # so ground truth is: every match whose shard file actually exists in this directory,
    # regardless of which run downloaded it. force_download=True re-fetches index.json fresh
    # from the hub rather than trusting whatever's cached locally, since an earlier run's manual
    # overwrite here can leave that local cache in a state snapshot_download doesn't expect.
    index_dir = snapshot_download(
        args.repo_id, repo_type="dataset", allow_patterns=[f"{args.split}/index.json"], force_download=True,
    )
    full_index = RocketScienceDataset(Path(index_dir) / args.split / "index.json")
    present_shards = {e.shard for e in full_index.index.entries if (dataset.root / e.shard).exists()}
    full_index.index.entries = [e for e in full_index.index.entries if e.shard in present_shards]
    full_index.matches = {mid: e for mid, e in full_index.matches.items() if e.shard in present_shards}

    # dataset.index_path is a symlink into HuggingFace's content-addressed blob cache -- writing
    # through it, as this script always used to, overwrites the blob's bytes in place, corrupting
    # the cache's own bookkeeping (confirmed directly, not theoretical: this is *why* repeated runs
    # here gave inconsistent results at different times, not because the remote data was changing).
    # Unlink the symlink first so this write lands as a plain, ordinary file that a future run
    # can't corrupt the same way -- same fix already applied by hand once to recover this
    # particular directory; this makes the script itself safe to rely on going forward.
    if dataset.index_path.is_symlink():
        dataset.index_path.unlink()
    dataset.index_path.write_text(full_index.index.model_dump_json())
    print(f"Done. {len(full_index.matches)} match(es) available across {len(present_shards)} shard(s).")
    print(f"Local index path (pass this to train_codec.py's --index-path): {dataset.root}")


if __name__ == "__main__":
    main()
