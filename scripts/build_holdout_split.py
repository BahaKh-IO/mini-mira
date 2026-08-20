"""Isolates shards added by a LATER download_shards.py call into a separate directory + filtered
index.json -- a genuine held-out eval split, unlike evaluate_codec.py's --seed alone (which only
reshuffles the SAME shard pool the codec already trained on).

Usage: capture a shard list before downloading more (`ls .../train/*.tar | sort > before.txt`),
download more shards, capture the list again (`> after.txt`), then run this.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mira" / "src"))
from mira.data.dataset import RocketScienceDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", required=True, help="The .../train dir download_shards.py wrote to")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output-dir", default="eval_holdout")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    before = {Path(line).name for line in Path(args.before).read_text().splitlines() if line.strip()}
    after = {Path(line).name for line in Path(args.after).read_text().splitlines() if line.strip()}
    new_shards = after - before
    if not new_shards:
        raise SystemExit("No new shards found -- did the download add anything beyond --before?")
    print(f"{len(new_shards)} new shard(s): {sorted(new_shards)}")

    dataset = RocketScienceDataset(train_dir / "index.json")
    dataset.index.entries = [e for e in dataset.index.entries if e.shard in new_shards]
    dataset.matches = {mid: e for mid, e in dataset.matches.items() if e.shard in new_shards}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard in new_shards:
        shutil.copy(train_dir / shard, out_dir / shard)
    (out_dir / "index.json").write_text(dataset.index.model_dump_json())
    print(f"{len(dataset.matches)} held-out match(es) across {len(new_shards)} shard(s).")
    print(f"Pass this to evaluate_codec.py's --index-path: {out_dir}")


if __name__ == "__main__":
    main()
