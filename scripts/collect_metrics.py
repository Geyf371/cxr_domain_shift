"""Collect H1/H2/H3 from existing outputs; no model or LoRA training."""
import argparse
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_analysis import compare_feature_caches
from src.hypothesis_metrics import compare_prompts, summarise_transfer
from src.runtime import write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="hypothesis", required=True)
    h1 = sub.add_parser("h1")
    h1.add_argument("--summaries", nargs="+", required=True, help="Summary CSV files or directories to scan")
    h1.add_argument("--baseline", default="pretrained")
    h1.add_argument("--output", default="outputs/h1_transfer.csv")
    h2 = sub.add_parser("h2")
    h2.add_argument("--base", required=True)
    h2.add_argument("--method", required=True)
    h2.add_argument("--expect-frozen", action="store_true")
    h2.add_argument("--output", default="outputs/h2_drift.json")
    h3 = sub.add_parser("h3")
    h3.add_argument("--single", required=True)
    h3.add_argument("--five", required=True)
    h3.add_argument("--output", default="outputs/h3_prompts.json")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.hypothesis == "h1":
        files = set()
        for value in args.summaries:
            path = Path(value)
            files.update(path.glob("metrics_summary_*.csv") if path.is_dir() else [path])
        if not files:
            raise ValueError("No summary CSV files found")
        summarise_transfer(pd.concat([pd.read_csv(p) for p in sorted(files)], ignore_index=True), args.baseline).to_csv(args.output, index=False)
    elif args.hypothesis == "h2":
        result = compare_feature_caches(args.base, args.method, args.expect_frozen)
        write_json(args.output, result)
        if args.expect_frozen and not result["within_tolerance"]:
            raise AssertionError(f"Frozen feature difference exceeds tolerance; see {args.output}")
    else:
        write_json(args.output, compare_prompts(pd.read_csv(args.single), pd.read_csv(args.five)))
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
