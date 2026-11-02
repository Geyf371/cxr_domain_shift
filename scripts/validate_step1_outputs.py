"""Validate one A/B/C Step 1 prediction-feature-metadata artifact set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifact_contract import validate_step1_artifacts
from src.runtime import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--training", help="Required for trainable A/B methods; omit for frozen C inference")
    parser.add_argument("--output", help="Optional JSON validation report")
    args = parser.parse_args()
    report = validate_step1_artifacts(
        predictions=args.predictions,
        features=args.features,
        metadata=args.metadata,
        training=args.training,
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
