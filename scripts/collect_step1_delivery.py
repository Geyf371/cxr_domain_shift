"""Validate and index A/B/C Step 1 artifacts from one or more output folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifact_contract import validate_step1_artifacts
from src.runtime import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Output directories to scan recursively")
    parser.add_argument("--output-csv", default="outputs/step1/step1_runs.csv")
    parser.add_argument("--output-json", default="outputs/step1/step1_delivery_report.json")
    parser.add_argument("--require-methods", nargs="*", default=[])
    args = parser.parse_args()

    metadata_paths = sorted({path.resolve() for value in args.inputs for path in Path(value).rglob("metadata_*.json")})
    if not metadata_paths:
        raise FileNotFoundError("No metadata_*.json files found")
    rows, reports = [], []
    for metadata_path in metadata_paths:
        tag = metadata_path.name.removeprefix("metadata_").removesuffix(".json")
        predictions = metadata_path.with_name(f"predictions_{tag}.csv")
        features = metadata_path.with_name(f"features_{tag}.npz")
        if not predictions.is_file() or not features.is_file():
            raise FileNotFoundError(f"Incomplete prediction/feature set for {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training = None
        if int(metadata.get("trainable_parameters", 0)) > 0:
            prefix = "" if metadata["scope"] == "experiment" else f"{metadata['scope']}_"
            training = metadata_path.parent / f"training_{prefix}{metadata['method']}_{metadata['budget']}_seed{metadata['seed']}.json"
            if not training.is_file():
                raise FileNotFoundError(f"Trainable run is missing {training.name}")
        report = validate_step1_artifacts(
            predictions=predictions,
            features=features,
            metadata=metadata_path,
            training=training,
        )
        training_resources = None
        if training:
            training_resources = json.loads(training.read_text(encoding="utf-8"))["resources"]
        else:
            training_resources = {
                "training_performed": False,
                "training_seconds": 0.0,
                "training_epochs": 0,
                "peak_gpu_memory_mb": None,
            }
        reports.append({"metadata": str(metadata_path), **report})
        row = {
            **report["identity"],
            **report["resources"],
            "num_samples": report["num_samples"],
            "predictions": str(predictions),
            "features": str(features),
            "metadata": str(metadata_path),
            "training": str(training) if training else None,
            "training_performed": training_resources["training_performed"],
            "training_seconds": training_resources["training_seconds"],
            "training_epochs": training_resources["training_epochs"],
            "training_peak_gpu_memory_mb": training_resources["peak_gpu_memory_mb"],
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    keys = ["method", "budget", "seed", "domain", "split", "scope"]
    if frame.duplicated(keys).any():
        duplicates = frame.loc[frame.duplicated(keys, keep=False), keys].to_dict("records")
        raise ValueError(f"Duplicate Step 1 run identities: {duplicates[:4]}")
    missing_methods = sorted(set(args.require_methods).difference(frame.method.astype(str)))
    if missing_methods:
        raise ValueError(f"Required methods are missing: {missing_methods}")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(keys).to_csv(args.output_csv, index=False)
    write_json(args.output_json, {
        "valid": True,
        "num_runs": len(frame),
        "methods": sorted(frame.method.astype(str).unique()),
        "runs_csv": str(Path(args.output_csv).resolve()),
        "reports": reports,
    })
    print(f"PASS: indexed {len(frame)} Step 1 runs -> {args.output_csv}")


if __name__ == "__main__":
    main()
