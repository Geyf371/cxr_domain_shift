"""Merge already-built domain manifests without rereading raw datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CLASS_NAMES, assert_patient_disjoint, write_class_frequencies


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge MIMIC and CheXpert manifests")
    parser.add_argument("--config", default="config/step0.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_dir = Path(config["data"].get("data_dir", "data"))
    paths = [data_dir / "mimic_manifest.csv", data_dir / "chexpert_manifest.csv"]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Build both domain manifests first: {missing}")
    frames = [pd.read_csv(path) for path in paths]
    required = {"domain", "patient_id", "study_id", "image_id", "image_path", "split"}
    required.update(f"{prefix}_{name}" for prefix in ("raw", "label", "mask") for name in CLASS_NAMES)
    for frame, domain in zip(frames, ("mimic", "chexpert")):
        missing_columns = required.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{domain} manifest is missing columns: {sorted(missing_columns)}")
        if frame.empty or not frame["domain"].eq(domain).all():
            raise ValueError(f"Expected a non-empty {domain}-only manifest")
        if not frame["patient_id"].is_unique:
            raise ValueError(f"{domain} manifest must contain one image per patient")
        if "image_exists" in frame and not frame["image_exists"].eq(True).all():
            raise ValueError(f"{domain} contains unavailable images; rebuild the final manifest first")
        assert_patient_disjoint([group for _, group in frame.groupby("split")])
        if domain == "chexpert" and not frame["split"].eq("test").all():
            raise ValueError("CheXpert is a test-only target domain")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(data_dir / "all_manifest.csv", index=False)
    write_class_frequencies(combined, data_dir / "class_frequencies.csv")
    print(f"Wrote {len(combined)} rows to {data_dir / 'all_manifest.csv'}")


if __name__ == "__main__":
    main()
