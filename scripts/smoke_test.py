"""Read exactly 16 images from one domain; no model inference is performed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CLASS_NAMES, assert_patient_disjoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Step 0 16-row data smoke test")
    parser.add_argument("--config", default="config/step0.yaml")
    parser.add_argument("--domain", choices=("mimic", "chexpert"), default="mimic")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    args = parser.parse_args()
    if args.num_samples != 16:
        raise ValueError("The Step 0 smoke test is defined for exactly 16 samples")
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_dir = Path(config["data"].get("data_dir", "data"))
    split = args.split or ("test" if args.domain == "chexpert" else "train")
    manifest_path = data_dir / f"{args.domain}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Build the manifests first: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    if not frame["domain"].eq(args.domain).all():
        raise ValueError(f"Expected only {args.domain} rows in {manifest_path}")
    if "image_exists" in frame.columns and not frame["image_exists"].all():
        raise FileNotFoundError(
            "This manifest was built with --allow-missing-images. Extract the selected JPGs "
            "and rebuild it before running the 16-image smoke test."
        )
    if args.domain == "chexpert" and not frame["split"].eq("test").all():
        raise ValueError("CheXpert is a test-only target domain")
    assert_patient_disjoint([group for _, group in frame.groupby("split")])
    sample = frame[frame["split"].eq(split)].head(args.num_samples).copy()
    if len(sample) != args.num_samples:
        raise ValueError(f"Only {len(sample)} rows are available for domain={args.domain}, split={split}")
    missing_images = [path for path in sample["image_path"] if not Path(path).exists()]
    if missing_images:
        raise FileNotFoundError(f"Missing image paths, first examples: {missing_images[:3]}")
    for path in sample["image_path"]:
        with Image.open(path) as image:
            image.convert("RGB").load()
    label_columns = [f"label_{name}" for name in CLASS_NAMES]
    mask_columns = [f"mask_{name}" for name in CLASS_NAMES]
    if sample[label_columns + mask_columns].isna().any().any():
        raise ValueError("Labels and masks contain NaN in the smoke-test rows")
    valid_values = set(sample[label_columns + mask_columns].to_numpy().ravel().tolist())
    if not valid_values.issubset({0, 1}):
        raise ValueError(f"Labels and masks must be binary, got {valid_values}")
    report = {
        "test_type": "data_only",
        "domain": args.domain,
        "num_samples": len(sample),
        "split": split,
        "unique_patients": int(sample["patient_id"].nunique()),
        "all_images_exist": True,
        "all_sample_images_decoded": True,
        "patients_disjoint_across_splits": True,
        "classes": CLASS_NAMES,
        "label_shape": [len(sample), len(label_columns)],
        "mask_shape": [len(sample), len(mask_columns)],
        "patient_ids_unique_in_batch": bool(sample["patient_id"].is_unique),
    }
    output_dir = Path(config["data"].get("output_dir", config.get("outputs", {}).get("output_dir", "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"smoke_test_{args.domain}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

