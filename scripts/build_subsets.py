"""Build nested source-domain patient subsets for seeds 42, 43, and 44."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested source patient subsets")
    parser.add_argument("--config", default="config/step0.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_cfg = config["data"]
    data_dir = Path(data_cfg.get("data_dir", "data"))
    manifest_path = data_dir / "mimic_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Build the source manifest first: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    patients = np.array(sorted(manifest.loc[manifest["split"].eq("train"), "patient_id"].unique()))
    if len(patients) == 0:
        raise ValueError("The MIMIC manifest has no train patients")
    for seed in config["split"].get("subset_seeds", [42, 43, 44]):
        rng = np.random.default_rng(int(seed))
        ordered = patients.copy()
        rng.shuffle(ordered)
        full = np.sort(ordered)
        for budget in config["split"].get("budgets", [0.1, 1.0]):
            budget = float(budget)
            count = len(ordered) if budget >= 1 else max(1, int(np.ceil(len(ordered) * budget)))
            subset = pd.DataFrame({"patient_id": np.sort(ordered[:count]), "budget": budget, "seed": int(seed)})
            label = "100pct" if budget >= 1 else f"{int(round(budget * 100))}pct"
            subset.to_csv(data_dir / f"source_train_seed{int(seed)}_{label}.csv", index=False)
        pd.DataFrame({"patient_id": full, "budget": 1.0, "seed": int(seed)}).to_csv(
            data_dir / f"source_train_seed{int(seed)}_100pct.csv", index=False
        )
    print(f"Wrote nested subsets for seeds {config['split'].get('subset_seeds', [42, 43, 44])} to {data_dir}")


if __name__ == "__main__":
    main()

