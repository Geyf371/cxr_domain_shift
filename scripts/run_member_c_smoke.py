"""Run the frozen single-vs-five-prompt Step 1 acceptance on fixed patients."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifact_contract import validate_step1_artifacts
from src.data import CLASS_NAMES, assert_patient_disjoint, load_manifest, make_loader
from src.evaluator import evaluate_model
from src.feature_analysis import compare_feature_caches
from src.hypothesis_metrics import compare_prompts
from src.model_adapter import MedCLIPAdapter
from src.model_loader import load_medclip
from src.prompts import PROMPT_SET_VERSION
from src.runtime import output_dir, project_path, read_config, seed_everything, write_json


def select_patients(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select deterministic evaluable patients and retain all their images."""

    masks = [f"mask_{name}" for name in CLASS_NAMES]
    valid_rows = frame[masks].astype(bool).any(axis=1)
    eligible = frame.assign(_valid=valid_rows).groupby("patient_id")["_valid"].any()
    patient_ids = sorted(str(value) for value in eligible[eligible].index)[:count]
    if len(patient_ids) != count:
        raise ValueError(f"Requested {count} evaluable patients but only found {len(patient_ids)}")
    selected = frame[frame.patient_id.astype(str).isin(patient_ids)].copy()
    if selected.patient_id.astype(str).nunique() != count:
        raise AssertionError("Patient selection count changed unexpectedly")
    return selected.sort_values(["patient_id", "study_id", "image_id"])


def artifact_report(result: dict) -> dict:
    return validate_step1_artifacts(
        predictions=result["paths"]["predictions"],
        features=result["paths"]["features"],
        metadata=result["paths"]["metadata"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step0.local.yaml")
    parser.add_argument("--patients", type=int, default=128)
    parser.add_argument("--domains", nargs="+", choices=["mimic", "chexpert"], default=["mimic", "chexpert"])
    parser.add_argument("--output-dir", help="Defaults to <configured output>/step1/member_c_smoke")
    args = parser.parse_args()
    if args.patients < 1:
        raise ValueError("--patients must be positive")

    config = read_config(args.config)
    exp = config.setdefault("experiment", {})
    seed = int(exp.get("seed", 42))
    budget = str(exp.get("budget", "10pct"))
    batch_size = int(exp.get("batch_size", 4))
    seed_everything(seed)
    torch.set_num_threads(int(config.get("model", {}).get("cpu_threads", 4)))
    root_output = Path(args.output_dir) if args.output_dir else output_dir(config) / "step1" / "member_c_smoke"
    root_output.mkdir(parents=True, exist_ok=True)
    data_dir = project_path(config["data"].get("data_dir", "data"))
    bundle = load_medclip(config)
    models = {1: MedCLIPAdapter(bundle, templates=1), 5: MedCLIPAdapter(bundle, templates=5)}

    report: dict = {
        "status": "running",
        "patients_per_domain": args.patients,
        "domains": args.domains,
        "seed": seed,
        "budget_label": budget,
        "prompt_set_version": PROMPT_SET_VERSION,
        "training": {
            "training_performed": False,
            "training_seconds": 0.0,
            "trainable_parameters": 0,
            "reason": "Member C frozen prompt inference has no training stage",
        },
        "results": {},
    }
    write_json(root_output / f"member_c_smoke_{args.patients}.json", report)

    for domain in args.domains:
        frame = load_manifest(data_dir / f"{domain}_manifest.csv")
        assert_patient_disjoint([group for _, group in frame.groupby("split")])
        split = "test"
        selected = select_patients(frame[frame.split.eq(split)], args.patients)
        selection_text = "\n".join(selected.patient_id.astype(str)) + "\n"
        domain_results = {}
        for templates, method in ((1, "pretrained"), (5, "pretrained_5prompts")):
            loader = make_loader(
                selected,
                bundle.processor,
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
            )
            result = evaluate_model(
                models[templates],
                loader,
                domain=domain,
                split=split,
                method=method,
                budget=budget,
                seed=seed,
                output_dir=root_output,
                threshold=float(exp.get("threshold", 0.5)),
                scope=f"smoke{args.patients}",
                provenance=bundle.provenance,
            )
            domain_results[templates] = result

        single, five = domain_results[1], domain_results[5]
        prompt_comparison = compare_prompts(single["predictions"], five["predictions"])
        frozen_check = compare_feature_caches(
            single["paths"]["features"],
            five["paths"]["features"],
            expect_frozen=True,
        )
        if not frozen_check["within_tolerance"]:
            raise AssertionError(f"Frozen image features changed for {domain}")
        report["results"][domain] = {
            "selected_patients": int(selected.patient_id.astype(str).nunique()),
            "selected_images": len(selected),
            "patient_selection_sha256": hashlib.sha256(selection_text.encode()).hexdigest(),
            "single": artifact_report(single),
            "five": artifact_report(five),
            "prompt_comparison": prompt_comparison,
            "frozen_feature_check": frozen_check,
            "paths": {"single": single["paths"], "five": five["paths"]},
        }
        write_json(root_output / f"member_c_smoke_{args.patients}.json", report)

    report["status"] = "passed"
    write_json(root_output / f"member_c_smoke_{args.patients}.json", report)
    print(f"PASS: Member C {args.patients}-patient smoke test -> {root_output}")


if __name__ == "__main__":
    main()
