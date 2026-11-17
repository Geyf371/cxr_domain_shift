"""Run the frozen single-vs-five-prompt comparison on both full test sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifact_contract import validate_step1_artifacts
from src.data import assert_patient_disjoint, load_manifest, make_loader
from src.evaluator import evaluate_model
from src.feature_analysis import compare_feature_caches
from src.hypothesis_metrics import compare_prompts
from src.model_adapter import MedCLIPAdapter
from src.model_loader import load_medclip
from src.prompts import PROMPT_SET_VERSION
from src.runtime import output_dir, project_path, read_config, seed_everything, write_json


def artifact_report(result: dict) -> dict:
    return validate_step1_artifacts(
        predictions=result["paths"]["predictions"],
        features=result["paths"]["features"],
        metadata=result["paths"]["metadata"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step0.local.yaml")
    parser.add_argument("--domains", nargs="+", choices=["mimic", "chexpert"], default=["mimic", "chexpert"])
    parser.add_argument("--output-dir", help="Defaults to <configured output>/step1/member_c_formal")
    parser.add_argument("--budget", default="100pct", help="Frozen method grouping label; no training budget is consumed")
    args = parser.parse_args()

    config = read_config(args.config)
    exp = config.setdefault("experiment", {})
    seed = int(exp.get("seed", 42))
    batch_size = int(exp.get("batch_size", 4))
    threshold = float(exp.get("threshold", 0.5))
    seed_everything(seed)
    torch.set_num_threads(int(config.get("model", {}).get("cpu_threads", 4)))

    root_output = Path(args.output_dir) if args.output_dir else output_dir(config) / "step1" / "member_c_formal"
    root_output.mkdir(parents=True, exist_ok=True)
    data_dir = project_path(config["data"].get("data_dir", "data"))
    bundle = load_medclip(config)
    models = {1: MedCLIPAdapter(bundle, templates=1), 5: MedCLIPAdapter(bundle, templates=5)}

    report: dict = {
        "status": "running",
        "scope": "experiment",
        "domains": args.domains,
        "seed": seed,
        "budget_label": args.budget,
        "batch_size": batch_size,
        "prompt_set_version": PROMPT_SET_VERSION,
        "training": {
            "training_performed": False,
            "training_seconds": 0.0,
            "trainable_parameters": 0,
            "reason": "Member C frozen prompt inference has no training stage",
        },
        "results": {},
    }
    report_path = root_output / "member_c_formal.json"
    write_json(report_path, report)
    summary_rows: list[dict] = []

    for domain in args.domains:
        frame = load_manifest(data_dir / f"{domain}_manifest.csv")
        assert_patient_disjoint([group for _, group in frame.groupby("split")])
        selected = frame[frame.split.eq("test")].copy().sort_values(["patient_id", "study_id", "image_id"])
        if selected.empty:
            raise ValueError(f"No test rows found for {domain}")

        domain_results = {}
        for templates, method in ((1, "pretrained"), (5, "pretrained_5prompts")):
            loader = make_loader(selected, bundle.processor, batch_size=batch_size, shuffle=False, seed=seed)
            result = evaluate_model(
                models[templates],
                loader,
                domain=domain,
                split="test",
                method=method,
                budget=args.budget,
                seed=seed,
                output_dir=root_output,
                threshold=threshold,
                scope="experiment",
                provenance=bundle.provenance,
            )
            domain_results[templates] = result
            summary_path = Path(result["paths"]["metrics_summary"])
            summary_rows.append(pd.read_csv(summary_path).iloc[0].to_dict())

        single, five = domain_results[1], domain_results[5]
        frozen_check = compare_feature_caches(single["paths"]["features"], five["paths"]["features"], expect_frozen=True)
        if not frozen_check["within_tolerance"]:
            raise AssertionError(f"Frozen image features changed for {domain}")
        report["results"][domain] = {
            "input_images": int(len(selected)),
            "single": artifact_report(single),
            "five": artifact_report(five),
            "prompt_comparison": compare_prompts(single["predictions"], five["predictions"]),
            "frozen_feature_check": frozen_check,
            "paths": {"single": single["paths"], "five": five["paths"]},
        }
        write_json(report_path, report)

    pd.DataFrame(summary_rows).to_csv(root_output / "member_c_metrics.csv", index=False)
    report["status"] = "passed"
    write_json(report_path, report)
    print(f"PASS: Member C formal full-test comparison -> {root_output}")


if __name__ == "__main__":
    main()
