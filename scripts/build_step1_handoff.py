"""Build the tracked Step 1 A/B/C handoff tables from completed experiment records."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime import write_json


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("macro_auc", "macro_f1", "macro_auprc")


def parse_value(value: str) -> tuple[float, float | None]:
    parts = [part.strip() for part in value.split("±")]
    return float(parts[0]), float(parts[1]) if len(parts) == 2 else None


def markdown_rows(section: str) -> list[list[str]]:
    rows = []
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] not in {"方法", "预算"}:
            rows.append(cells)
    return rows


def member_a_metrics(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    output = []
    locations = (("mimic/val", "mimic", "val"), ("mimic/test", "mimic", "test"),
                 ("chexpert/test", "chexpert", "test"))
    for metric in METRICS:
        section = text.split(f"## {metric}", 1)[1].split("\n## ", 1)[0]
        for cells in markdown_rows(section):
            method, budget = cells[:2]
            if method == "pretrained":
                continue  # replaced below by the exact member-C rerun of the same baseline
            for value, (label, domain, split) in zip(cells[2:], locations):
                mean, std = parse_value(value)
                output.append({
                    "method": method, "budget": budget, "domain": domain, "split": split,
                    "metric": metric, "mean": mean, "std": std, "seeds": 3,
                    "source": "member_a/docs/step1_results_tables.md",
                    "notes": f"three-seed summary; source table column {label}; values rounded to 4 decimals",
                })
    return output


def member_b_metrics(path: Path) -> list[dict]:
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) < 4 or rows[2][:3] != ["method", "budget", "domain"]:
        raise ValueError("Unexpected member-B mean/std CSV layout")
    output = []
    for row in rows[3:]:
        for offset, metric in enumerate(METRICS):
            output.append({
                "method": row[0], "budget": row[1], "domain": row[2], "split": "test",
                "metric": metric, "mean": float(row[3 + 2 * offset]), "std": float(row[4 + 2 * offset]),
                "seeds": 3, "source": "member_b/test_metrics_mean_std.csv", "notes": "three-seed summary",
            })
    return output


def member_c_metrics(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    output = []
    for row in frame.to_dict("records"):
        for metric in METRICS:
            output.append({
                "method": row["method"], "budget": row["budget"], "domain": row["domain"],
                "split": row["split"], "metric": metric, "mean": row[metric], "std": None,
                "seeds": 1, "source": "member_c/member_c_metrics.csv",
                "notes": "frozen inference; seed is an execution label and no training was performed",
            })
    return output


def member_a_training(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    section = text.split("### 训练", 1)[1].split("\n### ", 1)[0]
    output = []
    for cells in markdown_rows(section):
        if cells[0] != "linear_probe":
            continue
        output.append({
            "method": cells[0], "budget": cells[1], "seed": int(cells[2]),
            "epochs": int(cells[3]), "best_epoch": None, "training_seconds": float(cells[4]),
            "training_validation_seconds": None, "peak_memory_mb": float(cells[5].replace(",", "")),
            "total_parameters": int(cells[6].replace(",", "")),
            "trainable_parameters": int(cells[7].replace(",", "")), "training_performed": True,
            "source": "member_a/docs/step1_results_tables.md",
            "timing_definition": "sum of training epochs; validation excluded",
        })
    return output


def member_b_training(path: Path) -> list[dict]:
    output = []
    for row in pd.read_csv(path).to_dict("records"):
        output.append({
            "method": f"lora_r{int(row['rank'])}", "budget": row["budget"], "seed": int(row["seed"]),
            "epochs": int(row["epochs"]), "best_epoch": int(row["best_epoch"]),
            "training_seconds": float(row["train_only_seconds"]),
            "training_validation_seconds": float(row["training_validation_seconds"]),
            "peak_memory_mb": float(row["peak_training_validation_allocated_bytes"]) / 1024 ** 2,
            "total_parameters": None, "trainable_parameters": int(row["trainable_parameters"]),
            "training_performed": True, "source": "member_b/run_resources.csv",
            "timing_definition": "resource replay train-only time; validation/initialization/checkpoint I/O excluded",
        })
    return output


def frozen_training_rows() -> list[dict]:
    return [{
        "method": method, "budget": "100pct", "seed": 42, "epochs": 0, "best_epoch": None,
        "training_seconds": 0.0, "training_validation_seconds": 0.0, "peak_memory_mb": None,
        "total_parameters": 136616059, "trainable_parameters": 0, "training_performed": False,
        "source": "member_c/member_c_formal_summary.json", "timing_definition": "no training stage",
    } for method in ("pretrained", "pretrained_5prompts")]


def member_a_inference(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    section = text.split("### 推理", 1)[1]
    output = []
    for cells in markdown_rows(section):
        if cells[0] != "linear_probe":
            continue
        domain, split = cells[3].split("/")
        output.append({
            "method": cells[0], "budget": cells[1], "seed": int(cells[2]), "domain": domain,
            "split": split, "input_images": int(cells[4]), "evaluable_images": None,
            "model_seconds": None, "end_to_end_seconds": float(cells[5]),
            "peak_memory_mb": float(cells[7].replace(",", "")), "batch_size": 64,
            "device": "CUDA", "repeats": 1, "source": "member_a/docs/step1_results_tables.md",
            "timing_definition": "eval + no_grad; reported 1000-image elapsed time",
        })
    return output


def member_b_inference(path: Path) -> list[dict]:
    output = []
    for row in pd.read_csv(path).to_dict("records"):
        output.append({
            "method": f"lora_r{int(row['rank'])}", "budget": row["budget"], "seed": int(row["seed"]),
            "domain": "chexpert", "split": "test", "input_images": int(row["benchmark_images"]),
            "evaluable_images": None, "model_seconds": float(row["inference_model_seconds_mean"]),
            "end_to_end_seconds": float(row["inference_end_to_end_seconds_mean"]),
            "peak_memory_mb": float(row["inference_peak_allocated_bytes_max"]) / 1024 ** 2,
            "batch_size": int(row["benchmark_batch_size"]), "device": "CUDA",
            "repeats": int(row["benchmark_repeats"]), "source": "member_b/run_resources.csv",
            "timing_definition": "1000 images; warmup 5 batches; mean of 3 repeats; FP32",
        })
    return output


def member_c_inference(path: Path) -> list[dict]:
    output = []
    for row in pd.read_csv(path).to_dict("records"):
        output.append({
            "method": row["method"], "budget": row["budget"], "seed": int(row["seed"]),
            "domain": row["domain"], "split": row["split"],
            "input_images": int(row["num_samples"] + row["skipped_no_valid_samples"]),
            "evaluable_images": int(row["num_samples"]), "model_seconds": float(row["inference_model_seconds"]),
            "end_to_end_seconds": float(row["inference_end_to_end_seconds"]), "peak_memory_mb": None,
            "batch_size": 4, "device": row["device"], "repeats": 1,
            "source": "member_c/member_c_metrics.csv",
            "timing_definition": "model forward and end-to-end times; CPU run; invalid-label rows skipped before forward",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-b-dir", required=True)
    parser.add_argument("--member-c-dir", required=True)
    parser.add_argument("--output-dir", default="docs/results")
    args = parser.parse_args()

    output = ROOT / args.output_dir
    b_source, c_source = Path(args.member_b_dir), Path(args.member_c_dir)
    b_output, c_output = output / "member_b", output / "member_c"
    b_output.mkdir(parents=True, exist_ok=True)
    c_output.mkdir(parents=True, exist_ok=True)
    for name in ("test_metrics.csv", "test_metrics_mean_std.csv", "run_resources.csv", "manifest.json"):
        source = b_source / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, b_output / name)
    shutil.copy2(c_source / "member_c_metrics.csv", c_output / "member_c_metrics.csv")

    c_report = json.loads((c_source / "member_c_formal.json").read_text(encoding="utf-8"))
    sanitized = {key: c_report[key] for key in ("status", "scope", "domains", "seed", "budget_label",
                                                 "batch_size", "prompt_set_version", "training")}
    sanitized["results"] = {}
    for domain, value in c_report["results"].items():
        sanitized["results"][domain] = {key: value[key] for key in (
            "input_images", "single", "five", "prompt_comparison", "frozen_feature_check")}
    write_json(c_output / "member_c_formal_summary.json", sanitized)

    b_test = pd.read_csv(b_output / "test_metrics.csv")
    expected_b = {(method, budget, seed, domain) for method in ("lora_r8", "lora_r16")
                  for budget in ("10pct", "100pct") for seed in (42, 43, 44)
                  for domain in ("mimic", "chexpert")}
    actual_b = set(zip(b_test.method, b_test.budget, b_test.seed, b_test.domain))
    if len(b_test) != 24 or actual_b != expected_b:
        raise ValueError("Member-B test metric coverage is incomplete")

    a_results = ROOT / "docs/step1_results_tables.md"
    metrics = member_a_metrics(a_results) + member_b_metrics(b_output / "test_metrics_mean_std.csv") \
        + member_c_metrics(c_output / "member_c_metrics.csv")
    metrics_frame = pd.DataFrame(metrics).sort_values(["method", "budget", "domain", "split", "metric"])
    if set(metrics_frame.method) != {"pretrained", "pretrained_5prompts", "linear_probe", "lora_r8", "lora_r16"}:
        raise ValueError("Unified metrics table does not contain all Step 1 methods")
    metrics_frame.to_csv(output / "step1_metrics_summary.csv", index=False)

    training = member_a_training(a_results) + member_b_training(b_output / "run_resources.csv") + frozen_training_rows()
    pd.DataFrame(training).sort_values(["method", "budget", "seed"]).to_csv(
        output / "step1_training_resources.csv", index=False)
    inference = member_a_inference(a_results) + member_b_inference(b_output / "run_resources.csv") \
        + member_c_inference(c_output / "member_c_metrics.csv")
    pd.DataFrame(inference).sort_values(["method", "budget", "seed", "domain"]).to_csv(
        output / "step1_inference_resources.csv", index=False)

    hashes = set(b_test.groupby("domain")["evaluation_set_sha256"].nunique())
    if hashes != {1}:
        raise ValueError("Member-B seeds did not evaluate the same samples")
    write_json(output / "step1_delivery_manifest.json", {
        "status": "aggregate_results_passed_raw_handoff_pending",
        "methods": sorted(metrics_frame.method.unique()),
        "member_a": {"branch": "feat/step1-clean", "metrics_source": "docs/step1_results_tables.md"},
        "member_b": {"branch": "step1-member-b", "test_metric_rows": len(b_test), "training_runs": 12},
        "member_c": {"branch": "step1-member-c", "formal_runs": 4, "prompt_set_version": "clinical_v1"},
        "coverage": {
            "member_b_test_matrix_complete": True,
            "member_b_same_evaluation_set_across_seeds": True,
            "member_c_artifact_validation": c_report["status"],
            "all_five_methods_present": True,
            "member_a_raw_predictions_features_checkpoints_received": False,
            "member_b_raw_predictions_features_checkpoints_received": False,
            "member_c_raw_predictions_features_available_locally": True,
        },
        "handoff_readiness": {
            "integrated_code": True,
            "aggregate_metrics_and_resources": True,
            "direct_step2_raw_artifact_consumption": False,
        },
        "tracked_tables": [
            "docs/results/step1_metrics_summary.csv",
            "docs/results/step1_training_resources.csv",
            "docs/results/step1_inference_resources.csv",
        ],
        "large_artifacts": (
            "Member C full-test predictions and v/q/z are available in the local output directory. "
            "Member A/B raw predictions, v/q/z and checkpoints were not present in the received branches/files; "
            "collect their original experiment packages before declaring the Step 2 raw-artifact handoff complete."
        ),
    })
    print(f"PASS: built Step 1 handoff tables -> {output}")


if __name__ == "__main__":
    main()
