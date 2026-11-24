"""Recompute and summarize Step 2A from Step 1 predictions. No GPU is required."""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from cases import select_cases, write_case_review
from plots import write_plots


CLASSES = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")
MASKS = [f"mask_{name}" for name in CLASSES]
LABELS = [f"label_{name}" for name in CLASSES]
PROBABILITIES = [f"prob_{name}" for name in CLASSES]
DOMAINS = ("mimic", "chexpert")
BUDGETS = ("10pct", "100pct")
SEEDS = (42, 43, 44)
TRAINED = ("linear_probe", "lora_r8", "lora_r16")
BASELINE = "pretrained"
FIVE_PROMPTS = "pretrained_5prompts"
CLASS_METRICS = ("auc", "f1", "auprc")
MACRO_METRICS = tuple(f"macro_{name}" for name in CLASS_METRICS)
THRESHOLD = 0.5
IDENTITY = ["method", "budget", "seed", "domain"]


def boolean_masks(frame):
    # Step 1 exports booleans, whereas the manifests store the same masks as 0/1.
    values = frame[MASKS].astype(str).apply(lambda column: column.str.strip().str.lower())
    if not values.isin({"0", "1", "false", "true", "0.0", "1.0"}).all().all():
        raise ValueError("Unknown value in label mask")
    return values.isin({"1", "true", "1.0"})


def read_manifest(path, domain):
    manifest = pd.read_csv(
        path, dtype=str, keep_default_na=False
    ).query("split == 'test'").copy()
    if manifest.empty or set(manifest["domain"]) != {domain}:
        raise ValueError(f"{path} has no test records for {domain}")
    id_columns = ["domain", "patient_id", "study_id", "image_id"]
    manifest["sample_id"] = manifest[id_columns].agg("|".join, axis=1)
    manifest = manifest.set_index("sample_id").sort_index()
    if not manifest.index.is_unique:
        raise ValueError(f"Duplicate sample IDs in manifest: {path}")
    manifest[MASKS] = boolean_masks(manifest)
    manifest[LABELS] = manifest[LABELS].apply(pd.to_numeric, errors="raise")
    return manifest


def load_predictions(root, manifests):
    expected = set(product(TRAINED, BUDGETS, SEEDS, DOMAINS))
    expected.update((BASELINE, "100pct", 42, domain) for domain in DOMAINS)
    optional = {(FIVE_PROMPTS, "100pct", 42, domain) for domain in DOMAINS}
    predictions = {}
    files = {}
    for path in sorted(root.rglob("predictions_*.csv")):
        if path.name.startswith("predictions_step0_"):
            continue
        frame = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str, "image_id": str})
        if frame.empty:
            raise ValueError(f"Empty predictions file: {path}")
        # Read identity from the CSV itself. Filenames vary across Step 1 owners.
        identity = frame[IDENTITY + ["split", "scope"]].drop_duplicates()
        if len(identity) != 1:
            raise ValueError(f"Mixed run identities: {path}")
        method, budget, seed, domain, split, scope = identity.iloc[0]
        key = (method, budget, int(seed), domain)
        if split != "test" or scope != "experiment" or key not in expected | optional:
            continue
        if key in predictions:
            # Prefer the frozen-prompt export used in the repository's formal summary.
            new_is_official = "frozen_prompts" in path.relative_to(root).parts
            old_is_official = "frozen_prompts" in files[key].relative_to(root).parts
            if method != BASELINE or new_is_official == old_is_official:
                raise ValueError(f"Duplicate run {key}: {files[key]} and {path}")
            if old_is_official:
                continue

        frame = frame.set_index("sample_id").sort_index()
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate sample IDs in predictions: {path}")
        manifest = manifests[domain]
        valid_manifest = manifest.loc[manifest[MASKS].any(axis=1)]
        if not frame.index.equals(valid_manifest.index):
            raise ValueError(f"{path} does not cover the expected test samples")
        frame[MASKS] = boolean_masks(frame)
        mask = frame[MASKS].to_numpy(dtype=bool)
        if not np.array_equal(mask, valid_manifest[MASKS].to_numpy(dtype=bool)):
            raise ValueError(f"Mask differs from the manifest: {path}")
        if not np.array_equal(
            frame[LABELS].to_numpy()[mask], valid_manifest[LABELS].to_numpy()[mask]
        ):
            raise ValueError(f"Valid labels differ from the manifest: {path}")
        predictions[key] = frame
        files[key] = path

    missing = expected - predictions.keys()
    if missing:
        raise ValueError(f"Missing formal Step 1 test predictions: {sorted(missing)}")
    return predictions, files


def compute_results(predictions, files, manifests, compute_metrics):
    macro_rows = []
    class_rows = []
    audited_macros = 0
    for key, frame in sorted(predictions.items()):
        method, _, _, domain = key
        result = compute_metrics(
            frame[LABELS].to_numpy(),
            frame[PROBABILITIES].to_numpy(),
            frame[MASKS].to_numpy(dtype=bool),
            CLASSES,
            THRESHOLD,
        )
        path = files[key]
        step1_summary = path.with_name("metrics_summary_" + path.name.removeprefix("predictions_"))
        if step1_summary.is_file():
            original = pd.read_csv(step1_summary)
            if len(original) != 1 or float(original.iloc[0]["threshold"]) != THRESHOLD:
                raise ValueError(f"Unexpected threshold or summary shape: {step1_summary}")
            if not np.allclose(
                [result[metric] for metric in MACRO_METRICS],
                original.loc[0, list(MACRO_METRICS)].to_numpy(dtype=float),
                rtol=0, atol=1e-9, equal_nan=True,
            ):
                raise ValueError(f"Recomputed macro metrics differ from {step1_summary}")
            audited_macros += 1
        if method == FIVE_PROMPTS:
            continue  # Available as context in case pages. B owns its main evaluation.

        identity = dict(zip(IDENTITY, key))
        class_metrics = result.pop("per_class")
        total = len(manifests[domain])
        macro_rows.append({
            **identity, **result, "test_manifest_count": total,
            "skipped_all_labels_invalid": total - len(frame),
        })
        for row in class_metrics:
            valid = row["valid_count"]
            class_rows.append({
                **identity, **row, "threshold": THRESHOLD, "test_manifest_count": total,
                "unknown_count": total - valid,
                "positive_rate": row["positive_count"] / valid if valid else np.nan,
            })
    return pd.DataFrame(macro_rows), pd.DataFrame(class_rows), audited_macros


def summarize(frame, group_columns, metric_columns, fixed_columns=()):
    grouped = frame.groupby(group_columns, sort=False)
    summary = grouped[list(metric_columns)].agg(["mean", "std", "count"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    if fixed_columns:
        summary = summary.join(grouped[list(fixed_columns)].first())
    return summary.join(grouped.size().rename("n_seeds")).reset_index()


def transfer_results(macro, per_class):
    keys = ["method", "budget", "seed"]
    source = macro.loc[macro["domain"] == "mimic", keys + list(MACRO_METRICS)]
    target = macro.loc[macro["domain"] == "chexpert", keys + list(MACRO_METRICS)]
    transfer = source.merge(target, on=keys, validate="one_to_one", suffixes=("_source", "_target"))
    baseline = transfer.loc[transfer["method"] == BASELINE].iloc[0]
    probe = transfer.loc[transfer["method"] == "linear_probe"].set_index(["budget", "seed"])
    transfer = transfer.set_index(["budget", "seed"])
    for metric in MACRO_METRICS:
        # Compute paired differences before aggregating seeds. A negative drop indicates
        # that the target test score exceeds the source test score for that metric.
        source_metric, target_metric = f"{metric}_source", f"{metric}_target"
        transfer[f"{metric}_drop_source_minus_target"] = transfer[source_metric] - transfer[target_metric]
        for column in (source_metric, target_metric):
            transfer[f"{column}_gain_vs_pretrained"] = transfer[column] - baseline[column]
            transfer[f"{column}_gain_vs_linear_probe"] = (
                transfer[column] - probe[column].reindex(transfer.index).to_numpy()
            )
            transfer.loc[transfer["method"] == BASELINE, f"{column}_gain_vs_linear_probe"] = np.nan
    transfer = transfer.reset_index()[keys + [col for col in transfer if col != "method"]]

    # The main macro includes every class defined within each domain. Restrict both
    # sides to the same diseases in this companion calculation to check that choice.
    shared = per_class.pivot(index=[*keys, "class_name"], columns="domain", values="auc")
    shared = shared.dropna(subset=list(DOMAINS)).groupby(level=keys)[list(DOMAINS)].agg(["mean", "count"])
    shared.columns = [f"{domain}_{stat}" for domain, stat in shared.columns]
    shared = shared.rename(columns={
        "mimic_mean": "shared_class_auc_source",
        "chexpert_mean": "shared_class_auc_target",
        "mimic_count": "shared_auc_class_count",
    }).drop(columns="chexpert_count")
    shared["shared_class_auc_drop_source_minus_target"] = (
        shared["shared_class_auc_source"] - shared["shared_class_auc_target"]
    )
    return transfer.merge(shared.reset_index(), on=keys, how="left", validate="one_to_one")


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NA")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Path to cxr_domain_shift")
    parser.add_argument("--predictions", type=Path, required=True, help="Step 1 delivery CSV directory")
    folder = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=folder / "results")
    parser.add_argument("--private-output", type=Path, default=folder / "private")
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    inputs = args.predictions.resolve(strict=True)
    output, private = args.output.resolve(), args.private_output.resolve()
    if private == output or private.is_relative_to(output) or output.is_relative_to(private):
        parser.error("--output and --private-output must be separate directories")
    if not (repo / "src/metrics.py").is_file():
        parser.error("--repo must point to the directory containing src/metrics.py")
    sys.path.insert(0, str(repo))
    from src.metrics import compute_multilabel_metrics  # Reuse the group's metric definitions.
    import sklearn

    manifest_paths = {domain: repo / f"data/{domain}_manifest.csv" for domain in DOMAINS}
    manifests = {domain: read_manifest(path, domain) for domain, path in manifest_paths.items()}
    predictions, files = load_predictions(inputs, manifests)
    macro, per_class, audited = compute_results(
        predictions, files, manifests, compute_multilabel_metrics
    )

    fixed_counts = (
        "macro_auc_valid_classes", "macro_f1_valid_classes", "macro_auprc_valid_classes",
        "num_samples", "test_manifest_count", "skipped_all_labels_invalid", "threshold",
    )
    class_counts = (
        "valid_count", "positive_count", "negative_count", "unknown_count",
        "test_manifest_count", "positive_rate", "threshold",
    )
    macro_summary = summarize(macro, ["method", "budget", "domain"], MACRO_METRICS, fixed_counts)
    class_summary = summarize(
        per_class, ["method", "budget", "domain", "class_name"], CLASS_METRICS, class_counts,
    )
    prevalence = (
        class_summary[["domain", "class_name", *class_counts[:-1]]]
        .drop_duplicates().sort_values(["domain", "class_name"])
    )
    transfer = transfer_results(macro, per_class)
    transfer_summary = summarize(
        transfer, ["method", "budget"], [col for col in transfer if col not in IDENTITY]
    )
    cases, counts = select_cases(
        predictions, manifests["chexpert"], class_summary, CLASSES, BUDGETS, SEEDS, THRESHOLD
    )

    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    tables = {
        "macro_by_seed": macro, "macro_summary": macro_summary,
        "class_by_seed": per_class, "class_summary": class_summary,
        "class_prevalence": prevalence, "transfer_by_seed": transfer,
        "transfer_summary": transfer_summary, "case_candidate_counts": counts,
    }
    for name, frame in tables.items():
        save_csv(frame, output / f"{name}.csv")
    write_plots(macro_summary, transfer_summary, class_summary, prevalence, output / "figures")
    write_case_review(cases, predictions, SEEDS, THRESHOLD, private)
    metadata = {
        "protocol": {
            "test_domains": DOMAINS, "class_order": CLASSES, "budgets": BUDGETS, "seeds": SEEDS,
            "f1_threshold": THRESHOLD, "auprc_definition": "average_precision_score",
            "transfer_drop": "MIMIC test minus CheXpert test",
            "shared_class_auc": "mean over diseases with an AUC defined on both domains",
            "frozen_baseline": "one fixed single-template run per domain; no seed or budget sweep",
        },
        "run_counts": {
            "main_evaluation_runs": len(macro), "loaded_prediction_runs": len(predictions),
            "step1_macro_summaries_matched": audited, "case_pages": len(cases),
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
        "manifests": {
            path.name: sha256(path.read_bytes()).hexdigest() for path in manifest_paths.values()
        },
        "predictions": {
            files[key].relative_to(inputs).as_posix(): sha256(files[key].read_bytes()).hexdigest()
            for key in sorted(files)
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Evaluated {len(macro)} main runs; matched {audited} Step 1 summaries; selected {len(cases)} cases.")
    print(f"Aggregate results: {output}\nPrivate case review: {private}")


if __name__ == "__main__":
    main()
