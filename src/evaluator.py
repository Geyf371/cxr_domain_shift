"""Method-agnostic batch evaluation interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import hashlib
import time

import numpy as np
import pandas as pd
import torch

from .data import CLASS_NAMES
from .metrics import compute_multilabel_metrics
from .model_interface import validate_model_batch_output
from .resource_metrics import (
    inference_statistics,
    model_device,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    timed_device_block,
)
from .runtime import write_json


def select_batch(batch, keep):
    indices = np.flatnonzero(_to_numpy(keep)).tolist()
    return {key: value[indices] if hasattr(value, "shape") else [value[i] for i in indices]
            for key, value in batch.items()}


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def evaluate_model(
    model: Any,
    dataloader: Iterable,
    *,
    domain: str,
    split: str,
    method: str,
    budget: str,
    seed: int,
    output_dir: str | Path,
    threshold: float = 0.5,
    scope: str = "experiment",
    provenance: dict | None = None,
) -> dict:
    """Run one domain/split and save predictions, metrics, and features.

    The model adapter must expose ``forward_batch(batch)`` and return an object
    with ``logits`` plus optional ``image_features_v``, ``prompt_scores_q``,
    and ``joint_features_z``. This evaluator is independent of LoRA.
    """

    model.eval()
    records: list[dict[str, Any]] = []
    feature_arrays: dict[str, list[np.ndarray]] = {"v": [], "q": [], "z": []}
    skipped = 0
    batches = 0
    model_seconds = 0.0
    device = model_device(model)
    reset_peak_gpu_memory(device)
    end_to_end_started = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            keep = _to_numpy(batch["mask"]).astype(bool).any(axis=1)
            skipped += int((~keep).sum())
            if not keep.any():
                continue
            batch = select_batch(batch, keep)
            with timed_device_block(device) as timing:
                output = model.forward_batch(batch)
            output = validate_model_batch_output(output, len(batch["patient_id"]))
            model_seconds += timing["seconds"]
            batches += 1
            logits = _to_numpy(output.logits)
            labels = _to_numpy(batch["labels"])
            mask = _to_numpy(batch["mask"])
            if logits.shape != labels.shape or logits.shape[1] != len(CLASS_NAMES) or not np.isfinite(logits).all():
                raise ValueError("Model must return finite [batch, 5] logits")
            probabilities = torch.sigmoid(torch.as_tensor(logits)).numpy()
            patient_ids = batch["patient_id"]
            for row_index, patient_id in enumerate(patient_ids):
                record: dict[str, Any] = {
                    "method": method,
                    "budget": budget,
                    "seed": seed,
                    "domain": domain,
                    "split": split,
                    "patient_id": patient_id,
                    "image_id": batch["image_id"][row_index],
                    "sample_id": batch["sample_id"][row_index],
                    "scope": scope,
                }
                for class_index, class_name in enumerate(CLASS_NAMES):
                    record[f"label_{class_name}"] = labels[row_index, class_index]
                    record[f"mask_{class_name}"] = mask[row_index, class_index]
                    record[f"logit_{class_name}"] = logits[row_index, class_index]
                    record[f"prob_{class_name}"] = probabilities[row_index, class_index]
                records.append(record)
            for key, attribute in (("v", "image_features_v"), ("q", "prompt_scores_q"), ("z", "joint_features_z")):
                value = getattr(output, attribute, None)
                if value is not None:
                    feature_arrays[key].append(_to_numpy(value))

    if not records:
        raise ValueError(f"No evaluable samples; skipped {skipped} samples without valid labels")
    end_to_end_seconds = float(time.perf_counter() - end_to_end_started)
    resources = inference_statistics(
        model,
        samples=len(records),
        batches=batches,
        model_seconds=model_seconds,
        end_to_end_seconds=end_to_end_seconds,
        peak_memory_mb=peak_gpu_memory_mb(device),
    )
    if resources["trainable_parameters"] == 0:
        resources.update({
            "training_performed": False,
            "training_seconds": 0.0,
            "training_epochs": 0,
            "training_peak_gpu_memory_mb": None,
        })
    prediction_frame = pd.DataFrame(records)
    if prediction_frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample identities in evaluation")
    labels = prediction_frame[[f"label_{name}" for name in CLASS_NAMES]].to_numpy()
    probabilities = prediction_frame[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy()
    mask = prediction_frame[[f"mask_{name}" for name in CLASS_NAMES]].to_numpy()
    metrics = compute_multilabel_metrics(labels, probabilities, mask, CLASS_NAMES, threshold)
    metrics["skipped_no_valid_samples"] = skipped
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scope}_{method}_{domain}_{split}_{budget}_seed{seed}"
    prediction_path = output_dir / f"predictions_{tag}.csv"
    per_class_path = output_dir / f"metrics_per_class_{tag}.csv"
    summary_path = output_dir / f"metrics_summary_{tag}.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    pd.DataFrame(metrics["per_class"]).assign(method=method, budget=budget, seed=seed, domain=domain, split=split).to_csv(per_class_path, index=False)
    summary = {key: value for key, value in metrics.items() if key != "per_class"}
    summary.update({"method": method, "budget": budget, "seed": seed, "domain": domain, "split": split, "scope": scope})
    summary.update(resources)
    identity_columns = ["sample_id"] + [f"{p}_{c}" for p in ("label", "mask") for c in CLASS_NAMES]
    summary["evaluation_set_sha256"] = hashlib.sha256(prediction_frame[identity_columns].sort_values("sample_id").to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    feature_output = {}
    for key, values in feature_arrays.items():
        if values:
            feature_output[key] = np.concatenate(values, axis=0)
    feature_path = output_dir / f"features_{tag}.npz"
    if feature_output:
        for name in ("sample_id", "patient_id", "image_id"):
            feature_output[name] = prediction_frame[name].astype(str).to_numpy(dtype=str)
        feature_output["labels"] = labels
        feature_output["mask"] = mask
        feature_output["classes"] = np.asarray(CLASS_NAMES, dtype=str)
        text = getattr(model, "text_features", None)
        if text is not None:
            feature_output["text_features"] = _to_numpy(text)
        np.savez_compressed(feature_path, **feature_output)
    metadata = dict(provenance or {})
    metadata.update(summary)
    metadata["feature_shapes"] = {key: list(value.shape) for key, value in feature_output.items()}
    metadata["feature_layer"] = "vision_model.projection_head -> L2 normalization"
    metadata["prompt_protocol"] = "fixed length 77; mean template logits then sigmoid"
    metadata["prompt_set_version"] = getattr(model, "prompt_set_version", None)
    metadata["prompts"] = getattr(model, "prompt_texts", None)
    metadata_path = output_dir / f"metadata_{tag}.json"
    write_json(metadata_path, metadata)
    return {
        "predictions": prediction_frame,
        "metrics": metrics,
        "feature_arrays": feature_output,
        "paths": {
            "predictions": str(prediction_path),
            "metrics_per_class": str(per_class_path),
            "metrics_summary": str(summary_path),
            "features": str(feature_path) if feature_output else None,
            "metadata": str(metadata_path),
        },
        "resources": resources,
    }

