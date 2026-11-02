"""Validation of Step 1 prediction, feature, metadata, and training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import CLASS_NAMES


IDENTITY_COLUMNS = ["method", "budget", "seed", "domain", "split", "scope"]
RESOURCE_COLUMNS = [
    "total_parameters",
    "trainable_parameters",
    "device",
    "inference_samples",
    "inference_batches",
    "inference_model_seconds",
    "inference_end_to_end_seconds",
    "peak_gpu_memory_mb",
    "peak_gpu_memory_supported",
]


def _single_value(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"Prediction artifact needs one unique {column}; found {values.tolist()}")
    value = values.iloc[0]
    return value.item() if isinstance(value, np.generic) else value


def validate_predictions(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    frame = pd.read_csv(path)
    required = set(IDENTITY_COLUMNS + ["patient_id", "image_id", "sample_id"])
    for prefix in ("label", "mask", "logit", "prob"):
        required.update(f"{prefix}_{name}" for name in CLASS_NAMES)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {missing}")
    if frame.empty or frame.sample_id.astype(str).duplicated().any():
        raise ValueError("Prediction artifact must contain nonempty, unique sample_id rows")
    logits = frame[[f"logit_{name}" for name in CLASS_NAMES]].to_numpy(float)
    probabilities = frame[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(float)
    labels = frame[[f"label_{name}" for name in CLASS_NAMES]].to_numpy(float)
    masks = frame[[f"mask_{name}" for name in CLASS_NAMES]].to_numpy(float)
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("Prediction logits/probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Prediction probabilities must lie in [0, 1]")
    if not np.isin(masks, [0, 1]).all() or not np.isin(labels[masks.astype(bool)], [0, 1]).all():
        raise ValueError("Masks and valid labels must be binary")
    identity = {column: _single_value(frame, column) for column in IDENTITY_COLUMNS}
    return frame, identity


def validate_features(path: str | Path, predictions: pd.DataFrame) -> dict[str, list[int]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as cache:
        required = {"v", "q", "z", "sample_id", "labels", "mask", "classes"}
        missing = sorted(required.difference(cache.files))
        if missing:
            raise ValueError(f"Feature artifact is missing arrays: {missing}")
        sample_ids = cache["sample_id"].astype(str)
        expected_ids = predictions.sample_id.astype(str).to_numpy()
        if not np.array_equal(sample_ids, expected_ids):
            raise ValueError("Feature sample_id order must exactly match the prediction CSV")
        if not np.array_equal(cache["classes"].astype(str), np.asarray(CLASS_NAMES, dtype=str)):
            raise ValueError("Feature artifact uses a different class order")
        n = len(predictions)
        v, q, z = cache["v"], cache["q"], cache["z"]
        if v.ndim != 2 or v.shape[0] != n or q.shape != (n, len(CLASS_NAMES)):
            raise ValueError("Expected v=[samples, dim] and q=[samples, 5]")
        if z.shape != (n, v.shape[1] + len(CLASS_NAMES)):
            raise ValueError("Expected z=[samples, v_dim + 5]")
        if not all(np.isfinite(value).all() for value in (v, q, z)):
            raise ValueError("Feature arrays must be finite")
        expected_labels = predictions[[f"label_{name}" for name in CLASS_NAMES]].to_numpy()
        expected_masks = predictions[[f"mask_{name}" for name in CLASS_NAMES]].to_numpy()
        if not np.array_equal(cache["labels"], expected_labels) or not np.array_equal(cache["mask"], expected_masks):
            raise ValueError("Feature labels/masks must exactly match the prediction CSV")
        return {name: list(cache[name].shape) for name in cache.files}


def validate_metadata(path: str | Path, identity: dict[str, Any], sample_count: int) -> dict[str, Any]:
    path = Path(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in IDENTITY_COLUMNS + RESOURCE_COLUMNS if name not in metadata]
    if missing:
        raise ValueError(f"Metadata artifact is missing fields: {missing}")
    for name, expected in identity.items():
        if str(metadata[name]) != str(expected):
            raise ValueError(f"Metadata {name} does not match predictions")
    if int(metadata["inference_samples"]) != sample_count:
        raise ValueError("Metadata inference_samples does not match predictions")
    if int(metadata["total_parameters"]) < int(metadata["trainable_parameters"]) or int(metadata["trainable_parameters"]) < 0:
        raise ValueError("Invalid total/trainable parameter counts")
    supported = bool(metadata["peak_gpu_memory_supported"])
    if supported != (metadata["peak_gpu_memory_mb"] is not None):
        raise ValueError("peak_gpu_memory_supported and peak_gpu_memory_mb disagree")
    return metadata


def validate_training(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    training = json.loads(path.read_text(encoding="utf-8"))
    required = {"history", "checkpoint", "best_val_macro_auc", "resources"}
    missing = sorted(required.difference(training))
    if missing:
        raise ValueError(f"Training artifact is missing fields: {missing}")
    resources = training["resources"]
    for name in ("total_parameters", "trainable_parameters", "device", "training_performed",
                 "training_epochs", "training_seconds", "peak_gpu_memory_mb", "peak_gpu_memory_supported"):
        if name not in resources:
            raise ValueError(f"Training resources are missing {name}")
    if not resources["training_performed"] or int(resources["trainable_parameters"]) <= 0:
        raise ValueError("Trainable Step 1 methods must report performed training and positive trainable parameters")
    if int(resources["training_epochs"]) != len(training["history"]):
        raise ValueError("training_epochs does not match history")
    checkpoint = Path(training["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = path.parent / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint}")
    return training


def validate_step1_artifacts(
    *,
    predictions: str | Path,
    features: str | Path,
    metadata: str | Path,
    training: str | Path | None = None,
) -> dict[str, Any]:
    frame, identity = validate_predictions(predictions)
    shapes = validate_features(features, frame)
    metadata_value = validate_metadata(metadata, identity, len(frame))
    training_value = validate_training(training) if training else None
    return {
        "valid": True,
        "identity": identity,
        "num_samples": len(frame),
        "feature_shapes": shapes,
        "resources": {name: metadata_value[name] for name in RESOURCE_COLUMNS},
        "training_checked": training_value is not None,
    }
