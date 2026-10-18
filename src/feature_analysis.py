"""Small feature-analysis primitives used by later Step 3 analyses."""

from __future__ import annotations

import numpy as np
from pathlib import Path


def cosine_drift(base_v: np.ndarray, method_v: np.ndarray) -> np.ndarray:
    base_v = np.asarray(base_v, dtype=float)
    method_v = np.asarray(method_v, dtype=float)
    if base_v.shape != method_v.shape:
        raise ValueError("base_v and method_v must have the same shape")
    if base_v.ndim != 2 or not len(base_v) or not np.isfinite(base_v).all() or not np.isfinite(method_v).all():
        raise ValueError("Features must be nonempty finite [samples, dim] arrays")
    numerator = (base_v * method_v).sum(axis=1)
    denominator = np.linalg.norm(base_v, axis=1) * np.linalg.norm(method_v, axis=1)
    if (denominator <= 0).any():
        raise ValueError("Cannot compare zero-norm feature vectors")
    return np.clip(1.0 - numerator / denominator, 0.0, 2.0)


def summarise_drift(base_v: np.ndarray, method_v: np.ndarray) -> dict[str, float]:
    values = cosine_drift(base_v, method_v)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "max_absolute_difference": float(np.max(np.abs(np.asarray(base_v) - np.asarray(method_v)))),
    }


def compare_feature_caches(base_path, method_path, expect_frozen=False, tolerance=1e-5):
    with np.load(base_path, allow_pickle=False) as base, np.load(method_path, allow_pickle=False) as method:
        a_ids, b_ids = base["sample_id"].astype(str), method["sample_id"].astype(str)
        if len(set(a_ids)) != len(a_ids) or len(set(b_ids)) != len(b_ids):
            raise ValueError("Feature caches contain duplicate image identities")
        if set(a_ids) != set(b_ids):
            raise ValueError("Feature drift requires exactly the same image identities")
        if not np.array_equal(base["classes"], method["classes"]):
            raise ValueError("Feature caches use different class orders")
        lookup = {identity: i for i, identity in enumerate(b_ids)}
        order = [lookup[x] for x in a_ids]
        result = summarise_drift(base["v"], method["v"][order])
        result.update(num_samples=len(a_ids), tolerance=tolerance, expect_frozen=expect_frozen)
        result["within_tolerance"] = result["max_absolute_difference"] <= tolerance
        return result

