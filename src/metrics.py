"""Masked multilabel metrics used by the hypotheses."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def compute_multilabel_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
    class_names: Sequence[str],
    threshold: float = 0.5,
) -> dict:
    """Compute masked per-class and macro metrics.

    AUC is undefined when a valid class has only one label value. Undefined
    classes are reported as NaN and excluded from the corresponding macro mean.
    """

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    mask = np.asarray(mask).astype(bool)
    if labels.shape != probabilities.shape or labels.shape != mask.shape:
        raise ValueError("labels, probabilities, and mask must have the same shape")
    if labels.ndim != 2 or labels.shape[1] != len(class_names):
        raise ValueError("Expected arrays shaped [num_samples, num_classes]")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.isin(labels[mask], [0, 1]).all():
        raise ValueError("Valid labels must be binary")

    rows = []
    for index, class_name in enumerate(class_names):
        valid = mask[:, index]
        y_true = labels[valid, index].astype(int)
        y_prob = probabilities[valid, index].astype(float)
        y_pred = (y_prob >= threshold).astype(int)
        positives = int(y_true.sum()) if y_true.size else 0
        negatives = int(y_true.size - positives)
        auc = float("nan")
        if positives > 0 and negatives > 0:
            auc = float(roc_auc_score(y_true, y_prob))
        f1 = float(f1_score(y_true, y_pred, zero_division=0)) if y_true.size else float("nan")
        auprc = (
            float(average_precision_score(y_true, y_prob))
            if positives > 0
            else float("nan")
        )
        rows.append(
            {
                "class_name": class_name,
                "auc": auc,
                "f1": f1,
                "auprc": auprc,
                "valid_count": int(y_true.size),
                "positive_count": positives,
                "negative_count": negatives,
            }
        )

    def nanmean(key: str) -> float:
        values = np.asarray([row[key] for row in rows], dtype=float)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    return {
        "per_class": rows,
        "macro_auc": nanmean("auc"),
        "macro_f1": nanmean("f1"),
        "macro_auprc": nanmean("auprc"),
        "macro_auc_valid_classes": int(sum(np.isfinite(row["auc"]) for row in rows)),
        "macro_f1_valid_classes": int(sum(np.isfinite(row["f1"]) for row in rows)),
        "macro_auprc_valid_classes": int(sum(np.isfinite(row["auprc"]) for row in rows)),
        "num_samples": int(len(labels)),
        "no_valid_sample_count": int((~mask.any(axis=1)).sum()),
        "threshold": threshold,
    }

