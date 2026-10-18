"""H1/H3 result aggregation; independent of how a method is implemented."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import CLASS_NAMES
from .metrics import compute_multilabel_metrics


def summarise_transfer(frame, baseline="pretrained"):
    frame = frame[frame["split"].eq("test")].copy()
    keys = ["method", "budget", "seed", "scope", "domain"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Need unique test summaries for method/budget/seed/scope/domain")
    rows = []
    metrics = ("macro_auc", "macro_f1", "macro_auprc")
    for (method, budget, seed, scope), group in frame.groupby(keys[:-1]):
        if set(group.domain) != {"mimic", "chexpert"}:
            raise ValueError(f"Need both domains for {method}/{budget}/{seed}/{scope}")
        base = frame[frame.method.eq(baseline) & frame.budget.eq(budget)
                     & frame.seed.eq(seed) & frame.scope.eq(scope)]
        if set(base.domain) != {"mimic", "chexpert"}:
            raise ValueError("Baseline needs both domains with the same budget, seed and scope")
        group, base = group.set_index("domain"), base.set_index("domain")
        for domain in ("mimic", "chexpert"):
            for protocol in ("evaluation_set_sha256", "threshold"):
                if protocol not in group or group.loc[domain, protocol] != base.loc[domain, protocol]:
                    raise ValueError(f"H1 evaluation protocol mismatch: {domain}/{protocol}")
        row = dict(method=method, budget=budget, seed=seed, scope=scope)
        for metric in metrics:
            source, target = float(group.loc["mimic", metric]), float(group.loc["chexpert", metric])
            row.update({f"source_{metric}": source, f"target_{metric}": target,
                        f"source_gain_{metric}": source - float(base.loc["mimic", metric]),
                        f"target_gain_{metric}": target - float(base.loc["chexpert", metric]),
                        f"transfer_drop_{metric}": source - target})
        rows.append(row)
    return pd.DataFrame(rows)


def compare_prompts(single, ensemble):
    if single.sample_id.duplicated().any() or ensemble.sample_id.duplicated().any():
        raise ValueError("Duplicate sample identities")
    if set(single.sample_id) != set(ensemble.sample_id):
        raise ValueError("H3 requires the same images for both prompt settings")
    ensemble = ensemble.set_index("sample_id").loc[single.sample_id].reset_index()
    for column in ["domain", "split", "scope", "budget", "seed"]:
        if single[column].nunique() != 1 or not np.array_equal(single[column].to_numpy(), ensemble[column].to_numpy()):
            raise ValueError(f"H3 protocol mismatch: {column}")
    y_cols = [f"label_{c}" for c in CLASS_NAMES]
    m_cols = [f"mask_{c}" for c in CLASS_NAMES]
    p_cols = [f"prob_{c}" for c in CLASS_NAMES]
    if not np.array_equal(single[y_cols + m_cols], ensemble[y_cols + m_cols]):
        raise ValueError("Paired prediction labels/masks differ")
    a = compute_multilabel_metrics(single[y_cols].to_numpy(), single[p_cols].to_numpy(), single[m_cols].to_numpy(), CLASS_NAMES)
    b = compute_multilabel_metrics(ensemble[y_cols].to_numpy(), ensemble[p_cols].to_numpy(), ensemble[m_cols].to_numpy(), CLASS_NAMES)
    return {"num_samples": len(single), "single": a, "five": b,
            "delta_five_minus_single": {k: b[k] - a[k] for k in ("macro_auc", "macro_f1", "macro_auprc")}}
