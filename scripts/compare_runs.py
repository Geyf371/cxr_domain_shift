"""对比两个产物目录的 macro 指标（例如 batch=16 与 batch=64）。

用法::

    python scripts/compare_runs.py outputs/step1 outputs/step1_b64
    python scripts/compare_runs.py A B --budget 10pct

只比较 ``scope=experiment`` 的 metrics_summary，按
(method, budget, domain, split, seed) 对齐后求 3 seed 均值 ± 标准差，
再逐格给出 B − A 的差值。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["method", "budget", "domain", "split", "seed"]
METRICS = ["macro_auc", "macro_f1", "macro_auprc"]


def collect(directory: Path, budget: str | None) -> pd.DataFrame:
    files = sorted(directory.glob("metrics_summary_*.csv"))
    if not files:
        raise SystemExit(f"{directory} 里没有 metrics_summary_*.csv")
    frames = [pd.read_csv(f) for f in files]
    frame = pd.concat(frames, ignore_index=True)
    if "scope" in frame:
        frame = frame[frame.scope.eq("experiment")]
    if budget:
        frame = frame[frame.budget.eq(budget)]
    missing = [m for m in METRICS if m not in frame.columns]
    if missing:
        raise SystemExit(f"缺少指标列: {missing}")
    return frame[KEYS + METRICS]


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["method", "budget", "domain", "split"])[METRICS]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    count = grouped.count().iloc[:, :1].rename(columns={METRICS[0]: "n_seeds"})
    return mean.join(std).join(count).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="基准目录（例如 outputs/step1）")
    parser.add_argument("b", type=Path, help="对照目录（例如 outputs/step1_b64）")
    parser.add_argument("--budget", help="只比某个预算，例如 10pct")
    args = parser.parse_args()

    a = aggregate(collect(ROOT / args.a, args.budget))
    b = aggregate(collect(ROOT / args.b, args.budget))
    key = ["method", "budget", "domain", "split"]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    if merged.empty:
        raise SystemExit("两个目录没有可对齐的 (method,budget,domain,split)")

    print(f"A = {args.a}    B = {args.b}\n")
    for metric in METRICS:
        print(f"=== {metric} ===")
        header = f"{'method':<16}{'budget':<8}{'domain/split':<18}{'A':>18}{'B':>18}{'B-A':>10}"
        print(header)
        print("-" * len(header))
        for _, row in merged.iterrows():
            a_mean, a_std = row[f"{metric}_mean_a"], row[f"{metric}_std_a"]
            b_mean, b_std = row[f"{metric}_mean_b"], row[f"{metric}_std_b"]
            delta = b_mean - a_mean
            flag = "  <<<" if abs(delta) >= 0.01 else ""
            print(f"{row['method']:<16}{row['budget']:<8}"
                  f"{row['domain'] + '/' + row['split']:<18}"
                  f"{a_mean:>11.4f}±{a_std:.4f}"
                  f"{b_mean:>11.4f}±{b_std:.4f}"
                  f"{delta:>+10.4f}{flag}")
        print()


if __name__ == "__main__":
    main()
