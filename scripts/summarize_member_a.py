"""汇总成员A 的正式结果，生成报告用的 Markdown 表格。

读取 ``outputs/step1/`` 下的 metrics_summary_*.csv、measurement_*.json、
inference_*.json，按 方法 × 预算 × 域/划分 聚合，给出 3 个 seed 的
均值 ± 标准差，并列出实验要求第 5 条要的资源记录（训练时间、1000 张推理
时间、峰值显存、参数量）。

只统计 scope=experiment 的行；smoke / diagnostic / check16 一律排除，
避免把冒烟测试的数字混进正式表格。

用法（项目根目录）:
    python scripts/summarize_member_a.py
    python scripts/summarize_member_a.py --dir outputs/step1_b64 --output /tmp/t.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRICS = ["macro_auc", "macro_f1", "macro_auprc"]
DOMAIN_ORDER = [("mimic", "val"), ("mimic", "test"), ("chexpert", "test")]
METHOD_ORDER = ["pretrained", "linear_probe"]


def fmt(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def load_measurements(directory: Path) -> tuple[list[dict], list[dict]]:
    train, infer = [], []
    for path in sorted(directory.glob("measurement_*.json")):
        train.append(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted(directory.glob("inference_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("scope") == "experiment":
            infer.append(data)
    return train, infer


def measurement_section(train: list[dict], infer: list[dict]) -> list[str]:
    lines = ["## 资源与规模记录（实验要求第 5 条）", ""]
    if train:
        lines += ["### 训练", "",
                  "| 方法 | 预算 | seed | epoch | 训练时间 (s) | 峰值显存 (MB) | 参数量 | 可训练量 | 可训练占比 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in train:
            params = row["parameters"]
            lines.append(
                f"| {row['method']} | {row['budget']} | {row['seed']} | {row['epochs_run']} | "
                f"{row['training_seconds']} | {row['peak_memory_allocated_mb']} | "
                f"{params['total_parameters']:,} | {params['trainable_parameters']:,} | "
                f"{params['trainable_ratio']:.4%} |")
        lines.append("")
        # 逐层检查结果单独强调：这是实验要求里"逐层检查"的凭据
        lines += ["### 逐层更新检查", "",
                  "| 预算 | seed | 实际变化的层 | 冻结却变化 | 只有预期层变化 |",
                  "|---|---|---|---|---|"]
        for row in train:
            upd = row["layerwise_update"]
            lines.append(f"| {row['budget']} | {row['seed']} | "
                         f"{', '.join(upd['changed']) or '（无）'} | "
                         f"{', '.join(upd['frozen_but_changed']) or '无'} | "
                         f"{'是' if upd['only_expected_changed'] else '**否**'} |")
        lines.append("")

    if infer:
        lines += ["### 推理（eval + no_grad，按 1000 张折算）", "",
                  "| 方法 | 预算 | seed | 域/划分 | 实测张数 | 耗时 (s) | 1000 张耗时 (s) | 峰值显存 (MB) | 参数未变 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in infer:
            info = row.get("inference")
            if not info:
                continue
            lines.append(
                f"| {row['method']} | {row['budget']} | {row['seed']} | "
                f"{row['domain']}/{row['split']} | {info['images']} | {info['seconds']} | "
                f"{info['seconds_per_1000_images']} | {info['peak_memory_allocated_mb']} | "
                f"{'是' if info['parameters_unchanged'] else '**否**'} |")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="outputs/step1", help="产物目录")
    parser.add_argument("--output", help="默认写到 <dir>/member_a_tables.md")
    args = parser.parse_args()

    directory = ROOT / args.dir
    files = sorted(directory.glob("metrics_summary_*.csv"))
    if not files:
        sys.exit(f"{directory} 里没有 metrics_summary_*.csv")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    total = len(frame)
    frame = frame[frame.scope.eq("experiment")]
    print(f"汇总: 共 {total} 份 summary，其中 scope=experiment 的 {len(frame)} 份")

    lines = ["# 成员A 正式结果", ""]
    lines.append(f"数据来源：`{args.dir}/metrics_summary_*.csv`，只取 `scope=experiment`。")
    lines.append(f"共 {len(frame)} 次评测；数值为 3 个 seed 的 均值 ± 标准差。")
    lines.append("")

    for metric in METRICS:
        lines.append(f"## {metric}")
        lines.append("")
        header = "| 方法 | 预算 | " + " | ".join(f"{d}/{s}" for d, s in DOMAIN_ORDER) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (2 + len(DOMAIN_ORDER)))
        for method in METHOD_ORDER:
            for budget in ("10pct", "100pct"):
                subset = frame[frame.method.eq(method) & frame.budget.eq(budget)]
                if subset.empty:
                    continue
                cells = []
                for domain, split in DOMAIN_ORDER:
                    group = subset[subset.domain.eq(domain) & subset.split.eq(split)][metric]
                    cells.append(fmt(group.mean(), group.std(ddof=1) if len(group) > 1 else np.nan))
                lines.append(f"| {method} | {budget} | " + " | ".join(cells) + " |")
        lines.append("")

    # 样本量单独列出来，报告里必须交代
    lines.append("## 每次评测的样本量")
    lines.append("")
    lines.append("| 方法 | 预算 | 域/划分 | num_samples | 跳过（无有效标签） |")
    lines.append("|---|---|---|---|---|")
    for method in METHOD_ORDER:
        for budget in ("10pct", "100pct"):
            subset = frame[frame.method.eq(method) & frame.budget.eq(budget)]
            for domain, split in DOMAIN_ORDER:
                group = subset[subset.domain.eq(domain) & subset.split.eq(split)]
                if group.empty:
                    continue
                lines.append(f"| {method} | {budget} | {domain}/{split} | "
                             f"{int(group.num_samples.iloc[0])} | "
                             f"{int(group.skipped_no_valid_samples.iloc[0])} |")
    lines.append("")

    train_measure, infer_measure = load_measurements(directory)
    lines += measurement_section(train_measure, infer_measure)

    target = ROOT / (args.output or f"{args.dir}/member_a_tables.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"写入 {target}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
