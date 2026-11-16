"""Step 1 运行入口：在 Step 0 的 train/eval 流程上补测量与逐层检查。

为什么不直接改 ``scripts/run_experiment.py``：
那是 Step 0 的共享代码，成员 B、C 也在用。本脚本复制它的配置/数据装载
逻辑（约 30 行），把 ``fit_source`` / ``evaluate_model`` 包一层，从而在不
改动共享代码的前提下补上实验要求第 5 条要的记录：

    • 训练时间
    • 1000 张图像推理时间
    • 峰值显存
    • 参数量（总量 / 可训练量 / 逐层 requires_grad）
    • 逐层更新检查（只有预期层变化）

命令行与 ``run_experiment.py`` 保持一致，只是多两个开关::

    python scripts/step1_run.py --config C --action train --domain mimic
    python scripts/step1_run.py --config C --action eval --domain chexpert \
        --split test --checkpoint outputs/step1/checkpoint_xxx.pt --measure-inference

产物（除 Step 0 原有的 predictions / metrics / features / metadata 外）：

    measurement_{method}_{budget}_seed{seed}.json     训练时间、显存、参数量、逐层更新
    inference_{method}_{domain}_{split}_{budget}_seed{seed}.json   推理时间、显存
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import assert_patient_disjoint, load_manifest, make_loader
from src.evaluator import evaluate_model
from src.model_adapter import build_model
from src.model_loader import load_medclip
from src.runtime import output_dir, project_path, read_config, seed_everything, write_json
from src.step1_methods.measure import TrainingMeter, measure_inference, parameter_report
from src.trainer import fit_source, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step1_linear_probe.local.yaml")
    parser.add_argument("--action", choices=["train", "eval"], required=True)
    parser.add_argument("--domain", choices=["mimic", "chexpert"], default="mimic")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint")
    parser.add_argument("--measure-inference", action="store_true",
                        help="额外测量纯推理耗时（eval + no_grad，按 1000 张折算）")
    args = parser.parse_args()

    config = read_config(args.config)
    exp = config.setdefault("experiment", {})
    seed = int(exp.get("seed", 42))
    budget = exp.get("budget", "10pct")
    batch_size = int(exp.get("batch_size", 4))
    exp.update(seed=seed, budget=budget, method=exp.get("method", "pretrained"))
    seed_everything(seed)
    torch.set_num_threads(int(config.get("model", {}).get("cpu_threads", 4)))

    if args.domain == "chexpert" and (args.action == "train" or args.split != "test"):
        raise ValueError("CheXpert is test-only")

    data_dir = project_path(config["data"].get("data_dir", "data"))
    frame = load_manifest(data_dir / f"{args.domain}_manifest.csv")
    if set(frame.domain) != {args.domain}:
        raise ValueError("Manifest domain mismatch")
    assert_patient_disjoint([group for _, group in frame.groupby("split")])

    bundle = load_medclip(config)
    model = build_model(bundle, config)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, next(model.parameters()).device)
    out = output_dir(config)

    def loader(rows, shuffle=False):
        if rows.empty:
            raise ValueError("Requested split/subset is empty")
        return make_loader(rows.sort_values(["patient_id", "study_id", "image_id"]), bundle.processor,
                           batch_size=batch_size, shuffle=shuffle, seed=seed)

    if args.action == "train":
        ids = pd.read_csv(data_dir / f"source_train_seed{seed}_{budget}.csv").patient_id.astype(str)
        train = frame[frame.split.eq("train") & frame.patient_id.astype(str).isin(ids)]
        if set(train.patient_id.astype(str)) != set(ids):
            raise ValueError("Training subset contains missing or non-train patients")
        train_loader = loader(train, True)
        val_loader = loader(frame[frame.split.eq("val")])

        meter = TrainingMeter(model)
        with meter:
            result = fit_source(model, train_loader, val_loader, config, out, bundle.provenance)
        result["measurement"] = meter.report()
        write_json(out / f"training_{exp['method']}_{budget}_seed{seed}.json", result)

        # 训练结束时模型已载入最佳 checkpoint，此报告描述的就是最终交付的模型
        report = meter.report()
        report.update({
            "scope": "experiment",
            "method": exp["method"],
            "budget": budget,
            "seed": seed,
            "batch_size": batch_size,
            "epochs_run": len(result["history"]),
            "early_stopped": len(result["history"]) < int(config.get("training", {}).get("epochs", 10)),
            "best_val_macro_auc": result["best_val_macro_auc"],
            "checkpoint": result["checkpoint"],
            "train_images": len(train),
            "val_images": len(frame[frame.split.eq("val")]),
        })
        write_json(out / f"measurement_{exp['method']}_{budget}_seed{seed}.json", report)
        _print_layer_summary(report)
        return

    # ---- eval ----
    rows = frame[frame.split.eq(args.split)]
    eval_loader = loader(rows)
    method = exp["method"] + ("_5prompts" if exp.get("prompt_templates") == 5 else "")
    result = evaluate_model(model, eval_loader, domain=args.domain, split=args.split, method=method,
                            budget=budget, seed=seed, output_dir=out,
                            threshold=float(exp.get("threshold", 0.5)), scope="experiment",
                            provenance=bundle.provenance)
    print(result["paths"])

    report = parameter_report(model)
    report.update({"scope": "experiment", "method": method, "budget": budget, "seed": seed,
                   "domain": args.domain, "split": args.split, "batch_size": batch_size,
                   "images_evaluated": int(len(result["predictions"])),
                   "checkpoint": args.checkpoint})
    if args.measure_inference:
        report["inference"] = measure_inference(model, eval_loader, target_images=1000)
    tag = f"{method}_{args.domain}_{args.split}_{budget}_seed{seed}"
    write_json(out / f"inference_{tag}.json", report)


def _print_layer_summary(report: dict) -> None:
    params = report["parameters"]
    updates = report["layerwise_update"]
    print(f"\n参数量: {params['total_parameters']:,} 总 / "
          f"{params['trainable_parameters']:,} 可训练 "
          f"({params['trainable_ratio']:.4%})", flush=True)
    print(f"可训练层: {params['trainable_names']}", flush=True)
    print(f"实际变化层: {updates['changed']}", flush=True)
    print(f"冻结却变化: {updates['frozen_but_changed'] or '无'}", flush=True)
    print(f"训练耗时: {report['training_seconds']}s  "
          f"峰值显存: {report['peak_memory_allocated_mb']} MB", flush=True)
    if updates["frozen_but_changed"]:
        raise SystemExit(f"逐层检查失败：冻结层发生了变化 {updates['frozen_but_changed']}")


if __name__ == "__main__":
    main()
