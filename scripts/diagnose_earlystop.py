"""诊断：batch=64 下早停为何挑走训练不足的 checkpoint。

现象（100pct，batch=64）：
    seed42  ep1..ep5  loss 0.6392 → 0.5055（还在降）
             val AUC 0.9147, 0.9170, 0.9145, 0.9130, 0.9138（噪声波动）
    → 最佳 val AUC 出现在 ep2，连续 3 轮不升即早停
    → test F1 只有 0.51，而 batch=16 跑满 10 轮是 0.85

假设：源域 val 只有 191 张有效图，Macro-AUC 早已饱和，无法反映
分类头是否校准好。于是"早停 + 按 val AUC 选 checkpoint"两件事同时
失效——损失还在降，模型却被迫停在半路。

本脚本跑满 10 轮（禁用早停），逐轮同时记录 val 的 AUC / F1 / AUPRC
和训练 loss，最后分别评测**最佳 AUC checkpoint**与**最后一轮 checkpoint**
在源域 test 和目标域 test 上的表现。

产物 scope=diagnostic，不进入正式表格::

    outputs/step1_diag/earlystop_diagnosis.json
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
from src.trainer import load_checkpoint, masked_bce_loss, save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step1_linear_probe.local.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", default="100pct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="outputs/step1_diag")
    args = parser.parse_args()

    config = read_config(args.config)
    config["experiment"].update(seed=args.seed, budget=args.budget,
                                batch_size=args.batch_size, method="linear_probe")
    config.setdefault("data", {})["output_dir"] = args.out
    config.setdefault("training", {})["epochs"] = args.epochs
    seed_everything(args.seed)
    torch.set_num_threads(int(config.get("model", {}).get("cpu_threads", 4)))

    data_dir = project_path(config["data"].get("data_dir", "data"))
    frame = load_manifest(data_dir / "mimic_manifest.csv")
    assert_patient_disjoint([group for _, group in frame.groupby("split")])
    ids = pd.read_csv(data_dir / f"source_train_seed{args.seed}_{args.budget}.csv").patient_id.astype(str)
    train = frame[frame.split.eq("train") & frame.patient_id.astype(str).isin(ids)]
    val = frame[frame.split.eq("val")]

    bundle = load_medclip(config)
    model = build_model(bundle, config)
    out = output_dir(config)
    device = str(next(model.parameters()).device)

    def loader(rows, shuffle=False):
        return make_loader(rows.sort_values(["patient_id", "study_id", "image_id"]),
                           bundle.processor, batch_size=args.batch_size, shuffle=shuffle, seed=args.seed)

    train_loader, val_loader = loader(train, True), loader(val)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    best_path = out / "diag_best_auc.pt"
    last_path = out / "diag_last_epoch.pt"

    history, best_auc = [], -float("inf")
    print(f"训练 {len(train)} 张（{args.budget}, batch={args.batch_size}, "
          f"{args.epochs} 轮，禁用早停）", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            keep = batch["mask"].bool().any(dim=1)
            if not keep.any():
                continue
            idx = keep.nonzero(as_tuple=True)[0].tolist()
            batch = {k: (v[idx] if hasattr(v, "shape") else [v[i] for i in idx]) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_batch(batch)
            loss = masked_bce_loss(output.logits, batch["labels"].to(device), batch["mask"].to(device))
            loss.backward()
            optimizer.step()
            weight = int(batch["mask"].sum())
            total += float(loss.detach().cpu()) * weight
            count += weight
        # 每轮记录 val 的全部指标，而不只是 AUC
        result = evaluate_model(model, val_loader, domain="mimic", split="val", method="linear_probe",
                                budget=args.budget, seed=args.seed, output_dir=out,
                                threshold=0.5, scope="diagnostic", provenance=bundle.provenance)
        m = result["metrics"]
        row = {"epoch": epoch + 1, "train_loss": total / max(count, 1),
               "val_macro_auc": m["macro_auc"], "val_macro_f1": m["macro_f1"],
               "val_macro_auprc": m["macro_auprc"]}
        history.append(row)
        print(f"  ep{epoch+1}: loss={row['train_loss']:.4f}  "
              f"valAUC={row['val_macro_auc']:.4f}  valF1={row['val_macro_f1']:.4f}  "
              f"valAUPRC={row['val_macro_auprc']:.4f}", flush=True)
        if row["val_macro_auc"] > best_auc:
            best_auc = row["val_macro_auc"]
            save_checkpoint(best_path, model, dict(epoch=epoch + 1, val_macro_auc=best_auc))
    save_checkpoint(last_path, model, dict(epoch=args.epochs, val_macro_auc=history[-1]["val_macro_auc"]))

    # 两份 checkpoint 在同一批 test 数据上对比
    comparisons = {}
    for label, path in (("best_val_auc", best_path), ("last_epoch", last_path)):
        load_checkpoint(path, model, device)
        scores = {}
        for domain, split in (("mimic", "test"), ("chexpert", "test")):
            rows = frame[frame.split.eq(split)] if domain == "mimic" else \
                load_manifest(data_dir / "chexpert_manifest.csv").query("split == 'test'")
            r = evaluate_model(model, loader(rows), domain=domain, split=split,
                               method=f"linear_probe_diag_{label}", budget=args.budget,
                               seed=args.seed, output_dir=out, threshold=0.5,
                               scope="diagnostic", provenance=bundle.provenance)
            scores[f"{domain}/{split}"] = {k: r["metrics"][k]
                                           for k in ("macro_auc", "macro_f1", "macro_auprc")}
        comparisons[label] = scores

    report = {
        "scope": "diagnostic_early_stopping_analysis",
        "seed": args.seed, "budget": args.budget, "batch_size": args.batch_size,
        "epochs": args.epochs, "train_images": len(train), "val_images": len(val),
        "history": history,
        "best_val_auc_epoch": max(history, key=lambda h: h["val_macro_auc"])["epoch"],
        "best_val_f1_epoch": max(history, key=lambda h: h["val_macro_f1"])["epoch"],
        "test_comparison": comparisons,
    }
    write_json(out / "earlystop_diagnosis.json", report)
    print("\n=== 对比 ===", flush=True)
    for label, scores in comparisons.items():
        for key, value in scores.items():
            print(f"{label:14s} {key:15s} AUC {value['macro_auc']:.4f}  F1 {value['macro_f1']:.4f}",
                  flush=True)
    print(f"\n最佳 val AUC 出现在 ep{report['best_val_auc_epoch']}，"
          f"最佳 val F1 出现在 ep{report['best_val_f1_epoch']}", flush=True)


if __name__ == "__main__":
    main()
