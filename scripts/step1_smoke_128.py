"""Step 1 · 成员A 的 128 患者冒烟测试。

目的：在正式跑 6 组实验之前，用最小代价确认

  1. 真实图像能过 processor，前向不报错
  2. 只有 ``classifier.weight/bias`` 在更新，主干参数与 buffer 逐位不变
  3. 主干拿不到梯度（``p.grad is None``），确认真的冻结
  4. 训练 loss 下降（2 个 epoch，足以看出方向）
  5. 同一批图重复前向，``v`` 逐位一致（Step 3 漂移分析的前提）

不做早停、不保存正式 checkpoint、产物落到 ``outputs/step1/smoke128/``，
文件名带 ``smoke`` 前缀，不会被 collect_metrics 当成正式结果。

用法（项目根目录）:
    python scripts/step1_smoke_128.py
    python scripts/step1_smoke_128.py --config config/step1_linear_probe.local.yaml --epochs 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, assert_patient_disjoint, load_manifest, make_loader
from src.model_loader import load_medclip
from src.model_adapter import build_model
from src.runtime import output_dir, read_config, seed_everything
from src.trainer import masked_bce_loss

PATIENTS = 128
REPORT_NAME = "smoke128_linear_probe.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step1_linear_probe.local.yaml")
    parser.add_argument("--patients", type=int, default=PATIENTS)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = read_config(args.config)
    exp = config["experiment"]
    seed, budget = int(exp.get("seed", 42)), exp.get("budget", "10pct")
    seed_everything(seed)
    torch.set_num_threads(int(config.get("model", {}).get("cpu_threads", 4)))

    data_dir = ROOT / config["data"].get("data_dir", "data")
    frame = load_manifest(data_dir / "mimic_manifest.csv")
    assert_patient_disjoint([group for _, group in frame.groupby("split")])

    # 从 10pct 名单里取前 N 个患者，仍然只取 train 划分，保持患者不交叠
    subset = pd.read_csv(data_dir / f"source_train_seed{seed}_{budget}.csv").patient_id.astype(str)
    train_ids = set(frame.loc[frame.split.eq("train"), "patient_id"].astype(str))
    chosen = [pid for pid in subset if pid in train_ids][: args.patients]
    if len(chosen) < args.patients:
        sys.exit(f"训练名单里只有 {len(chosen)} 个可用患者，不足 {args.patients}")
    train = frame[frame.patient_id.astype(str).isin(chosen)]
    val = frame[frame.split.eq("val")]

    print(f"训练患者 {len(chosen)}, 图像 {len(train)}; 验证图像 {len(val)}", flush=True)
    bundle = load_medclip(config)
    model = build_model(bundle, config)
    trainable = model.trainable_parameter_names()
    print(f"可训练参数: {trainable}", flush=True)
    if trainable != ["classifier.weight", "classifier.bias"]:
        sys.exit(f"可训练参数不符预期: {trainable}")

    # 记录主干全部参数和 buffer 的快照
    backbone = model.backbone
    snapshot = {name: p.detach().cpu().clone() for name, p in backbone.named_parameters()}
    buffers = {name: b.detach().cpu().clone() for name, b in backbone.named_buffers()}
    versions = {name: p._version for name, p in backbone.named_parameters()}
    head_before = {n: p.detach().cpu().clone() for n, p in model.classifier.named_parameters()}

    loader = make_loader(train, bundle.processor, batch_size=args.batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(val, bundle.processor, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    device = str(next(model.parameters()).device)

    history = []
    for epoch in range(args.epochs):
        model.train()
        started, total, count, batches = time.time(), 0.0, 0, 0
        for batch in loader:
            keep = batch["mask"].bool().any(dim=1)
            if not keep.any():
                continue
            indices = keep.nonzero(as_tuple=True)[0].tolist()
            batch = {k: (v[indices] if hasattr(v, "shape") else [v[i] for i in indices])
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_batch(batch)
            loss = masked_bce_loss(output.logits, batch["labels"].to(device), batch["mask"].to(device))
            if not torch.isfinite(loss):
                sys.exit("训练 loss 非有限值")
            loss.backward()
            optimizer.step()
            weight = int(batch["mask"].sum())
            total += float(loss.detach().cpu()) * weight
            count += weight
            batches += 1
        elapsed = time.time() - started
        history.append({"epoch": epoch + 1, "loss": total / max(count, 1),
                        "batches": batches, "seconds": round(elapsed, 1)})
        print(f"  epoch {epoch+1}: loss={total/max(count,1):.4f}  "
              f"{batches} 批  {elapsed:.0f}s", flush=True)

    # ---- 断言 ----
    failures = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""), flush=True)
        if not ok:
            failures.append(label)

    # 1. 主干无梯度
    grads = [name for name, p in backbone.named_parameters() if p.grad is not None]
    check(not grads, "主干参数没有梯度", f"有梯度的: {grads[:3]}" if grads else "")

    # 2. 主干参数与 buffer 逐位不变
    changed = [n for n, p in backbone.named_parameters()
               if not torch.equal(p.detach().cpu(), snapshot[n])]
    check(not changed, "主干参数逐位不变", f"变化的: {changed[:3]}" if changed else "")
    changed_buf = [n for n, b in backbone.named_buffers()
                   if not torch.equal(b.detach().cpu(), buffers[n])]
    check(not changed_buf, "主干 buffer 逐位不变", f"变化的: {changed_buf[:3]}" if changed_buf else "")
    same_version = all(p._version == versions[n] for n, p in backbone.named_parameters())
    check(same_version, "主干参数版本号未变（无原地写入）")

    # 3. 分类头确实更新了
    moved = [n for n, p in model.classifier.named_parameters()
             if not torch.equal(p.detach().cpu(), head_before[n])]
    check(len(moved) == 2, "分类头权重已更新", f"{moved}")

    # 4. loss 下降
    check(len(history) >= 2 and history[-1]["loss"] < history[0]["loss"],
          "训练 loss 下降", f"{history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}")

    # 5. v 确定性：同一批图两次前向逐位一致
    model.eval()
    probe = next(iter(val_loader))
    probe = {k: (v[:8] if hasattr(v, "shape") else v[:8]) for k, v in probe.items()}
    with torch.no_grad():
        v1 = model.forward_batch(probe).image_features_v
        v2 = model.forward_batch(probe).image_features_v
    check(torch.equal(v1, v2), "v 重复前向逐位一致")
    check(bool(torch.allclose(v1.norm(dim=-1), torch.ones(len(v1), device=v1.device), atol=1e-5)),
          "v 已 L2 归一化", f"norm={float(v1.norm(dim=-1).mean()):.6f}")

    # 6. logits 形状与有限性
    with torch.no_grad():
        out = model.forward_batch(probe)
    check(tuple(out.logits.shape) == (len(v1), len(CLASS_NAMES)), "logits 形状 [B,5]",
          str(tuple(out.logits.shape)))
    check(bool(torch.isfinite(out.logits).all()), "logits 有限")
    check(tuple(out.joint_features_z.shape)[1] == model.joint_dim, "z 维数 = d+5",
          f"{tuple(out.joint_features_z.shape)[1]}")

    # 7. 特征里没有 NaN
    check(bool(np.isfinite(out.joint_features_z.cpu().numpy()).all()), "z 无 NaN/Inf")

    report = {
        "scope": "smoke_128_patients_not_a_formal_experiment",
        "patients": len(chosen),
        "train_images": len(train),
        "val_images": len(val),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "trainable_parameters": trainable,
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "history": history,
        "backbone_gradients": grads,
        "backbone_parameters_changed": changed,
        "backbone_buffers_changed": changed_buf,
        "classifier_updated": moved,
        "v_bitwise_stable": bool(torch.equal(v1, v2)),
        "joint_dim": int(model.joint_dim),
        "failures": failures,
    }
    out_dir = output_dir(config)
    target = out_dir / REPORT_NAME
    import json
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"\n报告: {target}", flush=True)
    if failures:
        sys.exit(f"冒烟测试失败 {len(failures)} 项: {failures}")
    print("冒烟测试全部通过")


if __name__ == "__main__":
    main()
