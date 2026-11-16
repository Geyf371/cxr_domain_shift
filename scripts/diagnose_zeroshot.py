"""零样本口径诊断：量化 MedCLIP 嵌入的公共分量，并对比三种打分方式。

背景
----
Step 0 的零样本 (`src/model_adapter.py::MedCLIPAdapter`) 用的是

    v = L2norm(vision_model(pixel_values))          # 单位向量
    t = L2norm(text_model(prompt))                  # 单位向量
    q[c] = v·t_pos[c] - v·t_neg[c]                  # 正负差分
    logit = q * logit_scale.exp()                   # 再乘预训练温度

实测发现这条链路在真实权重上退化：

  * 图像嵌入范数约 38.9，其中**公共均值向量**占 97.8%，真正区分不同
    胸片的残差只占 23.5%
  * 文本嵌入同理，不同语义的句子两两余弦都在 0.99 以上
    （"A chest X-ray showing Atelectasis." vs "hello world" 也有 0.99）
  * L2 归一化把残差压到 1/4，正负差分再把两边的公共分量相消，
    最终 logits 量级约 1e-3，sigmoid 后恒为 0.500

结果是零样本 AUC 落在 0.33—0.50，等同于随机。

本脚本不改动 Step 0 任何代码，只做测量与对照，产出
`outputs/step1/zeroshot_diagnosis.json` 供报告引用。

三种打分方式
------------
  A. step0_difference   Step 0 原口径：归一化后 正-负 差分（正式结果用它）
  B. centered_onevsall  用文本集合均值中心化后，只用正提示的相似度
  C. centered_difference 中心化后仍做正负差分

用法（项目根目录）:
    python scripts/diagnose_zeroshot.py
    python scripts/diagnose_zeroshot.py --domain mimic --split val --limit 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, load_manifest
from src.model_loader import load_medclip
from src.prompts import build_prompts
from src.runtime import output_dir, read_config


def auc_or_none(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step1_linear_probe.local.yaml")
    parser.add_argument("--domain", choices=("mimic", "chexpert"), default="mimic")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 张，用于快速核查")
    args = parser.parse_args()

    config = read_config(args.config)
    torch.set_num_threads(int(config.get("model", {}).get("cpu_threads", 4)))
    data_dir = ROOT / config["data"].get("data_dir", "data")
    frame = load_manifest(data_dir / f"{args.domain}_manifest.csv")
    rows = frame[frame.split.eq(args.split)]
    rows = rows[rows[[f"mask_{c}" for c in CLASS_NAMES]].sum(axis=1) > 0]
    if args.limit:
        rows = rows.head(args.limit)
    if rows.empty:
        sys.exit("没有可评测样本")

    bundle = load_medclip(config)
    tokenizer = bundle.processor.tokenizer
    prompts = build_prompts(templates=1)
    # 顺序必须与 src/model_adapter.py 一致：按类别交错 正/负，
    # 即 p0,n0,p1,n1,...；写成 positives+negatives 再 reshape 会错配。
    texts = [text for class_name in CLASS_NAMES for sign in ("positive", "negative")
             for text in prompts[class_name][sign]]
    inputs = tokenizer(texts, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt")
    with torch.no_grad():
        text = bundle.model.text_model(inputs["input_ids"].cuda(), inputs["attention_mask"].cuda())
    images = [Image.open(path).convert("RGB") for path in rows.image_path]
    # 4GB 显存放不下上千张图一次前向，必须分批。
    batch_size = int(config.get("experiment", {}).get("batch_size", 16))
    chunks = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        pixel_values = bundle.processor(images=batch, return_tensors="pt")["pixel_values"].cuda()
        with torch.no_grad():
            chunks.append(bundle.model.vision_model(pixel_values).cpu())
        del pixel_values
        torch.cuda.empty_cache()
    image = torch.cat(chunks, dim=0)
    # 图像嵌入已搬到 CPU；后续打分也在 CPU 上做，避免再次爆显存。
    image = image.cpu()
    text = text.cpu()
    scale = float(bundle.model.logit_scale.clamp(0, 4.6052).exp())

    # ---- 公共分量量化 ----
    image_mean = image.mean(dim=0)
    image_norm = float(image.norm(dim=-1).mean())
    residual_norm = float((image - image_mean).norm(dim=-1).mean())
    text_norm = float(text.norm(dim=-1).mean())
    text_residual = float((text - text.mean(dim=0)).norm(dim=-1).mean())

    def cosine_matrix(value: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(value, dim=-1)
        return normalized @ normalized.t()

    image_cosine = cosine_matrix(image)
    text_cosine = cosine_matrix(text)
    n = len(image)
    off_diagonal = image_cosine[~torch.eye(n, dtype=torch.bool, device=image_cosine.device)]

    # ---- 三种打分 ----
    v_unit = F.normalize(image, dim=-1)
    t_unit = F.normalize(text, dim=-1)
    difference = torch.einsum("bd,cstd->bcst", v_unit, t_unit.reshape(len(CLASS_NAMES), 2, 1, -1))
    diff_scores = (difference[:, :, 0, :] - difference[:, :, 1, :]).mean(-1) * scale
    # 正提示在交错序列里的位置是 0,2,4,6,8
    positive_unit = t_unit[0::2]
    onevsall_scores = torch.einsum("bd,cd->bc", v_unit, positive_unit) * scale

    text_mean = text.mean(dim=0, keepdim=True)
    v_centered = F.normalize(image - text_mean, dim=-1)
    t_centered = F.normalize(text - text_mean, dim=-1)
    c_diff = torch.einsum("bd,cstd->bcst", v_centered, t_centered.reshape(len(CLASS_NAMES), 2, 1, -1))
    centered_diff_scores = (c_diff[:, :, 0, :] - c_diff[:, :, 1, :]).mean(-1) * scale
    centered_ova_scores = torch.einsum("bd,cd->bc", v_centered, t_centered[0::2]) * scale

    results = {}
    for name, scores in (("step0_difference", diff_scores),
                         ("step0_onevsall", onevsall_scores),
                         ("centered_onevsall", centered_ova_scores),
                         ("centered_difference", centered_diff_scores)):
        per_class = {}
        values = []
        for index, class_name in enumerate(CLASS_NAMES):
            mask = rows[f"mask_{class_name}"].astype(bool).to_numpy()
            labels = rows.loc[mask, f"label_{class_name}"].astype(int).to_numpy()
            score = scores[mask, index].detach().cpu().numpy()
            auc = auc_or_none(labels, score)
            per_class[class_name] = auc
            if auc is not None:
                values.append(auc)
        results[name] = {
            "per_class_auc": per_class,
            "macro_auc": float(np.mean(values)) if values else None,
            "scored_classes": len(values),
        }

    report = {
        "scope": "diagnosis_only_not_a_formal_result",
        "domain": args.domain,
        "split": args.split,
        "num_images": len(rows),
        "logit_scale": scale,
        "common_component": {
            "image_embedding_mean_norm": image_norm,
            "image_common_vector_norm": float(image_mean.norm()),
            "image_common_share": float(image_mean.norm()) / image_norm,
            "image_residual_norm": residual_norm,
            "image_residual_share": residual_norm / image_norm,
            "text_embedding_mean_norm": text_norm,
            "text_residual_norm": text_residual,
            "text_residual_share": text_residual / text_norm,
            "image_pairwise_cosine_mean": float(off_diagonal.mean()),
            "image_pairwise_cosine_min": float(off_diagonal.min()),
            "text_pairwise_cosine_mean": float(text_cosine[~torch.eye(len(text), dtype=torch.bool, device=text_cosine.device)].mean()),
            "positive_negative_cosine_mean": float(F.cosine_similarity(
                t_unit[0::2], t_unit[1::2], dim=-1).mean()),
        },
        "raw_score_magnitude": {
            "step0_difference_abs_mean": float(diff_scores.abs().mean()),
            "step0_difference_abs_max": float(diff_scores.abs().max()),
        },
        "methods": results,
        "conclusion": (
            "MedCLIP 的嵌入有约 98% 的公共分量，L2 归一化后正负提示相减会把区分信号抵消到 1e-3 量级，"
            "sigmoid 后恒为 0.5。Step 0 的零样本口径在真实权重上因此退化为随机。"
            "中心化后 one-vs-all 能恢复部分区分度，但这属于修改口径，正式结果仍按 Step 0 原样报告。"
        ),
    }
    target = output_dir(config) / f"zeroshot_diagnosis_{args.domain}_{args.split}.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float),
                      encoding="utf-8", newline="\n")
    print(json.dumps(report["common_component"], indent=2, ensure_ascii=False))
    print()
    for name, value in results.items():
        macro = value["macro_auc"]
        print(f"{name:22s} macro_auc = " + (f"{macro:.4f}" if macro is not None else "不可定义"))
        for class_name, auc in value["per_class_auc"].items():
            print(f"    {class_name:20s} " + (f"{auc:.4f}" if auc is not None else "不可定义"))
    print(f"\n报告: {target}")


if __name__ == "__main__":
    main()
