"""Identify target-domain errors with an independently measured source-domain gain."""

from itertools import product
from pathlib import Path

import pandas as pd


def select_cases(predictions, target_manifest, class_summary, classes, budgets, seeds, threshold):
    """Select five distinct patients, covering eligible diseases before repeating one."""
    scores = class_summary.set_index(["method", "budget", "domain", "class_name"])["auc_mean"]
    candidates = []
    counts = []
    for budget, method, disease in product(budgets, ("lora_r8", "lora_r16"), classes):
        source_gain = scores[method, budget, "mimic", disease] - scores["linear_probe", budget, "mimic", disease]
        target_gain = scores[method, budget, "chexpert", disease] - scores["linear_probe", budget, "chexpert", disease]
        relevant = source_gain > 0 and target_gain < 0
        mask = target_manifest.loc[target_manifest[f"mask_{disease}"]]
        truth = mask[f"label_{disease}"].astype(int)
        probability_key = f"prob_{disease}"
        seed_errors = pd.DataFrame(index=mask.index)
        margins = pd.DataFrame(index=mask.index)
        for seed in seeds:
            probe = predictions["linear_probe", budget, seed, "chexpert"].loc[mask.index, probability_key]
            lora = predictions[method, budget, seed, "chexpert"].loc[mask.index, probability_key]
            # Same target image, label, budget and seed. The source-domain gain comes
            # from disease-level AUC computed on the separate MIMIC test cohort.
            seed_errors[seed] = (probe.ge(threshold) == truth) & (lora.ge(threshold) != truth)
            margins[seed] = (lora - threshold).abs().where(seed_errors[seed])
        n_discordant = seed_errors.sum(axis=1)
        repeats = n_discordant.ge(2)
        counts.append({
            "method": method, "budget": budget, "class_name": disease,
            "source_auc_gain_vs_linear_probe": source_gain, "target_auc_gain_vs_linear_probe": target_gain,
            "source_gain_and_target_loss": relevant, "target_valid_labels": len(mask),
            "target_probe_correct_lora_wrong_in_at_least_2_seeds": int(repeats.sum()),
        })
        if not relevant:
            continue
        for sample_id in repeats.index[repeats]:
            row = mask.loc[sample_id]
            label = int(truth[sample_id])
            candidates.append({
                "sample_id": sample_id, "patient_id": row["patient_id"],
                "image_path": row["image_path"], "view_position": row["view_position"],
                "budget": budget, "method": method, "class_name": disease,
                "label": label, "error": "假阴性" if label else "假阳性",
                "source_auc_gain_vs_linear_probe": float(source_gain),
                "target_auc_gain_vs_linear_probe": float(target_gain),
                "discordant_seeds": [int(seed) for seed in seeds if seed_errors.at[sample_id, seed]],
                "mean_wrong_margin": float(margins.loc[sample_id].mean()),
                "valid_labels": {name: int(row[f"label_{name}"]) for name in classes if row[f"mask_{name}"]},
                "unknown_labels": [name for name in classes if not row[f"mask_{name}"]],
                "raw_labels": {name: row[f"raw_{name}"] for name in classes},
            })

    candidates.sort(key=lambda row: (
        -len(row["discordant_seeds"]), row["target_auc_gain_vs_linear_probe"],
        -row["mean_wrong_margin"], row["sample_id"], row["budget"], row["method"],
    ))
    selected = []
    patients, diseases = set(), set()
    for require_new_disease in (True, False):
        for row in candidates:
            if row["patient_id"] in patients:
                continue
            if require_new_disease and row["class_name"] in diseases:
                continue
            selected.append({"case_id": f"A{len(selected) + 1:02d}", **row})
            patients.add(row["patient_id"])
            diseases.add(row["class_name"])
            if len(selected) == 5:
                return selected, pd.DataFrame(counts)
    return selected, pd.DataFrame(counts)


def write_case_review(cases, predictions, seeds, threshold, folder):
    notes = [
        "# Step 2A 病例页（限组内）", "",
        "在源域病种 AUC 相对线性探针提高、目标域同病种 AUC 降低的配置中，"
        "选择目标域至少 2 个种子出现线性探针判对、对应 LoRA 判错的病例。",
        "源域支持证据来自病种层面的独立测试队列。",
        f"主评测和病例判决均使用固定阈值 {threshold:.2f}。按病种覆盖优先选择不同患者。", "",
        "**原图复核状态：未复核。** 病例分析依据逐样本预测和数据清单，AP／PA 来自索引。", "",
    ]
    review_rows = []
    probability_rows = []
    if not cases:
        notes.append("本次数据中未找到满足预设筛选条件的病例。")
    for case in cases:
        sample = case["sample_id"]
        budget = case["budget"]
        disease = case["class_name"]
        code = case["case_id"]
        missing = "、".join(case["unknown_labels"]) or "无"
        unmasked = "、".join(f"{name}={value}" for name, value in case["valid_labels"].items())
        raw_labels = "、".join(f"{name}={value or '空白'}" for name, value in case["raw_labels"].items())
        image_status = "索引路径可访问" if Path(case["image_path"]).is_file() else "索引路径当前不可访问"
        notes += [
            f"## {code} · {disease} · {case['error']}", "",
            f"- 标注和实验配置为标签 {case['label']}、{case['view_position']} 位"
            f"（manifest 记录）、{budget}、{case['method']}。",
            f"- 该病种相对线性探针的 AUC 差值为源域 {case['source_auc_gain_vs_linear_probe']:+.4f}，"
            f"目标域 {case['target_auc_gain_vs_linear_probe']:+.4f}。"
            f"目标域在种子 {', '.join(map(str, case['discordant_seeds']))} 出现线性探针判对、LoRA 判错。",
            f"- 可用标签为 {unmasked}。未参与评测的标签为 {missing}。",
            f"- 原始标签记录为 {raw_labels}。",
            f"- 胸片状态为{image_status}。设备标记和可见病灶未做原图复核。", "",
            "| 方法 | 种子 | 阳性概率 | 阈值 | 判决 |", "|---|---:|---:|---:|---|",
        ]
        runs = [
            (method, budget, seed, "chexpert")
            for method, seed in product(("linear_probe", "lora_r8", "lora_r16"), seeds)
        ]
        runs += [(method, "100pct", 42, "chexpert") for method in ("pretrained", "pretrained_5prompts")]
        for key in runs:
            if key not in predictions:
                continue
            method, _, seed, _ = key
            probability = float(predictions[key].at[sample, f"prob_{disease}"])
            decision = int(probability >= threshold)
            grade = "正确" if decision == case["label"] else "错误"
            notes.append(f"| {method} | {seed} | {probability:.6f} | {threshold:.2f} | {decision}（{grade}） |")
            probability_rows.append({
                "case_id": code, "class_name": disease, "label": case["label"], "method": method,
                "case_budget": budget, "seed": seed, "probability": probability,
                "threshold": threshold, "decision": decision, "correct": decision == case["label"],
            })
        notes.append("")
        review_rows.append({
            "case_id": code, "sample_id": sample, "patient_id": case["patient_id"],
            "image_path": case["image_path"], "view_position": case["view_position"],
            "image_path_status": image_status, "class_name": disease, "label": case["label"],
            "budget": budget, "method": case["method"], "error": case["error"],
            "valid_labels": unmasked, "unknown_labels": missing, "raw_labels": raw_labels,
            "device_markers": "未做原图复核", "visible_findings": "未做原图复核",
        })
    (folder / "病例页_仅组内.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    pd.DataFrame(review_rows, columns=[
        "case_id", "sample_id", "patient_id", "image_path", "view_position", "image_path_status",
        "class_name", "label", "budget", "method", "error", "valid_labels", "unknown_labels",
        "raw_labels", "device_markers", "visible_findings",
    ]).to_csv(folder / "case_review_template.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(probability_rows, columns=[
        "case_id", "class_name", "label", "method", "case_budget", "seed", "probability",
        "threshold", "decision", "correct",
    ]).to_csv(folder / "case_probabilities.csv", index=False, encoding="utf-8-sig")
