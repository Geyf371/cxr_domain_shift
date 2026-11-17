# Step1 A/B/C 整合与交接说明

更新日期：2026-11-17

整合分支：`step1-member-c`

任务范围：仅 Step1，未开始 Step2—Step4 实验。

## 1. 当前结论

- A 的 `feat/step1-clean`、B 的 `step1-member-b`和 C 的公共接口/多提示已合入同一分支。
- 五种方法均有代码与汇总结果：`pretrained`、`pretrained_5prompts`、`linear_probe`、`lora_r8`、`lora_r16`。
- B 的正式结果覆盖 `2 ranks × 2 budgets × 3 seeds × 2 test domains = 24` 条评测记录，资源表覆盖 12 次训练运行。
- C 已在完整 MIMIC test（500 张，206 张标签有效）和 CheXpert test（1000 张，762 张标签有效）上重跑单模板/五模板，预测、`v/q/z`、指标和元数据均通过自动校验。
- 全量测试为 `33 passed`。

## 2. 正式测试集结果摘要

表中可训练方法为 3 seeds 的均值±标准差；冻结方法为 seed 42 的确定性推理结果。

| 方法 | 预算 | MIMIC AUC | MIMIC F1 | CheXpert AUC | CheXpert F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| pretrained | 100pct标签 | 0.4862 | 0.3819 | 0.3352 | 0.2892 |
| pretrained_5prompts | 100pct标签 | 0.4057 | 0.3686 | 0.3979 | 0.3866 |
| linear_probe | 10pct | 0.8737±0.0057 | 0.4569±0.0914 | 0.8638±0.0368 | 0.4381±0.0796 |
| linear_probe | 100pct | 0.8837±0.0016 | 0.5156±0.0072 | 0.8748±0.0261 | 0.5124±0.0046 |
| lora_r8 | 10pct | 0.8842±0.0044 | 0.4946±0.0233 | 0.8541±0.0414 | 0.4932±0.0149 |
| lora_r8 | 100pct | 0.8909±0.0066 | 0.5174±0.0014 | 0.8616±0.0320 | 0.5109±0.0016 |
| lora_r16 | 10pct | 0.8853±0.0037 | 0.4848±0.0071 | 0.8401±0.0587 | 0.4727±0.0128 |
| lora_r16 | 100pct | 0.8914±0.0077 | 0.5193±0.0010 | 0.8674±0.0295 | 0.5126±0.0008 |

完整 AUC/F1/AUPRC、val/test 和来源精度见 [`results/step1_metrics_summary.csv`](results/step1_metrics_summary.csv)。

## 3. 成员 C 多提示结论

`clinical_v1` 每类包含 5 组阳性/阴性提示；第 1 组严格保留 Step0/A 的单模板，保证受控对比。五模板采用“先平均 logits，再 sigmoid”。

| 域 | ΔAUC（五-单） | ΔF1（五-单） | ΔAUPRC（五-单） |
| --- | ---: | ---: | ---: |
| MIMIC test | -0.0805 | -0.0133 | -0.0478 |
| CheXpert test | +0.0627 | +0.0975 | -0.0037 |

结果表明五提示在外部 CheXpert 上提高 AUC/F1，但没有在所有域和所有指标上一致受益，因此不应表述为“五提示全面优于单提示”。两种提示的图像特征 `v` 最大绝对差为 0，符合冻结视觉编码器预期。

## 4. 资源统计口径

- 训练时间、峰值显存和可训练参数：[`results/step1_training_resources.csv`](results/step1_training_resources.csv)。
- 推理时间与显存：[`results/step1_inference_resources.csv`](results/step1_inference_resources.csv)。
- A 使用 CUDA/batch64，B 使用 CUDA/batch16/FP32/3 次重复，C 本次正式补跑使用 CPU/batch4。这些记录可用于交付和核查，但不能把跨硬件/跨 batch 的时间、显存直接当成公平排名。
- 冻结方法训练时间为 0、可训练参数为 0；CPU 无法提供 CUDA 峰值显存，依规记录为空，未估算填充。

## 5. 交接文件

| 内容 | 路径 |
| --- | --- |
| 自动检查清单 | [`results/step1_delivery_manifest.json`](results/step1_delivery_manifest.json) |
| B 的 24 条测试指标 | [`results/member_b/test_metrics.csv`](results/member_b/test_metrics.csv) |
| B 的均值±标准差 | [`results/member_b/test_metrics_mean_std.csv`](results/member_b/test_metrics_mean_std.csv) |
| B 的 12 次资源记录 | [`results/member_b/run_resources.csv`](results/member_b/run_resources.csv) |
| C 的脱敏正式报告 | [`results/member_c/member_c_formal_summary.json`](results/member_c/member_c_formal_summary.json) |
| 重建汇总表的脚本 | [`../scripts/build_step1_handoff.py`](../scripts/build_step1_handoff.py) |
| C 正式全集推理脚本 | [`../scripts/run_member_c_formal.py`](../scripts/run_member_c_formal.py) |

## 6. 交给 Step2 前的唯一待收项

代码、聚合指标和资源表已完成；但当前收到的 A/B 分支与 B 私发目录中，没有 A/B 的原始 `predictions_*.csv`、`features_*.npz`（`v/q/z`）和正式 checkpoint。B 的 `manifest.json` 仅引用了外部 6.18 GiB 基础交付包，不包含该包本身。

因此交付状态为：

- **Step1 集成代码 + 指标/资源汇总：已完成。**
- **Step2 可直接消费的 A/B 原始大文件交接：待 A/B 补充。**

建议向 A/B 各收一个不进 Git 的压缩包：

- A：6 个 Linear Probe checkpoint，以及正式 `predictions/features/metadata/metrics/training`。
- B：12 个 LoRA checkpoint，以及 24 组两域 `predictions/features/metadata/metrics/training`。
- 两包都应携带 SHA-256 清单，并保留 `sample_id`、五类顺序与现有数据划分。

C 的全集原始产物已在本机 `outputs/step1/member_c_formal/`，不提交 Git，可在正式网盘交接时与 A/B 产物一起打包。
