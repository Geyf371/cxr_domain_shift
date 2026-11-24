# Step 2A 源域与目标域分类评测

使用 Step 1 的正式逐患者预测，在 MIMIC test（源域）和 CheXpert test（目标域）比较原始预训练单模板、线性探针、LoRA-r8 和 LoRA-r16。本目录包含 A 部分的分类评测代码和汇总产物。

## 复现

在仓库根目录运行以下命令，把预测路径改成你本机的 Step 1 正式交接 CSV 目录。代码复用 `src/metrics.py`。已有项目依赖时可跳过安装。

```powershell
$predictions = '../step1_handoff/step1_csv'
python -m pip install -r ./step2/member_a/requirements.txt
python ./step2/member_a/evaluate.py --repo . --predictions $predictions
```

脚本递归读取正式测试的 `predictions_*.csv`。可用 `--output` 和 `--private-output` 指定输出目录。

主评测使用两域固定测试名单、五类顺序和标签 mask，F1 使用 0.5 固定阈值。线性探针及 LoRA 以 10% 和 100% 的源域训练标签预算分别报告种子 42、43、44 的均值、样本标准差和有效种子数。单模板使用 `frozen_prompts` 中与正式汇总一致的同一基线，重复导出不会增加种子数。CSV 中 `NA` 表示缺少定义条件或不足以计算标准差。AUPRC 使用 Average Precision 口径。

## 产物

| 位置 | 内容 |
|---|---|
| [分工说明](../分工说明.md) | A、B 的职责、各自输入及最终交接 |
| [结果说明](结果说明.md) | 分类总表、迁移结果、病例分析及阅片边界 |
| `results/macro_by_seed.csv`、`results/macro_summary.csv` | 两域 ROC-AUC、F1、AUPRC 的逐种子和汇总结果，含有效类别数 |
| `results/class_by_seed.csv`、`results/class_summary.csv` | 五病种逐种子与聚合指标、有效标签数和阳性率 |
| `results/transfer_by_seed.csv`、`results/transfer_summary.csv` | 源域减目标域的有符号差值、两域相同病种的 AUC 对照，以及相对单模板、线性探针的两域收益 |
| `results/class_prevalence.csv`、`results/case_candidate_counts.csv` | 每域逐病种标签分布和病例筛选计数 |
| `results/figures/`、`results/run_metadata.json` | 预算曲线、迁移下降图、逐病种图，以及复现所需的参数、文件哈希和版本 |
| `private/` | 病例页、逐病例概率与阅片模板。模板包含原始图像定位和患者伪标识 |

`private/` 已列入 Git 忽略规则。逐患者预测保留在本地 Step 1 交接目录，组内有权限的成员可以据此复核。此次病例分析使用预测与数据清单，设备标记和可见病灶均标为未做原图复核。若后续补充阅片观察，可另存为 `private/case_review_notes.csv`，以便保留人工记录。
