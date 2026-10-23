# 数据准备及交接说明

本项目以 MIMIC-CXR 和 Stanford CheXpert 为两个机构来源。原始数据和模型缓存放在项目外；本次交接通过 Git 提供固定患者级索引、嵌套名单和验收产物，通过网盘提供对应图像、原始 CSV 与模型。本机配置不提交。具体资源清单和接手命令见 [Step 1 接手说明](step1_handoff.md)。

## 源域：MIMIC-CXR-JPG 2.1.0

使用 JPG、metadata 和 CheXpert 标签 CSV，不需要报告文本。接受以下文件名，`.csv.gz` 直接读取，无需先解压：

- `mimic-cxr-2.0.0-metadata.csv.gz`，或 `mimic-cxr-metadata.csv.gz`（也支持对应 `.csv`）；
- `mimic-cxr-2.0.0-chexpert.csv.gz`（也支持 `.csv`）。

`data.mimic_root` 指向这两个表和图像树所在根目录。

先保留 `ViewPosition=AP/PA`。对每名患者按数值检查 ID、图像 ID 排序，取一次符合条件的检查和一张图像，再以 selection seed 2026 抽 5000 名患者；以 split seed 42 按患者划分 4000 train、500 val、500 test。患者 ID 添加 `mimic:` 前缀。

源域 train 再以种子 42、43、44 各构造一次 10%/100% 嵌套患者子集，分别为 400/4000 人；同一种子内小预算是大预算子集，所有方法共享这些名单。预算按患者数定义，不因某人有效标签较少而重新抽样。

## 目标域：CheXpert

使用 small 图像和公开 `train.csv`。

只保留 `Frontal/Lateral=Frontal` 且 `AP/PA` 列为 `AP` 或 `PA` 的记录；从 Path 的 patient 目录解析患者、study 目录解析检查，按检查 ID、路径排序取每人一图。以 selection seed 2026 抽 1000 名不同患者，全部作为外部 test。患者 ID 添加 `chexpert:` 前缀。本项目不使用该数据集进行训练、验证、提示选择或阈值选择。

## 标签和无效样本

固定五类顺序：Atelectasis、Cardiomegaly、Consolidation、Edema、Pleural Effusion。支持常见同义列，如 `Effusion`→`Pleural Effusion`、`Pulmonary edema`→`Edema`；匹配时忽略大小写、空格和下划线。缺少必需标签列会报错，不会静默当作全未知。

每类均保存三列：

- `raw_<class>`：原始数值，包含 `-1` 和空白；
- `label_<class>`：训练/评测的二值标签；原始无效项临时填 0；
- `mask_<class>`：仅原始 `0/1` 为 1，其余为 0；填 0 的无效标签不当作阴性。

五类 mask 全为 0 的样本在 manifest 中保留，以保证患者划分与抽样可复现；训练、评测中跳过并计数。可用图像数不等于有监督有效图像数。每域验收报告记录选中/存在图像数、至少一类有效标签的图像数及全无效数量，类别统计记录各划分阳性、阴性、未知和全无效样本数。数据缺标严重时，某类 test 可能没有正例或负例，AUC 应为 NA，不能补成 0 或 1。

## 文件生成与两人交接

| 文件 | 内容和生成方式 |
| --- | --- |
| `data/<domain>_manifest.csv` | `build_manifest.py`：来源、患者、检查、图像、绝对路径、AP/PA、划分和三组标签列 |
| `data/class_frequencies_<domain>.csv` | 同一命令：各划分、类别的阳性/阴性/未知/全无效数量 |
| `data/source_train_seed*_10pct.csv`、`*_100pct.csv` | 源域 `build_subsets.py`：三个种子的嵌套 train 患者清单 |
| `outputs/step0_<domain>.json` | `run_step0.py`：数据审计、16 图真实概率和特征检查、模型/环境/代码版本，无患者标识和本机路径 |
| `outputs/class_frequencies_<domain>.csv` | `run_step0.py`：供 Git 交接的聚合类别频数副本 |
| `outputs/step0_handoff.json` | `check_handoff.py`：两域代码、模型资产、抽样协议一致，且两域已通过验收 |


Git 提交脚本、共享配置、文档、两域索引、源域嵌套名单、类别频数以及现有验收预测/特征/报告。网盘只需提供名单对应的全部图像、原始 CSV 和一份完整模型缓存；不必交接完整图像下载包或 Python 虚拟环境。

现有 manifest 含负责人的绝对路径。Step 1 在新机器上须同时修改个人配置与 manifest 的 `image_path`，保持患者、划分及标签不变；仅改配置不会修复索引。

## 官方模型和预处理

`prepare_model.py` 固定官方 MedCLIP 代码提交、公开 ViT 医疗预训练权重（实际视觉骨干为 Swin Tiny）以及 Bio_ClinicalBERT 词表与配置，记录资产 SHA256。所有模型文件放项目外。

`load_medclip` 严格恢复完整权重，使用官方 `MedCLIPProcessor` 的图像处理；文本用同一官方 BERT 词表的 slow tokenizer，统一 padding/truncation 至 77。通过官方底层图像/文本模块支持 CPU/CUDA，规避上游封装写死 CUDA 的问题。不会重新训练预训练模型，也不将 ImageNet 初始化当作医疗预训练权重。
