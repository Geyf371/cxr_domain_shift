# 数据准备及交接说明

本项目以 MIMIC-CXR 和 Stanford CheXpert 为两个机构来源，不额外编造 `hospital_id`。原始数据放在项目外，各域负责人在有访问资格的环境保留数据、患者级索引与本机配置。

## 源域：MIMIC-CXR-JPG 2.1.0

访问需满足 PhysioNet 的资质认证、培训与数据使用协议。使用 JPG、metadata 和 CheXpert 标签 CSV，不需要报告文本。接受以下文件名，`.csv.gz` 直接读取，无需先解压：

- `mimic-cxr-2.0.0-metadata.csv.gz`，或 `mimic-cxr-metadata.csv.gz`（也支持对应 `.csv`）；
- `mimic-cxr-2.0.0-chexpert.csv.gz`（也支持 `.csv`）。

`data.mimic_root` 指向这两个表和图像树所在根目录。支持 `files/p10/p########/s########/<dicom_id>.jpg`，以及根目录直接含 `p10/...` 或 `file/p10/...` 的结构。ZIP 内 `p10/...` 的结构可由 `extract_selected_mimic.py` 按选中图像提取至 `<mimic_root>/files/p10/...`。

先保留 `ViewPosition=AP/PA`。对每名患者按数值检查 ID、图像 ID 排序，取一次符合条件的检查和一张图像，再以 selection seed 2026 抽 5000 名患者；以 split seed 42 按患者划分 4000 train、500 val、500 test。患者 ID 添加 `mimic:` 前缀。

源域 train 再以种子 42、43、44 各构造一次 10%/100% 嵌套患者子集，分别为 400/4000 人；同一种子内小预算是大预算子集，所有方法共享这些名单。预算按患者数定义，不因某人有效标签较少而重新抽样。

### 只有 p10 分片时

选择和重建 manifest 均指定 `--mimic-buckets p10`，以免抽到未下载分片的患者。提取前 `--allow-missing-images` 只生成 `mimic_selection.csv`；提取后不带该标记重建正式索引，确认 5000 张选中图像全部存在。

p10 是患者 ID 分片限定子队列，不能代表全 MIMIC 的随机样本；这一限制由组长确定是否接受，并在下游结论中保留。如需全队列抽样，应提供对应图像分片，重新生成源域索引、子集和报告。验收通过只表示当前子队列的接口与配置检查通过，不消除这一抽样限制。

## 目标域：CheXpert

从 Stanford AIMI 官方页面进入 AIMI Redivis，完成注册并接受条款。优先使用 small 图像和公开 `train.csv`。`data.chexpert_root` 指向 CSV 与已解压图像的共同根目录，例如：

```text
CheXpert-v1.0-small/
  train.csv
  train/
    patient00001/
      study1/
        view1_frontal.jpg
```

CSV 必须含 `Path`、`Frontal/Lateral`、`AP/PA` 和五类标签列。路径带 `CheXpert-v1.0-small/` 前缀时自动解析；也支持根目录配置为外面一层。

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

两个域的构建、模型验收独立执行；目标域同学无需运行 `build_subsets.py`，因为少样本预算只针对源域训练患者。详细命令见项目 README。

两人各保留自己的患者级索引和数据；Git 提交脚本、共享配置、文档与指定聚合报告。向 Step 1 交接时同时说明原始数据、索引、模型缓存的保存位置及访问方式；需要直接传递患者清单或特征时，在许可的数据环境进行。Step 1 在新机器上可按相同数据与配置重建索引，并核对报告中的不含路径的 selection SHA256；若图像子队列不同，不能假定会重建出同一清单。

如获准将两个 manifest 集中到一台机器，可用 `merge_manifests.py` 生成 `all_manifest.csv` 和合并频数，但不是 Step 0 独立域交接的必需命令；合并不会复制图像或修复另一台机器的绝对路径。

## 官方模型和预处理

`prepare_model.py` 固定官方 MedCLIP 代码提交、公开 ViT 医疗预训练权重（实际视觉骨干为 Swin Tiny）以及 Bio_ClinicalBERT 词表与配置，记录资产 SHA256。所有模型文件放项目外。

`load_medclip` 严格恢复完整权重，使用官方 `MedCLIPProcessor` 的图像处理；文本用同一官方 BERT 词表的 slow tokenizer，统一 padding/truncation 至 77。通过官方底层图像/文本模块支持 CPU/CUDA，规避上游封装写死 CUDA 的问题。不会重新训练预训练模型，也不将 ImageNet 初始化当作医疗预训练权重。

官方代码：https://github.com/RyanWangZf/MedCLIP ，提交 `9c3396f20d5d54e4fae241b8cb06ca45848e98c9`。预训练使用过 MIMIC-CXR 与 CheXpert，结论限于本次下游划分。具体版本、权重来源和软件环境写在每域验收报告中。
