# 跨机构胸片多标签分类：Step 0

Step 0 提供固定的数据划分、标签映射、prompt、官方 MedCLIP 初始化，以及共用的源域训练、两域评测和 H1—H3 指标采集代码，供后续步骤接入方法并开展正式实验。

**当前两域均已通过 Step 0 验收**：见 [源域报告](outputs/step0_mimic.json)、[目标域报告](outputs/step0_chexpert.json) 和 [最终交接报告](outputs/step0_handoff.json)（`step0_ready: true`）。新接手的同学请按 [Step 1 接手说明](docs/step1_handoff.md) 获取网盘资源、安装环境、调整路径。

## 已交付什么

| 交付项 | 文件与用途 |
| --- | --- |
| 数据挑选与固定划分 | `scripts/build_manifest.py`、`scripts/build_subsets.py`；`data/` 内两域索引、源域嵌套训练名单及类别统计 |
| 数据说明与实验假设 | [datasets.md](docs/datasets.md)、[hypotheses.md](docs/hypotheses.md) |
| 预训练模型、标签和 prompt | `scripts/prepare_model.py`、`src/model_loader.py`、`src/data.py`、`src/prompts.py`；实际模型文件由网盘提供 |
| 共用实验与指标采集代码 | `scripts/run_experiment.py`、`scripts/collect_metrics.py`；`src/` 内训练、评测、特征分析及 H1—H3 指标模块 |
| 验收证据 | `outputs/` 内两域 16 图概率、预测、特征缓存、指标及验收报告；源域另验证训练循环与 checkpoint 保存、重载 |

当前源域为 MIMIC-CXR-JPG 2.1.0 的 **p10 分片子队列**，5000 人/图，train/val/test 为 4000/500/500；种子 42、43、44 各有 400/4000 人的 10%/100% 嵌套训练名单。目标域为 CheXpert，1000 人/图，全部用于外部 test。两域均每人一张 AP/PA 正位图。

五类顺序固定为 `Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion`。保留原始标签，仅原始 0/1 参与监督，-1/空白通过 mask 排除；五类全无效的图像保留在索引中，训练和评测时跳过并计数。

## 目录与资源位置

```text
config/        共享配置模板；*.local.yaml 为个人配置
scripts/       数据准备、模型准备、验收、训练/评测及指标入口
src/           共用数据、模型、训练、评测和特征分析模块
data/          两域索引、源域嵌套名单、类别统计
outputs/       验收报告、预测、特征和指标的统一输出目录
docs/          数据说明、假设和 Step 1 接手说明
tests/         共用代码的回归检查
requirements.txt
```

**Git 提供代码、配置模板、说明、固定索引/名单及现有验收产物；网盘提供图像、原始 CSV 和完整 `step0_model/`。** 原始数据和模型继续放在项目外，不需要将完整 p10.zip、CheXpert 下载包或 `.venv/` 上传。网盘清单、源域约 9.45 GB 的资源构成及目标域需补齐的文件，见 [接手说明](docs/step1_handoff.md)。

## Step 0 与后续步骤的边界

Step 0 已验证每域 16 张有效图像的 5 维概率、`v/q/z` 特征缓存、重复推理、标签映射、患者互斥和模型冻结状态。**这些是接口验收，不能作为正式性能结果或 H1—H3 的结论。** `outputs/training_loop_check.pt` 是临时小头的训练检查文件，不是后续实验应加载的预训练权重或正式基线。

Step 1 按任务要求实现其可训练方法，通过配置接入共用训练和评测入口；本版本不包含正式线性探针、LoRA 方法或完整预算实验。默认 `experiment.method: pretrained` 为冻结推理模型，不可直接训练。

MedCLIP 预训练已使用过 MIMIC-CXR 和 CheXpert，结论限定于本次下游划分。具体限制见 [数据说明](docs/datasets.md) 和 [接手说明](docs/step1_handoff.md)。
