# Step 1 接手说明

接手需要两部分：**最新 Git 仓库 + 网盘中的图像和模型资源**。Git 内固定名单决定实验使用哪些患者；网盘提供这些名单实际引用的图像。不要重新随机抽样来代替现有名单。

## 1. 交接状态与 Git 文件

两域已完成数据与真实 MedCLIP 的 16 图验收，`outputs/step0_handoff.json` 中 `step0_ready` 为 `true`。源域负责人还需将本机现有源域文件与本说明一起提交；接手时应拉取这次最终提交，确认以下内容齐全：

| 文件 | 接手用途 |
| --- | --- |
| `data/mimic_manifest.csv`、`data/chexpert_manifest.csv` | 两域固定图像、患者、划分、原始标签、二值标签和 mask |
| `data/source_train_seed{42,43,44}_{10,100}pct.csv`，共 6 份 | 源域固定嵌套训练名单；分别为 400/4000 人 |
| `data/class_frequencies_*.csv`、`outputs/class_frequencies_*.csv` | 类别阳性、阴性、未知及全无效样本统计 |
| `docs/hypotheses.md`、`docs/datasets.md` | H1—H3 比较与指标约定、数据来源与抽样限制 |
| `config/step0.yaml`、`requirements.txt`、`src/`、`scripts/` | 环境、协议和后续共用代码 |
| `outputs/step0_mimic.json`、`step0_chexpert.json`、`step0_handoff.json` | 两域验收及模型、环境、代码版本证据 |
| `outputs/` 内 `step0_smoke` 相关预测、特征、指标及检查 JSON | 16 图接口示例，帮助核对输出格式；不作为正式实验结果 |

`data/mimic_selection.csv` 及对应 selection 频数是解压前挑选记录；正式实验读取 `mimic_manifest.csv`。`outputs/training_loop_check.pt` 仅检查公共训练循环和保存/重载，不能用作正式基线 checkpoint。正式预训练模型位于网盘的 `step0_model/`。

## 2. 网盘内容：由 Step 0 两位负责人准备

以下为目录结构：

```text
Step0交接资源/
  mimic/
    files/p10/p########/s########/<image_id>.jpg
    mimic-cxr-2.0.0-metadata.csv.gz
    mimic-cxr-2.0.0-chexpert.csv.gz
  chexpert/
    train.csv
    train/patient#####/study#/view#_frontal.jpg
  step0_model/
    pytorch_model.bin
    model_manifest.json
    official/
    bert/
    swin/
    python/
    hf_cache/
```

## 3. 接手后的首次准备

先获取包含源域文件的最终 Git 提交，再下载并解压网盘资源，保持上述内部目录。以下命令均在**项目根目录**、同一个 Python 环境执行。推荐干净的 Python 3.10—3.12 环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config/step0.yaml config/step0.local.yaml
```

已有满足依赖的环境可复用；个人配置已存在时跳过 `Copy-Item`。不要另装任意版本的 `transformers/tokenizers`。网盘模型缓存不等于完整 Python 环境，仍需安装上述依赖。

在完整的 `config/step0.local.yaml` 内只修改以下三个路径字段，其余模板设置保留。假设资源放在 `F:/Step0交接资源`：

```yaml
data:
  mimic_root: "F:/Step0交接资源/mimic"
  chexpert_root: "F:/Step0交接资源/chexpert"
model:
  cache_dir: "F:/Step0交接资源/step0_model"
```

这是需要修改的字段示意，**不要用这段替换整个配置文件**。

### 更新索引中的本机路径

现有索引的源域路径以 `E:\data` 开头，目标域以 `D:\AI_Projects\CheXpert_Small` 开头。**仅修改 YAML 不会更新 manifest 的 `image_path`。** 下载到其他位置后，在项目根目录 PowerShell 完整复制执行下面一段：它只替换两份正式 manifest 的路径，检查 5000/1000 张图全部存在后才写回，保留患者、标签、划分和六份嵌套名单。

```powershell
@'
import csv
from pathlib import Path, PureWindowsPath
import yaml

config = yaml.safe_load(Path("config/step0.local.yaml").read_text(encoding="utf-8"))
data_dir = Path(config["data"].get("data_dir", "data"))
pending = []
for domain, old_root, count in (
    ("mimic", "E:/data", 5000),
    ("chexpert", "D:/AI_Projects/CheXpert_Small", 1000),
):
    new_root = Path(config["data"][f"{domain}_root"]).resolve()
    manifest = data_dir / f"{domain}_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    assert len(rows) == count, f"{domain}: expected {count} rows"
    for row in rows:
        current = Path(row["image_path"])
        if current.is_absolute() and current.is_relative_to(new_root):
            updated = current
        else:
            relative = PureWindowsPath(row["image_path"]).relative_to(PureWindowsPath(old_root))
            updated = new_root.joinpath(*relative.parts)
        if not updated.is_file():
            raise FileNotFoundError(updated)
        row["image_path"] = str(updated)
    pending.append((manifest, fields, rows))
for manifest, fields, rows in pending:
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"OK: {manifest}, {len(rows)} images found")
'@ | python -
```

这段只更新本机路径；不要将自己的路径改动再次提交覆盖共享索引。若缺图，先补齐网盘资源。不要修改患者名单或删除缺图行。重建原始数据时源域须指定 `--mimic-buckets p10`；正常接手无需重新运行挑选、解压和嵌套抽样。

### 准备兼容依赖并核对交接

```powershell
python scripts/prepare_model.py --config config/step0.local.yaml --install-deps
python scripts/check_handoff.py --config config/step0.local.yaml
```

第一条复用完整模型缓存，已有权重不重复下载，为当前环境准备兼容依赖。第二条核对已有两域报告、代码、模型资产记录和抽样协议，预期输出 `PASS: Step 0 handoff ready`；**它不检查本机图像或实际加载模型**，图像存在性已由上面的路径更新代码检查。

如需确认新机器实际推理可用，可在开始修改实验代码之前分别运行 `python scripts/run_step0.py --config config/step0.local.yaml --domain mimic` 和同命令的 `--domain chexpert`，在本机重做 16 图检查。现有报告证明 Step 0 交付版本已通过，后续修改方法代码后应保留原版报告作为交接记录，不要手改报告中的指纹。

## 4. Step 1 从哪里接入

Step 0 已提供公共训练循环和多标签评测/指标代码；Step 1 按课程分工实现其方法，不需要复制另一套数据与评测逻辑。

- 方法入口：`src/model_adapter.py` 的 `build_model`。后续方法配置 `experiment.method` 与 `experiment.factory: "包名.模块名:函数名"`，factory 按 `fn(bundle=bundle, config=config)` 调用。
- 返回对象：`torch.nn.Module`，提供 `forward_batch(batch)`，返回 `src/model_interface.py` 的 `ModelBatchOutput`；为评测与特征分析输出 `logits`、`image_features_v`、`prompt_scores_q`、`joint_features_z`。可训练参数及设备移动由方法实现明确处理。
- 公共入口：`scripts/run_experiment.py`；训练只使用源域 train，早停/模型选择只使用源域 val；CheXpert 只允许 test。默认 `pretrained` 为冻结模型，直接执行训练会因无可训练参数而报错。
- 配置切换：`experiment.seed` 用 42/43/44，`budget` 用 `10pct`/`100pct`；`prompt_templates` 用 1/5，五模板先平均 logits，再 sigmoid。固定五类顺序和 mask 规则继续复用。

接好环境后，可用默认冻结模型执行全量 test 评测；下面是后续正式实验入口示例，Step 0 交接不要求现在运行：

```powershell
python scripts/run_experiment.py --config config/step0.local.yaml --action eval --domain mimic --split test
python scripts/run_experiment.py --config config/step0.local.yaml --action eval --domain chexpert --split test

# 仅在实现并配置可训练方法 factory 之后执行
python scripts/run_experiment.py --config config/step0.local.yaml --action train --domain mimic
```

训练后评测须保留相同方法配置，并在两域 eval 命令中追加 `--checkpoint` 和训练实际生成的 checkpoint 路径。`--limit` 只用于诊断，不作正式全量结果。建议将个人配置的 `data.output_dir` 设为新的实验目录（如 `outputs/step1`），与原始验收文件区分。

统一评测会输出逐样本预测、每类/汇总指标、元信息，以及 `v/q/z` 和文本特征缓存。正式结果齐备后按 [hypotheses.md](hypotheses.md) 采集指标，例如输出仍在 `outputs/` 时：

```powershell
python scripts/collect_metrics.py h1 --summaries outputs --output outputs/h1_transfer.csv
python scripts/collect_metrics.py h2 --base BASE_FEATURES.npz --method METHOD_FEATURES.npz --output outputs/h2_drift.json
python scripts/collect_metrics.py h3 --single SINGLE_PREDICTIONS.csv --five FIVE_PREDICTIONS.csv --output outputs/h3_prompts.json
```

大写路径需替换为实际文件；若使用 `outputs/step1`，相应修改输入/输出路径。H1 需要相同方法/预算/种子的两域 test 结果及预训练对照；H2 比较相同图像的特征漂移；H3 比较相同图像/标签的单模板与五模板预测。既有 `step0_smoke`/`check16` 结果只展示接口，不能替代这些正式结果。

