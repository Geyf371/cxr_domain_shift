# 跨机构胸片多标签分类：Step 0

本项目交付 Step 0 的数据准备、固定标签和 prompt、官方 MedCLIP 初始化，以及后续共用的源域训练、两域评测和 H1—H3 指标采集代码。两人可各自在本机准备一个数据域，通过 Git 中的验收报告交接，无需先把两个原始数据集集中到同一台电脑。

`run_step0.py` 真实加载模型，对 16 张图像检查概率、特征、冻结参数及重复推理；源域还验证公共训练循环和 checkpoint 保存、重载。它不会执行 Step 1 的正式线性探针、LoRA、128 人预实验或完整预算实验。16 张图的指标只用于接口验收，不能作为假设成立的实验结论。

## 简洁目录

```text
config/       共享配置模板；个人 *.local.yaml 不提交
docs/         数据说明、hypotheses.md
scripts/      数据准备、模型准备、验收和公共实验命令
src/          数据、模型、训练、评测、指标和特征分析模块
tests/        标签 mask、样本对齐、指标及目标域接入的回归检查
data/         本机索引和源域嵌套子集，Git 忽略
outputs/      统一输出目录；只提交指定的聚合统计和验收报告
```

原始图像、下载包和模型缓存放在项目外。已有的 `smoke_test.py` 仍可单独检查图像读取；新的 `run_step0.py` 包含数据审计和真实模型检查，完成交接时不必再单独运行旧命令。

## 当前源域负责人怎么交接

当前本机已生成 MIMIC 的 5000 患者索引、4000/500/500 划分、六份嵌套子集，并完成真实模型验收。最终结果见 `outputs/step0_mimic.json` 和 `outputs/class_frequencies_mimic.csv`。沿用已有数据、个人配置和外部模型缓存即可，不必重复解压。

提交列表应包含源域验收报告与类别统计；个人配置、`data/` 中的患者级文件、原始数据、权重、预测和特征不会加入。此时尚未完成目标域验收，不能宣称两域 Step 0 已全部通过。

## 目标域同学：一次环境准备，再运行四条命令

先拉取源域负责人提交的代码和报告。推荐使用干净的 Python 3.10—3.12 虚拟环境，避免已有新版 `transformers/tokenizers` 与官方旧版处理器冲突。若已建好满足依赖的干净环境，可直接使用。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item config/step0.yaml config/step0.local.yaml
```

`Copy-Item` 仅在个人配置不存在时执行。已有 `step0.local.yaml` 就修改它，避免覆盖自己的设置。在个人配置中只改本机路径，例如：

```yaml
data:
  mimic_root: null
  chexpert_root: "E:/CheXpert-v1.0-small"
  # 其余 data 字段保留模板原值
model:
  cache_dir: "E:/step0_model"
  # 其余 model 字段保留模板原值
```

以上是修改字段示意，不要用它替换整个配置文件。`chexpert_root` 应指向包含 `train.csv` 和 `train/patient.../` 图像目录的文件夹；也支持它们共同的上一层目录。CSV 中带 `CheXpert-v1.0-small/` 前缀的路径会自动解析。原始 ZIP 需先解压，索引脚本不会在 ZIP 内读取 JPG。模型缓存所在盘预留约 2 GB，下载结束会删除模型 ZIP；图像所需空间另计。

在项目根目录、同一个 Python 环境依次运行：

```powershell
python scripts/prepare_model.py --config config/step0.local.yaml --install-deps
python scripts/build_manifest.py --config config/step0.local.yaml --domain chexpert
python scripts/run_step0.py --config config/step0.local.yaml --domain chexpert
python scripts/check_handoff.py --config config/step0.local.yaml
```

这四条命令分别执行：

1. 下载固定版本的官方代码、官方 ViT/Swin 医疗预训练权重、文本词表与配置；将兼容依赖安装到项目外的模型缓存，不必另行安装 MedCLIP。
2. 从 `train.csv` 确定性选出 1000 名患者，各取一张 AP/PA 正位图，统一为目标域 test，生成五类标签和 mask。
3. 审计全部选中图像路径与标签，真实跑通 16 张有效图像的单模板、重复推理、五模板及 `v/q/z` 缓存，输出目标域报告和类别统计。
4. 读取 Git 带来的源域报告与本机目标域报告，检查代码、模型资产和抽样协议一致，生成 `outputs/step0_handoff.json`。该命令不读取源域图像或源域患者索引。

最后一条输出 `PASS: Step 0 handoff ready`，且 `step0_handoff.json` 中 `step0_ready=true` 后，目标域同学提交：

```powershell
git add .
git status --short
git commit -m "Add target acceptance and complete Step 0 handoff"
git push
```

新增的可提交文件是 `outputs/step0_chexpert.json`、`outputs/class_frequencies_chexpert.csv`、`outputs/step0_handoff.json`。至此完成 Step 0 版本，后续正式实验交给 Step 1 负责人。若报告代码版本不一致，应拉取一致版本后重跑受影响域的 `run_step0.py`，不要手工修改报告。

## 各类文件是什么、怎么得到

| 产物 | 生成方式与用途 | 是否进入 Git |
| --- | --- | --- |
| `data/mimic_manifest.csv`、`data/chexpert_manifest.csv` | `build_manifest.py` 生成；来源、患者、图像路径、划分、原始标签、二值标签和 mask | 本地保留；在获准的数据环境交接 |
| `data/source_train_seed*_10pct.csv`、`*_100pct.csv` | 源域运行 `build_subsets.py`；三个种子的嵌套患者清单 | 本地保留 |
| `data/class_frequencies_*.csv` | 构建索引时生成；各类别阳性、阴性、未知及全无效样本数 | 本地保留；验收后导出聚合版本至 outputs |
| `docs/datasets.md`、`docs/hypotheses.md` | 数据与抽样限制说明；预先固定 H1—H3 比较组和测量方式 | 提交 |
| `config/step0.yaml`、`requirements.txt` | 共享协议与环境依赖；真实软件版本另记入各域验收报告 | 提交 |
| `src/prompts.py` | 固定每类阳性/阴性提示，以及五模板列表 | 提交 |
| 项目外 `step0_model/` | `prepare_model.py` 生成；权重、官方源码、兼容依赖及校验清单 | 各自准备，不提交大文件 |
| `outputs/step0_mimic.json`、`step0_chexpert.json` | `run_step0.py` 生成；不含患者标识或本机路径的验收证据及环境/权重版本 | 提交 |
| `outputs/class_frequencies_mimic.csv`、`class_frequencies_chexpert.csv` | `run_step0.py` 导出的聚合类别统计 | 提交 |
| `outputs/step0_handoff.json` | `check_handoff.py` 检查两域报告后生成 | 最终通过后提交 |
| `outputs/predictions_*.csv`、`features_*.npz` | 统一评测生成；逐样本概率、logits、`v/q/z`、提示向量、标签和 mask | 含患者级信息，本地保留 |
| `outputs/metrics_*.csv`、`metadata_*.json` 等 | 统一评测的汇总指标、有效类别数和配置记录 | 默认本地保留 |

索引是 Step 0 交付物，但因含患者标识和本机绝对路径，默认不放普通 Git 仓库。两人分别保留自己的索引、原始数据和个人配置，并向 Step 1 说明保存位置；如需直接共享索引，在许可的数据环境传递并调整路径。Step 1 要重新跑源域训练或目标域评测，仍须能访问相应原始图像。只有代码和聚合报告不能替代数据集。

## 源域从原件重新构建（已有产物可跳过）

MIMIC 根目录包含两个 `.csv.gz` 文件和 `p10.zip`，个人配置设好 `mimic_root` 后：

```powershell
python scripts/build_manifest.py --config config/step0.local.yaml --domain mimic --mimic-buckets p10 --allow-missing-images
python scripts/extract_selected_mimic.py --config config/step0.local.yaml --archive-dir "E:/data"
python scripts/build_manifest.py --config config/step0.local.yaml --domain mimic --mimic-buckets p10
python scripts/build_subsets.py --config config/step0.local.yaml
python scripts/prepare_model.py --config config/step0.local.yaml --install-deps
python scripts/run_step0.py --config config/step0.local.yaml --domain mimic
```

两个 CSV 压缩文件直接读取，无需解压。第一条生成仅用于提取的 `mimic_selection.csv`；第二条只提取选中的 JPG 至外部数据目录的 `files/p10/...`，第三条才写正式 manifest。已解压完整数据时可省略前两条；有全量分片时不使用 `--mimic-buckets p10`。

**当前源域仅来自 p10 患者分片。** 它满足人数与患者划分检查，但属于分片限定子队列，不能视为全 MIMIC 的随机样本。是否接受这个抽样范围需由组长确定；如要求从全队列抽样，需要获得相关分片并重新生成源域产物。

## 后续共用接口（供 Step 1 接入）

后续新增方法通过配置 `experiment.method` 和 `experiment.factory: "包名.模块名:函数名"` 接入。factory 接收 `bundle, config`，返回有 `forward_batch(batch)` 的 `torch.nn.Module`；输出定义见 `src/model_interface.py`。需要导出 `logits`、`image_features_v`、`prompt_scores_q`、`joint_features_z`，并由方法实现明确设置可训练参数。图像 tensor 的设备移动由模型适配器处理。

```powershell
# 冻结预训练模型的完整指定域评测；Step 0 交接无需额外跑全量
python scripts/run_experiment.py --config config/step0.local.yaml --action eval --domain mimic --split test
python scripts/run_experiment.py --config config/step0.local.yaml --action eval --domain chexpert --split test

# 接入可训练方法 factory 后，复用公共源域训练/验证及 checkpoint
python scripts/run_experiment.py --config config/step0.local.yaml --action train --domain mimic

# 后续结果生成后采集假设指标；大写路径替换成实际输出
python scripts/collect_metrics.py h1 --summaries outputs --output outputs/h1_transfer.csv
python scripts/collect_metrics.py h2 --base BASE_FEATURES.npz --method METHOD_FEATURES.npz --output outputs/h2_drift.json
python scripts/collect_metrics.py h3 --single SINGLE_PREDICTIONS.csv --five FIVE_PREDICTIONS.csv --output outputs/h3_prompts.json
```

训练只读取源域 train 和 val；目标域仅可执行 test 评测。冻结预训练模型本身不含可训练参数，直接以该配置调用 `--action train` 会明确报错。方法实现由 Step 1 提供，评测和指标代码共用。单/五模板通过 `experiment.prompt_templates: 1` 或 `5` 切换。

H1 需要同一方法/预算/种子的两域 test 汇总及对应预训练对照；分机器运行时，在获准环境汇总两边 CSV。H2、H3 自动对齐同域同图像，拒绝不同样本或标签协议混用。正式实验规则见 `docs/hypotheses.md`。

开发回归检查可用 `python -m pytest tests -q`（开发环境另装 pytest）。不需要目标域同学为交接重复开发测试。
