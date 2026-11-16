# Step 1 · 成员A 交接说明

分支: `feat/step1-baseline-linearprobe`
负责方法: 原始预训练零样本 + Linear Probe

---

## 1. 我做了什么

### 零样本：不需要写代码

Step 0 的 `src/model_adapter.py::MedCLIPAdapter` 已经实现了零样本，且口径与实验要求一致：

```python
similarities = einsum(v, text_features)          # (B, 5, 2, T)
differences  = similarities[:,:,0,:] - similarities[:,:,1,:]   # 正 - 负
logits       = differences * logit_scale.exp()   # 乘预训练温度
# templates 平均后 sigmoid（evaluator 里做）
```

`experiment.prompt_templates: 1` 即单模板零样本，`5` 即多模板集成。
**成员A 的零样本直接跑 Step 0 现有代码即可，不新增文件。**

### Linear Probe：新增一个 factory

| 文件 | 说明 |
|---|---|
| `src/step1_methods/linear_probe.py` | Linear Probe 实现 |
| `src/step1_methods/measure.py` | 训练/推理耗时、峰值显存、参数量、逐层更新检查 |
| `config/step1_linear_probe.yaml` | 完整配置模板（可复制为 `.local.yaml` 后填路径） |
| `scripts/step1_run.py` | Step 1 运行入口（带测量），命令行与 `run_experiment.py` 一致 |
| `scripts/run_member_a.py` | 成员A 完整实验矩阵，一条命令跑完 |
| `scripts/update_manifest_paths.py` | 把 manifest 的 `image_path` 改到本机 |
| `scripts/compare_runs.py` | 对比两个产物目录的 macro 指标（逐格给差值） |
| `tests/test_step1_linear_probe.py` | 20 项结构测试 |
| `tests/test_step1_measure.py` | 22 项测量模块测试 |

核心设计：
- 只训练 `Linear(d+5, 5)`，输入是 `z = [v; q]`
- 复用 Step 0 的 `fit_source`（AdamW + masked BCE + 早停 + checkpoint）
- 复用 Step 0 的 `evaluate_model`（预测/指标/`v q z` 导出）
- `q` 的算法与 `MedCLIPAdapter` **逐字一致**，保证与零样本可比

### 为什么另开 `step1_run.py` 而不改 `run_experiment.py`

`scripts/run_experiment.py` 是 Step 0 的共享入口，成员 B、C 都在用。
实验要求第 5 条要的「训练时间 / 1000 张推理时间 / 峰值显存 / 参数量 /
逐层检查」在 `fit_source` 里都没有出口，而 `fit_source` 同样是共享代码。

所以 `scripts/step1_run.py` 复制了 `run_experiment.py` 的约 30 行配置与数据
装载逻辑，把 `fit_source` / `evaluate_model` 包一层，**Step 0 共享代码
一个字没动**。新增的产物：

| 文件 | 内容 |
|---|---|
| `measurement_{method}_{budget}_seed{seed}.json` | 训练耗时、峰值显存、参数量、逐层更新检查 |
| `inference_{method}_{domain}_{split}_{budget}_seed{seed}.json` | 推理耗时（含 1000 张折算）、峰值显存、参数未变校验 |

`measurement_*.json` 里的 `layerwise_update.frozen_but_changed` 必须为空，
否则 `step1_run.py` 直接以非零码退出——这就是「逐层检查线性探针只更新头部」
的机器可验证凭据。

---

## 2. 怎么跑

### 前置：补全模型缓存

Step 0 交接说明要求的 `step0_model/` 应有 `official/` 和 `python/` 两个目录，
实际交接的资源里缺失。补法：

```bash
CACHE="<你的路径>/step0_model"

# 1) official/ —— 从 MedCLIP 官方仓库取（commit 9c3396f2）
git clone https://github.com/RyanWangZf/MedCLIP
cd MedCLIP && git checkout 9c3396f20d5d54e4fae241b8cb06ca45848e98c9
mkdir -p "$CACHE/official" && cp -r medclip "$CACHE/official/"

# 2) .py 必须是 LF，否则 sha256 校验失败
python -c "
import pathlib
for p in pathlib.Path('$CACHE/official').rglob('*.py'):
    p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
"

# 3) python/ —— 装 nltk 等依赖
pip install --target "$CACHE/python" nltk==3.9.1 regex requests tqdm filelock packaging
```

校验通过的标准：`load_medclip` 不抛 `Model asset checksum mismatch`。

### 运行

```bash
# 1) 复制配置并填本机路径
cp config/step1_linear_probe.yaml config/step1_linear_probe.local.yaml
#    改三处: data.mimic_root / data.chexpert_root / model.cache_dir
#    （.local.yaml 已在 .gitignore 里，不会被提交）

# 2) 更新 manifest 里的 image_path
python scripts/update_manifest_paths.py
#    先在内存里校验完 5000/1000 张图全部存在才写回，缺一张就整体中止

# 3) 一条命令跑完整个矩阵（27 次调用，见下）
python scripts/run_member_a.py
#    先看会跑什么：python scripts/run_member_a.py --dry-run
#    只跑一部分：  python scripts/run_member_a.py --only zeroshot
```

`scripts/run_member_a.py` 展开后是：

| 步骤 | 次数 | 内容 |
|---|---|---|
| 零样本 | 3 | `pretrained` 直接评测 mimic/val、mimic/test、chexpert/test |
| Linear Probe 训练 | 6 | budget ∈ {10pct, 100pct} × seed ∈ {42,43,44} |
| Linear Probe 评测 | 18 | 每组 checkpoint 评测上述三域 |

零样本与预算无关，只跑一次（`budget` 记为 `100pct`）；
报告里 10%/100% 两列复用同一结果，**需要在表格脚注里注明**。

### 常用开关

```bash
# 只跑 10pct 三个 seed，产物单独放（不覆盖已有结果）
python scripts/run_member_a.py --only linearprobe --budgets 10pct --out outputs/step1_b64

# 换 batch_size（默认 64，与实验要求一致）
python scripts/run_member_a.py --batch-size 32

# 对比两个目录的 macro 指标，逐格给出差值
python scripts/compare_runs.py outputs/step1 outputs/step1_b64 --budget 10pct

# 生成报告表格（含耗时/显存/参数量/逐层检查）
python scripts/summarize_member_a.py --dir outputs/step1_b64
```

---

## 3. 验证状态

### 已通过

| 检查 | 结果 |
|---|---|
| 结构测试 20 项 | 全部通过 |
| 测量模块测试 22 项 | 全部通过 |
| 真实权重加载 | `vision_dim=512`, `logit_scale=13.34` |
| 可训练参数 | 仅 `classifier.weight/bias`（2,590 / 136,618,649 = 0.0019%） |
| 主干梯度 | 训练后仍为 `None`（确认冻结） |
| 主干参数/buffer | 逐位不变、版本号未变 |
| 逐层更新检查 | 实际变化层只有 `classifier.weight/bias`，冻结层零变化 |
| 维度 | `joint_dim=517`, `logits [B,5]` |
| v 归一化 | `norm = 1.000000` |
| v 重复前向一致性 | 逐位相同（Step 3 前提） |
| 16 图数据冒烟 | 两域均通过 |
| Step 0 交接校验 | `PASS: Step 0 handoff ready` |
| 128 患者冒烟 | 12 项断言全通过，loss 0.6767 → 0.6546 |
| 推理无副作用 | 参数与 buffer 逐位不变（3 个 seed 均验证） |

128 患者冒烟测试用 `python scripts/step1_smoke_128.py`，产物是
`outputs/step1/smoke128_linear_probe.json`（scope 标为
`smoke_128_patients_not_a_formal_experiment`，不会被当作正式结果）。

---

## 4. 重要发现：零样本口径在真实权重上很弱

**这一节请全组都看，它影响成员 B、C 的对照基线。**

### 现象

零样本（Step 0 原口径）三域实测：

| 域 / 划分 | macro AUC | macro F1 |
|---|---|---|
| mimic / val | 0.5009 | 0.4615 |
| mimic / test | 0.4862 | 0.3819 |
| chexpert / test | 0.3352 | 0.2892 |

源域接近随机，目标域低于随机。

### 根因：MedCLIP 嵌入有约 98% 的公共分量

用 `python scripts/diagnose_zeroshot.py` 量化（产物
`outputs/step1/zeroshot_diagnosis_*.json`）：

| 量 | 数值 |
|---|---|
| 图像嵌入范数均值 | 38.56 |
| 其中公共均值向量范数 | 37.69（**占 97.8%**） |
| 图像残差范数（真正的区分信号） | 9.29（占 24.1%） |
| 文本嵌入残差占比 | **1.7%** |
| 文本两两余弦均值 | 0.9997 |
| 同一类别的 正/负 提示余弦 | 0.9998 |

也就是说：文本塔对不同句子的输出方向几乎一样（"A chest X-ray showing
Atelectasis." 与 "hello world" 的余弦也有 0.99）。Step 0 的口径是

```python
v = L2norm(image);  t = L2norm(text)
q = v·t_pos - v·t_neg          # 两边公共分量相消，只剩 1e-3 量级
logit = q * exp(logit_scale)   # 13.34 倍后仍然约 0.001
```

sigmoid(0.001) ≈ 0.5000，所以早期排查时看到"所有概率恰好 0.5"。
不同类别之间的差分方向仍有细微差别，因此没有完全退化成常数，
但区分度很弱。

### 三种打分方式对照（同一批图，`step0_difference` 与正式评测逐位一致）

| 打分方式 | mimic/val | mimic/test | chexpert/test |
|---|---|---|---|
| `step0_difference`（**正式结果**） | 0.5009 | 0.4862 | 0.3352 |
| `step0_onevsall` | 0.1498 | — | 0.1375 |
| `centered_onevsall` | 0.6110 | 0.6215 | 0.5636 |
| `centered_difference` | 0.4846 | 0.5859 | 0.3654 |

`centered_*` 是用文本集合均值中心化后重算，**只作诊断对照，不是正式结果**。
它说明嵌入里的信息还在，只是被公共分量压住了。

### 处理方式

**正式结果一律按 Step 0 原口径报告，不擅自改动共享代码。**
`src/model_adapter.py` 一个字都没动，成员 B、C 的基线因此不受影响。
上面这组对照只作为报告里的"局限与讨论"，供全组决定是否在 Step 2/3 统一修正口径。

如果全组决定修正，建议的方向是：对 `v` 和 `t` 做中心化（减训练集均值）
再算相似度，且必须**三个方法同时改**，否则不可比。

---

## 4.5 核心结论：特征里有信息，零样本取不出来

### 正式结果（batch=64，3 seed 均值 ± 标准差）

> 口径：`batch=64` + 早停 `patience=3`，与实验要求一致。
> 产物目录 `outputs/step1_b64/`。

macro AUC：

| 方法 | 预算 | mimic/val | mimic/test | chexpert/test |
|---|---|---|---|---|
| 零样本 | 100pct | 0.5009 | 0.4862 | 0.3352 |
| Linear Probe | 10pct | 0.9116 ± 0.0125 | 0.8737 ± 0.0057 | 0.8638 ± 0.0368 |
| Linear Probe | 100pct | 0.9129 ± 0.0021 | 0.8837 ± 0.0016 | 0.8748 ± 0.0261 |

macro F1：

| 方法 | 预算 | mimic/val | mimic/test | chexpert/test |
|---|---|---|---|---|
| 零样本 | 100pct | 0.4615 | 0.3819 | 0.2892 |
| Linear Probe | 10pct | 0.4758 ± 0.0461 | 0.4569 ± 0.0914 | 0.4381 ± 0.0796 |
| Linear Probe | 100pct | 0.5895 ± 0.0215 | 0.5156 ± 0.0072 | 0.5124 ± 0.0046 |

macro AUPRC：

| 方法 | 预算 | mimic/val | mimic/test | chexpert/test |
|---|---|---|---|---|
| 零样本 | 100pct | 0.6777 | 0.6074 | 0.5618 |
| Linear Probe | 10pct | 0.9344 ± 0.0121 | 0.8959 ± 0.0071 | 0.9152 ± 0.0043 |
| Linear Probe | 100pct | 0.9292 ± 0.0048 | 0.9052 ± 0.0024 | 0.9264 ± 0.0016 |

零样本与预算无关，只跑一次（`budget` 记为 `100pct`），
表格里 10%/100% 两列复用同一结果。

### 对照：batch=16（早期探索口径，非正式）

产物目录 `outputs/step1/`。列在这里是为了说明 batch 口径的影响，
**不作为正式结果**。

| 指标 | 方法/预算 | mimic/val | mimic/test | chexpert/test |
|---|---|---|---|---|
| AUC | LP 10pct | 0.9140 ± 0.0113 | 0.8793 ± 0.0042 | 0.8795 ± 0.0188 |
| AUC | LP 100pct | 0.9203 ± 0.0000 | 0.8953 ± 0.0022 | 0.9112 ± 0.0014 |
| F1 | LP 10pct | 0.5305 ± 0.0063 | 0.5205 ± 0.0237 | 0.5174 ± 0.0150 |
| F1 | LP 100pct | 0.8681 ± 0.0008 | 0.8422 ± 0.0004 | 0.8480 ± 0.0017 |
| AUPRC | LP 10pct | 0.9357 ± 0.0137 | 0.9025 ± 0.0031 | 0.9215 ± 0.0083 |
| AUPRC | LP 100pct | 0.9313 ± 0.0001 | 0.9191 ± 0.0031 | 0.9351 ± 0.0010 |

两种口径的差异及根因见第 4.6 节。**AUC/AUPRC 基本一致（差 0.002~0.036），
差异集中在 F1**，且 batch=64 的 F1 更低——原因是早停提前终止，不是方法更差。

### 资源与规模记录（实验要求第 5 条）

**训练**（batch=64 正式口径）：

| 预算 | seed | 轮数 | 训练时间 (s) | 峰值显存 (MB) | 参数量 | 可训练量 | 占比 |
|---|---|---|---|---|---|---|---|
| 10pct | 42 | 8 | 526.83 | 1231.4 | 136,618,649 | 2,590 | 0.0019% |
| 10pct | 43 | 8 | 463.51 | 1250.9 | 136,618,649 | 2,590 | 0.0019% |
| 10pct | 44 | 10 | 561.59 | 1231.4 | 136,618,649 | 2,590 | 0.0019% |
| 100pct | 42 | 5 | 2735.07 | 1272.4 | 136,618,649 | 2,590 | 0.0019% |
| 100pct | 43 | 4 | 2097.35 | 1272.4 | 136,618,649 | 2,590 | 0.0019% |
| 100pct | 44 | 4 | 2071.17 | 1272.4 | 136,618,649 | 2,590 | 0.0019% |

可训练参数只有 `classifier.weight` (5×517=2,585) 和 `classifier.bias` (5)，
共 **2,590**，占全部 136,618,649 参数的 **0.0019%**。

**推理**（eval + no_grad，1000 张 chexpert/test 实测）：

| 方法 | 预算 | seed | 耗时 (s) | 峰值显存 (MB) | 参数未变 |
|---|---|---|---|---|---|
| 零样本 | 100pct | 42 | 16.24 | 1831.0 | 是 |
| Linear Probe | 10pct | 42/43/44 | 11.06 / 10.23 / 10.32 | 1831.0 | 是 |
| Linear Probe | 100pct | 42/43/44 | 16.29 / 16.13 / 16.81 | 1831.0 | 是 |

同一批 1000 张图，**零样本 16.24s、线性探针 16.29s，几乎没有差别**——
线性头只是 `Linear(517,5)`，计算量可忽略，开销全在冻结主干的视觉前向。

**逐层更新检查**（6 组全部通过）：

| 预算 | seed | 实际变化的层 | 冻结却变化 |
|---|---|---|---|
| 10pct | 42/43/44 | `classifier.weight`, `classifier.bias` | 无 |
| 100pct | 42/43/44 | `classifier.weight`, `classifier.bias` | 无 |

`step1_run.py` 在 `frozen_but_changed` 非空时会直接以非零码退出，
所以"只有头部被更新"是机器强制的，不依赖人工检查。

### 三点解读

**① 特征里有信息，零样本取不出来。**
同一个冻结主干，只加一个 `Linear(517, 5)`，目标域 AUC 从 0.335
跳到 0.875~0.911。说明 MedCLIP 的视觉特征确实编码了这 5 个类别的信息，
只是零样本的正负提示差分读不出来（与第 4 节量化的公共分量问题一致）。

**② 跨院迁移几乎不掉点。**
batch=64 正式口径下，100pct 的 mimic/test 0.8837 → chexpert/test 0.8748，
目标域只低 0.009。10pct 只用了 155 张有效训练图，目标域仍有 0.8638。

**③ AUC 靠特征，F1 靠训练是否充分。**
batch=64 口径下 10pct → 100pct，目标域 AUC 只涨 0.011（0.8638 → 0.8748），
F1 涨 0.074（0.4381 → 0.5124）；而在 batch=16 口径下 F1 从 0.5174 涨到 0.8480（+0.33）。
**F1 对优化步数和早停时机极度敏感**，AUC/AUPRC 则稳健得多。
报告里应以 AUC/AUPRC 为主要结论，F1 必须注明训练配置（batch、轮数、是否早停）。

这为成员 B、C 的 H1（源域增益能否迁移）提供了基准线：
**目标域 AUC 要超过 0.88（batch=64 口径）/ 0.91（batch=16 口径）
才算 LoRA 真的带来了可迁移的增益**，仅仅在源域涨点是不够的。

### 训练细节（batch=64 正式口径）

| 预算 | seed | 轮数 | 起始 loss | 结束 loss | 最佳 val AUC | 早停 |
|---|---|---|---|---|---|---|
| 10pct | 42 | 8 | 0.6845 | 0.6154 | 0.9188 | 是 |
| 10pct | 43 | 8 | 0.6875 | 0.5975 | 0.9217 | 是 |
| 10pct | 44 | 10 | 0.6956 | 0.5819 | 0.8979 | 否（跑满） |
| 100pct | 42 | 5 | 0.6392 | 0.5055 | 0.9170 | 是 |
| 100pct | 43 | 4 | 0.6441 | 0.5233 | 0.9151 | 是 |
| 100pct | 44 | 4 | 0.6477 | 0.5248 | 0.9148 | 是 |

batch=16 对照口径下 6 组全部跑满 10 轮、从未早停：

| 预算 | seed | 起始 loss | 结束 loss | 最佳 val AUC |
|---|---|---|---|---|
| 10pct | 42 | 0.6746 | 0.5464 | 0.9214 |
| 10pct | 43 | 0.6720 | 0.5103 | 0.9211 |
| 10pct | 44 | 0.6801 | 0.5024 | 0.9010 |
| 100pct | 42 | 0.5921 | 0.3753 | 0.9203 |
| 100pct | 43 | 0.5929 | 0.3735 | 0.9207 |
| 100pct | 44 | 0.5949 | 0.3747 | 0.9203 |

**注意一个反常现象**：100pct 的 val AUC（约 0.9148~0.9170）和 10pct
（约 0.9116）几乎相同，但 test 上的 F1 差了 0.28~0.34。

原因是源域验证集只有 **191 张有效图**，样本量太小，AUC 已经饱和，
无法区分"训练不足"和"训练充分"这两种状态。详见第 4.6 节。

这说明**用源域 val AUC 做模型选择在这个数据规模下不可靠**，
报告里应当提到这一点。如果后续要改进，可以考虑：
用更大的验证集、或用 val 上的 loss/F1 做早停。

---

## 4.6 batch_size 口径：为什么最终用 64

实验要求写死了 `batch=64`。我最初为省显存用了 16，后来按原要求改回 64，
并把两种口径都跑了一遍以量化差异（`scripts/compare_runs.py`）。

### 10pct 预算：batch=64 明显更差

| 指标 | 域/划分 | batch=16 | batch=64 | 差值 |
|---|---|---|---|---|
| macro AUC | mimic/val | 0.9140 ± 0.0113 | 0.9116 ± 0.0125 | −0.0024 |
| macro AUC | mimic/test | 0.8793 ± 0.0042 | 0.8737 ± 0.0057 | −0.0056 |
| macro AUC | chexpert/test | 0.8795 ± 0.0188 | 0.8638 ± 0.0368 | −0.0157 |
| macro F1 | mimic/val | 0.5305 ± 0.0063 | 0.4758 ± 0.0461 | −0.0547 |
| macro F1 | mimic/test | 0.5205 ± 0.0237 | 0.4569 ± 0.0914 | −0.0635 |
| macro F1 | chexpert/test | 0.5174 ± 0.0150 | 0.4381 ± 0.0796 | −0.0793 |
| macro AUPRC | mimic/val | 0.9357 ± 0.0137 | 0.9344 ± 0.0121 | −0.0013 |
| macro AUPRC | mimic/test | 0.9025 ± 0.0031 | 0.8959 ± 0.0071 | −0.0066 |
| macro AUPRC | chexpert/test | 0.9215 ± 0.0083 | 0.9152 ± 0.0043 | −0.0063 |

AUC 和 AUPRC 基本持平（掉 0.001~0.016），但 **F1 掉了 0.05~0.08**，
且方差明显变大（chexpert F1 标准差从 0.015 涨到 0.080）。

原因很直接：10pct 只有 **155 张有效训练图**，`batch=64` 下每轮只有
**3 个梯度步**，10 轮总共 30 步；`batch=16` 是每轮 10 步、共 100 步。
优化步数少了 3 倍多，线性头没收敛到稳定解，输出概率分布更容易
卡在 0.5 附近，固定阈值 0.5 的 F1 因此大幅波动。

### 结论

- **正式结果一律用 batch=64**，因为这是实验要求写死的口径，
  且同一张表内所有方法必须一致才能横向比较
- 但报告里应当注明：**10pct 预算下 batch=64 只相当于 30 个梯度步，
  F1 的低值和高方差部分是优化步数不足造成的，不全是方法本身的性能**
- 若全组希望在小预算下更公平，应统一改成 `batch=16` 或加梯度累积，
  但这会偏离原始要求，**必须三个方法一起改并在报告里说明**

### 100pct 预算：差异比 10pct 更大，但根因不同

| 指标 | 域/划分 | batch=16 | batch=64 | 差值 |
|---|---|---|---|---|
| macro AUC | mimic/val | 0.9203 ± 0.0000 | 0.9129 ± 0.0021 | −0.0075 |
| macro AUC | mimic/test | 0.8953 ± 0.0022 | 0.8837 ± 0.0016 | −0.0116 |
| macro AUC | chexpert/test | 0.9112 ± 0.0014 | 0.8748 ± 0.0261 | −0.0363 |
| macro F1 | mimic/val | 0.8681 ± 0.0008 | 0.5895 ± 0.0215 | −0.2785 |
| macro F1 | mimic/test | 0.8422 ± 0.0004 | 0.5156 ± 0.0072 | −0.3265 |
| macro F1 | chexpert/test | 0.8480 ± 0.0017 | 0.5124 ± 0.0046 | −0.3356 |
| macro AUPRC | mimic/val | 0.9313 ± 0.0001 | 0.9292 ± 0.0048 | −0.0021 |
| macro AUPRC | mimic/test | 0.9191 ± 0.0031 | 0.9052 ± 0.0024 | −0.0139 |
| macro AUPRC | chexpert/test | 0.9351 ± 0.0010 | 0.9264 ± 0.0016 | −0.0086 |

AUC/AUPRC 只掉 0.002~0.036，**但 F1 掉了 0.28~0.34**。
这次不是"步数不够"——100pct 下 batch=64 每轮 25 步，10 轮 250 步，
远多于 10pct 的 30 步。真正的原因是**早停提前终止了训练**：

| seed | batch=16 轮数 | batch=16 最佳 val AUC | batch=64 轮数 | batch=64 最佳 val AUC |
|---|---|---|---|---|
| 42 | 10（跑满） | 0.9203 | **5（早停）** | 0.9170 |
| 43 | 10（跑满） | 0.9207 | **4（早停）** | 0.9151 |
| 44 | 10（跑满） | 0.9203 | **4（早停）** | 0.9148 |

逐轮曲线暴露了矛盾（batch=64, seed42）：

| epoch | train loss | val AUC |
|---|---|---|
| 1 | 0.6392 | 0.9147 |
| 2 | 0.5777 | **0.9170** ← 最佳，被选为 checkpoint |
| 3 | 0.5459 | 0.9145 |
| 4 | 0.5237 | 0.9130 |
| 5 | 0.5055 | 0.9138 |

**loss 一路稳定下降（0.639 → 0.506），val AUC 却已经饱和并在噪声里抖动。**
于是"连续 3 轮不升"在第 5 轮就触发了，最终交付的是 ep2 的 checkpoint——
那时分类头只训练了 50 步，输出概率还挤在 0.5 附近，固定阈值 0.5 的 F1 自然崩。

batch=16 之所以没暴露这个问题，是因为每轮只有 10 步、更新更慢更平滑，
val AUC 恰好在 10 轮里一直缓慢爬升，早停从未触发，反而"因祸得福"跑满了。

### 这暴露了评测协议的一个真实缺陷

**源域 val 只有 191 张有效图，Macro-AUC 早已饱和，无法分辨"训练不足"
和"训练充分"。** 用它同时做早停判据和 checkpoint 选择，在 100pct 这种
"loss 还能继续降"的情况下会系统性地提前终止。

更精确地说：**AUC 衡量排序能力，F1 还依赖概率校准，两者收敛速度差一个量级。**
线性头第 1 轮就把排序学到位（val AUC 0.9147），但要把 logit 的尺度/截距
调整到与标签频率匹配、让固定阈值 0.5 有意义，需要 8-10 轮。
用 AUC 做早停等于在"排序刚学会、校准还没开始"时就喊停。

这是 Step 0 共享代码 `fit_source` 的既定行为，成员 B、C 的 LoRA 也会遇到
同样的问题（LoRA 收敛更慢，早停更可能提前掐断）。

**建议全组统一考虑**（需要三个人一起改，否则不可比）：
- 早停判据换成 **val loss**（loss 一路降到 0.4456，没饱和），或
- 保留 AUC 判据但把 patience 从 3 放宽到 ≥5（实测 10 轮才收敛），或
- 用 val F1 做判据，或
- 扩大源域验证集

### 诊断证据：跑满 10 轮就能拿回 F1

`scripts/diagnose_earlystop.py --seed 42 --budget 100pct` 禁用早停、跑满 10 轮，
每轮同时记录 val 的 AUC / F1 / AUPRC。产物
`outputs/step1_diag/earlystop_diagnosis.json`（scope=diagnostic）。

| epoch | train loss | val AUC | **val F1** | val AUPRC |
|---|---|---|---|---|
| 1 | 0.6392 | 0.9147 | 0.5310 | 0.9346 |
| 2 | 0.5777 | **0.9170** ← AUC 最佳 | 0.5435 | 0.9366 |
| 3 | 0.5459 | 0.9145 | 0.5451 | 0.9327 |
| 4 | 0.5237 | 0.9130 | 0.5759 | 0.9274 |
| 5 | 0.5055 | 0.9138 | 0.6045 | 0.9283 |
| 6 | 0.4906 | 0.9140 | 0.7268 | 0.9285 |
| 7 | 0.4774 | 0.9153 | 0.7810 | 0.9296 |
| 8 | 0.4656 | 0.9147 | 0.8194 | 0.9288 |
| 9 | 0.4549 | 0.9156 | 0.8214 | 0.9294 |
| 10 | 0.4456 | 0.9159 | **0.8351** ← F1 最佳 | 0.9298 |

**val AUC 全程在 0.9130–0.9170 之间抖动（极差 0.004），
val F1 却从 0.5310 单调爬到 0.8351（涨 0.30）。** 两者完全脱钩。

同一份训练的两份 checkpoint 在 test 上的对比：

| checkpoint | mimic/test AUC | mimic/test **F1** | chexpert/test AUC | chexpert/test **F1** |
|---|---|---|---|---|
| 最佳 val AUC（ep2） | 0.8832 | 0.5239 | 0.8902 | 0.5176 |
| 最后一轮（ep10） | 0.8876 | **0.7980** | 0.9043 | **0.7941** |

**只因为早停挑错 checkpoint，F1 就损失了 0.27–0.28**，
AUC 也顺带损失 0.004–0.014。

机理清楚了：**线性头学"排序"很快（AUC 第 1 轮就到位），
学"校准"很慢（要把 logit 的截距/尺度挪到和标签频率匹配，
F1 才上得来）**。val AUC 只看得见排序，看不见校准，
于是它在第 2 轮就"满意"了。

> **正式结果仍按原始要求（batch=64 + 早停 patience=3）报告**，
> 上面的分析写进报告的"局限与讨论"，不擅自改动共享代码。

---

## 5. 数据协议上的两个注意点

这两点不是 bug，是 Step 0 既定协议的结果，但写报告时必须交代，
否则数字会被误读。

### ① 大量样本没有有效标签

MIMIC-CXR 的 chexpert 标签表里，多数研究对多数类别是空白（NaN），
只有报告明确提到的才标 1.0/0.0/−1.0。Step 0 的规则是 NaN 和 −1.0
都算无效（`mask=0`），评测时整行无有效标签的样本被跳过。

| 集合 | 总图数 | 有有效标签 | 跳过 |
|---|---|---|---|
| mimic / train（完整） | 4000 | 1567 | 2433 |
| mimic / train（10pct 子集） | 400 | 155 | 245 |
| mimic / val | 500 | 191 | 309 |
| mimic / test | 500 | 206 | 294 |
| chexpert / test | 1000 | 762 | 238 |

所以 10% 预算下实际只有 155 张训练图、191 张验证图。
报告里写"10% 预算"时要注明这是**有效样本数**，不是 400。

### ② F1 在 10pct 预算下失效，100pct 恢复正常

实验要求 F1 用固定阈值 0.5。10pct 预算下 F1 只有约 0.52，
但 **100pct 下升到约 0.85**，不是阈值设置的问题。

原因是 10pct 只有 155 张有效训练图，模型训练不足，输出概率被压在
很窄的区间里，固定阈值几乎全判正或全判负：

| 类别 | 10pct 概率 p25–p75 | 10pct 判正比例 | 100pct 概率 p25–p75 | 100pct 判正比例 |
|---|---|---|---|---|
| Atelectasis | 0.824–0.834 | 100% | 0.920–0.957 | 100% |
| Cardiomegaly | 0.624–0.658 | 100% | 0.541–0.797 | 78% |
| Consolidation | 0.395–0.417 | 0% | 0.245–0.613 | 41% |
| Edema | 0.403–0.466 | 1% | 0.186–0.579 | 35% |
| Pleural Effusion | 0.439–0.513 | 33% | 0.205–0.779 | 47% |

100pct 下概率分布明显展开，阈值 0.5 才有区分作用。
所以 **F1 低是 10pct 训练不足的体现，不是评测口径的缺陷** ——
这一点在报告里要写清楚，不能笼统说"固定阈值不合理"。

---

## 6. 阻塞项

原先的两个阻塞项都已解决：

- 图像：已从 `10.60.38.174:55555` 取回 8.71 GB 压缩包，解压到
  `D:\step0_data\extracted`，6000 张图全部到位，manifest 路径已更新
- 模型缓存：`official/` 和 `python/` 已补齐，`load_medclip` 校验通过

---

## 7. 给成员 B、C 的提示

### Swin 的 Q/V 是融合层

`official/medclip/vision_model.py:102`：

```python
self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # Q/K/V 在一个层里
```

实验要求写"在 query、value 线性层注入 LoRA"，但这里没有独立的 Q/V 层。
两种做法：
- **拆层**：改成 `q_proj/k_proj/v_proj` 三个独立 Linear，只给 Q/V 加 LoRA（严格符合要求）
- **整层**：对 `qkv` 注入，同时影响 Q/K/V（实现简单但不符要求）

建议拆层，并在 `trainable_parameter_names()` 里明确列出注入的层名。

### 5 模板已由 Step 0 提供

`src/prompts.py` 里 `POSITIVE_TEMPLATES` / `NEGATIVE_TEMPLATES` 各 5 条，
第一条就是单模板。成员C 直接 `build_prompts(templates=5)` 即可，
**不需要自己设计模板**；且天然满足"单模板是多模板之一"的受控对比要求。

### 特征顺序一致

`evaluate_model` 按 `sample_id` 保存特征，同一域的各方法顺序自动一致，
Step 3 可直接逐图比对。但要注意：不同方法的 `v` 只有线性探针和零样本应完全相同，
LoRA 会改变 `v`，这是正常的、也是要测量的。

### 早停会在 100pct 下提前掐断训练（对 LoRA 影响更大）

这是我在补 batch=64 时发现的，**成员 B 必须注意**：

源域 val 只有 191 张有效图，Macro-AUC 早已饱和。100pct + batch=64 时，
训练 loss 还在稳定下降（0.639 → 0.506），val AUC 却在噪声里抖动，
于是 `patience=3` 在第 4-5 轮就触发，交付的 checkpoint 只训练了几十步。
后果是 test F1 掉 0.33（0.848 → 0.512），而 AUC 只掉 0.03。

LoRA 收敛比线性探针更慢，**早停更可能提前掐断**。建议：
- 训练时同时记录 val loss，观察它是否还在降
- 若 val AUC 饱和而 loss 仍降，考虑放宽 patience 或改用 val loss 做判据
- 三个方法必须一起改，否则不可比

`scripts/diagnose_earlystop.py` 可以跑满固定轮数、对比"最佳 AUC checkpoint"
与"最后一轮 checkpoint"，用来量化早停造成的损失。
