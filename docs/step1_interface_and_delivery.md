# Step1 接口与交付规范（给成员 A、B、C）

版本：1.0
维护人：成员 C
适用分支：`step1-member-a`、`step1-member-b`、`step1-member-c`

本规范锁定 Step1 的模型入口、数据协议、输出格式、资源统计和验收方式。A、B 不应复制数据加载、训练循环或评测代码；只实现各自模型适配器，并通过公共入口运行。

## 1. 方法名称与负责人

| 方法名 | 负责人 | 训练参数 |
| --- | --- | --- |
| `pretrained` | A/C 共用 | 无；单提示冻结基线 |
| `pretrained_5prompts` | C | 无；五提示冻结集成 |
| `linear_probe` | A | 只训练线性分类头 |
| `lora_r8` | B | Q/V LoRA-r8 与分类头 |
| `lora_r16` | B | Q/V LoRA-r16 与分类头 |

五类顺序固定为：`Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion`。任何方法不得改变顺序。

## 2. 数据与实验矩阵

- 训练与早停：只使用 MIMIC 的 train/val。
- 外部测试：CheXpert 只能作为 test，不参与训练、早停或阈值选择。
- 正式预算：`10pct`（400名训练患者）和 `100pct`（4000名训练患者）。
- 正式随机种子：42、43、44。
- 正式可训练方法共运行 `2 budgets × 3 seeds`，每个 checkpoint 分别评测 MIMIC test 和 CheXpert test。
- 开始正式实验前，每个可训练方法必须先用 `--patient-limit 128` 跑源域 train/val 小测试。
- 冻结提示方法没有训练预算；为与总表配对，可沿用相同的 `budget/seed` 标签，但必须记录 `training_performed=false`、训练时间0、可训练参数0。

不得重新抽患者、删除缺图患者或改变标签 mask。现有六份 `source_train_seed{42,43,44}_{10,100}pct.csv` 是唯一正式训练名单。

## 3. 模型接入接口

公共入口在 `src/model_adapter.py::build_model`。A、B 在自己的模块提供工厂函数，并在 YAML 中配置：

```yaml
experiment:
  method: "linear_probe"       # 或 lora_r8 / lora_r16
  factory: "src.linear_probe:build_linear_probe"
  prompt_templates: 1
  seed: 42
  budget: "10pct"
```

工厂签名固定为：

```python
def build_linear_probe(*, bundle, config):
    return model  # torch.nn.Module
```

模型必须实现：

```python
def forward_batch(self, batch) -> ModelBatchOutput:
    ...
```

返回值必须是 `src.model_interface.ModelBatchOutput`：

| 字段 | 形状 | 定义 |
| --- | --- | --- |
| `logits` | `[B, 5]` | 五类分类 logits，禁止提前 sigmoid |
| `image_features_v` | `[B, D]` | L2 归一化后的视觉特征；Linear Probe 为冻结特征，LoRA 为适配后特征 |
| `prompt_scores_q` | `[B, 5]` | 固定提示下的五类正负文本相似度差，sigmoid 前 |
| `joint_features_z` | `[B, D+5]` | `torch.cat([v, q], dim=1)` |

公共代码会检查类型、维度、有限值、五类顺序和 `z` 宽度，不符合时立即终止。A/B 应复用同一套固定文本特征计算 `q`，不要用概率代替 `q`。

## 4. 参数冻结规则

### Linear Probe

- MedCLIP 视觉与文本编码器全部 `requires_grad=False`，并保持 `eval()`。
- 仅五输出线性分类头可训练。
- `v` 必须与同图像冻结基线在容差 `1e-5` 内一致。

### LoRA-r8 / LoRA-r16

- 只在 Swin 注意力 Q/V 投影加入 LoRA。
- 原始 Q/V 权重、其他视觉参数和全部文本编码器冻结。
- 只允许 LoRA A/B 矩阵及分类头可训练。
- 工厂必须明确暴露可训练参数，资源统计会自动计数。

## 5. 提示集成协议（成员 C）

- 提示版本：`clinical_v1`，定义于 `src/prompts.py`。
- 每种疾病各有五条阳性和五条阴性提示；第一对同时作为单提示基线。
- 每条提示固定最大长度77。
- 每个模板先计算 `positive_similarity - negative_similarity`。
- 五模板必须先平均 logits，再执行 sigmoid：

```python
probability = sigmoid(mean(template_logits, dim=-1))
```

禁止先对各模板 sigmoid 再平均概率。

## 6. 统一运行方式

128患者训练验收（A/B）：

```bash
python scripts/run_experiment.py \
  --config config/YOUR_METHOD.local.yaml \
  --action train --domain mimic --patient-limit 128
```

128患者冻结提示验收（C，同时检查两域、单/五提示和冻结特征）：

```bash
python scripts/run_member_c_smoke.py \
  --config config/step0.local.yaml --patients 128
```

正式训练（A/B）：

```bash
python scripts/run_experiment.py --config config/YOUR_METHOD.local.yaml --action train --domain mimic
```

正式两域评测（A/B，保持训练时 method/budget/seed 配置）：

```bash
python scripts/run_experiment.py --config config/YOUR_METHOD.local.yaml --action eval \
  --domain mimic --split test --checkpoint PATH_TO_CHECKPOINT
python scripts/run_experiment.py --config config/YOUR_METHOD.local.yaml --action eval \
  --domain chexpert --split test --checkpoint PATH_TO_CHECKPOINT
```

## 7. 每次运行的强制产物

每个 `method × budget × seed × domain × split × scope` 必须形成以下文件：

```text
predictions_<tag>.csv       # sample_id、标签/mask、logits、probabilities
features_<tag>.npz          # v、q、z、sample_id、labels、mask、classes
metrics_per_class_<tag>.csv
metrics_summary_<tag>.csv
metadata_<tag>.json         # 配置、提示、模型来源、资源统计
```

可训练方法还必须提供：

```text
checkpoint_<method>_<budget>_seed<seed>.pt
training_<method>_<budget>_seed<seed>.json
```

128患者测试文件会增加 `smoke128_` 前缀或使用 `scope=smoke128`，不得覆盖正式实验。

## 8. 资源统计口径

公共评测自动记录：

- `total_parameters`、`trainable_parameters`；
- `inference_model_seconds`：只计 `forward_batch`，CUDA 前后同步；
- `inference_end_to_end_seconds`：包含图像读取、预处理、模型前向和结果收集；
- `inference_seconds_per_sample`、`inference_samples_per_second`；
- `peak_gpu_memory_mb`：CUDA `max_memory_allocated`。

公共训练自动记录训练 epoch 总耗时和训练峰值显存，验证时间不计入 `training_seconds`。

CPU/MPS 环境无法提供 CUDA 显存时，必须写 `peak_gpu_memory_mb=null`、`peak_gpu_memory_supported=false`，禁止填写估算值。最终显存表应在同一台 CUDA 机器、相同 batch size 下重跑。

## 9. 交付前自动检查

检查单组输出：

```bash
python scripts/validate_step1_outputs.py \
  --predictions PREDICTIONS.csv \
  --features FEATURES.npz \
  --metadata METADATA.json \
  --training TRAINING.json
```

冻结 C 方法省略 `--training`。汇总多个目录并检查方法是否齐全：

```bash
python scripts/collect_step1_delivery.py \
  --inputs outputs/step1/member_a outputs/step1/member_b outputs/step1/member_c \
  --require-methods pretrained pretrained_5prompts linear_probe lora_r8 lora_r16
```

汇总器会拒绝缺失文件、重复实验标识、样本顺序不一致、特征维度错误、缺训练文件或资源字段不完整的交付。

## 10. Git 与交付清单

- A/B/C 分别使用独立分支，不直接在 `main` 混合开发。
- 不提交本机绝对路径、原始图像、模型缓存或大体积正式特征文件。
- 合并前先更新自己的分支，再由三人共同检查冲突。
- A/B 交付时应提供：分支/commit、配置文件、运行命令、128人验收报告、六个正式 checkpoint、两域预测与特征、训练日志。
- C 最终接收 A/B 产物后运行总汇程序，生成统一实验索引和资源对比表。
