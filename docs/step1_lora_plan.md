# Step 1：LoRA-r8/r16 实现与实验计划

## 1. 范围与执行方式

依据 `README.md`、`step1_handoff.md`、`hypotheses.md`、`datasets.md`、现有代码及 用户提供的本地 `dialogue.md`（不随交付提交）整理。本文保留分阶段执行计划，已完成事项见文末执行记录；`dialogue.md` 中“立即执行阶段 A”属于原始任务背景，该指令不自动启动后续阶段。

本成员负责 LoRA-r8/r16；Linear Probe、预训练基线与冻结多提示由其他成员负责。按用户后续指令逐阶段实施，每阶段报告检查内容、修改文件、原因、测试命令与结果、限制及下一步。环境变更前说明最小方案；**12 组正式训练必须另获用户确认**。

初始整理阶段仅编写文档；后续按用户逐阶段授权执行。保留 Step0 验收成果、固定划分与公共接口契约；公共训练器仅作经过测试的兼容扩展。

## 2. 实验配置与固定协议

| 项目 | 配置 |
| --- | --- |
| 初始化 | 每次独立加载同一官方 MedCLIP 预训练权重，记录 SHA256；禁止跨组续训 |
| 注入 | Swin 每层注意力 Q/V 线性投影；K 不注入 |
| Rank / Alpha | r8 / 16；r16 / 32，即 alpha=2r |
| LoRA dropout | 0.05 |
| 分类头 | 与 Linear Probe 对齐：`Linear(d+5, 5)` |
| 可训练参数 | 仅 LoRA A/B 矩阵与分类头；其余视觉权重、文本端、原始投影层和 logit scale 冻结 |
| 优化器 | AdamW；LoRA LR=1e-4，分类头 LR=1e-3；weight decay=1e-4 |
| Batch / 累积 | 16 / 4，名义有效 batch=64 |
| Epoch / 早停 | 最多 10；源域 val Macro-AUC 连续 3 轮无提升停止，恢复最佳 checkpoint |
| Loss | 公共 masked BCEWithLogitsLoss，仅有效标签参与 |
| 数据预算 / 种子 | `10pct`（400 人）、`100pct`（4000 人）；42、43、44 |
| 提示与指标 | LoRA 主实验固定单模板；阈值 0.5；Macro-AUC、Macro-F1、Macro-AUPRC 及逐类指标 |

共 `2 ranks × 2 budgets × 3 seeds = 12` 次正式训练。每组最佳 checkpoint 分别评测固定 MIMIC test 与 CheXpert test。目标域仅用于最终测试，不用于方案选择或排错调参。

`d` 从实际 bundle 获取；现有官方权重投影后 d=512，因此 `v/q/z` 维度为 512/5/517。`q` 复用未乘 logit scale 的正负提示余弦相似度差，`z=[v;q]`，分类头产生五维 logits。不要把 Swin 投影前 768 维误作 d。与 Linear Probe 负责人核对特征定义，保持其实现不变。

## 3. 代码接入与待补能力

已具备：`src/model_loader.py` 严格加载固定官方权重；`src/model_adapter.py` 通过 `experiment.factory` 调用 `fn(bundle=bundle, config=config)`；公共模型返回 `ModelBatchOutput`。`src/trainer.py` 已支持 `optimizer_groups(config)`、masked loss、源域 val 早停、最佳 checkpoint 保存与恢复；`scripts/run_experiment.py` 已支持固定预算名单及统一评测。

后续拟新增独立 LoRA 模块、配置模板和测试，具体文件名在阶段 C 确定。使用不同方法名（建议 `lora_r8`、`lora_r16`）及独立输出目录，避免结果覆盖。新增配置键属于待实现设计，不能把配置文件存在视为能力已实现。

实现顺序：

1. 检查真实 `named_modules()`、投影形状及参数路径，记录 Swin Q/K/V 结构；不能仅凭命名猜测。独立 Q/V 时包装对应线性层；融合 QKV 时仅向 Q/V 切片加入增量，K 切片增量恒为零。
2. 使用 `W_eff=W+(alpha/r)BA`；低秩支路施加 dropout，以一个矩阵为零初始化使初始增量为零。保持原权重不变，验证注入前后基础投影等价。线性层新增参数量为 `r*(in_features+out_features)`，分类头为 `5*(d+5)+5`。
3. 缓存冻结文本向量，计算可微的 v/q/z，接入分类头；图像前向不能整体包在 `no_grad()` 中。明确训练模式：只启用 LoRA dropout，保持冻结骨干及文本端确定性。
4. 通过已有 `optimizer_groups(config)` 为 LoRA 与分类头建立无重复参数组；检查设备移动、可训练参数清单及参数量。
5. 公共训练器目前每个有效 batch 都更新一次，尚无梯度累积；提出兼容默认行为的最小扩展。明确尾部不足累积步数、全无效 batch 和有效标签数量不等时的 loss 归一化，按有效监督量验证累积结果，避免尾组缩放错误。
6. 增加耗时、峰值显存与实际优化步数记录；保留现有 checkpoint 契约并记录方法、rank、配置、权重来源和最佳轮次。混合精度为可选项，须另行验证数值稳定性。

## 4. 分阶段安排与验收

### 阶段 A：只读环境与交接检查

检查 Git 状态/分支、目录、Python/Conda、PyTorch/CUDA、驱动及 GPU 显存。用户提供的 Ubuntu 24.04、RTX 3060 12GB、i5-12500 仅为预期，不视为已核实。检查 manifest、六份训练名单、两域实际图像、完整外部 `step0_model/` 与资产清单；核对现有测试/验收入口及依赖是否可用。

验收：给出已存在、缺失、未验证清单与最小环境方案。缺少资源时列出交接目录、网盘获取方式和下一步；不自动下载大型数据，不用随机输入冒充真实验收。阶段 A 不执行路径迁移或写出验收报告。

### 阶段 B：真实 Step0 接口复验

资源齐全且用户指令进入本阶段后，按交接说明准备个人配置和必要路径迁移；仅迁移 `image_path`，保留样本、标签、划分及名单，不提交个人路径。先核对原版交接指纹，再做真实加载与两域 16 张有效图像检查，运行既有回归测试。

**报告保护：**`check_handoff.py` 和 `run_step0.py` 会写报告。复验须将个人配置 `data.output_dir` 指向独立目录（如 `outputs/step1_validation`）；检查既有报告时先将必要报告和聚合频数复制到该目录。不得覆盖 `outputs/` 中原始验收记录，不能手改指纹。新增 `src/*.py` 等也会改变指纹，因此原版交接核验应在实现前完成，之后保留原报告作为历史证据。

验收：真实权重加载、GPU 推理、两域加载、五维输出、v/q/z 维度及现有测试通过。16 图结果仅证明接口可用。兼容性问题先报告原因与最小方案，不擅自修改公共接口；总结后等待下一阶段指令。

### 阶段 C：实现 LoRA 与兼容训练扩展

按第 3 节顺序实施，优先 r8，再参数化为 r16。复用现有数据、评测、特征缓存和 checkpoint 逻辑；不更改 `config/step0.yaml` 或其他成员方法。说明低秩参数为何减少训练量、Q/V 的作用、实际 Swin 结构、冻结策略及 rank 差异。

验收：方法 factory 可构建两种 rank；配置覆盖第 2 节要求；新增能力与公共入口兼容，默认冻结预训练方法行为保持不变。

### 阶段 D：测试与数值验证

以 `python -m pytest tests/` 为回归入口，新增针对行为的测试：

- Q/V 注入位置与参数量符合公式；独立/融合结构按实际支持范围覆盖，融合 K 增量为零。
- 仅 LoRA 与分类头可训练；执行优化后逐项比较冻结权重及相关 buffers 数值，不能只查 `requires_grad`。
- 初始零增量、五维有限 logits、v/q/z 形状及公共预测/缓存格式正确。
- 无效标签不影响 loss/梯度；全无效 batch 不更新参数；拒绝目标域训练。
- 累积步数、尾组、不同有效标签数、参数分组学习率正确；既有默认路径保持兼容。
- LoRA 获得有限梯度并实际更新；零初始化可能使某一矩阵首步梯度为零，验证应覆盖后续优化步。
- 早停依据源域 val，最佳 checkpoint 保存/严格重载后在 eval 模式复现预测；记录数值容差。

验收：新旧相关测试通过；说明合成单测与真实模型验证各自覆盖范围。pytest 不在现有 `requirements.txt`，缺失时列入环境方案，不默默安装。测试阶段不启动正式训练。

### 阶段 E：RTX 3060 小规模训练

用户指令进入后，从固定源域 train 名单确定性选取 128 名患者用于诊断，记录选择规则、seed 和名单；不改正式名单，报告有效标签患者数。使用 r8 训练 2 epoch，验证 loss 趋势、有限梯度、LoRA 更新、冻结权重不变、checkpoint 重载、耗时及峰值显存。诊断使用独立目录，绝不作为正式结果。

优先 batch=16、累积=4；若 OOM，可改为 8/8，保持名义有效 batch=64。可选 FP16，必须检测 NaN/Inf；所有偏离记录原因和实际配置，不根据目标 test 结果调整。loss 不下降时报告并排查，不宣称通过。

验收：展示真实诊断记录及是否具备正式训练条件的判断，等待用户明确确认。

### 阶段 F：正式训练与汇总（确认后）

逐组执行 12 次独立训练，保存配置、Git 版本、权重摘要、训练参数量、历史、最佳轮次、耗时与峰值显存。新建实验配置模板和个人路径副本；个人副本须排除提交，现有 `.gitignore` 只明确忽略 `config/step0.local.yaml`，不能假设其他 `*.local.yaml` 已被忽略。

以下仅示例，`LORA_CONFIG.yaml` 与 `BEST_CHECKPOINT.pt` 必须替换为阶段 C 后实际配置和产物，当前不执行：

```bash
python scripts/run_experiment.py --config LORA_CONFIG.yaml --action train --domain mimic
python scripts/run_experiment.py --config LORA_CONFIG.yaml --action eval --domain mimic --split test --checkpoint BEST_CHECKPOINT.pt
python scripts/run_experiment.py --config LORA_CONFIG.yaml --action eval --domain chexpert --split test --checkpoint BEST_CHECKPOINT.pt
```

正式评测不使用 `--limit`；训练和评测保持方法、rank、预算、seed 一致。不同阶段/实验使用独立目录，避免公共命名规则覆盖已有结果。

验收：每组两域完整 test 预测、逐类/汇总指标、元信息和 v/q/z 缓存齐备；使用 `collect_metrics.py h1/h2` 与其他成员的同协议基线对齐，按预算/rank 汇总三种子结果及波动。H2 在各域内对齐同一图像，报告相对预训练的漂移；不把两机构不同图像直接配对。H3 冻结多提示仍由相应成员负责，不据 LoRA 结果更换提示。报告数据与预训练重叠限制，不以性能必须提升作为实现正确性的标准。

## 5. 执行记录（2026-10-30）

A/B 已完成环境、资源及真实两域接口复验。按用户授权，C/D 已实现独立 Q/V 的 LoRA-r8/r16、配置模板、兼容梯度累积和训练统计；16 项新旧测试通过，两种 rank 的两张真实源域图像 GPU 前向/反向检查通过。实现说明见 [lora_implementation.md](lora_implementation.md)。E 已完成128人、r8、2轮诊断：53人标签有效，loss 0.693802→0.688076，batch16/累积4无OOM，峰值已分配显存1.27GiB，冻结状态与checkpoint重载通过。F 已获用户授权并完成12组正式训练及24份两域test，交付说明见 lora_delivery.md。
