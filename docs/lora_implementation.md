# LoRA 实现与验证

## 接入与参数

`src.lora:build_lora` 复用 MedCLIPAdapter 的固定文本向量及 v/q/z 计算，保持 ModelBatchOutput 和公共评测接口不变。分类头为 `Linear(517,5)`，其中 d 从实际 bundle 读取；q 是未乘 logit scale 的正负提示相似度差。当前仓库未包含其他成员的正式 Linear Probe 实现，因此分类头按交接约定实现，尚未进行跨成员实现的数值对照。

已检查真实预训练模型的 named_modules：12 个 Swin 注意力块分别含独立 query/key/value，维数为 96、192、384、768。只包装 24 个 Q/V 投影；K、文本端、原视觉参数、投影层及 logit scale 保持冻结。未支持的融合 QKV 结构会直接报错，不静默注入整个 QKV。

LoRA 增量为 `(alpha/r)BA`，A 随机初始化、B 为零，因此初始投影等价。每个线性层只新增 `r*(in+out)` 个参数，显著少于更新完整权重；r16 的低秩容量和参数量约为 r8 两倍。选择 Q/V 用于改变注意力查询和内容表示，遵循本项目固定实验方案，不通过目标域调参。

只有 LoRA 与分类头可训练；原骨干始终 eval，仅 LoRA dropout 随 train/eval 切换。保留视觉端 autograd，使梯度可穿过冻结投影和特征计算回传到 LoRA。

| 配置 | Rank/Alpha | LoRA 参数 | 分类头参数 | 总可训练参数 |
| --- | --- | --- | --- | --- |
| config/lora_r8.yaml | 8/16 | 141312 | 2590 | 143902 |
| config/lora_r16.yaml | 16/32 | 282624 | 2590 | 285214 |

两份完整配置模板使用 dropout=.05、AdamW、LoRA LR=1e-4、head LR=1e-3、weight decay=1e-4、batch=16、累积=4、最多10轮及源域 val Macro-AUC patience=3。个人资源路径不写入模板。每次正式实验须重新加载相同预训练权重，不能跨 rank/预算/种子续训。

## 公共训练器的兼容扩展

train_source_epoch 新增可选关键词 gradient_accumulation_steps，默认1。累积 masked loss 的有效标签总和，更新前将梯度除以整组有效标签数；尾组单独归一化，全无效 batch 不推进累计步数或执行优化器更新。此行为与相同有效样本合并为大 batch 的目标一致。

复用 optimizer_groups、早停、最佳 checkpoint 保存及重载。新增实际优化步数、每轮/总耗时、CUDA 峰值已分配显存及训练参数量记录。checkpoint 元数据加入训练/LoRA 配置和预训练来源，保持原 state_dict/metadata 契约。暂未加入 FP16；若后续诊断需要，另行实现并验证数值稳定性。

## 如何验证

```bash
python -m pytest tests/ -p no:cacheprovider
python scripts/check_lora.py --config config/step0.local.yaml --output /tmp/lora_check.json
```

单测检查初始零增量、两种 rank 参数量、Q/V 定位、融合结构拒绝、冻结权重和 buffers 数值不变、A/B 后续非零梯度及实际更新、参数组学习率、输出契约、严格 checkpoint 重载、mask/NaN 排除、累计尾组和全无效跳过、源域早停及最佳权重恢复。

真实验证脚本仅使用 seed42 固定10%训练名单中的两张有效源域图像，检查前向、初始图像特征等价、五维 logits、v/q/z、有限梯度及原权重无梯度。不执行优化器更新，不使用目标域选择方案，也不替代128人训练或完整性能实验。

当前已完成两种 rank 的真实 RTX 3060 前向/反向验证；结果保存在仓库外 `resources/validation/lora_forward_backward.json`。随后已完成128人诊断及12组正式训练，batch16/累积4运行通过。交付与结果边界见 [lora_delivery.md](lora_delivery.md)。

新增源代码与训练器变更会改变 Step0 代码指纹；保留原验收报告及训练前复验记录，不修改其指纹，也不将旧报告冒充当前代码的重新验收。
