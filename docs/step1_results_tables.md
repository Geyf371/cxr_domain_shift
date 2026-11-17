# 成员A 正式结果

数据来源：`outputs/step1_b64/metrics_summary_*.csv`，只取 `scope=experiment`。
共 21 次评测；数值为 3 个 seed 的 均值 ± 标准差。

## macro_auc

| 方法 | 预算 | mimic/val | mimic/test | chexpert/test |
|---|---|---|---|---|
| pretrained | 100pct | 0.5009 | 0.4862 | 0.3352 |
| linear_probe | 10pct | 0.9116 ± 0.0125 | 0.8737 ± 0.0057 | 0.8638 ± 0.0368 |
| linear_probe | 100pct | 0.9129 ± 0.0021 | 0.8837 ± 0.0016 | 0.8748 ± 0.0261 |

## macro_f1

| 方法 | 预算 | mimic/val | mimic/test | chexpert/test |
|---|---|---|---|---|
| pretrained | 100pct | 0.4615 | 0.3819 | 0.2892 |
| linear_probe | 10pct | 0.4758 ± 0.0461 | 0.4569 ± 0.0914 | 0.4381 ± 0.0796 |
| linear_probe | 100pct | 0.5895 ± 0.0215 | 0.5156 ± 0.0072 | 0.5124 ± 0.0046 |

## macro_auprc

| 方法 | 预算 | mimic/val | mimic/test | chexpert/test |
|---|---|---|---|---|
| pretrained | 100pct | 0.6777 | 0.6074 | 0.5617 |
| linear_probe | 10pct | 0.9344 ± 0.0121 | 0.8959 ± 0.0071 | 0.9152 ± 0.0043 |
| linear_probe | 100pct | 0.9292 ± 0.0048 | 0.9052 ± 0.0024 | 0.9264 ± 0.0016 |

## 每次评测的样本量

| 方法 | 预算 | 域/划分 | num_samples | 跳过（无有效标签） |
|---|---|---|---|---|
| pretrained | 100pct | mimic/val | 191 | 309 |
| pretrained | 100pct | mimic/test | 206 | 294 |
| pretrained | 100pct | chexpert/test | 762 | 238 |
| linear_probe | 10pct | mimic/val | 191 | 309 |
| linear_probe | 10pct | mimic/test | 206 | 294 |
| linear_probe | 10pct | chexpert/test | 762 | 238 |
| linear_probe | 100pct | mimic/val | 191 | 309 |
| linear_probe | 100pct | mimic/test | 206 | 294 |
| linear_probe | 100pct | chexpert/test | 762 | 238 |

## 资源与规模记录（实验要求第 5 条）

### 训练

| 方法 | 预算 | seed | epoch | 训练时间 (s) | 峰值显存 (MB) | 参数量 | 可训练量 | 可训练占比 |
|---|---|---|---|---|---|---|---|---|
| linear_probe | 100pct | 42 | 5 | 2735.07 | 1272.4 | 136,618,649 | 2,590 | 0.0019% |
| linear_probe | 100pct | 43 | 4 | 2097.35 | 1272.4 | 136,618,649 | 2,590 | 0.0019% |
| linear_probe | 100pct | 44 | 4 | 2071.17 | 1272.4 | 136,618,649 | 2,590 | 0.0019% |
| linear_probe | 10pct | 42 | 8 | 526.83 | 1231.4 | 136,618,649 | 2,590 | 0.0019% |
| linear_probe | 10pct | 43 | 8 | 463.51 | 1250.9 | 136,618,649 | 2,590 | 0.0019% |
| linear_probe | 10pct | 44 | 10 | 561.59 | 1231.4 | 136,618,649 | 2,590 | 0.0019% |

### 逐层更新检查

| 预算 | seed | 实际变化的层 | 冻结却变化 | 只有预期层变化 |
|---|---|---|---|---|
| 100pct | 42 | classifier.weight, classifier.bias | 无 | 是 |
| 100pct | 43 | classifier.weight, classifier.bias | 无 | 是 |
| 100pct | 44 | classifier.weight, classifier.bias | 无 | 是 |
| 10pct | 42 | classifier.weight, classifier.bias | 无 | 是 |
| 10pct | 43 | classifier.weight, classifier.bias | 无 | 是 |
| 10pct | 44 | classifier.weight, classifier.bias | 无 | 是 |

### 推理（eval + no_grad，按 1000 张折算）

| 方法 | 预算 | seed | 域/划分 | 实测张数 | 耗时 (s) | 1000 张耗时 (s) | 峰值显存 (MB) | 参数未变 |
|---|---|---|---|---|---|---|---|---|
| linear_probe | 100pct | 42 | chexpert/test | 1000 | 16.29 | 16.29 | 1831.0 | 是 |
| linear_probe | 100pct | 43 | chexpert/test | 1000 | 16.13 | 16.13 | 1831.0 | 是 |
| linear_probe | 100pct | 44 | chexpert/test | 1000 | 16.81 | 16.81 | 1831.0 | 是 |
| linear_probe | 10pct | 42 | chexpert/test | 1000 | 11.06 | 11.06 | 1831.0 | 是 |
| linear_probe | 10pct | 43 | chexpert/test | 1000 | 10.23 | 10.23 | 1831.0 | 是 |
| linear_probe | 10pct | 44 | chexpert/test | 1000 | 10.32 | 10.32 | 1831.0 | 是 |
| pretrained | 100pct | 42 | chexpert/test | 1000 | 16.24 | 16.24 | 1831.0 | 是 |
