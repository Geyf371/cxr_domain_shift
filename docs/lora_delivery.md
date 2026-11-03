# LoRA 交付说明

## 范围与协议

成员B交付LoRA-r8/r16：两个预算（10%/100%）、三个种子（42/43/44），共12组独立正式训练，及每组MIMIC test/CheXpert test的预测、指标和v/q/z。使用原始Step0固定单模板；C的五模板属于独立冻结分支，不替换本交付的默认提示。

每组重新加载原始MedCLIP权重，只训练图像Q/V LoRA与Linear(517,5)。batch16/累积4、FP32、AdamW、LoRA LR=1e-4、head LR=1e-3、weight decay=1e-4；r8/alpha16、r16/alpha32、dropout=.05，最多10轮，源域val Macro-AUC patience=3。目标域不参与训练或模型选择。源域test Macro-AUC有效类别数为4，目标域为5；说明NA类别后再解释域间差值。

## 接收包结构

交付包在仓库外生成，不将医学影像、个人配置、大型checkpoint或逐患者特征提交Git：

```text
README.md
manifest.json                 # 12组索引、checkpoint相对包根路径、资源统计口径
SHA256SUMS                    # 除自身之外每个文件的SHA256
code.bundle                   # step1-member-b的离线Git分支及历史
runs/<method>_<budget>_seed<seed>/
  config.portable.yaml        # 无个人路径；运行前配置自己的资源根和索引副本
  preflight.json              # 实际训练源码摘要与原Git提交
  integrity.json              # 冻结参数/buffers和可训练张量更新证据
  completed.json
  run.log
  outputs/                   # 最佳checkpoint、训练history、val/test预测/指标/特征
validation/                  # 单测、真实两图梯度检查与128人诊断证据
training_environment.freeze.txt
 test_metrics.csv             # 24行两域test汇总
 test_metrics_mean_std.csv    # 按方法/预算/域汇总三种子均值±样本标准差
```

预训练资源、原始影像和本机绝对路径不在包内。预测和特征包含患者伪标识，应按课程数据协议私下交接，不发布到公开Git。原始固定索引已在仓库历史中，不在本包重新抽样。

## 校验与离线取得代码

在解压后的包根目录执行：

```bash
sha256sum -c SHA256SUMS
git clone -b step1-member-b code.bundle cxr_domain_shift
cd cxr_domain_shift
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -p no:cacheprovider
```

推荐Python3.10；实际训练Python3.10.22/PyTorch2.14.1/torchvision0.29.1/CUDA13.0。requirements-dev用于安装测试依赖；training_environment.freeze.txt记录训练环境，模型缓存另提供固定transformers4.24.0及huggingface_hub0.25.2。勿另装任意MedCLIP/tokenizers替换缓存运行时。

bundle提交是整理后的交付提交，不等于训练时提交：训练时LoRA代码尚未提交，准确源码摘要保留在各组preflight.json。未改写原训练记录中的提交来伪装重跑。

## 模型复用

将某组config.portable.yaml复制到仓库外的个人配置，填写data.mimic_root、data.chexpert_root、model.cache_dir、data.data_dir；按原交接说明迁移个人manifest的image_path，患者/标签/划分不变。为data.output_dir填写新的输出目录，不覆盖交付包证据。manifest.json内checkpoint路径相对包根；训练JSON中的checkpoint路径相对该JSON父目录。

```bash
python scripts/run_experiment.py --config /path/to/personal.yaml --action eval --domain mimic --split test --checkpoint /path/to/package/runs/RUN/outputs/CHECKPOINT.pt
python scripts/run_experiment.py --config /path/to/personal.yaml --action eval --domain chexpert --split test --checkpoint /path/to/package/runs/RUN/outputs/CHECKPOINT.pt
```

这些是恢复/评测示例，不要求接收方重新训练12组。保持对应rank、budget、seed及原提示版本。下一阶段分析以完整test缓存为输入，不以诊断结果替代正式结果。

## 与成员C的交付工具对接

公共模型接口保持factory/forward_batch/ModelBatchOutput不变。本包使用原Step0产物schema并增加包级manifest，不假装满足C新分支的全部强制字段。对接时需要schema兼容层，并修复其相对checkpoint路径重复拼接问题。C新增资源统计可以选择性整合，不能用其旧trainer覆盖本分支的梯度累积实现。

已有elapsed_seconds包含训练、每轮验证及checkpoint保存/恢复；peak_cuda_memory_bytes是训练/验证阶段CUDA峰值已分配张量内存，不是整卡显存占用。无法从旧日志精确恢复纯训练耗时，原交付manifest中以null表示。后续已完成同配置纯训练资源测量和1000图像独立推理计时，合并表及口径见 [资源记录](lora_resource_records.md)，补充包v2与原包配套使用，原实验结果不改写。

## 已有验证与待办

16项新旧测试通过；真实Swin两种rank前向/反向通过；128人/r8/2轮诊断通过（53人标签有效），诊断checkpoint未打包，仅提供证据；12组最佳checkpoint均已严格恢复后完成两域test，冻结状态检查通过。交付包校验针对已有结果，不触发训练。

尚需配合同协议预训练/Linear Probe基线完成H1/H2、图像表征漂移及组内后续分析。clinical_v1五模板可以独立比较，但不得把旧checkpoint缓存文本向量标记成clinical_v1。此次分支整理不合并成员C、不推送远端。

## 重新导出

在已提交的step1-member-b分支运行，目标目录必须不存在：

```bash
python scripts/export_lora_delivery.py --source /path/to/formal --destination /path/to/new_delivery
```

导出会验证12组完成标记和24份test预测/特征对应，复制产物、去除个人路径、重算汇总并生成校验清单。交付中的配置路径占位、日志路径脱敏和checkpoint相对路径转换属于包装处理；checkpoint张量与原预测/特征保持原样。
