# LoRA 资源记录

12组资源总表见 [lora_resource_summary.csv](results/lora_resource_summary.csv)。时间来自同配置资源测量，三个种子的统计汇总应保留标准差。

- `train_only_seconds`：训练epoch总和，含数据读取、预处理、前向、反向和优化器更新；不含验证、模型初始化、checkpoint保存/恢复。
- `replay_validation_seconds`及各分项：同配置测量中的源域验证、初始化、checkpoint保存与恢复时间。
- `training_validation_seconds`：原正式运行的训练/验证总时间，保留原口径。
- `inference_model_seconds_mean/std`：固定CheXpert全部1000张图像的forward_batch耗时，batch16、FP32，每个checkpoint独立进程，预热5个batch后重复3次。
- `inference_end_to_end_seconds_mean/std`：含图像读取、预处理、H2D、模型前向、sigmoid与CPU结果收集，排除模型加载和磁盘导出。无效标签图像也参与计时；分类指标仍只计算有效标签。
- `inference_peak_allocated_bytes_max`：PyTorch峰值已分配张量显存，不等于整卡显存占用。

原12个正式checkpoint保留。同配置资源测量的训练轮数、逐轮loss和源域val AUC与原记录一致，临时checkpoint在恢复检查后删除。历史JSON不重写；补充包v2内保留逐轮分项、验证结果及时间来源。

复现单组计时（会重新执行训练，仅为资源测量）：

```bash
python scripts/measure_lora_training_time.py --formal-run /path/to/formal/lora_r8_10pct_seed42 --output-root /path/to/new_time_records
```

对固定checkpoint做推理计时（不训练）：

```bash
python scripts/benchmark_lora.py --formal-root /path/to/formal --run-name lora_r8_10pct_seed42 --output /path/to/new_inference_records
```

每个checkpoint分别启动上述推理进程，避免同进程加载多个模型影响显存口径。

合并已有记录，不训练或改写旧包：

```bash
python scripts/merge_lora_time_records.py --records /path/to/v1_records --replay /path/to/time_replay --destination /path/to/new_v2_records
```

补充包v2提供资源总表、原始分项记录、曲线、逐病种指标、清单和校验值，不重复携带大型checkpoint；与原6.18GiB模型交付包配套使用。
