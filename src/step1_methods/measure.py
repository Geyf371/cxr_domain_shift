"""Step 1 要求的运行测量：训练时间、推理时间、峰值显存、参数量、逐层更新检查。

实验要求第 5 条原文：

    保存"方法、标注预算、种子、数据域、患者、5 类 logits／概率、v／q／z"；
    记录训练时间、1000 张图像推理时间、峰值显存和参数量。
    先用源域 128 名患者训练 2 轮，确认分类损失下降；逐层检查线性探针
    只更新头部、LoRA 只更新注入矩阵和头部、冻结提示分支无梯度更新。

`src/trainer.py::fit_source` 是 Step 0 的共享代码，不返回耗时和显存，
也不做逐层检查。本模块用装饰/包裹的方式补上，不改动共享代码。

用法::

    from measure import TrainingMeter, measure_inference, parameter_report

    meter = TrainingMeter(model)
    with meter:
        result = fit_source(...)
    record = meter.report()
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def _device_of(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    """参数总量、可训练量、以及逐层的 requires_grad / 是否变化。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layers = []
    for name, param in model.named_parameters():
        layers.append({
            "name": name,
            "shape": list(param.shape),
            "numel": int(param.numel()),
            "requires_grad": bool(param.requires_grad),
        })
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_ratio": float(trainable) / total if total else 0.0,
        "trainable_names": [x["name"] for x in layers if x["requires_grad"]],
        "frozen_names": [x["name"] for x in layers if not x["requires_grad"]],
        "layer_count": len(layers),
        "layers": layers,
    }


def layerwise_update_check(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> dict[str, Any]:
    """逐层比对训练前后的参数，确认只有预期层发生变化。

    参数:
        model:    训练后的模型
        snapshot: 训练前 {参数名: 克隆的 tensor}

    返回:
        changed:            实际发生变化的层名
        unchanged:          未变化的层名
        frozen_but_changed: 被冻结却发生变化（必须为空）
        trainable_but_unchanged: 可训练但没变化（通常意味着 lr 太小或没参与前向）
    """
    changed, unchanged, frozen_but_changed, trainable_but_unchanged = [], [], [], []
    for name, param in model.named_parameters():
        if name not in snapshot:
            continue
        same = torch.equal(param.detach().cpu(), snapshot[name])
        if same:
            unchanged.append(name)
            if param.requires_grad:
                trainable_but_unchanged.append(name)
        else:
            changed.append(name)
            if not param.requires_grad:
                frozen_but_changed.append(name)
    return {
        "changed": changed,
        "unchanged_count": len(unchanged),
        "unchanged": unchanged,
        "frozen_but_changed": frozen_but_changed,
        "trainable_but_unchanged": trainable_but_unchanged,
        "only_expected_changed": not frozen_but_changed,
    }


class TrainingMeter:
    """包住 fit_source，记录耗时、峰值显存，并做逐层更新检查。"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = _device_of(model)
        self.snapshot: dict[str, torch.Tensor] = {}
        self.seconds = 0.0
        self.peak_memory_mb: float | None = None
        self.peak_memory_reserved_mb: float | None = None

    def __enter__(self) -> "TrainingMeter":
        self.snapshot = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters()}
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            self.peak_memory_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
        self.seconds = time.perf_counter() - self._start

    def report(self) -> dict[str, Any]:
        params = parameter_report(self.model)
        updates = layerwise_update_check(self.model, self.snapshot)
        return {
            "training_seconds": round(self.seconds, 2),
            "peak_memory_allocated_mb": None if self.peak_memory_mb is None else round(self.peak_memory_mb, 1),
            "peak_memory_reserved_mb": None if self.peak_memory_reserved_mb is None else round(self.peak_memory_reserved_mb, 1),
            "device": str(self.device),
            "parameters": params,
            "layerwise_update": updates,
        }


def measure_inference(model: torch.nn.Module, loader, target_images: int = 1000) -> dict[str, Any]:
    """测量纯推理耗时（eval + no_grad），按 target_images 折算。

    不修改模型状态；调用前后参数和 buffer 逐位比对，确认推理无副作用。
    """
    device = _device_of(model)
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    buffers_before = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
    was_training = model.training
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    count = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            model.forward_batch(batch)
            count += len(batch["patient_id"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if was_training:
        model.train()

    unchanged = all(torch.equal(p.detach().cpu(), before[n]) for n, p in model.named_parameters())
    buffers_unchanged = all(torch.equal(b.detach().cpu(), buffers_before[n]) for n, b in model.named_buffers())
    return {
        "images": count,
        "seconds": round(elapsed, 2),
        "seconds_per_1000_images": round(elapsed / count * target_images, 2) if count else None,
        "images_per_second": round(count / elapsed, 2) if elapsed else None,
        "peak_memory_allocated_mb": (round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                                     if device.type == "cuda" else None),
        "parameters_unchanged": bool(unchanged),
        "buffers_unchanged": bool(buffers_unchanged),
    }
