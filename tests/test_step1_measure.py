"""``src/step1_methods/measure.py`` 的结构测试。

覆盖实验要求第 5 条要的四个量：训练时间、推理时间、峰值显存、参数量，
以及"逐层检查只有预期层变化"。

用法::

    python -m tests.test_step1_measure
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.step1_methods.measure import (  # noqa: E402
    TrainingMeter, layerwise_update_check, measure_inference, parameter_report,
)

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


class Toy(nn.Module):
    """最小可测模型：一个冻结子模块 + 一个可训练头。"""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 8, bias=False)
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(8, 2)
        self.register_buffer("scale", torch.ones(1))

    def forward_batch(self, batch):
        z = self.backbone(batch["x"])
        return type("O", (), {"logits": self.head(z)})()


def toy_loader(n=10, batch=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=generator)
    for start in range(0, n, batch):
        chunk = x[start:start + batch]
        yield {"x": chunk, "patient_id": [f"p{i}" for i in range(len(chunk))]}


def test_parameter_report():
    print("\n=== test_parameter_report ===")
    model = Toy()
    report = parameter_report(model)
    check(report["total_parameters"] == 64 + 16 + 2, "参数总量正确",
          str(report["total_parameters"]))
    check(report["trainable_parameters"] == 18, "可训练量只算头部",
          str(report["trainable_parameters"]))
    check(report["trainable_names"] == ["head.weight", "head.bias"], "可训练层名正确",
          str(report["trainable_names"]))
    check("backbone.weight" in report["frozen_names"], "主干在冻结名单里")
    check(report["layer_count"] == 3, "层数正确", str(report["layer_count"]))


def test_layerwise_update_check():
    print("\n=== test_layerwise_update_check ===")
    model = Toy()
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        model.head.weight.add_(1.0)
        model.head.bias.add_(1.0)
    result = layerwise_update_check(model, snapshot)
    check(result["changed"] == ["head.weight", "head.bias"], "只有头部被记为变化",
          str(result["changed"]))
    check(result["frozen_but_changed"] == [], "没有冻结层被误改")
    check(result["only_expected_changed"], "only_expected_changed 为真")


def test_layerwise_detects_frozen_violation():
    print("\n=== test_layerwise_detects_frozen_violation ===")
    model = Toy()
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        model.backbone.weight.add_(1.0)   # 故意违反冻结
    result = layerwise_update_check(model, snapshot)
    check(result["frozen_but_changed"] == ["backbone.weight"],
          "冻结层被改动能被检出", str(result["frozen_but_changed"]))
    check(not result["only_expected_changed"], "only_expected_changed 为假")


def test_training_meter():
    print("\n=== test_training_meter ===")
    model = Toy()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    meter = TrainingMeter(model)
    with meter:
        for batch in toy_loader():
            optimizer.zero_grad(set_to_none=True)
            loss = model.forward_batch(batch).logits.square().mean()
            loss.backward()
            optimizer.step()
    report = meter.report()
    check(report["training_seconds"] > 0, "记录到训练耗时", f"{report['training_seconds']}s")
    check(report["peak_memory_allocated_mb"] is None, "CPU 上不报显存（应为 None）")
    check(report["layerwise_update"]["changed"] == ["head.weight", "head.bias"],
          "训练后只有头部变化", str(report["layerwise_update"]["changed"]))
    check(report["layerwise_update"]["frozen_but_changed"] == [], "主干逐位不变")
    check(report["parameters"]["trainable_parameters"] == 18, "参数量记录正确")


def test_measure_inference():
    print("\n=== test_measure_inference ===")
    model = Toy()
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    report = measure_inference(model, toy_loader(n=20, batch=5), target_images=1000)
    check(report["images"] == 20, "统计到全部图像", str(report["images"]))
    check(report["seconds"] >= 0, "记录到推理耗时", f"{report['seconds']}s")
    check(report["seconds_per_1000_images"] is not None
          and report["seconds_per_1000_images"] >= report["seconds"],
          "1000 张折算值不小于实测耗时")
    check(report["parameters_unchanged"], "推理后参数逐位不变")
    check(report["buffers_unchanged"], "推理后 buffer 逐位不变")
    check(all(torch.equal(p, params_before[n]) for n, p in model.named_parameters()),
          "独立复核：参数确实没动")


def test_inference_leaves_eval_mode():
    print("\n=== test_inference_leaves_eval_mode ===")
    model = Toy()
    model.train()
    measure_inference(model, toy_loader(n=4, batch=4))
    check(model.training, "调用前是 train 的，调用后仍恢复 train")


def main() -> None:
    test_parameter_report()
    test_layerwise_update_check()
    test_layerwise_detects_frozen_violation()
    test_training_meter()
    test_measure_inference()
    test_inference_leaves_eval_mode()
    print("\n" + "=" * 50)
    if FAILURES:
        print(f"失败 {len(FAILURES)} 项: {FAILURES}")
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
