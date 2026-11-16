"""Step 1 · 成员A: Linear Probe 的结构与冻结检查。

不加载真实权重，用假的 bundle 验证:
  1. 只有分类头可训练，主干一个 requires_grad 都不能有
  2. z 的维数是 d+5
  3. forward_batch 返回合法的 ModelBatchOutput，logits 形状 [B, 5] 且有限
  4. train() 不会把主干切回训练态
  5. 同一输入重复前向，v 逐位一致（Step 3 漂移分析的前提）

用法:
    python -m tests.test_step1_linear_probe
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CLASS_NAMES
from src.model_interface import ModelBatchOutput
from src.step1_methods.linear_probe import LinearProbeAdapter

passed, failed = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"[PASS] {name}")
    else:
        failed += 1
        print(f"[FAIL] {name}" + (f"  ({detail})" if detail else ""))


# ---------------- 假 backbone ----------------
class FakeVision(nn.Module):
    """把 (B,1,224,224) 压成 (B,512)，模拟 Swin + projection_head 的输出。

    真实 MedCLIP 的 vision_model 返回 projection_head 之后的结果，
    维数是 512（不是 Swin 的 768），这里必须一致。
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(224 * 224, 512)

    def forward(self, pixel_values, project=True):
        x = pixel_values.flatten(1)
        out = self.proj(x)
        if project:
            out = F.normalize(out, dim=-1)
        return out


class FakeText(nn.Module):
    """按 token id 的均值造一个 (B,512) 向量，只为让形状对上。"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 512)

    def forward(self, input_ids, attention_mask):
        x = input_ids.float().mean(dim=1, keepdim=True)
        return self.proj(x)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = FakeVision()
        self.text_model = FakeText()
        self.logit_scale = nn.Parameter(torch.zeros(()))


class FakeTokenizer:
    def __call__(self, texts, padding=None, max_length=None, truncation=None,
                 return_tensors=None):
        n = len(texts)
        ids = torch.arange(1, max_length + 1).unsqueeze(0).repeat(n, 1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeBundle:
    def __init__(self):
        self.model = FakeBackbone()
        self.processor = type("P", (), {"tokenizer": FakeTokenizer()})()
        self.vision_feature_dim = 512
        self.provenance = {}


def make_adapter(templates: int = 1) -> LinearProbeAdapter:
    return LinearProbeAdapter(FakeBundle(), templates=templates)


def make_batch(batch_size: int = 3) -> dict:
    return {
        "pixel_values": torch.randn(batch_size, 1, 224, 224),
        "labels": torch.randint(0, 2, (batch_size, 5)).float(),
        "mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "patient_id": [f"p{i}" for i in range(batch_size)],
        "image_id": [f"i{i}" for i in range(batch_size)],
        "sample_id": [f"s{i}" for i in range(batch_size)],
        "domain": ["mimic"] * batch_size,
        "split": ["train"] * batch_size,
    }


def test_frozen_backbone() -> None:
    model = make_adapter()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    check("只有分类头可训练",
          all(n.startswith("classifier.") for n in trainable), str(trainable))
    check("分类头参数数为 2", len(trainable) == 2, str(trainable))

    backbone_trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    check("主干无可训练参数", backbone_trainable == [], str(backbone_trainable))


def test_joint_dim() -> None:
    model = make_adapter()
    check("z 维数 = d+5", model.joint_dim == 512 + 5, f"{model.joint_dim}")
    check("分类头输入维数正确",
          model.classifier.in_features == 512 + 5, f"{model.classifier.in_features}")
    check("分类头输出维数 = 5", model.classifier.out_features == 5)


def test_forward_shape() -> None:
    model = make_adapter()
    model.eval()
    with torch.no_grad():
        out = model.forward_batch(make_batch(3))
    check("返回 ModelBatchOutput", isinstance(out, ModelBatchOutput))
    check("logits 形状 [3,5]", tuple(out.logits.shape) == (3, 5), str(tuple(out.logits.shape)))
    check("logits 有限", bool(torch.isfinite(out.logits).all()))
    check("v 形状 [3,512]", tuple(out.image_features_v.shape) == (3, 512))
    check("q 形状 [3,5]", tuple(out.prompt_scores_q.shape) == (3, 5))
    check("z 形状 [3,517]", tuple(out.joint_features_z.shape) == (3, 517))
    check("v 已 L2 归一化",
          torch.allclose(out.image_features_v.norm(dim=-1),
                         torch.ones(3), atol=1e-5))


def test_train_keeps_backbone_eval() -> None:
    model = make_adapter()
    model.train()
    check("train() 后主干仍是 eval", not model.backbone.training)
    check("train() 后自身是 train", model.training)


def test_v_deterministic() -> None:
    """同一输入两次前向，v 必须逐位一致 —— Step 3 漂移分析的前提。"""
    model = make_adapter()
    model.eval()
    batch = make_batch(2)
    with torch.no_grad():
        a = model.encode(batch)[0]
        b = model.encode(batch)[0]
    check("v 重复前向逐位一致", torch.equal(a, b),
          f"max|diff|={(a - b).abs().max().item():.3e}")


def test_templates() -> None:
    m1, m5 = make_adapter(1), make_adapter(5)
    check("1 模板 text_features 形状",
          tuple(m1.text_features.shape) == (5, 2, 1, 512),
          str(tuple(m1.text_features.shape)))
    check("5 模板 text_features 形状",
          tuple(m5.text_features.shape) == (5, 2, 5, 512),
          str(tuple(m5.text_features.shape)))
    # q 的维数与模板数无关
    m5.eval()
    with torch.no_grad():
        out = m5.forward_batch(make_batch(2))
    check("5 模板下 q 仍是 5 维", tuple(out.prompt_scores_q.shape) == (2, 5))


def test_optimizer_groups_absent() -> None:
    """没定义 optimizer_groups 时，公共 fit_source 用默认单组。"""
    model = make_adapter()
    check("无 optimizer_groups（走公共默认）",
          getattr(model, "optimizer_groups", None) is None)


def main() -> None:
    for fn in (test_frozen_backbone, test_joint_dim, test_forward_shape,
               test_train_keeps_backbone_eval, test_v_deterministic,
               test_templates, test_optimizer_groups_absent):
        print(f"\n=== {fn.__name__} ===")
        fn()
    print(f"\n{'='*50}\n通过 {passed}, 失败 {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
