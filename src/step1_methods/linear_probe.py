"""Step 1 · 成员A: Linear Probe factory.

只训练 ``Linear(d+5, 5)``；MedCLIP 主干、原投影层和文本端全部冻结。
输入是联合特征 ``z = [v; q]``（实验要求第 2 条），不是只有 ``v``。

接入方式（不改动 Step 0 的公共训练与评测代码）::

    experiment:
      method: "linear_probe"
      factory: "src.step1_methods.linear_probe:build_model"

``scripts/run_experiment.py`` 会调用 ``build_model(bundle=..., config=...)``，
返回值必须提供 ``forward_batch(batch) -> ModelBatchOutput``。
公共的 ``fit_source`` 负责 AdamW、masked BCE、早停和 checkpoint 保存；
公共的 ``evaluate_model`` 负责预测、指标和 ``v/q/z`` 特征导出。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.data import CLASS_NAMES
from src.model_interface import ModelBatchOutput
from src.prompts import build_prompts


class LinearProbeAdapter(nn.Module):
    """冻结 MedCLIP + 可训练线性头。

    参数:
        bundle:   Step 0 的 ``MedCLIPBundle``（含 model / processor / 特征维）
        templates: 提示模板数，1 或 5；与 Step 0 零样本保持同一口径

    冻结断言:
        构造后立即调用 ``backbone.requires_grad_(False)``，
        唯一 ``requires_grad=True`` 的参数是 ``classifier.weight/bias``。
        ``tests/test_step1_linear_probe.py`` 会检查这一点。
    """

    def __init__(self, bundle, templates: int = 1):
        super().__init__()
        self.backbone = bundle.model
        self.backbone.requires_grad_(False).eval()
        self.templates = int(templates)

        # 文本端冻结，提示向量只算一次并缓存为 buffer。
        # 与 src/model_adapter.py 的 MedCLIPAdapter 保持完全相同的取法，
        # 这样 q 的定义在零样本和线性探针之间一致。
        self.prompt_texts = build_prompts(templates=self.templates)
        texts = [text for disease in CLASS_NAMES for sign in ("positive", "negative")
                 for text in self.prompt_texts[disease][sign]]
        device = next(self.backbone.parameters()).device
        inputs = bundle.processor.tokenizer(
            texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        with torch.no_grad():
            features = self.backbone.text_model(
                inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
            features = F.normalize(features, dim=-1).reshape(5, 2, self.templates, -1)
        self.register_buffer("text_features", features)

        # z = [v; q]，维数 d + 5
        # 主干可能已被 load_medclip 搬到 CUDA，线性头必须同设备，否则 forward 报
        # "Expected all tensors to be on the same device"。
        self.feature_dim = int(bundle.vision_feature_dim)
        self.joint_dim = self.feature_dim + len(CLASS_NAMES)
        self.classifier = nn.Linear(self.joint_dim, len(CLASS_NAMES), device=device)

    # 冻结主干在任何 train() 调用下都保持 eval（BN/dropout 不生效），
    # 保证 v 在训练前后逐位一致 —— Step 3 的漂移分析依赖这一点。
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (v, q, z)。v 已 L2 归一化；q 是正负差分在模板上的平均。"""
        device = next(self.backbone.parameters()).device
        v = F.normalize(self.backbone.vision_model(batch["pixel_values"].to(device)), dim=-1)
        similarities = torch.einsum("bd,cstd->bcst", v, self.text_features)
        differences = similarities[:, :, 0, :] - similarities[:, :, 1, :]
        q = differences.mean(dim=-1)
        z = torch.cat((v, q), dim=-1)
        return v, q, z

    def forward_batch(self, batch) -> ModelBatchOutput:
        v, q, z = self.encode(batch)
        logits = self.classifier(z)
        return ModelBatchOutput(logits=logits, image_features_v=v,
                                prompt_scores_q=q, joint_features_z=z)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, p in self.named_parameters() if p.requires_grad]


def build_model(bundle, config):
    """Step 1 factory 入口。配置见 config/step1_linear_probe.yaml。"""
    templates = int(config.get("experiment", {}).get("prompt_templates", 1))
    return LinearProbeAdapter(bundle, templates=templates)
