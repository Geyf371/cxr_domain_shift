"""Frozen pretrained inference plus a common factory contract for later steps."""
from __future__ import annotations
import importlib
import torch
from torch import nn
from torch.nn import functional as F
from .data import CLASS_NAMES
from .model_interface import ModelBatchOutput
from .prompts import build_prompts


class MedCLIPAdapter(nn.Module):
    def __init__(self, bundle, templates=1):
        super().__init__()
        self.backbone = bundle.model
        self.backbone.requires_grad_(False).eval()
        self.templates = templates
        self.feature_dim = bundle.vision_feature_dim
        self.prompt_texts = build_prompts(templates=templates)
        texts = [text for disease in CLASS_NAMES for sign in ("positive", "negative")
                 for text in self.prompt_texts[disease][sign]]
        device = next(self.backbone.parameters()).device
        inputs = bundle.processor.tokenizer(texts, padding="max_length", max_length=77,
                                             truncation=True, return_tensors="pt")
        with torch.no_grad():
            # Official encode_text hard-codes CUDA; its underlying module is
            # identical here and supports the configured CPU/CUDA device.
            features = self.backbone.text_model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
            features = F.normalize(features, dim=-1).reshape(5, 2, templates, -1)
        self.register_buffer("text_features", features)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen image/text features always deterministic
        return self

    def forward_batch(self, batch):
        device = next(self.backbone.parameters()).device
        v = F.normalize(self.backbone.vision_model(batch["pixel_values"].to(device)), dim=-1)
        similarities = torch.einsum("bd,cstd->bcst", v, self.text_features)
        differences = similarities[:, :, 0, :] - similarities[:, :, 1, :]
        scale = self.backbone.logit_scale.clamp(0, 4.6052).exp()
        self.template_logits = differences * scale
        q = differences.mean(dim=-1)
        logits = self.template_logits.mean(dim=-1)  # ensemble logits BEFORE sigmoid
        return ModelBatchOutput(logits=logits, image_features_v=v, prompt_scores_q=q,
                                joint_features_z=torch.cat((v, q), dim=-1))


def build_model(bundle, config):
    method = config.get("experiment", {}).get("method", "pretrained")
    templates = int(config.get("experiment", {}).get("prompt_templates", 1))
    if method == "pretrained":
        return MedCLIPAdapter(bundle, templates=templates)
    factory = config.get("experiment", {}).get("factory")
    if not factory or ":" not in factory:
        raise ValueError(f"Method {method!r} needs experiment.factory='module:function' supplied by Step 1")
    module, function = factory.split(":", 1)
    model = getattr(importlib.import_module(module), function)(bundle=bundle, config=config)
    if not isinstance(model, nn.Module) or not callable(getattr(model, "forward_batch", None)):
        raise TypeError("The factory must return nn.Module with forward_batch(batch) -> ModelBatchOutput")
    return model
