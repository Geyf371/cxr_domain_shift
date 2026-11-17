"""LoRA on the pinned Swin's independent attention Q/V projections."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model_adapter import MedCLIPAdapter


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA requires Linear, positive rank/alpha and dropout in [0,1)")
        self.base = base.requires_grad_(False)
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs):
        return self.base(inputs) + self.scale * F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)


def inject_qv(vision, rank, alpha, dropout):
    """Fail closed on fused or unrecognized layouts; never wrap K."""
    targets = [(name, module) for name, module in vision.named_modules()
               if name.endswith('attention.self')]
    if not targets:
        raise ValueError("No independent Swin attention.self Q/K/V found; fused QKV is unsupported")
    for name, module in targets:
        if not all(isinstance(getattr(module, key, None), nn.Linear) for key in ('query', 'key', 'value')):
            raise ValueError(f"Unsupported Q/K/V structure at {name}; no injection performed")
    paths = []
    for name, module in targets:
        for key in ('query', 'value'):
            setattr(module, key, LoRALinear(getattr(module, key), rank, alpha, dropout))
            paths.append(f'{name}.{key}')
    return paths


class LoRAMedCLIPAdapter(MedCLIPAdapter):
    def __init__(self, bundle, config):
        templates = int(config.get('experiment', {}).get('prompt_templates', 1))
        if templates != 1:
            raise ValueError("LoRA main experiment requires the fixed single template")
        super().__init__(bundle, templates=templates)
        cfg = config['lora']
        rank = int(cfg['rank'])
        if rank not in (8, 16):
            raise ValueError("This experiment supports ranks 8 and 16")
        self.injected_paths = inject_qv(self.backbone.vision_model, rank,
                                        float(cfg.get('alpha', 2 * rank)), float(cfg.get('dropout', .05)))
        self.head = nn.Linear(self.feature_dim + 5, 5).to(next(self.backbone.parameters()).device)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        # Keep original backbone dropout/buffers deterministic, enable only
        # the newly added low-rank dropout while preserving autograd.
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                module.dropout.train(mode)
        return self

    def forward_batch(self, batch):
        output = super().forward_batch(batch)
        output.logits = self.head(output.joint_features_z)
        return output

    def optimizer_groups(self, config):
        cfg = config.get('training', {})
        low_rank = [p for name, p in self.backbone.named_parameters() if p.requires_grad]
        return [{'params': low_rank, 'lr': float(cfg.get('lora_learning_rate', 1e-4))},
                {'params': list(self.head.parameters()), 'lr': float(cfg.get('head_learning_rate', 1e-3))}]


def build_lora(*, bundle, config):
    return LoRAMedCLIPAdapter(bundle, config)
