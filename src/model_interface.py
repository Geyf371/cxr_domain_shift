"""Common model output contract for later pretrained, linear, and LoRA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .data import CLASS_NAMES


@dataclass
class ModelBatchOutput:
    """Outputs consumed by the shared evaluator."""

    logits: Any
    image_features_v: Any | None = None
    prompt_scores_q: Any | None = None
    joint_features_z: Any | None = None


def validate_model_batch_output(output: Any, batch_size: int) -> ModelBatchOutput:
    """Validate the shared Step 1 output contract at the producer boundary."""

    if not isinstance(output, ModelBatchOutput):
        raise TypeError("forward_batch must return ModelBatchOutput")
    if not isinstance(output.logits, torch.Tensor):
        raise TypeError("ModelBatchOutput.logits must be a torch.Tensor")
    if tuple(output.logits.shape) != (batch_size, len(CLASS_NAMES)):
        raise ValueError(f"logits must have shape [{batch_size}, {len(CLASS_NAMES)}]")
    if not torch.isfinite(output.logits).all():
        raise ValueError("logits must be finite")

    expected = {
        "image_features_v": 2,
        "prompt_scores_q": 2,
        "joint_features_z": 2,
    }
    for name, dimensions in expected.items():
        value = getattr(output, name)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.ndim != dimensions or value.shape[0] != batch_size:
            raise ValueError(f"{name} must be a [{batch_size}, dim] torch.Tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if output.prompt_scores_q is not None and output.prompt_scores_q.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"prompt_scores_q must have {len(CLASS_NAMES)} columns in the fixed class order")
    if output.image_features_v is not None and output.prompt_scores_q is not None and output.joint_features_z is not None:
        expected_z = output.image_features_v.shape[1] + output.prompt_scores_q.shape[1]
        if output.joint_features_z.shape[1] != expected_z:
            raise ValueError("joint_features_z width must equal image_features_v width + prompt_scores_q width")
    return output

