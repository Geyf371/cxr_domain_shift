"""Common model output contract for later pretrained, linear, and LoRA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBatchOutput:
    """Outputs consumed by the shared evaluator."""

    logits: Any
    image_features_v: Any | None = None
    prompt_scores_q: Any | None = None
    joint_features_z: Any | None = None

