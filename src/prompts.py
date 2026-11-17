"""Versioned, disease-specific prompts for frozen MedCLIP inference."""

from __future__ import annotations

from typing import Iterable

from .data import CLASS_NAMES

PROMPT_SET_VERSION = "clinical_v1"

# The first pair exactly preserves the Step 0/A single-template baseline.  The five-prompt condition
# uses all five pairs for the same disease and averages their logits.  Keeping
# explicit disease entries makes the protocol auditable and avoids silently
# treating a generic sentence template as disease-specific prompt design.
DISEASE_PROMPTS = {
    "Atelectasis": {
        "positive": [
            "A chest X-ray showing Atelectasis.",
            "The radiograph demonstrates an atelectatic pulmonary opacity.",
            "There is lung volume loss consistent with atelectasis.",
            "This chest radiograph is positive for atelectasis.",
            "Findings include linear or dependent atelectatic change.",
        ],
        "negative": [
            "No evidence of Atelectasis on the chest X-ray.",
            "The radiograph shows no focal or subsegmental atelectasis.",
            "There is no atelectatic opacity or associated lung volume loss.",
            "This chest radiograph is negative for atelectasis.",
            "The lungs show no findings of collapse or underexpansion.",
        ],
    },
    "Cardiomegaly": {
        "positive": [
            "A chest X-ray showing Cardiomegaly.",
            "The radiograph demonstrates an enlarged cardiac silhouette.",
            "The cardiac size is increased on this chest radiograph.",
            "This chest X-ray is positive for cardiomegaly.",
            "The cardiomediastinal silhouette is abnormally enlarged.",
        ],
        "negative": [
            "No evidence of Cardiomegaly on the chest X-ray.",
            "The cardiac silhouette is not enlarged.",
            "The heart size is within normal limits on this radiograph.",
            "This chest X-ray is negative for cardiomegaly.",
            "The cardiomediastinal silhouette is normal in size.",
        ],
    },
    "Consolidation": {
        "positive": [
            "A chest X-ray showing Consolidation.",
            "The radiograph demonstrates a focal air-space consolidation.",
            "There is dense pulmonary air-space opacity consistent with consolidation.",
            "This chest X-ray is positive for consolidation.",
            "Findings include a lobar or segmental consolidative opacity.",
        ],
        "negative": [
            "No evidence of Consolidation on the chest X-ray.",
            "The radiograph shows no focal air-space consolidation.",
            "There is no dense consolidative pulmonary opacity.",
            "This chest X-ray is negative for consolidation.",
            "No lobar or segmental consolidation is identified.",
        ],
    },
    "Edema": {
        "positive": [
            "A chest X-ray showing Edema.",
            "The radiograph demonstrates interstitial or alveolar pulmonary edema.",
            "There is pulmonary vascular congestion consistent with edema.",
            "This chest X-ray is positive for pulmonary edema.",
            "Bilateral perihilar opacities are compatible with pulmonary edema.",
        ],
        "negative": [
            "No evidence of Edema on the chest X-ray.",
            "The radiograph shows no interstitial or alveolar edema.",
            "There is no pulmonary vascular congestion.",
            "This chest X-ray is negative for pulmonary edema.",
            "No perihilar edema pattern is identified.",
        ],
    },
    "Pleural Effusion": {
        "positive": [
            "A chest X-ray showing Pleural Effusion.",
            "The radiograph demonstrates pleural fluid.",
            "There is blunting of a costophrenic angle consistent with pleural effusion.",
            "This chest X-ray is positive for pleural effusion.",
            "Findings include a layering or dependent pleural effusion.",
        ],
        "negative": [
            "No evidence of Pleural Effusion on the chest X-ray.",
            "The radiograph shows no pleural fluid.",
            "The costophrenic angles are sharp without pleural effusion.",
            "This chest X-ray is negative for pleural effusion.",
            "No layering or dependent pleural fluid is identified.",
        ],
    },
}


def build_prompts(
    class_names: Iterable[str] = CLASS_NAMES,
    *,
    templates: int = 1,
) -> dict[str, dict[str, list[str]]]:
    class_names = list(class_names)
    if templates not in (1, 5):
        raise ValueError("templates must be 1 or 5")
    missing = [name for name in class_names if name not in DISEASE_PROMPTS]
    if missing:
        raise ValueError(f"No {PROMPT_SET_VERSION} prompts for classes: {missing}")
    return {
        disease: {
            "positive": DISEASE_PROMPTS[disease]["positive"][:templates],
            "negative": DISEASE_PROMPTS[disease]["negative"][:templates],
        }
        for disease in class_names
    }

