"""Fixed positive/negative prompt templates for the five classes."""

from __future__ import annotations

from typing import Iterable

from .data import CLASS_NAMES

SINGLE_POSITIVE = "A chest X-ray showing {disease}."
SINGLE_NEGATIVE = "No evidence of {disease} on the chest X-ray."

POSITIVE_TEMPLATES = [
    "A chest X-ray showing {disease}.",
    "The radiograph demonstrates {disease}.",
    "There is evidence of {disease} on this chest radiograph.",
    "This chest X-ray is positive for {disease}.",
    "Findings are consistent with {disease} on the chest X-ray.",
]
NEGATIVE_TEMPLATES = [
    "No evidence of {disease} on the chest X-ray.",
    "The radiograph shows no {disease}.",
    "There is no evidence of {disease} on this chest radiograph.",
    "This chest X-ray is negative for {disease}.",
    "The chest X-ray does not show {disease}.",
]


def build_prompts(
    class_names: Iterable[str] = CLASS_NAMES,
    *,
    templates: int = 1,
) -> dict[str, dict[str, list[str]]]:
    if templates not in (1, 5):
        raise ValueError("templates must be 1 or 5")
    positives = POSITIVE_TEMPLATES[:templates]
    negatives = NEGATIVE_TEMPLATES[:templates]
    return {
        disease: {
            "positive": [template.format(disease=disease) for template in positives],
            "negative": [template.format(disease=disease) for template in negatives],
        }
        for disease in class_names
    }

