"""Shared data and label utilities for the two-domain experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# Normalize case, spaces and underscores, and explicitly support common aliases.
LABEL_ALIASES = {
    "Atelectasis": ["Atelectasis", "Atelectases"],
    "Cardiomegaly": ["Cardiomegaly", "Enlarged heart"],
    "Consolidation": ["Consolidation"],
    "Edema": ["Edema", "Pulmonary edema", "Oedema"],
    "Pleural Effusion": ["Pleural Effusion", "Pleural Effusions", "Effusion"],
}

def resolve_label_columns(frame):
    def key(value):
        return "".join(c for c in str(value).casefold() if c.isalnum())
    lookup = {key(column): column for column in frame.columns}
    mapping = {}
    for canonical, aliases in LABEL_ALIASES.items():
        found = [lookup[key(alias)] for alias in aliases if key(alias) in lookup]
        if not found:
            raise ValueError(f"Missing required label column {canonical}; accepted aliases: {aliases}")
        mapping[canonical] = found[0]
    return mapping


def normalise_label(value: object) -> tuple[int, int]:
    """Map a CheXpert/MIMIC label to (binary_label, valid_mask)."""

    if pd.isna(value) or str(value).strip() == "":
        return 0, 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0, 0
    if number in (0.0, 1.0):
        return int(number), 1
    return 0, 0


def build_mask_columns(class_names: Iterable[str] = CLASS_NAMES) -> list[str]:
    return [f"mask_{name}" for name in class_names]


def write_class_frequencies(frame: pd.DataFrame, path: str | Path) -> None:
    rows = []
    for (domain, split), group in frame.groupby(["domain", "split"], dropna=False):
        for class_name in CLASS_NAMES:
            mask = group[f"mask_{class_name}"].astype(bool)
            labels = group.loc[mask, f"label_{class_name}"].astype(int)
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "class_name": class_name,
                    "positive_count": int((labels == 1).sum()),
                    "negative_count": int((labels == 0).sum()),
                    "unknown_count": int((~mask).sum()),
                    "no_valid_sample_count": int((group[[f"mask_{c}" for c in CLASS_NAMES]].sum(axis=1) == 0).sum()),
                }
            )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a generated manifest and verify the required identity columns."""

    frame = pd.read_csv(path)
    required = {"domain", "patient_id", "image_path", "split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return frame


class ManifestDataset:
    """Small PyTorch-compatible dataset for a generated manifest.

    The processor is optional so the indexing and label pipeline can be tested
    without loading MedCLIP. When supplied, it is called with one PIL image.
    """

    def __init__(self, frame: pd.DataFrame, processor=None):
        self.frame = frame.reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        if self.processor is not None:
            image_input = self.processor(images=image, return_tensors="pt")
        else:
            image_input = image
        labels = [int(row[f"label_{name}"]) for name in CLASS_NAMES]
        masks = [int(row[f"mask_{name}"]) for name in CLASS_NAMES]
        return {
            "image": image_input,
            "patient_id": str(row["patient_id"]),
            "image_id": str(row["image_id"]),
            "sample_id": "|".join(str(row[name]) for name in ("domain", "patient_id", "study_id", "image_id")),
            "domain": str(row["domain"]),
            "split": str(row["split"]),
            "labels": labels,
            "mask": masks,
        }


def label_columns(prefix: str = "label_") -> list[str]:
    return [f"{prefix}{name}" for name in CLASS_NAMES]


class ManifestCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, items):
        import torch
        result = self.processor(images=[item["image"] for item in items], return_tensors="pt")
        result["labels"] = torch.tensor([item["labels"] for item in items], dtype=torch.float32)
        result["mask"] = torch.tensor([item["mask"] for item in items], dtype=torch.bool)
        for name in ("patient_id", "image_id", "sample_id", "domain", "split"):
            result[name] = [item[name] for item in items]
        return result


def make_loader(frame, processor, batch_size=4, shuffle=False, seed=42):
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(ManifestDataset(frame), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=ManifestCollator(processor),
                      generator=torch.Generator().manual_seed(seed))


def mask_columns(prefix: str = "mask_") -> list[str]:
    return [f"{prefix}{name}" for name in CLASS_NAMES]


def assert_patient_disjoint(frames: Iterable[pd.DataFrame]) -> None:
    """Raise if any patient appears in more than one supplied split."""

    seen: set[str] = set()
    for frame in frames:
        patients = set(frame["patient_id"].astype(str))
        overlap = seen.intersection(patients)
        if overlap:
            raise ValueError(f"Patient overlap detected: {sorted(overlap)[:5]}")
        seen.update(patients)

