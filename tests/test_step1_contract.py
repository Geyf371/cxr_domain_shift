"""Step 1 Member C prompt, resource, and artifact-contract checks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_member_c_smoke import select_patients
from src.artifact_contract import validate_step1_artifacts
from src.data import CLASS_NAMES
from src.prompts import DISEASE_PROMPTS, PROMPT_SET_VERSION, build_prompts


def test_disease_specific_prompt_protocol():
    prompts = build_prompts(templates=5)
    assert PROMPT_SET_VERSION == "clinical_v1"
    assert list(prompts) == CLASS_NAMES
    assert set(DISEASE_PROMPTS) == set(CLASS_NAMES)
    for disease in CLASS_NAMES:
        assert len(prompts[disease]["positive"]) == len(prompts[disease]["negative"]) == 5
        assert build_prompts(templates=1)[disease]["positive"] == prompts[disease]["positive"][:1]
        assert build_prompts(templates=1)[disease]["negative"] == prompts[disease]["negative"][:1]
        assert all(disease.casefold().split()[0] in text.casefold() or disease == "Edema"
                   for text in prompts[disease]["positive"][:1])
    assert len({tuple(value["positive"]) for value in prompts.values()}) == len(CLASS_NAMES)


def test_patient_smoke_selection_is_deterministic_and_patient_level():
    rows = []
    for patient in ("p3", "p1", "p2"):
        for image in range(2):
            row = {"patient_id": patient, "study_id": f"s{image}", "image_id": f"i{image}"}
            row.update({f"mask_{name}": int(patient != "p1" or image == 1) for name in CLASS_NAMES})
            rows.append(row)
    selected = select_patients(pd.DataFrame(rows), 2)
    assert selected.patient_id.astype(str).unique().tolist() == ["p1", "p2"]
    assert len(selected) == 4  # every image belonging to the two selected patients is retained


def test_step1_artifact_contract(tmp_path):
    n, dim = 3, 4
    identity = {"method": "pretrained", "budget": "10pct", "seed": 42,
                "domain": "mimic", "split": "test", "scope": "smoke128"}
    rows = []
    for index in range(n):
        row = {**identity, "patient_id": f"p{index}", "image_id": f"i{index}", "sample_id": f"s{index}"}
        for name in CLASS_NAMES:
            row.update({f"label_{name}": index % 2, f"mask_{name}": 1,
                        f"logit_{name}": float(index), f"prob_{name}": float(1 / (1 + np.exp(-index)))})
        rows.append(row)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions, index=False)
    v = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    q = np.arange(n * 5, dtype=np.float32).reshape(n, 5)
    features = tmp_path / "features.npz"
    labels = np.repeat(np.asarray([0, 1, 0])[:, None], 5, axis=1)
    np.savez(features, v=v, q=q, z=np.concatenate([v, q], axis=1),
             sample_id=[f"s{i}" for i in range(n)], labels=labels,
             mask=np.ones((n, 5)), classes=CLASS_NAMES)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        **identity,
        "total_parameters": 100,
        "trainable_parameters": 0,
        "device": "cpu",
        "inference_samples": n,
        "inference_batches": 1,
        "inference_model_seconds": 0.1,
        "inference_end_to_end_seconds": 0.2,
        "peak_gpu_memory_mb": None,
        "peak_gpu_memory_supported": False,
    }), encoding="utf-8")
    report = validate_step1_artifacts(predictions=predictions, features=features, metadata=metadata)
    assert report["valid"] and report["feature_shapes"]["z"] == [n, dim + 5]

    np.savez(features, v=v, q=q[:, :4], z=np.concatenate([v, q[:, :4]], axis=1),
             sample_id=[f"s{i}" for i in range(n)], labels=labels,
             mask=np.ones((n, 5)), classes=CLASS_NAMES)
    with pytest.raises(ValueError, match="q=\\[samples, 5\\]"):
        validate_step1_artifacts(predictions=predictions, features=features, metadata=metadata)
