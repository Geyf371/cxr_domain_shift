"""Regression checks for masking, sample alignment, and ensemble mathematics."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from src.data import CLASS_NAMES, normalise_label, resolve_label_columns
from src.metrics import compute_multilabel_metrics
from src.model_adapter import MedCLIPAdapter
from src.model_interface import ModelBatchOutput
from src.evaluator import evaluate_model
from src.feature_analysis import compare_feature_caches
from src.hypothesis_metrics import summarise_transfer, compare_prompts
from src.trainer import train_source_epoch, masked_bce_loss
from scripts.build_manifest import build_chexpert
from scripts.run_step0 import audit_manifest

def test_label_schema():
    assert normalise_label(-1) == (0, 0)
    assert normalise_label(np.nan) == (0, 0)
    frame = pd.DataFrame(columns=["atelectasis", "cardiomegaly", "consolidation", "pulmonary_edema", "Effusion"])
    assert resolve_label_columns(frame)["Pleural Effusion"] == "Effusion"
    with pytest.raises(ValueError, match="Missing required label"):
        resolve_label_columns(frame.drop(columns="Effusion"))

def test_masked_metrics():
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    p = np.array([[.9, .2], [.1, .3], [.0, .4], [.6, .8]])
    m = np.array([[1, 1], [1, 1], [0, 1], [0, 0]])
    result = compute_multilabel_metrics(y, p, m, ["a", "b"])
    assert result["macro_auc"] == 1.0
    assert result["macro_auc_valid_classes"] == result["macro_auprc_valid_classes"] == 1
    assert np.isnan(result["per_class"][1]["auprc"])
    assert result["no_valid_sample_count"] == 1
    logits = torch.tensor([[.3, .8]], requires_grad=True)
    loss = masked_bce_loss(logits, torch.tensor([[1., float("nan")]]), torch.tensor([[1, 0]]))
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[0, 1] == 0

class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(3, 4, bias=False)
        self.embedding = nn.Embedding(50, 4)
        self.logit_scale = nn.Parameter(torch.tensor(2.))

    def text_model(self, ids, attention):
        return self.embedding(ids[:, 0])

def fake_bundle():
    torch.manual_seed(8)
    tokenizer = lambda texts, **kwargs: {"input_ids": torch.arange(len(texts)).view(-1, 1), "attention_mask": torch.ones(len(texts), 1)}
    return SimpleNamespace(model=FakeBackbone(), processor=SimpleNamespace(tokenizer=tokenizer), vision_feature_dim=4)

def test_prompt_ensemble_is_mean_logits():
    model = MedCLIPAdapter(fake_bundle(), templates=5)
    output = model.forward_batch({"pixel_values": torch.tensor([[1., 2., 3.], [3., 1., 2.]])})
    assert output.logits.shape == (2, 5)
    assert output.joint_features_z.shape == (2, 9)
    assert torch.allclose(output.image_features_v.norm(dim=-1), torch.ones(2))
    assert torch.allclose(output.logits.sigmoid(), model.template_logits.mean(-1).sigmoid())
    assert not torch.allclose(output.logits.sigmoid(), model.template_logits.sigmoid().mean(-1))
    assert not any(p.requires_grad for p in model.parameters())

def batch():
    return {"pixel_values": torch.ones(3, 3), "labels": torch.tensor([[1., 0., 1., 0., 1.], [0., 1., 0., 1., 0.], [0., 0., 0., 0., 0.]]),
            "mask": torch.tensor([[1]*5, [1]*5, [0]*5]), "domain": ["mimic"]*3,
            "patient_id": ["a", "b", "c"], "image_id": ["ia", "ib", "ic"], "sample_id": ["sa", "sb", "sc"]}

def test_evaluator_filters_invalid_and_aligns_features(tmp_path):
    result = evaluate_model(MedCLIPAdapter(fake_bundle()), [batch()], domain="mimic", split="test", method="pretrained", budget="10pct", seed=42, output_dir=tmp_path)
    assert result["metrics"]["skipped_no_valid_samples"] == 1
    assert result["predictions"].sample_id.tolist() == ["sa", "sb"]
    with np.load(result["paths"]["features"], allow_pickle=False) as cache:
        assert cache["sample_id"].tolist() == ["sa", "sb"]
        assert len(cache["v"]) == 2
        assert cache["text_features"].shape == (5, 2, 1, 4)

def test_drift_aligns_and_rejects_different_images(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    features = np.eye(2, dtype=np.float32)
    np.savez(a, sample_id=["a", "b"], v=features, classes=CLASS_NAMES)
    np.savez(b, sample_id=["b", "a"], v=features[::-1], classes=CLASS_NAMES)
    assert compare_feature_caches(a, b, True)["max_absolute_difference"] == 0
    np.savez(b, sample_id=["b", "c"], v=features, classes=CLASS_NAMES)
    with pytest.raises(ValueError, match="same image"):
        compare_feature_caches(a, b)

class Trainable(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 5)

    def forward_batch(self, data):
        return ModelBatchOutput(logits=self.head(data["pixel_values"]))

def test_invalid_batches_skip_updates_and_target_training_rejected():
    model, data = Trainable(), batch()
    before = model.head.weight.detach().clone()
    data["mask"] = torch.zeros(3, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.1)
    result = train_source_epoch(model, [data], optimizer, "cpu")
    assert result["batches"] == 0 and result["skipped_no_valid_samples"] == 3
    assert torch.equal(before, model.head.weight)
    data["domain"] = ["chexpert"]*3
    with pytest.raises(ValueError, match="Only MIMIC"):
        train_source_epoch(model, [data], optimizer, "cpu")

def test_h1_gains_and_protocol_mismatch():
    rows = []
    for method, source, target in [("pretrained", .7, .6), ("new_method", .9, .5)]:
        for domain, value in [("mimic", source), ("chexpert", target)]:
            rows.append(dict(method=method, budget="10pct", seed=42, scope="experiment", split="test", domain=domain,
                             macro_auc=value, macro_f1=value, macro_auprc=value, threshold=.5, evaluation_set_sha256=domain))
    frame = pd.DataFrame(rows)
    result = summarise_transfer(frame).set_index("method").loc["new_method"]
    assert result["source_gain_macro_auc"] == pytest.approx(.2)
    assert result["target_gain_macro_auc"] == pytest.approx(-.1)
    assert result["transfer_drop_macro_auc"] == pytest.approx(.4)
    frame.loc[3, "evaluation_set_sha256"] = "different"
    with pytest.raises(ValueError, match="protocol mismatch"):
        summarise_transfer(frame)

def test_h3_pairs_images_and_checks_labels(tmp_path):
    result = evaluate_model(MedCLIPAdapter(fake_bundle()), [batch()], domain="mimic", split="test", method="pretrained", budget="10pct", seed=42, output_dir=tmp_path)
    a = result["predictions"]
    b = a.iloc[::-1].copy()
    assert compare_prompts(a, b)["delta_five_minus_single"]["macro_auc"] == 0
    b.loc[b.index[0], "label_Atelectasis"] = 1
    with pytest.raises(ValueError, match="labels/masks differ"):
        compare_prompts(a, b)


@pytest.mark.parametrize("root_has_prefix", [False, True])
def test_chexpert_layout_selection_and_audit(tmp_path, root_has_prefix):
    """Official prefixed CSV paths work from either data-root layout."""
    from PIL import Image
    folder = tmp_path / "CheXpert-v1.0-small"
    rows = []
    for patient in range(1, 6):
        for study, orientation, view in [(1, "Frontal", "AP"), (2, "Frontal", "PA"), (3, "Lateral", "")]:
            relative = f"train/patient{patient:05d}/study{study}/view1.jpg"
            image = folder / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (12, 16), color=patient).save(image)
            rows.append({"Path": f"CheXpert-v1.0-small/{relative}", "Frontal/Lateral": orientation, "AP/PA": view,
                         "Atelectasis": 1, "Cardiomegaly": 0, "Consolidation": -1, "Pulmonary_edema": np.nan, "Effusion": 1})
    pd.DataFrame(rows).to_csv(folder / "train.csv", index=False)
    root = tmp_path if root_has_prefix else folder
    frame = build_chexpert(root, n_patients=4, selection_seed=2026)
    assert frame.patient_id.is_unique and set(frame.split) == {"test"}
    assert set(frame.study_id) == {"study1"} and frame.image_exists.all()
    assert set(frame["mask_Consolidation"]) == {0} and set(frame["mask_Edema"]) == {0}
    assert set(frame["raw_Consolidation"]) == {-1} and set(frame["label_Pleural Effusion"]) == {1}
    audit = audit_manifest(frame, {"data": {"target_patients": 4}}, "chexpert")
    assert audit["num_patients"] == 4 and audit["labels_verified_against_raw"]
    assert frame.equals(build_chexpert(root, n_patients=4, selection_seed=2026))


def test_handoff_requires_both_reports_and_matching_sampling(tmp_path, monkeypatch):
    """No patient files are needed, but one domain alone cannot pass handoff."""
    import copy
    import json
    import sys
    from scripts import check_handoff
    from src.runtime import ROOT, read_config, write_json
    from scripts.run_step0 import code_fingerprint, data_protocol
    shared = read_config(ROOT / "config/step0.yaml")
    local = copy.deepcopy(shared)
    local["data"]["output_dir"] = str(tmp_path / "outputs")
    config_path = tmp_path / "local.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(local), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_handoff.py", "--config", str(config_path)])
    out = tmp_path / "outputs"
    report = {"domain": "mimic", "domain_ready": True, "status": "passed", "code_sha256": code_fingerprint(),
              "data_protocol": data_protocol(shared), "classes": CLASS_NAMES, "model_feature_dim": 512,
              "joint_feature_dim": 517, "model": {"sha256": {"test_asset": "test_digest"}, "weights_sha256": "test_digest"}}
    write_json(out / "step0_mimic.json", report)
    (out / "class_frequencies_mimic.csv").write_text("aggregate_only\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="step0_chexpert"):
        check_handoff.main()
    target = copy.deepcopy(report)
    target["domain"] = "chexpert"
    write_json(out / "step0_chexpert.json", target)
    (out / "class_frequencies_chexpert.csv").write_text("aggregate_only\n", encoding="utf-8")
    check_handoff.main()
    assert json.loads((out / "step0_handoff.json").read_text())["step0_ready"]
    target["data_protocol"]["data"]["target_patients"] = 16
    write_json(out / "step0_chexpert.json", target)
    with pytest.raises(ValueError, match="sampling settings"):
        check_handoff.main()
    assert not json.loads((out / "step0_handoff.json").read_text())["step0_ready"]
