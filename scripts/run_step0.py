"""Verify one domain with 16 real MedCLIP images and export a shareable report."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CLASS_NAMES, load_manifest, assert_patient_disjoint, normalise_label, make_loader, write_class_frequencies
from src.evaluator import evaluate_model
from src.feature_analysis import compare_feature_caches
from src.hypothesis_metrics import compare_prompts
from src.model_adapter import MedCLIPAdapter
from src.model_interface import ModelBatchOutput
from src.model_loader import load_medclip
from src.runtime import ROOT, read_config, project_path, output_dir, seed_everything, write_json
from src.trainer import train_source_epoch, save_checkpoint, load_checkpoint


def code_fingerprint():
    digest = hashlib.sha256()
    files = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py"), ROOT / "config/step0.yaml",
                    ROOT / "docs/hypotheses.md"])
    for file in files:
        digest.update(file.relative_to(ROOT).as_posix().encode())
        # Normalize Git CRLF conversion across computers.
        digest.update(file.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return digest.hexdigest()


def data_protocol(config):
    """Share sampling settings without sharing machine paths or patient IDs."""
    return {"classes": config["classes"],
            "data": {key: config["data"][key] for key in
                     ("source_patients", "target_patients", "source_selection_seed", "split_seed")},
            "split": {key: config["split"][key] for key in
                      ("train_ratio", "val_ratio", "test_ratio", "subset_seeds", "budgets")}}


def audit_manifest(frame, config, domain):
    if set(frame.domain) != {domain} or frame.patient_id.duplicated().any():
        raise ValueError("Expected one image per unique patient in the requested domain")
    expected = int(config["data"]["source_patients" if domain == "mimic" else "target_patients"])
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} patients, found {len(frame)}")
    assert_patient_disjoint([g for _, g in frame.groupby("split")])
    if not frame.view_position.isin(["AP", "PA"]).all():
        raise ValueError("Manifest contains non-frontal images")
    if not frame.image_path.map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("Manifest contains missing images")
    for disease in CLASS_NAMES:
        mapped = frame[f"raw_{disease}"].map(normalise_label)
        if not np.array_equal(mapped.map(lambda x: x[0]), frame[f"label_{disease}"]):
            raise ValueError(f"Incorrect binary label mapping for {disease}")
        if not np.array_equal(mapped.map(lambda x: x[1]), frame[f"mask_{disease}"]):
            raise ValueError(f"Incorrect mask for {disease}")
    counts = {str(k): int(v) for k, v in frame.split.value_counts().items()}
    nested = {}
    if domain == "mimic":
        expected_counts = {"train": round(expected * .8), "val": round(expected * .1)}
        expected_counts["test"] = expected - sum(expected_counts.values())
        if counts != expected_counts:
            raise ValueError(f"Source split must be 8:1:1: {counts}")
        train_ids = set(frame.loc[frame.split.eq("train"), "patient_id"].astype(str))
        for seed in config["split"]["subset_seeds"]:
            previous = set()
            for budget in sorted(config["split"]["budgets"]):
                name = f"source_train_seed{seed}_{round(float(budget) * 100)}pct.csv"
                subset = pd.read_csv(project_path(config["data"].get("data_dir", "data")) / name)
                selected = set(subset.patient_id.astype(str))
                required = int(np.ceil(len(train_ids) * float(budget)))
                if len(subset) != required or len(selected) != required or not previous <= selected <= train_ids:
                    raise ValueError(f"Invalid or nonnested training subset: {name}")
                previous = selected
                nested[name] = len(selected)
    elif counts != {"test": expected}:
        raise ValueError("Target manifest must be entirely test-only")
    identity_columns = ["domain", "patient_id", "study_id", "image_id", "view_position", "split"]
    identity_columns += [f"{prefix}_{c}" for prefix in ("raw", "label", "mask") for c in CLASS_NAMES]
    digest = hashlib.sha256(frame[identity_columns].sort_values("patient_id").to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    no_valid = int((frame[[f"mask_{c}" for c in CLASS_NAMES]].sum(axis=1) == 0).sum())
    result = {"num_patients": len(frame), "num_images": len(frame), "num_evaluable_images": len(frame) - no_valid,
              "split_counts": counts, "all_selected_images_exist": True,
              "patients_disjoint": True, "labels_verified_against_raw": True, "nested_subsets": nested,
              "selection_sha256_without_local_paths": digest,
              "no_valid_sample_count": no_valid}
    if domain == "mimic":
        result["patient_buckets"] = sorted({x.split(":")[-1][:3] for x in frame.patient_id.astype(str)})
        result["sampling_scope"] = "Only the available patient buckets; not a random full-MIMIC cohort"
    return result


class TrainingLoopCheck(nn.Module):
    """Disposable tiny head used only to test the Step 0 shared training loop."""
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Linear(dim, 5)

    def forward_batch(self, batch):
        return ModelBatchOutput(logits=self.head(batch["z"]))


def check_training_loop(arrays, out):
    # Real source features, but no official Step 1 method, budget sweep or pilot.
    model = TrainingLoopCheck(arrays["z"].shape[1])
    initial = model.head.weight.detach().clone()
    batch = {name: torch.as_tensor(arrays[name], dtype=torch.float32) for name in ("z", "labels", "mask")}
    batch["domain"] = ["mimic"] * len(arrays["z"])
    stats = train_source_epoch(model, [batch], torch.optim.AdamW(model.parameters(), lr=1e-3), "cpu")
    if torch.equal(initial, model.head.weight) or stats["batches"] != 1:
        raise AssertionError("Source training loop did not update temporary head parameters")
    model.eval()
    with torch.no_grad():
        before = model.forward_batch(batch).logits.clone()
    checkpoint = out / "training_loop_check.pt"
    save_checkpoint(checkpoint, model, {"purpose": "step0_interface_check_only"})
    restored = TrainingLoopCheck(arrays["z"].shape[1])
    load_checkpoint(checkpoint, restored)
    restored.eval()
    with torch.no_grad():
        difference = float((before - restored.forward_batch(batch).logits).abs().max())
    if difference > 1e-7:
        raise AssertionError("Checkpoint roundtrip changed outputs")
    return dict(stats, parameters_updated=True, checkpoint_reload_max_abs=difference,
                purpose="Disposable interface check on 16 source features; not a Step 1 trained baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step0.local.yaml")
    parser.add_argument("--domain", choices=["mimic", "chexpert"], required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(42)
    torch.set_num_threads(int(config.get("model", {}).get("cpu_threads", 4)))
    out = output_dir(config)
    report_path = out / f"step0_{args.domain}.json"
    # Invalidate a previous success if any part of this new check fails.
    write_json(report_path, {"domain": args.domain, "domain_ready": False, "status": "running"})
    frame = load_manifest(project_path(config["data"].get("data_dir", "data")) / f"{args.domain}_manifest.csv")
    audit = audit_manifest(frame, config, args.domain)
    write_class_frequencies(frame, out / f"class_frequencies_{args.domain}.csv")
    split = "train" if args.domain == "mimic" else "test"
    valid = frame[[f"mask_{c}" for c in CLASS_NAMES]].sum(axis=1).gt(0)
    sample = frame[frame.split.eq(split) & valid].sort_values(["patient_id", "study_id", "image_id"]).head(16)
    if len(sample) != 16:
        raise ValueError("Need 16 evaluable images for the model check")
    print(f"Loading official MedCLIP for {args.domain}; checking 16 images", flush=True)
    bundle = load_medclip(config)
    adapter = MedCLIPAdapter(bundle, templates=1)
    trainable = [name for name, p in adapter.named_parameters() if p.requires_grad]
    if trainable:
        raise AssertionError("The pretrained model must be entirely frozen")
    versions = {name: p._version for name, p in bundle.model.named_parameters()}
    buffers = {name: value.detach().cpu().clone() for name, value in bundle.model.named_buffers()}
    loader = make_loader(sample, bundle.processor, batch_size=int(config.get("experiment", {}).get("batch_size", 4)))
    def evaluate(model, method):
        print(f"Checking {method}", flush=True)
        return evaluate_model(model, loader, domain=args.domain, split=split, method=method, budget="check16", seed=42,
                              output_dir=out, scope="step0_smoke", provenance=bundle.provenance)
    single = evaluate(adapter, "pretrained")
    repeated = evaluate(adapter, "pretrained_repeat")
    drift = compare_feature_caches(single["paths"]["features"], repeated["paths"]["features"], expect_frozen=True)
    p_columns = [f"prob_{c}" for c in CLASS_NAMES]
    probability_difference = float(np.max(np.abs(single["predictions"][p_columns].to_numpy() - repeated["predictions"][p_columns].to_numpy())))
    if not drift["within_tolerance"] or probability_difference > 1e-5:
        raise AssertionError("Repeated eval outputs exceed tolerance")
    five = evaluate(MedCLIPAdapter(bundle, templates=5), "pretrained_5prompts")
    prompt_image_drift = compare_feature_caches(single["paths"]["features"], five["paths"]["features"], expect_frozen=True)
    if not prompt_image_drift["within_tolerance"]:
        raise AssertionError("Prompt ensemble unexpectedly changed frozen image features")
    prompt_comparison = compare_prompts(single["predictions"], five["predictions"])
    write_json(out / f"h2_repeat_check_{args.domain}.json", drift)
    write_json(out / f"h3_interface_check_{args.domain}.json", prompt_comparison)
    arrays = single["feature_arrays"]
    d = arrays["v"].shape[1]
    if arrays["q"].shape != (16, 5) or arrays["z"].shape != (16, d + 5):
        raise AssertionError("Incorrect v/q/z dimensions")
    for key in ("v", "q", "z", "text_features"):
        if not np.isfinite(arrays[key]).all():
            raise AssertionError(f"Nonfinite {key} features")
    if not np.allclose(np.linalg.norm(arrays["v"], axis=1), 1.0, atol=1e-5):
        raise AssertionError("Image vectors must have unit norm")
    state_unchanged = all(p._version == versions[name] for name, p in bundle.model.named_parameters())
    state_unchanged &= all(torch.equal(value.detach().cpu(), buffers[name]) for name, value in bundle.model.named_buffers())
    if not state_unchanged:
        raise AssertionError("Frozen model weights or buffers changed during inference")
    training_check = check_training_loop(arrays, out) if args.domain == "mimic" else {"not_required": "target is test-only"}
    packages = {}
    for name in ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow", "PyYAML", "transformers", "huggingface-hub"):
        packages[name] = importlib.metadata.version(name)
    report = {"domain": args.domain, "domain_ready": True, "status": "passed", "code_sha256": code_fingerprint(),
              "classes": CLASS_NAMES, "data_protocol": data_protocol(config), "data_audit": audit,
              "environment": {"python": platform.python_version(), "packages": packages,
                              "device": str(next(bundle.model.parameters()).device), "cuda_available": torch.cuda.is_available()},
              "model": bundle.provenance, "model_feature_dim": d, "joint_feature_dim": d + 5,
              "model_check": {"num_samples": 16, "split": split, "probability_shape": [16, 5],
                              "feature_shapes": {key: list(arrays[key].shape) for key in ("v", "q", "z", "text_features")},
                              "trainable_parameters": trainable, "trainable_parameter_count": 0,
                              "total_parameter_count": sum(p.numel() for p in adapter.parameters()),
                              "repeat_probability_max_abs": probability_difference, "repeat_feature_drift": drift,
                              "five_prompt_image_drift": prompt_image_drift,
                              "model_weights_and_buffers_unchanged": bool(state_unchanged), "eval_mode": not adapter.training},
              "training_loop_check": training_check, "single_prompt_metrics": single["metrics"],
              "five_prompt_metrics": five["metrics"], "hypothesis_tools": {"h2_repeat_check": "passed", "h3_paired_comparison": "passed"},
              "scope": "Step 0 interface acceptance on 16 images, not full H1-H3 experiment results",
              "limitations": ["Medical pretraining includes both datasets; conclusions concern these downstream splits."]}
    write_json(report_path, report)
    print(f"PASS: {report_path}. Share this report and the aggregate class frequencies; image-level outputs stay local.")

if __name__ == "__main__":
    main()
