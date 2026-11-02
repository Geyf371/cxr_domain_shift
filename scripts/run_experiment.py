"""One configurable train/eval entry point; later steps supply method factories."""
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_manifest, assert_patient_disjoint, make_loader
from src.model_loader import load_medclip
from src.model_adapter import build_model
from src.evaluator import evaluate_model
from src.trainer import fit_source, load_checkpoint
from src.runtime import read_config, project_path, output_dir, seed_everything, write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step0.local.yaml")
    parser.add_argument("--action", choices=["train", "eval"], required=True)
    parser.add_argument("--domain", choices=["mimic", "chexpert"], default="mimic")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint")
    parser.add_argument("--limit", type=int, help="Diagnostic subset only; never labelled as a full experiment")
    parser.add_argument("--patient-limit", type=int, help="Deterministic diagnostic patient limit; keeps all selected-patient images")
    args = parser.parse_args()
    config = read_config(args.config)
    exp = config.setdefault("experiment", {})
    seed, budget, batch_size = int(exp.get("seed", 42)), exp.get("budget", "10pct"), int(exp.get("batch_size", 4))
    exp.update(seed=seed, budget=budget, method=exp.get("method", "pretrained"))
    seed_everything(seed)
    torch.set_num_threads(int(config.get("model", {}).get("cpu_threads", 4)))
    if args.domain == "chexpert" and (args.action == "train" or args.split != "test"):
        raise ValueError("CheXpert is test-only")
    if args.limit is not None and args.patient_limit is not None:
        raise ValueError("Use either --limit or --patient-limit, not both")
    if args.limit is not None and (args.limit < 1 or args.action == "train"):
        raise ValueError("--limit is a positive evaluation-only diagnostic option")
    if args.patient_limit is not None and args.patient_limit < 1:
        raise ValueError("--patient-limit must be positive")
    scope = f"smoke{args.patient_limit}" if args.patient_limit else "diagnostic" if args.limit else "experiment"
    exp["scope"] = scope
    data_dir = project_path(config["data"].get("data_dir", "data"))
    frame = load_manifest(data_dir / f"{args.domain}_manifest.csv")
    if set(frame.domain) != {args.domain}:
        raise ValueError("Manifest domain mismatch")
    assert_patient_disjoint([group for _, group in frame.groupby("split")])
    bundle = load_medclip(config)
    model = build_model(bundle, config)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, next(model.parameters()).device)
    out = output_dir(config)
    def loader(rows, shuffle=False):
        if rows.empty:
            raise ValueError("Requested split/subset is empty")
        return make_loader(rows.sort_values(["patient_id", "study_id", "image_id"]), bundle.processor,
                           batch_size=batch_size, shuffle=shuffle, seed=seed)
    if args.action == "train":
        ids = pd.read_csv(data_dir / f"source_train_seed{seed}_{budget}.csv").patient_id.astype(str)
        train = frame[frame.split.eq("train") & frame.patient_id.astype(str).isin(ids)]
        if set(train.patient_id.astype(str)) != set(ids):
            raise ValueError("Training subset contains missing or non-train patients")
        validation = frame[frame.split.eq("val")]
        if args.patient_limit:
            train_ids = sorted(train.patient_id.astype(str).unique())[:args.patient_limit]
            val_ids = sorted(validation.patient_id.astype(str).unique())[:args.patient_limit]
            if len(train_ids) != args.patient_limit or len(val_ids) != args.patient_limit:
                raise ValueError("Not enough source train/val patients for the requested smoke test")
            train = train[train.patient_id.astype(str).isin(train_ids)]
            validation = validation[validation.patient_id.astype(str).isin(val_ids)]
        result = fit_source(model, loader(train, True), loader(validation), config, out, bundle.provenance)
        prefix = "" if scope == "experiment" else f"{scope}_"
        write_json(out / f"training_{prefix}{exp['method']}_{budget}_seed{seed}.json", result)
    else:
        rows = frame[frame.split.eq(args.split)]
        if args.patient_limit:
            patient_ids = sorted(rows.patient_id.astype(str).unique())[:args.patient_limit]
            rows = rows[rows.patient_id.astype(str).isin(patient_ids)]
            if rows.patient_id.astype(str).nunique() != args.patient_limit:
                raise ValueError(f"Requested {args.patient_limit} patients but the split provides fewer")
        elif args.limit:
            rows = rows.sort_values(["patient_id", "study_id", "image_id"]).head(args.limit)
        result = evaluate_model(model, loader(rows), domain=args.domain, split=args.split,
                                method=exp["method"] + ("_5prompts" if exp.get("prompt_templates") == 5 else ""),
                                budget=budget, seed=seed, output_dir=out, threshold=float(exp.get("threshold", 0.5)),
                                scope=scope, provenance=bundle.provenance)
        print(result["paths"])

if __name__ == "__main__":
    main()
