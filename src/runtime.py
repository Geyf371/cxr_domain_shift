"""Shared configuration, reproducibility and report helpers."""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from .data import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]

def read_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("classes") != CLASS_NAMES:
        raise ValueError("The five classes and their order must match src.data.CLASS_NAMES")
    return config

def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

def model_cache(config):
    value = config.get("model", {}).get("cache_dir")
    if not value:
        root = config["data"].get("mimic_root") or config["data"].get("chexpert_root")
        if not root:
            raise ValueError("Set model.cache_dir or one local data root")
        value = Path(root) / "step0_model"
    return project_path(value)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def choose_device(value="auto"):
    value = "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Set model.device to cpu or auto.")
    return str(value)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")

def output_dir(config):
    path = project_path(config["data"].get("output_dir", "outputs"))
    path.mkdir(parents=True, exist_ok=True)
    return path
