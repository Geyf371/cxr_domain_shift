"""Download pinned official assets once; never download patient data."""
from __future__ import annotations
import argparse
import importlib.util
import importlib.metadata
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_loader import MEDCLIP_COMMIT, BERT_COMMIT, SWIN_COMMIT, WEIGHT_URL
from src.runtime import model_cache, read_config, sha256_file, write_json

def download(url, target):
    target = Path(target)
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {target.name}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Step0-MedCLIP/1.0"})
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as dest:
        shutil.copyfileobj(source, dest, length=1024 * 1024)
    temporary.replace(target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step0.local.yaml")
    parser.add_argument("--install-deps", action="store_true")
    args = parser.parse_args()
    try:
        tokenizer_version = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        tokenizer_version = None
    if tokenizer_version and not tokenizer_version.startswith(("0.11.", "0.12.", "0.13.")):
        raise RuntimeError(
            f"This environment has tokenizers {tokenizer_version}, incompatible with official transformers 4.24. "
            "Create and activate a clean venv, install requirements.txt, then run this command again. "
            "Do not install transformers or tokenizers separately; prepare_model.py supplies the pinned runtime."
        )
    cache = model_cache(read_config(args.config))
    cache.mkdir(parents=True, exist_ok=True)
    if args.install_deps:
        # Official transformers 4.24 supports slow BertTokenizer without the
        # optional Rust tokenizers package (which has no old Python 3.12 wheel).
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "--upgrade",
                        "--target", str(cache / "python"), "transformers==4.24.0", "huggingface-hub==0.25.2"], check=True)
        for name, requirement in (("nltk", "nltk==3.9.1"), ("regex", "regex"), ("requests", "requests"),
                                  ("tqdm", "tqdm"), ("filelock", "filelock"), ("packaging", "packaging")):
            if importlib.util.find_spec(name) is None:
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--target",
                                str(cache / "python"), requirement], check=True)
    assets = {}
    for name in ("__init__.py", "modeling_medclip.py", "dataset.py", "prompts.py", "constants.py"):
        relative = f"official/medclip/{name}"
        download(f"https://raw.githubusercontent.com/RyanWangZf/MedCLIP/{MEDCLIP_COMMIT}/medclip/{name}", cache / relative)
        assets[relative] = sha256_file(cache / relative)
    for repo, revision, folder, names in (
        ("emilyalsentzer/Bio_ClinicalBERT", BERT_COMMIT, "bert", ("config.json", "vocab.txt")),
        ("microsoft/swin-tiny-patch4-window7-224", SWIN_COMMIT, "swin", ("config.json",)),
    ):
        for name in names:
            relative = f"{folder}/{name}"
            download(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", cache / relative)
            assets[relative] = sha256_file(cache / relative)
    write_json(cache / "bert/tokenizer_config.json", {"do_lower_case": True, "model_max_length": 77,
                                                   "tokenizer_class": "BertTokenizer"})
    assets["bert/tokenizer_config.json"] = sha256_file(cache / "bert/tokenizer_config.json")
    weight = cache / "pytorch_model.bin"
    if not weight.is_file():
        if shutil.disk_usage(cache).free < 1600 * 1024**2:
            raise OSError("Need at least 1.6 GiB free in model.cache_dir for download and extraction")
        archive = cache / "medclip-vit-pretrained.zip"
        download(WEIGHT_URL, archive)
        with zipfile.ZipFile(archive) as packed:
            members = [x for x in packed.namelist() if Path(x).name == "pytorch_model.bin"]
            if len(members) != 1:
                raise ValueError("Official archive must contain exactly one pytorch_model.bin")
            with packed.open(members[0]) as source, weight.with_suffix(".part").open("wb") as dest:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
            weight.with_suffix(".part").replace(weight)
        archive.unlink()
    assets["pytorch_model.bin"] = sha256_file(weight)
    write_json(cache / "model_manifest.json", {
        "medclip_repository": "https://github.com/RyanWangZf/MedCLIP", "medclip_commit": MEDCLIP_COMMIT,
        "weights_url": WEIGHT_URL, "weights_sha256": assets["pytorch_model.bin"],
        "vision_backbone": "microsoft/swin-tiny-patch4-window7-224", "swin_config_commit": SWIN_COMMIT,
        "text_backbone": "emilyalsentzer/Bio_ClinicalBERT", "bert_config_commit": BERT_COMMIT,
        "pretraining_data": ["MIMIC-CXR", "CheXpert"],
        "scope": "Downstream splits only; institutions may have been seen during pretraining.",
        "transformers": "4.24.0 (official processor; slow BertTokenizer)", "sha256": assets,
    })
    print(f"Prepared {cache}; run scripts/run_step0.py to verify real inference.")

if __name__ == "__main__":
    main()
