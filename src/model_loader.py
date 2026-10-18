"""Load pinned official MedCLIP modules and weights offline, on CPU or CUDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import sys
import os
import torch
from torch import nn
from .runtime import choose_device, model_cache, sha256_file

MEDCLIP_COMMIT = "9c3396f20d5d54e4fae241b8cb06ca45848e98c9"
BERT_COMMIT = "d5892b39a4adaed74b92212a44081509db72f87b"
SWIN_COMMIT = "d00d478bfaf1f417d34a1186673ad4b4be264c51"
WEIGHT_URL = "https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip"


@dataclass
class MedCLIPBundle:
    model: Any
    processor: Any
    vision_feature_dim: int
    provenance: dict


def load_medclip(config) -> MedCLIPBundle:
    cache = model_cache(config)
    manifest_path = cache / "model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run python scripts/prepare_model.py --config config/step0.local.yaml --install-deps first")
    provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
    if provenance.get("medclip_commit") != MEDCLIP_COMMIT:
        raise ValueError("Model cache uses a different official MedCLIP commit")
    for name, expected in provenance["sha256"].items():
        if sha256_file(cache / name) != expected:
            raise ValueError(f"Model asset checksum mismatch: {name}")
    sys.path.insert(0, str(cache / "python"))
    sys.path.insert(0, str(cache / "official"))
    os.environ["TRANSFORMERS_CACHE"] = str(cache / "hf_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import transformers
        from transformers import AutoModel, BertConfig, SwinConfig, BertTokenizer, CLIPProcessor
        # Old AutoTokenizer imports the optional Rust backend unconditionally.
        # The official BERT vocabulary only needs the supported slow tokenizer.
        # Import CLIPProcessor first (its legacy ProcessorMixin reloads the
        # transformers module), then bind the slow tokenizer for official code.
        import transformers
        transformers.AutoTokenizer = BertTokenizer
        from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPTextModel, MedCLIPProcessor
    except ImportError as exc:
        raise ImportError("Run prepare_model.py --install-deps with this same Python interpreter") from exc

    # Keep official forward implementations. Initialize from local configs
    # before strict checkpoint restoration to avoid redundant pretrained files.
    vision = MedCLIPVisionModelViT.__new__(MedCLIPVisionModelViT)
    nn.Module.__init__(vision)
    vision.vit_type = str(cache / "swin")
    vision.model = AutoModel.from_config(SwinConfig.from_pretrained(cache / "swin", local_files_only=True))
    vision.projection_head = nn.Linear(768, 512, bias=False)
    text = MedCLIPTextModel.__new__(MedCLIPTextModel)
    nn.Module.__init__(text)
    text.bert_type, text.last_n_layer = str(cache / "bert"), 4
    bert_config = BertConfig.from_pretrained(cache / "bert", local_files_only=True)
    bert_config.output_hidden_states = True
    text.model = AutoModel.from_config(bert_config)
    text.tokenizer = BertTokenizer.from_pretrained(cache / "bert", local_files_only=True)
    text.projection_head = nn.Linear(768, 512, bias=False)
    model = MedCLIPModel.__new__(MedCLIPModel)
    nn.Module.__init__(model)
    model.vision_model, model.text_model = vision, text
    model.logit_scale = nn.Parameter(torch.zeros(()))
    weights = torch.load(cache / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    from medclip.dataset import MedCLIPFeatureExtractor
    processor = MedCLIPProcessor.__new__(MedCLIPProcessor)
    processor.feature_extractor = MedCLIPFeatureExtractor()
    processor.tokenizer = text.tokenizer
    processor.tokenizer.model_max_length = 77
    processor.current_processor = processor.feature_extractor
    device = choose_device(config.get("model", {}).get("device", "auto"))
    model.requires_grad_(False)
    model.to(device).eval()
    return MedCLIPBundle(model, processor, int(vision.projection_head.out_features), provenance)

