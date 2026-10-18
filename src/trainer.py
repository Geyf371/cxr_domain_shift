"""Generic masked source-domain training loop for later model adapters."""

from __future__ import annotations

from typing import Callable
from pathlib import Path
import copy

import torch
from .data import CLASS_NAMES
from .evaluator import evaluate_model, select_batch


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if not valid.any():
        return logits.sum() * 0.0
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], labels.float()[valid])


def train_source_epoch(model, loader, optimizer, device: str, output_fn: Callable | None = None):
    model.train()
    total = 0.0
    batches = 0
    skipped = 0
    valid_items = 0
    for batch in loader:
        if any(value != "mimic" for value in batch.get("domain", [])):
            raise ValueError("Only MIMIC source data may enter the training loop")
        keep = batch["mask"].bool().any(dim=1)
        skipped += int((~keep).sum())
        if not keep.any():
            continue
        batch = select_batch(batch, keep)
        optimizer.zero_grad(set_to_none=True)
        output = output_fn(model, batch, device) if output_fn else model.forward_batch(batch)
        loss = masked_bce_loss(output.logits, batch["labels"].to(device), batch["mask"].to(device))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite source training loss")
        loss.backward()
        optimizer.step()
        count = int(batch["mask"].sum())
        total += float(loss.detach().cpu()) * count
        valid_items += count
        batches += 1
    return {"loss": total / max(valid_items, 1), "batches": batches,
            "valid_label_items": valid_items, "skipped_no_valid_samples": skipped}


def save_checkpoint(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata, classes=CLASS_NAMES)}, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["metadata"].get("classes") != CLASS_NAMES:
        raise ValueError("Checkpoint class order does not match")
    model.load_state_dict(saved["state_dict"], strict=True)
    return saved["metadata"]


def fit_source(model, train_loader, val_loader, config, output_dir, provenance=None):
    """Shared source training/validation; Step 1 supplies a trainable adapter."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters; Step 1 must supply a trainable model factory")
    cfg, exp = config.get("training", {}), config.get("experiment", {})
    grouping = getattr(model, "optimizer_groups", None)
    optimizer = torch.optim.AdamW(grouping(config) if grouping else params,
                                 lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    best, stale, history = -float("inf"), 0, []
    checkpoint = Path(output_dir) / f"checkpoint_{exp['method']}_{exp['budget']}_seed{exp['seed']}.pt"
    for epoch in range(int(cfg.get("epochs", 10))):
        train_stats = train_source_epoch(model, train_loader, optimizer, str(params[0].device))
        result = evaluate_model(model, val_loader, domain="mimic", split="val", method=exp["method"],
                                budget=exp["budget"], seed=int(exp["seed"]), output_dir=output_dir,
                                threshold=float(exp.get("threshold", 0.5)), provenance=provenance)
        score = result["metrics"]["macro_auc"]
        if not torch.isfinite(torch.tensor(score)):
            raise ValueError("Source validation has no defined class AUC; cannot select a checkpoint")
        history.append(dict(epoch=epoch + 1, **train_stats, val_macro_auc=score))
        if score > best:
            best, stale = score, 0
            save_checkpoint(checkpoint, model, dict(epoch=epoch + 1, val_macro_auc=score, experiment=exp))
        else:
            stale += 1
        if stale >= int(cfg.get("patience", 3)):
            break
    load_checkpoint(checkpoint, model, str(params[0].device))
    return {"history": history, "checkpoint": str(checkpoint), "best_val_macro_auc": best}

