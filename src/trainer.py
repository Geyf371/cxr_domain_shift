"""Generic masked source-domain training loop for later model adapters."""

from __future__ import annotations

from typing import Callable
from pathlib import Path
import time

import torch
from .data import CLASS_NAMES
from .evaluator import evaluate_model, select_batch
from .model_interface import validate_model_batch_output
from .resource_metrics import (
    model_device,
    parameter_statistics,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    timed_device_block,
)


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if not valid.any():
        return logits.sum() * 0.0
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], labels.float()[valid])


def train_source_epoch(model, loader, optimizer, device: str, output_fn: Callable | None = None,
                       *, gradient_accumulation_steps: int = 1):
    if isinstance(gradient_accumulation_steps, bool) or not isinstance(gradient_accumulation_steps, int) or gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    model.train()
    pending_batches = pending_items = optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)

    def update(items):
        # Normalize by valid labels across the whole accumulation group,
        # including the final partial group, rather than by microbatch count.
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.div_(items)
                    if not torch.isfinite(parameter.grad).all():
                        raise ValueError("Non-finite source gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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
        output = output_fn(model, batch, device) if output_fn else model.forward_batch(batch)
        output = validate_model_batch_output(output, len(batch["domain"]))
        loss = masked_bce_loss(output.logits, batch["labels"].to(device), batch["mask"].to(device))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite source training loss")
        count = int(batch["mask"].sum())
        (loss * count).backward()
        pending_items += count
        pending_batches += 1
        if pending_batches == gradient_accumulation_steps:
            update(pending_items)
            optimizer_steps += 1
            pending_batches = pending_items = 0
        total += float(loss.detach().cpu()) * count
        valid_items += count
        batches += 1
    if pending_batches:
        update(pending_items)
        optimizer_steps += 1
    return {"loss": total / max(valid_items, 1), "batches": batches, "optimizer_steps": optimizer_steps,
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
    device = model_device(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best, stale, history = -float("inf"), 0, []
    training_seconds = 0.0
    training_peak_memory: float | None = None
    training_validation_peak_memory: float | None = None
    scope = str(exp.get("scope", "experiment"))
    prefix = "" if scope == "experiment" else f"{scope}_"
    checkpoint = Path(output_dir) / f"checkpoint_{prefix}{exp['method']}_{exp['budget']}_seed{exp['seed']}.pt"
    for epoch in range(int(cfg.get("epochs", 10))):
        reset_peak_gpu_memory(device)
        epoch_started = time.perf_counter()
        with timed_device_block(device) as timing:
            train_stats = train_source_epoch(
                model,
                train_loader,
                optimizer,
                str(device),
                gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            )
        training_seconds += timing["seconds"]
        epoch_peak = peak_gpu_memory_mb(device)
        if epoch_peak is not None:
            training_peak_memory = max(training_peak_memory or 0.0, epoch_peak)
            training_validation_peak_memory = max(training_validation_peak_memory or 0.0, epoch_peak)
        result = evaluate_model(model, val_loader, domain="mimic", split="val", method=exp["method"],
                                budget=exp["budget"], seed=int(exp["seed"]), output_dir=output_dir,
                                threshold=float(exp.get("threshold", 0.5)), scope=scope, provenance=provenance)
        validation_peak = result.get("resources", {}).get("peak_gpu_memory_mb")
        if validation_peak is not None:
            training_validation_peak_memory = max(training_validation_peak_memory or 0.0, validation_peak)
        score = result["metrics"]["macro_auc"]
        if not torch.isfinite(torch.tensor(score)):
            raise ValueError("Source validation has no defined class AUC; cannot select a checkpoint")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(dict(epoch=epoch + 1, **train_stats, val_macro_auc=score,
                            elapsed_seconds=time.perf_counter() - epoch_started))
        if score > best:
            best, stale = score, 0
            save_checkpoint(checkpoint, model, dict(epoch=epoch + 1, val_macro_auc=score, experiment=exp,
                training=cfg, lora=config.get('lora', {}), provenance=provenance or {},
                trainable_parameter_count=sum(p.numel() for p in params)))
        else:
            stale += 1
        if stale >= int(cfg.get("patience", 3)):
            break
    load_checkpoint(checkpoint, model, str(params[0].device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    resources = {
        **parameter_statistics(model),
        "device": str(device),
        "training_performed": True,
        "training_epochs": len(history),
        "training_seconds": training_seconds,
        "peak_gpu_memory_mb": training_peak_memory,
        "peak_gpu_memory_supported": training_peak_memory is not None,
        "timing_definition": "sum of source training epochs, including data loading and optimizer updates; validation excluded",
    }
    return {"history": history, "checkpoint": str(checkpoint), "best_val_macro_auc": best,
            "resources": resources,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int((training_validation_peak_memory or 0.0) * 1024 ** 2),
            "trainable_parameter_count": sum(p.numel() for p in params)}

