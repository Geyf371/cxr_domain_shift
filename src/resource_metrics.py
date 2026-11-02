"""Comparable timing, memory, and parameter statistics for Step 1 methods."""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any, Iterator

import torch


def model_device(model: Any) -> torch.device:
    """Return the device of the first parameter/buffer, or CPU for empty models."""

    for value in list(model.parameters()) + list(model.buffers()):
        return value.device
    return torch.device("cpu")


def parameter_statistics(model: Any) -> dict[str, int]:
    parameters = list(model.parameters())
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad)),
    }


def synchronize(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_gpu_memory(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_memory_mb(device: torch.device | str) -> float | None:
    device = torch.device(device)
    if device.type != "cuda":
        return None
    synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


@contextmanager
def timed_device_block(device: torch.device | str) -> Iterator[dict[str, float]]:
    """Measure a device block, synchronizing CUDA so elapsed time is honest."""

    result: dict[str, float] = {}
    synchronize(device)
    started = time.perf_counter()
    try:
        yield result
    finally:
        synchronize(device)
        result["seconds"] = float(time.perf_counter() - started)


def inference_statistics(
    model: Any,
    *,
    samples: int,
    batches: int,
    model_seconds: float,
    end_to_end_seconds: float,
    peak_memory_mb: float | None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        **parameter_statistics(model),
        "device": str(model_device(model)),
        "inference_samples": int(samples),
        "inference_batches": int(batches),
        "inference_model_seconds": float(model_seconds),
        "inference_end_to_end_seconds": float(end_to_end_seconds),
        "inference_seconds_per_sample": float(model_seconds / samples) if samples else None,
        "inference_samples_per_second": float(samples / model_seconds) if model_seconds > 0 else None,
        "peak_gpu_memory_mb": peak_memory_mb,
        "peak_gpu_memory_supported": peak_memory_mb is not None,
        "timing_definition": (
            "model_seconds synchronizes CUDA and times forward_batch only; "
            "end_to_end_seconds also includes data loading and output collection"
        ),
    }
    return stats
