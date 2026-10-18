"""Feature cache helpers for v, prompt scores q, and joint features z."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_feature_cache(path: str | Path, **arrays: Any) -> None:
    """Save numeric feature arrays in a compact NumPy archive."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)

