"""Reproducible synthetic surface with scratch and bright corrosion-like defect."""
from __future__ import annotations

import numpy as np


def synthetic_case(seed: int = 21, size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if size < 64:
        raise ValueError("Synthetic demo requires size >= 64")
    rng = np.random.default_rng(seed)
    base = np.clip(rng.normal(125, 5, (size, size)), 0, 255)
    current = base + 3 + rng.normal(0, 1, (size, size))
    truth = np.zeros((size, size), dtype=bool)
    y1, y2 = size // 7, size // 2
    x1, x2 = size // 3, size // 3 + 2
    truth[y1:y2, x1:x2] = True  # narrow scratch
    yy, xx = np.ogrid[:size, :size]
    disk = (xx - size * 0.72) ** 2 + (yy - size * 0.70) ** 2 <= (size * 0.055) ** 2
    truth |= disk
    current[y1:y2, x1:x2] -= 75
    current[disk] += 65
    return base.clip(0, 255).astype(np.uint8), current.clip(0, 255).astype(np.uint8), truth
