"""Local image anomaly detection with robust illumination correction."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Defect:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    area: int
    peak_difference: float

    def as_dict(self) -> dict:
        return {
            "bbox_xyxy_inclusive": [self.x_min, self.y_min, self.x_max, self.y_max],
            "area_pixels": self.area,
            "peak_difference": round(self.peak_difference, 3),
        }


@dataclass
class Inspection:
    mask: np.ndarray
    difference: np.ndarray
    defects: list[Defect]
    illumination_offset: float
    threshold: float

    def as_dict(self) -> dict:
        return {
            "illumination_offset": round(self.illumination_offset, 3),
            "threshold": round(self.threshold, 3),
            "defects": [d.as_dict() for d in self.defects],
            "total_defect_pixels": int(self.mask.sum()),
        }


def _components(binary: np.ndarray, magnitude: np.ndarray, min_area: int) -> tuple[np.ndarray, list[Defect]]:
    height, width = binary.shape
    cleaned = np.zeros_like(binary, dtype=bool)
    seen = np.zeros_like(binary, dtype=bool)
    defects: list[Defect] = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or seen[y, x]:
                continue
            seen[y, x] = True
            queue = deque([(y, x)])
            pixels = []
            while queue:
                row, col = queue.popleft()
                pixels.append((row, col))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    r, c = row + dy, col + dx
                    if 0 <= r < height and 0 <= c < width and binary[r, c] and not seen[r, c]:
                        seen[r, c] = True
                        queue.append((r, c))
            if len(pixels) >= min_area:
                rows = [row for row, _ in pixels]
                cols = [col for _, col in pixels]
                for row, col in pixels:
                    cleaned[row, col] = True
                defects.append(Defect(
                    min(cols), min(rows), max(cols), max(rows), len(pixels),
                    max(float(magnitude[row, col]) for row, col in pixels),
                ))
    return cleaned, sorted(defects, key=lambda d: (-d.area, d.y_min, d.x_min))


def inspect(
    reference: ArrayLike,
    current: ArrayLike,
    min_difference: float = 25.0,
    sigma_factor: float = 5.0,
    min_area: int = 8,
    illumination_correction: bool = True,
) -> Inspection:
    base = np.asarray(reference, dtype=np.float64)
    now = np.asarray(current, dtype=np.float64)
    if base.ndim != 2 or base.shape != now.shape or base.size == 0:
        raise ValueError("Equal-sized nonempty grayscale arrays required")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(now)):
        raise ValueError("All image values must be finite")
    if min_difference <= 0 or sigma_factor < 0 or min_area < 1:
        raise ValueError("Thresholds and minimum area must be valid")
    delta = now - base
    offset = float(np.median(delta)) if illumination_correction else 0.0
    residual = np.abs(delta - offset)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    adaptive_threshold = median + sigma_factor * 1.4826 * mad
    threshold = max(min_difference, adaptive_threshold)
    detected, defects = _components(residual >= threshold, residual, min_area)
    return Inspection(detected, residual, defects, offset, threshold)


def pixel_metrics(predicted: ArrayLike, truth: ArrayLike) -> dict:
    p, t = np.asarray(predicted, dtype=bool), np.asarray(truth, dtype=bool)
    if p.ndim != 2 or p.shape != t.shape:
        raise ValueError("Predicted and truth masks must share a 2D shape")
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(f1, 6), "true_positive": tp, "false_positive": fp, "false_negative": fn}
