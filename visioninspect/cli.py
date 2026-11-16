"""Inspect supplied PGM images or generate an independently testable demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import inspect, pixel_metrics
from .io import read_pgm, write_pgm, write_png
from .synthetic import synthetic_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-based surface inspection")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--demo", action="store_true", help="Generate synthetic defects")
    parser.add_argument("--out", type=Path, default=Path("inspection-output"))
    parser.add_argument("--min-difference", type=float, default=25)
    parser.add_argument("--min-area", type=int, default=8)
    args = parser.parse_args()
    if args.demo:
        if args.reference or args.current:
            parser.error("--demo cannot be combined with image paths")
        reference, current, truth = synthetic_case()
    elif args.reference and args.current:
        reference, current = read_pgm(args.reference), read_pgm(args.current)
        truth = None
    else:
        parser.error("Use --demo or both --reference and --current")
    result = inspect(reference, current, min_difference=args.min_difference,
                     min_area=args.min_area)
    args.out.mkdir(parents=True, exist_ok=True)
    overlay = np.repeat(current[:, :, None], 3, axis=2)
    overlay[result.mask] = np.array([255, 0, 40], dtype=np.uint8)
    write_png(args.out / "reference.png", reference)
    write_png(args.out / "current.png", current)
    write_png(args.out / "overlay.png", overlay)
    write_pgm(args.out / "detected-mask.pgm", result.mask.astype(np.uint8) * 255)
    report = result.as_dict()
    if truth is not None:
        report["synthetic_pixel_metrics"] = pixel_metrics(result.mask, truth)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Images and report saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
