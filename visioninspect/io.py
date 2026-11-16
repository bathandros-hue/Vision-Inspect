"""Minimal P2 PGM reader and independent PNG writer for dependency-light demos."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def read_pgm(path: Path) -> np.ndarray:
    tokens = []
    for line in path.read_text(encoding="ascii").splitlines():
        tokens.extend(line.split("#", 1)[0].split())
    if len(tokens) < 4 or tokens[0] != "P2":
        raise ValueError("Only ASCII P2 PGM images are supported")
    width, height, maximum = map(int, tokens[1:4])
    if width < 1 or height < 1 or maximum != 255:
        raise ValueError("Require positive dimensions and PGM max value 255")
    data = [int(x) for x in tokens[4:]]
    if len(data) != width * height or any(not 0 <= x <= 255 for x in data):
        raise ValueError("Invalid PGM pixels or image dimensions")
    return np.asarray(data, dtype=np.uint8).reshape(height, width)


def write_pgm(path: Path, image: np.ndarray) -> None:
    pixels = np.asarray(image)
    if pixels.ndim != 2 or pixels.size == 0 or np.any((pixels < 0) | (pixels > 255)):
        raise ValueError("PGM image requires 2D 0..255 values")
    height, width = pixels.shape
    lines = ["P2", f"{width} {height}", "255"]
    lines.extend(" ".join(str(int(x)) for x in row) for row in pixels)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_png(path: Path, image: np.ndarray) -> None:
    """Write 8-bit grayscale or RGB PNG without Pillow."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim not in (2, 3) or arr.size == 0:
        raise ValueError("PNG image must be uint8 grayscale or RGB")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError("RGB images need exactly three channels")
    height, width = arr.shape[:2]
    color_type = 0 if arr.ndim == 2 else 2
    rows = b"".join(b"\0" + arr[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
        chunk(b"IDAT", zlib.compress(rows, level=9)) + chunk(b"IEND", b"")
    )
