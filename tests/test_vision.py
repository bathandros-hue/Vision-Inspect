import tempfile
import unittest
from pathlib import Path

import numpy as np

from visioninspect import inspect, pixel_metrics, synthetic_case
from visioninspect.io import read_pgm, write_pgm, write_png


class TestInspection(unittest.TestCase):
    def test_detect_synthetic_defects(self):
        reference, current, truth = synthetic_case()
        result = inspect(reference, current)
        metrics = pixel_metrics(result.mask, truth)
        self.assertGreater(metrics["f1"], 0.95)
        self.assertEqual(len(result.defects), 2)
        self.assertLess(abs(result.illumination_offset - 3), 1)

    def test_illumination_correction(self):
        base = np.ones((20, 20), dtype=np.uint8) * 100
        current = base.astype(float) + 30
        corrected = inspect(base, current, min_difference=20)
        not_corrected = inspect(base, current, min_difference=20, illumination_correction=False)
        self.assertEqual(corrected.mask.sum(), 0)
        self.assertEqual(not_corrected.mask.sum(), 400)

    def test_min_area_filter(self):
        base = np.zeros((20, 20), dtype=np.uint8)
        current = base.copy()
        current[4, 5] = 100
        self.assertEqual(inspect(base, current, min_area=2).mask.sum(), 0)

    def test_formats_and_png_signature(self):
        image = np.array([[0, 100, 255], [200, 40, 10]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            pgm, png = Path(temp) / "a.pgm", Path(temp) / "a.png"
            write_pgm(pgm, image)
            self.assertTrue(np.array_equal(read_pgm(pgm), image))
            write_png(png, image)
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_validation(self):
        with self.assertRaises(ValueError):
            inspect(np.zeros((2, 2)), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            inspect(np.zeros((2, 2)), np.zeros((2, 2)), min_area=0)
        with self.assertRaises(ValueError):
            synthetic_case(size=10)

    def test_mask_scoring(self):
        truth = np.array([[1, 0], [0, 1]], dtype=bool)
        predicted = np.array([[1, 1], [0, 0]], dtype=bool)
        result = pixel_metrics(predicted, truth)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
