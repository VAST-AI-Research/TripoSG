import tempfile
import unittest
from pathlib import Path

from PIL import Image

from character_sheet import (
    crop_reference,
    normalize_bbox,
    validate_bbox,
    validate_reference_regions,
)


class CharacterSheetTests(unittest.TestCase):
    def test_normalize_drag_coordinates(self):
        self.assertEqual(
            normalize_bbox((80, 60), (20, 10), (100, 80)),
            [0.2, 0.125, 0.8, 0.75],
        )

    def test_crop_uses_normalized_source_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sheet.png"
            Image.new("RGB", (100, 80), "white").save(path)
            crop = crop_reference(path, [0.2, 0.25, 0.7, 0.75])
            self.assertEqual(crop.size, (50, 40))

    def test_primary_must_be_a_full_body_view(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sheet.png"
            Image.new("RGB", (8, 8), "white").save(path)
            region = {
                "id": "ref-001",
                "path": str(path),
                "view": "eyes_detail",
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "primary": True,
            }
            with self.assertRaises(ValueError):
                validate_reference_regions([region])

    def test_only_one_primary_view_is_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sheet.png"
            Image.new("RGB", (8, 8), "white").save(path)
            regions = [
                {
                    "id": f"ref-{view}",
                    "path": str(path),
                    "view": view,
                    "bbox": [0.1, 0.1, 0.9, 0.9],
                    "primary": True,
                }
                for view in ("front", "side")
            ]
            with self.assertRaises(ValueError):
                validate_reference_regions(regions)

    def test_rejects_out_of_bounds_and_zero_area_boxes(self):
        for bbox in ([0, 0, 1.1, 1], [0.5, 0.5, 0.5, 0.8], [0, 0, float("nan"), 1]):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                validate_bbox(bbox)


if __name__ == "__main__":
    unittest.main()
