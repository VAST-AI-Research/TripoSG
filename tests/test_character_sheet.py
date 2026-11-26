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
from aethryx_engines import analyze_reference_image, engine_catalog, missing_reference_views


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

    def test_missing_views_are_not_predictions(self):
        result = missing_reference_views([
            {"view": "front", "confirmed": True},
            {"view": "side", "confirmed": False},
        ])
        self.assertEqual(result["confirmed_views"], ["front"])
        self.assertEqual(result["missing"], ["side", "back", "three_quarter"])

    def test_engine_catalog_keeps_multiview_explicitly_unavailable(self):
        engines = {engine.engine_id: engine for engine in engine_catalog()}
        self.assertTrue(engines["triposg"].single_view)
        self.assertFalse(engines["triposg"].multi_view)
        self.assertFalse(engines["multiview-character"].available)

    def test_reference_analysis_returns_an_explicit_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blank.png"
            Image.new("RGB", (64, 64), "white").save(path)
            result = analyze_reference_image(path)
            self.assertIn(result["status"], ("available", "completed", "unavailable"))
            self.assertIsInstance(result["regions"], list)


if __name__ == "__main__":
    unittest.main()
