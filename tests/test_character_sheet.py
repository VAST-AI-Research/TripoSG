import tempfile
import unittest
from pathlib import Path

from PIL import Image

from character_sheet import (
    crop_reference,
    normalize_bbox,
    select_primary_region,
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

    def test_synthetic_anime_reference_has_real_explicit_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "anime.png"
            image = Image.new("RGB", (240, 320), "#e8e1d8")
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            draw.ellipse((70, 30, 170, 150), fill="#f1c7a9", outline="#171717", width=3)
            draw.polygon(
                [(70, 70), (90, 20), (150, 20), (170, 70), (145, 50), (120, 42), (95, 52)],
                fill="#d4a0b4",
            )
            draw.ellipse((90, 80, 112, 96), fill="white", outline="#171717")
            draw.ellipse((128, 80, 150, 96), fill="white", outline="#171717")
            draw.ellipse((98, 84, 108, 94), fill="#55aadd")
            draw.ellipse((136, 84, 146, 94), fill="#55aadd")
            draw.rectangle((85, 150, 155, 280), fill="#6b78a8", outline="#171717", width=3)
            image.save(path)
            result = analyze_reference_image(path)
            self.assertIn(result["status"], ("available", "completed", "unavailable"))
            for region in result["regions"]:
                self.assertIn(region["origin"], (
                    "opencv-haar",
                    "mediapipe-face",
                    "mediapipe-pose",
                    "onnx-segmentation",
                ))
                self.assertFalse(region["confirmed"])

    def test_numpy_coordinates_are_accepted_by_bbox_validation(self):
        import numpy as np
        result = validate_bbox(np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32))
        self.assertTrue(all(type(value) is float for value in result))
        for actual, expected in zip(result, (0.1, 0.2, 0.8, 0.9)):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_primary_selection_across_multiple_images_is_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_a = root / "a.png"
            image_b = root / "b.png"
            Image.new("RGB", (32, 32), "white").save(image_a)
            Image.new("RGB", (32, 32), "white").save(image_b)
            regions = [
                {
                    "id": "side-a",
                    "path": str(image_a),
                    "view": "side",
                    "bbox": [0.1, 0.1, 0.9, 0.9],
                    "primary": False,
                    "confirmed": True,
                },
                {
                    "id": "front-b",
                    "path": str(image_b),
                    "view": "front",
                    "bbox": [0.1, 0.1, 0.9, 0.9],
                    "primary": True,
                    "confirmed": True,
                },
            ]
            selected = select_primary_region(regions)
            primaries = [region["id"] for region in selected if region["primary"]]
            self.assertEqual(primaries, ["front-b"])

    def test_mediapipe_tasks_api_is_detected_but_weights_remain_optional(self):
        import mediapipe as mp
        self.assertTrue(hasattr(mp, "tasks"))
        self.assertTrue(hasattr(mp.tasks, "vision"))
        self.assertTrue(hasattr(mp.tasks.vision, "PoseLandmarker"))

    def test_semantic_segmentation_is_unavailable_until_a_local_model_is_configured(self):
        import os
        os.environ.pop("AETHRYX_SEMANTIC_SEGMENTATION_MODEL", None)
        os.environ.pop("AETHRYX_SEMANTIC_SEGMENTATION_LABELS", None)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "character.png"
            Image.new("RGB", (128, 128), "white").save(path)
            result = analyze_reference_image(path)
            self.assertNotIn("onnx-segmentation", result.get("providers", []))

    def test_complete_rle_mask_is_validated_and_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "character.png"
            Image.new("RGB", (32, 32), "white").save(path)
            region = {
                "id": "ref-mask",
                "path": str(path),
                "view": "other",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "primary": False,
                "origin": "onnx-segmentation",
                "confirmed": False,
                "mask": {
                    "format": "rle_u8",
                    "width": 32,
                    "height": 32,
                    "class_index": 1,
                    "runs": [0, 512, 1, 512],
                },
            }
            validated = validate_reference_regions([region])
            self.assertEqual(validated[0]["mask"]["format"], "rle_u8")

    def test_pose_region_metadata_is_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pose.png"
            Image.new("RGB", (64, 64), "white").save(path)
            region = {
                "id": "ref-pose",
                "path": str(path),
                "view": "other",
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "primary": False,
                "origin": "mediapipe-pose",
                "confidence": None,
                "confirmed": False,
                "keypoints": [
                    {"name": "pose-0", "x": 0.5, "y": 0.2},
                ],
                "metadata": {"provider": "mediapipe-pose", "landmark_count": 1},
            }
            validated = validate_reference_regions([region])
            self.assertEqual(validated[0]["keypoints"][0]["name"], "pose-0")
            self.assertEqual(validated[0]["metadata"]["provider"], "mediapipe-pose")


if __name__ == "__main__":
    unittest.main()
