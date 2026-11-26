import tempfile
import unittest
from pathlib import Path
from PIL import Image

import numpy as np
import trimesh

from aethryx_avatar import (
    analyze_mesh_symmetry,
    build_avatar_scene_graph,
    repair_mesh_copy,
    split_disconnected_components,
)
from studio_project import save_project, load_project


class AvatarSceneGraphTests(unittest.TestCase):
    def test_scene_graph_has_humanoid_required_nodes(self):
        nodes = build_avatar_scene_graph()
        ids = {node["id"] for node in nodes}
        for required in (
            "avatar-root",
            "body-base",
            "head",
            "eyes",
            "hair",
            "clothing",
            "accessories",
            "armature",
        ):
            self.assertIn(required, ids)

    def test_pose_evidence_marks_body_detected_not_rigged(self):
        nodes = build_avatar_scene_graph(
            [{"view": "other", "origin": "mediapipe-pose", "confirmed": False}],
        )
        body = next(node for node in nodes if node["id"] == "body-base")
        armature = next(node for node in nodes if node["id"] == "armature")
        self.assertEqual(body["status"], "detected-needs-confirmation")
        self.assertEqual(armature["status"], "unavailable")

    def test_disconnected_geometry_components_are_split_without_anatomy_claims(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "components.glb"
            scene = trimesh.Scene([
                trimesh.creation.box(extents=(1, 1, 1)),
                trimesh.creation.box(extents=(0.5, 0.5, 0.5)),
            ])
            scene.export(str(path), file_type="glb")
            result = split_disconnected_components(path)
            self.assertEqual(result["status"], "available")
            self.assertGreaterEqual(len(result["components"]), 2)
            self.assertTrue(all(
                component["classification"] == "unassigned"
                for component in result["components"]
            ))

    def test_repair_copy_is_real_and_preserves_original_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.glb"
            output = Path(folder) / "repaired.glb"
            mesh = trimesh.creation.box(extents=(2, 2, 2))
            mesh.export(str(path), file_type="glb")
            result = repair_mesh_copy(path, output)
            self.assertEqual(result["status"], "available")
            self.assertTrue(output.is_file())
            original = trimesh.load(str(path), force="mesh", process=False)
            self.assertEqual(result["before"]["vertices"], len(original.vertices))
            self.assertGreaterEqual(result["after"]["vertices"], 1)

    def test_symmetry_analysis_runs_on_a_real_mesh(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "symmetric.glb"
            trimesh.creation.box(extents=(2, 2, 2)).export(str(path), file_type="glb")
            result = analyze_mesh_symmetry(path, axis="x")
            self.assertEqual(result["status"], "available")
            self.assertGreaterEqual(result["mean_vertex_mirror_distance"], 0)

    def test_avatar_scene_graph_round_trip_in_project_version_2(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "character.png"
            Image.new("RGB", (16, 16), "white").save(image)
            project = root / "avatar.aeth"
            nodes = build_avatar_scene_graph([{
                "view": "face_detail",
                "origin": "mediapipe-face",
                "confirmed": False,
            }])
            save_project(
                project,
                image=image,
                model=None,
                output=root / "mesh.glb",
                steps=50,
                faces=5000,
                seed=42,
                guidance=7.0,
                references=[],
                avatar_nodes=nodes,
            )
            loaded = load_project(project)
            self.assertEqual(loaded["avatar_nodes"], nodes)


if __name__ == "__main__":
    unittest.main()
