import tempfile
import unittest
from pathlib import Path
from PIL import Image

from aethryx_avatar import build_avatar_scene_graph
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
