import json
from pathlib import Path
import struct
import tempfile
import unittest
from studio_project import save_project, load_project, validate_settings, validate_glb

class ProjectTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "image.png"
            image.touch()
            project = root / "test.aeth"
            save_project(project, image=image, model=None, output=root/"mesh.glb",
                         steps=50, faces=-1, seed=42, guidance=7.0)
            self.assertEqual(json.loads(project.read_text())["image"], "image.png")
            self.assertEqual(load_project(project)["image"], str(image.resolve()))
            image.unlink()
            with self.assertRaises(ValueError):
                load_project(project)
    def test_invalid_settings(self):
        for values in [(0,-1,42,7), (50,0,42,7), (50,-1,-1,7), (50,-1,42,float("nan"))]:
            with self.assertRaises(ValueError):
                validate_settings(*values)
    def test_glb_header_and_chunks(self):
        data = b'{"meshes":[{}]}'
        data += b" " * (-len(data) % 4)
        content = struct.pack("<4sII", b"glTF", 2, 20+len(data)) + struct.pack("<I4s",len(data),b"JSON") + data
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"mesh.glb"
            path.write_bytes(content)
            self.assertIn("meshes", validate_glb(path))
            path.write_bytes(content[:-1])
            with self.assertRaises(ValueError):
                validate_glb(path)
if __name__ == "__main__":
    unittest.main()
