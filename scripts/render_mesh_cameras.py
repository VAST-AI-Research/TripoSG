import argparse
from pathlib import Path

import pyvista as pv
import trimesh


def render_mesh_cameras(input_path, output_dir, prefix):
    mesh = trimesh.load(input_path, force="mesh", process=False)
    poly = pv.wrap(mesh)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = {
        "front": [(0, 0, 3), (0, 0, 0), (0, 1, 0)],
        "side": [(3, 0, 0), (0, 0, 0), (0, 1, 0)],
        "rear": [(0, 0, -3), (0, 0, 0), (0, 1, 0)],
    }
    rendered = []
    for name, position in cameras.items():
        plotter = pv.Plotter(off_screen=True, window_size=(900, 900))
        plotter.set_background("#11141b")
        plotter.add_mesh(poly, color="#c9d4dc", smooth_shading=True)
        plotter.camera_position = position
        plotter.add_axes()
        output = output_dir / f"{prefix}-{name}.png"
        plotter.show(screenshot=str(output), auto_close=True)
        rendered.append(str(output))
    return rendered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="mesh")
    args = parser.parse_args()
    print("\n".join(render_mesh_cameras(args.input, args.output_dir, args.prefix)))
