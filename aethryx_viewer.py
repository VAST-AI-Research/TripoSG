"""Viewer backend contract. GPU acceleration is not assumed until a backend is tested."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ViewerBackend:
    backend_id: str
    name: str
    available: bool
    gpu_accelerated: bool
    reason: str


def viewer_catalog():
    return [
        ViewerBackend(
            "matplotlib-tk",
            "Matplotlib/Tk",
            True,
            False,
            "Visor funcional actual; renderizado no certificado como GPU.",
        ),
        ViewerBackend(
            "pyvista-vtk",
            "PyVista/VTK",
            True,
            True,
            "PyVista/VTK instalado y reportó la RTX 5070 como renderizador OpenGL; sin benchmark interactivo.",
        ),
    ]


def viewer_catalog_dict():
    return [asdict(item) for item in viewer_catalog()]
