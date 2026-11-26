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
            False,
            False,
            "PyVista/VTK no instalado; aceleración no probada.",
        ),
    ]


def viewer_catalog_dict():
    return [asdict(item) for item in viewer_catalog()]
