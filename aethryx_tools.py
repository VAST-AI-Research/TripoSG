"""Typed read-only tool registry for future KIRIA and AETHRYX Connect clients."""
from dataclasses import asdict, dataclass
from pathlib import Path

import trimesh


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict
    permissions: tuple
    available: bool


def list_scene_nodes(model_path):
    path = Path(model_path)
    if not path.is_file():
        return {"status": "unavailable", "reason": "Malla no encontrada"}
    scene = trimesh.load(str(path), force="scene", process=False)
    nodes = []
    for name, geometry in scene.geometry.items():
        nodes.append({
            "id": name,
            "name": name,
            "vertices": int(len(geometry.vertices)),
            "faces": int(len(geometry.faces)),
            "visible": True,
        })
    return {"status": "available", "nodes": nodes}


def get_mesh_statistics(model_path):
    path = Path(model_path)
    if not path.is_file():
        return {"status": "unavailable", "reason": "Malla no encontrada"}
    scene = trimesh.load(str(path), force="scene", process=False)
    vertex_count = sum(len(geometry.vertices) for geometry in scene.geometry.values())
    face_count = sum(len(geometry.faces) for geometry in scene.geometry.values())
    bounds = scene.bounds
    if bounds is None:
        return {"status": "unavailable", "reason": "No se pudieron calcular los límites"}
    return {
        "status": "available",
        "vertices": int(vertex_count),
        "faces": int(face_count),
        "bounds_min": [float(value) for value in bounds[0]],
        "bounds_max": [float(value) for value in bounds[1]],
    }


def validate_avatar_placeholder():
    return {
        "status": "unavailable",
        "reason": "La validación Unity/VRChat requiere un rig y SDK compatibles",
    }


TOOL_REGISTRY = {
    "list_scene_nodes": ToolDefinition(
        "list_scene_nodes",
        "Lista nodos reales y recuentos básicos de geometría.",
        {"model_path": "string"},
        ("read",),
        True,
    ),
    "get_mesh_statistics": ToolDefinition(
        "get_mesh_statistics",
        "Calcula vértices, caras y límites de una malla existente.",
        {"model_path": "string"},
        ("read",),
        True,
    ),
    "validate_avatar": ToolDefinition(
        "validate_avatar",
        "Valida requisitos reales de avatar humanoide.",
        {"project_id": "string", "revision": "string"},
        ("read",),
        False,
    ),
}


def tool_catalog():
    return [asdict(tool) for tool in TOOL_REGISTRY.values()]
