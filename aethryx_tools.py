"""Typed read-only tool registry for future KIRIA and AETHRYX Connect clients."""
from dataclasses import asdict, dataclass
from pathlib import Path

import trimesh
from aethryx_avatar import build_avatar_scene_graph, split_disconnected_components


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict
    permissions: tuple
    available: bool
    project_context_required: bool = False


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


def get_avatar_scene_graph(context):
    if not isinstance(context, dict):
        return {"status": "unavailable", "reason": "Contexto de proyecto no proporcionado"}
    project_id = context.get("project_id")
    revision = context.get("revision")
    if not project_id or not revision:
        return {"status": "unavailable", "reason": "Faltan project_id o revision"}
    nodes = build_avatar_scene_graph(
        context.get("references"),
        context.get("mesh_stats"),
    )
    return {
        "status": "available",
        "project_id": project_id,
        "revision": revision,
        "nodes": nodes,
    }


def validate_avatar_placeholder():
    return {
        "status": "unavailable",
        "reason": "La validación Unity/VRChat requiere un rig y SDK compatibles",
    }


def get_project_context(context):
    if not isinstance(context, dict):
        return {"status": "unavailable", "reason": "Contexto de proyecto no proporcionado"}
    project_id = context.get("project_id")
    revision = context.get("revision")
    if not project_id or not revision:
        return {"status": "unavailable", "reason": "Faltan project_id o revision"}
    allowed = {}
    for key in ("image", "model", "output", "references", "settings", "mesh_stats"):
        if key in context:
            allowed[key] = context[key]
    return {
        "status": "available",
        "project_id": project_id,
        "revision": revision,
        "context": allowed,
    }


def analyze_project_diagnostics(context):
    project = get_project_context(context)
    if project["status"] != "available":
        return project
    data = project["context"]
    references = data.get("references", [])
    full_views = {
        region.get("view")
        for region in references
        if isinstance(region, dict) and region.get("confirmed")
    }
    missing = [
        view for view in ("front", "side", "back", "three_quarter")
        if view not in full_views
    ]
    findings = []
    if not references:
        findings.append("No hay referencias registradas.")
    if missing:
        findings.append("Vistas completas no confirmadas: " + ", ".join(missing) + ".")
    mesh_stats = data.get("mesh_stats")
    if not mesh_stats:
        findings.append("No hay una malla cargada o estadisticas 3D disponibles.")
    else:
        vertices = mesh_stats.get("vertices", 0)
        faces = mesh_stats.get("faces", 0)
        findings.append(f"Malla real cargada: {vertices} vertices y {faces} caras.")
    if not data.get("model"):
        findings.append("No hay modelo 3D importado en el proyecto.")
    return {
        "status": "available",
        "project_id": project["project_id"],
        "revision": project["revision"],
        "findings": findings,
        "missing_full_views": missing,
        "source": "deterministic-local",
    }


TOOL_REGISTRY = {
    "split_disconnected_components": ToolDefinition(
        "split_disconnected_components",
        "Separa componentes geométricos desconectados sin asignar anatomía.",
        {"model_path": "string"},
        ("read",),
        True,
    ),
    "get_avatar_scene_graph": ToolDefinition(
        "get_avatar_scene_graph",
        "Devuelve el scene graph humanoide y el estado real de sus componentes.",
        {
            "project_id": "string",
            "revision": "string",
            "references": "array",
            "mesh_stats": "object",
        },
        ("read",),
        True,
        True,
    ),
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
    "get_project_context": ToolDefinition(
        "get_project_context",
        "Devuelve solo el contexto de proyecto autorizado para lectura.",
        {
            "project_id": "string",
            "revision": "string",
            "fields": "array<string>",
        },
        ("read",),
        True,
        True,
    ),
    "analyze_project_diagnostics": ToolDefinition(
        "analyze_project_diagnostics",
        "Produce diagnosticos locales deterministas basados en referencias y malla.",
        {
            "project_id": "string",
            "revision": "string",
            "references": "array",
            "mesh_stats": "object",
            "model": "string",
        },
        ("read",),
        True,
        True,
    ),
}


def tool_catalog():
    return [asdict(tool) for tool in TOOL_REGISTRY.values()]


def execute_tool(name, inputs, project_context=None, write_allowed=False):
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return {"status": "unavailable", "reason": "Herramienta no registrada"}
    if not tool.available:
        return {"status": "unavailable", "reason": "Herramienta no implementada"}
    if not isinstance(inputs, dict):
        return {"status": "error", "reason": "Entrada invalida"}
    if "write" in tool.permissions and not write_allowed:
        return {"status": "denied", "reason": "La herramienta requiere permiso de escritura"}
    if tool.project_context_required:
        context = dict(project_context or {})
        context.update({
            key: value for key, value in inputs.items()
            if key in ("project_id", "revision", "references", "mesh_stats", "model")
        })
        if name == "get_project_context":
            return get_project_context(context)
        if name == "analyze_project_diagnostics":
            return analyze_project_diagnostics(context)
        if name == "get_avatar_scene_graph":
            return get_avatar_scene_graph(context)
    if name == "list_scene_nodes":
        return list_scene_nodes(inputs.get("model_path", ""))
    if name == "get_mesh_statistics":
        return get_mesh_statistics(inputs.get("model_path", ""))
    if name == "split_disconnected_components":
        return split_disconnected_components(inputs.get("model_path", ""))
    if name == "validate_avatar":
        return validate_avatar_placeholder()
    return {"status": "unavailable", "reason": "Sin ejecutor implementado"}
