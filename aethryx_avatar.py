"""Deterministic humanoid avatar scene graph for AETHRYX Studio."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AvatarNode:
    id: str
    name: str
    parent_id: str | None
    kind: str
    status: str
    source: str
    confidence: float | None = None


def build_avatar_scene_graph(references=None, mesh_stats=None):
    references = references or []
    nodes = [
        AvatarNode("avatar-root", "Avatar Root", None, "root", "available", "deterministic"),
        AvatarNode("body-base", "Body Base", "avatar-root", "body", "unavailable", "mesh"),
        AvatarNode("head", "Head", "avatar-root", "face", "unavailable", "mesh"),
        AvatarNode("eyes", "Eyes", "head", "face", "unavailable", "vision"),
        AvatarNode("hair", "Hair", "head", "hair", "unavailable", "vision"),
        AvatarNode("clothing", "Clothing", "body-base", "clothing", "unavailable", "vision"),
        AvatarNode("accessories", "Accessories", "avatar-root", "accessory", "unavailable", "vision"),
        AvatarNode("armature", "Humanoid Armature", "avatar-root", "rig", "unavailable", "rig"),
    ]

    views = {
        region.get("view")
        for region in references
        if isinstance(region, dict)
    }
    origins = {
        region.get("origin")
        for region in references
        if isinstance(region, dict)
    }
    pose_present = "mediapipe-pose" in origins
    face_present = "mediapipe-face" in origins or "opencv-haar" in origins

    updated = []
    for node in nodes:
        status = node.status
        source = node.source
        if node.id == "eyes" and face_present:
            status = "detected-needs-confirmation"
            source = "vision"
        if node.id == "head" and face_present:
            status = "detected-needs-confirmation"
            source = "vision"
        if node.id == "body-base" and pose_present:
            status = "detected-needs-confirmation"
            source = "mediapipe-pose"
        if node.id == "hair" and "hair_detail" in views:
            status = "detected-needs-confirmation"
            source = "vision"
        if node.id == "clothing" and "clothing_detail" in views:
            status = "detected-needs-confirmation"
            source = "vision"
        if node.id == "accessories" and "accessory_detail" in views:
            status = "detected-needs-confirmation"
            source = "vision"
        if node.id == "body-base" and mesh_stats:
            status = "mesh-imported-not-segmented"
            source = "mesh"
        updated.append(AvatarNode(
            node.id,
            node.name,
            node.parent_id,
            node.kind,
            status,
            source,
            node.confidence,
        ))
    return [asdict(node) for node in updated]


def split_disconnected_components(model_path):
    from pathlib import Path
    import trimesh

    path = Path(model_path)
    if not path.is_file():
        return {"status": "unavailable", "reason": "Malla no encontrada"}
    try:
        scene = trimesh.load(str(path), force="scene", process=False)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"No se pudo cargar la malla: {exc}"}

    components = []
    for geometry_name, geometry in scene.geometry.items():
        try:
            pieces = geometry.split(only_watertight=False)
        except Exception:
            pieces = [geometry]
        for index, piece in enumerate(pieces):
            bounds = piece.bounds
            components.append({
                "id": f"{geometry_name}-component-{index}",
                "name": f"Componente geométrico {index + 1}",
                "vertices": int(len(piece.vertices)),
                "faces": int(len(piece.faces)),
                "bounds": [
                    [float(value) for value in bounds[0]],
                    [float(value) for value in bounds[1]],
                ],
                "classification": "unassigned",
                "editable": True,
            })
    return {
        "status": "available",
        "components": components,
        "note": "Separación topológica real; anatomía no inferida.",
    }


def export_disconnected_components(model_path, output_dir):
    from pathlib import Path
    import trimesh

    path = Path(model_path)
    output_dir = Path(output_dir)
    if not path.is_file():
        return {"status": "unavailable", "reason": "Malla no encontrada"}
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        scene = trimesh.load(str(path), force="scene", process=False)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"No se pudo cargar la malla: {exc}"}

    exported = []
    component_index = 0
    for geometry in scene.geometry.values():
        try:
            pieces = geometry.split(only_watertight=False)
        except Exception:
            pieces = [geometry]
        for piece in pieces:
            component_index += 1
            output_path = output_dir / f"component-{component_index}.glb"
            trimesh.Scene([piece]).export(str(output_path), file_type="glb")
            exported.append({
                "path": str(output_path),
                "vertices": int(len(piece.vertices)),
                "faces": int(len(piece.faces)),
            })
    return {
        "status": "available",
        "components": exported,
        "classification": "unassigned",
    }


def repair_mesh_copy(model_path, output_path):
    from pathlib import Path
    import trimesh

    path = Path(model_path)
    output_path = Path(output_path)
    if not path.is_file():
        return {"status": "unavailable", "reason": "Malla no encontrada"}
    try:
        mesh = trimesh.load(str(path), force="mesh", process=False)
    except Exception as exc:
        return {"status": "unavailable", "reason": f"No se pudo cargar la malla: {exc}"}

    before_vertices = int(len(mesh.vertices))
    before_faces = int(len(mesh.faces))
    mesh = mesh.copy()
    mesh.merge_vertices()
    import numpy as np
    face_keys = np.sort(np.asarray(mesh.faces, dtype=np.int64), axis=1)
    _, unique_indices = np.unique(face_keys, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    mesh.update_faces(np.isin(np.arange(len(mesh.faces)), unique_indices))
    triangles = mesh.triangles
    if len(triangles):
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        degenerate = np.linalg.norm(cross, axis=1) <= 1e-12
        mesh.update_faces(~degenerate)
    mesh.remove_unreferenced_vertices()
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Scene([mesh]).export(str(output_path), file_type="glb")
    validate = trimesh.load(str(output_path), force="mesh", process=False)
    return {
        "status": "available",
        "path": str(output_path),
        "before": {"vertices": before_vertices, "faces": before_faces},
        "after": {
            "vertices": int(len(validate.vertices)),
            "faces": int(len(validate.faces)),
            "watertight": bool(validate.is_watertight),
        },
    }


def analyze_mesh_symmetry(model_path, axis="x"):
    from pathlib import Path
    import numpy as np
    import trimesh

    path = Path(model_path)
    if not path.is_file():
        return {"status": "unavailable", "reason": "Malla no encontrada"}
    mesh = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
    if axis_index is None:
        return {"status": "unavailable", "reason": "Eje de simetría no compatible"}
    if len(vertices) == 0:
        return {"status": "unavailable", "reason": "Malla sin vértices"}
    mirrored = vertices.copy()
    mirrored[:, axis_index] *= -1
    kd = None
    try:
        from scipy.spatial import cKDTree
        kd = cKDTree(mirrored)
    except ImportError:
        pass
    if kd is None:
        distances = [
            float(np.min(np.linalg.norm(vertices - point, axis=1)))
            for point in mirrored
        ]
    else:
        distances = kd.query(vertices, k=1)[0]
    return {
        "status": "available",
        "axis": axis.lower(),
        "mean_vertex_mirror_distance": float(np.mean(distances)),
        "max_vertex_mirror_distance": float(np.max(distances)),
    }
