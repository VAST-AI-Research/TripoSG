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
