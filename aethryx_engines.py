"""Intercambiable engine contracts and local, dependency-light implementations."""
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image


@dataclass(frozen=True)
class EngineCapabilities:
    engine_id: str
    name: str
    license: str
    commercial_review: str
    single_view: bool
    multi_view: bool
    texturing: bool
    rigging: bool
    local: bool
    available: bool
    version: str


def engine_catalog():
    return [
        EngineCapabilities(
            "triposg", "TripoSG",
            "Revisar licencia upstream y pesos antes de redistribuir",
            "PENDIENTE", True, False, False, False, True, True, "repository",
        ),
        EngineCapabilities(
            "multiview-character", "Reconstrucción multivista especializada",
            "No integrado", "NO EVALUADO",
            False, True, True, True, True, False, "none",
        ),
    ]


def _opencv_status():
    try:
        import cv2
    except ImportError:
        return None, "opencv-python no está instalado"
    cascades = getattr(cv2, "data", None)
    if cascades is None:
        return None, "OpenCV no proporciona rutas de clasificadores locales"
    face_path = Path(cascades.haarcascades) / "haarcascade_frontalface_default.xml"
    eye_path = Path(cascades.haarcascades) / "haarcascade_eye.xml"
    if not face_path.is_file():
        return None, "Clasificador frontal Haar no disponible"
    return (cv2, str(face_path), str(eye_path) if eye_path.is_file() else None), None


def analyze_reference_image(source):
    """Run local OpenCV face detection. It never guesses lateral, rear or 3/4 views."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        return {"status": "unavailable", "reason": "Imagen de referencia no encontrada", "regions": []}

    loaded, reason = _opencv_status()
    if loaded is None:
        return {"status": "unavailable", "reason": reason, "regions": []}

    cv2, face_cascade_path, eye_cascade_path = loaded
    with Image.open(source) as pil_image:
        rgb = pil_image.convert("RGB")
        width, height = rgb.size
        image = cv2.cvtColor(__import__("numpy").array(rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        return {"status": "unavailable", "reason": "No se pudo cargar el clasificador frontal Haar", "regions": []}
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    regions = []
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path) if eye_cascade_path else None
    for x, y, box_width, box_height in faces:
        regions.append({
            "id": f"ref-{uuid4().hex[:16]}",
            "path": str(source),
            "view": "face_detail",
            "bbox": [
                max(0.0, x / width),
                max(0.0, y / height),
                min(1.0, (x + box_width) / width),
                min(1.0, (y + box_height) / height),
            ],
            "primary": False,
            "origin": "opencv-haar",
            "confidence": None,
            "confirmed": False,
        })
        if eye_cascade is not None and not eye_cascade.empty():
            face_gray = gray[y:y + box_height, x:x + box_width]
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)
            for ex, ey, eye_width, eye_height in eyes:
                regions.append({
                    "id": f"ref-{uuid4().hex[:16]}",
                    "path": str(source),
                    "view": "eyes_detail",
                    "bbox": [
                        max(0.0, (x + ex) / width),
                        max(0.0, (y + ey) / height),
                        min(1.0, (x + ex + eye_width) / width),
                        min(1.0, (y + ey + eye_height) / height),
                    ],
                    "primary": False,
                    "origin": "opencv-haar",
                    "confidence": None,
                    "confirmed": False,
                })

    return {
        "status": "available" if regions else "completed",
        "reason": "Rostros frontales detectados por un modelo local Haar" if regions else "No se detectaron rostros frontales",
        "regions": regions,
    }


def missing_reference_views(regions):
    present = {region["view"] for region in regions if region["confirmed"]}
    missing = [
        view for view in ("front", "side", "back", "three_quarter")
        if view not in present
    ]
    return {
        "missing": missing,
        "confirmed_views": sorted(present.intersection(("front", "side", "back", "three_quarter"))),
        "note": "Las vistas no confirmadas no son predicciones.",
    }


def engine_catalog_dict():
    return [asdict(engine) for engine in engine_catalog()]
