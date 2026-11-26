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
        EngineCapabilities(
            "lgm", "LGM / candidatos gaussianos multivista",
            "Revisar código, pesos y licencia antes de integrar",
            "NO EVALUADO", True, True, False, False, True, False, "candidate",
        ),
        EngineCapabilities(
            "unique3d", "Unique3D / candidato multivista",
            "Revisar código, pesos y licencia antes de integrar",
            "NO EVALUADO", True, True, False, False, True, False, "candidate",
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
        x_int, y_int, box_width_int, box_height_int = (
            int(x), int(y), int(box_width), int(box_height)
        )
        x, y, box_width, box_height = (
            float(x_int), float(y_int), float(box_width_int), float(box_height_int)
        )
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
            face_gray = gray[
                y_int:y_int + box_height_int,
                x_int:x_int + box_width_int,
            ]
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)
            for ex, ey, eye_width, eye_height in eyes:
                ex, ey, eye_width, eye_height = (
                    float(ex), float(ey), float(eye_width), float(eye_height)
                )
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

    mediapipe_regions, mediapipe_reason = _mediapipe_reference_regions(source, width, height)
    regions.extend(mediapipe_regions)
    if regions:
        return {
            "status": "available",
            "reason": "Detecciones locales disponibles" if mediapipe_regions else
                      "Rostros frontales detectados por Haar",
            "regions": regions,
            "providers": ["opencv-haar"] + (["mediapipe"] if mediapipe_regions else []),
        }
    return {
        "status": "completed",
        "reason": mediapipe_reason or "No se detectaron rostros frontales",
        "regions": [],
        "providers": ["opencv-haar"],
    }


def _mediapipe_reference_regions(source, width, height):
    try:
        import mediapipe as mp
    except ImportError:
        return [], "MediaPipe no está instalado; Haar es la alternativa disponible"

    import numpy as np

    regions = []
    with Image.open(source) as pil_image:
        rgb = np.asarray(pil_image.convert("RGB"))

    solutions = getattr(mp, "solutions", None)
    if solutions is None:
        return [], "Esta instalación de MediaPipe no expone proveedores de imagen compatibles"

    try:
        with solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        ) as pose:
            result = pose.process(rgb)
        if result.pose_landmarks:
            points = []
            xs = []
            ys = []
            for index, landmark in enumerate(result.pose_landmarks.landmark):
                x = float(landmark.x)
                y = float(landmark.y)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    xs.append(x)
                    ys.append(y)
                    points.append({"name": f"pose-{index}", "x": x, "y": y})
            if xs and ys:
                regions.append({
                    "id": f"ref-{uuid4().hex[:16]}",
                    "path": str(source),
                    "view": "other",
                    "bbox": [
                        max(0.0, min(xs)),
                        max(0.0, min(ys)),
                        min(1.0, max(xs)),
                        min(1.0, max(ys)),
                    ],
                    "primary": False,
                    "origin": "mediapipe-pose",
                    "confidence": None,
                    "confirmed": False,
                    "keypoints": points,
                    "metadata": {"provider": "mediapipe-pose", "landmark_count": len(points)},
                })
    except Exception as exc:
        return regions, f"MediaPipe pose no estuvo disponible: {exc}"

    try:
        with solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
        ) as face_mesh:
            result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            for face in result.multi_face_landmarks:
                xs = [float(point.x) for point in face.landmark]
                ys = [float(point.y) for point in face.landmark]
                regions.append({
                    "id": f"ref-{uuid4().hex[:16]}",
                    "path": str(source),
                    "view": "face_detail",
                    "bbox": [
                        max(0.0, min(xs)),
                        max(0.0, min(ys)),
                        min(1.0, max(xs)),
                        min(1.0, max(ys)),
                    ],
                    "primary": False,
                    "origin": "mediapipe-face",
                    "confidence": None,
                    "confirmed": False,
                    "keypoints": [
                        {"name": f"face-{index}", "x": float(point.x), "y": float(point.y)}
                        for index, point in enumerate(face.landmark)
                    ],
                    "metadata": {"provider": "mediapipe-face", "landmark_count": len(face.landmark)},
                })
    except Exception as exc:
        return regions, f"MediaPipe face no estuvo disponible: {exc}"

    if not regions:
        return [], "MediaPipe no detectó rostro ni pose"
    return regions, "MediaPipe detectó características locales"


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
