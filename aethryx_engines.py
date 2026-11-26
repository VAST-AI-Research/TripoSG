"""Intercambiable engine contracts and local, dependency-light implementations."""
from dataclasses import asdict, dataclass
import os
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
        EngineCapabilities(
            "triposr", "TripoSR / alternativa monovista",
            "MIT de código y pesos según model card",
            "COMPILACIÓN WINDOWS PENDIENTE", True, False, False, False, True, False, "1.0",
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
    segmentation_regions, segmentation_reason = _onnx_segmentation_regions(source, width, height)
    regions.extend(segmentation_regions)
    if regions:
        return {
            "status": "available",
            "reason": (
                "Detecciones locales disponibles"
                if mediapipe_regions or segmentation_regions
                else "Rostros frontales detectados por Haar"
            ),
            "regions": regions,
            "providers": (
                ["opencv-haar"]
                + (["mediapipe"] if mediapipe_regions else [])
                + (["onnx-segmentation"] if segmentation_regions else [])
            ),
        }
    return {
        "status": "completed",
        "reason": mediapipe_reason or "No se detectaron rostros frontales",
        "regions": [],
        "providers": ["opencv-haar"],
    }


def _mask_rle(mask):
    flat = mask.astype("uint8").reshape(-1)
    runs = []
    if flat.size == 0:
        return runs
    current = int(flat[0])
    count = 1
    for value in flat[1:]:
        value = int(value)
        if value == current:
            count += 1
        else:
            runs.extend((current, count))
            current = value
            count = 1
    runs.extend((current, count))
    return runs


def _rectangle_mask(width, height, bbox):
    import numpy as np
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    left = max(0, min(width - 1, int(x0 * width)))
    top = max(0, min(height - 1, int(y0 * height)))
    right = max(left + 1, min(width, int(x1 * width)))
    bottom = max(top + 1, min(height, int(y1 * height)))
    mask[top:bottom, left:right] = 1
    return mask


def _onnx_segmentation_regions(source, width, height):
    model_path = os.environ.get("AETHRYX_SEMANTIC_SEGMENTATION_MODEL", "")
    labels_text = os.environ.get("AETHRYX_SEMANTIC_SEGMENTATION_LABELS", "")
    if not model_path or not labels_text:
        return [], "Segmentación semántica no configurada"
    model = Path(model_path)
    if not model.is_file():
        return [], "Modelo ONNX de segmentación no encontrado"
    try:
        import cv2
        import numpy as np
    except ImportError:
        return [], "OpenCV no está disponible para segmentación"

    labels = [label.strip() for label in labels_text.split(",") if label.strip()]
    label_to_view = {
        "body": "other",
        "character": "other",
        "torso": "clothing_detail",
        "face": "face_detail",
        "hair": "hair_detail",
        "eye": "eyes_detail",
        "eyes": "eyes_detail",
        "mouth": "mouth_detail",
        "clothing": "clothing_detail",
        "clothes": "clothing_detail",
        "accessory": "accessory_detail",
        "accessories": "accessory_detail",
    }
    try:
        for package_name in ("nvidia/cudnn", "nvidia/cublas", "nvidia/cuda_runtime", "nvidia/cuda_nvrtc"):
            package_bin = Path(__import__("sys").prefix) / "Lib" / "site-packages" / package_name / "bin"
            if package_bin.is_dir():
                os.environ["PATH"] = str(package_bin) + os.pathsep + os.environ.get("PATH", "")
        with Image.open(source) as pil_image:
            rgb = np.asarray(pil_image.convert("RGB"))
        import onnxruntime as ort
        providers = []
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        session = ort.InferenceSession(str(model), providers=providers)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        model_size = (
            int(input_shape[2]) if isinstance(input_shape[2], int) else 1024,
            int(input_shape[3]) if isinstance(input_shape[3], int) else 1024,
        )
        resized = cv2.resize(rgb, model_size, interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        output = session.run(None, {input_name: tensor})[0]
        squeezed = np.squeeze(output)
        if squeezed.ndim == 2:
            mask_probs = squeezed.astype(np.float32)
            if mask_probs.min() >= 0.0 and mask_probs.max() <= 1.0:
                threshold = float(mask_probs.max()) * 0.5
                binary = mask_probs >= threshold
            else:
                binary = (mask_probs - mask_probs.min()) > (
                    (mask_probs.max() - mask_probs.min()) * 0.5
                )
            mask = cv2.resize(
                binary.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        elif squeezed.ndim == 3:
            channel_mask = np.argmax(squeezed, axis=0).astype(np.uint8)
            mask = cv2.resize(
                channel_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            return [], "El modelo ONNX devolvió una forma de máscara no compatible"
        regions = []
        for class_index, label in enumerate(labels):
            view = label_to_view.get(label.lower())
            if view is None:
                continue
            ys, xs = np.where(mask == class_index)
            if xs.size == 0:
                continue
            x0, x1 = float(xs.min()), float(xs.max() + 1)
            y0, y1 = float(ys.min()), float(ys.max() + 1)
            regions.append({
                "id": f"ref-{uuid4().hex[:16]}",
                "path": str(source),
                "view": view,
                "bbox": [
                    max(0.0, x0 / width),
                    max(0.0, y0 / height),
                    min(1.0, x1 / width),
                    min(1.0, y1 / height),
                ],
                "primary": False,
                "origin": "onnx-segmentation",
                "confidence": None,
                "confirmed": False,
                "mask": {
                    "format": "rle_u8",
                    "width": width,
                    "height": height,
                    "class_index": class_index,
                    "runs": _mask_rle(mask == class_index),
                },
                "metadata": {
                    "provider": "onnx-segmentation",
                    "label": label,
                    "model": str(model),
                    "execution_provider": session.get_providers()[0],
                },
            })
        if not regions:
            return [], "El modelo ONNX no produjo regiones compatibles con el esquema"
        return regions, "Segmentación ONNX local producida"
    except Exception as exc:
        return [], f"Segmentación ONNX no disponible: {exc}"


def _mediapipe_reference_regions(source, width, height):
    try:
        import mediapipe as mp
    except ImportError:
        return [], "MediaPipe no está instalado; Haar es la alternativa disponible"

    pose_model = os.environ.get("AETHRYX_MEDIAPIPE_POSE_MODEL", "")
    face_model = os.environ.get("AETHRYX_MEDIAPIPE_FACE_MODEL", "")
    if not pose_model and not face_model:
        return [], (
            "MediaPipe 0.10 Tasks está instalado, pero no hay pesos locales "
            "autorizados configurados; usa Haar hasta incorporar modelos"
        )

    import numpy as np
    source = str(source)
    regions = []

    with Image.open(source) as pil_image:
        rgb = np.asarray(pil_image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    if pose_model:
        try:
            if not Path(pose_model).is_file():
                raise FileNotFoundError("Modelo pose no encontrado")
            base = mp.tasks.BaseOptions(model_asset_path=pose_model)
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
            )
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                result = landmarker.detect(mp_image)
            pose_landmarks = getattr(result, "pose_landmarks", None)
            if pose_landmarks:
                landmarks = pose_landmarks[0]
                points = []
                xs = []
                ys = []
                for index, landmark in enumerate(landmarks):
                    x = float(landmark.x)
                    y = float(landmark.y)
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        xs.append(x)
                        ys.append(y)
                        points.append({"name": f"pose-{index}", "x": x, "y": y})
                if xs and ys:
                    regions.append({
                        "id": f"ref-{uuid4().hex[:16]}",
                        "path": source,
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
                    component_specs = (
                        ("face_detail", (0, 1, 2, 5, 4, 6, 7, 8, 9, 10), "face"),
                        ("eyes_detail", (1, 2, 3, 4, 5, 6), "eyes"),
                        ("clothing_detail", (11, 12, 23, 24), "clothing"),
                        ("hair_detail", (0, 1, 2, 3, 4, 5, 6, 7, 8), "hair"),
                    )
                    point_by_index = {
                        index: {"x": float(landmark.x), "y": float(landmark.y)}
                        for index, landmark in enumerate(landmarks)
                    }
                    for view, indices, component_name in component_specs:
                        selected = [
                            point_by_index[index]
                            for index in indices
                            if index in point_by_index
                            and 0 <= point_by_index[index]["x"] <= 1
                            and 0 <= point_by_index[index]["y"] <= 1
                        ]
                        if len(selected) < 2:
                            continue
                        comp_xs = [point["x"] for point in selected]
                        comp_ys = [point["y"] for point in selected]
                        bbox = [
                            max(0.0, min(comp_xs)),
                            max(0.0, min(comp_ys)),
                            min(1.0, max(comp_xs)),
                            min(1.0, max(comp_ys)),
                        ]
                        if component_name == "hair":
                            face_bottom = bbox[3]
                            bbox = [
                                max(0.0, bbox[0]),
                                max(0.0, max(0.0, bbox[1] - max(0.08, (face_bottom - bbox[1]) * 0.8))),
                                min(1.0, bbox[2]),
                                min(1.0, face_bottom),
                            ]
                        component_mask = _rectangle_mask(width, height, bbox)
                        regions.append({
                            "id": f"ref-{uuid4().hex[:16]}",
                            "path": source,
                            "view": view,
                            "bbox": bbox,
                            "primary": False,
                            "origin": "mediapipe-pose-heuristic",
                            "confidence": None,
                            "confirmed": False,
                            "keypoints": [
                                {"name": f"{component_name}-{index}", "x": point["x"], "y": point["y"]}
                                for index, point in zip(indices, selected)
                            ],
                            "mask": {
                                "format": "rle_u8",
                                "width": width,
                                "height": height,
                                "class_index": 1,
                                "runs": _mask_rle(component_mask),
                            },
                            "metadata": {
                                "provider": "mediapipe-pose-heuristic",
                                "component": component_name,
                                "status": "provisional-region-not-semantic-mask",
                            },
                        })
        except Exception as exc:
            return regions, f"MediaPipe pose no estuvo disponible: {exc}"

    if face_model:
        try:
            if not Path(face_model).is_file():
                raise FileNotFoundError("Modelo facial no encontrado")
            base = mp.tasks.BaseOptions(model_asset_path=face_model)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=5,
            )
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
                result = landmarker.detect(mp_image)
            for face in getattr(result, "face_landmarks", []):
                xs = [float(point.x) for point in face]
                ys = [float(point.y) for point in face]
                regions.append({
                    "id": f"ref-{uuid4().hex[:16]}",
                    "path": source,
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
                        for index, point in enumerate(face)
                    ],
                    "metadata": {"provider": "mediapipe-face", "landmark_count": len(face)},
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


def prepare_character_alpha(source, output_path):
    source = Path(source).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not source.is_file():
        return {"status": "unavailable", "reason": "Imagen de referencia no encontrada"}
    model_path = os.environ.get("AETHRYX_SEMANTIC_SEGMENTATION_MODEL", "")
    labels_text = os.environ.get("AETHRYX_SEMANTIC_SEGMENTATION_LABELS", "")
    if not model_path or not labels_text:
        model_path = str(Path(__file__).parent / "models" / "anime" / "isnetis.onnx")
        labels_text = "background,character"
    os.environ["AETHRYX_SEMANTIC_SEGMENTATION_MODEL"] = model_path
    os.environ["AETHRYX_SEMANTIC_SEGMENTATION_LABELS"] = labels_text
    with Image.open(source) as pil_image:
        rgb = pil_image.convert("RGB")
        width, height = rgb.size
    regions, reason = _onnx_segmentation_regions(source, width, height)
    character = next(
        (region for region in regions if region.get("view") == "other"),
        None,
    )
    if character is None:
        return {"status": "unavailable", "reason": reason}
    mask = character["mask"]
    import numpy as np
    alpha = np.zeros((height, width), dtype=np.uint8)
    runs = mask["runs"]
    position = 0
    for index in range(0, len(runs), 2):
        value = runs[index]
        count = runs[index + 1]
        alpha.flat[position:position + count] = 255 if value else 0
        position += count
    rgba = np.dstack((np.asarray(rgb), alpha))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(output_path, format="PNG")
    return {
        "status": "available",
        "path": str(output_path),
        "provider": "onnx-segmentation",
        "model": model_path,
    }
