from math import ceil, floor, isfinite
from pathlib import Path

from PIL import Image


REFERENCE_VIEW_LABELS = {
    "front": "Frontal",
    "side": "Lateral",
    "back": "Posterior",
    "three_quarter": "Tres cuartos",
    "face_detail": "Detalle del rostro",
    "eyes_detail": "Detalle de los ojos",
    "mouth_detail": "Detalle de la boca",
    "hair_detail": "Detalle del cabello",
    "clothing_detail": "Detalle de la ropa",
    "accessory_detail": "Detalle de accesorios",
    "other": "Otro",
}
FULL_BODY_VIEWS = frozenset(("front", "side", "back", "three_quarter"))
MAX_REFERENCE_REGIONS = 128


def validate_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("El recorte debe contener cuatro coordenadas.")
    if any(type(value) not in (int, float) or not isfinite(value) for value in bbox):
        raise ValueError("Las coordenadas del recorte deben ser números finitos.")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("El recorte debe estar dentro de la imagen y tener área.")
    return [x0, y0, x1, y1]


def normalize_bbox(start, end, image_size):
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("La imagen tiene dimensiones inválidas.")
    x0 = min(start[0], end[0]) / width
    y0 = min(start[1], end[1]) / height
    x1 = max(start[0], end[0]) / width
    y1 = max(start[1], end[1]) / height
    return validate_bbox(
        [
            max(0.0, x0),
            max(0.0, y0),
            min(1.0, x1),
            min(1.0, y1),
        ]
    )


def validate_reference_regions(regions):
    if not isinstance(regions, (list, tuple)) or len(regions) > MAX_REFERENCE_REGIONS:
        raise ValueError(f"El proyecto admite hasta {MAX_REFERENCE_REGIONS} recortes.")

    validated = []
    seen_ids = set()
    primary_count = 0
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("Cada referencia debe ser un objeto.")
        region_id = region.get("id")
        source = region.get("path")
        view = region.get("view")
        primary = region.get("primary", False)
        origin = region.get("origin", "manual")
        confidence = region.get("confidence")
        confirmed = region.get("confirmed", True)
        if not isinstance(region_id, str) or not region_id or len(region_id) > 64:
            raise ValueError("Identificador de referencia inválido.")
        if region_id in seen_ids:
            raise ValueError("Los identificadores de referencia deben ser únicos.")
        if not isinstance(source, str) or not source:
            raise ValueError("La ruta de la referencia no es válida.")
        if not isinstance(view, str) or view not in REFERENCE_VIEW_LABELS:
            raise ValueError("Tipo de vista de referencia no compatible.")
        if type(primary) is not bool:
            raise ValueError("El estado de vista principal debe ser booleano.")
        if primary and view not in FULL_BODY_VIEWS:
            raise ValueError("Solo una vista de cuerpo completo puede ser principal.")
        if origin not in ("manual", "opencv-haar"):
            raise ValueError("Origen de referencia no compatible.")
        if confidence is not None:
            if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
                raise ValueError("La confianza debe estar entre 0 y 1.")
        if type(confirmed) is not bool:
            raise ValueError("El estado de confirmación debe ser booleano.")

        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"No se encontró la imagen de referencia: {path}")
        validated.append(
            {
                "id": region_id,
                "path": str(path),
                "view": view,
                "bbox": validate_bbox(region.get("bbox")),
                "primary": primary,
                "origin": origin,
                "confidence": confidence,
                "confirmed": confirmed,
            }
        )
        seen_ids.add(region_id)
        primary_count += int(primary)

    if primary_count > 1:
        raise ValueError("Solo puede haber una vista principal.")
    return validated


def crop_reference(source, bbox):
    x0, y0, x1, y1 = validate_bbox(bbox)
    with Image.open(source) as image:
        width, height = image.size
        left = max(0, min(width - 1, floor(x0 * width)))
        top = max(0, min(height - 1, floor(y0 * height)))
        right = max(left + 1, min(width, ceil(x1 * width)))
        bottom = max(top + 1, min(height, ceil(y1 * height)))
        crop = image.crop((left, top, right, bottom))
        if "A" in crop.getbands() or "transparency" in image.info:
            background = Image.new("RGBA", crop.size, (255, 255, 255, 255))
            background.alpha_composite(crop.convert("RGBA"))
            return background.convert("RGB")
        return crop.convert("RGB")
