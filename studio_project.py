"""Project manifests and GLB checks, independent of the GUI and GPU."""
import json
import os
from pathlib import Path
import struct
import tempfile

def validate_settings(steps, faces, seed, guidance):
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("Pasos: 1–100.")
    if type(faces) is not int or (faces != -1 and not 100 <= faces <= 2000000):
        raise ValueError("Caras: -1 (sin límite) o 100–2.000.000.")
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("Semilla: 0–2147483647.")
    if type(guidance) not in (int, float) or not 0 <= guidance <= 20:
        raise ValueError("Guidance: 0–20.")

def validate_glb(path):
    """Check GLB 2 header, JSON chunk and mesh presence; not a full glTF validator."""
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError("Cabecera GLB incompleta.")
        magic, version, length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2 or length != path.stat().st_size:
            raise ValueError("GLB 2 inválido o truncado.")
        chunk = stream.read(8)
        if len(chunk) != 8:
            raise ValueError("GLB sin bloque JSON.")
        size, kind = struct.unpack("<I4s", chunk)
        if kind != b"JSON" or size % 4 or size > length - 20 or size > 64 * 1024 * 1024:
            raise ValueError("Bloque JSON GLB inválido.")
        data = json.loads(stream.read(size))
        if not isinstance(data, dict) or not data.get("meshes"):
            raise ValueError("GLB sin mallas.")
        while stream.tell() < length:
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ValueError("Bloque GLB truncado.")
            size, _ = struct.unpack("<I4s", chunk)
            if size % 4 or size > length - stream.tell():
                raise ValueError("Longitud de bloque GLB inválida.")
            stream.seek(size, 1)
    return data

def _relative(value, parent):
    if not value:
        return ""
    path = Path(value).expanduser().resolve()
    try:
        return os.path.relpath(path, parent)
    except ValueError:  # Windows volumes differ
        return str(path)

def save_project(path, *, image, model, output, steps, faces, seed, guidance):
    validate_settings(steps, faces, seed, guidance)
    path = Path(path).resolve()
    if path.suffix.lower() != ".aeth":
        raise ValueError("El proyecto debe tener extensión .aeth.")
    data = {"format": "AETHRYX", "version": 1,
            "image": _relative(image, path.parent), "model": _relative(model, path.parent),
            "output": _relative(output, path.parent),
            "settings": {"steps": steps, "faces": faces, "seed": seed, "guidance": guidance}}
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temp = handle.name
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)

def load_project(path):
    path = Path(path).resolve()
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Proyecto demasiado grande.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != "AETHRYX" or data.get("version") != 1:
        raise ValueError("Formato de proyecto no compatible.")
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Proyecto sin configuración.")
    validate_settings(*(settings.get(key) for key in ("steps", "faces", "seed", "guidance")))
    result = {"settings": settings}
    for key in ("image", "model", "output"):
        value = data.get(key, "")
        if not isinstance(value, str):
            raise ValueError("Ruta de proyecto inválida.")
        result[key] = str((path.parent / value).resolve()) if value else ""
    for key in ("image", "model"):
        if result[key] and not Path(result[key]).is_file():
            raise ValueError("Recurso no encontrado: " + result[key])
    if result["output"] and Path(result["output"]).suffix.lower() != ".glb":
        raise ValueError("La salida del proyecto debe ser GLB.")
    return result
