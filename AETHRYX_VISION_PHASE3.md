# AETHRYX Vision — tercera fase

## Validación

- La suite existente pasa con 22 pruebas en el entorno local.
- NumPy es aceptado por `validate_bbox` mediante `numbers.Real/Integral`.
- MediaPipe 0.10.35 expone la API Tasks, no `mp.solutions`.
- No se descargaron ni incorporaron pesos de MediaPipe. Sin
  `AETHRYX_MEDIAPIPE_POSE_MODEL` o `AETHRYX_MEDIAPIPE_FACE_MODEL`, los resultados
  Tasks se reportan explícitamente como no disponibles.
- La prueba sintética estilo anime produjo solo el resultado real disponible por
  Haar o el estado no disponible de MediaPipe; no se etiquetó ninguna visión
  completa ni característica especial como automática.

## Funcionamiento real

- Análisis Haar en hilo secundario.
- Reclasificación y reemplazo manual de recortes.
- Selección determinista de una vista principal entre varias imágenes:
  confirma preferencia frontal, tres cuartos, lateral y posterior.
- Puntos MediaPipe solamente con archivos de modelo locales configurados.
- Segmentación semántica únicamente con un modelo ONNX local configurado por
  variables de entorno. Sin modelo no produce máscaras ni regiones.
- Diagnóstico local basado en referencias y estadísticas reales de malla.

## Configuración opcional

No descarga modelos automáticamente:

```powershell
$env:AETHRYX_MEDIAPIPE_POSE_MODEL="C:\ruta\pose_landmarker.task"
$env:AETHRYX_MEDIAPIPE_FACE_MODEL="C:\ruta\face_landmarker.task"
$env:AETHRYX_SEMANTIC_SEGMENTATION_MODEL="C:\ruta\segmentacion.onnx"
$env:AETHRYX_SEMANTIC_SEGMENTATION_LABELS="body,face,hair,eye,mouth,clothing,accessory"
```

Las etiquetas provistas deben corresponder a índices reales del modelo.

## Evaluación de reconstrucción multivista

| Motor | Estado | Licencia detectada | Compatibilidad |
|---|---|---|---|
| TripoSG | Disponible en este repositorio | MIT de código; revisar pesos/activos | Monovista |
| LGM | Candidato no integrado | MIT de código | CUDA/PyTorch por medir; no especializado en personajes |
| Unique3D | Candidato no integrado | MIT de código | Windows reportado por upstream; dependencias Triton/CUDA por medir |
| Motor humanoide especializado | No identificado con licencia/verificación suficientes | No evaluado | No integrar |

RTX 5070 observada: 12.227 MiB VRAM totales. Esto no confirma compatibilidad,
consumo ni licencias de pesos para LGM/Unique3D.

No se descargaron pesos de reconstrucción ni se realizó una generación 3D con estos
candidatos. TripoSG no se ejecutó en esta fase porque el entorno local no tiene
PyTorch y los pesos son grandes; descargar ambos requeriría preparación explícita.

## Limitación actual

- No hay segmentación validada con un personaje anime real.
- No hay generación multivista de avatar humanoide probada.
- Matplotlib/Tk sigue siendo el visor actual; no se certifica aceleración GPU.
- No hay rig, skinning, blendshapes ni exportación Unity/VRChat probados.
