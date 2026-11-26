# AETHRYX Vision — fase funcional 4

## Entorno confirmado

- PyTorch `2.11.0+cu128`, CUDA disponible en NVIDIA RTX 5070.
- MediaPipe `0.10.35`, API Tasks confirmada.
- ONNX Runtime `1.23.2`, OpenCV, Trimesh y PyVista/VTK instalados.
- OpenCV DNN no soportó el ONNX anime por una diferencia de `Concat`;
  se sustituyó por ONNX Runtime CPU.
- ONNX Runtime intentó CUDA pero la DLL cuFFT 11 faltaba; el modelo quedó
  registrado con `CPUExecutionProvider`.
- PyVista/VTK reportó la RTX 5070 como `NVIDIA GeForce RTX 5070/PCIe/SSE2`
  y `vtkWin32OpenGLRenderWindow`.

## Modelos locales descargados

| Modelo | Licencia | Tamaño | Uso |
|---|---:|---:|---|
| MediaPipe pose landmarker heavy | Apache-2.0 | 30.7 MB | Pose; 33 puntos en la prueba |
| MediaPipe face landmarker | Apache-2.0 | 3.7 MB | Rostro/facial; pendiente de precisión en anime |
| SkyTNT anime-seg isnetis ONNX | Apache-2.0 | 176 MB | Máscara binaria character/background |
| TripoSG weights | MIT card | ~7.9 GB | Generación monovista de prueba |

RMBG 1.4 no se descargó: su licencia es no comercial/fuente disponible con
acuerdo comercial para uso comercial. TripoSG se ejecutó con una máscara alfa
Apache-2.0 generada por ISNet y sin RMBG.

## Prueba visual real

Se usó una imagen anime pública de `jerryli27/AniSeg`, Apache-2.0.
Los proveedores locales produjeron:

- Pose MediaPipe: 33 puntos y región corporal.
- Segmentación ONNX: máscara RLE completa de carácter/background.
- `authorized-anime-vision-review.png`: evidencia visual verificable.

## Generación TripoSG

- Entrada: PNG RGBA con máscara alpha ISNet.
- Pasos: 20.
- Semilla: 42.
- Límite: 5.000 caras.
- Tiempo: 65.47 segundos.
- VRAM pico observada: 11.480 MiB de 12.227 MiB.
- Resultado: 2.476 vértices y 5.000 caras.
- Watertight: no.
- Límites: aproximadamente -0.609..0.605 X, -0.952..0.950 Y,
  -0.425..0.401 Z.

El render PyVista muestra una figura reconocible pero defectos visibles:
proporciones irregulares, extremidades deformadas, anatomía incompleta y
geometría no cerrada. No se etiqueta como avatar listo.

## Limitaciones y próximos pasos

- El ONNX de SkyTNT es binario: separa carácter/fondo, no cabello, ropa, ojos
  o accesorios por clase.
- MediaPipe aporta cara/pose pero no produce aún máscaras semánticas completas
  de las partes solicitadas.
- Falta un modelo ONNX multiclase anime autorizado y probado.
- No se integró LGM/Unique3D para generación multivista: requieren clonado,
  pesos, instalación CUDA y prueba antes de integrarse.
- El visor PyVista/VTK está disponible para render, pero falta benchmark
  interactivo y operaciones de edición/transacción.
