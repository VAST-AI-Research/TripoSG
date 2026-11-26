# AETHRYX Genesis — Character Quality Phase 3

## Referencia

Se creó una referencia original de 800×1200 con figura completa, pose T,
manos/pies visibles, fondo limpio y perspectiva frontal. No se reutilizó como
entrada principal la imagen oscura de AniSeg.

Archivos:

- `test_assets/anime/genesis-phase3/original-anime-tpose.png`
- `test_assets/anime/genesis-phase3/original-anime-tpose-alpha.png`

ISNet produjo el canal alfa local. Si ISNet no está disponible o no produce
una máscara válida, AETHRYX cancela la generación y muestra el error; no
continúa con una imagen sin alfa.

## Generación

- Motor: TripoSG.
- Pasos: 75.
- Semilla: 42.
- Guidance: 7.
- Decimación: desactivada (`--faces -1`).
- Adaptive Engine: perfil RTX 5070 de 12 GiB, FP16.
- CPU offload: no aplicado porque TripoSG no expone `model_cpu_offload_seq`.
- Tiempo: 69.21 s.

## VRAM independiente

- `nvidia-smi` pico total: 11.298 GiB (11.298 MiB? nvidia-smi reported 11298 MiB).
- PyTorch allocated al final: 3.798 GiB.
- PyTorch reserved al final: 5.112 GiB.
- PyTorch peak allocated: 5.330 GiB.

Estas mediciones son independientes y no deben confundirse.

## Malla

- Vértices: 284.697.
- Caras: 569.394.
- Watertight: sí.
- Bounds: X -0.671..0.673, Y -0.953..0.951, Z -0.258..0.256.

Archivo:

- `test_assets/anime/genesis-phase3/genesis-phase3-75-original.glb`

## Cámaras corregidas

El script reproducible `scripts/render_mesh_cameras.py` usa:

- Frontal: eje Z positivo.
- Lateral: eje X positivo.
- Posterior: eje Z negativo.
- Eje Y vertical.

Renders:

- `genesis-phase3-75-front.png`
- `genesis-phase3-75-side.png`
- `genesis-phase3-75-rear.png`

Las vistas muestran al personaje completo, vertical y sin recortes.

## Calidad observada

La figura es reconocible como humanoide T-pose, pero conserva defectos:

- Cara simplificada y sin detalle fiable.
- Cabello no separado.
- Manos y pies visibles pero deformados.
- Anatomía interna no verificable.
- Ropa y accesorios no son mallas independientes.
- No hay rig, skinning ni exportación VRChat probada.

## Reparación y topología

Operaciones reales sobre copias:

- `repair_mesh_copy`: eliminó una cara duplicada; 569.393 caras restantes;
  watertight quedó negativo, por lo que no se declara reparación completa.
- `analyze_mesh_symmetry`: simetría X media 0.004985417 y máxima
  0.026089532 unidades.
- `export_disconnected_components`: exportó un componente GLB desconectado
  real. Sigue sin asignar anatomía.

La separación por componentes no convierte un componente unido en cabeza,
brazo, ropa u ojo; esas categorías requieren segmentación/modelado específico.

## Motor multivista

LGM y Unique3D permanecen como candidatos no integrados. No se descargaron
pesos ni se declaró compatibilidad VRAM/comercial hasta una prueba real.
