# AETHRYX Genesis — Character Quality Phase 2

## Hardware y generación

Hardware confirmado: NVIDIA RTX 5070, 12 GiB VRAM, Windows, PyTorch
`2.11.0+cu128`, CUDA disponible.

El proceso de generación desde AETHRYX Studio activa:

- Perfil Adaptive Engine FP16.
- ISNet local como preparación predeterminada de canal alfa.
- Sin RMBG.
- Sin decimación cuando `--faces -1`.
- Reintento y limpieza ante CUDA OOM.

## CPU offloading

La prueba real encontró que TripoSG no expone `model_cpu_offload_seq` y el
pipeline personalizado rechazó `enable_model_cpu_offload`. Por ello el perfil 12
GiB marca el CPU offloading como no disponible para TripoSG en esta versión.
No se fuerza un método incompatible. La gestión de memoria conserva limpieza
CUDA y reintento único ante OOM.

## Referencia

Se conserva una referencia anime autorizada de AniSeg. La imagen usada no cumple
una pose A/T limpia ni una hoja de vistas: no se etiqueta como tal.
El canal alfa fue producido por ISNet isnetis ONNX Apache-2.0.

## Comparación 50/75 pasos

| Pasos | Decimación | Tiempo | VRAM alloc pico | Vértices | Caras | Watertight |
|---:|---|---:|---:|---:|---:|---|
| 50 | Ninguna | 64.88 s | 5.345 MiB | 350.480 | 700.884 | No |
| 75 | Ninguna | 77.60 s | 5.347 MiB | 342.252 | 684.448 | Sí |

La generación 75 produjo una malla cerrada y se conserva como geometría original.

## Copias de malla

| Caras | Vértices | Watertight | Estado |
|---:|---:|---|---|
| 25.000 | 12.482 | No | Copia de revisión |
| 100.000 | 49.997 | No | Copia de revisión |
| Original 75 | 342.252 | Sí | Geometría original preservada |

25.000 y 100.000 son reducciones posteriores del original; no se aplicó
decimación durante la generación original.

## Calidad observada

Las vistas 75 frontal, lateral y posterior son verificables en
`test_assets/anime/`.

- Rostro: reconocible, pero con proporciones y estructura anatómica defectuosas.
- Cabello: sugerido por el volumen exterior, no separado como malla.
- Manos: deformadas o fusionadas en varias vistas.
- Pies: presentes visualmente pero no validados ni editables.
- Anatomía: no cumple todavía un avatar humanoide VRChat.
- La malla original 75 está cerrada, pero no contiene rig, skin, ojos separados,
  cabello separado ni ropa/accesorios independientes.

## Separación tridimensional

Existe separación topológica real de componentes desconectados mediante
trimesh. Cada componente queda marcado como `unassigned`; no se asigna
cabeza/cuerpo/ropa por posición o silueta. La separación semántica 3D y la
reconstrucción anatómica siguen pendientes.

## Motor multivista

No se integró LGM/Unique3D en esta fase. Se requieren licencia comercial de
pesos, compatibilidad Windows/CUDA 12.8 y una prueba controlada con RTX 5070
antes de reemplazar TripoSG.
