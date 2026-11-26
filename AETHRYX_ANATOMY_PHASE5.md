# AETHRYX Anatomy — fase funcional 5

## Progreso implementado

- Scene graph humanoide persistible en proyectos `.aeth` versión 2.
- Nodos raíz, cuerpo, cabeza, ojos, cabello, ropa, accesorios y armature.
- Estados reales: disponible, detectado pendiente de confirmación,
  malla importada no segmentada y rig no disponible.
- Herramienta KIRIA `get_avatar_scene_graph` de solo lectura.
- El panel Escena expone el estado de los componentes.
- Regiones derivadas de pose MediaPipe para rostro, ojos, ropa y cabello.
- Máscaras RLE conservadas para las regiones derivadas.

## Interpretación anatómica

Las regiones derivadas de pose son candidatos geométricos, no reconstrucciones
exactas. Se etiquetan `mediapipe-pose-heuristic` y requieren confirmación manual.

- Rostro: recorte alrededor de puntos faciales disponibles.
- Ojos: recorte entre landmarks laterales/circulares disponibles.
- Cabello: región ampliada sobre los landmarks superiores de la cabeza.
- Ropa: recorte entre hombros y caderas.
- Accesorios: no inferidos.
- Rig humanoide: no creado ni anunciado.

No se declara separación semántica exacta hasta disponer de una máscara
multiclase probada para cada componente.

## Proyectos

Los `.aeth` guardan `avatar_nodes` y referencias versionadas.
Los manifiestos versión 1 siguen cargándose; no se elimina compatibilidad
hacia atrás.

## Hardware

A partir de esta fase se conserva como referencia exclusiva la RTX 5070 de
12 GiB con PyTorch/CUDA compatible. Los perfiles 4/6/8/16/24 GiB permanecen
pendientes y no se presentan como probados.
