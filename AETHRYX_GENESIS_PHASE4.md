# AETHRYX Genesis — Character Quality Phase 4

## Referencia

Se seleccionó `test_assets/anime/genesis-phase4/reference/authorized-anime-799906.jpg`,
una ilustración detallada del repositorio AniSeg. El repositorio AniSeg está
licenciado Apache-2.0.

Esta referencia conserva un pose dinámico y no cubre pies ni una pose neutral
completa. No se declara como referencia perfecta para entrenamiento anatómico.

## Motor alternativo

Se evaluó TripoSR:

- Código MIT: VAST-AI-Research/TripoSR.
- Pesos MIT según README/model card.
- Peso local: 1.677.246.742 bytes (aprox. 1.56 GiB).
- Tiempo/VRAM no publicados como prueba válida hasta pasar generación.
- Resultado: no integrado en Windows debido a fallo de compilación de
  `torchmcubes` aun con MSVC disponible.
- No reemplaza TripoSG ni se declara compatible con la RTX 5070.

LGM/Unique3D siguen como candidatos no integrados. No se descargaron pesos
adicionales ni se publicó una comparación falsa.

## Retopología

`retopologize_mesh_copy` realiza reducción quadric real sobre una copia.
Rechaza el resultado si las dimensiones cambian más de 5%.

## Reparación segura

`repair_mesh_copy` limpia duplicados y caras degeneradas sobre una copia.
Rechaza exportar si altera las dimensiones más de 5%. No modifica el original.

## Componentes

Los componentes topológicos reales pueden exportarse como GLB independientes.
La aplicación los importa al scene graph como objetos 3D seleccionables.
El componente no recibe automáticamente etiqueta anatómica.

## Edición y deshacer

AETHRYX conserva hasta 20 checkpoints de escena en memoria y ofrece
Deshacer/Rehacer por menú y Ctrl+Z/Ctrl+Y.

La confirmación y preview son deterministas para operaciones de copia:
el original permanece disponible y cada salida debe validarse antes de cargarla.

## Hardware

La única referencia activa sigue siendo RTX 5070 de 12 GiB, Windows y CUDA
compatible. No se probarán ni certificará hardware inferior en esta fase.
