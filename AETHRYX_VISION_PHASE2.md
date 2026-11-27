# AETHRYX Vision — segunda fase

## Implementado

- Detección Haar ejecutada fuera del hilo Tk y aplicada al volver al hilo de interfaz.
- Reclasificación manual de regiones y reemplazo de recortes mediante selección + arrastre.
- Procedencia ampliada para `opencv-haar`, `mediapipe-face` y `mediapipe-pose`.
- Puntos de pose y región corporal cuando MediaPipe está disponible.
- Contrato ampliado de motores con candidatos multivista y estado no integrado.
- Registro tipado de herramientas con permisos de lectura y contexto de proyecto.
- Diagnóstico local determinista basado en referencias, malla y archivo de proyecto.

## Reconocimiento visual

Haar permanece como alternativa básica de rostro frontal. MediaPipe es un proveedor
local opcional compatible con retratos y puede aportar malla facial o articulaciones.
Su precisión en personajes anime debe medirse con fixtures sintéticos; no se etiqueta
como especializado en anime ni se promete una segmentación semántica completa.

### Segmentación progresiva

- Real: regiones faciales y de ojos mediante OpenCV Haar.
- Real cuando MediaPipe esté instalado: región facial y puntos corporales.
- No implementado: máscara de cabello, ropa, accesorios y manos/pies.
- No implementado: separación anatómica visible/oculta ni reconstrucción de superficies.

### Pose

Las articulaciones MediaPipe se conservan como puntos normalizados y metadatos.
No se convierte esa hipótesis en huesos, skinning ni animación de avatar.

## Motor de reconstrucción

TripoSG continúa como motor monovista disponible. Se mantienen los candidatos
LGM y Unique3D como entradas de evaluación, no como motores integrados.
Ningún candidato se instala ni descarga automáticamente.

La comparación comercial requiere confirmar código, pesos, datos de entrenamiento,
licencia de redistribución, VRAM, Windows/RTX y soporte de rig humanoide.

## Visor 3D

El visor actual usa Matplotlib/Tk. Esta fase comienza la migración mediante
contratos de backend; no se afirma aceleración GPU ni se añade un falso renderador.

## Herramientas KIRIA

Las herramientas actuales son de solo lectura y usan esquemas básicos. Toda operación
de modificación exige permiso explícito y no tiene ejecutor disponible todavía.

## Licencias relevantes

- OpenCV: Apache 2.0 para el runtime; revisar versiones y clasificadores incluidos.
- MediaPipe: Apache 2.0 para el runtime; revisar los modelos y assets incluidos antes
  de redistribución comercial.
- TripoSG: licencia upstream y pesos pendientes de revisión.
- RMBG 1.4: licencia upstream pendiente de revisión.
- LGM/Unique3D: candidatos no integrados y no evaluados comercialmente.
