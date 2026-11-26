# AETHRYX Vision — primera fase probada

Esta entrega implementa el primer tramo real y local del Character Sheet Analyzer.
No reconstruye vistas faltantes ni fusiona varias imágenes.

## Capacidades reales

- Detecta rostros frontales y posibles regiones de ojos con los clasificadores Haar
  incluidos por `opencv-python`.
- Muestra esas detecciones como recortes pendientes de aprobación.
- Permite confirmarlas manualmente, reasignarlas y elegir una vista principal.
- Conserva la procedencia `manual` o `opencv-haar`, confirmación y confianza opcional
  en el manifiesto `.aeth`.
- Mantiene TripoSG como motor monovista: el flujo aprobado usa solamente el recorte
  frontal o la imagen completa si no existe recorte principal.
- Informa vistas completas sin confirmar como faltantes; no las predice ni las inventa.
- Expone un catálogo de motores y un registro tipado de herramientas de lectura.

## Estados que no deben interpretarse como implementados

- Clasificación lateral, posterior o tres cuartos: no disponible.
- Detección de boca, cabello, ropa y accesorios: no disponible.
- Asociación automática de varias imágenes al mismo personaje: no disponible.
- Reconstrucción multivista y rigging especializado: no disponible.
- Servidor MCP, proveedores de IA remotos y herramientas de edición: no disponibles.
- Visor GPU acelerado específico de AETHRYX: no integrado.

## Motores evaluables

El registro conserva el contrato de capacidades de cada motor:

| Motor | Estado | Capacidad |
|---|---|---|
| TripoSG | Disponible | Una imagen |
| Motor multivista humanoide especializado | No integrado | Interfaz pendiente de evaluación |

No concatenar recortes en hojas ni enviar una hoja completa como multivista.
Registrar en el proyecto qué imagen o recorte recibió realmente el motor.

## Licencias y comercialización

Este archivo es inventario técnico, no dictamen legal.

- OpenCV: `opencv-python` se distribuye bajo Apache 2.0 en el proyecto OpenCV;
  revisar la licencia exacta de la versión y de los clasificadores instalados antes
  de redistribuir una aplicación comercial.
- TripoSG: revisar licencia del código, pesos y activos del modelo antes de uso
  comercial o redistribución.
- RMBG 1.4: revisar la licencia upstream y el peso antes de redistribuir.
- Modelos locales o remotos futuros deben agregarse al inventario con código,
  pesos, dataset, runtime y restricciones antes de marcarse disponibles.

No descargar pesos nuevos ni incorporar datos personales mediante esta fase.

## Ejecución local

`python -m unittest discover -s tests`

OpenCV y NumPy son las únicas dependencias nuevas del análisis automático ya usadas
por el proyecto. La ausencia de OpenCV o de sus clasificadores produce un estado
`unavailable`, no una detección falsa.
