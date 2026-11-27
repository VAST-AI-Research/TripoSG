# AETHRYX Adaptive Engine

## Alcance actual

AETHRYX Adaptive Engine es un requisito transversal que selecciona perfiles según
hardware local y aplica políticas verificables al cargar o ejecutar motores.

## Hardware detectado

En la sesión RTX 5070 probada:

- GPU: NVIDIA GeForce RTX 5070
- VRAM: 12.227 GiB totales
- RAM: detectada automáticamente desde `psutil`
- CPU: núcleos detectados automáticamente
- Almacenamiento: espacio libre del directorio de trabajo

La prueba TripoSG observó 11.480 GiB de pico VRAM, por lo que el sistema selecciona
el perfil objetivo de 12 GiB.

## Perfiles

| Objetivo VRAM | Precisión | CPU offload | Carga secuencial | Estado |
|---:|---|---|---|---|
| 4 GiB | FP32/FP16 solo si el motor lo acepta | Sí | Sí | No probado |
| 6 GiB | FP32/FP16 solo si el motor lo acepta | Sí | Sí | No probado |
| 8 GiB | FP32/FP16 solo si el motor lo acepta | Sí | Sí | No probado |
| 12 GiB | FP16 | Sí | No | TripoSG probado |
| 16 GiB | FP16 | Sí | No | No probado |
| 24 GiB | FP16 | No | No | No probado |

Los perfiles no declaran compatibilidad de equipos que no hayan ejecutado el motor.

## Políticas implementadas

- Detección RAM/VRAM/CPU/GPU/almacenamiento libre.
- Selección de perfil objetivo.
- Carga/modelo CPU offload disponible en Diffusers.
- Carga secuencial para perfiles <=8 GiB.
- Memoria compartida y limpieza CUDA con `empty_cache`/`ipc_collect`.
- Descarga y liberación de objetos al finalizar.
- Reintento único ante CUDA OOM mediante caché limpia y offload secuencial.
- TripoSG conserva FP16 en GPU y FP32 en CPU.
- INT8/INT4 permanecen marcados como `none` o `engine-dependent`, nunca como
  compatibles con TripoSG sin una implementación probada.
- RAM <8 GiB activa modo edición básica y deshabilita generación.
- Prioridad de calidad: no reduce pasos ni recorta resolución por defecto.

## Benchmark comparativo

Prueba real registrada:

- Motor: TripoSG
- Entrada: personaje anime autorizado con máscara alfa.
- Pasos: 20
- Caras: 5.000
- Tiempo: 65.47 s
- VRAM pico: 11.480 GiB
- Resultado: 2.476 vértices / 5.000 caras
- Estado: válido como geometría de prueba, no watertight y no riggeada

Cada perfil futuro debe registrar las mismas métricas con el mismo asset:
tiempo, calidad visual, VRAM pico, VRAM final, caras, vértices, fallos y OOM.

## Limitaciones

- No existe soporte confirmado de INT8/INT4 en TripoSG.
- 4, 6, 8, 16 y 24 GiB no están certificados.
- CPU offloading puede aumentar significativamente el tiempo.
- El modo de edición básica no prueba todavía el visor PyVista con <=8 GiB RAM.
- 4, 6, 8, 16 y 24 GiB quedan suspendidos por decisión del director.
- El hardware de referencia activo es exclusivamente RTX 5070 de 12 GiB
  hasta completar el avatar humanoide VRChat.
- TripoSG no expone `model_cpu_offload_seq`; el CPU offload y la carga secuencial
  no se aplican al pipeline personalizado hasta que el motor declare soporte.
