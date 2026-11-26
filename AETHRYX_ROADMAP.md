# AETHRYX Studio — encargo de producto para Copilot

Nombre visible: AETHRYX Studio (provisional hasta verificar marca). Marca empresarial de trabajo: NIKIX. Asistente futuro: KIRIA. Eslogan: Más allá de la imaginación, más allá de las dimensiones. Diseño 3D con IA local para Unity/VRChat y futuros usos. Centralizar metadatos de producto, cambiar títulos, launcher y documentación de usuario, conservar TripoSG como motor interno y sus créditos/licencias. No inventar datos legales de la empresa ni renombrar el repositorio remoto.

## Fase 1: interfaz profesional
Tema oscuro, rosa #FF268F, celeste #00D9FF, blanco/dorado de acento, iconos consistentes, DPI, paneles redimensionables. Árbol de escena, visor central, inspector, consola y estado GPU. Menús nativos:
- Archivo: nuevo/abrir/guardar proyecto, importar imagen/malla, exportar, recientes, salir.
- Editar: deshacer/rehacer, duplicar, borrar, preferencias.
- Seleccionar: todo/nada/invertir, modos objeto/vértice/arista/cara.
- Vista: frontal/lateral/trasera/superior, ortográfica/perspectiva, encuadrar, rejilla, sólido/wireframe/materiales/huesos.
- Imagen: referencia, recorte, fondo, resolución, generación IA.
- Capas/Escena: jerarquía, grupos, renombrar, ocultar, aislar, bloquear.
- Colores/Materiales: paleta, UV, PBR, normales, emisión.
- Herramientas: transformación, reparación, simetría, retopología, rigging, editor facial.
- Filtros: limpieza, suavizado, reducción, análisis de malla con vista previa.
- Animación: pose de reposo, huesos, pesos, pruebas de movimiento, ojos, visemas.
- Ventanas: paneles, consola, restablecer espacio.
- Ayuda: manual, diagnóstico, créditos, licencias, Acerca de AETHRYX (marca NIKIX, versión real, descripción, créditos; no inventar registro, web, domicilio).
Atajos solo si funcionan. Ocultar o marcar no disponibles las acciones pendientes. Proyectos .aeth (propuesta, esquema versionado, rutas relativas, recursos, guardado seguro), separados de la exportación.

## Fase 2: robustez y visor
Partir de copilot/development c95d9d4, app.py Tkinter/Matplotlib; comparar con kiria/studio-mvp sin fusionar automáticamente. Evaluar visor GPU/editor y PySide6 gradualmente. Mantener lo que funciona. Respetar transformaciones de nodos GLB, jerarquía y materiales; decimación de vista previa no debe alterar original. Evitar cargas concurrentes obsoletas, validar entradas y GLB, capturar parámetros en hilo UI, proceso cancelable, logs acotados, rutas portables en vez de pythonw.exe fijo, salida única/atómica. Diagnóstico CUDA/VRAM/pesos y modo offline; no instalar sin autorización. Estadísticas reales, no inventar progreso. Validar normales, agujeros/no manifold, escala/ejes y componentes.

## Fase 3: editor y automatización de avatares
- Limpieza, simetría, proporciones, retopología anatómica (no solo decimación), separar cuerpo, pelo, ojos, dientes/lengua, ropa cuando sea viable; preservar originales y permitir correcciones manuales. No afirmar reconstrucción exacta de geometría ausente.
- UV y texturas PBR: color base, normal, roughness, metallic, emisión. Realismo de geometría y materiales configurable sin perder estilo anime.
- Auto-rig por adaptador (evaluar UniRig y licencia): jerarquía humanoide, dedos, ojos, mandíbula, cabello opcional, pesos normalizados, pose T/A, pruebas de deformación de hombros/codos/rodillas/dedos.
- Ojos independientes, pivotes y mirada; parpadeo, mandíbula, boca, expresiones y visemas VRChat (15 si se elige ese modo de lipsync), blendshapes reales y previsualización de animación. No simular con texturas.
- Asistente de flujo automático con checkpoints, cancelación, recuperación y edición por etapa. Sin Blender como requisito, ni GUI ni backend obligatorio.
- Exportación GLB/OBJ inicial; GLB riggeado intermedio; FBX con skin, bind pose, morphs, materiales y animaciones mediante SDK/biblioteca compatible y licenciada. OBJ solo estático. Validar en Unity escala/ejes, Humanoid, mapeo de huesos, texturas. Complemento Unity opcional para Avatar Descriptor, eye look, lip sync, PhysBones, Poiyomi si disponible y UnityPackage desde Unity. No afirmar avatar listo hasta validación SDK.

## Fase 4: comercialización
Inventario de licencias de código y pesos, NOTICE y atribuciones; revisar BRIA RMBG 1.4 y alternativas comerciales, licencias de rigging/exportadores. Marca AETHRYX provisional por posible coincidencia; verificar antes del lanzamiento. Arquitectura modular para futuras ediciones Community/Professional, sin DRM ni cobros todavía. Privacidad local; no publicar imágenes personales, pesos, credenciales ni entornos virtuales.

## Entrega inicial
Priorizar identidad, menús funcionales y robustez del visor. Documentar pruebas reales y pendientes, captura de interfaz, instrucciones. Trabajar en copilot/development, abrir PR para revisión, no fusionar main ni modificar kiria/studio-mvp. Las fases avanzadas son hoja de ruta, no se exige completar todo en un commit.