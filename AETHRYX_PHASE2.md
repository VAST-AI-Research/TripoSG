# AETHRYX Studio — siguiente fase: estabilidad y proyectos
Basado en copilot/development f3097e7. Se mantiene Tkinter/Matplotlib y el motor existente. Sin nuevas dependencias ni Blender.
- Proyectos .aeth v1: JSON con referencias externas relativas cuando es posible, parámetros de generación y rutas; no empaqueta recursos ni guarda ediciones de malla, materiales o animaciones. Mover un proyecto requiere conservar sus recursos. No ejecuta comandos al abrirlo.
- Semilla, guidance y opción sin decimación (-1) por defecto. Modelos más grandes aumentan tiempo/memoria; el visor mantiene su reducción de vista previa.
- Bloqueo de generación concurrente, cierre durante trabajo, confirmación de reemplazo, indicador de actividad no porcentual y comprobación básica del GLB antes de publicar la salida.
- Para GLB importados, exportación de copia binaria original, sin reconstrucción por Trimesh. No es un editor de rig; archivos de otros formatos se convierten como escena estática.
- Pendiente: pruebas GUI, GPU, cancelación, ejes/cámaras, visor acelerado por GPU, edición de geometría, rigging, texturas, FBX y validación Unity. El cierre durante diagnóstico en segundo plano y cancelación de subprocesos descendientes requieren pruebas adicionales.
Pruebas unitarias propuestas: python -m unittest discover -s tests -p "test_studio_project.py"
Prueba manual: python app.py desde el entorno instalado; abrir/guardar .aeth, reabrir, generar dos veces, cancelar, confirmar que no se sobrescribe sin permiso, exportar copia GLB y comparar hashes.
