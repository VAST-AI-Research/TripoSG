import os
import subprocess
import sys


EXE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = EXE_DIR

candidate = EXE_DIR
for _ in range(3):
    if os.path.isfile(os.path.join(candidate, "app.py")):
        ROOT = candidate
        break
    candidate = os.path.dirname(candidate)

PYTHON = os.path.join(ROOT, ".venv", "Scripts", "pythonw.exe")
APP = os.path.join(ROOT, "app.py")


if not os.path.isfile(PYTHON):
    sys.exit("No se encontro el entorno Python de AETHRYX Studio.")

subprocess.Popen(
    [PYTHON, APP],
    cwd=ROOT,
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
    | getattr(subprocess, "DETACHED_PROCESS", 0)
    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
)
