import os
import re
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(ROOT, ".venv", "Scripts", "pythonw.exe")


class TripoSGApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nikix 3D Studio")
        self.geometry("1100x850")
        self.minsize(980, 740)

        self.image_path = tk.StringVar(value=os.path.join(ROOT, "assets", "example_data", "hjswed.png"))
        self.output_path = tk.StringVar(value=os.path.join(ROOT, "output.glb"))
        self.steps = tk.IntVar(value=50)
        self.faces = tk.IntVar(value=5000)
        self.preview_image = None
        self.current_model_path = None

        self._build_ui()
        self._load_preview()
        self._create_model_viewer()
        default_model = self.output_path.get()
        if os.path.isfile(default_model):
            self.load_model(default_model)

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        ttk.Label(self, text="Nikix 3D Studio", font=("Segoe UI", 18, "bold")).pack(pady=(10, 6))

        frame = ttk.Frame(self)
        frame.pack(fill="x", **pad)
        ttk.Label(frame, text="Imagen:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.image_path).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Elegir...", command=self.choose_image).grid(row=0, column=2)
        ttk.Label(frame, text="Salida:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.output_path).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Guardar como...", command=self.choose_output).grid(row=1, column=2)
        frame.columnconfigure(1, weight=1)

        settings = ttk.Frame(self)
        settings.pack(fill="x", **pad)
        ttk.Label(settings, text="Pasos:").grid(row=0, column=0)
        ttk.Spinbox(settings, from_=10, to=100, textvariable=self.steps, width=8).grid(row=0, column=1, padx=(6, 20))
        ttk.Label(settings, text="Caras:").grid(row=0, column=2)
        ttk.Spinbox(settings, from_=1000, to=100000, increment=1000, textvariable=self.faces, width=10).grid(row=0, column=3, padx=6)

        preview_header = ttk.Frame(self)
        preview_header.pack(fill="x", padx=12, pady=(4, 0))
        self.preview = ttk.Label(preview_header, text="Imagen de entrada")
        self.preview.pack(side="left")
        ttk.Button(preview_header, text="Abrir modelo 3D...", command=self.choose_model).pack(side="right")

        self.viewer_frame = ttk.Frame(self)
        self.viewer_frame.pack(fill="both", expand=True, padx=12, pady=4)

        self.status = tk.StringVar(value="Listo")
        ttk.Label(self, textvariable=self.status).pack()
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=12, pady=(4, 2))
        self.progress["value"] = 0

        self.log = tk.Text(self, height=5, width=110, wrap="word", state="disabled")
        self.log.pack(fill="x", padx=12, pady=4)

        self.generate_button = ttk.Button(self, text="Generar modelo 3D", command=self.generate)
        self.generate_button.pack(pady=12)
        self.open_button = ttk.Button(self, text="Abrir carpeta", command=self.open_output_folder, state="disabled")
        self.open_button.pack(pady=(0, 10))

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Elegir imagen",
            filetypes=[("Imagen", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Todos", "*.*")],
        )
        if path:
            self.image_path.set(path)
            self._load_preview()

    def choose_output(self):
        path = filedialog.asksaveasfilename(
            title="Guardar modelo 3D",
            defaultextension=".glb",
            filetypes=[("Modelo GLB", "*.glb"), ("Todos", "*.*")],
        )
        if path:
            self.output_path.set(path)

    def _load_preview(self):
        path = self.image_path.get()
        if not os.path.isfile(path):
            return
        image = Image.open(path)
        image.thumbnail((620, 360))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_image)

    def _create_model_viewer(self):
        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.figure.patch.set_facecolor("#f4f4f4")
        self.model_axes = self.figure.add_subplot(111, projection="3d")
        self.model_axes.set_title("Arrastra para girar; usa la barra para mover y acercar")
        self.model_axes.set_axis_off()
        self.model_canvas = FigureCanvasTkAgg(self.figure, master=self.viewer_frame)
        self.model_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.model_canvas, self.viewer_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")
        self.model_canvas.mpl_connect("scroll_event", self._zoom_model)

    def _zoom_model(self, event):
        if event.inaxes is not self.model_axes or event.button not in ("up", "down"):
            return
        factor = 0.85 if event.button == "up" else 1.18
        for getter, setter in (
            (self.model_axes.get_xlim3d, self.model_axes.set_xlim3d),
            (self.model_axes.get_ylim3d, self.model_axes.set_ylim3d),
            (self.model_axes.get_zlim3d, self.model_axes.set_zlim3d),
        ):
            low, high = getter()
            center = (low + high) / 2
            half_range = (high - low) * factor / 2
            setter(center - half_range, center + half_range)
        self.model_canvas.draw_idle()

    def choose_model(self):
        path = filedialog.askopenfilename(
            title="Abrir modelo 3D",
            filetypes=[
                ("Modelos 3D", "*.glb *.gltf *.obj *.stl *.ply"),
                ("Todos", "*.*"),
            ],
        )
        if path:
            self.load_model(path)

    def load_model(self, path):
        self.current_model_path = path
        self.status.set("Cargando modelo 3D...")
        threading.Thread(target=self._prepare_model, args=(path,), daemon=True).start()

    def _prepare_model(self, path):
        try:
            import numpy as np
            import pymeshlab
            import trimesh

            loaded = trimesh.load(path, force="scene")
            meshes = [mesh for mesh in loaded.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
            if not meshes and isinstance(loaded, trimesh.Trimesh):
                meshes = [loaded]
            if not meshes:
                raise ValueError("El archivo no contiene una malla 3D.")
            mesh = trimesh.util.concatenate(meshes)
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            if len(faces) > 25000:
                mesh_set = pymeshlab.MeshSet()
                mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
                mesh_set.meshing_decimation_quadric_edge_collapse(targetfacenum=25000)
                vertices = mesh_set.current_mesh().vertex_matrix().astype(np.float32)
                faces = mesh_set.current_mesh().face_matrix().astype(np.int32)
            self.after(0, lambda: self._show_model(path, vertices, faces))
        except Exception as exc:
            self.after(0, lambda error=str(exc): self._model_load_failed(error))

    def _show_model(self, path, vertices, faces):
        self.model_axes.clear()
        self.model_axes.set_axis_off()
        self.model_axes.set_title(os.path.basename(path))
        self.model_axes.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            color="#80a9ce",
            edgecolor="none",
            linewidth=0,
            antialiased=True,
        )
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        center = (minimum + maximum) / 2
        radius = max(float((maximum - minimum).max()) / 2, 1e-4)
        self.model_axes.set_xlim(center[0] - radius, center[0] + radius)
        self.model_axes.set_ylim(center[1] - radius, center[1] + radius)
        self.model_axes.set_zlim(center[2] - radius, center[2] + radius)
        self.model_axes.set_box_aspect((1, 1, 1))
        self.model_axes.view_init(elev=15, azim=225)
        self.model_canvas.draw_idle()
        self.status.set("Modelo cargado: " + os.path.basename(path))

    def _model_load_failed(self, error):
        self.status.set("No se pudo cargar el modelo")
        messagebox.showerror("Error al abrir el modelo", error)

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def open_output_folder(self):
        path = os.path.dirname(self.output_path.get())
        if os.path.isdir(path):
            os.startfile(path)

    def generate(self):
        image = self.image_path.get()
        output = self.output_path.get()
        if not os.path.isfile(image):
            messagebox.showerror("Error", "No se encontró la imagen seleccionada.")
            return
        if not output:
            messagebox.showerror("Error", "Selecciona una ruta de salida.")
            return

        self.generate_button.state(["disabled"])
        self.open_button.state(["disabled"])
        self.progress.configure(mode="determinate")
        self.progress["value"] = 0
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.status.set("Generando modelo...")
        thread = threading.Thread(target=self._run_generation, args=(image, output), daemon=True)
        thread.start()

    def _run_generation(self, image, output):
        command = [
            PYTHON,
            "-m",
            "scripts.inference_triposg",
            "--image-input",
            image,
            "--output-path",
            output,
            "--num-inference-steps",
            str(self.steps.get()),
            "--faces",
            str(self.faces.get()),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                self.after(0, lambda text=line: self.append_log(text))
                match = re.search(r"(\d+)%", line)
                if match:
                    percent = int(match.group(1))
                    self.after(0, lambda value=percent: self.progress.configure(value=value))
                if "Loading pipeline" in line:
                    self.after(0, lambda: self.status.set("Cargando modelo IA"))
                if "step " in line.lower():
                    self.after(0, lambda: self.status.set("Creando malla 3D"))
            process.wait()
            if process.returncode == 0 and os.path.isfile(output):
                self.after(0, lambda: self._finished(output))
            else:
                error = "".join(output_lines[-40:])
                self.after(0, lambda: self._failed(error))
        except Exception as exc:
            self.after(0, lambda: self._failed(str(exc)))

    def _finished(self, output):
        self.generate_button.state(["!disabled"])
        self.open_button.state(["!disabled"])
        self.progress["value"] = 100
        self.status.set("Modelo generado")
        self.load_model(output)
        messagebox.showinfo("Completado", f"Modelo guardado en:\n{output}")

    def _failed(self, error):
        self.generate_button.state(["!disabled"])
        self.status.set("Error al generar")
        messagebox.showerror("Error", error)


if __name__ == "__main__":
    TripoSGApp().mainloop()
