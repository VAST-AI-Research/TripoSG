import os
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import traceback
from collections import deque
from tkinter import filedialog, messagebox, ttk

import numpy as np
import trimesh
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from studio_project import validate_settings, validate_glb, save_project, load_project

from studio_metadata import (
    ASSISTANT_NAME,
    ENGINE_NAME,
    PRODUCT_NAME,
    TAGLINE,
    VERSION_LABEL,
    WORKING_BRAND,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
README_PATH = os.path.join(ROOT, "README.md")
ACCENT = "#ff268f"
CYAN = "#00d9ff"
GOLD = "#d6b766"
BG = "#11141b"
PANEL = "#191e28"
TEXT = "#eef1f7"
MUTED = "#9ca7b8"
SUPPORTED_IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SUPPORTED_MODELS = (".glb", ".gltf", ".obj", ".stl", ".ply")
MAX_PREVIEW_FACES = 8000
MAX_LOG_LINES = 500


class AethryxStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(PRODUCT_NAME)
        self.geometry("1420x920")
        self.minsize(1120, 760)
        self.configure(background=BG)

        self.image_path = tk.StringVar(value=os.path.join(ROOT, "assets", "example_data", "hjswed.png"))
        self.output_path = tk.StringVar(value=os.path.join(ROOT, "output.glb"))
        self.steps = tk.IntVar(value=50)
        self.faces = tk.IntVar(value=-1)
        self.seed = tk.IntVar(value=42)
        self.guidance = tk.DoubleVar(value=7.0)
        self.project_path = None
        self.generation_active = False
        self.model_loading = False
        self.projection = tk.StringVar(value="persp")
        self.display_mode = tk.StringVar(value="solid")
        self.selection_mode = tk.StringVar(value="object")
        self.grid_visible = tk.BooleanVar(value=False)
        self.console_visible = tk.BooleanVar(value=True)

        self.preview_image = None
        self.loaded_scene = None
        self.current_model_path = None
        self.model_parts = []
        self.model_stats = {}
        self.mesh_artists = []
        self.tree_to_part = {}
        self.part_to_tree = {}
        self.load_request_id = 0
        self.generation_process = None
        self.cancel_event = None
        self.log_line_count = 0

        self._configure_styles()
        self._build_menu()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close_window)
        self.bind("<Control-n>", lambda event: self.new_project())
        self.bind("<Control-o>", lambda event: self.open_project())
        self.bind("<Control-s>", lambda event: self.save_current_project())
        self._load_reference_preview()
        self._refresh_system_status()

        if os.path.isfile(self.output_path.get()):
            self.load_model(self.output_path.get())

    def report_callback_exception(self, exc, value, tb):
        details = "".join(traceback.format_exception(exc, value, tb)).strip()
        self.status.set("Error en la interfaz")
        self.append_log(details)
        messagebox.showerror("Error en la interfaz", f"{exc.__name__}: {value}")

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=BG, foreground=TEXT)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 20, "bold"))
        style.configure("Brand.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        style.configure("Section.TLabel", background=PANEL, foreground=CYAN, font=("Segoe UI", 11, "bold"))
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#11141b",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 8),
        )
        style.map("Accent.TButton", background=[("active", "#ff5aa9"), ("disabled", "#55515b")])
        style.configure("TButton", background="#293140", foreground=TEXT, padding=(8, 6))
        style.map("TButton", background=[("active", "#3a4659")])
        style.configure("TEntry", fieldbackground="#202633", foreground=TEXT, insertcolor=TEXT)
        style.configure("TSpinbox", fieldbackground="#202633", foreground=TEXT)
        style.configure("TCombobox", fieldbackground="#202633", foreground=TEXT)
        style.configure(
            "Treeview",
            background="#151a23",
            foreground=TEXT,
            fieldbackground="#151a23",
            rowheight=27,
        )
        style.map("Treeview", background=[("selected", "#5a2443")], foreground=[("selected", TEXT)])
        style.configure("Treeview.Heading", background="#252d3b", foreground=TEXT)
        style.configure(
            "Horizontal.TProgressbar",
            background=ACCENT,
            troughcolor="#252d3b",
            bordercolor="#252d3b",
        )

    def _build_menu(self):
        menu_colors = {
            "bg": PANEL,
            "fg": TEXT,
            "activebackground": "#3a2637",
            "activeforeground": TEXT,
            "disabledforeground": "#667080",
            "tearoff": False,
        }
        self.main_menu = tk.Menu(self, **menu_colors)
        self.config(menu=self.main_menu)

        file_menu = tk.Menu(self.main_menu, **menu_colors)
        file_menu.add_command(label="Nuevo proyecto (.aeth)", accelerator="Ctrl+N", command=self.new_project)
        file_menu.add_command(label="Abrir proyecto...", accelerator="Ctrl+O", command=self.open_project)
        file_menu.add_command(label="Guardar proyecto", accelerator="Ctrl+S", command=self.save_current_project)
        file_menu.add_command(label="Guardar proyecto como...", command=lambda: self.save_current_project(True))
        file_menu.add_separator()
        file_menu.add_command(label="Importar imagen de referencia...", command=self.choose_image)
        file_menu.add_command(label="Importar malla 3D...", command=self.choose_model)
        file_menu.add_command(label="Exportar escena original como GLB...", command=self.export_model)
        file_menu.add_command(label="Archivos recientes - proximo", state="disabled")
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.close_window)
        self.main_menu.add_cascade(label="Archivo", menu=file_menu)

        edit_menu = tk.Menu(self.main_menu, **menu_colors)
        for label in ("Deshacer - proximo", "Rehacer - proximo"):
            edit_menu.add_command(label=label, state="disabled")
        edit_menu.add_separator()
        for label in ("Duplicar objeto - proximo", "Eliminar objeto - proximo", "Preferencias - proximo"):
            edit_menu.add_command(label=label, state="disabled")
        self.main_menu.add_cascade(label="Editar", menu=edit_menu)

        select_menu = tk.Menu(self.main_menu, **menu_colors)
        select_menu.add_command(label="Seleccionar todo", command=self.select_all_objects)
        select_menu.add_command(label="Deseleccionar todo", command=self.deselect_all_objects)
        select_menu.add_command(label="Invertir seleccion", command=self.invert_object_selection)
        select_menu.add_separator()
        select_menu.add_radiobutton(label="Modo objeto", value="object", variable=self.selection_mode)
        for label in ("Modo vertices - proximo", "Modo aristas - proximo", "Modo caras - proximo"):
            select_menu.add_command(label=label, state="disabled")
        self.main_menu.add_cascade(label="Seleccionar", menu=select_menu)

        view_menu = tk.Menu(self.main_menu, **menu_colors)
        for label, camera in (
            ("Frontal", "front"),
            ("Lateral", "side"),
            ("Trasera", "back"),
            ("Superior", "top"),
        ):
            view_menu.add_command(label=label, command=lambda name=camera: self.set_camera(name))
        view_menu.add_command(label="Encuadrar todo", command=self.frame_model)
        view_menu.add_separator()
        view_menu.add_radiobutton(
            label="Perspectiva",
            variable=self.projection,
            value="persp",
            command=self.apply_view_settings,
        )
        view_menu.add_radiobutton(
            label="Ortografica",
            variable=self.projection,
            value="ortho",
            command=self.apply_view_settings,
        )
        view_menu.add_checkbutton(
            label="Rejilla",
            variable=self.grid_visible,
            command=self.apply_view_settings,
        )
        view_menu.add_separator()
        view_menu.add_radiobutton(
            label="Solido",
            variable=self.display_mode,
            value="solid",
            command=self.apply_display_mode,
        )
        view_menu.add_radiobutton(
            label="Estructura alambre",
            variable=self.display_mode,
            value="wireframe",
            command=self.apply_display_mode,
        )
        view_menu.add_command(label="Materiales / texturas - proximo", state="disabled")
        view_menu.add_command(label="Huesos - proximo", state="disabled")
        self.main_menu.add_cascade(label="Vista", menu=view_menu)

        image_menu = tk.Menu(self.main_menu, **menu_colors)
        image_menu.add_command(label="Cargar referencia...", command=self.choose_image)
        image_menu.add_command(label="Generar malla 3D", command=self.generate)
        for label in ("Recortar - proximo", "Quitar fondo - proximo", "Resolucion de salida - proximo"):
            image_menu.add_command(label=label, state="disabled")
        self.main_menu.add_cascade(label="Imagen", menu=image_menu)

        scene_menu = tk.Menu(self.main_menu, **menu_colors)
        scene_menu.add_command(label="Alternar visibilidad seleccionada", command=self.toggle_selected_visibility)
        scene_menu.add_command(label="Aislar seleccion", command=self.isolate_selection)
        scene_menu.add_command(label="Mostrar todos los objetos", command=self.show_all_objects)
        scene_menu.add_command(label="Renombrar objeto - proximo", state="disabled")
        self.main_menu.add_cascade(label="Escena", menu=scene_menu)

        materials_menu = tk.Menu(self.main_menu, **menu_colors)
        materials_menu.add_command(label="Editor de color, UV y PBR - proximo", state="disabled")
        self.main_menu.add_cascade(label="Colores / materiales", menu=materials_menu)

        tools_menu = tk.Menu(self.main_menu, **menu_colors)
        tools_menu.add_command(label="Estadisticas de malla...", command=self.show_mesh_statistics)
        for label in ("Reparacion - proximo", "Simetria - proximo", "Retopologia - proximo", "Rigging - proximo"):
            tools_menu.add_command(label=label, state="disabled")
        self.main_menu.add_cascade(label="Herramientas", menu=tools_menu)

        filters_menu = tk.Menu(self.main_menu, **menu_colors)
        filters_menu.add_command(label="Analizar malla...", command=self.show_mesh_statistics)
        for label in ("Suavizar - proximo", "Reducir malla original - proximo"):
            filters_menu.add_command(label=label, state="disabled")
        self.main_menu.add_cascade(label="Filtros", menu=filters_menu)

        animation_menu = tk.Menu(self.main_menu, **menu_colors)
        for label in ("Pose y huesos - proximo", "Pesos - proximo", "Prueba de movimiento - proximo", "Visemas - proximo"):
            animation_menu.add_command(label=label, state="disabled")
        self.main_menu.add_cascade(label="Animacion", menu=animation_menu)

        windows_menu = tk.Menu(self.main_menu, **menu_colors)
        windows_menu.add_checkbutton(
            label="Consola",
            variable=self.console_visible,
            command=self.toggle_console,
        )
        windows_menu.add_command(label="Restablecer espacio de trabajo", command=self.reset_workspace)
        self.main_menu.add_cascade(label="Ventanas", menu=windows_menu)

        help_menu = tk.Menu(self.main_menu, **menu_colors)
        help_menu.add_command(label="Manual del proyecto", command=self.open_manual)
        help_menu.add_command(label="Diagnostico...", command=self.show_diagnostics)
        help_menu.add_command(label="Creditos y licencias", command=self.show_credits)
        help_menu.add_command(label=f"Acerca de {PRODUCT_NAME}", command=self.show_about)
        self.main_menu.add_cascade(label="Ayuda", menu=help_menu)

    def _build_ui(self):
        header = ttk.Frame(self, padding=(18, 12, 18, 8))
        header.pack(fill="x")
        brand_block = ttk.Frame(header)
        brand_block.pack(side="left")
        ttk.Label(brand_block, text=PRODUCT_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(brand_block, text=TAGLINE, foreground=CYAN).pack(anchor="w")
        ttk.Label(
            header,
            text=f"{WORKING_BRAND}  /  IA LOCAL  /  MOTOR {ENGINE_NAME.upper()}",
            style="Brand.TLabel",
        ).pack(side="right", anchor="e", pady=8)

        quickbar = ttk.Frame(self, padding=(14, 4))
        quickbar.pack(fill="x")
        ttk.Button(quickbar, text="Importar imagen", command=self.choose_image).pack(side="left", padx=(0, 6))
        ttk.Button(quickbar, text="Abrir malla", command=self.choose_model).pack(side="left", padx=6)
        ttk.Button(quickbar, text="Exportar GLB", command=self.export_model).pack(side="left", padx=6)
        ttk.Button(quickbar, text="Carpeta de salida", command=self.open_output_folder).pack(side="left", padx=6)
        ttk.Label(quickbar, text="Vista previa; la malla fuente se conserva sin modificar", foreground=MUTED).pack(
            side="right", padx=8
        )

        self.main_panes = ttk.Panedwindow(self, orient="horizontal")
        self.main_panes.pack(fill="both", expand=True, padx=10, pady=(4, 8))
        self.scene_panel = self._build_scene_panel(self.main_panes)
        self.viewport_panel = self._build_viewport_panel(self.main_panes)
        self.inspector_panel = self._build_inspector_panel(self.main_panes)
        self.main_panes.add(self.scene_panel, weight=1)
        self.main_panes.add(self.viewport_panel, weight=4)
        self.main_panes.add(self.inspector_panel, weight=1)

        self.console_frame = ttk.Frame(self, style="Panel.TFrame", padding=(10, 6))
        console_header = ttk.Frame(self.console_frame, style="Panel.TFrame")
        console_header.pack(fill="x")
        ttk.Label(console_header, text="CONSOLA", style="Section.TLabel").pack(side="left")
        ttk.Button(console_header, text="Limpiar", command=self.clear_log).pack(side="right")
        self.log = tk.Text(
            self.console_frame,
            height=6,
            bg="#0b0e13",
            fg="#d3d9e3",
            insertbackground=TEXT,
            relief="flat",
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.log.pack(fill="x", expand=False, pady=(5, 0))
        self.console_frame.pack(fill="x", padx=10, pady=(0, 6))

        footer = ttk.Frame(self, padding=(12, 3, 12, 8))
        footer.pack(fill="x")
        self.status = tk.StringVar(value="Listo")
        ttk.Label(footer, textvariable=self.status).pack(side="left")
        self.gpu_status = tk.StringVar(value="GPU: comprobando...")
        ttk.Label(footer, textvariable=self.gpu_status, foreground=CYAN).pack(side="right")

    def _build_scene_panel(self, parent):
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        ttk.Label(panel, text="ESCENA", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(panel, text="Objetos de la malla", style="Muted.TLabel").pack(anchor="w")
        self.scene_tree = ttk.Treeview(panel, show="tree", selectmode="extended", height=12)
        self.scene_tree.pack(fill="both", expand=True, pady=(5, 8))
        self.scene_tree.bind("<Double-1>", lambda _event: self.toggle_selected_visibility())
        self.scene_tree.bind("<<TreeviewSelect>>", lambda _event: self._update_object_inspector())
        ttk.Label(panel, text="REFERENCIA", style="Section.TLabel").pack(anchor="w", pady=(8, 4))
        self.reference_preview = ttk.Label(panel, text="Importa una imagen", style="Muted.TLabel")
        self.reference_preview.pack(fill="x", pady=(0, 6))
        self.object_summary = tk.StringVar(value="Sin malla cargada")
        ttk.Label(panel, textvariable=self.object_summary, style="Muted.TLabel", wraplength=220).pack(anchor="w")
        self.scene_panel_ref = panel
        return panel

    def _build_viewport_panel(self, parent):
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=8)
        toolbar = ttk.Frame(panel, style="Panel.TFrame")
        toolbar.pack(fill="x", pady=(0, 4))
        ttk.Label(toolbar, text="VISOR 3D", style="Section.TLabel").pack(side="left")
        ttk.Button(toolbar, text="Encuadrar", command=self.frame_model).pack(side="right", padx=3)
        ttk.Button(toolbar, text="Superior", command=lambda: self.set_camera("top")).pack(side="right", padx=3)
        ttk.Button(toolbar, text="Frontal", command=lambda: self.set_camera("front")).pack(side="right", padx=3)

        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor=PANEL)
        self.model_axes = self.figure.add_subplot(111, projection="3d")
        self.model_axes.set_facecolor(PANEL)
        self.model_axes.set_title("Importa una malla 3D para comenzar", color=MUTED, pad=18)
        self.model_axes.set_axis_off()
        self.model_canvas = FigureCanvasTkAgg(self.figure, master=panel)
        self.model_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.model_canvas, panel, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")
        self.model_canvas.mpl_connect("scroll_event", self._zoom_model)
        self.viewport_panel_ref = panel
        return panel

    def _build_inspector_panel(self, parent):
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        ttk.Label(panel, text="GENERACION", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        ttk.Label(panel, text="Imagen de entrada", style="Muted.TLabel").pack(anchor="w")
        image_row = ttk.Frame(panel, style="Panel.TFrame")
        image_row.pack(fill="x", pady=(3, 9))
        ttk.Entry(image_row, textvariable=self.image_path).pack(side="left", fill="x", expand=True)
        ttk.Button(image_row, text="...", width=3, command=self.choose_image).pack(side="right", padx=(4, 0))

        ttk.Label(panel, text="Guardar GLB en", style="Muted.TLabel").pack(anchor="w")
        output_row = ttk.Frame(panel, style="Panel.TFrame")
        output_row.pack(fill="x", pady=(3, 10))
        ttk.Entry(output_row, textvariable=self.output_path).pack(side="left", fill="x", expand=True)
        ttk.Button(output_row, text="...", width=3, command=self.choose_output).pack(side="right", padx=(4, 0))

        settings = ttk.Frame(panel, style="Panel.TFrame")
        settings.pack(fill="x", pady=4)
        ttk.Label(settings, text="Pasos", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(settings, from_=10, to=100, textvariable=self.steps, width=8).grid(row=1, column=0, sticky="ew", pady=(3, 8))
        ttk.Label(settings, text="Limite de caras", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(settings, from_=-1, to=2000000, increment=1000, textvariable=self.faces, width=8).grid(
            row=3, column=0, sticky="ew", pady=(3, 8)
        )

        ttk.Label(settings, text="-1 conserva el detalle original", style="Muted.TLabel").grid(row=4, column=0, sticky="w")
        ttk.Label(settings, text="Semilla", style="Muted.TLabel").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(settings, from_=0, to=2147483647, textvariable=self.seed, width=10).grid(row=6, column=0, sticky="ew")
        ttk.Label(settings, text="Guidance", style="Muted.TLabel").grid(row=7, column=0, sticky="w")
        ttk.Spinbox(settings, from_=0, to=20, increment=0.5, textvariable=self.guidance, width=8).grid(row=8, column=0, sticky="ew")
        self.progress = ttk.Progressbar(panel, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(4, 6))
        self.generate_button = ttk.Button(
            panel,
            text="GENERAR MODELO 3D",
            style="Accent.TButton",
            command=self.generate,
        )
        self.generate_button.pack(fill="x", pady=(8, 5))
        self.cancel_button = ttk.Button(panel, text="Cancelar generacion", command=self.cancel_generation, state="disabled")
        self.cancel_button.pack(fill="x")

        ttk.Separator(panel).pack(fill="x", pady=14)
        ttk.Label(panel, text="INSPECTOR", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.inspector_text = tk.StringVar(value="Selecciona un objeto para ver sus datos.")
        ttk.Label(panel, textvariable=self.inspector_text, style="Muted.TLabel", wraplength=225, justify="left").pack(
            anchor="w", fill="x"
        )
        self.inspector_panel_ref = panel
        return panel

    def _load_reference_preview(self):
        path = self.image_path.get()
        if not os.path.isfile(path):
            self.reference_preview.configure(text="Importa una imagen", image="")
            return
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((220, 180))
            self.preview_image = ImageTk.PhotoImage(image)
            self.reference_preview.configure(image=self.preview_image, text="")
        except (OSError, ValueError) as exc:
            self.append_log(f"No se pudo mostrar la imagen: {exc}")

    def _workspace_available(self):
        if self.generation_active or self.model_loading:
            self.status.set("Espera a que termine la operación actual.")
            return False
        return True

    def _clear_scene(self):
        self.load_request_id += 1
        self.model_loading = False
        self.loaded_scene = None
        self.current_model_path = None
        self.model_parts = []
        self.model_stats = {}
        self.mesh_artists = []
        self.tree_to_part.clear()
        self.part_to_tree.clear()
        for item in self.scene_tree.get_children():
            self.scene_tree.delete(item)
        self.model_axes.clear()
        self.model_axes.set_axis_off()
        self.model_canvas.draw_idle()
        self._update_object_inspector()

    def new_project(self):
        if not self._workspace_available():
            return
        if not messagebox.askyesno("Nuevo proyecto", "¿Crear un proyecto nuevo? Guarda antes la configuración que quieras conservar. Los archivos existentes no se eliminan."):
            return
        self._clear_scene()
        self.project_path = None
        self.image_path.set("")
        self.output_path.set("")
        self.steps.set(50); self.faces.set(-1); self.seed.set(42); self.guidance.set(7.0)
        self._load_reference_preview()
        self.title(PRODUCT_NAME)
        self.status.set("Proyecto nuevo.")

    def open_project(self):
        if not self._workspace_available():
            return
        path = filedialog.askopenfilename(filetypes=[("Proyecto AETHRYX", "*.aeth")])
        if not path:
            return
        try:
            data = load_project(path)
            if data["model"] and os.path.splitext(data["model"])[1].lower() not in SUPPORTED_MODELS:
                raise ValueError("Formato de malla no compatible.")
            if not messagebox.askyesno("Abrir proyecto", "¿Reemplazar la configuración actual? Los archivos existentes se conservan."):
                return
            self._clear_scene()
            self.project_path = path
            self.image_path.set(data["image"]); self.output_path.set(data["output"])
            for key in ("steps", "faces", "seed", "guidance"):
                getattr(self, key).set(data["settings"][key])
            self._load_reference_preview()
            self.title(f"{PRODUCT_NAME} — {os.path.basename(path)}")
            if data["model"]:
                self.load_model(data["model"])
            else:
                self.status.set("Proyecto abierto.")
        except Exception as exc:
            messagebox.showerror("Abrir proyecto", str(exc))

    def save_current_project(self, save_as=False):
        if not self._workspace_available():
            return
        path = self.project_path if not save_as else None
        if not path:
            path = filedialog.asksaveasfilename(defaultextension=".aeth", filetypes=[("Proyecto AETHRYX", "*.aeth")])
        if not path:
            return
        try:
            save_project(path, image=self.image_path.get(), model=self.current_model_path,
                         output=self.output_path.get(), steps=self.steps.get(), faces=self.faces.get(),
                         seed=self.seed.get(), guidance=self.guidance.get())
            self.project_path = path
            self.title(f"{PRODUCT_NAME} — {os.path.basename(path)}")
            self.status.set("Proyecto guardado. Los recursos se conservan como archivos externos.")
        except Exception as exc:
            messagebox.showerror("Guardar proyecto", str(exc))

    def close_window(self):
        if self.generation_active:
            messagebox.showinfo("Generación activa", "Cancela la generación y espera a que termine antes de cerrar.")
            return
        if self.model_loading:
            messagebox.showinfo("Carga activa", "Espera a que termine la carga antes de cerrar.")
            return
        self.destroy()

    def choose_image(self):
        if not self._workspace_available():
            return
        path = filedialog.askopenfilename(
            title="Importar imagen de referencia",
            filetypes=[("Imagenes", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.image_path.set(path)
            self._load_reference_preview()
            self.status.set(f"Imagen cargada: {os.path.basename(path)}")

    def choose_output(self):
        path = filedialog.asksaveasfilename(
            title="Guardar modelo generado",
            defaultextension=".glb",
            filetypes=[("GLB", "*.glb")],
        )
        if path:
            self.output_path.set(path)

    def choose_model(self):
        if not self._workspace_available():
            return
        path = filedialog.askopenfilename(
            title="Importar malla 3D",
            filetypes=[
                ("Mallas 3D", "*.glb *.gltf *.obj *.stl *.ply"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self.load_model(path)

    def load_model(self, path):
        if not os.path.isfile(path):
            messagebox.showerror("No se encontro el archivo", path)
            return
        if os.path.splitext(path)[1].lower() not in SUPPORTED_MODELS:
            messagebox.showerror("Formato no compatible", "El visor acepta GLB, GLTF, OBJ, STL y PLY.")
            return

        self.load_request_id += 1
        request_id = self.load_request_id
        self.model_loading = True
        self.status.set(f"Cargando malla: {os.path.basename(path)}")
        self.append_log(f"Importando malla: {path}")
        threading.Thread(target=self._prepare_model, args=(request_id, path), daemon=True).start()

    def _prepare_model(self, request_id, path):
        try:
            scene = trimesh.load(path, force="scene")
            if isinstance(scene, trimesh.Trimesh):
                wrapper = trimesh.Scene()
                wrapper.add_geometry(scene, node_name=os.path.basename(path), geom_name="mesh")
                scene = wrapper

            parts = []
            nodes = list(scene.graph.nodes_geometry)
            for index, node_name in enumerate(nodes):
                transform, geometry_name = scene.graph[node_name]
                source_mesh = scene.geometry[geometry_name]
                if not isinstance(source_mesh, trimesh.Trimesh):
                    continue
                preview_mesh = source_mesh.copy()
                preview_mesh.apply_transform(transform)
                vertices = np.asarray(preview_mesh.vertices, dtype=np.float32)
                faces = np.asarray(preview_mesh.faces, dtype=np.int32)
                if len(faces) == 0:
                    continue
                color = self._preview_color(preview_mesh, index)
                if len(faces) > MAX_PREVIEW_FACES:
                    mesh_set = self._decimate_preview(vertices, faces)
                    vertices = mesh_set.current_mesh().vertex_matrix().astype(np.float32)
                    faces = mesh_set.current_mesh().face_matrix().astype(np.int32)
                parts.append(
                    {
                        "node": str(node_name),
                        "geometry": str(geometry_name),
                        "vertices": vertices,
                        "faces": faces,
                        "source_vertices": len(source_mesh.vertices),
                        "source_faces": len(source_mesh.faces),
                        "color": color,
                    }
                )

            if not parts:
                raise ValueError("El archivo no contiene geometria de malla visible.")

            geometries = [
                mesh for mesh in scene.geometry.values() if isinstance(mesh, trimesh.Trimesh)
            ]
            bounds = scene.bounds
            stats = {
                "objects": len(parts),
                "geometries": len(geometries),
                "vertices": sum(part["source_vertices"] for part in parts),
                "faces": sum(part["source_faces"] for part in parts),
                "watertight": all(mesh.is_watertight for mesh in geometries),
                "bounds": bounds,
            }
            self.after(0, lambda: self._show_model(request_id, path, scene, parts, stats))
        except Exception as exc:
            self.after(0, lambda error=str(exc): self._model_load_failed(request_id, error))

    @staticmethod
    def _preview_color(mesh, index):
        visual = getattr(mesh, "visual", None)
        colors = None
        if visual is not None and getattr(visual, "kind", None) == "vertex":
            colors = getattr(visual, "vertex_colors", None)
        elif visual is not None and getattr(visual, "kind", None) == "face":
            colors = getattr(visual, "face_colors", None)
        if colors is not None and len(colors):
            color = np.asarray(colors, dtype=np.float64)[:, :3].mean(axis=0)
            if color.max() > 1:
                color /= 255.0
            return tuple(np.clip(color, 0, 1))

        material = getattr(visual, "material", None)
        for attribute in ("baseColorFactor", "diffuse"):
            factor = getattr(material, attribute, None)
            if factor is not None:
                color = np.asarray(factor, dtype=np.float64)[:3]
                if color.max() > 1:
                    color /= 255.0
                return tuple(np.clip(color, 0, 1))

        palette = (ACCENT, CYAN, GOLD, "#a78bfa", "#6ee7b7")
        return palette[index % len(palette)]

    @staticmethod
    def _decimate_preview(vertices, faces):
        import pymeshlab

        mesh_set = pymeshlab.MeshSet()
        mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
        mesh_set.meshing_decimation_quadric_edge_collapse(targetfacenum=MAX_PREVIEW_FACES)
        return mesh_set

    def _show_model(self, request_id, path, scene, parts, stats):
        if request_id != self.load_request_id:
            return
        self.model_loading = False
        self.loaded_scene = scene
        self.model_parts = parts
        self.model_stats = stats
        self.mesh_artists = []
        self.tree_to_part.clear()
        self.part_to_tree.clear()

        for item in self.scene_tree.get_children():
            self.scene_tree.delete(item)
        self.model_axes.clear()
        self.model_axes.set_facecolor(PANEL)
        self.model_axes.set_title(os.path.basename(path), color=TEXT, pad=14)

        for index, part in enumerate(parts):
            artist = Poly3DCollection(
                part["vertices"][part["faces"]],
                facecolors=part["color"],
                edgecolor="none",
                linewidth=0,
                antialiased=False,
                shade=True,
            )
            self.model_axes.add_collection3d(artist)
            self.mesh_artists.append(artist)
            part["shaded_color"] = artist.get_facecolor()
            item_id = f"object-{index}"
            label = part["node"] if part["node"] != part["geometry"] else part["node"]
            self.scene_tree.insert("", "end", iid=item_id, text=f"[x] {label}")
            self.tree_to_part[item_id] = index
            self.part_to_tree[index] = item_id

        self.apply_display_mode()
        self._fit_bounds(stats.get("bounds"))
        self.model_axes.view_init(elev=15, azim=225)
        self.apply_view_settings()
        self.model_canvas.draw_idle()
        self.current_model_path = path
        self._update_object_inspector()
        self.status.set(f"Malla cargada: {os.path.basename(path)}")
        self.append_log(
            f"Malla cargada: {stats['objects']} objeto(s), {stats['vertices']:,} vertices, {stats['faces']:,} caras."
        )

    def _model_load_failed(self, request_id, error):
        if request_id != self.load_request_id:
            return
        self.model_loading = False
        self.status.set("No se pudo cargar la malla")
        self.append_log(f"Error al importar malla: {error}")
        messagebox.showerror("Error al importar malla", error)

    def _fit_bounds(self, bounds=None):
        if not self.model_parts:
            return
        if bounds is None:
            vertices = np.concatenate([part["vertices"] for part in self.model_parts], axis=0)
            minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        else:
            minimum, maximum = np.asarray(bounds[0]), np.asarray(bounds[1])
        center = (minimum + maximum) / 2
        radius = max(float((maximum - minimum).max()) / 2, 1e-4)
        self.model_axes.set_xlim(center[0] - radius, center[0] + radius)
        self.model_axes.set_ylim(center[1] - radius, center[1] + radius)
        self.model_axes.set_zlim(center[2] - radius, center[2] + radius)
        self.model_axes.set_box_aspect((1, 1, 1))

    def frame_model(self):
        if not self.model_parts:
            self.status.set("Importa una malla para encuadrar la vista.")
            return
        self._fit_bounds(self.model_stats.get("bounds"))
        self.model_canvas.draw_idle()

    def set_camera(self, name):
        cameras = {
            "front": (0, -90),
            "side": (0, 0),
            "back": (0, 90),
            "top": (90, -90),
        }
        elevation, azimuth = cameras[name]
        self.model_axes.view_init(elev=elevation, azim=azimuth)
        self.model_canvas.draw_idle()

    def apply_view_settings(self):
        self.model_axes.set_proj_type(self.projection.get())
        if self.grid_visible.get():
            self.model_axes.set_axis_on()
            self.model_axes.grid(True, color="#3a4352", alpha=0.45)
            self.model_axes.tick_params(colors=MUTED, labelsize=7)
        else:
            self.model_axes.set_axis_off()
        self.model_canvas.draw_idle()

    def apply_display_mode(self):
        wireframe = self.display_mode.get() == "wireframe"
        for index, artist in enumerate(self.mesh_artists):
            if wireframe:
                artist.set_edgecolor("#d8e0ec")
                artist.set_linewidth(0.18)
                artist.set_facecolor((0.08, 0.1, 0.14, 0.06))
            else:
                artist.set_edgecolor("none")
                artist.set_linewidth(0)
                artist.set_facecolor(self.model_parts[index]["shaded_color"])
                artist.set_alpha(1.0)
        self.model_canvas.draw_idle()

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

    def _selected_part_indices(self):
        return [
            self.tree_to_part[item]
            for item in self.scene_tree.selection()
            if item in self.tree_to_part
        ]

    def select_all_objects(self):
        self.scene_tree.selection_set(self.scene_tree.get_children())
        self._update_object_inspector()

    def deselect_all_objects(self):
        self.scene_tree.selection_remove(self.scene_tree.selection())
        self._update_object_inspector()

    def invert_object_selection(self):
        selected = set(self.scene_tree.selection())
        items = self.scene_tree.get_children()
        self.scene_tree.selection_set([item for item in items if item not in selected])
        self._update_object_inspector()

    def toggle_selected_visibility(self):
        indices = self._selected_part_indices()
        if not indices:
            self.status.set("Selecciona uno o mas objetos de la escena.")
            return
        for index in indices:
            artist = self.mesh_artists[index]
            visible = not artist.get_visible()
            artist.set_visible(visible)
            item = self.part_to_tree[index]
            label = self.model_parts[index]["node"]
            self.scene_tree.item(item, text=f"[{'x' if visible else ' '}] {label}")
        self.model_canvas.draw_idle()

    def isolate_selection(self):
        selected = set(self._selected_part_indices())
        if not selected:
            self.status.set("Selecciona los objetos que quieras aislar.")
            return
        for index, artist in enumerate(self.mesh_artists):
            visible = index in selected
            artist.set_visible(visible)
            label = self.model_parts[index]["node"]
            self.scene_tree.item(self.part_to_tree[index], text=f"[{'x' if visible else ' '}] {label}")
        self.model_canvas.draw_idle()

    def show_all_objects(self):
        for index, artist in enumerate(self.mesh_artists):
            artist.set_visible(True)
            self.scene_tree.item(
                self.part_to_tree[index],
                text=f"[x] {self.model_parts[index]['node']}",
            )
        self.model_canvas.draw_idle()

    def _update_object_inspector(self):
        indices = self._selected_part_indices()
        if not indices:
            stats = self.model_stats
            if stats:
                self.object_summary.set(f"{stats['objects']} objeto(s) en escena")
                self.inspector_text.set(
                    f"Escena completa\n"
                    f"{stats['vertices']:,} vertices\n"
                    f"{stats['faces']:,} caras\n"
                    f"Manifold cerrado: {'si' if stats['watertight'] else 'no'}"
                )
            else:
                self.object_summary.set("Sin malla cargada")
                self.inspector_text.set("Importa un modelo para inspeccionar sus datos.")
            return
        index = indices[0]
        part = self.model_parts[index]
        self.inspector_text.set(
            f"Objeto: {part['node']}\n"
            f"Geometria: {part['geometry']}\n"
            f"Vertices de vista: {len(part['vertices']):,}\n"
            f"Caras de vista: {len(part['faces']):,}\n"
            "La decimacion afecta solo la vista previa."
        )

    def show_mesh_statistics(self):
        if not self.model_stats:
            messagebox.showinfo("Estadisticas de malla", "Importa primero una malla 3D.")
            return
        stats = self.model_stats
        bounds = stats["bounds"]
        bounds = np.asarray(bounds) if bounds is not None else np.empty((0, 3))
        dimensions = bounds[1] - bounds[0] if bounds.shape == (2, 3) else np.zeros(3)
        messagebox.showinfo(
            "Estadisticas de malla",
            f"Objetos: {stats['objects']}\n"
            f"Geometrias: {stats['geometries']}\n"
            f"Vertices fuente: {stats['vertices']:,}\n"
            f"Caras fuente: {stats['faces']:,}\n"
            f"Cerrada (watertight): {'si' if stats['watertight'] else 'no'}\n"
            f"Dimensiones locales: {dimensions[0]:.4g} x {dimensions[1]:.4g} x {dimensions[2]:.4g}",
        )

    def open_output_folder(self):
        path = os.path.dirname(os.path.abspath(self.output_path.get()))
        if os.path.isdir(path):
            os.startfile(path)
        else:
            self.status.set("La carpeta de salida todavia no existe.")

    def export_model(self):
        if not self._workspace_available():
            return
        if self.loaded_scene is None:
            self.status.set("Importa una malla antes de exportar.")
            return
        path = filedialog.asksaveasfilename(
            title="Exportar escena original como GLB",
            defaultextension=".glb",
            filetypes=[("GLB", "*.glb")],
        )
        if not path:
            return
        try:
            if self.current_model_path and self.current_model_path.lower().endswith(".glb"):
                # Preserve the original bytes, including skin/morph/animation data not supported by trimesh.
                self._copy_original_glb(self.current_model_path, path)
            else:
                self._write_scene_glb(self.loaded_scene, path)
            self.status.set(f"Escena original exportada: {os.path.basename(path)}")
            self.append_log(f"Exportacion GLB completada: {path}")
        except Exception as exc:
            self.status.set("Error al exportar la escena")
            self.append_log(f"Error al exportar GLB: {exc}")
            messagebox.showerror("Error al exportar", str(exc))

    @staticmethod
    def _copy_original_glb(source, path):
        if os.path.realpath(source) == os.path.realpath(path):
            return
        directory = os.path.dirname(os.path.abspath(path))
        handle, temp_path = tempfile.mkstemp(prefix=".aethryx-", suffix=".glb", dir=directory)
        os.close(handle)
        try:
            shutil.copyfile(source, temp_path)
            validate_glb(temp_path)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def _write_scene_glb(scene, path):
        directory = os.path.dirname(os.path.abspath(path))
        handle, temp_path = tempfile.mkstemp(prefix=".aethryx-", suffix=".glb", dir=directory)
        os.close(handle)
        try:
            scene.export(temp_path, file_type="glb")
            validate_glb(temp_path)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _refresh_system_status(self):
        def inspect():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.free,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                name, free, total = (value.strip() for value in result.stdout.splitlines()[0].split(","))
                text = f"GPU: {name} | VRAM libre {int(free) / 1024:.1f}/{int(total) / 1024:.1f} GB"
            except (OSError, subprocess.SubprocessError, IndexError, ValueError):
                text = "GPU: estado disponible en Diagnostico de PyTorch"
            weights = os.path.isdir(os.path.join(ROOT, "pretrained_weights", "TripoSG"))
            text += " | pesos TripoSG descargados" if weights else " | faltan pesos TripoSG"
            self.after(0, lambda: self.gpu_status.set(text))

        threading.Thread(target=inspect, daemon=True).start()

    def show_diagnostics(self):
        import torch

        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "No detectada"
        weights = os.path.isdir(os.path.join(ROOT, "pretrained_weights", "TripoSG"))
        messagebox.showinfo(
            "Diagnostico local",
            f"Version de PyTorch: {torch.__version__}\n"
            f"CUDA disponible: {'si' if cuda else 'no'}\n"
            f"GPU: {gpu}\n"
            f"Pesos TripoSG: {'descargados' if weights else 'no descargados'}",
        )

    def show_about(self):
        messagebox.showinfo(
            f"Acerca de {PRODUCT_NAME}",
            f"{PRODUCT_NAME} (nombre provisional)\n"
            f"Marca de trabajo: {WORKING_BRAND}\n"
            f"{TAGLINE}\n\n"
            f"Version: {VERSION_LABEL}\n"
            f"Motor de generacion: {ENGINE_NAME}\n"
            f"Asistente previsto: {ASSISTANT_NAME}\n\n"
            "Aplicacion local experimental. No representa una entidad legal ni una marca registrada.",
        )

    def show_credits(self):
        messagebox.showinfo(
            "Creditos y licencias",
            f"Motor de generacion: {ENGINE_NAME} (VAST-AI-Research).\n"
            "El modelo y los pesos de RMBG conservan sus licencias originales.\n"
            "Revisa las licencias upstream antes de redistribuir o usar comercialmente.\n"
            "Esta interfaz no implica afiliacion con Unity o VRChat.",
        )

    def open_manual(self):
        if os.path.isfile(README_PATH):
            os.startfile(README_PATH)

    def reset_workspace(self):
        self.geometry("1420x920")
        self.after(100, lambda: self.main_panes.sashpos(0, 250))
        self.after(100, lambda: self.main_panes.sashpos(1, 1120))
        self.status.set("Espacio de trabajo restablecido.")

    def toggle_console(self):
        if self.console_visible.get():
            self.console_frame.pack(fill="x", padx=10, pady=(0, 6))
        else:
            self.console_frame.pack_forget()

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log_line_count += 1
        if self.log_line_count > MAX_LOG_LINES:
            self.log.delete("1.0", "2.0")
            self.log_line_count -= 1
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.log_line_count = 0

    def generate(self):
        if not self._workspace_available():
            return
        image = self.image_path.get().strip()
        output = self.output_path.get().strip()
        if not os.path.isfile(image):
            messagebox.showerror("Imagen no valida", "Selecciona un archivo de imagen existente.")
            return
        if os.path.splitext(image)[1].lower() not in SUPPORTED_IMAGES:
            messagebox.showerror("Formato no compatible", "La imagen debe ser PNG, JPG, WEBP o BMP.")
            return
        if not output.lower().endswith(".glb"):
            messagebox.showerror("Formato de salida", "La salida de generacion debe tener extension .glb.")
            return
        if not os.path.isfile(PYTHON):
            messagebox.showerror("Entorno no encontrado", "Instala primero las dependencias con requirements.txt.")
            return
        try:
            steps, faces = self.steps.get(), self.faces.get()
            seed, guidance = self.seed.get(), self.guidance.get()
            validate_settings(steps, faces, seed, guidance)
            output = os.path.abspath(output)
            if os.path.exists(output) and not messagebox.askyesno("Reemplazar GLB", f"¿Reemplazar el archivo existente si la generación es válida?\n{output}"):
                return
            os.makedirs(os.path.dirname(output), exist_ok=True)
        except (ValueError, OSError, tk.TclError) as exc:
            messagebox.showerror("Configuración no válida", str(exc))
            return
        self.generation_active = True
        self.cancel_event = threading.Event()
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.generate_button.state(["disabled"])
        self.cancel_button.state(["!disabled"])
        self.status.set("Preparando generacion...")
        self.append_log(f"Entrada: {image}")
        self.append_log(f"Salida: {output}")
        thread = threading.Thread(
            target=self._run_generation,
            args=(image, output, steps, faces, seed, guidance, self.cancel_event),
            daemon=True,
        )
        thread.start()

    def cancel_generation(self):
        if self.cancel_event is None:
            return
        self.cancel_event.set()
        process = self.generation_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError as exc:
                self.append_log(f"No se pudo cancelar el proceso: {exc}")
        self.status.set("Cancelando generacion...")

    def _run_generation(self, image, output, steps, faces, seed, guidance, cancel_event):
        directory = os.path.dirname(output)
        temp_path = None
        process = None
        try:
            handle, temp_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(output)}.",
                suffix=".tmp.glb",
                dir=directory,
            )
            os.close(handle)
            os.remove(temp_path)
            command = [
                PYTHON,
                "-m",
                "scripts.inference_triposg",
                "--image-input",
                image,
                "--output-path",
                temp_path,
                "--num-inference-steps",
                str(steps),
                "--faces",
                str(faces),
                "--seed", str(seed),
                "--guidance-scale", str(guidance),
            ]
            if cancel_event.is_set():
                self.after(0, self._generation_cancelled)
                return

            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.generation_process = process
            if cancel_event.is_set() and process.poll() is None:
                process.terminate()
            recent_output = deque(maxlen=40)
            for line in process.stdout:
                recent_output.append(line)
                self.after(0, lambda text=line: self.append_log(text))
                if "Loading pipeline" in line:
                    self.after(0, lambda: self.status.set("Cargando componentes del modelo..."))
                elif "step " in line.lower():
                    self.after(0, lambda: self.status.set("Reconstruyendo geometria 3D..."))

            process.wait()
            if cancel_event.is_set():
                self.after(0, self._generation_cancelled)
            elif process.returncode == 0 and temp_path and os.path.isfile(temp_path):
                validate_glb(temp_path)
                candidate = trimesh.load(temp_path, force="scene")
                if not any(isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) and np.isfinite(mesh.vertices).all() for mesh in candidate.geometry.values()):
                    raise ValueError("La salida no contiene una malla válida.")
                if cancel_event.is_set():
                    self.after(0, self._generation_cancelled)
                    return
                os.replace(temp_path, output)
                temp_path = None
                self.after(0, lambda: self._generation_finished(output))
            else:
                error = "".join(recent_output) or "El proceso termino sin producir un modelo."
                self.after(0, lambda message=error: self._generation_failed(message))
        except Exception as exc:
            self.after(0, lambda error=str(exc): self._generation_failed(error))
        finally:
            self.generation_process = None
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _generation_finished(self, output):
        self.generation_active = False
        self.progress.stop()
        self.progress.configure(mode="determinate", value=100)
        self.generate_button.state(["!disabled"])
        self.cancel_button.state(["disabled"])
        self.status.set(f"Generacion completada: {output}")
        self.append_log(f"Modelo guardado: {output}")
        self.load_model(output)

    def _generation_cancelled(self):
        self.generation_active = False
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.generate_button.state(["!disabled"])
        self.cancel_button.state(["disabled"])
        self.status.set("Generacion cancelada; se conservo cualquier salida anterior.")
        self.append_log("Generacion cancelada. El archivo temporal se descarto.")

    def _generation_failed(self, error):
        self.generation_active = False
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self.generate_button.state(["!disabled"])
        self.cancel_button.state(["disabled"])
        self.status.set("Error durante la generacion")
        self.append_log(error)
        messagebox.showerror("Error al generar", error[-4000:])


if __name__ == "__main__":
    AethryxStudio().mainloop()
