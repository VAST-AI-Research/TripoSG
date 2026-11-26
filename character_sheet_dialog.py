import os
import tkinter as tk
from tkinter import messagebox, ttk
from uuid import uuid4

from PIL import Image, ImageTk

from character_sheet import (
    FULL_BODY_VIEWS,
    REFERENCE_VIEW_LABELS,
    normalize_bbox,
    validate_reference_regions,
)


class CharacterSheetDialog(tk.Toplevel):
    def __init__(self, master, image_path, regions=()):
        super().__init__(master)
        self.title("AETHRYX — Character Sheet Analyzer")
        self.geometry("1120x780")
        self.minsize(900, 640)
        self.configure(background="#11141b")
        self.transient(master)
        self.grab_set()

        self.image_path = os.path.abspath(image_path)
        self.regions = validate_reference_regions(regions)
        self.result = None
        self.pending_bbox = None
        self.drag_start = None
        self.drag_rectangle = None
        self.display_scale = 1.0
        self.image_offset = (0, 0)
        self.displayed_image = None

        try:
            with Image.open(self.image_path) as image:
                mode = "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
                self.source_image = image.convert(mode)
        except (OSError, ValueError) as exc:
            messagebox.showerror("No se pudo abrir la hoja", str(exc), parent=self)
            self.destroy()
            return

        self._build_ui()
        self._populate_regions()
        self.canvas.bind("<Configure>", self._render_image)
        self.canvas.bind("<ButtonPress-1>", self._begin_selection)
        self.canvas.bind("<B1-Motion>", self._update_selection)
        self.canvas.bind("<ButtonRelease-1>", self._finish_selection)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.after_idle(self._render_image)

    def _build_ui(self):
        ttk.Label(
            self,
            text="Marca manualmente cada vista o detalle y asígnale una categoría.",
            background="#11141b",
            foreground="#eef1f7",
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 3))
        ttk.Label(
            self,
            text="Esta primera versión no reconoce vistas automáticamente. TripoSG recibe solo el recorte principal aprobado; no fusiona varias vistas.",
            background="#11141b",
            foreground="#9ca7b8",
            wraplength=1080,
        ).pack(anchor="w", padx=14, pady=(0, 10))

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=4)
        image_panel = ttk.Frame(body, padding=4)
        controls = ttk.Frame(body, padding=(12, 4))
        body.add(image_panel, weight=4)
        body.add(controls, weight=2)

        self.canvas = tk.Canvas(
            image_panel,
            background="#191e28",
            highlightthickness=1,
            highlightbackground="#303847",
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        ttk.Label(
            controls,
            text=f"HOJA: {os.path.basename(self.image_path)}",
            foreground="#00d9ff",
            wraplength=300,
        ).pack(anchor="w", pady=(4, 8))
        ttk.Label(
            controls,
            text="Tipo de región",
            foreground="#9ca7b8",
        ).pack(anchor="w")
        self.view_var = tk.StringVar(value=REFERENCE_VIEW_LABELS["front"])
        self.view_combo = ttk.Combobox(
            controls,
            textvariable=self.view_var,
            values=tuple(REFERENCE_VIEW_LABELS.values()),
            state="readonly",
        )
        self.view_combo.pack(fill="x", pady=(3, 8))
        ttk.Button(controls, text="Añadir recorte", command=self.add_region).pack(fill="x")
        ttk.Button(controls, text="Marcar como vista principal", command=self.set_primary).pack(
            fill="x", pady=(5, 12)
        )

        self.region_tree = ttk.Treeview(
            controls,
            columns=("view", "role", "bounds"),
            show="headings",
            selectmode="browse",
            height=14,
        )
        self.region_tree.heading("view", text="Región")
        self.region_tree.heading("role", text="Uso")
        self.region_tree.heading("bounds", text="Recorte")
        self.region_tree.column("view", width=125, stretch=True)
        self.region_tree.column("role", width=75, stretch=False)
        self.region_tree.column("bounds", width=120, stretch=False)
        self.region_tree.pack(fill="both", expand=True, pady=(0, 6))
        self.region_tree.bind("<<TreeviewSelect>>", self._select_region)
        ttk.Button(controls, text="Eliminar recorte seleccionado", command=self.remove_region).pack(
            fill="x"
        )

        self.status = tk.StringVar(value="Arrastra sobre la imagen para marcar una región.")
        ttk.Label(
            self,
            textvariable=self.status,
            background="#11141b",
            foreground="#00d9ff",
            wraplength=1060,
        ).pack(anchor="w", padx=14, pady=(7, 3))
        buttons = ttk.Frame(self, padding=(12, 8))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancelar", command=self.cancel).pack(side="right", padx=(6, 0))
        ttk.Button(buttons, text="Aplicar regiones", command=self.accept).pack(side="right")

    def _populate_regions(self):
        for region in self.regions:
            self._insert_region(region)

    def _insert_region(self, region):
        x0, y0, x1, y1 = region["bbox"]
        bounds = f"{x0:.2f},{y0:.2f}–{x1:.2f},{y1:.2f}"
        self.region_tree.insert(
            "",
            "end",
            iid=region["id"],
            values=(
                REFERENCE_VIEW_LABELS[region["view"]],
                "Principal" if region["primary"] else "Referencia",
                bounds,
            ),
        )

    def _render_image(self, _event=None):
        if not hasattr(self, "source_image") or not self.canvas.winfo_exists():
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        source_width, source_height = self.source_image.size
        self.display_scale = min(width / source_width, height / source_height)
        display_width = max(1, round(source_width * self.display_scale))
        display_height = max(1, round(source_height * self.display_scale))
        self.image_offset = ((width - display_width) // 2, (height - display_height) // 2)
        preview = self.source_image.copy()
        preview.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        self.displayed_image = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        offset_x, offset_y = self.image_offset
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.displayed_image)
        for region in self.regions:
            self._draw_bbox(region["bbox"], region["id"], region["primary"])
        if self.pending_bbox:
            self._draw_bbox(self.pending_bbox, "Nuevo recorte", False, color="#00d9ff")

    def _event_to_image(self, event):
        canvas_x = self.canvas.canvasx(event.x) - self.image_offset[0]
        canvas_y = self.canvas.canvasy(event.y) - self.image_offset[1]
        width, height = self.source_image.size
        x = max(0.0, min(width, canvas_x / self.display_scale))
        y = max(0.0, min(height, canvas_y / self.display_scale))
        return x, y

    def _begin_selection(self, event):
        self.drag_start = self._event_to_image(event)
        self.pending_bbox = None
        if self.drag_rectangle:
            self.canvas.delete(self.drag_rectangle)
        self.drag_rectangle = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#00d9ff", width=2
        )

    def _update_selection(self, event):
        if self.drag_rectangle and self.drag_start:
            self.canvas.coords(self.drag_rectangle, *self._canvas_start_and_current(event))

    def _canvas_start_and_current(self, event):
        start_x = self.image_offset[0] + self.drag_start[0] * self.display_scale
        start_y = self.image_offset[1] + self.drag_start[1] * self.display_scale
        return start_x, start_y, event.x, event.y

    def _finish_selection(self, event):
        if not self.drag_start:
            return
        end = self._event_to_image(event)
        try:
            self.pending_bbox = normalize_bbox(self.drag_start, end, self.source_image.size)
            self.status.set("Recorte marcado. Asígnale un tipo y pulsa «Añadir recorte».")
        except ValueError:
            self.pending_bbox = None
            self.status.set("El recorte es demasiado pequeño o no contiene un área válida.")
        self.drag_start = None
        self._render_image()

    def _draw_bbox(self, bbox, label, primary, color=None):
        x0, y0, x1, y1 = bbox
        offset_x, offset_y = self.image_offset
        coords = (
            offset_x + x0 * self.source_image.width * self.display_scale,
            offset_y + y0 * self.source_image.height * self.display_scale,
            offset_x + x1 * self.source_image.width * self.display_scale,
            offset_y + y1 * self.source_image.height * self.display_scale,
        )
        outline = color or ("#ff268f" if primary else "#d6b766")
        self.canvas.create_rectangle(*coords, outline=outline, width=2)
        self.canvas.create_text(
            coords[0] + 4,
            coords[1] + 4,
            text=f"{label}{' · principal' if primary else ''}",
            anchor="nw",
            fill=outline,
            font=("Segoe UI", 9, "bold"),
        )

    def add_region(self):
        if not self.pending_bbox:
            self.status.set("Primero marca un rectángulo sobre la imagen.")
            return
        reverse_labels = {label: key for key, label in REFERENCE_VIEW_LABELS.items()}
        view = reverse_labels.get(self.view_var.get())
        if not view:
            self.status.set("Selecciona un tipo de región válido.")
            return
        region = {
            "id": f"ref-{uuid4().hex[:16]}",
            "path": self.image_path,
            "view": view,
            "bbox": self.pending_bbox,
            "primary": False,
        }
        try:
            self.regions = validate_reference_regions(self.regions + [region])
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self._insert_region(self.regions[-1])
        self.pending_bbox = None
        self.status.set(f"Recorte añadido: {REFERENCE_VIEW_LABELS[view]}.")
        self._render_image()

    def set_primary(self):
        selected = self.region_tree.selection()
        if not selected:
            self.status.set("Selecciona una vista de cuerpo completo.")
            return
        region_id = selected[0]
        region = next((item for item in self.regions if item["id"] == region_id), None)
        if not region or region["view"] not in FULL_BODY_VIEWS:
            self.status.set("Solo frontal, lateral, posterior o tres cuartos puede ser vista principal.")
            return
        for item in self.regions:
            item["primary"] = item["id"] == region_id
        self._refresh_region_tree()
        self.status.set(
            f"Vista principal: {REFERENCE_VIEW_LABELS[region['view']]}. El motor actual usará solo este recorte."
        )
        self._render_image()

    def remove_region(self):
        selected = self.region_tree.selection()
        if not selected:
            self.status.set("Selecciona un recorte para eliminarlo.")
            return
        region_id = selected[0]
        self.regions = [region for region in self.regions if region["id"] != region_id]
        self.region_tree.delete(region_id)
        self.status.set("Recorte eliminado de la configuración; la imagen original no se modifica.")
        self._render_image()

    def _refresh_region_tree(self):
        for item in self.region_tree.get_children():
            self.region_tree.delete(item)
        self._populate_regions()

    def _select_region(self, _event=None):
        selected = self.region_tree.selection()
        if not selected:
            return
        region = next((item for item in self.regions if item["id"] == selected[0]), None)
        if region:
            self.pending_bbox = None
            self.status.set(f"Región seleccionada: {REFERENCE_VIEW_LABELS[region['view']]}.")
            self._render_image()

    def accept(self):
        try:
            self.result = validate_reference_regions(self.regions)
        except ValueError as exc:
            messagebox.showerror("Regiones inválidas", str(exc), parent=self)
            return
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()
