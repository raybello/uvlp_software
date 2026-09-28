"""UVLP Maskless Wafer Stepper — desktop control UI (simulation)."""

from __future__ import annotations

__version__ = "0.3.0"
__appname__ = "UVLP Maskless Wafer Stepper"

import json
import math
import sys
import time
from enum import Enum
from pathlib import Path

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import screeninfo

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_PATTERN = APP_DIR / "pattern.png"


class StepperStatus(Enum):
    IDLE = "Idle"
    RUNNING = "Running"
    HOMING = "Homing"
    ERROR = "Error"
    PAUSED = "Paused"
    EXPOSING = "Exposing"


def to_rgba(image: np.ndarray) -> np.ndarray:
    """Convert grayscale or BGR uint8/float image to flat RGBA float32 for Dear PyGui."""
    if image.ndim == 2:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
        rgba = np.dstack([gray, gray, gray, np.ones_like(gray)])
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image.astype(np.float32)
        if bgr.max() > 1.0:
            bgr = bgr / 255.0
        rgb = bgr[:, :, ::-1]
        alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        rgba = np.dstack([rgb, alpha])
    elif image.ndim == 3 and image.shape[2] == 4:
        rgba = image.astype(np.float32)
        if rgba.max() > 1.0:
            rgba = rgba / 255.0
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return np.ascontiguousarray(rgba.flatten(), dtype=np.float32)


class WaferStage:
    def __init__(self) -> None:
        self.x_position = 0.0
        self.y_position = 0.0
        self.z_position = 0.0

        self.x_min = -100.0
        self.x_max = 100.0
        self.y_min = -100.0
        self.y_max = 100.0
        self.z_min = 0.0
        self.z_max = 50.0

        self.speed = 10.0
        self.acceleration = 50.0
        self.step_size = 1.0
        self.jog_speed = 2.0

        self.status = StepperStatus.IDLE
        self.is_homed = False

        self.exposure_time = 0.5
        self.exposure_power = 50.0

        self.grid_rows = 5
        self.grid_columns = 5
        self.grid_spacing_x = 10.0
        self.grid_spacing_y = 10.0

        self.wafer_diameter = 100.0
        self.wafer_type = "Silicon"
        self.edge_exclusion = 5.0

    def move_to(self, x=None, y=None, z=None):
        if x is not None:
            self.x_position = max(self.x_min, min(self.x_max, float(x)))
        if y is not None:
            self.y_position = max(self.y_min, min(self.y_max, float(y)))
        if z is not None:
            self.z_position = max(self.z_min, min(self.z_max, float(z)))
        return self.x_position, self.y_position, self.z_position

    def jog(self, x_delta=0.0, y_delta=0.0, z_delta=0.0):
        return self.move_to(
            self.x_position + x_delta,
            self.y_position + y_delta,
            self.z_position + z_delta,
        )

    def home(self) -> bool:
        self.status = StepperStatus.HOMING
        self.x_position = 0.0
        self.y_position = 0.0
        self.z_position = self.z_max
        self.is_homed = True
        self.status = StepperStatus.IDLE
        return True

    def calculate_grid_positions(self) -> list[tuple[float, float]]:
        positions = []
        usable_radius = (self.wafer_diameter / 2.0) - self.edge_exclusion
        for row in range(self.grid_rows):
            for col in range(self.grid_columns):
                x = col * self.grid_spacing_x - ((self.grid_columns - 1) * self.grid_spacing_x / 2)
                y = row * self.grid_spacing_y - ((self.grid_rows - 1) * self.grid_spacing_y / 2)
                if math.hypot(x, y) <= usable_radius:
                    positions.append((x, y))
        return positions

    def to_dict(self) -> dict:
        return {
            "step_size": self.step_size,
            "speed": self.speed,
            "acceleration": self.acceleration,
            "jog_speed": self.jog_speed,
            "exposure_time": self.exposure_time,
            "exposure_power": self.exposure_power,
            "grid_rows": self.grid_rows,
            "grid_columns": self.grid_columns,
            "grid_spacing_x": self.grid_spacing_x,
            "grid_spacing_y": self.grid_spacing_y,
            "wafer_diameter": self.wafer_diameter,
            "wafer_type": self.wafer_type,
            "edge_exclusion": self.edge_exclusion,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }

    def load_dict(self, data: dict) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Engine:
    def __init__(self) -> None:
        self.name = f"{__appname__} {__version__}"
        self.ui_image_size = 420
        self.pattern_image = np.ones((self.ui_image_size, self.ui_image_size), dtype=np.float32) * 0.15
        self.wafer_view = np.zeros((self.ui_image_size, self.ui_image_size, 3), dtype=np.uint8)
        self.pattern_path: Path | None = None

        self.stage = WaferStage()
        self.exposure_points: list[tuple[float, float]] = []

        self.window_width = 1280
        self.window_height = 820
        self.layout_size: tuple[int, int] = (0, 0)
        self.fullscreen = False

        self.job_running = False
        self.job_cancelled = False
        self.job_queue: list[tuple[float, float]] = []
        self.job_index = 0
        self.job_phase = "idle"  # idle | move | expose
        self.job_phase_until = 0.0
        self.selected_monitor_index = 0
        self._dmd_window_open = False

        self.update_wafer_view()

    def mm_to_px(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        center = self.ui_image_size // 2
        half_diameter = max(self.stage.wafer_diameter / 2.0, 1e-6)
        scale = (self.ui_image_size * 0.42) / half_diameter
        return (
            int(center + x_mm * scale),
            int(center - y_mm * scale),  # +Y up in UI
        )

    def update_wafer_view(self) -> None:
        size = self.ui_image_size
        view = np.zeros((size, size, 3), dtype=np.uint8)
        view[:] = (18, 18, 22)

        center = (size // 2, size // 2)
        radius = int(size * 0.42)

        # Wafer body + edge exclusion ring
        cv2.circle(view, center, radius, (48, 48, 52), -1)
        excl = int(radius * (self.stage.edge_exclusion / max(self.stage.wafer_diameter / 2.0, 1e-6)))
        cv2.circle(view, center, max(radius - excl, 1), (58, 58, 64), 1)
        cv2.circle(view, center, radius, (90, 90, 100), 2)

        # Grid lines
        for frac in np.linspace(-1.0, 1.0, 9):
            offset = int(frac * radius)
            color = (36, 36, 48) if abs(frac) > 1e-6 else (50, 50, 90)
            cv2.line(view, (center[0] - radius, center[1] + offset), (center[0] + radius, center[1] + offset), color, 1)
            cv2.line(view, (center[0] + offset, center[1] - radius), (center[0] + offset, center[1] + radius), color, 1)

        # Planned grid sites
        for gx, gy in self.stage.calculate_grid_positions():
            px, py = self.mm_to_px(gx, gy)
            cv2.circle(view, (px, py), 3, (0, 200, 220), 1)

        # Completed exposures
        for ex, ey in self.exposure_points:
            px, py = self.mm_to_px(ex, ey)
            cv2.circle(view, (px, py), 4, (40, 40, 220), -1)

        # Stage crosshair
        sx, sy = self.mm_to_px(self.stage.x_position, self.stage.y_position)
        cv2.line(view, (sx - 12, sy), (sx + 12, sy), (80, 220, 80), 2)
        cv2.line(view, (sx, sy - 12), (sx, sy + 12), (80, 220, 80), 2)
        cv2.circle(view, (sx, sy), 6, (80, 220, 80), 1)

        # Legend strip
        cv2.putText(view, "cyan=grid  red=exposed  green=stage", (8, size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 150), 1, cv2.LINE_AA)

        self.wafer_view = view

    def expose_at_position(self) -> None:
        self.exposure_points.append((self.stage.x_position, self.stage.y_position))
        self.update_wafer_view()

    def load_pattern(self, path: str | Path) -> None:
        path = Path(path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        resized = cv2.resize(image, (self.ui_image_size, self.ui_image_size))
        self.pattern_image = resized.astype(np.float32) / 255.0
        self.pattern_path = path


ng = Engine()


# --- UI helpers -----------------------------------------------------------------

def show_status(message: str, error: bool = False) -> None:
    color = (220, 90, 90, 255) if error else (180, 180, 190, 255)
    if dpg.does_item_exist("status_text"):
        dpg.set_value("status_text", message)
        dpg.configure_item("status_text", color=color)


def show_error(message: str) -> None:
    show_status(message, error=True)
    if dpg.does_item_exist("error_modal_text"):
        dpg.set_value("error_modal_text", message)
        dpg.configure_item("error_modal", show=True)


def refresh_position_labels() -> None:
    if not dpg.does_item_exist("position_x"):
        return
    dpg.set_value("position_x", f"X: {ng.stage.x_position:.3f} mm")
    dpg.set_value("position_y", f"Y: {ng.stage.y_position:.3f} mm")
    dpg.set_value("position_z", f"Z: {ng.stage.z_position:.3f} mm")
    dpg.set_value("stage_status", f"Status: {ng.stage.status.value}")
    homed = "Homed" if ng.stage.is_homed else "Not homed"
    dpg.set_value("homed_status", homed)


def refresh_textures() -> None:
    if dpg.does_item_exist("pattern_texture"):
        dpg.set_value("pattern_texture", to_rgba(ng.pattern_image))
    if dpg.does_item_exist("wafer_texture"):
        dpg.set_value("wafer_texture", to_rgba(ng.wafer_view))


def refresh_job_progress() -> None:
    if not dpg.does_item_exist("job_progress"):
        return
    total = max(len(ng.job_queue), 1)
    done = ng.job_index
    dpg.set_value("job_progress", done / total)
    dpg.set_value("job_progress_text", f"Job: {done}/{len(ng.job_queue)}" if ng.job_queue else "Job: idle")


def sync_settings_to_ui() -> None:
    mapping = {
        "input_step_size": ng.stage.step_size,
        "exposure_time": ng.stage.exposure_time,
        "exposure_power": ng.stage.exposure_power,
        "grid_rows": ng.stage.grid_rows,
        "grid_columns": ng.stage.grid_columns,
        "grid_spacing_x": ng.stage.grid_spacing_x,
        "grid_spacing_y": ng.stage.grid_spacing_y,
        "wafer_diameter": ng.stage.wafer_diameter,
        "edge_exclusion": ng.stage.edge_exclusion,
        "stage_speed": ng.stage.speed,
        "stage_accel": ng.stage.acceleration,
        "jog_speed": ng.stage.jog_speed,
    }
    for tag, value in mapping.items():
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, value)
    if dpg.does_item_exist("wafer_type"):
        dpg.set_value("wafer_type", ng.stage.wafer_type)


def push_wafer_and_status(message: str | None = None) -> None:
    ng.update_wafer_view()
    refresh_textures()
    refresh_position_labels()
    if message:
        show_status(message)


# --- Callbacks ------------------------------------------------------------------

def file_dialog_callback(sender, app_data) -> None:
    path = app_data.get("file_path_name")
    if not path:
        return
    try:
        ng.load_pattern(path)
        refresh_textures()
        show_status(f"Loaded pattern: {Path(path).name}")
    except Exception as exc:
        show_error(f"Failed to load pattern: {exc}")


def move_to_callback() -> None:
    if ng.job_running:
        show_status("Stop the running job before moving manually.", error=True)
        return
    x = dpg.get_value("input_x")
    y = dpg.get_value("input_y")
    z = dpg.get_value("input_z")
    ng.stage.move_to(x, y, z)
    push_wafer_and_status(f"Moved to ({ng.stage.x_position:.2f}, {ng.stage.y_position:.2f}, {ng.stage.z_position:.2f})")


def jog_callback(sender, app_data, user_data) -> None:
    if ng.job_running:
        show_status("Stop the running job before jogging.", error=True)
        return
    dx, dy, dz = user_data
    step = float(dpg.get_value("input_step_size")) if dpg.does_item_exist("input_step_size") else ng.stage.step_size
    ng.stage.step_size = step
    ng.stage.jog(dx * step, dy * step, dz * step)
    push_wafer_and_status(
        f"Jog → ({ng.stage.x_position:.2f}, {ng.stage.y_position:.2f}, {ng.stage.z_position:.2f})"
    )


def home_callback() -> None:
    if ng.job_running:
        show_status("Stop the running job before homing.", error=True)
        return
    ng.stage.home()
    if dpg.does_item_exist("input_x"):
        dpg.set_value("input_x", ng.stage.x_position)
        dpg.set_value("input_y", ng.stage.y_position)
        dpg.set_value("input_z", ng.stage.z_position)
    push_wafer_and_status("Stage homed (Z raised).")


def expose_callback() -> None:
    if ng.job_running:
        show_status("Job already running.", error=True)
        return
    ng.stage.status = StepperStatus.EXPOSING
    ng.expose_at_position()
    ng.stage.status = StepperStatus.IDLE
    refresh_textures()
    refresh_position_labels()
    show_status(
        f"Exposed at ({ng.stage.x_position:.2f}, {ng.stage.y_position:.2f}) — "
        f"{len(ng.exposure_points)} site(s)"
    )


def clear_exposures_callback() -> None:
    if ng.job_running:
        show_status("Stop the running job first.", error=True)
        return
    ng.exposure_points.clear()
    push_wafer_and_status("Cleared exposure history.")


def update_grid_params_callback() -> None:
    ng.stage.grid_rows = int(dpg.get_value("grid_rows"))
    ng.stage.grid_columns = int(dpg.get_value("grid_columns"))
    ng.stage.grid_spacing_x = float(dpg.get_value("grid_spacing_x"))
    ng.stage.grid_spacing_y = float(dpg.get_value("grid_spacing_y"))
    sites = len(ng.stage.calculate_grid_positions())
    push_wafer_and_status(f"Grid updated — {sites} site(s) inside usable wafer.")


def update_exposure_params_callback() -> None:
    ng.stage.exposure_time = float(dpg.get_value("exposure_time"))
    ng.stage.exposure_power = float(dpg.get_value("exposure_power"))
    show_status(f"Exposure: {ng.stage.exposure_time:.2f}s @ {ng.stage.exposure_power:.0f}%")


def update_wafer_params_callback() -> None:
    ng.stage.wafer_diameter = float(dpg.get_value("wafer_diameter"))
    ng.stage.edge_exclusion = float(dpg.get_value("edge_exclusion"))
    if dpg.does_item_exist("wafer_type"):
        ng.stage.wafer_type = dpg.get_value("wafer_type")
    sites = len(ng.stage.calculate_grid_positions())
    push_wafer_and_status(f"Wafer settings applied — {sites} grid site(s).")


def update_stage_params_callback() -> None:
    ng.stage.speed = float(dpg.get_value("stage_speed"))
    ng.stage.acceleration = float(dpg.get_value("stage_accel"))
    ng.stage.jog_speed = float(dpg.get_value("jog_speed"))
    show_status("Stage motion parameters updated.")


def start_grid_job() -> None:
    if ng.job_running:
        show_status("Job already running.", error=True)
        return
    update_grid_params_callback()
    update_exposure_params_callback()
    positions = ng.stage.calculate_grid_positions()
    if not positions:
        show_error("No grid sites fall inside the usable wafer. Adjust spacing or exclusion.")
        return
    ng.job_queue = positions
    ng.job_index = 0
    ng.job_running = True
    ng.job_cancelled = False
    ng.job_phase = "move"
    ng.job_phase_until = 0.0
    ng.stage.status = StepperStatus.RUNNING
    refresh_job_progress()
    refresh_position_labels()
    show_status(f"Starting grid job: {len(positions)} exposures…")
    if dpg.does_item_exist("btn_run_grid"):
        dpg.configure_item("btn_run_grid", enabled=False)
        dpg.configure_item("btn_stop_grid", enabled=True)


def stop_grid_job() -> None:
    if not ng.job_running:
        return
    ng.job_cancelled = True
    show_status("Stopping job after current step…")


def finish_job(message: str) -> None:
    ng.job_running = False
    ng.job_phase = "idle"
    ng.job_queue = []
    ng.stage.status = StepperStatus.IDLE
    refresh_job_progress()
    refresh_position_labels()
    show_status(message)
    if dpg.does_item_exist("btn_run_grid"):
        dpg.configure_item("btn_run_grid", enabled=True)
        dpg.configure_item("btn_stop_grid", enabled=False)


def tick_job() -> None:
    """Advance simulated step-and-expose job one frame at a time."""
    if not ng.job_running:
        return

    now = time.time()

    if ng.job_cancelled:
        finish_job(f"Job cancelled after {ng.job_index} exposure(s).")
        return

    if ng.job_index >= len(ng.job_queue):
        finish_job(f"Grid complete — {len(ng.exposure_points)} total exposure(s).")
        return

    if now < ng.job_phase_until:
        return

    x, y = ng.job_queue[ng.job_index]

    if ng.job_phase == "move":
        ng.stage.move_to(x, y)
        # Simulated travel time based on speed
        travel = max(0.05, 0.15)
        ng.job_phase = "expose"
        ng.job_phase_until = now + travel
        ng.stage.status = StepperStatus.RUNNING
        push_wafer_and_status(f"Moving to site {ng.job_index + 1}/{len(ng.job_queue)}…")
        refresh_job_progress()
        return

    if ng.job_phase == "expose":
        ng.stage.status = StepperStatus.EXPOSING
        ng.expose_at_position()
        refresh_textures()
        refresh_position_labels()
        ng.job_index += 1
        refresh_job_progress()
        expose_s = max(0.05, float(ng.stage.exposure_time) * 0.25)  # sped-up sim
        ng.job_phase = "move"
        ng.job_phase_until = now + expose_s
        show_status(f"Exposed {ng.job_index}/{len(ng.job_queue)}")
        if ng.job_index >= len(ng.job_queue):
            finish_job(f"Grid complete — {len(ng.exposure_points)} total exposure(s).")


def save_config_callback() -> None:
    try:
        payload = {
            "version": __version__,
            "stage": ng.stage.to_dict(),
            "pattern_path": str(ng.pattern_path) if ng.pattern_path else None,
            "selected_monitor_index": ng.selected_monitor_index,
        }
        DEFAULT_CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        show_status(f"Saved configuration → {DEFAULT_CONFIG_PATH.name}")
    except Exception as exc:
        show_error(f"Could not save config: {exc}")


def load_config_callback(path: Path | None = None) -> None:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        show_status("No saved configuration found.", error=True)
        return
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        ng.stage.load_dict(payload.get("stage", {}))
        ng.selected_monitor_index = int(payload.get("selected_monitor_index", 0))
        pattern = payload.get("pattern_path")
        if pattern and Path(pattern).exists():
            ng.load_pattern(pattern)
        sync_settings_to_ui()
        push_wafer_and_status(f"Loaded configuration from {config_path.name}")
    except Exception as exc:
        show_error(f"Could not load config: {exc}")


def set_fullscreen() -> None:
    ng.fullscreen = not ng.fullscreen
    dpg.toggle_viewport_fullscreen()
    ng.layout_size = (0, 0)


LAYOUT_MIN_WIDTH = 960
LAYOUT_MIN_HEIGHT = 640
LAYOUT_PAD = 10
LAYOUT_COL_GAP = 8
LAYOUT_ROW_GAP = 8
LAYOUT_FOOTER_H = 34
LAYOUT_MENU_H = 22

_SETTINGS_FIELD_TAGS = (
    "exposure_time",
    "exposure_power",
    "grid_rows",
    "grid_columns",
    "grid_spacing_x",
    "grid_spacing_y",
    "wafer_type",
    "wafer_diameter",
    "edge_exclusion",
    "stage_speed",
    "stage_accel",
    "jog_speed",
    "monitor_combo",
)


def apply_responsive_layout() -> None:
    """Size columns, panes, and widgets from the current viewport (symmetric 2×2 grid)."""
    if not dpg.does_item_exist("layout_body"):
        return

    vw = max(dpg.get_viewport_client_width(), LAYOUT_MIN_WIDTH)
    vh = max(dpg.get_viewport_client_height(), LAYOUT_MIN_HEIGHT)
    if (vw, vh) == ng.layout_size:
        return
    ng.layout_size = (vw, vh)

    body_w = vw - LAYOUT_PAD * 2
    body_h = vh - LAYOUT_MENU_H - LAYOUT_FOOTER_H - LAYOUT_PAD * 2
    body_h = max(body_h, 320)
    col_w = max(280, (body_w - LAYOUT_COL_GAP) // 2)

    top_h = max(180, int(body_h * 0.46))
    bottom_h = max(200, body_h - top_h - LAYOUT_ROW_GAP)

    dpg.configure_item("layout_body", width=body_w, height=body_h)
    dpg.configure_item("col_left", width=col_w, height=body_h)
    dpg.configure_item("col_right", width=col_w, height=body_h)

    pane_inner_w = col_w - 12
    for tag, h in (
        ("pane_pattern", top_h),
        ("pane_wafer", top_h),
        ("pane_stage", bottom_h),
        ("pane_settings", bottom_h),
    ):
        dpg.configure_item(tag, width=pane_inner_w, height=h)

    img_sz = min(pane_inner_w - 16, top_h - 72)
    img_sz = max(128, img_sz)
    for tag in ("pattern_image_widget", "wafer_image_widget"):
        dpg.configure_item(tag, width=img_sz, height=img_sz)

    side_pad = max(0, (pane_inner_w - img_sz) // 2)
    for tag in ("pattern_pad_l", "pattern_pad_r", "wafer_pad_l", "wafer_pad_r"):
        dpg.configure_item(tag, width=side_pad)

    btn_row_w = min(pane_inner_w - 16, 280)
    btn_pad = max(0, (pane_inner_w - btn_row_w) // 2)
    for tag in ("pattern_btn_pad_l", "pattern_btn_pad_r", "wafer_btn_pad_l", "wafer_btn_pad_r"):
        dpg.configure_item(tag, width=btn_pad)

    jog_pad = max(0, (pane_inner_w - 44) // 2)
    jog_row3_pad = max(0, (pane_inner_w - 44 * 3) // 2)
    jog_z_pad = max(0, (pane_inner_w - 44 * 2) // 2)
    for tag in ("jog_y_pad_l", "jog_y_pad_r", "jog_y_minus_pad_l", "jog_y_minus_pad_r"):
        dpg.configure_item(tag, width=jog_pad)
    for tag in ("jog_pad_l", "jog_pad_r"):
        dpg.configure_item(tag, width=jog_row3_pad)
    for tag in ("jog_z_pad_l", "jog_z_pad_r"):
        dpg.configure_item(tag, width=jog_z_pad)

    dpg.configure_item("job_progress", width=max(120, pane_inner_w - 20))

    field_w = max(120, min(pane_inner_w - 40, 260))
    for tag in _SETTINGS_FIELD_TAGS:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, width=field_w)

    if dpg.does_item_exist("status_text"):
        dpg.configure_item("status_text", wrap=body_w - 24)


def get_monitor_labels() -> list[str]:
    labels = []
    for i, mon in enumerate(screeninfo.get_monitors()):
        name = getattr(mon, "name", None) or f"Display {i}"
        labels.append(f"{i}: {name} ({mon.width}x{mon.height})")
    return labels or ["0: Primary"]


def refresh_monitor_list() -> None:
    labels = get_monitor_labels()
    if dpg.does_item_exist("monitor_combo"):
        dpg.configure_item("monitor_combo", items=labels)
        idx = min(ng.selected_monitor_index, len(labels) - 1)
        dpg.set_value("monitor_combo", labels[idx])


def on_monitor_selected(sender, app_data) -> None:
    try:
        ng.selected_monitor_index = int(str(app_data).split(":", 1)[0])
        show_status(f"DMD output monitor set to #{ng.selected_monitor_index}")
    except ValueError:
        pass


def show_dmd_output() -> None:
    """Fullscreen pattern on the selected monitor via OpenCV (blocks until key)."""
    monitors = list(screeninfo.get_monitors())
    if not monitors:
        show_error("No monitors detected.")
        return
    idx = min(max(ng.selected_monitor_index, 0), len(monitors) - 1)
    screen = monitors[idx]

    if ng.pattern_path and ng.pattern_path.exists():
        image = cv2.imread(str(ng.pattern_path), cv2.IMREAD_GRAYSCALE)
    else:
        image = (ng.pattern_image * 255).astype(np.uint8)

    if image is None:
        show_error("No pattern available to display.")
        return

    image = cv2.resize(image, (screen.width, screen.height), interpolation=cv2.INTER_NEAREST)
    window_name = "UVLP_DMD_OUTPUT"
    show_status(f"DMD output on monitor {idx} — press any key in that window to close.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, int(screen.x), int(screen.y))
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    show_status("DMD output closed.")


def exit_app() -> None:
    if ng.job_running:
        ng.job_cancelled = True
    try:
        cv2.destroyWindow("UVLP_DMD_OUTPUT")
    except cv2.error:
        pass
    dpg.stop_dearpygui()


# --- Layout ---------------------------------------------------------------------

def build_ui() -> None:
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(
            width=ng.ui_image_size,
            height=ng.ui_image_size,
            default_value=to_rgba(ng.pattern_image),
            tag="pattern_texture",
        )
        dpg.add_dynamic_texture(
            width=ng.ui_image_size,
            height=ng.ui_image_size,
            default_value=to_rgba(ng.wafer_view),
            tag="wafer_texture",
        )

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        modal=True,
        callback=file_dialog_callback,
        tag="file_dialog",
        width=700,
        height=400,
    ):
        dpg.add_file_extension("Image files (*.png *.jpg *.jpeg *.bmp){.png,.jpg,.jpeg,.bmp}")
        dpg.add_file_extension(".*")

    with dpg.window(
        label="Error",
        modal=True,
        show=False,
        tag="error_modal",
        no_title_bar=False,
        width=420,
        height=140,
    ):
        dpg.add_text("", tag="error_modal_text", wrap=380)
        dpg.add_spacer(height=8)
        dpg.add_button(label="OK", width=80, callback=lambda: dpg.configure_item("error_modal", show=False))

    with dpg.window(
        label=__appname__,
        tag="_main_window",
        no_resize=False,
        no_move=False,
        no_title_bar=True,
        width=ng.window_width,
        height=ng.window_height,
        pos=(0, 0),
    ):
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Open Pattern…", callback=lambda: dpg.show_item("file_dialog"))
                dpg.add_menu_item(label="Save Configuration", callback=save_config_callback)
                dpg.add_menu_item(label="Load Configuration", callback=lambda: load_config_callback())
                dpg.add_separator()
                dpg.add_menu_item(label="Exit", callback=exit_app)
            with dpg.menu(label="Stage"):
                dpg.add_menu_item(label="Home", callback=home_callback)
                dpg.add_menu_item(label="Clear Exposures", callback=clear_exposures_callback)
            with dpg.menu(label="View"):
                dpg.add_menu_item(label="Toggle Fullscreen", callback=set_fullscreen)
                dpg.add_menu_item(label="Refresh Monitors", callback=refresh_monitor_list)
                dpg.add_menu_item(label="Show Pattern on DMD…", callback=show_dmd_output)

        with dpg.child_window(tag="layout_body", border=False, no_scrollbar=True):
            with dpg.group(horizontal=True):
                with dpg.child_window(tag="col_left", border=True, no_scrollbar=False):
                    with dpg.child_window(tag="pane_pattern", border=False, no_scrollbar=True):
                        dpg.add_text("Exposure Pattern", color=(200, 200, 210))
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(tag="pattern_pad_l", width=1)
                            dpg.add_image(
                                "pattern_texture",
                                tag="pattern_image_widget",
                                width=ng.ui_image_size,
                                height=ng.ui_image_size,
                            )
                            dpg.add_spacer(tag="pattern_pad_r", width=1)
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(tag="pattern_btn_pad_l", width=1)
                            dpg.add_button(label="Open Pattern…", callback=lambda: dpg.show_item("file_dialog"))
                            dpg.add_button(label="Show on DMD", callback=show_dmd_output)
                            dpg.add_spacer(tag="pattern_btn_pad_r", width=1)

                    dpg.add_spacer(height=LAYOUT_ROW_GAP)

                    with dpg.child_window(tag="pane_stage", border=False, no_scrollbar=False):
                        dpg.add_text("Stage Control", color=(200, 200, 210))
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"X: {ng.stage.x_position:.3f} mm", tag="position_x")
                            dpg.add_text(f"Y: {ng.stage.y_position:.3f} mm", tag="position_y")
                            dpg.add_text(f"Z: {ng.stage.z_position:.3f} mm", tag="position_z")
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"Status: {ng.stage.status.value}", tag="stage_status", color=(220, 200, 80))
                            dpg.add_text("Not homed", tag="homed_status")

                        dpg.add_spacer(height=4)
                        with dpg.group(horizontal=True):
                            dpg.add_input_float(label="X", width=90, default_value=0.0, tag="input_x", step=0.1)
                            dpg.add_input_float(label="Y", width=90, default_value=0.0, tag="input_y", step=0.1)
                            dpg.add_input_float(label="Z", width=90, default_value=0.0, tag="input_z", step=0.1)
                            dpg.add_button(label="Move To", callback=move_to_callback)

                        dpg.add_input_float(
                            label="Step (mm)",
                            width=100,
                            default_value=ng.stage.step_size,
                            tag="input_step_size",
                            step=0.1,
                        )

                        dpg.add_text("Jog", color=(180, 180, 190))
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(tag="jog_y_pad_l", width=1)
                            dpg.add_button(label="Y+", width=44, height=28, callback=jog_callback, user_data=(0, 1, 0))
                            dpg.add_spacer(tag="jog_y_pad_r", width=1)
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(tag="jog_pad_l", width=1)
                            dpg.add_button(label="X-", width=44, height=28, callback=jog_callback, user_data=(-1, 0, 0))
                            dpg.add_button(label="Home", width=44, height=28, callback=home_callback)
                            dpg.add_button(label="X+", width=44, height=28, callback=jog_callback, user_data=(1, 0, 0))
                            dpg.add_spacer(tag="jog_pad_r", width=1)
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(tag="jog_y_minus_pad_l", width=1)
                            dpg.add_button(label="Y-", width=44, height=28, callback=jog_callback, user_data=(0, -1, 0))
                            dpg.add_spacer(tag="jog_y_minus_pad_r", width=1)
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(tag="jog_z_pad_l", width=1)
                            dpg.add_button(label="Z+", width=44, height=28, callback=jog_callback, user_data=(0, 0, 1))
                            dpg.add_button(label="Z-", width=44, height=28, callback=jog_callback, user_data=(0, 0, -1))
                            dpg.add_spacer(tag="jog_z_pad_r", width=1)

                        dpg.add_spacer(height=6)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Expose Here", callback=expose_callback)
                            dpg.add_button(label="Run Grid", tag="btn_run_grid", callback=start_grid_job)
                            dpg.add_button(label="Stop", tag="btn_stop_grid", callback=stop_grid_job, enabled=False)
                            dpg.add_button(label="Clear", callback=clear_exposures_callback)

                        dpg.add_progress_bar(tag="job_progress", default_value=0.0, width=-1)
                        dpg.add_text("Job: idle", tag="job_progress_text")

                dpg.add_spacer(width=LAYOUT_COL_GAP)

                with dpg.child_window(tag="col_right", border=True, no_scrollbar=False):
                    with dpg.child_window(tag="pane_wafer", border=False, no_scrollbar=True):
                        dpg.add_text("Wafer View", color=(200, 200, 210))
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(tag="wafer_pad_l", width=1)
                            dpg.add_image(
                                "wafer_texture",
                                tag="wafer_image_widget",
                                width=ng.ui_image_size,
                                height=ng.ui_image_size,
                            )
                            dpg.add_spacer(tag="wafer_pad_r", width=1)
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(tag="wafer_btn_pad_l", width=1)
                            dpg.add_button(label="Refresh Grid", callback=update_grid_params_callback)
                            dpg.add_button(label="Clear Exposures", callback=clear_exposures_callback)
                            dpg.add_spacer(tag="wafer_btn_pad_r", width=1)

                    dpg.add_spacer(height=LAYOUT_ROW_GAP)

                    with dpg.child_window(tag="pane_settings", border=False, no_scrollbar=False):
                        dpg.add_text("Settings", color=(200, 200, 210))
                        with dpg.tab_bar():
                            with dpg.tab(label="Exposure"):
                                dpg.add_input_float(
                                    label="Time (s)",
                                    default_value=ng.stage.exposure_time,
                                    tag="exposure_time",
                                    callback=update_exposure_params_callback,
                                    width=140,
                                )
                                dpg.add_slider_float(
                                    label="Power (%)",
                                    default_value=ng.stage.exposure_power,
                                    min_value=0,
                                    max_value=100,
                                    tag="exposure_power",
                                    callback=update_exposure_params_callback,
                                    width=220,
                                )

                            with dpg.tab(label="Grid"):
                                dpg.add_input_int(
                                    label="Rows",
                                    default_value=ng.stage.grid_rows,
                                    tag="grid_rows",
                                    min_value=1,
                                    max_value=40,
                                    width=140,
                                )
                                dpg.add_input_int(
                                    label="Columns",
                                    default_value=ng.stage.grid_columns,
                                    tag="grid_columns",
                                    min_value=1,
                                    max_value=40,
                                    width=140,
                                )
                                dpg.add_input_float(
                                    label="X Spacing (mm)",
                                    default_value=ng.stage.grid_spacing_x,
                                    tag="grid_spacing_x",
                                    width=140,
                                )
                                dpg.add_input_float(
                                    label="Y Spacing (mm)",
                                    default_value=ng.stage.grid_spacing_y,
                                    tag="grid_spacing_y",
                                    width=140,
                                )
                                dpg.add_button(label="Update Grid Preview", callback=update_grid_params_callback)

                            with dpg.tab(label="Wafer"):
                                dpg.add_combo(
                                    label="Type",
                                    items=["Silicon", "Glass", "Quartz", "Other"],
                                    default_value=ng.stage.wafer_type,
                                    tag="wafer_type",
                                    width=160,
                                )
                                dpg.add_input_float(
                                    label="Diameter (mm)",
                                    default_value=ng.stage.wafer_diameter,
                                    tag="wafer_diameter",
                                    width=140,
                                )
                                dpg.add_input_float(
                                    label="Edge Exclusion (mm)",
                                    default_value=ng.stage.edge_exclusion,
                                    tag="edge_exclusion",
                                    width=140,
                                )
                                dpg.add_button(label="Apply Wafer Settings", callback=update_wafer_params_callback)

                            with dpg.tab(label="Stage"):
                                dpg.add_input_float(
                                    label="Speed (mm/s)",
                                    default_value=ng.stage.speed,
                                    tag="stage_speed",
                                    width=140,
                                )
                                dpg.add_input_float(
                                    label="Accel (mm/s²)",
                                    default_value=ng.stage.acceleration,
                                    tag="stage_accel",
                                    width=140,
                                )
                                dpg.add_input_float(
                                    label="Jog Speed (mm/s)",
                                    default_value=ng.stage.jog_speed,
                                    tag="jog_speed",
                                    width=140,
                                )
                                dpg.add_button(label="Apply Motion Settings", callback=update_stage_params_callback)

                            with dpg.tab(label="Display"):
                                dpg.add_text("DMD / pattern output monitor")
                                dpg.add_combo(
                                    items=get_monitor_labels(),
                                    default_value=get_monitor_labels()[0],
                                    tag="monitor_combo",
                                    callback=on_monitor_selected,
                                    width=360,
                                )
                                dpg.add_button(label="Refresh Monitor List", callback=refresh_monitor_list)

        dpg.add_separator()
        dpg.add_text(
            "Ready — simulation mode (no hardware connected).",
            tag="status_text",
            color=(180, 180, 190),
        )


def main(*, capture_smoke: bool = False) -> None:
    dpg.create_context()
    dpg.create_viewport(
        title=ng.name,
        width=ng.window_width,
        height=ng.window_height,
        min_width=LAYOUT_MIN_WIDTH,
        min_height=LAYOUT_MIN_HEIGHT,
        resizable=True,
    )

    # Optional default pattern for a nicer first-run look
    if DEFAULT_PATTERN.exists():
        try:
            ng.load_pattern(DEFAULT_PATTERN)
        except Exception:
            pass

    if DEFAULT_CONFIG_PATH.exists():
        try:
            payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
            ng.stage.load_dict(payload.get("stage", {}))
            ng.selected_monitor_index = int(payload.get("selected_monitor_index", 0))
            pattern = payload.get("pattern_path")
            if pattern and Path(pattern).exists():
                ng.load_pattern(pattern)
        except Exception:
            pass

    ng.update_wafer_view()
    build_ui()
    sync_settings_to_ui()
    refresh_textures()
    refresh_position_labels()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("_main_window", True)
    ng.layout_size = (0, 0)
    apply_responsive_layout()

    dpg.set_viewport_resize_callback(callback=lambda: setattr(ng, "layout_size", (0, 0)))

    smoke_dir = APP_DIR / "screenshots"
    frame = 0
    while dpg.is_dearpygui_running():
        apply_responsive_layout()
        tick_job()
        dpg.render_dearpygui_frame()
        frame += 1

        if capture_smoke:
            if frame == 6:
                home_callback()
                push_wafer_and_status("Smoke test: homed stage.")
            if frame == 10:
                expose_callback()
            if frame == 14:
                smoke_dir.mkdir(exist_ok=True)
                dpg.output_frame_buffer(str(smoke_dir / "01_default_size.png"))
            if frame == 16:
                ng.layout_size = (0, 0)
                dpg.set_viewport_width(1560)
                dpg.set_viewport_height(960)
            if frame == 26:
                dpg.output_frame_buffer(str(smoke_dir / "02_resized_window.png"))
            if frame == 38:
                dpg.stop_dearpygui()

    dpg.destroy_context()


if __name__ == "__main__":
    main(capture_smoke="--capture-smoke" in sys.argv)
