#!/usr/bin/env python3
"""
Smooth Transition — UI con alineación por 3 puntos (ojo izq, ojo der, boca).

Flujo:
1. Sube imagen X e Y.
2. Para cada imagen, selecciona qué punto colocar (radio) y haz click en la imagen.
3. Repite para los 3 puntos en ambas imágenes.
4. Click "Generar" → alineación afín + transición.
"""

import shutil
import tempfile

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import (
    EASINGS,
    blend_frames,
    load_and_warp_to_target,
    write_gif,
    write_mp4,
)

POINT_NAMES = ["Ojo izquierdo", "Ojo derecho", "Boca"]
POINT_COLORS = [(0, 180, 255), (0, 255, 100), (255, 100, 100)]  # cyan, green, red
DEFAULT_POINTS = {
    "Ojo izquierdo": (0.38, 0.38),
    "Ojo derecho": (0.62, 0.38),
    "Boca": (0.50, 0.62),
}
DISPLAY_MAX = 500


# ---------------------------------------------------------------------------
# Render overlay
# ---------------------------------------------------------------------------

def render_overlay(img_path: str, points: dict) -> np.ndarray | None:
    if img_path is None:
        return None
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    ratio = DISPLAY_MAX / max(w, h)
    nw, nh = int(w * ratio), int(h * ratio)
    img = img.resize((nw, nh), Image.LANCZOS)
    draw = ImageDraw.Draw(img, "RGBA")

    px_points = []
    for i, name in enumerate(POINT_NAMES):
        fx, fy = points.get(name, DEFAULT_POINTS[name])
        px, py = int(fx * nw), int(fy * nh)
        px_points.append((px, py))
        r, g, b = POINT_COLORS[i]
        # Filled circle
        draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(r, g, b, 220))
        draw.ellipse([px - 8, py - 8, px + 8, py + 8], outline=(255, 255, 255, 200), width=2)
        # Label
        draw.text((px + 12, py - 8), name, fill=(r, g, b, 255))

    # Draw triangle connecting the 3 points
    if len(px_points) == 3:
        tri = [tuple(p) for p in px_points]
        draw.polygon(tri, fill=(200, 200, 255, 30), outline=(200, 200, 255, 120))

    return np.array(img)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def persist_image(filepath):
    """Copy uploaded file to a persistent PNG (OpenCV can't read .avif/.webp)."""
    if filepath is None:
        return None
    persistent = tempfile.mktemp(suffix=".png", prefix="smooth_")
    Image.open(filepath).convert("RGB").save(persistent)
    return persistent


def on_upload(filepath):
    """When an image is uploaded, render overlay with default points."""
    if filepath is None:
        return None, DEFAULT_POINTS.copy(), None
    persistent = persist_image(filepath)
    overlay = render_overlay(persistent, DEFAULT_POINTS)
    return overlay, DEFAULT_POINTS.copy(), persistent


def on_click(filepath, points_state, active_point, evt: gr.SelectData):
    """When user clicks on the overlay image, move the active point there."""
    if filepath is None:
        return None, points_state

    # Get display dimensions to convert pixel click → fraction
    img = Image.open(filepath)
    w, h = img.size
    ratio = DISPLAY_MAX / max(w, h)
    nw, nh = int(w * ratio), int(h * ratio)

    cx_frac = evt.index[0] / nw
    cy_frac = evt.index[1] / nh
    cx_frac = max(0.02, min(0.98, cx_frac))
    cy_frac = max(0.02, min(0.98, cy_frac))

    points = dict(points_state) if points_state else DEFAULT_POINTS.copy()
    points[active_point] = (cx_frac, cy_frac)

    overlay = render_overlay(filepath, points)
    return overlay, points


def generate(
    x_path, y_path,
    pts_x, pts_y,
    path_x_persist, path_y_persist,
    seconds, fps, easing, fmt, reverse, hold, size,
):
    # Use persistent paths (immune to Gradio temp cleanup)
    x_path = path_x_persist or x_path
    y_path = path_y_persist or y_path
    if x_path is None or y_path is None:
        raise gr.Error("Debes subir ambas imágenes (X e Y).")
    if pts_x is None or pts_y is None:
        raise gr.Error("Marca los 3 puntos en ambas imágenes.")

    size = int(size)

    # Target points: canonical face position in the output frame
    # Eyes at 35% from top, mouth at 70%, horizontally centered with 30% eye spread
    target = np.array([
        [size * 0.35, size * 0.35],  # left eye
        [size * 0.65, size * 0.35],  # right eye
        [size * 0.50, size * 0.70],  # mouth
    ], dtype=np.float32)

    # Convert fractional points to pixel coordinates in original images
    def frac_to_pixels(path, pts):
        pil_img = Image.open(path)
        w, h = pil_img.size
        return np.array([
            [pts["Ojo izquierdo"][0] * w, pts["Ojo izquierdo"][1] * h],
            [pts["Ojo derecho"][0] * w, pts["Ojo derecho"][1] * h],
            [pts["Boca"][0] * w, pts["Boca"][1] * h],
        ], dtype=np.float32)

    src_x = frac_to_pixels(x_path, pts_x)
    src_y = frac_to_pixels(y_path, pts_y)

    img_x = load_and_warp_to_target(x_path, src_x, target, size)
    img_y = load_and_warp_to_target(y_path, src_y, target, size)

    frames = blend_frames(
        img_x, img_y,
        fps=int(fps),
        seconds=float(seconds),
        easing_name=easing,
        hold=float(hold),
        reverse=reverse,
    )

    ext = ".gif" if fmt == "GIF" else ".mp4"
    out_path = tempfile.mktemp(suffix=ext)

    if ext == ".gif":
        write_gif(frames, out_path, int(fps))
    else:
        write_mp4(frames, out_path, int(fps))

    return out_path, out_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Smooth Transition") as app:
    gr.Markdown(
        "# Smooth Transition\n"
        "Genera una animación suave entre dos imágenes con alineación facial.\n\n"
        "**Flujo:** Sube las imágenes → Marca 3 puntos en cada una (ojo izq, ojo der, boca) → Genera"
    )

    # State for the 3 points per image (dict: name → (fx, fy))
    pts_x_state = gr.State(DEFAULT_POINTS.copy())
    pts_y_state = gr.State(DEFAULT_POINTS.copy())
    # Persistent image paths (copies that won't be cleaned by Gradio)
    path_x_state = gr.State(None)
    path_y_state = gr.State(None)

    # --- Step 1: Upload ---
    gr.Markdown("### 1. Sube las imágenes")
    with gr.Row():
        img_x = gr.Image(label="Imagen X (origen)", type="filepath", height=180)
        img_y = gr.Image(label="Imagen Y (destino)", type="filepath", height=180)

    # --- Step 2: Align ---
    gr.Markdown(
        "### 2. Marca los puntos de referencia\n"
        "Selecciona qué punto quieres colocar y luego **haz click en la imagen** donde corresponda. "
        "Repite para los 3 puntos en cada imagen."
    )
    with gr.Row():
        active_x = gr.Radio(POINT_NAMES, value="Ojo izquierdo", label="Punto activo (imagen X)")
        active_y = gr.Radio(POINT_NAMES, value="Ojo izquierdo", label="Punto activo (imagen Y)")

    with gr.Row():
        overlay_x = gr.Image(label="Imagen X — click para colocar el punto activo", interactive=False)
        overlay_y = gr.Image(label="Imagen Y — click para colocar el punto activo", interactive=False)

    # Upload → persist + render default overlay
    img_x.change(fn=on_upload, inputs=[img_x], outputs=[overlay_x, pts_x_state, path_x_state])
    img_y.change(fn=on_upload, inputs=[img_y], outputs=[overlay_y, pts_y_state, path_y_state])

    # Click on overlay → reposition active point (use persistent path)
    overlay_x.select(
        fn=on_click,
        inputs=[path_x_state, pts_x_state, active_x],
        outputs=[overlay_x, pts_x_state],
    )
    overlay_y.select(
        fn=on_click,
        inputs=[path_y_state, pts_y_state, active_y],
        outputs=[overlay_y, pts_y_state],
    )

    # --- Step 3: Generate ---
    gr.Markdown("### 3. Configura y genera")
    with gr.Row():
        seconds = gr.Slider(0.5, 10, value=2.5, step=0.5, label="Duración (segundos)")
        fps = gr.Slider(10, 60, value=24, step=1, label="FPS")
        size = gr.Slider(128, 1024, value=512, step=64, label="Tamaño (px)")

    with gr.Row():
        easing = gr.Dropdown(list(EASINGS.keys()), value="smoothstep", label="Easing")
        fmt = gr.Radio(["GIF", "MP4"], value="MP4", label="Formato")
        hold = gr.Slider(0, 2, value=0.3, step=0.1, label="Hold (segundos)")
        reverse = gr.Checkbox(label="Reverse (X→Y→X)", value=False)

    btn = gr.Button("Generar", variant="primary", size="lg")

    with gr.Row():
        preview = gr.Video(label="Vista previa")
        download = gr.File(label="Descargar")

    btn.click(
        fn=generate,
        inputs=[
            img_x, img_y,
            pts_x_state, pts_y_state,
            path_x_state, path_y_state,
            seconds, fps, easing, fmt, reverse, hold, size,
        ],
        outputs=[preview, download],
    )

if __name__ == "__main__":
    app.launch()
