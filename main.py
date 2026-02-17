#!/usr/bin/env python3
"""
smooth_transition — CLI que genera una animación suave de transición entre dos imágenes.

Uso rápido:
    python main.py --x A.png --y B.png --out morph.gif
    python main.py --x A.png --y B.png --out morph.mp4 --fps 30 --seconds 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Carga y preprocesado
# ---------------------------------------------------------------------------

def load_and_preprocess(path: str, size: int) -> np.ndarray:
    """Carga una imagen, hace crop centrado cuadrado y redimensiona a (size, size)."""
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")

    # Crop centrado cuadrado
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    img = img[y0 : y0 + side, x0 : x0 + side]

    # Resize
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)

    # BGR → RGB (todas las salidas trabajan en RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def load_and_warp_to_target(
    path: str,
    src_points: np.ndarray,
    target_points: np.ndarray,
    size: int,
) -> np.ndarray:
    """Carga una imagen y aplica transformación afín para alinear src_points → target_points.

    src_points: array (3, 2) con coordenadas en píxeles de la imagen original
                [ojo_izq, ojo_der, boca].
    target_points: array (3, 2) con coordenadas destino en el frame de salida (size×size).
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")

    src = src_points.astype(np.float32)
    dst = target_points.astype(np.float32)
    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return warped.astype(np.float32)


# ---------------------------------------------------------------------------
# Funciones de easing
# ---------------------------------------------------------------------------

EASINGS = {}

def _register(name):
    def decorator(fn):
        EASINGS[name] = fn
        return fn
    return decorator

@_register("linear")
def _linear(t: float) -> float:
    return t

@_register("smoothstep")
def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)

@_register("smootherstep")
def _smootherstep(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

@_register("ease-in")
def _ease_in(t: float) -> float:
    return t * t

@_register("ease-out")
def _ease_out(t: float) -> float:
    return t * (2.0 - t)

@_register("ease-in-out")
def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return -1.0 + (4.0 - 2.0 * t) * t


def easing_fn(name: str):
    """Devuelve la función de easing por nombre."""
    if name not in EASINGS:
        raise ValueError(
            f"Easing '{name}' no soportado. Opciones: {', '.join(EASINGS)}"
        )
    return EASINGS[name]


# ---------------------------------------------------------------------------
# Generación de frames
# ---------------------------------------------------------------------------

def blend_frames(
    img_x: np.ndarray,
    img_y: np.ndarray,
    fps: int,
    seconds: float,
    easing_name: str,
    hold: float,
    reverse: bool,
) -> list[np.ndarray]:
    """Genera la lista completa de frames (uint8, RGB)."""
    ease = easing_fn(easing_name)
    n_transition = max(int(round(fps * seconds)), 2)
    n_hold = max(int(round(fps * hold)), 0)

    frames: list[np.ndarray] = []

    # Hold primer frame
    first = np.clip(img_x, 0, 255).astype(np.uint8)
    for _ in range(n_hold):
        frames.append(first)

    # Transición X → Y
    transition_forward: list[np.ndarray] = []
    for i in range(n_transition):
        t = i / (n_transition - 1)
        te = ease(t)
        blended = (1.0 - te) * img_x + te * img_y
        transition_forward.append(np.clip(blended, 0, 255).astype(np.uint8))
    frames.extend(transition_forward)

    # Hold último frame
    last = np.clip(img_y, 0, 255).astype(np.uint8)
    for _ in range(n_hold):
        frames.append(last)

    # Reverse: Y → X
    if reverse:
        frames.extend(reversed(transition_forward))
        for _ in range(n_hold):
            frames.append(first)

    return frames


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------

def write_gif(frames: list[np.ndarray], out_path: str, fps: int) -> None:
    """Escribe un GIF optimizado usando Pillow."""
    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / fps)

    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"GIF guardado: {out_path}  ({len(frames)} frames, {fps} fps)")


def write_mp4(frames: list[np.ndarray], out_path: str, fps: int) -> None:
    """Escribe un MP4 usando imageio + ffmpeg."""
    try:
        import imageio.v3 as iio

        with iio.imopen(out_path, "w", plugin="pyav") as writer:
            for f in frames:
                writer.write(f, codec="libx264", fps=fps)
    except Exception:
        # Fallback: imageio v2 con ffmpeg
        import imageio

        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()

    print(f"MP4 guardado: {out_path}  ({len(frames)} frames, {fps} fps)")


def write_gif_or_mp4(frames: list[np.ndarray], out_path: str, fps: int) -> None:
    ext = Path(out_path).suffix.lower()
    if ext == ".gif":
        write_gif(frames, out_path, fps)
    elif ext in (".mp4", ".avi", ".mov", ".mkv"):
        write_mp4(frames, out_path, fps)
    else:
        raise ValueError(f"Formato de salida no soportado: {ext}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera una animación suave de transición entre dos imágenes."
    )
    p.add_argument("--x", required=True, help="Ruta a la imagen de origen")
    p.add_argument("--y", required=True, help="Ruta a la imagen de destino")
    p.add_argument("--out", required=True, help="Ruta de salida (.gif o .mp4)")
    p.add_argument("--fps", type=int, default=24, help="Frames por segundo (default: 24)")
    p.add_argument("--seconds", type=float, default=2.5, help="Duración de la transición en segundos (default: 2.5)")
    p.add_argument("--size", type=int, default=512, help="Tamaño cuadrado de salida en px (default: 512)")
    p.add_argument(
        "--easing",
        default="smoothstep",
        choices=list(EASINGS.keys()),
        help="Función de easing (default: smoothstep)",
    )
    p.add_argument("--hold", type=float, default=0.3, help="Segundos de pausa en primer/último frame (default: 0.3)")
    p.add_argument("--reverse", action="store_true", help="Generar loop X→Y→X")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validar que existan los archivos
    for label, path in [("--x", args.x), ("--y", args.y)]:
        if not Path(path).is_file():
            sys.exit(f"Error: archivo no encontrado para {label}: {path}")

    # Validar extensión de salida
    ext = Path(args.out).suffix.lower()
    if ext not in (".gif", ".mp4", ".avi", ".mov", ".mkv"):
        sys.exit(f"Error: formato de salida no soportado: {ext}")

    print(f"Cargando imágenes ({args.size}×{args.size}) …")
    img_x = load_and_preprocess(args.x, args.size)
    img_y = load_and_preprocess(args.y, args.size)

    total_frames = int(round(args.fps * args.seconds))
    hold_frames = int(round(args.fps * args.hold))
    expected = total_frames + 2 * hold_frames
    if args.reverse:
        expected += total_frames + hold_frames
    print(
        f"Generando {expected} frames "
        f"({args.seconds}s transición, {args.hold}s hold, easing={args.easing}"
        f"{', reverse' if args.reverse else ''}) …"
    )

    frames = blend_frames(
        img_x, img_y,
        fps=args.fps,
        seconds=args.seconds,
        easing_name=args.easing,
        hold=args.hold,
        reverse=args.reverse,
    )

    write_gif_or_mp4(frames, args.out, args.fps)
    print("¡Listo!")


if __name__ == "__main__":
    main()
