# smooth_transition

CLI que genera una animación suave de transición (cross-fade con easing) entre dos imágenes.

## Instalación

```bash
pip install opencv-python numpy Pillow imageio imageio-ffmpeg
```

## Uso

### GIF básico

```bash
python main.py --x X.png --y Y.png --out morph.gif
```

### MP4 con parámetros personalizados

```bash
python main.py --x X.png --y Y.png --out morph.mp4 --fps 30 --seconds 3
```

### GIF en loop (X → Y → X)

```bash
python main.py --x X.png --y Y.png --out loop.gif --reverse
```

### Todas las opciones

```
--x           Imagen de origen (requerido)
--y           Imagen de destino (requerido)
--out         Archivo de salida: .gif o .mp4 (requerido)
--fps         Frames por segundo (default: 24)
--seconds     Duración de la transición (default: 2.5)
--size        Tamaño cuadrado en px — crop centrado + resize (default: 512)
--easing      Función de easing (default: smoothstep)
              Opciones: linear, smoothstep, smootherstep, ease-in, ease-out, ease-in-out
--hold        Pausa en primer/último frame en segundos (default: 0.3)
--reverse     Generar animación X→Y→X (útil para GIF en loop)
```

## Ejemplos de easing

| Easing        | Descripción                          |
|---------------|--------------------------------------|
| `linear`      | Velocidad constante                  |
| `smoothstep`  | Aceleración y desaceleración suave   |
| `smootherstep`| Aún más suave (derivada segunda = 0) |
| `ease-in`     | Empieza lento, termina rápido        |
| `ease-out`    | Empieza rápido, termina lento        |
| `ease-in-out` | Suave en ambos extremos              |
