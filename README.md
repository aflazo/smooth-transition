# Smooth Transition

Genera animaciones suaves de transición entre dos imágenes con alineación facial.

## Instalación

```bash
pip install opencv-python numpy Pillow imageio imageio-ffmpeg gradio
```

## UI Web (Gradio)

```bash
python app.py
```

Abre `http://127.0.0.1:7860` en el navegador.

### Flujo:
1. **Sube las imágenes** X (origen) e Y (destino).
2. **Marca 3 puntos** en cada imagen: ojo izquierdo, ojo derecho y boca. Selecciona el punto activo con el radio button y haz click en la imagen para posicionarlo.
3. **Configura** duración, FPS, tamaño, easing, formato (GIF/MP4), hold y reverse.
4. **Genera** la animación. El sistema aplica una transformación afín para alinear ambas caras antes de hacer la transición.

## CLI

```bash
python main.py --x A.png --y B.png --out morph.gif
python main.py --x A.png --y B.png --out morph.mp4 --fps 30 --seconds 3
python main.py --x A.png --y B.png --out loop.gif --reverse
```

### Opciones

```
--x           Imagen de origen (requerido)
--y           Imagen de destino (requerido)
--out         Archivo de salida: .gif o .mp4 (requerido)
--fps         Frames por segundo (default: 24)
--seconds     Duración de la transición (default: 2.5)
--size        Tamaño cuadrado en px (default: 512)
--easing      Función de easing (default: smoothstep)
--hold        Pausa en primer/último frame en segundos (default: 0.3)
--reverse     Generar animación X→Y→X
```

## Easings disponibles

| Easing        | Descripción                          |
|---------------|--------------------------------------|
| `linear`      | Velocidad constante                  |
| `smoothstep`  | Aceleración y desaceleración suave   |
| `smootherstep`| Aún más suave (derivada segunda = 0) |
| `ease-in`     | Empieza lento, termina rápido        |
| `ease-out`    | Empieza rápido, termina lento        |
| `ease-in-out` | Suave en ambos extremos              |
