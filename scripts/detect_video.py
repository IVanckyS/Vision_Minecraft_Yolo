from ultralytics import YOLO
import cv2
import os
import time

# ------------------------
# CONFIGURACIÓN
# ------------------------

# Ruta correcta al modelo entrenado
model_path = r"C:\Users\ivanc\Desktop\Yoloprueba\runs\detect\train8\weights\best.pt"

# Video de entrada (MKV o MP4)
video_input = r"C:\Users\ivanc\Videos\wolf2.mkv"

# Video procesado de salida
video_output = r"C:\Users\ivanc\Desktop\Yoloprueba\wolf2_detected.mp4"

# ------------------------
# CARGAR MODELO
# ------------------------
print("\n📦 Cargando modelo YOLOv11...")
model = YOLO(model_path)

# ------------------------
# VALIDAR VIDEO
# ------------------------
print("\n Verificando archivo de entrada...")
print("Existe:", os.path.exists(video_input))

cap = cv2.VideoCapture(video_input)

if not cap.isOpened():
    print("\n ERROR: OpenCV NO pudo abrir el video.")
    print("➡ Problema del formato. Usa MKV o remux a MP4.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\n Información del video:")
print(f"FPS: {fps}")
print(f"Frames totales: {total_frames}")
print(f"Resolución: {width} x {height}")

if total_frames == 0:
    print("\n ERROR: El archivo está dañado o no contiene frames.")
    exit()

# ------------------------
# COLORES POR TIPO DE MOB
# ------------------------
colors = {
    # Pacíficos -> amarillo
    "cow": (0, 255, 255),
    "parrot": (0, 255, 255),
    "Villager": (0, 255, 255),

    # “Neutrales” -> violeta
    "wolf": (255, 0, 255),
    "Enderman": (255, 0, 255),
    "Golem": (255, 0, 255),

    # Hostiles -> rojo
    "creeper": (0, 0, 255),
    "whiter skeleton": (0, 0, 255),
    "skeleton": (0, 0, 255),
}

# ------------------------
# SALIDA
# ------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

print("\n Procesando video...\n")

start = time.time()
frame_count = 0

# ------------------------
# LOOP DE PROCESAMIENTO
# ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("\n✔ Fin del video.")
        break

    frame_count += 1

    if frame_count % 30 == 0:
        elapsed = time.time() - start
        progress = (frame_count / total_frames) * 100
        eta = (elapsed / frame_count) * (total_frames - frame_count)
        print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - ETA: {eta:.1f}s")

    # Ejecutar YOLO
    results = model(frame, imgsz=640)[0]

    # Dibujar cajas a mano con colores personalizados
    for box in results.boxes:
        # Coordenadas de la bounding box
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()

        # Clase y confianza
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        conf = float(box.conf[0])

        # Color según el nombre de la clase (default blanco si no está)
        color = colors.get(label, (255, 255, 255))

        # Caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Texto etiqueta + confianza
        text = f"{label} {conf:.2f}"
        cv2.putText(
            frame, text, (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    # Escribir frame anotado al video de salida
    out.write(frame)

cap.release()
out.release()

print("\n==============================")
print(" PROCESO COMPLETADO")
print(f" Frames procesados: {frame_count}")
print(f" Video guardado en: {video_output}")
print("==============================\n")
