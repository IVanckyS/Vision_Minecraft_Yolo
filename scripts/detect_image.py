from ultralytics import YOLO
import cv2
import os
import time

# ------------------------
# CONFIGURACIÓN
# ------------------------

# Ruta correcta al modelo entrenado
model_path = r"C:\Users\ivanc\Desktop\Yoloprueba\runs\detect\train7\weights\best.pt"

# Imagen de entrada
image_input = r"C:\Users\ivanc\Desktop\Yoloprueba\minecraft_test.png"  # <- cambia esto

# Imagen de salida anotada
image_output = r"C:\Users\ivanc\Desktop\Yoloprueba\minecraft_test_detected.png"

# ------------------------
# CARGAR MODELO
# ------------------------
print("\n Cargando modelo YOLOv11...")
model = YOLO(model_path)

# ------------------------
# VALIDAR IMAGEN
# ------------------------
print("\n Verificando archivo de entrada...")
print("Existe:", os.path.exists(image_input))

img = cv2.imread(image_input)

if img is None:
    print("\n ERROR: OpenCV NO pudo abrir la imagen.")
    exit()

height, width = img.shape[:2]
print("\n🖼 Información de la imagen:")
print(f"Resolución: {width} x {height}")

# ------------------------
# COLORES POR TIPO DE MOB
# ------------------------
colors = {
    # Pacíficos -> amarillo (BGR)
    "cow": (0, 255, 255),
    "parrot": (0, 255, 255),
    "Villager": (0, 255, 255),
    "villager": (0, 255, 255),

    # “Neutrales” -> violeta (BGR)
    "wolf": (255, 0, 255),
    "Enderman": (255, 0, 255),
    "enderman": (255, 0, 255),
    "Golem": (255, 0, 255),
    "golem": (255, 0, 255),

    # Hostiles -> rojo (BGR)
    "creeper": (0, 0, 255),
    "skeleton": (0, 0, 255),
    "whiter skeleton": (0, 0, 255),
    "wither skeleton": (0, 0, 255),
}

# ------------------------
# INFERENCIA EN LA IMAGEN
# ------------------------
print("\n🔍 Ejecutando YOLO sobre la imagen...")

start = time.time()
results = model(img, imgsz=640)[0]
elapsed = time.time() - start

print(f" Inferencia completada en {elapsed:.3f} s")
print(f" Objetos detectados: {len(results.boxes)}")

# ------------------------
# DIBUJAR CAJAS CON COLORES PERSONALIZADOS
# ------------------------
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
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Texto etiqueta + confianza
    text = f"{label} {conf:.2f}"
    cv2.putText(
        img, text, (x1, max(y1 - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
    )

# ------------------------
# GUARDAR RESULTADO
# ------------------------
cv2.imwrite(image_output, img)

print("\n==============================")
print(" PROCESO COMPLETADO")
print(f" Imagen de salida guardada en: {image_output}")
print("==============================\n")

# (Opcional) mostrar la imagen en una ventana
# cv2.imshow("Detecciones", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
