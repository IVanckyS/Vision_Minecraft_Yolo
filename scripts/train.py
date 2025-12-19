from ultralytics import YOLO

def main():
    # Ruta al data.yaml del dataset v9
    data_yaml = r"C:\Users\ivanc\Desktop\Yoloprueba\datasets\Minecraft mobs.v9i.yolov11\data.yaml"

    # Cargar modelo base YOLOv11n
    model = YOLO("yolo11n.pt")

    model.train(
        data=data_yaml,

        # Entrenamiento + early stopping
        epochs=200,        # Máximo de épocas
        patience=40,       # Si no mejora en 40 épocas, se detiene solo

        # Parámetros básicos
        batch=16,
        imgsz=640,
        device=0,
        workers=0,         # Importante en Windows para evitar problemas de multiprocessing
        lr0=0.01,
        pretrained=True,
        optimizer="SGD",
        amp=True,

        # Augmentations pensadas para Minecraft
        hsv_h=0.015,       # Variación leve de tono
        hsv_s=0.7,         # Variación fuerte de saturación (antorchas, biomas, etc.)
        hsv_v=0.4,         # Variación de brillo (día/noche, cuevas)
        degrees=10.0,      # Rotación leve
        fliplr=0.5,        # Flip horizontal sí tiene sentido
        mosaic=1.0,        # Mosaic activado (bueno para objetos pequeños y variados)
    )

    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()
