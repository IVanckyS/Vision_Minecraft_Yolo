### Detección de Mobs en Minecraft utilizando YOLOv11

---

## 📝 Descripción del proyecto
Este proyecto corresponde al trabajo final del curso **Taller de Introducción a Visión por Computadora** de la **Universidad del Bío-Bío**. El objetivo principal es la implementación de un modelo de Deep Learning basado en **YOLOv11** orientado a la **detección de objetos** en un entorno digital controlado (Minecraft)[cite: 7, 9].

El sistema es capaz de detectar y clasificar distintos mobs del juego, enfrentando desafíos críticos de visión por computador como variaciones de iluminación y fondos complejos.

---

## 👾 Mobs Detectados
El modelo fue entrenado para reconocer las siguientes categorías:

* **Pacíficos:** Cow, Villager, Parrot.
* **Neutrales:** Wolf, Iron Golem, Enderman.
* **Hostiles:** Creeper, Skeleton, Wither Skeleton.

---

## 🛠️ Tecnologías Utilizadas
* **Python 3.10**
***YOLOv11 (Ultralytics)**
* **PyTorch & OpenCV**
***Roboflow:** Utilizado para el análisis, selección y etiquetado de datos

---

## 📂 Estructura del Repositorio

```text
Vision_Minecraft_Yolo/
│
├── models/
│   └── best.pt             # Modelo entrenado (.pt) [cite: 59]
│
├── scripts/
│   ├── train.py            # Código de entrenamiento [cite: 50]
│   ├── detect_image.py     # Inferencia en imágenes
│   └── detect_video.py     # Inferencia en videos
│
├── demo_results/
│   └── *.gif               # Evidencia de funcionamiento 
│
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Documentación
└── .gitignore              # Archivos excluidos (entornos y datasets pesados)
