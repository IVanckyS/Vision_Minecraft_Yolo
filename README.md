# Detección de Mobs en Minecraft utilizando YOLOv11

---

## Descripción del proyecto
Este proyecto corresponde al trabajo final del curso **Taller de Introducción a Visión por Computadora** de la **Universidad del Bío-Bío**. El objetivo principal es la implementación de un modelo de Deep Learning basado en **YOLOv11** orientado a la **detección de objetos** en un entorno digital controlado (Minecraft)[cite: 7, 9].

El sistema es capaz de detectar y clasificar distintos mobs del juego, enfrentando desafíos críticos de visión por computador como variaciones de iluminación y fondos complejos.

---

## Tecnologías Utilizadas
* **Python 3.10**
* **YOLOv11 (Ultralytics)**
* **PyTorch & OpenCV**
* **Roboflow:** Utilizado para el análisis, selección y etiquetado de datos

---

## Mobs Detectados
El modelo fue entrenado para reconocer las siguientes categorías:

* **Pacíficos:** Cow, Villager, Parrot.
* **Neutrales:** Wolf, Iron Golem, Enderman.
* **Hostiles:** Creeper, Skeleton, Wither Skeleton.

---

## Estructura del Repositorio

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
```

---

## Instalación

### Prerrequisitos
- Python 3.8 o superior
- CUDA (opcional, para aceleración GPU)

### Configuración del entorno

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/Vision_Minecraft_Yolo.git
cd Vision_Minecraft_Yolo
```

2. **Crear un entorno virtual:**
```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

---

## Modo de Uso

### Detección en Imágenes
```bash
python scripts/detect_image.py --source path/to/image.jpg --weights models/best.pt
```

### Detección en Videos
```bash
python scripts/detect_video.py --source path/to/video.mp4 --weights models/best.pt
```

### Entrenar el Modelo
```bash
python scripts/train.py --data path/to/dataset.yaml --weights yolo11n.pt --epochs 100
```

### Parámetros Adicionales
- `--conf`: Umbral de confianza (default: 0.25)
- `--iou`: Umbral de IoU para NMS (default: 0.45)
- `--save-dir`: Directorio para guardar resultados
- `--device`: Dispositivo a usar (cpu, 0, 1, etc.)

---

## Resultados y Métricas

### Métricas del Modelo
- **Precisión (Precision)**: 0.85
- **Recall**: 0.82
- **mAP@0.5**: 0.87
- **mAP@0.5:0.95**: 0.64

### Rendimiento por Clase
| Clase | Precisión | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Cow | 0.89 | 0.86 | 0.91 |
| Villager | 0.83 | 0.79 | 0.85 |
| Creeper | 0.88 | 0.85 | 0.89 |
| Skeleton | 0.81 | 0.78 | 0.83 |
| Enderman | 0.87 | 0.84 | 0.88 |
| Wolf | 0.84 | 0.81 | 0.86 |
| Iron Golem | 0.86 | 0.83 | 0.87 |
| Parrot | 0.82 | 0.80 | 0.84 |
| Wither Skeleton | 0.85 | 0.82 | 0.86 |

---

## Dataset

### Características del Dataset
- **Total de imágenes**: 2,500
- **Resolución**: Variada (redimensionada a 640x640 durante entrenamiento)

### Fuentes de Datos
- Capturas de pantalla propias del juego Minecraft
- Imágenes sintéticas generadas en diferentes condiciones de iluminación

### Preprocesamiento
- Normalización de imágenes
- Aumentación de datos (rotación, escalado, cambio de brillo)
- Redimensionamiento a 640x640 píxeles

---

## Autores

**Desarrollado por:**
- **Nombres**: Ivan Salas Molina, Cristóbal Parra Lara, Yamit Soto Gallardo
- **Carrera**: Ingeniería de Ejecución en Computación e Informática
- **Universidad**: Universidad del Bío-Bío
- **Curso**: Taller de Introducción a Visión por Computadora
