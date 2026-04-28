# 🤖 NAO Audio Cleaner · RoboCup HSL

**NAO Audio Cleaner** es una herramienta web interactiva construida con [Gradio](https://gradio.app/) para la limpieza de audio y la detección de silbatos de árbitro. Está diseñada específicamente para procesar grabaciones de los robots **NAO v6** en el contexto de la **RoboCup Humanoid Soccer League (HSL)**.

Esta aplicación no solo limpia y analiza audios en lote, sino que incorpora un **sistema de aprendizaje adaptativo** que mejora la precisión de la detección de silbatos a medida que el usuario proporciona feedback sobre los resultados.

---

## ✨ Características Principales

### 🎧 Limpieza de Audio Avanzada
- **Reducción de Ruido:** Filtra ruido de fondo usando un perfil de ruido estimado del mismo clip (`noisereduce`).
- **Filtro Paso-Banda (Band-pass):** Retiene solo las frecuencias útiles (por defecto 100 Hz - 10 kHz), eliminando las vibraciones mecánicas de los motores del robot.
- **Normalización de Volumen:** Maximiza la amplitud del audio sin causar distorsión.

### 🕵️ Detección de Silbatos
El algoritmo analiza el espectro del audio buscando características típicas de un silbato de árbitro:
- Concentración de energía en el rango de frecuencias esperado (2000 Hz - 8000 Hz).
- Duración mínima sostenida del pico de energía.
- Prominencia de picos espectrales (peaks) por encima de la media de ruido.

### 🧠 Aprendizaje Adaptativo (Feedback)
- **Interfaz de Revisión:** Permite al usuario escuchar cada audio y validar si la clasificación ("Silbato" o "Solo Ruido") es correcta.
- **Auto-Ajuste de Umbrales:** A medida que el usuario corrige falsos positivos o falsos negativos, el sistema realiza una búsqueda en cuadrícula (grid-search) para optimizar los umbrales de detección (`energy_ratio_threshold` y `min_duration_s`), maximizando el F1-Score.
- **Persistencia:** El feedback se guarda localmente en `feedback_data.json` para no perder el progreso del entrenamiento entre sesiones.

### 📦 Procesamiento por Lotes (Batch)
- Sube múltiples archivos simultáneamente.
- Descarga los archivos originales y los audios limpios de manera individual o empaquetados en un archivo `.zip`.

---

## 🛠️ Requisitos Previos

Para ejecutar la aplicación necesitas Python 3.8 o superior y las siguientes librerías. Asegúrate de tener instalado un entorno virtual o instalar directamente las dependencias.

```bash
pip install numpy pandas librosa soundfile noisereduce scipy gradio
```

> **Nota:** `librosa` puede requerir la instalación de `ffmpeg` a nivel de sistema operativo para procesar ciertos formatos de audio como `.mp3` o `.ogg`.

---

## 🚀 Instalación y Ejecución

1. Clona o descarga el código fuente del proyecto.
2. Abre una terminal en el directorio del proyecto (donde se encuentra `audio_cleaner_app.py`).
3. Ejecuta el script principal:

```bash
python audio_cleaner_app.py
```

4. Tras unos segundos, la terminal mostrará un mensaje indicando que el servidor está activo. Abre tu navegador web en la dirección indicada (por defecto `http://localhost:7860`).

---

## 📖 Guía de Uso

1. **Subir Archivos:** Arrastra o selecciona tus archivos de audio (`.wav`, `.mp3`, etc.) en la sección correspondiente.
2. **Ajustar Parámetros de Limpieza:** 
   - Modifica la "Fuerza de reducción de ruido" si el audio limpio suena demasiado metálico (se recomienda bajar de 0.75 a 0.5 - 0.6).
   - Activa/desactiva el filtro paso-banda y la normalización según tus necesidades.
3. **Analizar:** Haz clic en **"🚀 Limpiar y analizar"**.
4. **Revisar Resultados:** En el panel derecho verás una tabla con el análisis de cada audio, mostrando si se detectó un silbato, la energía máxima y la duración.
5. **Entrenar el Modelo (Feedback):** 
   - Ve a la sección **"Revisar y Corregir Clasificación"**.
   - Selecciona un audio, escúchalo usando el reproductor.
   - Indica si **"Tiene silbato"** o **"No tiene silbato"**.
   - Al llegar a al menos 5 muestras con diferentes clases etiquetadas, el sistema comenzará a ajustar sus umbrales automáticamente.
   - Presiona **"🔄 Volver a analizar originales con nuevos umbrales"** para aplicar lo que ha aprendido a la lista actual de archivos.
6. **Descargar:** Usa los botones de descarga `.zip` o los reproductores individuales para obtener tus archivos de audio procesados.

---

## 📁 Estructura del Proyecto

```text
/silbato
│
├── audio_cleaner_app.py    # Script principal y lógica de la aplicación (Gradio)
├── feedback_data.json      # Base de datos local (JSON) con el feedback del usuario
└── README.md               # Este archivo de documentación
```

---

## ⚙️ Configuración del Algoritmo

Si deseas modificar los valores iniciales (antes del entrenamiento adaptativo), puedes editar el diccionario `CFG` al inicio de `audio_cleaner_app.py`:

- `whistle_freq_min`: Frecuencia mínima del silbato (2000 Hz)
- `whistle_freq_max`: Frecuencia máxima del silbato (8000 Hz)
- `energy_ratio_threshold`: Umbral de energía base (0.25)
- `peak_prominence_db`: Prominencia mínima para detección de picos (12.0 dB)
- `min_duration_s`: Duración mínima sostenida (0.08 s)
