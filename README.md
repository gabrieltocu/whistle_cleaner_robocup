# 🤖 NAO Audio Cleaner · RoboCup HSL (Pro Version)

**NAO Audio Cleaner** es una herramienta web interactiva orientada a la limpieza de audio y la detección de silbatos de árbitro. Desarrollada para equipos de la **RoboCup Humanoid Soccer League (HSL)**, procesa audios capturados en los robots **NAO v6**.

Originalmente construida como un monolito en Gradio, **se ha refactorizado a una arquitectura Cliente-Servidor premium** utilizando **FastAPI (Python)** para el motor analítico profundo y **HTML/CSS/JS (Vainilla)** puro para lograr un *Dashboard* con una estética Cyber-Punk espectacular (Glassmorphism).

---

## ✨ Características Principales

### 🎧 Limpieza de Audio Avanzada
- **Reducción de Ruido:** Filtra ruido de fondo usando un perfil de ruido estimado del mismo clip (`noisereduce`).
- **Filtro Paso-Banda (Band-pass):** Retiene solo las frecuencias útiles (por defecto 100 Hz - 10 kHz), eliminando las vibraciones mecánicas de los motores del robot.
- **Normalización de Volumen Automática:** Maximiza la amplitud del audio sin causar distorsión.

### 🕵️ Detección de Silbatos (Inteligencia del Algoritmo)
El algoritmo analiza el espectro del audio buscando características típicas de un silbato de árbitro:
- Concentración de energía en el rango de frecuencias esperado (2000 Hz - 8000 Hz).
- Duración mínima sostenida del pico de energía.
- Prominencia de picos espectrales (peaks) por encima de la media de ruido.

### 🧠 Aprendizaje Adaptativo (Feedback)
- **Interfaz de Revisión:** Permite al usuario escuchar cada audio en el nuevo dashboard y validar si la clasificación ("Silbato" o "Solo Ruido") es correcta.
- **Auto-Ajuste de Umbrales:** A medida que el usuario marca validaciones, el sistema realiza una búsqueda en cuadrícula (grid-search) para optimizar los umbrales de detección (`energy_ratio_threshold` y `min_duration_s`), maximizando el F1-Score.
- **Persistencia:** El feedback se guarda localmente en un JSON para no perder el progreso del entrenamiento entre sesiones.

### 📦 Procesamiento por Lotes (Batch)
- Sube múltiples archivos simultáneamente mediante Drag & Drop al dashboard.
- Todos son analizados en lista a velocidad luz.

---

## 🛠️ Requisitos Previos

Para ejecutar la aplicación necesitas Python 3.8 o superior y las siguientes librerías:

```bash
pip install fastapi uvicorn python-multipart numpy pandas librosa soundfile noisereduce scipy
```

> **Nota:** `librosa` puede requerir la instalación de `ffmpeg` a nivel de sistema operativo para procesar ciertos formatos de audio como `.mp3` o `.ogg`.

---

## 🚀 Instalación y Ejecución

Siéntete libre de abrir este entorno en tu IDE preferido (como WebStorm para el frontend o PyCharm/VSCode para el backend).

### 1. Levantar el Backend (FastAPI + Frontend Unificado)
Tanto la API como el Frontend renovado se corren unificadamente desde el puerto seguro 8001 para que jamás interfiera con configuraciones pasadas.

Abre la terminal de tu entorno, activa tu entorno virtual y ejecuta el servidor de Python:
```bash
.venv\Scripts\python.exe -m uvicorn backend.server:app --reload --port 8001
```

### 2. Acceder al Panel de Control
Abre tu navegador web de preferencia y visita:
👉 **[http://127.0.0.1:8001](http://127.0.0.1:8001)**

---

## 📖 Guía Rápida de Uso

1. **Subir Archivos:** Arrastra tus archivos de audio en el panel de la izquierda.
2. **Ajustar Parámetros de Limpieza:** 
   - Modifica el medidor de reducción de ruido si el audio limpio suena demasiado metálico.
   - Activa/desactiva los interruptores de los filtros según lo requieras.
3. **Analizar:** Haz clic en **"⚡ INICIAR ANÁLISIS"**.
4. **Revisar Resultados:** En el panel central oscuro (flotante) verás los audios con gráficas devolviéndose en tarjetas a mano derecha y su clasificación de Silbato o Ruido.
5. **Entrenar el Modelo (Feedback):** Usa los botones azules y rojos individuales en cada tarjeta. El sistema empezará a aprender a partir de las 5 muestras que decidas corregir o verificar.

---

## ⚙️ Configuración del Algoritmo Céntrico

Si deseas modificar los valores base iniciales (antes del entrenamiento adaptativo del usuario), puedes editar el diccionario `CFG_DEFAULTS` directamente en **`backend/core.py`**:

- `whistle_freq_min`: Frecuencia mínima del silbato (2000 Hz)
- `whistle_freq_max`: Frecuencia máxima del silbato (8000 Hz)
- `energy_ratio_threshold`: Umbral de energía base (0.25)
- `peak_prominence_db`: Prominencia mínima para detección de picos (12.0 dB)
- `min_duration_s`: Duración mínima sostenida (0.08 s)

---

## 📘 Más Información
Revisa el archivo **[arquitectura.md](arquitectura.md)** (incluido en este repositorio) para entender cómo está estructurada internamente la separación moderna de las carpetas Frontend y Backend.
