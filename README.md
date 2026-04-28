# 🤖 NAO Audio Cleaner & Whistle Detector — RoboCup HSL

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)
![RoboCup](https://img.shields.io/badge/RoboCup-Humanoid_Soccer_League-success.svg)

Una herramienta de interfaz web interactiva construida para limpiar grabaciones provenientes de los micrófonos de los robots NAO v6 y detectar automáticamente los silbatos del árbitro (Game Controller). Diseñada específicamente para lidiar con el alto ruido de los motores del NAO y la acústica compleja de los entornos de competencia.

## ✨ Características Principales

* **Limpieza de Audio en Batch:** Aplica algoritmos avanzados de sustracción espectral (reducción de ruido), filtros paso-banda (100 Hz - 10 kHz) y normalización de volumen.
* **Detección Espectral de Silbatos:** Analiza la relación de energía en la banda de frecuencia del silbato (2 kHz - 8 kHz) y busca picos prominentes (Prominence Peaks) en la transformada de Fourier.
* **Aprendizaje Adaptativo (Feedback Loop):** Incorpora un sistema "Human-in-the-Loop". El etiquetado manual optimiza automáticamente los parámetros del algoritmo ejecutando un Grid Search en segundo plano para maximizar el F1-score.
* **Re-análisis en Caliente:** Permite volver a evaluar los audios de la sesión actual inmediatamente después de que el sistema ajusta sus umbrales.
* **Identificación Trazable:** Cada entrada de feedback genera un `id` secuencial base 0 para exportar a futuros análisis.

---

## ⚙️ Requisitos Previos

* **Python 3.8 o superior.**

---

## 🛠️ Instalación y Configuración del Entorno

Para garantizar que la aplicación funcione correctamente y no haya conflictos con otras librerías instaladas en tu sistema (algo muy común al trabajar con procesamiento de audio y machine learning), es altamente recomendable utilizar un entorno virtual.

### 1. Clonar el repositorio
Primero, descarga los archivos del proyecto y entra a la carpeta:
```bash
git clone <URL_DE_TU_REPOSITORIO>
cd <NOMBRE_DE_LA_CARPETA>



2. Crear un entorno virtual (recomendado)
Un entorno virtual aísla las dependencias de este proyecto, evitando conflictos con otras instalaciones de Python.

# Crea el entorno virtual (se creará una carpeta .venv)
python -m venv .venv

# Activa el entorno virtual
# En Windows
.venv\Scripts\activate

# En macOS o Linux
source .venv/bin/activate

3. Instalar dependencias
Con el entorno activado, instala todas las librerías necesarias usando pip:

pip install gradio librosa numpy scipy noisereduce soundfile pandas

---

## 🚀 Ejecución de la Aplicación

Una vez completada la instalación, puedes iniciar la aplicación web con un simple comando:

python audio_cleaner_app.py

El programa iniciará un servidor local y mostrará una URL en la terminal (generalmente http://localhost:7860).

Abre esa dirección en tu navegador web para comenzar a usar el limpiador y detector.

---

## 🎯 Uso Básico

1. **Carga de Audios:**
   - Arrastra y suelta la carpeta con tus archivos `.wav` o usa el botón de carga.
   - Selecciona el "Número de fila" para navegar por los archivos.
2. **Limpieza:**
   - Ajusta los parámetros de reducción de ruido y filtros a tu gusto.
   - Pulsa "Procesar" para ver los resultados.
3. **Corrección y Feedback:**
   - Revisa la clasificación (SILBATO o SOLO RUIDO).
   - Si la clasificación es errónea, pulsa los botones de corrección.
   - El sistema aprenderá de tus correcciones para ajustar automáticamente los umbrales futuros.

---

## 📂 Estructura del Proyecto

```
whistle_detector/
├── audio_cleaner_app.py     # 👈 Aplicación principal Gradio (interfaz web).
├── whistle_detector.py      # Script auxiliar para análisis por línea de comandos.
├── audio_recordThree.py     # Script para grabación en el robot NAO.
├── feedback_data.json       # 💾 Base de datos de feedback y umbrales aprendidos.
├── dataWhistles/            # 📁 Carpeta contenedora de los audios de prueba.
└── ... (otros archivos auxiliares de la aplicación)
```

---

## 🧪 Pruebas de Integración y Tests

Para verificar que el sistema funciona correctamente después de la instalación, puedes ejecutar un test automatizado que valida el pipeline completo de procesamiento y detección.

### Comando de Prueba

Ejecuta el siguiente comando para iniciar la suite de tests:

```bash
python whistle_detector.py --test
```

### ¿Qué Verifica Este Test?

El script realizará las siguientes validaciones internas:

1.  **Módulo de Reducción de Ruido:**
    *   Verifica que la función `reduce_noise` pueda procesar un array NumPy sin generar errores.
2.  **Módulo de Detección de Silbatos:**
    *   **Construcción Espectral:** Confirma que el STFT (Short-Time Fourier Transform) se calcula correctamente.
    *   **Envoltura Espectral:** Verifica la capacidad de crear el ratio de energía en la banda del silbato.
    *   **Detección de Picos:** Valida que la función `find_peaks` localiza armónicos (picos) en el espectro.
3.  **Módulo de Feedback y Adaptación:**
    *   **Ajuste de Umbrales:** Ejecuta el algoritmo de Grid Search para simular el aprendizaje adaptativo y valida que devuelve valores numéricos coherentes.
4.  **Validación Global:**
    *   Ejecuta el análisis completo con parámetros de prueba estándar y verifica que retorna una clasificación válida.

Si todos los tests pasan, verás un mensaje de éxito como este:

```
✅ All tests passed successfully!
```

---

## 🎛️ Parámetros de Configuración

La aplicación permite ajustar varios parámetros para optimizar la detección de silbatos. Estos se encuentran definidos en el bloque `CFG` del archivo principal.

| Parámetro | Descripción | Valores Sugeridos |
|-----------|-------------|-------------------|
| `whistle_freq_min` | Frecuencia mínima a considerar como silbato (Hz). | `2000` (2 kHz) |
| `whistle_freq_max` | Frecuencia máxima a considerar como silbato (Hz). | `8000` (8 kHz) |
| `energy_ratio_threshold` | Umbral de energía relativa del silbato. | 0.03 - 0.70 |
| `peak_prominence_db` | Prominencia mínima de los picos espectrales (dB). | 12.0 - 20.0 |
| `min_duration_s` | Duración mínima del silbato detectado (segundos). | 0.05 - 0.2 |
| `noise_reduction_strength` | Fuerza de la reducción de ruido (0.0 a 1.0). | 0.5 - 0.8 |
| `use_band_pass_filter` | Activa/desactiva el filtro paso-banda. | `True` / `False` |
| `normalize_audio` | Normaliza el volumen antes del análisis. | `True` / `False` |

---

## 🤝 Colaboración y Contribuciones

Si planeas contribuir a este proyecto o integrar tus propios algoritmos, considera seguir estas pautas:

1.  **Código Limpio:** Mantén una estructura modular, separando el preprocesamiento, la detección y la lógica de aprendizaje.
2.  **Documentación:** Añade siempre docstrings claros a las nuevas funciones.
3.  **Nuevos Algoritmos:**
    *   Si implementas un nuevo método de detección, asegúrate de que sea compatible con la estructura de datos de `feedback_data.json`.
    *   Considera usar `metrics.py` para añadir nuevas métricas de evaluación.

---

## 📝 Licencia

Este proyecto es de código abierto y está disponible para uso académico y de investigación.