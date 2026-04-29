# 🏗️ Arquitectura del Sistema: Cliente-Servidor (Pro Version)

Dejando de lado el patrón monolítico original con Gradio, el proyecto interactúa ahora mediante un patrón **Cliente-Servidor (API REST)**. Esta separación radical garantiza:

1. Modificabilidad visual del entorno sin interferir con la matemática pesada (Librosa/NumPy).
2. Estabilidad ampliada para futuros equipos o desarrolladores robocups.
3. Posibilidad de utilizar editores y entornos de desarrollo front-end puros (como WebStorm).

El árbol base del proyecto es el siguiente:

```text
/whistle_cleaner_robocup
│
├── frontend/                   # Interfaz de Usuario (Cliente)
│   ├── index.html              # Estructura del Pro Dashboard visual
│   ├── styles.css              # Patrones estéticos modernos y animaciones fluidas
│   └── app.js                  # Lógica para interactuar asíncronamente con la API (.fetch)
│
├── backend/                    # Cerebro de Procesamiento (Servidor)
│   ├── core.py                 # Algoritmos numéricos; cálculos espectrales y aprendizaje profundo.
│   └── server.py               # Eje central de FastAPI. Abre puertos, rutas y une Frontend con Core.
│
├── audio_cleaner_app.py        # 🗑️ Legado del monolito en Gradio (seguro de eliminar)
├── README.md                   # Instrucciones de setup y quickstart
├── arquitectura.md             # Esta documentación
└── requirements.txt            # Dependencias actualizadas del ecosistema Python
```

---

## 🗃️ Detalle por Componente

### 1. El Backend (`backend/`)
Es el motor asíncrono y enrutador maestro. 
- **`core.py`**: No tiene idea de que existe una interfaz de usuario o una API web. Solo recibe diccionarios de configuración y bytes (archivos temporales), procesa el ruido a nivel numérico matricial, aplica algoritmos limitantes de paso-banda, y ejecuta procesos de evaluación por rangos (`whistle_freq_min - whistle_freq_max`).
- **`server.py`**: Es el envoltorio (wrapper) FastAPI. Instancia CORS (para seguridad), registra múltiples Endpoints (`/api/process`, `/api/feedback`) y, por último, su magia monta el directorio estático del frontend en la raíz (`/`) para servirse unificadamente.

### 2. El Frontend (`frontend/`)
Pensado a bajo nivel y con alta optimización mediante Vainilla JS. 
Libre de *Node.js* o *React* para su ejecución veloz en cualquier panel táctil o computadora.
- **Diseño**: Basado enteramente en un patrón "Glassmorphism" con un CSS modular para micro-animaciones (barras deslizables, cargas dinámicas de spinner).
- **Consumo API**: Emplea Promesas y asincronía purista (`await fetch`) en `app.js` enviando `FormDatas` transparentes en lugar de bloquear el lazo interactivo y dando agilidad durante el proceso profundo de FastAPI.

### 🔄 Flujo de Datos

1. El usuario sube `X` cantidad de `.wav` al Frontend vía Drag and Drop.
2. `app.js` encapsula los archivos como multipart y golpea `POST /api/process`.
3. `server.py` delega archivo por archivo hacia `core.py` mediante guardado en archivos temporales `tempfile`.
4. El procesamiento matemático culmina y responde con JSON detallado (estadísticas, umbral).
5. Frontend intercepta la respuesta inyectando al arbol (DOM) reproductores nativos `HTML5 <audio>` e iterando estadísticas estéticas por cada elemento sin recargar jamás la página.
