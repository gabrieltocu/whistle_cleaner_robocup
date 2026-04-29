from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import tempfile
from backend import core

app = FastAPI(title="NAO Audio Cleaner API")

# Habilitar CORS para que el frontend pueda hacer peticiones sin problemas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite a LiveServer o WebStorm conectar desde otro puerto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeedbackRequest(BaseModel):
    filename: str
    predicted: str
    corrected: str
    features: dict

@app.post("/api/process")
async def process_audio(
    files: list[UploadFile] = File(...),
    noise_strength: float = Form(0.75),
    apply_bp: bool = Form(True),
    do_normalize: bool = Form(True)
):
    results = []
    
    config = {
        "noise_strength": noise_strength,
        "apply_bp": apply_bp,
        "do_normalize": do_normalize
    }
    
    for file in files:
        contents = await file.read()
        
        # Save temp original to process
        temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_in.write(contents)
        temp_in.close()
        
        fname = file.filename
        name_clean = os.path.splitext(fname)[0] + "_limpio.wav"
        
        res = core.process_file(temp_in.name, name_clean, config)
        
        os.remove(temp_in.name)
        
        if res["success"]:
            results.append({
                "filename": fname,
                "status": "OK",
                "classification": res["etiqueta"],
                "energia_max": round(res["energia_max"], 3),
                "duracion_s": round(res["duracion_s"], 2),
                "pico_espectral": res["pico_espectral"],
                "features": res["features"],
                "clean_url": res["clean_url"]
            })
        else:
            results.append({
                "filename": fname,
                "status": "ERROR",
                "error": res["error"]
            })
            
    return JSONResponse({"results": results, "stats": core.get_stats()})


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    _, adapted = core.push_feedback(req.filename, req.predicted, req.corrected, req.features)
    
    return JSONResponse({
        "success": True,
        "adapted": adapted,
        "stats": core.get_stats()
    })

@app.post("/api/reset")
def reset_feedback():
    if os.path.exists(core.FEEDBACK_FILE):
        os.remove(core.FEEDBACK_FILE)
    core.CFG["energy_ratio_threshold"] = core.CFG_DEFAULTS["energy_ratio_threshold"]
    core.CFG["min_duration_s"] = core.CFG_DEFAULTS["min_duration_s"]
    
    return JSONResponse({"success": True, "stats": core.get_stats()})

@app.get("/api/stats")
def get_stats():
    return core.get_stats()

@app.get("/api/audios/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(core.TMP_OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio no encontrado")

# Montar el frontend para que se sirva desde la raíz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(BASE_DIR, "frontend")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))

app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
