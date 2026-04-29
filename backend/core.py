import os, json, tempfile
from datetime import datetime

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import find_peaks, butter, sosfilt

# ────────────────────────────────────────────────────────────
#  CONSTANTES
# ────────────────────────────────────────────────────────────
MAX_PLAYERS   = 20
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_data.json")
TMP_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_audios")

os.makedirs(TMP_OUTPUT_DIR, exist_ok=True)

# Umbrales base (se sobreescriben con los valores adaptados al arrancar)
CFG = {
    "whistle_freq_min":       2000,
    "whistle_freq_max":       8000,
    "energy_ratio_threshold": 0.25,
    "peak_prominence_db":     12.0,
    "min_duration_s":         0.08,
    "sr":                     44100,
    "n_fft":                  2048,
    "hop_length":             512,
}
CFG_DEFAULTS = {k: v for k, v in CFG.items()}   # copia de valores originales

def _bootstrap_cfg():
    """Aplica umbrales guardados desde feedback anterior al arrancar."""
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                fb = json.load(f)
            for k in ("energy_ratio_threshold", "min_duration_s", "peak_prominence_db"):
                if k in fb.get("adjusted_cfg", {}):
                    CFG[k] = fb["adjusted_cfg"][k]
    except Exception:
        pass

_bootstrap_cfg()

# ────────────────────────────────────────────────────────────
#  PROCESAMIENTO DE AUDIO
# ────────────────────────────────────────────────────────────

def reduce_noise(y, sr, strength=0.75):
    noise_clip = y[: max(int(sr * 0.3), 1)]
    return nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip,
                           prop_decrease=strength, stationary=False)

def band_pass_filter(y, sr, low=100.0, high=10000.0):
    nyq = sr / 2
    sos = butter(4, [max(low / nyq, 0.001), min(high / nyq, 0.999)],
                 btype="band", output="sos")
    return sosfilt(sos, y)

def normalize_audio(y):
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def detect_whistle(y, sr):
    D     = librosa.stft(y, n_fft=CFG["n_fft"], hop_length=CFG["hop_length"])
    S_db  = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=CFG["n_fft"])

    mask    = (freqs >= CFG["whistle_freq_min"]) & (freqs <= CFG["whistle_freq_max"])
    S_pwr   = librosa.db_to_power(S_db)
    ratio   = S_pwr[mask, :].sum(0) / (S_pwr.sum(0) + 1e-9)
    fps     = sr / CFG["hop_length"]
    dur_abv = (ratio >= CFG["energy_ratio_threshold"]).sum() / fps

    mean_sp  = S_db.mean(1)
    band_sp  = mean_sp[mask]
    median   = np.median(mean_sp)
    peaks, _ = find_peaks(band_sp, prominence=CFG["peak_prominence_db"])
    peak_ok  = len(peaks) > 0 and np.any(
                   band_sp[peaks] > median + CFG["peak_prominence_db"])

    has = dur_abv >= CFG["min_duration_s"] and peak_ok
    return {
        "etiqueta":       "🟢 SILBATO" if has else "🔴 SOLO RUIDO",
        "has_whistle":    bool(has),
        "energia_max":    float(ratio.max()),
        "duracion_s":     float(dur_abv),
        "pico_espectral": bool(peak_ok),
    }

# ────────────────────────────────────────────────────────────
#  SISTEMA DE FEEDBACK Y APRENDIZAJE ADAPTATIVO
# ────────────────────────────────────────────────────────────

def load_fb():
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"entries": [], "adjusted_cfg": {}}

def save_fb(data):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def compute_thresholds(entries):
    """
    Grid-search para encontrar los umbrales (energy_ratio_threshold, min_duration_s)
    que maximizan el F1-score sobre los datos etiquetados por el usuario.
    """
    labeled = [e for e in entries if "corrected" in e]
    if len(labeled) < 5:
        return None

    X_e = np.array([e["features"]["energia_max"] for e in labeled])
    X_d = np.array([e["features"]["duracion_s"]  for e in labeled])
    y   = np.array([1 if "SILBATO" in e["corrected"] else 0 for e in labeled])

    if len(np.unique(y)) < 2:
        return None   # necesitamos ambas clases

    best_f1 = -1.0
    best_et = CFG["energy_ratio_threshold"]
    best_dt = CFG["min_duration_s"]

    for et in np.linspace(0.03, 0.70, 60):
        for dt in np.linspace(0.02, 0.40, 25):
            pred = ((X_e >= et) & (X_d >= dt)).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            if tp + fp == 0 or tp + fn == 0:
                continue
            p  = tp / (tp + fp)
            r  = tp / (tp + fn)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1, best_et, best_dt = f1, float(et), float(dt)

    return {
        "energy_ratio_threshold": round(best_et, 4),
        "min_duration_s":         round(best_dt, 4),
        "_f1":                    round(best_f1, 4),
        "_n":                     len(labeled),
    }

def push_feedback(filename, predicted, corrected, features):
    """Guarda una entrada de feedback, asignando un ID basado en índice, y readapta los umbrales."""
    fb = load_fb()

    new_entry = {
        "id": 0, # Se asignará abajo
        "timestamp":  datetime.now().isoformat(),
        "filename":   filename,
        "predicted":  predicted,
        "corrected":  corrected,
        "features":   features,
        "correction": predicted != corrected,
    }

    for i, e in enumerate(fb["entries"]):
        if e.get("filename") == filename:
            new_entry["id"] = e.get("id", i)
            fb["entries"][i] = new_entry
            break
    else:
        new_entry["id"] = len(fb["entries"])
        fb["entries"].append(new_entry)

    # Adaptar umbrales
    new_cfg = compute_thresholds(fb["entries"])
    adapted = False
    if new_cfg:
        old_e = CFG["energy_ratio_threshold"]
        old_d = CFG["min_duration_s"]
        fb["adjusted_cfg"] = new_cfg
        CFG["energy_ratio_threshold"] = new_cfg["energy_ratio_threshold"]
        CFG["min_duration_s"]         = new_cfg["min_duration_s"]
        adapted = (abs(CFG["energy_ratio_threshold"] - old_e) > 0.0005 or
                   abs(CFG["min_duration_s"] - old_d) > 0.0005)

    save_fb(fb)
    return fb, adapted

def get_stats():
    fb  = load_fb()
    ent = fb.get("entries", [])
    n   = len(ent)
    nc  = sum(1 for e in ent if e.get("correction"))
    nfn = sum(1 for e in ent if e.get("correction") and "SILBATO"    in e.get("corrected", ""))
    nfp = sum(1 for e in ent if e.get("correction") and "SOLO RUIDO" in e.get("corrected", ""))
    adj = fb.get("adjusted_cfg", {})
    
    return {
        "samples_count": n,
        "corrections_count": nc,
        "false_negatives": nfn,
        "false_positives": nfp,
        "accuracy_pct": round((n - nc) / n * 100, 1) if n > 0 else 0,
        "f1_score": round(adj.get("_f1", 0) * 100, 1) if "_f1" in adj else None,
        "thresholds": {
            "energy_original": CFG_DEFAULTS["energy_ratio_threshold"],
            "energy_current": CFG["energy_ratio_threshold"],
            "duration_original": CFG_DEFAULTS["min_duration_s"],
            "duration_current": CFG["min_duration_s"]
        },
        "needs_more_data": n < 5
    }

def process_file(filepath, out_filename, config):
    try:
        y, sr = librosa.load(filepath, sr=CFG["sr"], mono=True)
        
        y_c = y.copy()
        if config["noise_strength"] > 0:
            y_c = reduce_noise(y_c, sr, strength=config["noise_strength"])
        if config["apply_bp"]:
            y_c = band_pass_filter(y_c, sr)
        if config["do_normalize"]:
            y_c = normalize_audio(y_c)
            
        det = detect_whistle(y_c, sr)
        
        out_path = os.path.join(TMP_OUTPUT_DIR, out_filename)
        sf.write(out_path, y_c, sr, subtype="PCM_16")
        
        return {
            "success": True,
            "etiqueta": det["etiqueta"],
            "energia_max": det["energia_max"],
            "duracion_s": det["duracion_s"],
            "pico_espectral": det["pico_espectral"],
            "features": det,
            "clean_url": f"/api/audios/{out_filename}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
