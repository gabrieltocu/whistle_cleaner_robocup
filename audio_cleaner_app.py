"""
audio_cleaner_app.py  ·  v3  (con feedback adaptativo, IDs y re-análisis)
=========================================================================
Interfaz web — Limpieza y detección de silbato para RoboCup HSL / NAO v6

Incluye:
  - Reproductor por fila en la sección de análisis
  - Tarjeta de clasificación con barras de progreso vs umbral
  - Botones de feedback: confirmar / marcar falso negativo / marcar falso positivo
  - Aprendizaje adaptativo: grid-search de umbrales óptimos sobre el feedback acumulado
  - Botón de re-análisis en caliente con nuevos umbrales
  - Panel de estadísticas con métricas y estado de adaptación
"""

import os, json, zipfile, tempfile, warnings
from datetime import datetime

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import find_peaks, butter, sosfilt
import gradio as gr
import pandas as pd

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
#  CONSTANTES
# ────────────────────────────────────────────────────────────
MAX_PLAYERS   = 20
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_data.json")

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

def to_int16(y):
    return (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)

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

    # Sobreescribir si ya existe una entrada para este archivo, manteniendo su ID original
    for i, e in enumerate(fb["entries"]):
        if e.get("filename") == filename:
            new_entry["id"] = e.get("id", i)
            fb["entries"][i] = new_entry
            break
    else:
        # Si es nuevo, el ID es la longitud actual de la lista (índice 0-based)
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

# ────────────────────────────────────────────────────────────
#  HTML HELPERS
# ────────────────────────────────────────────────────────────

def make_zip(paths, label):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip",
                                      prefix=f"nao_{label}_")
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))
    return tmp.name

def badge(text, color="#00e5ff"):
    return (f'<div style="display:inline-block;background:#0b1525;'
            f'border:1.5px solid {color};border-radius:20px;'
            f'padding:5px 18px;font-family:Space Mono,monospace;'
            f'font-size:0.78rem;color:{color};margin:4px 0 10px">{text}</div>')

def sec_title(icon, title):
    return (f'<div style="font-family:Space Mono,monospace;font-size:0.68rem;'
            f'text-transform:uppercase;letter-spacing:3px;color:#4a6070;'
            f'margin:22px 0 6px">{icon}&nbsp; {title}</div>')

def _bar(val, threshold):
    pct    = min(int(val / max(threshold * 2, 0.001) * 100), 100)
    color  = "#00e676" if val >= threshold else "#ff5252"
    return (f'<div style="background:#060e1c;border-radius:3px;height:5px;margin-top:3px">'
            f'<div style="background:{color};width:{pct}%;height:100%;border-radius:3px;'
            f'transition:width .3s ease"></div></div>')

def clf_info_html(fname, state):
    if not fname or not state or fname not in state.get("features", {}):
        return ('<div style="color:#4a6070;font-size:0.85rem;padding:10px 0">'
                'Selecciona un audio para ver su clasificación.</div>')

    feat = state["features"][fname]
    clf  = state["classifications"].get(fname, "—")
    is_w = "SILBATO" in clf
    col  = "#00e676" if is_w else "#ff5252"
    peak = "✅ detectado" if feat.get("pico_espectral") else "❌ no detectado"
    et   = CFG["energy_ratio_threshold"]
    dt   = CFG["min_duration_s"]

    return f"""
<div style="background:#060e1c;border:1px solid #192840;border-radius:10px;
            padding:14px 18px;font-family:DM Sans,sans-serif">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
    <span style="font-size:1.6rem">{'🟢' if is_w else '🔴'}</span>
    <div>
      <div style="color:{col};font-family:Space Mono,monospace;
                  font-size:0.9rem;font-weight:700;letter-spacing:0.5px">
        {'SILBATO DETECTADO' if is_w else 'SOLO RUIDO'}
      </div>
      <div style="color:#2a3a4a;font-family:Space Mono,monospace;font-size:0.7rem;
                  margin-top:2px;max-width:340px;overflow:hidden;
                  text-overflow:ellipsis;white-space:nowrap">{fname}</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">
    <div style="background:#0b1525;border-radius:8px;padding:9px 12px">
      <div style="color:#4a6070;font-size:0.65rem;text-transform:uppercase;
                  letter-spacing:1px">Energía máx</div>
      <div style="color:#00e5ff;font-family:Space Mono,monospace;
                  font-size:1.05rem;font-weight:700;margin-top:2px">
        {feat['energia_max']:.3f}</div>
      {_bar(feat['energia_max'], et)}
      <div style="color:#2a3a4a;font-size:0.62rem;margin-top:3px">
        umbral activo: {et:.3f}</div>
    </div>
    <div style="background:#0b1525;border-radius:8px;padding:9px 12px">
      <div style="color:#4a6070;font-size:0.65rem;text-transform:uppercase;
                  letter-spacing:1px">Duración</div>
      <div style="color:#00e5ff;font-family:Space Mono,monospace;
                  font-size:1.05rem;font-weight:700;margin-top:2px">
        {feat['duracion_s']:.2f}s</div>
      {_bar(feat['duracion_s'], dt)}
      <div style="color:#2a3a4a;font-size:0.62rem;margin-top:3px">
        umbral activo: {dt:.3f}s</div>
    </div>
    <div style="background:#0b1525;border-radius:8px;padding:9px 12px">
      <div style="color:#4a6070;font-size:0.65rem;text-transform:uppercase;
                  letter-spacing:1px">Pico espectral</div>
      <div style="color:#{'00e676' if feat.get('pico_espectral') else 'ff5252'};
                  font-size:0.85rem;font-weight:600;margin-top:4px">{peak}</div>
    </div>
  </div>
</div>
"""

def stats_html():
    fb  = load_fb()
    ent = fb.get("entries", [])
    n   = len(ent)
    nc  = sum(1 for e in ent if e.get("correction"))
    nfn = sum(1 for e in ent if e.get("correction") and "SILBATO"    in e.get("corrected", ""))
    nfp = sum(1 for e in ent if e.get("correction") and "SOLO RUIDO" in e.get("corrected", ""))
    adj = fb.get("adjusted_cfg", {})
    f1s = f"{adj['_f1']*100:.1f}%" if "_f1" in adj else "—"
    acc = f"{(n - nc) / n * 100:.1f}%" if n > 0 else "—"

    et_orig = CFG_DEFAULTS["energy_ratio_threshold"]
    dt_orig = CFG_DEFAULTS["min_duration_s"]
    et_now  = CFG["energy_ratio_threshold"]
    dt_now  = CFG["min_duration_s"]
    et_ch   = abs(et_now - et_orig) > 0.005
    dt_ch   = abs(dt_now - dt_orig) > 0.005
    adapted = et_ch or dt_ch

    def th_chip(label, orig, curr, unit=""):
        changed = abs(curr - orig) > 0.001
        arrow   = " ↓" if curr < orig else (" ↑" if curr > orig else "")
        bd_col  = "#ffd600" if changed else "#192840"
        tx_col  = "#ffd600" if changed else "#00e5ff"
        return (f'<span style="display:inline-block;background:#060e1c;'
                f'border:1px solid {bd_col};border-radius:6px;'
                f'padding:4px 10px;margin:3px;font-size:0.72rem">'
                f'<span style="color:#4a6070">{label} </span>'
                f'<span style="color:{tx_col};font-family:Space Mono,monospace">'
                f'{curr:.3f}{unit}{arrow}</span></span>')

    need_more = n < 5
    acc_color = "#00e676" if n > 0 and nc / n < 0.2 else (
                "#ffd600" if n > 0 and nc / n < 0.4 else "#ff5252")

    return f"""
<div style="background:#0b1525;border:1px solid #192840;border-radius:12px;
            padding:16px 20px;font-family:DM Sans,sans-serif">
  <div style="font-family:Space Mono,monospace;font-size:0.65rem;
              text-transform:uppercase;letter-spacing:3px;color:#4a6070;
              margin-bottom:12px">🧠&nbsp; Aprendizaje Adaptativo</div>

  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px">
    <div style="flex:1;min-width:70px;background:#060e1c;border-radius:8px;
                padding:10px 12px;text-align:center">
      <div style="font-size:1.5rem;font-weight:700;color:#00e5ff;
                  font-family:Space Mono,monospace">{n}</div>
      <div style="font-size:0.65rem;color:#4a6070">muestras</div>
    </div>
    <div style="flex:1;min-width:70px;background:#060e1c;border-radius:8px;
                padding:10px 12px;text-align:center">
      <div style="font-size:1.5rem;font-weight:700;font-family:Space Mono,monospace;
                  color:{'#ff5252' if nc > 0 else '#00e676'}">{nc}</div>
      <div style="font-size:0.65rem;color:#4a6070">correcciones</div>
    </div>
    <div style="flex:1;min-width:70px;background:#060e1c;border-radius:8px;
                padding:10px 12px;text-align:center">
      <div style="font-size:1.5rem;font-weight:700;color:{acc_color};
                  font-family:Space Mono,monospace">{acc}</div>
      <div style="font-size:0.65rem;color:#4a6070">precisión obs.</div>
    </div>
  </div>

  {f'<div style="font-size:0.72rem;color:#00e5ff;margin-bottom:4px">F1 (feedback): <span style="font-family:Space Mono,monospace">{f1s}</span></div>' if f1s != "—" else ""}
  {f'<div style="font-size:0.72rem;color:#ff8f00;margin-bottom:6px">FN: {nfn} &nbsp;·&nbsp; FP: {nfp}</div>' if nc > 0 else ""}

  <div style="font-size:0.72rem;color:#4a6070;margin-bottom:5px">Umbrales activos:</div>
  <div style="margin-bottom:8px">
    {th_chip("Energía", et_orig, et_now)}
    {th_chip("Duración", dt_orig, dt_now, "s")}
  </div>

  {('<div style="background:rgba(255,214,0,0.08);border:1px solid #ffd600;border-radius:6px;'
     'padding:6px 10px;font-size:0.72rem;color:#ffd600">'
     '⚡ Umbrales adaptados automáticamente desde el feedback acumulado</div>')
    if adapted else ""}
  {('<div style="font-size:0.72rem;color:#4a6070;font-style:italic;margin-top:6px">'
     f'Necesitas al menos 5 muestras con ambas clases · tienes {n} hasta ahora</div>')
    if need_more else ""}
</div>
"""

# ────────────────────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ────────────────────────────────────────────────────────────

def process_audio(files, noise_strength, apply_bp, do_normalize):
    def empties():
        return [gr.update(value=None, visible=False, label="")] * MAX_PLAYERS

    EMPTY_STATE = {"filenames": [], "audios": {}, "features": {}, "classifications": {}}

    if not files:
        return (
            *empties(), *empties(),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            pd.DataFrame(),
            "_Sube archivos y haz clic en **Limpiar y analizar**._",
            badge("0 archivos subidos"),
            badge("0 archivos listos", color="#4a6070"),
            EMPTY_STATE,
            gr.update(choices=[], value=None, visible=False),
            stats_html(),
            gr.update(value=None, visible=False),
            clf_info_html(None, None),
        )

    tmp_dir      = tempfile.mkdtemp()
    rows         = []
    orig_paths   = []
    clean_paths  = []
    orig_arrays  = []
    clean_arrays = []
    feat_list    = []

    for fp in files:
        fname      = os.path.basename(fp)
        name_clean = os.path.splitext(fname)[0] + "_limpio.wav"
        try:
            y, sr = librosa.load(fp, sr=CFG["sr"], mono=True)
            orig_paths.append(fp)
            orig_arrays.append((sr, y))

            y_c = y.copy()
            if noise_strength > 0:
                y_c = reduce_noise(y_c, sr, strength=noise_strength)
            if apply_bp:
                y_c = band_pass_filter(y_c, sr)
            if do_normalize:
                y_c = normalize_audio(y_c)

            det = detect_whistle(y_c, sr)
            clean_arrays.append((sr, y_c))
            feat_list.append(det)

            out = os.path.join(tmp_dir, name_clean)
            sf.write(out, y_c, sr, subtype="PCM_16")
            clean_paths.append(out)

            rows.append({
                "Archivo":          fname,
                "Clasificación":    det["etiqueta"],
                "Energía máx":      f"{det['energia_max']:.3f}",
                "Dur. silbato (s)": f"{det['duracion_s']:.2f}",
                "Pico espectral":   "✅" if det["pico_espectral"] else "❌",
                "Estado":           "✅ OK",
            })
        except Exception as e:
            feat_list.append(None)
            rows.append({
                "Archivo":          fname,
                "Clasificación":    "—",
                "Energía máx":      "—",
                "Dur. silbato (s)": "—",
                "Pico espectral":   "—",
                "Estado":           f"❌ {str(e)[:55]}",
            })

    df         = pd.DataFrame(rows)
    n_ok       = len(clean_paths)
    n_silbato  = int(df["Clasificación"].str.contains("SILBATO", na=False).sum())
    n_ruido    = int(df["Clasificación"].str.contains("RUIDO",   na=False).sum())

    summary = (f"✅ Procesados: **{n_ok}** &nbsp;&nbsp;|&nbsp;&nbsp; "
               f"🟢 Con silbato: **{n_silbato}** &nbsp;&nbsp;|&nbsp;&nbsp; "
               f"🔴 Solo ruido: **{n_ruido}**")

    # ── Estado de sesión ──────────────────────────────────────
    state = {"filenames": [], "audios": {}, "features": {}, "classifications": {}}
    for i, row in enumerate(rows):
        fname = row["Archivo"]
        state["filenames"].append(fname)
        state["classifications"][fname] = row["Clasificación"]
        if i < len(clean_arrays) and feat_list[i] is not None:
            sr_i, y_i = clean_arrays[i]
            state["audios"][fname] = (sr_i, to_int16(y_i))
            state["features"][fname] = {
                "energia_max":    feat_list[i]["energia_max"],
                "duracion_s":     feat_list[i]["duracion_s"],
                "pico_espectral": feat_list[i]["pico_espectral"],
            }

    # ── Reproductores de audio ────────────────────────────────
    def build_updates(arrays):
        upds = []
        for i in range(MAX_PLAYERS):
            if i < len(arrays):
                sr_i, y_i = arrays[i]
                lbl = rows[i]["Archivo"] if i < len(rows) else f"audio_{i+1}"
                upds.append(gr.update(value=(sr_i, to_int16(y_i)),
                                      visible=True, label=lbl))
            else:
                upds.append(gr.update(value=None, visible=False, label=""))
        return upds

    orig_upds  = build_updates(orig_arrays)
    clean_upds = build_updates(clean_arrays)

    # ── ZIP ───────────────────────────────────────────────────
    n_o, n_c = len(orig_paths), len(clean_paths)
    zip_o = gr.update(
        value=make_zip(orig_paths, "originales") if orig_paths else None,
        visible=bool(orig_paths),
        label=f"📦  Descargar {n_o} original{'es' if n_o != 1 else ''} (.zip)",
    )
    zip_c = gr.update(
        value=make_zip(clean_paths, "limpios") if clean_paths else None,
        visible=bool(clean_paths),
        label=f"📦  Descargar {n_c} limpio{'s' if n_c != 1 else ''} (.zip)",
    )

    b_orig  = badge(f"{n_o} archivo{'s' if n_o!=1 else ''} · subido{'s' if n_o!=1 else ''}")
    b_clean = badge(
        f"{n_c} archivo{'s' if n_c!=1 else ''} · listo{'s' if n_c!=1 else ''} para descargar",
        color="#00e676",
    )

    # ── Dropdown + preview del primer archivo ─────────────────
    fnames = state["filenames"]
    dd_upd = gr.update(choices=fnames, value=fnames[0] if fnames else None,
                       visible=bool(fnames))

    first_audio = None
    first_info  = clf_info_html(None, None)
    if fnames and fnames[0] in state["audios"]:
        first_audio = gr.update(value=state["audios"][fnames[0]], visible=True)
        first_info  = clf_info_html(fnames[0], state)
    else:
        first_audio = gr.update(value=None, visible=False)

    return (
        *orig_upds, *clean_upds,
        zip_o, zip_c,
        df, summary,
        b_orig, b_clean,
        state, dd_upd, stats_html(),
        first_audio, first_info,
    )

# ────────────────────────────────────────────────────────────
#  MANEJADORES DE FEEDBACK
# ────────────────────────────────────────────────────────────

def on_file_select(fname, state):
    if not fname or not state or "audios" not in state:
        return gr.update(value=None, visible=False), clf_info_html(None, None)
    audio = state["audios"].get(fname)
    info  = clf_info_html(fname, state)
    if audio is None:
        return gr.update(value=None, visible=False), info
    return gr.update(value=audio, visible=True), info

def _status(msg, color):
    return (f'<div style="color:{color};font-size:0.85rem;padding:6px 2px;'
            f'font-family:DM Sans,sans-serif">{msg}</div>')

def on_mark_whistle(fname, state):
    if not fname or not state or "classifications" not in state:
        return _status("⚠️  Selecciona un audio primero.", "#ff5252"), stats_html()
    predicted = state["classifications"].get(fname, "UNKNOWN")
    features  = state["features"].get(fname, {})
    _, adapted = push_feedback(fname, predicted, "🟢 SILBATO", features)

    if "SILBATO" in predicted:
        msg = "✅  Clasificación confirmada (Verdadero Positivo)."
        col = "#00e676"
    else:
        msg = "📝  Falso Negativo registrado — el detector será más sensible."
        col = "#ffd600"
    if adapted:
        msg += " ⚡ Umbrales actualizados."
    return _status(msg, col), stats_html()

def on_mark_noise(fname, state):
    if not fname or not state or "classifications" not in state:
        return _status("⚠️  Selecciona un audio primero.", "#ff5252"), stats_html()
    predicted = state["classifications"].get(fname, "UNKNOWN")
    features  = state["features"].get(fname, {})
    _, adapted = push_feedback(fname, predicted, "🔴 SOLO RUIDO", features)

    if "RUIDO" in predicted:
        msg = "✅  Clasificación confirmada (Verdadero Negativo)."
        col = "#00e676"
    else:
        msg = "📝  Falso Positivo registrado — el detector será más específico."
        col = "#ffd600"
    if adapted:
        msg += " ⚡ Umbrales actualizados."
    return _status(msg, col), stats_html()

def on_reset_feedback():
    if os.path.exists(FEEDBACK_FILE):
        os.remove(FEEDBACK_FILE)
    CFG["energy_ratio_threshold"] = CFG_DEFAULTS["energy_ratio_threshold"]
    CFG["min_duration_s"]         = CFG_DEFAULTS["min_duration_s"]
    return (_status("🗑️  Feedback eliminado. Umbrales restaurados a valores por defecto.",
                    "#ff5252"),
            stats_html())

# ────────────────────────────────────────────────────────────
#  CSS
# ────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --primary : #00e5ff;
    --green   : #00e676;
    --bg      : #060e1c;
    --surface : #0b1525;
    --surface2: #101e30;
    --border  : #192840;
    --text    : #d8e6f8;
    --muted   : #4a6070;
}

body, .gradio-container {
    background : var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color      : var(--text) !important;
}

.nao-header { text-align:center; padding:36px 24px 20px; border-bottom:1px solid var(--border); margin-bottom:24px; }
.nao-header h1 { font-family:'Space Mono',monospace; font-size:2rem; color:var(--primary); letter-spacing:-1px; margin:0 0 6px; }
.nao-header p  { color:var(--muted); font-size:0.9rem; margin:0; }

.gr-group, .gr-box { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:12px !important; }

button.lg { background:linear-gradient(135deg,var(--primary),#0060e0) !important; border:none !important; border-radius:10px !important; font-family:'Space Mono',monospace !important; font-weight:700 !important; color:#04090f !important; font-size:13px !important; padding:14px !important; width:100% !important; transition:transform .15s,box-shadow .15s !important; }
button.lg:hover { transform:translateY(-2px) !important; box-shadow:0 0 28px rgba(0,229,255,.35) !important; }
button.reanalyze-btn { margin-top: 10px !important; border: 1px solid #00e5ff !important; color: #00e5ff !important; background: transparent !important; }
button.reanalyze-btn:hover { background: rgba(0,229,255,0.1) !important; }

.player-wrap { background:var(--surface2); border:1px solid var(--border); border-radius:10px; padding:10px 12px; margin-bottom:8px; }
.player-wrap label { font-size:10px !important; color:var(--muted) !important; font-family:'Space Mono',monospace !important; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
audio { border-radius:8px; width:100%; }

.zip-wrap .file-preview { background:var(--surface2) !important; border:1px dashed var(--green) !important; border-radius:10px !important; }

table { background:var(--surface2) !important; border-radius:10px !important; font-size:0.84rem !important; }
th { background:var(--surface) !important; color:var(--primary) !important; font-family:'Space Mono',monospace !important; font-size:9px !important; text-transform:uppercase !important; letter-spacing:1.5px !important; }
td { color:var(--text) !important; }

input[type=range]    { accent-color:var(--primary) !important; }
input[type=checkbox] { accent-color:var(--primary) !important; }
label  { color:var(--text)  !important; font-size:0.88rem !important; }
.gr-info { color:var(--muted) !important; font-size:0.78rem !important; }

.divider { border:none; border-top:1px solid var(--border); margin:20px 0; }
.summary p { font-size:0.94rem !important; }

/* ── Sección de feedback ── */
.review-wrap { background:var(--surface2) !important; border:1px solid var(--border) !important; border-radius:12px !important; padding:16px 18px !important; }
.review-player audio { border-radius:10px; }

.fb-actions-row { display: flex; gap: 20px !important; margin-top: 10px; justify-content: center; }
.fb-whistle { flex: 1; padding: 14px !important; font-size: 0.95rem !important; background:rgba(255,214,0,0.10)  !important; border:1px solid #ffd600 !important; color:#ffd600  !important; border-radius:8px !important; font-weight:bold !important; transition:background .15s !important; }
.fb-whistle:hover { background:rgba(255,214,0,0.20) !important; }
.fb-noise   { flex: 1; padding: 14px !important; font-size: 0.95rem !important; background:rgba(255,82,82,0.10)   !important; border:1px solid #ff5252 !important; color:#ff5252  !important; border-radius:8px !important; font-weight:bold !important; transition:background .15s !important; }
.fb-noise:hover   { background:rgba(255,82,82,0.20) !important; }
.fb-reset   { background:transparent !important; border:1px solid #2a3a4a !important; color:#4a6070 !important; border-radius:6px !important; font-size:0.72rem !important; }
.fb-reset:hover { border-color:#ff5252 !important; color:#ff5252 !important; }
"""

# ────────────────────────────────────────────────────────────
#  INTERFAZ
# ────────────────────────────────────────────────────────────

def build_interface():
    with gr.Blocks(css=CSS, title="NAO Audio Cleaner · RoboCup HSL",
                   theme=gr.themes.Base()) as demo:

        # Estado de sesión (invisible)
        analysis_state = gr.State({})

        # ── Cabecera ─────────────────────────────────────────
        gr.HTML("""
        <div class="nao-header">
            <h1>🤖 NAO Audio Cleaner</h1>
            <p>Limpieza · Detección · Feedback adaptativo &nbsp;·&nbsp; RoboCup HSL &nbsp;·&nbsp; NAO v6</p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ══════════════════════════════════
            # COL IZQUIERDA — Controles
            # ══════════════════════════════════
            with gr.Column(scale=1, min_width=300):

                gr.HTML(sec_title("📂", "Archivos de audio"))
                file_input = gr.File(
                    label="Arrastra o haz clic para subir  (WAV · MP3 · OGG · FLAC)",
                    file_count="multiple",
                    file_types=[".wav", ".mp3", ".ogg", ".flac", ".aac"],
                )
                badge_upload_live = gr.HTML(badge("0 archivos subidos"))

                gr.HTML(sec_title("⚙️", "Limpieza"))
                with gr.Group():
                    noise_strength = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.75, step=0.05,
                        label="Fuerza de reducción de ruido",
                        info="0 = sin cambios  ·  1 = máxima reducción",
                    )
                    apply_bp = gr.Checkbox(
                        value=True,
                        label="Filtro paso-banda  (100 Hz – 10 kHz)",
                        info="Elimina vibraciones mecánicas fuera del rango útil",
                    )
                    do_normalize = gr.Checkbox(
                        value=True,
                        label="Normalizar volumen",
                        info="Sube el volumen al máximo posible sin distorsión",
                    )

                run_btn = gr.Button("🚀  Limpiar y analizar", variant="primary", elem_classes=["lg"])

                with gr.Accordion("ℹ️  Ayuda rápida", open=False):
                    gr.Markdown("""
**Pasos:**
1. Sube los archivos grabados por el NAO v6.
2. Ajusta los controles si es necesario.
3. Clic en **Limpiar y analizar**.
4. En la sección **Revisar**, escucha cada audio y verifica la clasificación.
5. Usa los botones de feedback para corregir errores — el sistema aprenderá.
6. Descarga los audios limpios individualmente o en ZIP.

> Si el audio limpio suena metálico, baja la fuerza de ruido a 0.5 – 0.6.

**Clasificación:**
- 🟢 **SILBATO** → contiene el silbato del árbitro
- 🔴 **SOLO RUIDO** → solo ruido de fondo

**Feedback:**
- Selecciona **"Tiene silbato"** o **"No tiene silbato"**.
- El sistema comparará tu respuesta con su clasificación y determinará si es un verdadero/falso positivo/negativo.

Con 5+ correcciones de ambas clases, los umbrales se optimizan automáticamente. Usa el botón de **Volver a analizar** para aplicar los nuevos umbrales a los audios actuales.
                    """)

            # ══════════════════════════════════
            # COL DERECHA — Resultados
            # ══════════════════════════════════
            with gr.Column(scale=2):

                # ── Tabla de análisis ─────────────────────────
                gr.HTML(sec_title("📊", "Análisis"))
                summary_md = gr.Markdown(
                    "_Los resultados aparecerán aquí._",
                    elem_classes=["summary"],
                )
                results_df = gr.Dataframe(
                    headers=["Archivo", "Clasificación", "Energía máx",
                             "Dur. silbato (s)", "Pico espectral", "Estado"],
                    interactive=False,
                    wrap=True,
                )

                gr.HTML('<hr class="divider">')

                # ── Sección de revisión y feedback ────────────
                gr.HTML(sec_title("🎧", "Revisar y Corregir Clasificación"))
                gr.HTML(
                    '<div style="color:#4a6070;font-size:0.82rem;margin-bottom:12px">'
                    'Escucha cada audio y verifica si la clasificación es correcta. '
                    'Tu feedback entrena el detector automáticamente.</div>'
                )

                with gr.Group(elem_classes=["review-wrap"]):
                    with gr.Row():
                        with gr.Column(scale=3):
                            file_selector = gr.Dropdown(
                                choices=[],
                                label="Seleccionar audio para revisar",
                                interactive=True,
                                visible=False,
                            )
                            review_player = gr.Audio(
                                value=None,
                                visible=False,
                                label="",
                                type="numpy",
                                buttons=["download"],
                                elem_classes=["review-player"],
                            )

                        with gr.Column(scale=2):
                            clf_info = gr.HTML(
                                '<div style="color:#4a6070;font-size:0.85rem;padding:10px 0">'
                                'Selecciona un audio para ver su clasificación.</div>'
                            )

                    gr.HTML(
                        '<div style="font-family:Space Mono,monospace;font-size:0.7rem;'
                        'color:#4a6070;text-transform:uppercase;letter-spacing:2px;'
                        'margin:14px 0 8px">¿Es correcta esta clasificación?</div>'
                    )

                    with gr.Row(elem_classes=["fb-actions-row"]):
                        btn_whistle = gr.Button(
                            "🟢  Tiene silbato",
                            elem_classes=["fb-whistle"],
                        )
                        btn_noise = gr.Button(
                            "🔴  No tiene silbato",
                            elem_classes=["fb-noise"],
                        )

                    feedback_status = gr.HTML("")

                    with gr.Row():
                        gr.HTML("")   # spacer
                        btn_reset = gr.Button(
                            "🗑️  Reiniciar todo el feedback",
                            size="sm",
                            elem_classes=["fb-reset"],
                        )

                gr.HTML('<hr class="divider">')

                # ── Panel de aprendizaje adaptativo ───────────
                gr.HTML(sec_title("🧠", "Aprendizaje Adaptativo"))
                feedback_stats = gr.HTML(stats_html())
                
                with gr.Row():
                    btn_reanalyze = gr.Button(
                        "🔄  Volver a analizar originales con nuevos umbrales", 
                        elem_classes=["reanalyze-btn"]
                    )

                gr.HTML('<hr class="divider">')

                # ── Audios originales ─────────────────────────
                gr.HTML(sec_title("🎙️", "Audios originales"))
                badge_orig = gr.HTML(badge("0 archivos subidos"))
                zip_orig   = gr.File(
                    label="📦  Descargar originales (.zip)",
                    interactive=False,
                    visible=False,
                    elem_classes=["zip-wrap"],
                )
                orig_players = []
                for row_i in range(0, MAX_PLAYERS, 2):
                    with gr.Row():
                        for col_j in range(2):
                            if (row_i + col_j) < MAX_PLAYERS:
                                p = gr.Audio(
                                    value=None, visible=False, label="",
                                    type="numpy", buttons=["download"],
                                    elem_classes=["player-wrap"],
                                )
                                orig_players.append(p)

                gr.HTML('<hr class="divider">')

                # ── Audios limpios ────────────────────────────
                gr.HTML(sec_title("✨", "Audios limpios"))
                badge_clean = gr.HTML(badge("0 archivos listos para descargar"))
                zip_clean   = gr.File(
                    label="📦  Descargar limpios (.zip)",
                    interactive=False,
                    visible=False,
                    elem_classes=["zip-wrap"],
                )
                clean_players = []
                for row_i in range(0, MAX_PLAYERS, 2):
                    with gr.Row():
                        for col_j in range(2):
                            if (row_i + col_j) < MAX_PLAYERS:
                                p = gr.Audio(
                                    value=None, visible=False, label="",
                                    type="numpy", buttons=["download"],
                                    elem_classes=["player-wrap"],
                                )
                                clean_players.append(p)

        # ─────────────────────────────────────────────────────
        #  EVENTOS
        # ─────────────────────────────────────────────────────

        # Badge en tiempo real al subir
        def live_badge(files):
            n = len(files) if files else 0
            s = "s" if n != 1 else ""
            return badge(f"{n} archivo{s} subido{s}")

        file_input.change(fn=live_badge, inputs=[file_input],
                          outputs=[badge_upload_live])

        # Procesar inicial y Re-procesar
        all_outputs = (
            orig_players
            + clean_players
            + [zip_orig, zip_clean,
               results_df, summary_md,
               badge_orig, badge_clean,
               analysis_state, file_selector, feedback_stats,
               review_player, clf_info]
        )

        run_btn.click(
            fn=process_audio,
            inputs=[file_input, noise_strength, apply_bp, do_normalize],
            outputs=all_outputs,
        )
        
        btn_reanalyze.click(
            fn=process_audio,
            inputs=[file_input, noise_strength, apply_bp, do_normalize],
            outputs=all_outputs,
        )

        # Cambio de archivo en dropdown → actualiza reproductor e info
        file_selector.change(
            fn=on_file_select,
            inputs=[file_selector, analysis_state],
            outputs=[review_player, clf_info],
        )

        # Botones de feedback
        btn_whistle.click(
            fn=on_mark_whistle,
            inputs=[file_selector, analysis_state],
            outputs=[feedback_status, feedback_stats],
        )
        btn_noise.click(
            fn=on_mark_noise,
            inputs=[file_selector, analysis_state],
            outputs=[feedback_status, feedback_stats],
        )
        btn_reset.click(
            fn=on_reset_feedback,
            inputs=[],
            outputs=[feedback_status, feedback_stats],
        )

    return demo


# ────────────────────────────────────────────────────────────
#  ARRANQUE
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🤖  NAO Audio Cleaner v3 — RoboCup HSL")
    print("=" * 55)
    print("  Abre en tu navegador:  http://localhost:7860")
    print("  Para salir presiona:   Ctrl + C")
    print("=" * 55 + "\n")

    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )