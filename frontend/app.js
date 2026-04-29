// Nota: Usa /api/... directamente porque el index.html se va a servir en la misma ruta que FastAPI (http://localhost:8000)
const API_BASE = '/api';

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsGrid = document.getElementById('resultsGrid');
const loader = document.getElementById('loader');
const statsContainer = document.getElementById('statsContainer');
const statsBoxes = document.getElementById('statsBoxes');
const noiseStrength = document.getElementById('noiseStrength');
const noiseValLabel = document.getElementById('noiseVal');

let selectedFiles = [];

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFiles(e.dataTransfer.files);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFiles(e.target.files);
    }
});

noiseStrength.addEventListener('input', (e) => {
    noiseValLabel.textContent = e.target.value;
});

function handleFiles(files) {
    selectedFiles = Array.from(files).filter(f => f.name.match(/\.(wav|mp3|ogg|flac|aac)$/i));
    fileList.innerHTML = '';
    
    if (selectedFiles.length === 0) {
        fileList.innerHTML = '<div style="color:var(--danger);font-size:0.85em;">Ningún archivo de audio válido seleccionado.</div>';
        analyzeBtn.disabled = true;
        return;
    }
    
    selectedFiles.forEach((file) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-pill';
        fileItem.innerHTML = `
            <span>📄 ${file.name}</span>
            <span style="color:var(--primary)">${(file.size / 1024 / 1024).toFixed(2)} MB</span>
        `;
        fileList.appendChild(fileItem);
    });
    
    analyzeBtn.disabled = false;
    dropzone.querySelector('h3').innerText = `${selectedFiles.length} Archivos Listos`;
}

// Processing
analyzeBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;
    
    analyzeBtn.disabled = true;
    loader.classList.remove('hidden');
    resultsGrid.innerHTML = '';
    statsContainer.classList.add('hidden');
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    
    formData.append('noise_strength', noiseStrength.value);
    formData.append('apply_bp', document.getElementById('applyBp').checked);
    formData.append('do_normalize', document.getElementById('doNormalize').checked);
    
    try {
        const response = await fetch(`${API_BASE}/process`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error("Error HTTP " + response.status);
        const data = await response.json();
        
        if(data.results){
            renderResults(data.results);
            renderStats(data.stats);
            // Hacer scroll a los resultados
            document.querySelector('.results-wrapper').scrollIntoView({ behavior: 'smooth' });
        } else {
            throw new Error("Respuesta del servidor no válida");
        }
    } catch (err) {
        console.error("Connection Error:", err);
        resultsGrid.innerHTML = `<div class="empty-state"><div class="empty-icon" style="color:var(--danger)">⚠️</div><p style="color:var(--danger)">Fallo de comunicación: ${err.message}.<br><br>Asegúrate de que estás ingresando por localhost:8000</p></div>`;
    } finally {
        analyzeBtn.disabled = false;
        loader.classList.add('hidden');
    }
});

function renderResults(results) {
    resultsGrid.innerHTML = '';
    
    results.forEach(res => {
        const card = document.createElement('div');
        
        if (res.status === "ERROR") {
            card.className = "res-card is-ruido";
            card.innerHTML = `<div style="grid-column: 1/-1; color: var(--danger)">❌ <b>${res.filename}</b>: ${res.error}</div>`;
            resultsGrid.appendChild(card);
            return;
        }

        const isWhistle = res.classification.includes('SILBATO');
        card.className = `res-card ${isWhistle ? 'is-silbato' : 'is-ruido'}`;
        const tagClass = isWhistle ? 'tag-silbato' : 'tag-ruido';
        const cardId = Math.random().toString(36).substring(7);
        
        // El atributo Features debe sanitizarse para HTML
        const featStr = JSON.stringify(res.features).replace(/"/g, '&quot;');
        const fileUrl = `${API_BASE}/audios/${res.clean_url.split('/').pop()}`;
        
        card.innerHTML = `
            <div class="rc-info">
                <span class="rc-tag ${tagClass}">${res.classification}</span>
                <h4>${res.filename}</h4>
                <div class="rc-metrics">
                    <div class="metric"><span>Pico Energía</span><span>${res.energia_max.toFixed(3)}</span></div>
                    <div class="metric"><span>Duración</span><span>${res.duracion_s.toFixed(2)}s</span></div>
                </div>
            </div>
            
            <div class="rc-audio">
                <audio controls src="${fileUrl}"></audio>
            </div>
            
            <div class="rc-feedback" id="actions-${cardId}">
                <p>Verificar Clasificación</p>
                <div class="f-btns">
                    <button class="f-btn f-yes" onclick="sendFeedback('${res.filename}', '${res.classification}', '🟢 SILBATO', ${featStr}, 'actions-${cardId}')">✔️ Silbato</button>
                    <button class="f-btn f-no" onclick="sendFeedback('${res.filename}', '${res.classification}', '🔴 SOLO RUIDO', ${featStr}, 'actions-${cardId}')">❌ Ruido</button>
                </div>
            </div>
        `;
        resultsGrid.appendChild(card);
    });
}

function renderStats(stats) {
    if (!stats || stats.samples_count === 0) return;
    statsContainer.classList.remove('hidden');
    
    statsBoxes.innerHTML = `
        <div class="stat-box">
            <span class="v">${stats.samples_count}</span>
            <span class="l">Muestras</span>
        </div>
        <div class="stat-box">
            <span class="v" style="color:var(--success)">${stats.accuracy_pct}%</span>
            <span class="l">Precisión</span>
        </div>
        <div class="stat-box">
            <span class="v" style="color:var(--primary)">${stats.thresholds.energy_current.toFixed(3)}</span>
            <span class="l">Umbral Energía (Act)</span>
        </div>
        <div class="stat-box">
            <span class="v" style="color:var(--primary)">${stats.thresholds.duration_current.toFixed(3)}s</span>
            <span class="l">Umbral Duración (Act)</span>
        </div>
    `;
}

window.sendFeedback = async function(filename, predicted, corrected, features, actionsId) {
    const parent = document.getElementById(actionsId);
    parent.innerHTML = '<span class="f-resp" style="color:var(--primary)">Actualizando...</span>';
    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, predicted, corrected, features })
        });
        const data = await response.json();
        if(data.success){
            const msg = data.adapted ? "⚡ Umbrales Ajustados" : "✅ Guardado";
            const clr = data.adapted ? "var(--warning)" : "var(--success)";
            parent.innerHTML = `<span class="f-resp" style="color:${clr}">${msg}</span>`;
            if (data.stats) renderStats(data.stats);
        }
    } catch(err){
        console.error(err);
        parent.innerHTML = '<span class="f-resp" style="color:var(--danger)">Error al guardar</span>';
    }
};

window.resetFeedback = async function() {
    if(!confirm("¿Estás seguro de borrar toda la base de datos de aprendizaje y volver a los umbrales de fábrica?")) return;
    try {
        const response = await fetch(`${API_BASE}/reset`, { method: 'POST' });
        const data = await response.json();
        if(data.success){
            alert("✅ Perfil de aprendizaje reseteado con éxito.");
            statsContainer.classList.add('hidden');
        }
    } catch(err){
        alert("Error al intentar reiniciar.");
    }
};

window.onload = async () => {
    try {
        const resp = await fetch(`${API_BASE}/stats`);
        if(resp.ok) {
            const stats = await resp.json();
            renderStats(stats);
        }
    } catch(e){}
};
