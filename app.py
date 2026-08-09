import os
from flask import Flask, render_template_string, jsonify, request
from google import genai

# ==========================================
# CONFIGURATION & FLASK BACKEND
# ==========================================
app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@app.route('/api/ai_chat', methods=['POST'])
def ai_chat():
    req = request.json or {}
    query = req.get('query', '').strip()
    lang = req.get('lang', 'en-IN')
    m = req.get('metrics', {})

    lang_names = {
        'en-IN': 'English', 'hi-IN': 'Hindi', 'mr-IN': 'Marathi',
        'gu-IN': 'Gujarati', 'pa-IN': 'Punjabi', 'ta-IN': 'Tamil', 'te-IN': 'Telugu'
    }
    target_lang = lang_names.get(lang, 'English')

    if ai_client:
        try:
            prompt = (
                f"You are SpecTantra AI, an Indian agricultural expert.\n"
                f"Live Soil Analysis Context:\n"
                f"- Nitrogen: {m.get('nitrogen', 'Optimal')}\n"
                f"- Phosphorus: {m.get('phosphorus', 'Optimal')}\n"
                f"- Potassium: {m.get('potassium', 'Optimal')}\n"
                f"- pH: {m.get('ph', 6.8)} ({m.get('ph_class', 'Neutral')})\n"
                f"- Quality Score: {m.get('score', 92)}%\n\n"
                f"Farmer Question: '{query}'\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Answer the question directly and practically.\n"
                f"2. Use the live soil readings if applicable, or give expert agricultural advice for general farming questions.\n"
                f"3. Keep response concise (2-3 sentences) and strictly in {target_lang}."
            )
            res = ai_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return jsonify({"status": "ok", "response": res.text.strip()})
        except Exception as e:
            print(f"Gemini API Error: {e}")

    # Smart Fallback Engine
    q_low = query.lower()
    if any(k in q_low for k in ["sugarcane", "गन्ना", "ऊस"]):
        ans = f"Sugarcane requires a pH of 6.0 to 7.5, while Wheat requires 6.0 to 7.0. Your current soil pH is {m.get('ph', 6.8)}."
    elif any(k in q_low for k in ["brand", "company", "fertilizer"]):
        ans = "Top trusted fertilizer brands in India are IFFCO, Mahadhan, Coromandel, and Kribhco."
    else:
        ans = f"Regarding '{query}': Soil pH is {m.get('ph', 6.8)}. {m.get('recommendation', 'Maintain organic crop rotation.')}"

    return jsonify({"status": "ok", "response": ans})

# ==========================================
# DASHBOARD INTERFACE & JS ENGINE
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpecTantra AI - Soil Spectroscopy Engine</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #0b1329; color: #f8fafc; font-family: system-ui, sans-serif; }
        .card { background-color: #131e3a; border: 1px solid #1e2d5a; border-radius: 12px; }
        .video-container { position: relative; width: 100%; cursor: crosshair; }
        canvas#displayCanvas { width: 100%; border-radius: 8px; border: 2px solid #00d2ff; background: #000; min-height: 260px; }
        .badge-val { font-size: 1.1rem; font-weight: 700; padding: 8px 16px; border-radius: 6px; display: inline-block; width: 100%; }
        .bg-optimal { background-color: #10b981; color: #fff; }
        .bg-deficient { background-color: #ef4444; color: #fff; }
        .bg-surplus { background-color: #f59e0b; color: #fff; }
        .metric-label { font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 4px; display: block; }
        .control-btn { font-weight: 600; text-transform: uppercase; font-size: 0.85rem; }
    </style>
</head>
<body class="p-3">
    <div class="container-fluid">
        <div class="d-flex justify-content-between align-items-center pb-3 mb-3 border-bottom border-secondary">
            <h3 class="m-0 text-info fw-bold">🔬 SpecTantra AI <span class="fs-6 text-light fw-normal">| Soil Spectroscopy Engine</span></h3>
            <button onclick="startCamera()" class="btn btn-sm btn-primary">📷 Enable Camera</button>
        </div>

        <div class="row g-3">
            <div class="col-lg-7">
                <div class="card p-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h5 class="m-0 text-warning fw-bold">📹 Live Spectral Stream & Graph</h5>
                        <small class="text-muted">Click image to position Target ROI Box</small>
                    </div>
                    
                    <div class="video-container" onclick="handleCanvasClick(event)">
                        <canvas id="displayCanvas"></canvas>
                        <video id="webcam" autoplay playsinline muted style="display: none;"></video>
                    </div>
                    
                    <div class="row g-2 mt-2 align-items-center">
                        <div class="col-auto"><small class="text-info fw-bold">ROI X:</small> <input type="number" id="roiX" value="150" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><small class="text-info fw-bold">Y:</small> <input type="number" id="roiY" value="100" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><small class="text-info fw-bold">Width:</small> <input type="number" id="roiW" value="340" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><small class="text-info fw-bold">Height:</small> <input type="number" id="roiH" value="60" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><button onclick="updateRoiFromInputs()" class="btn btn-sm btn-outline-info">Update ROI</button></div>
                    </div>

                    <!-- CONTROL BUTTONS -->
                    <div class="d-flex gap-2 mt-3">
                        <button onclick="saveTestLocally()" class="btn btn-success flex-fill control-btn">💾 [S] SAVE TEST DATA</button>
                        <button onclick="calibrateBaseline()" class="btn btn-info flex-fill control-btn">🎯 [C] CALIBRATE BASELINE</button>
                        <button onclick="flipGraph()" class="btn btn-secondary flex-fill control-btn">🔄 [F] FLIP GRAPH</button>
                        <button onclick="resetCalibration()" class="btn btn-outline-danger flex-fill control-btn">❌ [R] RESET</button>
                    </div>
                </div>
            </div>

            <div class="col-lg-5">
                <div class="card p-3 mb-3">
                    <h5 class="text-success fw-bold mb-3">📊 Real-Time Soil Analysis</h5>
                    
                    <div class="row g-2 text-center mb-3">
                        <div class="col-4"><div class="p-2 border border-secondary rounded bg-dark"><span class="metric-label">Nitrogen (N)</span><span id="valN" class="badge-val bg-optimal">--</span></div></div>
                        <div class="col-4"><div class="p-2 border border-secondary rounded bg-dark"><span class="metric-label">Phosphorus (P)</span><span id="valP" class="badge-val bg-optimal">--</span></div></div>
                        <div class="col-4"><div class="p-2 border border-secondary rounded bg-dark"><span class="metric-label">Potassium (K)</span><span id="valK" class="badge-val bg-optimal">--</span></div></div>
                    </div>

                    <div class="row g-2 text-center mb-3">
                        <div class="col-6"><div class="p-2 border border-secondary rounded bg-dark"><span class="metric-label">Soil pH</span><h3 id="valPh" class="m-0 text-info fw-bold">--</h3><small id="valPhClass" class="text-warning fw-bold">--</small></div></div>
                        <div class="col-6"><div class="p-2 border border-secondary rounded bg-dark"><span class="metric-label">Health Index</span><h3 id="valScore" class="m-0 text-success fw-bold">--%</h3><small class="text-light">Quality Score</small></div></div>
                    </div>

                    <div class="p-3 bg-dark rounded border border-secondary">
                        <small class="text-warning fw-bold d-block mb-1">💡 Advisory:</small>
                        <p id="valAdv" class="m-0 small text-light">Awaiting camera baseline calibration...</p>
                    </div>
                </div>

                <div class="card p-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h5 class="m-0 text-warning fw-bold">🤖 Multilingual Gemini AI</h5>
                        <select id="langSelect" class="form-select form-select-sm bg-dark text-light border-secondary" style="width: auto;">
                            <option value="en-IN" selected>English (India)</option>
                            <option value="hi-IN">Hindi (हिंदी)</option>
                            <option value="mr-IN">Marathi (मराठी)</option>
                            <option value="gu-IN">Gujarati (ગુજરાતી)</option>
                            <option value="pa-IN">Punjabi (ਪੰਜਾਬੀ)</option>
                            <option value="ta-IN">Tamil (தமிழ்)</option>
                            <option value="te-IN">Telugu (తెలుగు)</option>
                        </select>
                    </div>

                    <div class="input-group mb-2">
                        <input type="text" id="aiQueryInput" class="form-control bg-dark text-light border-secondary" placeholder="Ask crop, fertilizer, or soil questions...">
                        <button onclick="startVoice()" class="btn btn-outline-warning">🎙️ Speak</button>
                        <button onclick="sendAiQuery()" class="btn btn-info fw-bold">Ask Gemini</button>
                    </div>

                    <div class="p-3 bg-dark rounded border border-secondary" style="min-height: 80px;">
                        <small class="text-info fw-bold d-block mb-1">Gemini AI Response:</small>
                        <p id="aiResponseText" class="m-0 small text-light">Select language and ask a question...</p>
                    </div>

                    <div class="d-flex gap-2 mt-3">
                        <button onclick="shareWhatsApp()" class="btn btn-sm btn-outline-success flex-fill">💬 WhatsApp</button>
                        <button onclick="shareEmail()" class="btn btn-sm btn-outline-primary flex-fill">✉️ Email</button>
                        <button onclick="downloadCSVClient()" class="btn btn-sm btn-outline-warning flex-fill">📥 Download CSV (<span id="testCount">0</span>)</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let roi = { x: 150, y: 100, w: 340, h: 60 };
        let flipDir = false;
        let baselineProfile = null;
        let lastProfile = null;
        let currentAnalysis = {};

        async function startCamera() {
            const video = document.getElementById('webcam');
            let stream = null;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } }
                });
            } catch (err1) {
                try { stream = await navigator.mediaDevices.getUserMedia({ video: true }); }
                catch (err2) { return alert("Camera permission denied or unavailable."); }
            }

            if (stream) {
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play();
                    requestAnimationFrame(renderLoop);
                };
            }
        }

        function renderLoop() {
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('displayCanvas');
            if (!video.videoWidth) {
                requestAnimationFrame(renderLoop);
                return;
            }

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');

            // Draw Camera Video
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Process Spectral ROI
            const rx = Math.max(0, Math.min(roi.x, canvas.width - 20));
            const ry = Math.max(0, Math.min(roi.y, canvas.height - 20));
            const rw = Math.max(20, Math.min(roi.w, canvas.width - rx));
            const rh = Math.max(20, Math.min(roi.h, canvas.height - ry));

            const imgData = ctx.getImageData(rx, ry, rw, rh);
            const data = imgData.data;

            // Compute 1D Profile (Gray Intensity Average per column)
            let profile = new Float32Array(rw);
            for (let c = 0; c < rw; c++) {
                let sum = 0;
                for (let r = 0; r < rh; r++) {
                    let idx = (r * rw + c) * 4;
                    sum += (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
                }
                profile[c] = sum / rh;
            }

            if (flipDir) profile.reverse();

            let maxVal = 0;
            for (let i = 0; i < rw; i++) if (profile[i] > maxVal) maxVal = profile[i];
            if (maxVal === 0) maxVal = 1.0;

            let norm = new Float32Array(rw);
            for (let i = 0; i < rw; i++) norm[i] = profile[i] / maxVal;
            lastProfile = Array.from(norm);

            let absorbance = new Float32Array(rw);
            if (baselineProfile && baselineProfile.length === rw) {
                for (let i = 0; i < rw; i++) {
                    absorbance[i] = Math.max(0, Math.min(1.0, 1.0 - (norm[i] / (baselineProfile[i] + 1e-5))));
                }
            } else {
                absorbance = norm;
            }

            // Partition Wavelength Bands
            const bThird = Math.floor(rw / 3);
            let blueBand = 0, greenBand = 0, redBand = 0;

            for (let i = 0; i < bThird; i++) blueBand += absorbance[i];
            for (let i = bThird; i < 2 * bThird; i++) greenBand += absorbance[i];
            for (let i = 2 * bThird; i < rw; i++) redBand += absorbance[i];

            blueBand /= bThird;
            greenBand /= bThird;
            redBand /= (rw - 2 * bThird);

            function getStatus(val) { return val < 0.35 ? "Deficient" : (val > 0.75 ? "Surplus" : "Optimal"); }
            const nStat = getStatus(blueBand);
            const kStat = getStatus(greenBand);
            const pStat = getStatus(redBand);

            const ratio = (blueBand + 1e-5) / (redBand + 1e-5);
            const estPh = Math.round(Math.max(4.5, Math.min(8.5, 6.5 + (ratio - 1.0) * 1.2)) * 10) / 10;
            const phClass = estPh < 6.0 ? "Acidic (Needs Lime)" : (estPh > 7.5 ? "Alkaline (Needs Gypsum)" : "Neutral (Balanced)");
            const score = Math.round(Math.max(30, Math.min(98, 100 - (Math.abs(7.0 - estPh) * 12 + (nStat === "Optimal" ? 0 : 15) + (pStat === "Optimal" ? 0 : 15)))));

            let adv = [];
            if (nStat === "Deficient") adv.push("Apply Urea or Neem-coated Nitrogen.");
            if (pStat === "Deficient") adv.push("Apply Single Super Phosphate (SSP).");
            if (kStat === "Deficient") adv.push("Apply Muriate of Potash (MOP).");
            if (phClass.includes("Acidic")) adv.push("Apply Agricultural Lime.");
            if (phClass.includes("Alkaline")) adv.push("Apply Gypsum.");
            const rec = adv.length ? adv.join(" ") : "Soil health is optimal. Maintain current organic crop rotation.";

            currentAnalysis = { nitrogen: nStat, phosphorus: pStat, potassium: kStat, ph: estPh, ph_class: phClass, score: score, recommendation: rec };

            // Update UI Metrics
            updateBadge('valN', nStat);
            updateBadge('valP', pStat);
            updateBadge('valK', kStat);
            document.getElementById('valPh').innerText = estPh;
            document.getElementById('valPhClass').innerText = phClass;
            document.getElementById('valScore').innerText = score + "%";
            document.getElementById('valAdv').innerText = rec;

            // Draw ROI Box Overlay
            ctx.strokeStyle = "#00d2ff";
            ctx.lineWidth = 2;
            ctx.strokeRect(rx, ry, rw, rh);
            ctx.fillStyle = "#00d2ff";
            ctx.font = "14px sans-serif";
            ctx.fillText(`TARGET ROI (${rx},${ry})`, rx, Math.max(15, ry - 6));

            // Draw Wavelength Spectrum Graph
            const gh = 90, gw = canvas.width - 20, gx = 10, gy = canvas.height - 100;
            ctx.fillStyle = "rgba(15, 15, 15, 0.75)";
            ctx.fillRect(gx, gy, gw, gh);
            ctx.strokeStyle = "#666";
            ctx.strokeRect(gx, gy, gw, gh);

            // Rainbow Bar
            for (let c = 0; c < gw; c++) {
                let rC = c / gw;
                let color = rC < 0.5 
                    ? `rgb(0, ${Math.floor(rC * 510)}, ${Math.floor((1 - rC * 2) * 255)})`
                    : `rgb(${Math.floor((rC - 0.5) * 510)}, ${Math.floor((1 - (rC - 0.5) * 2) * 255)}, 0)`;
                ctx.fillStyle = color;
                ctx.fillRect(gx + c, gy + gh - 5, 1, 4);
            }

            // Reflectance Curve Line
            ctx.beginPath();
            ctx.strokeStyle = "#ffff00";
            ctx.lineWidth = 2;
            for (let i = 0; i < rw; i++) {
                let px = gx + Math.floor((i / rw) * gw);
                let py = gy + gh - 8 - Math.floor(norm[i] * (gh - 18));
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();

            requestAnimationFrame(renderLoop);
        }

        function updateBadge(id, status) {
            let el = document.getElementById(id);
            el.innerText = status;
            el.className = 'badge-val ' + (status === 'Optimal' ? 'bg-optimal' : (status === 'Deficient' ? 'bg-deficient' : 'bg-surplus'));
        }

        function handleCanvasClick(e) {
            const canvas = document.getElementById('displayCanvas');
            const rect = canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;

            const realX = Math.round(clickX * scaleX);
            const realY = Math.round(clickY * scaleY);

            const w = parseInt(document.getElementById('roiW').value) || 340;
            const h = parseInt(document.getElementById('roiH').value) || 60;

            roi.x = Math.max(0, realX - Math.round(w / 2));
            roi.y = Math.max(0, realY - Math.round(h / 2));
            document.getElementById('roiX').value = roi.x;
            document.getElementById('roiY').value = roi.y;
        }

        function updateRoiFromInputs() {
            roi.x = parseInt(document.getElementById('roiX').value) || 150;
            roi.y = parseInt(document.getElementById('roiY').value) || 100;
            roi.w = parseInt(document.getElementById('roiW').value) || 340;
            roi.h = parseInt(document.getElementById('roiH').value) || 60;
        }

        function calibrateBaseline() {
            if (lastProfile) {
                baselineProfile = Array.from(lastProfile);
                alert("🎯 Baseline calibrated successfully!");
            } else {
                alert("Enable camera first to capture baseline spectrum.");
            }
        }

        function flipGraph() { flipDir = !flipDir; }
        function resetCalibration() { baselineProfile = null; flipDir = false; alert("❌ Calibration reset."); }

        function getSavedTests() { return JSON.parse(localStorage.getItem('soil_tests') || '[]'); }

        function saveTestLocally() {
            if (!currentAnalysis.ph) return alert("Please enable camera to capture live test data first.");
            let tests = getSavedTests();
            tests.push({
                timestamp: new Date().toLocaleString(),
                nitrogen: currentAnalysis.nitrogen,
                phosphorus: currentAnalysis.phosphorus,
                potassium: currentAnalysis.potassium,
                ph: currentAnalysis.ph,
                ph_class: currentAnalysis.ph_class,
                score: currentAnalysis.score,
                recommendation: currentAnalysis.recommendation
            });
            localStorage.setItem('soil_tests', JSON.stringify(tests));
            updateTestCounter();
            alert("💾 Test record saved successfully! Total saved: " + tests.length);
        }

        function updateTestCounter() {
            document.getElementById('testCount').innerText = getSavedTests().length;
        }

        function downloadCSVClient() {
            let tests = getSavedTests();
            if (tests.length === 0) return alert("No saved tests found. Tap '[S] SAVE TEST DATA' first!");

            let csv = "data:text/csv;charset=utf-8,Timestamp,Nitrogen,Phosphorus,Potassium,pH,pH Classification,Health Score (%),Recommendation\n";
            tests.forEach(t => {
                csv += `"${t.timestamp}","${t.nitrogen}","${t.phosphorus}","${t.potassium}","${t.ph}","${t.ph_class}","${t.score}","${t.recommendation.replace(/"/g, '""')}"\n`;
            });

            let link = document.createElement("a");
            link.href = encodeURI(csv);
            link.download = `soil_database_${Date.now()}.csv`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        function sendAiQuery() {
            let text = document.getElementById('aiQueryInput').value, lang = document.getElementById('langSelect').value;
            if (!text) return;
            document.getElementById('aiResponseText').innerText = "Thinking...";

            fetch('/api/ai_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query: text, lang: lang, metrics: currentAnalysis })
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('aiResponseText').innerText = data.response;
                speakText(data.response, lang);
            });
        }

        function startVoice() {
            let lang = document.getElementById('langSelect').value;
            let SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SR) return alert("Speech recognition not supported in this browser.");
            let rec = new SR();
            rec.lang = lang;
            rec.onresult = e => { document.getElementById('aiQueryInput').value = e.results[0][0].transcript; sendAiQuery(); };
            rec.start();
        }

        function speakText(text, lang) {
            if (!('speechSynthesis' in window)) return;
            window.speechSynthesis.cancel();
            let msg = new SpeechSynthesisUtterance(text);
            msg.lang = lang;

            function executeSpeech() {
                let voices = window.speechSynthesis.getVoices();
                let prefix = lang.split('-')[0].toLowerCase();
                let match = voices.find(v => v.lang.toLowerCase() === lang.toLowerCase()) ||
                            voices.find(v => v.lang.toLowerCase().startsWith(prefix)) ||
                            voices.find(v => v.name.toLowerCase().includes('marathi') || v.name.toLowerCase().includes('hindi')) ||
                            voices.find(v => v.lang.toLowerCase().includes('in'));
                if (match) msg.voice = match;
                window.speechSynthesis.speak(msg);
            }

            let voices = window.speechSynthesis.getVoices();
            if (voices.length > 0) executeSpeech();
            else window.speechSynthesis.onvoiceschanged = executeSpeech;
        }

        function shareWhatsApp() {
            let txt = `SpecTantra Soil Report: N=${currentAnalysis.nitrogen}, P=${currentAnalysis.phosphorus}, K=${currentAnalysis.potassium}, pH=${currentAnalysis.ph}. Score: ${currentAnalysis.score}%. Advice: ${currentAnalysis.recommendation}`;
            window.open(`https://wa.me/?text=${encodeURIComponent(txt)}`, '_blank');
        }

        function shareEmail() {
            let txt = `SpecTantra Soil Report: N=${currentAnalysis.nitrogen}, P=${currentAnalysis.phosphorus}, K=${currentAnalysis.potassium}, pH=${currentAnalysis.ph}. Score: ${currentAnalysis.score}%. Advice: ${currentAnalysis.recommendation}`;
            window.open(`mailto:?subject=Soil Diagnostics Report&body=${encodeURIComponent(txt)}`);
        }

        document.addEventListener('keydown', function(e) {
            if (document.activeElement.tagName === 'INPUT') return;
            let k = e.key.toLowerCase();
            if (k === 's') saveTestLocally();
            if (k === 'c') calibrateBaseline();
            if (k === 'f') flipGraph();
            if (k === 'r') resetCalibration();
        });

        window.addEventListener('DOMContentLoaded', () => { 
            startCamera(); 
            updateTestCounter();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)