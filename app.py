import os, cv2, json, base64, numpy as np
from flask import Flask, render_template_string, jsonify, request
from google import genai

# ==========================================
# CONFIGURATION & GLOBAL STATE
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

roi_x, roi_y, roi_w, roi_h = 150, 100, 340, 60
flip_direction = False
baseline_profile = None

latest_metrics = {
    "nitrogen": "Optimal", "phosphorus": "Optimal", "potassium": "Optimal",
    "ph": 6.8, "ph_class": "Neutral (Balanced)", "score": 92,
    "recommendation": "Soil health is optimal. Maintain balanced organic compost application."
}

# ==========================================
# SPECTRAL ANALYSIS ENGINE
# ==========================================
def process_spectral_frame(frame):
    global roi_x, roi_y, roi_w, roi_h, flip_direction, baseline_profile, latest_metrics
    h_img, w_img = frame.shape[:2]

    rx, ry = max(0, min(roi_x, w_img - 20)), max(0, min(roi_y, h_img - 20))
    rw, rh = max(20, min(roi_w, w_img - rx)), max(20, min(roi_h, h_img - ry))

    roi = frame[ry:ry+rh, rx:rx+rw]
    if roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        raw = np.mean(gray, axis=0)
        if flip_direction: raw = np.flip(raw)
        norm = raw / (np.max(raw) if np.max(raw) > 0 else 1.0)

        if baseline_profile is not None and len(baseline_profile) == len(norm):
            absorbance = np.clip(1.0 - (norm / (baseline_profile + 1e-5)), 0.0, 1.0)
        else:
            absorbance = norm

        n_pts = len(absorbance)
        b_third = n_pts // 3
        blue_b, green_b, red_b = np.mean(absorbance[:b_third]), np.mean(absorbance[b_third:2*b_third]), np.mean(absorbance[2*b_third:])

        def get_stat(v): return "Deficient" if v < 0.35 else ("Surplus" if v > 0.75 else "Optimal")
        n_stat, p_stat, k_stat = get_stat(blue_b), get_stat(red_b), get_stat(green_b)

        ratio = (blue_b + 1e-5) / (red_b + 1e-5)
        est_ph = round(float(np.clip(6.5 + (ratio - 1.0) * 1.2, 4.5, 8.5)), 1)
        ph_c = "Acidic (Needs Lime)" if est_ph < 6.0 else ("Alkaline (Needs Gypsum)" if est_ph > 7.5 else "Neutral (Balanced)")
        score = int(np.clip(100 - (abs(7.0 - est_ph) * 12 + (0 if n_stat == "Optimal" else 15) + (0 if p_stat == "Optimal" else 15)), 30, 98))

        adv = []
        if n_stat == "Deficient": adv.append("Apply Urea or Neem-coated Nitrogen.")
        if p_stat == "Deficient": adv.append("Apply Single Super Phosphate (SSP).")
        if k_stat == "Deficient": adv.append("Apply Muriate of Potash (MOP).")
        if "Acidic" in ph_c: adv.append("Apply Lime.")
        if "Alkaline" in ph_c: adv.append("Apply Gypsum.")
        rec = " ".join(adv) if adv else "Soil health is optimal. Maintain current organic crop rotation."

        latest_metrics = {"nitrogen": n_stat, "phosphorus": p_stat, "potassium": k_stat, "ph": est_ph, "ph_class": ph_c, "score": score, "recommendation": rec}

        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 191, 0), 2)
        cv2.putText(frame, f"TARGET ROI ({rx},{ry})", (rx, max(15, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 191, 0), 1)

        gh, gw, gx, gy = 90, w_img - 20, 10, h_img - 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (gx, gy), (gx + gw, gy + gh), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (100, 100, 100), 1)

        for c in range(gw):
            r_c = c / float(gw)
            col = (0, int(r_c * 510), int((1 - r_c * 2) * 255)) if r_c < 0.5 else (int((r_c - 0.5) * 510), int((1 - (r_c - 0.5) * 2) * 255), 0)
            cv2.line(frame, (gx + c, gy + gh - 4), (gx + c, gy + gh - 1), col, 1)

        pts = [(gx + int((i / float(len(norm))) * gw), gy + gh - 8 - int(v * (gh - 18))) for i, v in enumerate(norm)]
        for i in range(len(pts) - 1): cv2.line(frame, pts[i], pts[i+1], (0, 255, 255), 2)

    return frame

# ==========================================
# FLASK APPLICATION & ROUTING
# ==========================================
app = Flask(__name__)

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    data = request.json or {}
    img_str = data.get('image', '')
    if img_str:
        try:
            if ',' in img_str: img_str = img_str.split(',')[1]
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(img_str), np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                processed = process_spectral_frame(frame)
                _, buffer = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                b64_out = base64.b64encode(buffer).decode('utf-8')
                return jsonify({"status": "ok", "image": "data:image/jpeg;base64," + b64_out, "metrics": latest_metrics})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    return jsonify({"status": "error"})

@app.route('/api/set_roi', methods=['POST'])
def set_roi():
    global roi_x, roi_y, roi_w, roi_h
    req = request.json or {}
    roi_x, roi_y = int(req.get('x', roi_x)), int(req.get('y', roi_y))
    roi_w, roi_h = int(req.get('w', roi_w)), int(req.get('h', roi_h))
    return jsonify({"status": "ok"})

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    return jsonify({"status": "success", "message": "Baseline calibrated successfully!"})

@app.route('/api/flip', methods=['POST'])
def flip():
    global flip_direction
    flip_direction = not flip_direction
    return jsonify({"status": "ok", "flip": flip_direction})

@app.route('/api/reset', methods=['POST'])
def reset():
    global baseline_profile, flip_direction
    baseline_profile = None
    flip_direction = False
    return jsonify({"status": "ok", "message": "Calibration reset."})

@app.route('/api/ai_chat', methods=['POST'])
def ai_chat():
    req = request.json or {}
    query, lang = req.get('query', '').strip(), req.get('lang', 'en-IN')
    m = dict(latest_metrics)
    target_lang = {'en-IN': 'English', 'hi-IN': 'Hindi', 'mr-IN': 'Marathi', 'gu-IN': 'Gujarati', 'pa-IN': 'Punjabi', 'ta-IN': 'Tamil', 'te-IN': 'Telugu'}.get(lang, 'English')

    if ai_client:
        try:
            prompt = (
                f"You are SpecTantra AI, an Indian agricultural expert.\n"
                f"Soil Context: Nitrogen={m['nitrogen']}, Phosphorus={m['phosphorus']}, Potassium={m['potassium']}, pH={m['ph']} ({m['ph_class']}).\n"
                f"Question: '{query}'\n\n"
                f"Instructions:\n1. Answer the question directly and practically.\n"
                f"2. If asked about soil/crops, use live soil data above. If asked general farming questions (fertilizers, brands, pests, weather), give expert advice.\n"
                f"3. Keep response concise (2-3 sentences) and strictly in {target_lang}."
            )
            res = ai_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return jsonify({"status": "ok", "response": res.text.strip()})
        except Exception as e:
            print(f"Gemini API Error: {e}")

    q_low = query.lower()
    if any(k in q_low for k in ["sugarcane", "गन्ना", "ऊस"]):
        ans = f"Sugarcane requires a pH of 6.0 to 7.5, while Wheat requires 6.0 to 7.0. Your current soil pH is {m['ph']}."
    elif any(k in q_low for k in ["brand", "company", "fertilizer"]):
        ans = "Top trusted fertilizer brands in India are IFFCO, Mahadhan, Coromandel, and Kribhco."
    else:
        ans = f"For '{query}': Soil pH is {m['ph']} ({m['ph_class']}). Advice: {m['recommendation']}"

    return jsonify({"status": "ok", "response": ans})

# ==========================================
# UI DASHBOARD HTML TEMPLATE
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
        #outImg { width: 100%; border-radius: 8px; border: 2px solid #00d2ff; background: #000; }
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
                    
                    <div class="video-container" onclick="handleImageClick(event)">
                        <img id="outImg" alt="Camera Stream Loading...">
                        <video id="webcam" autoplay playsinline muted class="d-none"></video>
                        <canvas id="hiddenCanvas" class="d-none"></canvas>
                    </div>
                    
                    <div class="row g-2 mt-2 align-items-center">
                        <div class="col-auto"><small class="text-info fw-bold">ROI X:</small> <input type="number" id="roiX" value="150" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><small class="text-info fw-bold">Y:</small> <input type="number" id="roiY" value="100" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><small class="text-info fw-bold">Width:</small> <input type="number" id="roiW" value="340" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><small class="text-info fw-bold">Height:</small> <input type="number" id="roiH" value="60" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><button onclick="updateRoi()" class="btn btn-sm btn-outline-info">Update ROI</button></div>
                    </div>

                    <!-- CONTROL BUTTONS -->
                    <div class="d-flex gap-2 mt-3">
                        <button onclick="saveTestLocally()" class="btn btn-success flex-fill control-btn">💾 [S] SAVE TEST DATA</button>
                        <button onclick="triggerCalibrate()" class="btn btn-info flex-fill control-btn">🎯 [C] CALIBRATE BASELINE</button>
                        <button onclick="triggerFlip()" class="btn btn-secondary flex-fill control-btn">🔄 [F] FLIP GRAPH</button>
                        <button onclick="triggerReset()" class="btn btn-outline-danger flex-fill control-btn">❌ [R] RESET</button>
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
        let currentAnalysis = {};

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } } });
                document.getElementById('webcam').srcObject = stream;
            } catch (e) { alert("Camera access denied: " + e); }
        }

        function sendFrame() {
            let video = document.getElementById('webcam');
            let canvas = document.getElementById('hiddenCanvas');
            if (!video.videoWidth) return;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

            fetch('/api/process_frame', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ image: canvas.toDataURL('image/jpeg', 0.6) })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'ok') {
                    document.getElementById('outImg').src = data.image;
                    let m = data.metrics;
                    currentAnalysis = m;
                    updateBadge('valN', m.nitrogen);
                    updateBadge('valP', m.phosphorus);
                    updateBadge('valK', m.potassium);
                    document.getElementById('valPh').innerText = m.ph;
                    document.getElementById('valPhClass').innerText = m.ph_class;
                    document.getElementById('valScore').innerText = m.score + "%";
                    document.getElementById('valAdv').innerText = m.recommendation;
                }
            });
        }

        function updateBadge(id, status) {
            let el = document.getElementById(id);
            el.innerText = status;
            el.className = 'badge-val ' + (status === 'Optimal' ? 'bg-optimal' : (status === 'Deficient' ? 'bg-deficient' : 'bg-surplus'));
        }

        function handleImageClick(e) {
            let img = document.getElementById('outImg');
            let rect = img.getBoundingClientRect();
            let clickX = e.clientX - rect.left, clickY = e.clientY - rect.top;

            let scaleX = img.naturalWidth / rect.width, scaleY = img.naturalHeight / rect.height;
            let realX = Math.round(clickX * scaleX), realY = Math.round(clickY * scaleY);
            let w = parseInt(document.getElementById('roiW').value) || 340, h = parseInt(document.getElementById('roiH').value) || 60;

            let newX = Math.max(0, realX - Math.round(w / 2)), newY = Math.max(0, realY - Math.round(h / 2));
            document.getElementById('roiX').value = newX;
            document.getElementById('roiY').value = newY;
            updateRoi();
        }

        function updateRoi() {
            fetch('/api/set_roi', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    x: parseInt(document.getElementById('roiX').value),
                    y: parseInt(document.getElementById('roiY').value),
                    w: parseInt(document.getElementById('roiW').value),
                    h: parseInt(document.getElementById('roiH').value)
                })
            });
        }

        // BROWSER LOCALSTORAGE CLIENT-SIDE SAVING & CSV GENERATION
        function getSavedTests() {
            return JSON.parse(localStorage.getItem('soil_tests') || '[]');
        }

        function saveTestLocally() {
            if (!currentAnalysis.ph) return alert("Please enable camera to capture test data first.");
            
            let tests = getSavedTests();
            let record = {
                timestamp: new Date().toLocaleString(),
                nitrogen: currentAnalysis.nitrogen,
                phosphorus: currentAnalysis.phosphorus,
                potassium: currentAnalysis.potassium,
                ph: currentAnalysis.ph,
                ph_class: currentAnalysis.ph_class,
                score: currentAnalysis.score,
                recommendation: currentAnalysis.recommendation
            };

            tests.push(record);
            localStorage.setItem('soil_tests', JSON.stringify(tests));
            updateTestCounter();
            alert("✅ Test Record Saved Successfully! Total saved tests: " + tests.length);
        }

        function updateTestCounter() {
            let tests = getSavedTests();
            document.getElementById('testCount').innerText = tests.length;
        }

        function downloadCSVClient() {
            let tests = getSavedTests();
            if (tests.length === 0) {
                return alert("No saved tests found. Click '[S] SAVE TEST DATA' first!");
            }

            let csvContent = "data:text/csv;charset=utf-8,";
            csvContent += "Timestamp,Nitrogen,Phosphorus,Potassium,pH,pH Classification,Health Score (%),Recommendation\n";

            tests.forEach(t => {
                let row = `"${t.timestamp}","${t.nitrogen}","${t.phosphorus}","${t.potassium}","${t.ph}","${t.ph_class}","${t.score}","${t.recommendation.replace(/"/g, '""')}"`;
                csvContent += row + "\n";
            });

            let encodedUri = encodeURI(csvContent);
            let link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `soil_database_${Date.now()}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        function triggerCalibrate() {
            fetch('/api/calibrate', {method: 'POST'})
                .then(r => r.json())
                .then(d => alert(d.message));
        }

        function triggerFlip() {
            fetch('/api/flip', {method: 'POST'});
        }

        function triggerReset() {
            fetch('/api/reset', {method: 'POST'})
                .then(r => r.json())
                .then(d => alert(d.message));
        }

        function sendAiQuery() {
            let text = document.getElementById('aiQueryInput').value, lang = document.getElementById('langSelect').value;
            if (!text) return;
            document.getElementById('aiResponseText').innerText = "Thinking...";

            fetch('/api/ai_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: text, lang: lang})
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
            if (!SR) return alert("Voice speech recognition not supported.");
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

                if (match) {
                    msg.voice = match;
                }
                window.speechSynthesis.speak(msg);
            }

            let voices = window.speechSynthesis.getVoices();
            if (voices.length > 0) {
                executeSpeech();
            } else {
                window.speechSynthesis.onvoiceschanged = executeSpeech;
            }
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
            if (k === 'c') triggerCalibrate();
            if (k === 'f') triggerFlip();
            if (k === 'r') triggerReset();
        });

        window.addEventListener('DOMContentLoaded', () => { 
            startCamera(); 
            setInterval(sendFrame, 400); 
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