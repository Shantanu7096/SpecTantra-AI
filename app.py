import os
import sys
import cv2
import time
import csv
import json
import threading
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request, send_file, send_from_directory
from google import genai
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION & PERSISTENCE
# ==========================================
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "soil_database.csv")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
SAVED_TESTS_DIR = os.path.join(BASE_DIR, "saved_tests")
os.makedirs(SAVED_TESTS_DIR, exist_ok=True)

# GEMINI API KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ai_client = None
if GEMINI_API_KEY:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini AI Client initialized successfully from .env!")
    except Exception as e:
        print(f"⚠️ Gemini API initialization warning: {e}")
else:
    print("⚠️ GEMINI_API_KEY not found in .env file!")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"camera_source": "0"}

def save_config(data):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Failed to save config: {e}")

config = load_config()
camera_source = config.get("camera_source", "0")

# ==========================================
# GLOBAL STATE
# ==========================================
state_lock = threading.Lock()

roi_x = 150
roi_y = 100
roi_w = 340
roi_h = 60

flip_direction = False
baseline_profile = None

latest_metrics = {
    "nitrogen": "Optimal",
    "nitrogen_val": 0.55,
    "phosphorus": "Optimal",
    "phosphorus_val": 0.52,
    "potassium": "Optimal",
    "potassium_val": 0.58,
    "ph": 6.8,
    "ph_class": "Neutral (Balanced)",
    "score": 92,
    "recommendation": "Soil health is optimal. Maintain balanced organic compost application.",
    "is_calibrated": False
}

# ==========================================
# SPECTRAL ANALYSIS ENGINE
# ==========================================
def process_spectral_frame(frame):
    global roi_x, roi_y, roi_w, roi_h, flip_direction, baseline_profile, latest_metrics

    h_img, w_img = frame.shape[:2]
    
    with state_lock:
        rx = max(0, min(roi_x, w_img - 20))
        ry = max(0, min(roi_y, h_img - 20))
        rw = max(20, min(roi_w, w_img - rx))
        rh = max(20, min(roi_h, h_img - ry))

    roi = frame[ry:ry+rh, rx:rx+rw]
    if roi.size == 0:
        return frame

    # Extract Spectral Profile
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    raw_profile = np.mean(gray_roi, axis=0)

    if flip_direction:
        raw_profile = np.flip(raw_profile)

    max_val = np.max(raw_profile) if np.max(raw_profile) > 0 else 1.0
    norm_profile = raw_profile / max_val

    with state_lock:
        if baseline_profile is not None and len(baseline_profile) == len(norm_profile):
            absorbance = np.clip(1.0 - (norm_profile / (baseline_profile + 1e-5)), 0.0, 1.0)
            is_calibrated = True
        else:
            absorbance = norm_profile
            is_calibrated = False

    n_pts = len(absorbance)
    b_third = n_pts // 3
    
    blue_band = np.mean(absorbance[:b_third])
    green_band = np.mean(absorbance[b_third:2*b_third])
    red_band = np.mean(absorbance[2*b_third:])

    n_status, n_val = classify_nutrient(blue_band)
    p_status, p_val = classify_nutrient(red_band)
    k_status, k_val = classify_nutrient(green_band)

    ratio = (blue_band + 1e-5) / (red_band + 1e-5)
    est_ph = round(float(np.clip(6.5 + (ratio - 1.0) * 1.2, 4.5, 8.5)), 1)
    
    if est_ph < 6.0:
        ph_class = "Acidic (Needs Lime)"
    elif est_ph > 7.5:
        ph_class = "Alkaline (Needs Gypsum)"
    else:
        ph_class = "Neutral (Balanced)"

    score = int(np.clip(100 - (abs(7.0 - est_ph) * 12 + (0 if n_status == "Optimal" else 15) + (0 if p_status == "Optimal" else 15)), 30, 98))
    recommendation = generate_advisory(n_status, p_status, k_status, ph_class)

    with state_lock:
        latest_metrics = {
            "nitrogen": n_status,
            "nitrogen_val": round(float(n_val), 2),
            "phosphorus": p_status,
            "phosphorus_val": round(float(p_val), 2),
            "potassium": k_status,
            "potassium_val": round(float(k_val), 2),
            "ph": est_ph,
            "ph_class": ph_class,
            "score": score,
            "recommendation": recommendation,
            "is_calibrated": is_calibrated
        }

    # Overlay Target Box
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 191, 0), 2)
    cv2.putText(frame, f"TARGET ROI ({rx},{ry},{rw}x{rh})", (rx, max(15, ry - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 191, 0), 1)

    # Render Spectrum Graph
    gh, gw, gx, gy = 110, w_img - 20, 10, h_img - 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (gx, gy), (gx + gw, gy + gh), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (100, 100, 100), 1)

    for c in range(gw):
        col_ratio = c / float(gw)
        if col_ratio < 0.5:
            r, g, b = 0, int(col_ratio * 2 * 255), int((1 - col_ratio * 2) * 255)
        else:
            r, g, b = int((col_ratio - 0.5) * 2 * 255), int((1 - (col_ratio - 0.5) * 2) * 255), 0
        cv2.line(frame, (gx + c, gy + gh - 6), (gx + c, gy + gh - 1), (b, g, r), 1)

    pts = []
    for i, val in enumerate(norm_profile):
        px = gx + int((i / float(len(norm_profile))) * gw)
        py = gy + gh - 10 - int(val * (gh - 25))
        pts.append((px, py))

    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], (0, 255, 255), 2)

    cal_tag = "CALIBRATED" if is_calibrated else "RAW (Press 'C' to Calibrate)"
    summary_txt = f"N:{n_status} | P:{p_status} | K:{k_status} | pH:{est_ph} | {cal_tag}"
    cv2.rectangle(frame, (0, 0), (w_img, 28), (0, 0, 0), -1)
    cv2.putText(frame, summary_txt, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame

def classify_nutrient(val):
    if val < 0.35: return "Deficient", val
    if val > 0.75: return "Surplus", val
    return "Optimal", val

def generate_advisory(n, p, k, ph_c):
    adv = []
    if n == "Deficient": adv.append("Apply Urea or Neem-coated Nitrogen.")
    if p == "Deficient": adv.append("Apply Single Super Phosphate (SSP).")
    if k == "Deficient": adv.append("Apply Muriate of Potash (MOP).")
    if "Acidic" in ph_c: adv.append("Apply Agricultural Lime.")
    if "Alkaline" in ph_c: adv.append("Apply Gypsum.")
    if not adv: adv.append("Soil parameters are optimal. Maintain current organic crop rotation.")
    return " ".join(adv)

# ==========================================
# CAMERA THREAD CONTROLLER
# ==========================================
class CameraStream:
    def __init__(self):
        self.cap = None
        self.running = False
        self.current_frame = None
        self.lock = threading.Lock()

    def start(self, source):
        self.stop()
        with self.lock:
            if str(source).isdigit():
                src = int(source)
                if sys.platform.startswith('win'):
                    self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(src)
            else:
                src = source
                self.cap = cv2.VideoCapture(src)
            self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    processed = process_spectral_frame(frame)
                    with self.lock:
                        self.current_frame = processed
                else:
                    time.sleep(0.05)
            else:
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

camera = CameraStream()
camera.start(camera_source)

# ==========================================
# FLASK WEB APP & ROUTING
# ==========================================
app = Flask(__name__)

def generate_mjpeg_stream():
    while True:
        frame = camera.get_frame()
        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "CONNECTING TO CAMERA STREAM...", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            frame = blank

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/get_analysis')
def get_analysis():
    with state_lock:
        data = dict(latest_metrics)
        data["roi"] = {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h}
        data["flip"] = flip_direction
        data["camera_source"] = camera_source
    return jsonify(data)

@app.route('/api/set_camera_ip', methods=['POST'])
def set_camera_ip():
    global camera_source
    req = request.json or {}
    src = req.get('source', '0').strip()
    
    camera_source = src
    save_config({"camera_source": camera_source})
    camera.start(camera_source)
    
    return jsonify({"status": "ok", "source": camera_source, "message": f"Camera set to: {camera_source}"})

@app.route('/api/set_roi', methods=['POST'])
def set_roi():
    global roi_x, roi_y, roi_w, roi_h
    req = request.json or {}
    with state_lock:
        roi_x = int(req.get('x', roi_x))
        roi_y = int(req.get('y', roi_y))
        roi_w = int(req.get('w', roi_w))
        roi_h = int(req.get('h', roi_h))
    return jsonify({"status": "ok", "roi": {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h}})

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global baseline_profile
    frame = camera.get_frame()
    if frame is not None:
        h_img, w_img = frame.shape[:2]
        rx, ry, rw, rh = roi_x, roi_y, roi_w, roi_h
        roi = frame[ry:ry+rh, rx:rx+rw]
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            profile = np.mean(gray_roi, axis=0)
            if flip_direction: profile = np.flip(profile)
            max_v = np.max(profile) if np.max(profile) > 0 else 1.0
            with state_lock:
                baseline_profile = profile / max_v
            return jsonify({"status": "success", "message": "Baseline calibrated successfully!"})
    return jsonify({"status": "error", "message": "Calibration failed. Ensure camera stream is visible."})

@app.route('/api/flip', methods=['POST'])
def flip():
    global flip_direction
    with state_lock:
        flip_direction = not flip_direction
    return jsonify({"status": "ok", "flip": flip_direction})

@app.route('/api/reset', methods=['POST'])
def reset():
    global baseline_profile, flip_direction
    with state_lock:
        baseline_profile = None
        flip_direction = False
    return jsonify({"status": "ok", "message": "Calibration reset."})

@app.route('/api/save_test', methods=['POST'])
def save_test():
    frame = camera.get_frame()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename = f"test_{timestamp_str}.png"
    img_path = os.path.join(SAVED_TESTS_DIR, img_filename)

    if frame is not None:
        cv2.imwrite(img_path, frame)

    with state_lock:
        m = dict(latest_metrics)

    headers = ["Timestamp", "Nitrogen Status", "Phosphorus Status", "Potassium Status",
               "Estimated pH", "pH Classification", "Soil Health Score (%)", "Recommendation", "Image_File_Path"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        m["nitrogen"], m["phosphorus"], m["potassium"],
        m["ph"], m["ph_class"], m["score"],
        m["recommendation"], os.path.relpath(img_path, BASE_DIR)
    ]

    target_csv = CSV_FILE
    try:
        file_exists = os.path.exists(target_csv) and os.path.getsize(target_csv) > 0
        with open(target_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Error saving to CSV: {e}")

    return jsonify({"status": "success", "message": f"Saved record to {os.path.basename(target_csv)}!"})

@app.route('/download/csv')
def download_csv():
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        return send_file(CSV_FILE, as_attachment=True, download_name="soil_database.csv")
    return jsonify({"status": "error", "message": "No CSV file created yet."}), 404

@app.route('/saved_tests/<filename>')
def serve_saved_test_image(filename):
    return send_from_directory(SAVED_TESTS_DIR, filename)

@app.route('/api/ai_chat', methods=['POST'])
def ai_chat():
    req = request.json or {}
    user_query = req.get('query', '').strip()
    lang = req.get('lang', 'en-IN')

    with state_lock:
        m = dict(latest_metrics)

    q_lower = user_query.lower()

    lang_names = {
        'en-IN': 'English',
        'hi-IN': 'Hindi',
        'mr-IN': 'Marathi',
        'gu-IN': 'Gujarati',
        'pa-IN': 'Punjabi',
        'ta-IN': 'Tamil',
        'te-IN': 'Telugu'
    }
    target_lang = lang_names.get(lang, 'English')

    # 1. LIVE GEMINI AI ENGINE
    if ai_client and GEMINI_API_KEY not in ["YOUR_ACTUAL_GEMINI_API_KEY_HERE", "", None]:
        try:
            system_prompt = (
                f"You are SpecTantra AI, an expert agricultural advisor for Indian farmers.\n"
                f"Live Soil Analysis Context:\n"
                f"- Nitrogen: {m['nitrogen']}\n"
                f"- Phosphorus: {m['phosphorus']}\n"
                f"- Potassium: {m['potassium']}\n"
                f"- Soil pH: {m['ph']} ({m['ph_class']})\n"
                f"- Quality Score: {m['score']}%\n\n"
                f"Farmer Question: '{user_query}'\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Answer the farmer's question directly in sentence 1.\n"
                f"2. Evaluate crop benefits, fertilizers, optimal soil conditions, or general farming queries accurately.\n"
                f"3. Keep response concise (2 to 3 sentences).\n"
                f"4. MANDATORY: Respond strictly in {target_lang}."
            )
            
            response = ai_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=system_prompt,
            )
            return jsonify({"status": "ok", "response": response.text.strip()})
        except Exception as e:
            print(f"⚠️ Gemini API Error: {e}")

    # 2. ENHANCED OFFLINE FALLBACK ENGINE
    if any(k in q_lower for k in ["wheat", "गेहूं", "गहू"]):
        ans_en = f"Wheat provides excellent crop yields in balanced soil. Your current pH of {m['ph']} is optimal for wheat cultivation."
        ans_hi = f"गेहूं की फसल इस मिट्टी के लिए बहुत लाभदायक है। आपका वर्तमान pH {m['ph']} गेहूं की बेहतर पैदावार के लिए अनुकूल है।"
        ans_mr = f"गहू पीक या मातीसाठी अत्यंत फायदेशीर आहे. तुमचा सध्याचा pH {m['ph']} गव्हाच्या उत्तम उत्पादनासाठी योग्य आहे."

    elif any(k in q_lower for k in ["sugarcane", "गन्ना", "ऊस"]):
        ans_en = f"Sugarcane grows best in soil with pH 6.0 to 7.5. Your soil pH of {m['ph']} is suitable."
        ans_hi = f"गन्ने की फसल के लिए pH 6.0 से 7.5 उत्तम रहता है। आपकी मिट्टी का pH {m['ph']} इसके अनुकूल है।"
        ans_mr = f"उसाच्या पिकासाठी pH 6.0 ते 7.5 उत्तम असतो. तुमच्या मातीचा pH {m['ph']} योग्य आहे."

    elif any(k in q_lower for k in ["brand", "company", "fertilizer", "खाद"]):
        ans_en = "Top trusted Indian fertilizer brands include IFFCO, Mahadhan, Coromandel, and Kribhco."
        ans_hi = "भारत में सबसे भरोसेमंद खाद ब्रांड इफ्को (IFFCO), महाधन (Mahadhan) और कोरोमंडल हैं।"
        ans_mr = "भारतातील प्रमुख खत ब्रँड इफको (IFFCO), महाधन (Mahadhan) आणि कोरोमंडल आहेत."

    else:
        ans_en = f"For query '{user_query}': Current soil pH is {m['ph']} ({m['ph_class']}). Advice: {m['recommendation']}"
        ans_hi = f"आपके प्रश्न के लिए: मिट्टी का pH {m['ph']} है। सलाह: {m['recommendation']}"
        ans_mr = f"तुमच्या प्रश्नासाठी: मातीचा pH {m['ph']} आहे. सल्ला: {m['recommendation']}"

    if lang == 'hi-IN': resp_text = ans_hi
    elif lang == 'mr-IN': resp_text = ans_mr
    else: resp_text = ans_en

    return jsonify({"status": "ok", "response": resp_text})

# ==========================================
# DASHBOARD INTERFACE HTML
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
        body { background-color: #0b1329; color: #f8fafc; font-family: 'Segoe UI', system-ui, sans-serif; }
        .card { background-color: #131e3a; border: 1px solid #1e2d5a; border-radius: 12px; }
        .video-container { position: relative; width: 100%; cursor: crosshair; }
        .video-container img { width: 100%; border-radius: 8px; border: 2px solid #00d2ff; min-height: 280px; background: #000; }
        .badge-val { font-size: 1.1rem; font-weight: 700; padding: 8px 16px; border-radius: 6px; display: inline-block; width: 100%; }
        .bg-optimal { background-color: #10b981; color: #ffffff; }
        .bg-deficient { background-color: #ef4444; color: #ffffff; }
        .bg-surplus { background-color: #f59e0b; color: #ffffff; }
        .metric-label { font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; display: block; }
        .control-btn { font-weight: 600; text-transform: uppercase; font-size: 0.85rem; }
    </style>
</head>
<body class="p-3">
    <div class="container-fluid">
        <!-- TOP NAV BAR -->
        <div class="d-flex justify-content-between align-items-center pb-3 mb-3 border-bottom border-secondary">
            <h3 class="m-0 text-info fw-bold">🔬 SpecTantra AI <span class="fs-6 text-light fw-normal">| Local System</span></h3>
            <div class="d-flex gap-2 align-items-center">
                <select id="camSelect" class="form-select form-select-sm bg-dark text-light border-secondary" style="width: auto;" onchange="handleCamSelectChange(this.value)">
                    <option value="0">Camera 0 (Laptop/Front)</option>
                    <option value="1">Camera 1 (External/Rear)</option>
                    <option value="custom">IP Stream URL...</option>
                </select>
                <input type="text" id="camIpInput" class="form-control form-control-sm bg-dark text-light border-secondary d-none" placeholder="http://192.168.x.x:8080/video" style="width: 220px;">
                <button id="camBtn" onclick="startCamera()" class="btn btn-sm btn-success fw-bold">📷 Start / Enable Camera</button>
            </div>
        </div>

        <div class="row g-3">
            <!-- LIVE VIDEO & GRAPH -->
            <div class="col-lg-7">
                <div class="card p-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h5 class="m-0 text-warning fw-bold">📹 Live Spectral Stream & Graph</h5>
                        <small class="text-muted">Click canvas to position Target ROI Box</small>
                    </div>
                    
                    <div class="video-container" onclick="handleCanvasClick(event)">
                    <canvas id="displayCanvas"></canvas>
                    <video id="webcam" autoplay playsinline muted style="position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none;"></video>
                    </div>
                    
                    <div class="row g-2 mt-2 align-items-center">
                        <div class="col-auto"><small class="text-info fw-bold">ROI X:</small> <input type="number" id="roiX" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><small class="text-info fw-bold">Y:</small> <input type="number" id="roiY" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><small class="text-info fw-bold">Width:</small> <input type="number" id="roiW" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><small class="text-info fw-bold">Height:</small> <input type="number" id="roiH" class="form-control form-control-sm bg-dark text-light border-secondary" style="width:75px;"></div>
                        <div class="col-auto"><button onclick="applyRoiInputs()" class="btn btn-sm btn-outline-info">Update ROI</button></div>
                    </div>

                    <!-- CONTROL BUTTONS -->
                    <div class="d-flex gap-2 mt-3">
                        <button onclick="triggerSave()" class="btn btn-success flex-fill control-btn">💾 [S] SAVE TEST DATA</button>
                        <button onclick="triggerCalibrate()" class="btn btn-info flex-fill control-btn">🎯 [C] CALIBRATE BASELINE</button>
                        <button onclick="triggerFlip()" class="btn btn-secondary flex-fill control-btn">🔄 [F] FLIP GRAPH</button>
                        <button onclick="triggerReset()" class="btn btn-outline-danger flex-fill control-btn">❌ [R] RESET</button>
                    </div>
                </div>
            </div>

            <!-- ANALYTICS & AI ASSISTANT -->
            <div class="col-lg-5">
                <div class="card p-3 mb-3">
                    <h5 class="text-success fw-bold mb-3">📊 Real-Time Soil Analysis</h5>
                    
                    <div class="row g-2 text-center mb-3">
                        <div class="col-4">
                            <div class="p-2 border border-secondary rounded bg-dark">
                                <span class="metric-label">Nitrogen (N)</span>
                                <span id="valN" class="badge-val bg-optimal">--</span>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="p-2 border border-secondary rounded bg-dark">
                                <span class="metric-label">Phosphorus (P)</span>
                                <span id="valP" class="badge-val bg-optimal">--</span>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="p-2 border border-secondary rounded bg-dark">
                                <span class="metric-label">Potassium (K)</span>
                                <span id="valK" class="badge-val bg-optimal">--</span>
                            </div>
                        </div>
                    </div>

                    <div class="row g-2 text-center mb-3">
                        <div class="col-6">
                            <div class="p-2 border border-secondary rounded bg-dark">
                                <span class="metric-label">Soil pH</span>
                                <h3 id="valPh" class="m-0 text-info fw-bold">--</h3>
                                <small id="valPhClass" class="text-warning fw-bold">--</small>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="p-2 border border-secondary rounded bg-dark">
                                <span class="metric-label">Health Index</span>
                                <h3 id="valScore" class="m-0 text-success fw-bold">--%</h3>
                                <small class="text-light">Quality Score</small>
                            </div>
                        </div>
                    </div>

                    <div class="p-3 bg-dark rounded border border-secondary">
                        <small class="text-warning fw-bold d-block mb-1">💡 Advisory:</small>
                        <p id="valAdv" class="m-0 small text-light">Awaiting baseline calibration...</p>
                    </div>
                </div>

                <!-- MULTILINGUAL AI ASSISTANT -->
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
                        <button onclick="startVoiceRecognition()" class="btn btn-outline-warning">🎙️ Speak</button>
                        <button onclick="sendAiQuery()" class="btn btn-info fw-bold">Ask Gemini</button>
                    </div>

                    <div class="p-3 bg-dark rounded border border-secondary" style="min-height: 85px;">
                        <small class="text-info fw-bold d-block mb-1">Gemini AI Response:</small>
                        <p id="aiResponseText" class="m-0 small text-light">Select language and ask a question...</p>
                    </div>

                    <div class="d-flex gap-2 mt-3">
                        <button onclick="shareWhatsApp()" class="btn btn-sm btn-outline-success flex-fill">💬 WhatsApp</button>
                        <button onclick="shareEmail()" class="btn btn-sm btn-outline-primary flex-fill">✉️ Email</button>
                        <a href="/download/csv" class="btn btn-sm btn-outline-warning flex-fill" target="_blank">📥 Download CSV</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentAnalysis = {};
        
        let cameraActive = false;

function drawPlaceholder() {
    const canvas = document.getElementById('displayCanvas');
    if (!canvas) return;
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = "#050b18";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#00d2ff";
    ctx.font = "bold 18px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("📷 CLICK 'START / ENABLE CAMERA' BUTTON ABOVE", canvas.width / 2, canvas.height / 2 - 10);
    ctx.fillStyle = "#a0aec0";
    ctx.font = "14px sans-serif";
    ctx.fillText("Grant camera permissions when prompted by your browser.", canvas.width / 2, canvas.height / 2 + 20);
}

async function startCamera() {
    const video = document.getElementById('webcam');
    let stream = null;

    const configs = [
        { video: { facingMode: { ideal: "environment" } } },
        { video: { facingMode: "user" } },
        { video: true }
    ];

    for (let cfg of configs) {
        try {
            stream = await navigator.mediaDevices.getUserMedia(cfg);
            if (stream) break;
        } catch (e) {
            console.warn("Camera constraint mode failed:", cfg, e);
        }
    }

    if (!stream) {
        alert("Camera access denied or unavailable. Check browser permissions.");
        return;
    }

    video.srcObject = stream;

    const onVideoReady = () => {
        cameraActive = true;
        const btn = document.getElementById('camBtn');
        if (btn) {
            btn.className = "btn btn-sm btn-outline-success fw-bold";
            btn.innerText = "✅ Camera Active";
        }
        requestAnimationFrame(renderLoop);
    };

    video.onloadedmetadata = onVideoReady;
    video.onloadeddata = onVideoReady;
    video.play().catch(e => console.error("Video play error:", e));
}

function renderLoop() {
    if (!cameraActive) return;

    const video = document.getElementById('webcam');
    const canvas = document.getElementById('displayCanvas');

    if (video && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        // Match canvas dimensions to webcam feed
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        const ctx = canvas.getContext('2d');

        // 1. DRAW LIVE SMOOTH VIDEO FRAME
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        try {
            // 2. SAFE ROI CLAMPING
            const rx = Math.max(0, Math.min(roi.x, canvas.width - 20));
            const ry = Math.max(0, Math.min(roi.y, canvas.height - 20));
            const rw = Math.max(20, Math.min(roi.w, canvas.width - rx));
            const rh = Math.max(20, Math.min(roi.h, canvas.height - ry));

            // 3. DRAW BLUE TARGET ROI BOX
            ctx.strokeStyle = "#00d2ff";
            ctx.lineWidth = 3;
            ctx.strokeRect(rx, ry, rw, rh);
            ctx.fillStyle = "#00d2ff";
            ctx.font = "bold 14px sans-serif";
            ctx.textAlign = "left";
            ctx.fillText(`TARGET ROI (${rx},${ry},${rw}x${rh})`, rx, Math.max(18, ry - 8));

            // 4. EXTRACT SPECTRAL DATA
            if (rw > 0 && rh > 0) {
                const imgData = ctx.getImageData(rx, ry, rw, rh);
                const data = imgData.data;

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

                // Partition Spectral Bands (N, P, K)
                const bThird = Math.floor(rw / 3);
                let blueBand = 0, greenBand = 0, redBand = 0;

                for (let i = 0; i < bThird; i++) blueBand += absorbance[i];
                for (let i = bThird; i < 2 * bThird; i++) greenBand += absorbance[i];
                for (let i = bThird * 2; i < rw; i++) redBand += absorbance[i];

                blueBand /= Math.max(1, bThird);
                greenBand /= Math.max(1, bThird);
                redBand /= Math.max(1, rw - 2 * bThird);

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

                // 5. DRAW RAINBOW SPECTRAL GRAPH OVERLAY
                const gh = 100, gw = canvas.width - 20, gx = 10, gy = canvas.height - 110;
                ctx.fillStyle = "rgba(15, 15, 15, 0.85)";
                ctx.fillRect(gx, gy, gw, gh);
                ctx.strokeStyle = "#00d2ff";
                ctx.lineWidth = 1;
                ctx.strokeRect(gx, gy, gw, gh);

                for (let c = 0; c < gw; c++) {
                    let rC = c / gw;
                    let color = rC < 0.5 
                        ? `rgb(0, ${Math.floor(rC * 510)}, ${Math.floor((1 - rC * 2) * 255)})`
                        : `rgb(${Math.floor((rC - 0.5) * 510)}, ${Math.floor((1 - (rC - 0.5) * 2) * 255)}, 0)`;
                    ctx.fillStyle = color;
                    ctx.fillRect(gx + c, gy + gh - 6, 1, 5);
                }

                ctx.beginPath();
                ctx.strokeStyle = "#ffff00";
                ctx.lineWidth = 2;
                for (let i = 0; i < rw; i++) {
                    let px = gx + Math.floor((i / rw) * gw);
                    let py = gy + gh - 10 - Math.floor(norm[i] * (gh - 25));
                    if (i === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.stroke();
            }
        } catch (err) {
            console.error("Frame processing notice:", err);
        }
    }

    // Always schedule next frame for continuous 60 FPS animation
    requestAnimationFrame(renderLoop);
}

        function fetchAnalysis() {
            fetch('/api/get_analysis')
                .then(res => res.json())
                .then(data => {
                    currentAnalysis = data;
                    
                    updateBadge('valN', data.nitrogen);
                    updateBadge('valP', data.phosphorus);
                    updateBadge('valK', data.potassium);

                    document.getElementById('valPh').innerText = data.ph;
                    document.getElementById('valPhClass').innerText = data.ph_class;
                    document.getElementById('valScore').innerText = data.score + "%";
                    document.getElementById('valAdv').innerText = data.recommendation;

                    if (document.activeElement.id !== 'camIpInput' && data.camera_source) {
                        document.getElementById('camIpInput').value = data.camera_source;
                    }

                    if (document.activeElement.tagName !== 'INPUT') {
                        document.getElementById('roiX').value = data.roi.x;
                        document.getElementById('roiY').value = data.roi.y;
                        document.getElementById('roiW').value = data.roi.w;
                        document.getElementById('roiH').value = data.roi.h;
                    }
                });
        }

        function updateBadge(id, status) {
            let el = document.getElementById(id);
            el.innerText = status;
            el.className = 'badge-val ' + (status === 'Optimal' ? 'bg-optimal' : (status === 'Deficient' ? 'bg-deficient' : 'bg-surplus'));
        }

        function handleVideoClick(e) {
            let img = document.getElementById('streamImg');
            let rect = img.getBoundingClientRect();
            let clickX = e.clientX - rect.left;
            let clickY = e.clientY - rect.top;

            let scaleX = img.naturalWidth / rect.width;
            let scaleY = img.naturalHeight / rect.height;

            let realX = Math.round(clickX * scaleX);
            let realY = Math.round(clickY * scaleY);

            let w = parseInt(document.getElementById('roiW').value) || 340;
            let h = parseInt(document.getElementById('roiH').value) || 60;

            let newX = Math.max(0, realX - Math.round(w / 2));
            let newY = Math.max(0, realY - Math.round(h / 2));

            postRoi(newX, newY, w, h);
        }

        function applyRoiInputs() {
            let x = parseInt(document.getElementById('roiX').value);
            let y = parseInt(document.getElementById('roiY').value);
            let w = parseInt(document.getElementById('roiW').value);
            let h = parseInt(document.getElementById('roiH').value);
            postRoi(x, y, w, h);
        }

        function postRoi(x, y, w, h) {
            fetch('/api/set_roi', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x: x, y: y, w: w, h: h})
            }).then(() => fetchAnalysis());
        }

        function handleCamSelectChange(val) {
            let inputEl = document.getElementById('camIpInput');
            if (val === 'custom') {
                inputEl.classList.remove('d-none');
            } else {
                inputEl.classList.add('d-none');
                inputEl.value = val;
                updateCameraIp();
            }
        }

        function updateCameraIp() {
            let src = document.getElementById('camIpInput').value;
            fetch('/api/set_camera_ip', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({source: src})
            }).then(r => r.json()).then(d => alert(d.message));
        }

        function triggerSave() {
            fetch('/api/save_test', {method: 'POST'})
                .then(r => r.json())
                .then(d => alert(d.message));
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
            let text = document.getElementById('aiQueryInput').value;
            let lang = document.getElementById('langSelect').value;
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

        function startVoiceRecognition() {
            let lang = document.getElementById('langSelect').value;
            let SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SR) return alert("Speech recognition not supported in this browser.");
            let rec = new SR();
            rec.lang = lang;
            rec.onresult = e => { 
                document.getElementById('aiQueryInput').value = e.results[0][0].transcript; 
                sendAiQuery(); 
            };
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
            if (k === 's') triggerSave();
            if (k === 'c') triggerCalibrate();
            if (k === 'f') triggerFlip();
            if (k === 'r') triggerReset();
        });

        window.addEventListener('DOMContentLoaded', () => { 
    drawPlaceholder();
    updateTestCounter();
    startCamera();
        });
        setInterval(fetchAnalysis, 1000);
        fetchAnalysis();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    print("=" * 65)
    print("🚀 SpecTantra AI Local Server Running")
    print("👉 Open Dashboard: http://localhost:5000")
    print("=" * 65)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)