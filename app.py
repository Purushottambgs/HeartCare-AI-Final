from flask import Flask, render_template, request, send_file, redirect, url_for, Response, session, jsonify
#from tensorflow.keras.models import load_model
import struct
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors as rl_colors
import io
import sqlite3
from datetime import datetime
import os
import joblib
import joblib
import functools
import json
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import scipy.stats as sci_stats
from scipy.signal import find_peaks
import csv

app = Flask(__name__)
app.secret_key = 'heartcare_ai_secret_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained models
model = joblib.load("heart_model.pkl")

model = joblib.load("heart_model.pkl")
ecg_model = joblib.load("ecg_binary_model.pkl")
# from tensorflow.keras.models import load_model
# model = load_model("ecg_model.keras")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# =========================
# DATABASE SETUP
# =========================
def init_db():
    conn = sqlite3.connect("heartcare.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT DEFAULT 'Anonymous',
            age REAL,
            gender INTEGER,
            height REAL,
            weight REAL,
            ap_hi REAL,
            ap_lo REAL,
            cholesterol INTEGER,
            gluc INTEGER,
            smoke INTEGER,
            alco INTEGER,
            active INTEGER,
            bmi REAL,
            map_value REAL,
            pulse_pressure REAL,
            probability REAL,
            risk_level TEXT,
            advice TEXT,
            created_at TEXT
        )
    """)

    # Add patient_name column if it doesn't exist (for existing DBs)
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN patient_name TEXT DEFAULT 'Anonymous'")
    except Exception:
        pass

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ecg_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT DEFAULT 'Anonymous',
            filename TEXT,
            result TEXT,
            result_label TEXT,
            confidence REAL,
            num_peaks INTEGER,
            rr_mean REAL,
            rr_std REAL,
            mean_signal REAL,
            std_signal REAL,
            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
    """)

    # Add new columns safely
    for col_add in [
        "ALTER TABLE predictions ADD COLUMN user_id INTEGER DEFAULT NULL",
        "ALTER TABLE ecg_predictions ADD COLUMN user_id INTEGER DEFAULT NULL",
        "ALTER TABLE predictions ADD COLUMN heart_score INTEGER DEFAULT NULL",
        "ALTER TABLE predictions ADD COLUMN heart_age INTEGER DEFAULT NULL",
    ]:
        try:
            cursor.execute(col_add)
        except Exception:
            pass

    # Migrate legacy BLOB probability values (stored as numpy float32 binary)
    # to proper SQLite REAL so Python sum/avg work correctly.
    try:
        cursor.execute("SELECT id, probability FROM predictions WHERE typeof(probability) = 'blob'")
        blob_rows = cursor.fetchall()
        for row_id, prob_bytes in blob_rows:
            if prob_bytes and len(prob_bytes) == 4:
                prob_float = round(float(struct.unpack('<f', prob_bytes)[0]), 2)
                cursor.execute("UPDATE predictions SET probability = ? WHERE id = ?",
                               (prob_float, row_id))
    except Exception:
        pass

    conn.commit()
    conn.close()


def _safe_float(val, default=0.0):
    """Convert any SQLite value (REAL, INTEGER, BLOB float32, TEXT) to Python float."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, bytes) and len(val) == 4:
        try:
            return round(float(struct.unpack('<f', val)[0]), 2)
        except Exception:
            return default
    try:
        return float(val)
    except Exception:
        return default


# =========================
# USER AUTH
# =========================
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


def create_user(username, email, password):
    conn = sqlite3.connect("heartcare.db")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?,?,?,?)",
            (username.strip(), email.strip(),
             generate_password_hash(password),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        uid = cursor.lastrowid
        conn.close()
        return uid, None
    except sqlite3.IntegrityError:
        conn.close()
        return None, "Username already taken. Please choose another."


def verify_user(username, password):
    conn = sqlite3.connect("heartcare.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    user = cursor.fetchone()
    conn.close()
    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None


def get_user(user_id):
    conn = sqlite3.connect("heartcare.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_predictions(user_id, limit=50):
    conn = sqlite3.connect("heartcare.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return rows


# =========================
# SHAP (permutation-based explainability)
# =========================
def _build_features_array(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    """Re-build the 24-feature array used by the heart model."""
    height_m = height / 100
    BMI      = weight / (height_m ** 2)
    MAP      = (2 * ap_lo + ap_hi) / 3
    PP       = ap_hi - ap_lo
    age_group = 1 if age > 50 else 0
    log_BMI   = np.log(BMI)   if BMI > 0 else 0
    log_MAP   = np.log(MAP)   if MAP > 0 else 0
    Age_BMI  = age * BMI
    Age_BP   = age * ap_hi
    BMI_Chol = BMI * cholesterol
    BP_Chol  = ap_hi * cholesterol
    Risk_Count = ((1 if cholesterol > 1 else 0) + (1 if gluc > 1 else 0)
                  + smoke + alco + (1 if active == 0 else 0))
    return np.array([[age, gender, height, weight, ap_hi, ap_lo,
                      cholesterol, gluc, smoke, alco, active,
                      age, height_m, BMI, MAP, PP,
                      age_group, log_BMI, log_MAP,
                      Age_BMI, Age_BP, BMI_Chol, BP_Chol, Risk_Count]])


def compute_shap_contributions(age, gender, height, weight, ap_hi, ap_lo,
                                cholesterol, gluc, smoke, alco, active):
    """
    Permutation SHAP: measure how much each clinical factor contributes to the
    predicted risk by resetting it to a healthy baseline and seeing the drop.
    Returns [(factor, delta_pct), …] sorted by |impact|, top 5 only.
    """
    base = _build_features_array(age, gender, height, weight, ap_hi, ap_lo,
                                  cholesterol, gluc, smoke, alco, active)
    base_prob = round(float(model.predict_proba(base)[0][1]) * 100, 2) / 100

    healthy_w = 22 * (height / 100) ** 2   # weight for BMI = 22

    scenarios = {
        'Age':              _build_features_array(30, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active),
        'Blood Pressure':   _build_features_array(age, gender, height, weight, 110, 70, cholesterol, gluc, smoke, alco, active),
        'Body Weight/BMI':  _build_features_array(age, gender, height, healthy_w, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active),
        'Cholesterol':      _build_features_array(age, gender, height, weight, ap_hi, ap_lo, 1, gluc, smoke, alco, active),
        'Glucose':          _build_features_array(age, gender, height, weight, ap_hi, ap_lo, cholesterol, 1, smoke, alco, active),
        'Smoking':          _build_features_array(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, 0, alco, active),
        'Alcohol':          _build_features_array(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, 0, active),
        'Physical Activity':_build_features_array(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, 1),
    }

    contribs = {}
    for factor, feats in scenarios.items():
        modified_prob = round(float(model.predict_proba(feats)[0][1]) * 100, 2) / 100
        contribs[factor] = round((base_prob - modified_prob) * 100, 1)

    return sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]


def save_prediction(data):
    conn = sqlite3.connect("heartcare.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            patient_name, age, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active,
            bmi, map_value, pulse_pressure,
            probability, risk_level, advice, created_at, user_id,
            heart_score, heart_age
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("patient_name", "Anonymous"),
        float(data["age"]), int(data["gender"]),
        float(data["height"]), float(data["weight"]),
        float(data["ap_hi"]), float(data["ap_lo"]),
        int(data["cholesterol"]), int(data["gluc"]),
        int(data["smoke"]), int(data["alco"]), int(data["active"]),
        float(data["bmi"]), float(data["map_value"]), float(data["pulse_pressure"]),
        float(data["probability"]),                    # explicit Python float
        str(data["risk_level"]), str(data["advice"]),
        str(data["created_at"]), data.get("user_id"),
        (int(data["heart_score"]) if data.get("heart_score") is not None else None),
        (int(data["heart_age"])   if data.get("heart_age")   is not None else None),
    ))
    conn.commit()
    conn.close()


def save_ecg_prediction(data):
    conn = sqlite3.connect("heartcare.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ecg_predictions (
            patient_name, filename, result, result_label, confidence,
            num_peaks, rr_mean, rr_std, mean_signal, std_signal, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["patient_name"], data["filename"], data["result"],
        data["result_label"], data["confidence"],
        data["num_peaks"], data["rr_mean"], data["rr_std"],
        data["mean_signal"], data["std_signal"], data["created_at"]
    ))
    conn.commit()
    conn.close()


def get_all_predictions(search="", risk_filter=""):
    conn = sqlite3.connect("heartcare.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if search:
        query += " AND (patient_name LIKE ? OR CAST(age AS TEXT) LIKE ?)"
        params += [f"%{search}%", f"%{search}%"]
    if risk_filter:
        query += " AND risk_level = ?"
        params.append(risk_filter)

    query += " ORDER BY id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_ecg_predictions():
    conn = sqlite3.connect("heartcare.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ecg_predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


# =========================
# ECG IMAGE PROCESSING  (v2 — 20 features, improved signal extraction)
# =========================
def extract_ecg_features_from_image(image_file):
    """
    Extract 20 ECG signal features from an uploaded ECG image.

    Improvements over v1:
    • Focuses on the middle horizontal band of the image to avoid lead labels,
      text annotations, and the top/bottom lead strips (which differ from the
      primary strip and confuse the column-wise scan).
    • Uses a percentile-based threshold per column to find the trace, instead
      of a plain centre-of-mass that is easily dominated by the grid pattern.
    • Applies a moving-average smoother to suppress grid-line artefacts.
    • Computes 7 extra features (rr_cv, dom_freq, lf_power, hf_power,
      lf_hf_ratio, spec_entropy, peak_amp_std) that are more discriminative
      for normal vs abnormal rhythms.
    """
    img = Image.open(image_file).convert('L')  # Convert to grayscale

    # ── 1. Resize to standard width ──────────────────────────────────────────
    width, height = img.size
    target_w = 1400
    if width != target_w:
        ratio = target_w / width
        img = img.resize((target_w, max(1, int(height * ratio))), Image.LANCZOS)

    arr = np.array(img, dtype=np.float64)
    h, w = arr.shape

    # ── 2. Invert if dark-background ECG ────────────────────────────────────
    if arr.mean() < 128:
        arr = 255.0 - arr

    # ── 3. Normalise contrast ─────────────────────────────────────────────────
    a_min, a_max = arr.min(), arr.max()
    if a_max > a_min:
        arr = (arr - a_min) / (a_max - a_min) * 255.0

    # ── 4. Isolate the primary ECG strip ────────────────────────────────────
    #   Real ECG printouts have multiple leads arranged in horizontal strips.
    #   The middle 50 % of image height reliably contains a continuous lead
    #   strip (Lead II or the chest leads), so we restrict scanning to it.
    strip_top = int(h * 0.25)
    strip_bot = int(h * 0.75)
    strip = 255.0 - arr[strip_top:strip_bot, :]   # invert: trace → bright
    strip_h = strip.shape[0]

    # ── 5. Column-wise trace extraction (percentile threshold) ───────────────
    signal = np.zeros(w)
    for col in range(w):
        col_data = strip[:, col]
        threshold = np.percentile(col_data, 85)
        trace_px = np.where(col_data >= threshold)[0]
        if len(trace_px) > 0:
            signal[col] = float(np.mean(trace_px))
        else:
            total = col_data.sum()
            if total > 0:
                signal[col] = float(np.average(np.arange(strip_h), weights=col_data))
            else:
                signal[col] = strip_h / 2.0

    # ── 6. Smooth to suppress grid-line noise ────────────────────────────────
    kernel_size = max(3, w // 150)
    signal = np.convolve(signal, np.ones(kernel_size) / kernel_size, mode='same')

    # ── 7. Normalise signal ───────────────────────────────────────────────────
    sig_mean = signal.mean()
    sig_std  = signal.std()
    if sig_std > 1e-6:
        signal_norm = (signal - sig_mean) / sig_std
    else:
        signal_norm = signal - sig_mean

    # ── 8. Extract 20 features ────────────────────────────────────────────────

    # Time-domain
    mean_val   = float(np.mean(signal_norm))
    std_val    = float(np.std(signal_norm))
    max_val    = float(np.max(signal_norm))
    min_val    = float(np.min(signal_norm))
    median_val = float(np.median(signal_norm))
    range_val  = float(max_val - min_val)
    rms_val    = float(np.sqrt(np.mean(signal_norm ** 2)))
    energy_val = float(np.sum(signal_norm ** 2))
    skew_val   = float(sci_stats.skew(signal_norm))
    kurt_val   = float(sci_stats.kurtosis(signal_norm))

    # Peak / rhythm
    peaks, _ = find_peaks(signal_norm, height=0.3, distance=max(5, w // 80))
    num_peaks = len(peaks)

    if len(peaks) > 1:
        rr_intervals = np.diff(peaks).astype(float)
        rr_mean = float(np.mean(rr_intervals))
        rr_std  = float(np.std(rr_intervals))
        rr_cv   = rr_std / (rr_mean + 1e-6)   # HRV: coefficient of variation
    else:
        rr_mean = 0.0
        rr_std  = 0.0
        rr_cv   = 0.0

    # Frequency-domain  (treat image pixels as pseudo-250 Hz samples)
    pseudo_fs = 250.0
    fft_mag = np.abs(np.fft.rfft(signal_norm))
    freqs   = np.fft.rfftfreq(len(signal_norm), d=1.0 / pseudo_fs)

    dom_freq = float(freqs[np.argmax(fft_mag)]) if len(fft_mag) > 0 else 0.0

    lf_mask = (freqs >= 0.5) & (freqs < 5.0)
    hf_mask = (freqs >= 5.0) & (freqs < 40.0)
    lf_power    = float(np.sum(fft_mag[lf_mask] ** 2)) if lf_mask.any() else 0.0
    hf_power    = float(np.sum(fft_mag[hf_mask] ** 2)) if hf_mask.any() else 0.0
    lf_hf_ratio = lf_power / (hf_power + 1e-6)

    psd      = fft_mag ** 2
    psd_norm = psd / (psd.sum() + 1e-6)
    spec_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-6)))

    # Peak amplitude variability
    if num_peaks > 1:
        peak_amp_std = float(np.std(signal_norm[peaks]))
    else:
        peak_amp_std = 0.0

    # ── 9. Pack into arrays ───────────────────────────────────────────────────
    features = np.array([[
        mean_val, std_val, max_val, min_val, median_val,
        range_val, rms_val, energy_val, skew_val, kurt_val,
        num_peaks, rr_mean, rr_std, rr_cv,
        dom_freq, lf_power, hf_power, lf_hf_ratio,
        spec_entropy, peak_amp_std
    ]])

    feature_info = {
        'mean':         round(mean_val, 4),
        'std':          round(std_val, 4),
        'max':          round(max_val, 4),
        'min':          round(min_val, 4),
        'median':       round(median_val, 4),
        'range':        round(range_val, 4),
        'rms':          round(rms_val, 4),
        'energy':       round(energy_val, 2),
        'skew':         round(skew_val, 4),
        'kurtosis':     round(kurt_val, 4),
        'num_peaks':    num_peaks,
        'rr_mean':      round(rr_mean, 2),
        'rr_std':       round(rr_std, 2),
        'rr_cv':        round(rr_cv, 4),
        'dom_freq':     round(dom_freq, 4),
        'lf_hf_ratio':  round(lf_hf_ratio, 4),
        'spec_entropy': round(spec_entropy, 4),
        'peak_amp_std': round(peak_amp_std, 4),
    }

    return features, feature_info


# =========================
# ECG IMAGE VALIDATOR
# =========================
def validate_ecg_image(feature_info):
    """
    Reject images that are clearly NOT ECG printouts (payment slips, photos,
    random documents, etc.) using discriminative feature thresholds.

    How it works:
    ┌─────────────────┬────────────────────┬────────────────────┐
    │ Feature         │ Real ECG range     │ Non-ECG (e.g. slip)│
    ├─────────────────┼────────────────────┼────────────────────┤
    │ lf_hf_ratio     │ 0.1 – 50           │ > 1,000            │
    │ kurtosis        │ -2  – 4            │ > 5                │
    │ num_peaks       │ 3   – 60           │ < 3 or chaotic     │
    │ rr_mean (px)    │ 30  – 250          │ < 20 (impossible)  │
    │ rr_cv           │ 0   – 1.5          │ > 2.5              │
    └─────────────────┴────────────────────┴────────────────────┘

    Returns (True, None) if the image looks like a real ECG.
    Returns (False, reason_string) if validation fails.
    """
    n_peaks   = feature_info['num_peaks']
    rr_mean   = feature_info['rr_mean']
    rr_cv     = feature_info['rr_cv']
    lf_hf     = feature_info['lf_hf_ratio']
    kurt      = feature_info['kurtosis']

    # ── Check 1: not enough waveform peaks ──────────────────────────────────
    if n_peaks < 3:
        return False, (
            "Too few waveform peaks detected. "
            "Please upload a clear, high-contrast ECG strip image. "
            "Ensure the ECG trace is visible and not cut off."
        )

    # ── Check 2: LF/HF ratio — the strongest indicator ──────────────────────
    #   Real ECG: ~0.1–50.  Non-ECG images (text, photos): commonly > 1,000.
    if lf_hf > 500:
        return False, (
            "This image does not appear to be an ECG. "
            "Its frequency profile does not match a cardiac waveform. "
            "Please upload an actual ECG strip image (PNG, JPG, TIFF)."
        )

    # ── Check 3: extremely high kurtosis = random spikes, not cardiac waves ─
    if kurt > 5.5:
        return False, (
            "The waveform shape is inconsistent with an ECG signal. "
            "Please upload an actual ECG printout, not a document or photo."
        )

    # ── Check 4: impossibly short RR intervals (fake 'peaks' from text/noise) ─
    #   Pixel spacing < 20 at 1400px width → > 350 BPM — physically impossible.
    if 0 < rr_mean < 20:
        return False, (
            "Detected peak spacing is too short to represent cardiac beats. "
            "Please upload a real ECG image rather than a document or receipt."
        )

    # ── Check 5: wildly irregular 'rhythm' = random noise, not ECG ──────────
    if n_peaks > 5 and rr_cv > 2.5:
        return False, (
            "Peak intervals are too irregular to represent cardiac rhythm. "
            "Please upload a clear ECG strip, not a photo of a document."
        )

    return True, None


# =========================
# HELPER FUNCTIONS
# =========================
def risk_level(prob):
    if prob < 0.30:
        return "Low Risk", "Maintain healthy lifestyle with regular checkups."
    elif prob < 0.60:
        return "Moderate Risk", "Monitor health regularly and consult a physician."
    elif prob < 0.80:
        return "High Risk", "Consult a cardiologist and improve lifestyle urgently."
    else:
        return "Critical Risk", "Seek immediate cardiology consultation."


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"


def confidence_band(prob):
    if prob < 0.30:
        return "Stable Zone"
    elif prob < 0.60:
        return "Watch Zone"
    elif prob < 0.80:
        return "Attention Zone"
    return "Critical Zone"


def get_risk_factors(age, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    factors = []
    if age > 50:
        factors.append("Age above 50")
    if bmi >= 30:
        factors.append("Obese BMI")
    elif bmi >= 25:
        factors.append("Overweight BMI")
    if ap_hi >= 140 or ap_lo >= 90:
        factors.append("High Blood Pressure")
    if cholesterol > 1:
        factors.append("High Cholesterol")
    if gluc > 1:
        factors.append("High Glucose")
    if smoke == 1:
        factors.append("Smoking Habit")
    if alco == 1:
        factors.append("Alcohol Consumption")
    if active == 0:
        factors.append("Low Physical Activity")
    if not factors:
        factors.append("No major visible risk factor")
    return factors


def get_health_tips(bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    tips = []
    if ap_hi >= 140 or ap_lo >= 90:
        tips.append("Reduce salt intake and monitor blood pressure regularly.")
    if bmi >= 25:
        tips.append("Maintain healthy weight through diet and regular exercise.")
    if cholesterol > 1:
        tips.append("Avoid oily and high-fat foods. Prefer a heart-healthy diet.")
    if gluc > 1:
        tips.append("Control sugar intake and check blood glucose regularly.")
    if smoke == 1:
        tips.append("Quit smoking — it's the single best cardiovascular improvement.")
    if alco == 1:
        tips.append("Limit alcohol to reduce BP and liver strain.")
    if active == 0:
        tips.append("Do at least 30 minutes of physical activity daily.")
    if not tips:
        tips.append("Continue your healthy lifestyle and get annual checkups.")
    return tips


def get_emergency_alert(ap_hi, ap_lo):
    if ap_hi >= 180 or ap_lo >= 120:
        return "EMERGENCY: BP is critically high. Seek immediate medical attention."
    if ap_hi >= 160:
        return "WARNING: Stage 2 hypertension detected. Please consult a doctor urgently."
    return None


def compute_heart_health_score(ap_hi, ap_lo, bmi, cholesterol, gluc, smoke, alco, active, age):
    """Heart Health Score 0-100. Higher = healthier."""
    score = 100
    if ap_hi >= 180 or ap_lo >= 120:   score -= 30
    elif ap_hi >= 160 or ap_lo >= 100: score -= 20
    elif ap_hi >= 140 or ap_lo >= 90:  score -= 12
    elif ap_hi >= 130 or ap_lo >= 80:  score -= 6
    if bmi >= 35:   score -= 18
    elif bmi >= 30: score -= 12
    elif bmi >= 25: score -= 6
    elif bmi < 18.5: score -= 4
    if cholesterol == 3: score -= 15
    elif cholesterol == 2: score -= 8
    if gluc == 3: score -= 12
    elif gluc == 2: score -= 6
    if smoke == 1:  score -= 15
    if alco == 1:   score -= 8
    if active == 1: score += 5
    else:           score -= 8
    if age > 70:   score -= 10
    elif age > 60: score -= 7
    elif age > 50: score -= 4
    return max(0, min(100, round(score)))


def heart_health_grade(score):
    if score >= 80:   return "Excellent", "good"
    elif score >= 65: return "Good", "moderate-good"
    elif score >= 50: return "Fair", "moderate"
    elif score >= 35: return "Poor", "high"
    else:             return "Critical", "critical"


def compute_heart_age(actual_age, ap_hi, ap_lo, bmi, cholesterol, gluc, smoke, alco, active):
    """Estimate biological heart age based on risk factors."""
    heart_age = float(actual_age)
    if ap_hi >= 180 or ap_lo >= 120:   heart_age += 15
    elif ap_hi >= 160 or ap_lo >= 100: heart_age += 10
    elif ap_hi >= 140 or ap_lo >= 90:  heart_age += 6
    elif ap_hi >= 130 or ap_lo >= 80:  heart_age += 3
    if bmi >= 35:   heart_age += 10
    elif bmi >= 30: heart_age += 6
    elif bmi >= 25: heart_age += 3
    if cholesterol == 3: heart_age += 8
    elif cholesterol == 2: heart_age += 4
    if gluc == 3: heart_age += 7
    elif gluc == 2: heart_age += 3
    if smoke == 1:  heart_age += 10
    if alco == 1:   heart_age += 4
    if active == 1: heart_age -= 3
    else:           heart_age += 4
    return max(actual_age, round(heart_age))


def get_recommended_tests(ap_hi, ap_lo, cholesterol, gluc, smoke, bmi, probability):
    """Suggest medical tests based on patient's risk profile."""
    tests = []
    if ap_hi >= 140 or ap_lo >= 90:
        tests.append({"category": "High Blood Pressure", "icon": "🫀",
                       "tests": ["ECG (Electrocardiogram)", "Echocardiogram", "Stress Test",
                                 "24-hr Ambulatory BP Monitor", "Renal Function Test"]})
    if cholesterol >= 2:
        tests.append({"category": "High Cholesterol", "icon": "🩸",
                       "tests": ["Lipid Profile (Full)", "LDL/HDL Ratio",
                                 "ApoB / ApoA1 Ratio", "Thyroid Function (TSH)"]})
    if gluc >= 2:
        tests.append({"category": "High Blood Glucose", "icon": "💉",
                       "tests": ["HbA1c (3-month average)", "Fasting Blood Glucose",
                                 "OGTT (Oral Glucose Tolerance)", "Insulin Resistance Test"]})
    if smoke == 1:
        tests.append({"category": "Smoking History", "icon": "🫁",
                       "tests": ["Chest X-Ray", "Lung Function Test (Spirometry)",
                                 "Carotid Ultrasound", "Peripheral Artery Disease Screening"]})
    if bmi >= 30:
        tests.append({"category": "Obesity / High BMI", "icon": "⚖️",
                       "tests": ["Liver Function Test", "Sleep Apnea Screening",
                                 "Insulin Sensitivity Test", "Inflammatory Markers (CRP)"]})
    if probability >= 60:
        tests.append({"category": "General Cardiac Screening", "icon": "🏥",
                       "tests": ["Coronary Artery Calcium Score (CAC)", "Troponin T/I",
                                 "BNP / NT-proBNP (heart failure marker)", "Holter Monitor (24-hr ECG)"]})
    if not tests:
        tests.append({"category": "Preventive Health Checkup", "icon": "✅",
                       "tests": ["Annual Blood Panel", "Blood Pressure Monitoring",
                                 "BMI & Weight Check", "Cholesterol Screening"]})
    return tests


def get_lifestyle_recommendations(bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    """Generate detailed lifestyle recommendations."""
    if bmi >= 30:
        exercise = "Start with 20-min daily walks, gradually reach 45 min. Add swimming, cycling, yoga. Target 300 min/week."
        cal_deficit = "500-600 kcal/day deficit recommended"
    elif bmi >= 25:
        exercise = "30 min daily brisk walk + 2 strength sessions/week. Mix cardio with strength training."
        cal_deficit = "300-400 kcal/day deficit recommended"
    else:
        exercise = "150 min/week aerobic activity. 2 strength sessions/week. Maintain balanced weight."
        cal_deficit = "Maintenance diet"

    diet = []
    if cholesterol >= 2:
        diet += ["Replace saturated fats with unsaturated (olive oil, avocado, nuts)",
                 "Eat 2-3 portions of oily fish/week (salmon, mackerel)",
                 "Increase fiber: oats, legumes, vegetables"]
    if gluc >= 2:
        diet += ["Reduce refined sugars and white carbohydrates",
                 "Choose low-GI foods: brown rice, quinoa, sweet potato",
                 "Eat small frequent meals to stabilize blood sugar"]
    if ap_hi >= 140 or ap_lo >= 90:
        diet += ["Limit sodium to <2,300 mg/day — avoid processed foods",
                 "DASH diet: fruits, vegetables, whole grains, lean proteins",
                 "Eat potassium-rich foods: bananas, spinach, beans"]
    if not diet:
        diet = ["Mediterranean diet: vegetables, fruits, whole grains, fish, olive oil",
                "Limit red meat and ultra-processed foods",
                "Include antioxidant-rich foods: berries, leafy greens, tomatoes"]

    calories = ("Increase intake 300-500 kcal/day — nutrient-dense foods." if bmi < 18.5
                else "Maintain caloric balance — focus on nutritional quality." if bmi < 25
                else "Moderate deficit 300-500 kcal/day. Avoid extreme restriction." if bmi < 30
                else "Structured deficit 500-750 kcal/day with medical supervision.")

    return {
        "exercise": exercise,
        "diet": diet,
        "water": "8-10 glasses (2.0-2.5 L) per day. More during exercise or hot weather.",
        "sleep": "7-9 hours per night. Consistent schedule. Avoid screens 1 hr before bed.",
        "calories": calories,
        "cal_deficit": cal_deficit,
        "quit_smoking": smoke == 1,
        "limit_alcohol": alco == 1,
    }


def get_doctor_recommendations(probability, ap_hi, ap_lo, cholesterol, gluc, smoke, bmi):
    """Recommend medical specialists based on risk factors."""
    docs = []
    if probability >= 60 or ap_hi >= 140:
        docs.append({"specialty": "Cardiologist", "icon": "🫀",
                     "reason": "High cardiovascular risk / hypertension management",
                     "urgency": "Urgent: Within 1 week" if probability >= 80 else "Within 2-4 weeks"})
    if gluc >= 2:
        docs.append({"specialty": "Endocrinologist / Diabetologist", "icon": "💉",
                     "reason": "High blood glucose / diabetes risk assessment",
                     "urgency": "Within 2-4 weeks"})
    if cholesterol == 3:
        docs.append({"specialty": "Lipidologist / Internal Medicine", "icon": "🩸",
                     "reason": "Very high cholesterol — medication evaluation needed",
                     "urgency": "Within 1 month"})
    if bmi >= 30:
        docs.append({"specialty": "Nutritionist / Bariatric Specialist", "icon": "⚖️",
                     "reason": "Weight management and metabolic health",
                     "urgency": "Within 1-2 months"})
    if smoke == 1:
        docs.append({"specialty": "Pulmonologist", "icon": "🫁",
                     "reason": "Lung health monitoring and smoking cessation",
                     "urgency": "Within 1-2 months"})
    if not docs:
        docs.append({"specialty": "General Physician", "icon": "👨‍⚕️",
                     "reason": "Annual health checkup and preventive screening",
                     "urgency": "Next annual checkup"})
    return docs


def validate_inputs(age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    errors = []
    if age < 1 or age > 100:
        errors.append("Age must be between 1 and 100 years.")
    if height < 100 or height > 250:
        errors.append("Height must be between 100 and 250 cm.")
    if weight < 20 or weight > 250:
        errors.append("Weight must be between 20 and 250 kg.")
    if ap_hi < 70 or ap_hi > 250:
        errors.append("Systolic BP must be between 70 and 250.")
    if ap_lo < 40 or ap_lo > 150:
        errors.append("Diastolic BP must be between 40 and 150.")
    if ap_hi <= ap_lo:
        errors.append("Systolic BP must be greater than diastolic BP.")
    if cholesterol not in [1, 2, 3]:
        errors.append("Invalid cholesterol value.")
    if gluc not in [1, 2, 3]:
        errors.append("Invalid glucose value.")
    if smoke not in [0, 1]:
        errors.append("Invalid smoking value.")
    if alco not in [0, 1]:
        errors.append("Invalid alcohol value.")
    if active not in [0, 1]:
        errors.append("Invalid physical activity value.")
    return errors


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        patient_name = request.form.get("patient_name", "Anonymous").strip() or "Anonymous"
        age          = float(request.form["age"])
        gender       = int(request.form["gender"])
        height       = float(request.form["height"])
        weight       = float(request.form["weight"])
        ap_hi        = float(request.form["ap_hi"])
        ap_lo        = float(request.form["ap_lo"])
        cholesterol  = int(request.form["cholesterol"])
        gluc         = int(request.form["gluc"])
        smoke        = int(request.form["smoke"])
        alco         = int(request.form["alco"])
        active       = int(request.form["active"])

        errors = validate_inputs(age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
        if errors:
            return render_template(
                "result.html",
                patient_name=patient_name,
                probability=0, level="Invalid Input",
                advice="Please correct the input values and try again.",
                bmi=0, map_value=0, pulse_pressure=0,
                risk_factors=errors,
                health_tips=["Use valid medical values."],
                emergency_alert=None, band="Not Available",
                bmi_status="Not Available",
                generated_at="Unavailable"
            )

        age_years  = age
        height_m   = height / 100
        BMI        = weight / (height_m ** 2)
        bmi_status = bmi_category(BMI)
        MAP        = (2 * ap_lo + ap_hi) / 3
        PP         = ap_hi - ap_lo

        age_group = 1 if age_years > 50 else 0
        log_BMI   = np.log(BMI) if BMI > 0 else 0
        log_MAP   = np.log(MAP) if MAP > 0 else 0

        Age_BMI  = age_years * BMI
        Age_BP   = age_years * ap_hi
        BMI_Chol = BMI * cholesterol
        BP_Chol  = ap_hi * cholesterol

        Risk_Count = (
            (1 if cholesterol > 1 else 0) +
            (1 if gluc > 1 else 0) +
            smoke + alco +
            (1 if active == 0 else 0)
        )

        features = np.array([[
            age, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active,
            age_years, height_m, BMI, MAP, PP,
            age_group, log_BMI, log_MAP,
            Age_BMI, Age_BP, BMI_Chol, BP_Chol, Risk_Count
        ]])

        prob  = float(model.predict_proba(features)[0][1])   # ensure Python float
        level, advice = risk_level(prob)
        band  = confidence_band(prob)

        risk_factors  = get_risk_factors(age, BMI, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
        health_tips   = get_health_tips(BMI, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
        emergency_alert = get_emergency_alert(ap_hi, ap_lo)

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # SHAP – permutation feature attribution
        shap_data = compute_shap_contributions(
            age, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active)

        # New advanced metrics
        heart_score = compute_heart_health_score(ap_hi, ap_lo, BMI, cholesterol, gluc, smoke, alco, active, age)
        hs_grade, hs_cls = heart_health_grade(heart_score)
        heart_age = compute_heart_age(age, ap_hi, ap_lo, BMI, cholesterol, gluc, smoke, alco, active)
        recommended_tests = get_recommended_tests(ap_hi, ap_lo, cholesterol, gluc, smoke, BMI, round(prob * 100, 2))
        lifestyle = get_lifestyle_recommendations(BMI, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
        doctor_recs = get_doctor_recommendations(round(prob * 100, 2), ap_hi, ap_lo, cholesterol, gluc, smoke, BMI)

        uid = session.get('user_id')
        save_prediction({
            "patient_name":   patient_name,
            "age":            age,
            "gender":         gender,
            "height":         height,
            "weight":         weight,
            "ap_hi":          ap_hi,
            "ap_lo":          ap_lo,
            "cholesterol":    cholesterol,
            "gluc":           gluc,
            "smoke":          smoke,
            "alco":           alco,
            "active":         active,
            "bmi":            round(BMI, 2),
            "map_value":      round(MAP, 2),
            "pulse_pressure": round(PP, 2),
            "probability":    round(prob * 100, 2),
            "risk_level":     level,
            "advice":         advice,
            "created_at":     generated_at,
            "user_id":        uid,
            "heart_score":    heart_score,
            "heart_age":      heart_age,
        })

        return render_template(
            "result.html",
            patient_name=patient_name,
            probability=round(prob * 100, 2),
            level=level, advice=advice,
            bmi=round(BMI, 2), bmi_status=bmi_status,
            map_value=round(MAP, 2),
            pulse_pressure=round(PP, 2),
            risk_factors=risk_factors,
            health_tips=health_tips,
            emergency_alert=emergency_alert,
            band=band,
            generated_at=generated_at,
            shap_data=shap_data,
            heart_score=heart_score,
            hs_grade=hs_grade,
            hs_cls=hs_cls,
            heart_age=int(heart_age),
            actual_age=int(age),
            recommended_tests=recommended_tests,
            lifestyle=lifestyle,
            doctor_recs=doctor_recs,
            # pass raw inputs for what-if pre-fill
            raw=dict(age=age, gender=gender, height=height, weight=weight,
                     ap_hi=ap_hi, ap_lo=ap_lo, cholesterol=cholesterol,
                     gluc=gluc, smoke=smoke, alco=alco, active=active),
        )

    except Exception as e:
        return render_template(
            "result.html",
            patient_name="Unknown",
            probability=0, level="Processing Error",
            advice="Something went wrong. Please check your inputs.",
            bmi=0, map_value=0, pulse_pressure=0,
            risk_factors=["Unable to process the request."],
            health_tips=["Please check inputs and try again."],
            emergency_alert=None, band="Unavailable",
            bmi_status="Unavailable",
            generated_at="Unavailable"
        )


# =========================
# ECG ANALYSIS
# =========================
@app.route("/ecg_analysis", methods=["GET", "POST"])
def ecg_analysis():
    if request.method == "GET":
        return render_template("ecg_analysis.html", result=None)

    patient_name = request.form.get("patient_name", "Anonymous").strip() or "Anonymous"
    file = request.files.get("ecg_image")

    if not file or file.filename == "":
        return render_template("ecg_analysis.html", result=None,
                               error="Please upload an ECG image.")
    if not allowed_file(file.filename):
        return render_template("ecg_analysis.html", result=None,
                               error="Invalid file type. Upload PNG, JPG, BMP, or TIFF.")

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = ts + filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.seek(0)
        file.save(save_path)

        # Extract features
        file.seek(0)
        features, feature_info = extract_ecg_features_from_image(file)

        # ── Validate: is this actually an ECG image? ─────────────────────────
        # Validate ECG image
        is_ecg = True
        # is_ecg, rejection_reason = validate_ecg_image(feature_info)
        # if not is_ecg:
        #     # Delete the saved upload so junk files don't accumulate
        #     try:
        #         os.remove(save_path)
        #     except Exception:
        #         pass
        #     return render_template("ecg_analysis.html", result=None,
        #                            error=f"Invalid image: {rejection_reason}")

        # Predict
        img = Image.open(file).resize((200,200))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        features = features.reshape(1, -1)
        proba = ecg_model.predict_proba(features)[0][1]

        if proba > 0.5:
            prediction = 1
            confidence = proba * 100
        else:
            prediction = 0
            confidence = (1 - proba) * 100

        if prediction == 1:
            result_label = "Abnormal ECG"
            result_msg   = "Possible cardiac abnormality detected"
            result_class = "danger"
            result_icon  = "🚨"
            details = (
                "The ECG pattern analysis suggests potential cardiac abnormality. "
                "This may indicate arrhythmia, ischemia, or other heart conditions. "
                "Please consult a cardiologist for a proper clinical evaluation."
            )
        else:
            result_label = "Normal ECG"
            result_msg   = "No significant cardiac abnormality detected"
            result_class = "success"
            result_icon  = "✅"
            details = (
                "The ECG pattern analysis appears within normal range. "
                "However, this is a screening tool only. Regular cardiac checkups "
                "are still recommended, especially if you have risk factors."
            )

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        save_ecg_prediction({
            "patient_name": patient_name,
            "filename":     filename,
            "result":       str(prediction),
            "result_label": result_label,
            "confidence":   confidence,
            "num_peaks":    feature_info["num_peaks"],
            "rr_mean":      feature_info["rr_mean"],
            "rr_std":       feature_info["rr_std"],
            "mean_signal":  feature_info["mean"],
            "std_signal":   feature_info["std"],
            "created_at":   generated_at
        })

        result = {
            "patient_name": patient_name,
            "label":        result_label,
            "msg":          result_msg,
            "cls":          result_class,
            "icon":         result_icon,
            "details":      details,
            "confidence":   confidence,
            "features":     feature_info,
            "image_path":   f"uploads/{filename}",
            "generated_at": generated_at,
        }

        return render_template("ecg_analysis.html", result=result, error=None)

    except Exception as e:
        return render_template("ecg_analysis.html", result=None,
                               error=f"Could not process image: {str(e)}")


# =========================
# AI ASSISTANT
# =========================
@app.route("/ai_assistant", methods=["GET", "POST"])
def ai_assistant():
    answer = None
    user_query = ""

    # Load latest prediction for the logged-in user (personalized context)
    user_ctx = None
    uid = session.get('user_id')
    if uid:
        rows = get_user_predictions(uid, limit=1)
        if rows:
            r = rows[0]
            user_ctx = {
                "name":        session.get('username'),
                "risk":        r["probability"],
                "level":       r["risk_level"],
                "bp":          f"{r['ap_hi']}/{r['ap_lo']}",
                "bmi":         r["bmi"],
                "smoke":       r["smoke"],
                "alco":        r["alco"],
                "active":      r["active"],
                "cholesterol": r["cholesterol"],
                "gluc":        r["gluc"],
            }

    if request.method == "POST":
        user_query = request.form.get("user_query", "")
        answer = get_ai_response(user_query, user_ctx)
    return render_template("ai_assistant.html", answer=answer,
                           user_query=user_query, user_ctx=user_ctx)


def get_ai_response(user_query, ctx=None):
    q = user_query.lower().strip()

    responses = {
        ("blood pressure", "bp", "high bp", "hypertension"):
            "High blood pressure (hypertension) strains the heart and vessels. Reduce salt, exercise regularly, manage stress, avoid smoking, and monitor BP frequently. Normal BP is below 120/80 mmHg.",
        ("cholesterol", "ldl", "hdl"):
            "High LDL cholesterol causes plaque in arteries, risking heart attack. Eat more fiber, avoid fried/fatty foods, exercise, and consider statins if your doctor recommends.",
        ("bmi", "weight", "obesity", "overweight"):
            "High BMI increases cardiovascular risk. Aim for BMI 18.5–24.9 through balanced diet and at least 150 minutes of moderate exercise weekly.",
        ("smoking", "smoke", "cigarette", "tobacco"):
            "Smoking is the #1 preventable cause of heart disease. It damages vessel walls and increases clot risk. Quitting even after years of smoking significantly reduces risk.",
        ("alcohol", "drink", "drinking"):
            "Excess alcohol raises BP, causes arrhythmia, and weakens the heart muscle. Men: max 2 drinks/day; Women: max 1 drink/day.",
        ("exercise", "workout", "physical activity", "walk"):
            "150+ minutes/week of moderate aerobic exercise (brisk walking, cycling, swimming) reduces heart disease risk by up to 35%. Start slow and increase gradually.",
        ("diet", "food", "eat", "nutrition"):
            "Heart-healthy diet: fruits, vegetables, whole grains, lean protein, healthy fats (olive oil, nuts). Limit red meat, processed food, sugar, and salt.",
        ("glucose", "diabetes", "sugar", "blood sugar"):
            "Uncontrolled diabetes doubles heart disease risk. Monitor blood glucose, follow a low-glycemic diet, exercise regularly, and take prescribed medication.",
        ("heart attack", "heart disease", "coronary", "cardiac", "heart problem"):
            "Heart disease covers coronary artery disease, heart failure, arrhythmia, and more. Early signs: chest pain, shortness of breath, fatigue. Risk factors: BP, cholesterol, diabetes, smoking, obesity.",
        ("reduce risk", "prevention", "prevent", "protect heart"):
            "Top 5 to protect your heart: 1) Don't smoke, 2) Exercise daily, 3) Eat heart-healthy, 4) Manage BP/cholesterol/glucose, 5) Maintain healthy weight.",
        ("ecg", "electrocardiogram", "ecg test", "heart rhythm"):
            "An ECG records the heart's electrical activity. Abnormal patterns may indicate arrhythmia, ischemia, or prior heart attack. Our ECG Analysis tool can screen uploaded ECG images.",
        ("stress", "anxiety", "mental"):
            "Chronic stress raises cortisol, increasing BP and inflammation. Practice mindfulness, deep breathing, regular exercise, and adequate sleep (7–8 hrs).",
        ("age", "elderly", "old age"):
            "Cardiovascular risk increases with age, especially after 45 (men) and 55 (women). More frequent screenings for BP, cholesterol, and glucose are recommended.",
        ("sleep", "insomnia", "rest"):
            "Poor sleep (< 6 hrs/night) is linked to hypertension, obesity, and heart disease. Aim for 7–9 hours and maintain a consistent sleep schedule.",
        ("bmi category", "bmi range"):
            "BMI categories: Underweight < 18.5 | Normal 18.5–24.9 | Overweight 25–29.9 | Obese ≥ 30",
        ("low risk",):
            "Low risk means fewer cardiovascular warning signs currently. Maintain this through a healthy lifestyle, regular check-ups, and monitoring key metrics.",
        ("high risk", "critical risk", "danger"):
            "High/critical risk indicates multiple warning signs. Consult a cardiologist, adopt lifestyle changes immediately, and follow medical advice closely.",
        ("hello", "hi", "hey", "help"):
            "Hello! I'm HeartCare AI Assistant. Ask me about: blood pressure, cholesterol, BMI, ECG, diet, exercise, smoking, diabetes, heart disease, or risk prevention.",
    }

    # Personalized responses using logged-in user's latest data
    if ctx and any(k in q for k in ("my risk", "my result", "my score", "my health", "how am i")):
        name = ctx['name']
        return (
            f"Hi {name}! Your latest risk score is <strong>{ctx['risk']}%</strong> "
            f"({ctx['level']}). Your BP is {ctx['bp']} mmHg and BMI is {ctx['bmi']}. "
            + ("You are currently smoking — quitting is the single most impactful change you can make. " if ctx['smoke'] else "")
            + ("Reducing alcohol intake will help lower your BP. " if ctx['alco'] else "")
            + ("Regular physical activity can reduce your risk by up to 35%. " if not ctx['active'] else "")
            + "Visit the What-if Simulation to explore how lifestyle changes affect your score."
        )

    if ctx and any(k in q for k in ("my bp", "my blood pressure", "my pressure")):
        return (f"Your last recorded blood pressure is <strong>{ctx['bp']} mmHg</strong>. "
                + ("This is in the hypertensive range — reduce salt, exercise, and consult a doctor. "
                   if int(ctx['bp'].split('/')[0]) >= 140 else
                   "This is in the normal-to-elevated range. Continue monitoring regularly."))

    if ctx and any(k in q for k in ("my bmi", "my weight", "my body")):
        bmi = ctx['bmi']
        cat = "Obese" if bmi >= 30 else "Overweight" if bmi >= 25 else "Normal" if bmi >= 18.5 else "Underweight"
        return (f"Your BMI is <strong>{bmi}</strong> ({cat}). "
                + ("Losing even 5 kg can meaningfully reduce your cardiovascular risk. "
                   if bmi >= 25 else "Keep maintaining your healthy weight!"))

    for keywords, reply in responses.items():
        if any(k in q for k in keywords):
            return reply

    if ctx:
        return (f"Hi {ctx['name']}! I can give you personalized advice based on your data. "
                "Try asking: 'my risk', 'my BP', 'my BMI', or general topics like "
                "blood pressure, cholesterol, diet, exercise, ECG, or stress.")

    return ("I can help with: blood pressure, cholesterol, BMI, ECG, smoking, alcohol, diet, "
            "exercise, glucose/diabetes, sleep, stress, heart disease prevention, and risk levels. "
            "Log in to get personalized advice based on your health data.")


# =========================
# ABOUT
# =========================
@app.route("/about_heart_disease")
def about_heart_disease():
    return render_template("about_heart_disease.html")


# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():
    data = get_dashboard_data()
    return render_template("dashboard.html", data=data)


def get_dashboard_data():
    conn = sqlite3.connect("heartcare.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    def qone(sql, params=()):
        cursor.execute(sql, params)
        return cursor.fetchone()

    total    = qone("SELECT COUNT(*) as c FROM predictions")["c"]
    low      = qone("SELECT COUNT(*) as c FROM predictions WHERE risk_level='Low Risk'")["c"]
    moderate = qone("SELECT COUNT(*) as c FROM predictions WHERE risk_level='Moderate Risk'")["c"]
    high     = qone("SELECT COUNT(*) as c FROM predictions WHERE risk_level='High Risk'")["c"]
    critical = qone("SELECT COUNT(*) as c FROM predictions WHERE risk_level='Critical Risk'")["c"]

    avg_bmi   = qone("SELECT ROUND(AVG(bmi),2) as v FROM predictions")["v"] or 0
    avg_ap_hi = qone("SELECT ROUND(AVG(ap_hi),2) as v FROM predictions")["v"] or 0
    avg_ap_lo = qone("SELECT ROUND(AVG(ap_lo),2) as v FROM predictions")["v"] or 0
    # probability may have been stored as BLOB before migration — use safe cast
    avg_prob  = qone("SELECT ROUND(AVG(CAST(probability AS REAL)),2) as v FROM predictions")["v"] or 0

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 8")
    recent_records = cursor.fetchall()

    # BMI distribution buckets
    bmi_under  = qone("SELECT COUNT(*) as c FROM predictions WHERE bmi < 18.5")["c"]
    bmi_normal = qone("SELECT COUNT(*) as c FROM predictions WHERE bmi >= 18.5 AND bmi < 25")["c"]
    bmi_over   = qone("SELECT COUNT(*) as c FROM predictions WHERE bmi >= 25 AND bmi < 30")["c"]
    bmi_obese  = qone("SELECT COUNT(*) as c FROM predictions WHERE bmi >= 30")["c"]

    # Last 7 predictions trend
    cursor.execute("SELECT probability, created_at FROM predictions ORDER BY id DESC LIMIT 7")
    trend_rows = cursor.fetchall()
    trend_probs  = [r["probability"] for r in reversed(trend_rows)]
    trend_labels = [r["created_at"][:10] for r in reversed(trend_rows)]

    # ECG stats
    ecg_total    = qone("SELECT COUNT(*) as c FROM ecg_predictions")["c"]
    ecg_abnormal = qone("SELECT COUNT(*) as c FROM ecg_predictions WHERE result='1'")["c"]
    ecg_normal   = qone("SELECT COUNT(*) as c FROM ecg_predictions WHERE result='0'")["c"]

    conn.close()

    high_risk_count = high + critical
    high_risk_pct = round((high_risk_count / total * 100), 1) if total > 0 else 0

    return {
        "total": total, "low": low, "moderate": moderate,
        "high": high, "critical": critical,
        "avg_bmi": avg_bmi, "avg_ap_hi": avg_ap_hi,
        "avg_ap_lo": avg_ap_lo, "avg_prob": avg_prob,
        "recent_records": recent_records,
        "bmi_under": bmi_under, "bmi_normal": bmi_normal,
        "bmi_over": bmi_over, "bmi_obese": bmi_obese,
        "trend_probs": trend_probs, "trend_labels": trend_labels,
        "ecg_total": ecg_total, "ecg_abnormal": ecg_abnormal,
        "ecg_normal": ecg_normal,
        "high_risk_pct": high_risk_pct,
    }


# =========================
# HISTORY
# =========================
@app.route("/history")
def history():
    search      = request.args.get("search", "").strip()
    risk_filter = request.args.get("risk", "").strip()
    records     = get_all_predictions(search, risk_filter)
    return render_template("history.html", records=records,
                           search=search, risk_filter=risk_filter)


@app.route("/ecg_history")
def ecg_history():
    records = get_all_ecg_predictions()
    return render_template("ecg_history.html", records=records)


@app.route("/export_csv")
def export_csv():
    records = get_all_predictions()

    def generate():
        header = ["ID", "Patient Name", "Date", "Age", "Gender", "Height", "Weight",
                  "Systolic BP", "Diastolic BP", "Cholesterol", "Glucose",
                  "Smoking", "Alcohol", "Active", "BMI", "MAP",
                  "Pulse Pressure", "Probability %", "Risk Level", "Advice"]
        yield ",".join(header) + "\n"
        for r in records:
            row = [
                str(r["id"]), r["patient_name"] or "Anonymous",
                r["created_at"], str(r["age"]),
                "Female" if r["gender"] == 1 else "Male",
                str(r["height"]), str(r["weight"]),
                str(r["ap_hi"]), str(r["ap_lo"]),
                str(r["cholesterol"]), str(r["gluc"]),
                str(r["smoke"]), str(r["alco"]), str(r["active"]),
                str(r["bmi"]), str(r["map_value"]), str(r["pulse_pressure"]),
                str(r["probability"]), r["risk_level"], r["advice"]
            ]
            yield ",".join(f'"{v}"' for v in row) + "\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=heartcare_history.csv"}
    )


# =========================
# PDF REPORT
# =========================
@app.route("/download_report", methods=["POST"])
def download_report():
    patient_name   = request.form.get("patient_name", "Anonymous")
    probability    = request.form.get("probability", "N/A")
    level          = request.form.get("level", "N/A")
    advice         = request.form.get("advice", "N/A")
    bmi            = request.form.get("bmi", "N/A")
    bmi_status_val = request.form.get("bmi_status", "N/A")
    map_value      = request.form.get("map_value", "N/A")
    pulse_pressure = request.form.get("pulse_pressure", "N/A")
    risk_factors   = request.form.get("risk_factors_str", "")
    health_tips    = request.form.get("health_tips_str", "")
    generated_at   = request.form.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    heart_score    = request.form.get("heart_score", "N/A")
    heart_age      = request.form.get("heart_age", "N/A")
    actual_age     = request.form.get("actual_age", "N/A")
    hs_grade       = request.form.get("hs_grade", "")
    shap_str       = request.form.get("shap_str", "")
    tests_str      = request.form.get("tests_str", "")

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    W, H = letter

    # ---- Header bar ----
    p.setFillColorRGB(0.07, 0.04, 0.18)
    p.rect(0, H - 72, W, 72, fill=1, stroke=0)

    p.setFillColorRGB(0.49, 0.23, 0.93)
    p.circle(40, H - 36, 8, fill=1, stroke=0)

    p.setFont("Helvetica-Bold", 18)
    p.setFillColorRGB(0.92, 0.94, 1)
    p.drawString(60, H - 42, "HeartCare AI — Heart Risk Report")

    p.setFont("Helvetica", 10)
    p.setFillColorRGB(0.7, 0.75, 0.9)
    p.drawString(60, H - 58, f"Generated: {generated_at}   |   Patient: {patient_name}")

    # ---- Risk box ----
    try:
        prob_float = float(probability)
    except Exception:
        prob_float = 0

    if prob_float < 30:
        r, g, b = 0.13, 0.77, 0.37
    elif prob_float < 60:
        r, g, b = 0.96, 0.62, 0.04
    elif prob_float < 80:
        r, g, b = 0.94, 0.27, 0.27
    else:
        r, g, b = 0.55, 0.05, 0.05

    p.setFillColorRGB(r, g, b)
    p.roundRect(40, H - 175, 240, 80, 10, fill=1, stroke=0)
    p.setFillColorRGB(1, 1, 1)
    p.setFont("Helvetica-Bold", 32)
    p.drawCentredString(160, H - 140, f"{probability}%")
    p.setFont("Helvetica-Bold", 13)
    p.drawCentredString(160, H - 162, f"Risk Score — {level}")

    # ---- Metrics ----
    p.setFillColorRGB(0.92, 0.94, 1)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(310, H - 105, "Clinical Metrics")
    p.setFont("Helvetica", 11)
    metrics = [
        ("BMI", f"{bmi}  ({bmi_status_val})"),
        ("Mean Arterial Pressure", f"{map_value} mmHg"),
        ("Pulse Pressure", f"{pulse_pressure} mmHg"),
        ("Recommendation", advice),
    ]
    y = H - 122
    for label, val in metrics:
        p.setFont("Helvetica-Bold", 10)
        p.setFillColorRGB(0.55, 0.6, 0.85)
        p.drawString(310, y, label + ":")
        p.setFont("Helvetica", 10)
        p.setFillColorRGB(0.92, 0.94, 1)
        # Wrap long text
        if len(val) > 45:
            val = val[:45] + "…"
        p.drawString(310, y - 13, val)
        y -= 34

    # ---- Divider ----
    p.setStrokeColorRGB(0.49, 0.23, 0.93)
    p.setLineWidth(1.5)
    p.line(40, H - 190, W - 40, H - 190)

    # ---- Risk Factors ----
    y = H - 210
    p.setFillColorRGB(0.92, 0.94, 1)
    p.setFont("Helvetica-Bold", 13)
    p.drawString(40, y, "Identified Risk Factors")
    y -= 18
    p.setFont("Helvetica", 10)
    if risk_factors:
        for factor in risk_factors.split("|"):
            factor = factor.strip()
            if factor:
                p.setFillColorRGB(0.94, 0.27, 0.27)
                p.circle(50, y + 3, 3, fill=1, stroke=0)
                p.setFillColorRGB(0.92, 0.94, 1)
                p.drawString(60, y, factor)
                y -= 16

    # ---- Health Tips ----
    y -= 8
    p.setFont("Helvetica-Bold", 13)
    p.setFillColorRGB(0.92, 0.94, 1)
    p.drawString(40, y, "Health Recommendations")
    y -= 18
    p.setFont("Helvetica", 10)
    if health_tips:
        for tip in health_tips.split("|"):
            tip = tip.strip()
            if tip:
                p.setFillColorRGB(0.13, 0.77, 0.37)
                p.circle(50, y + 3, 3, fill=1, stroke=0)
                p.setFillColorRGB(0.92, 0.94, 1)
                # Wrap tip at ~90 chars
                words = tip.split()
                line = ""
                for word in words:
                    if len(line) + len(word) + 1 > 90:
                        p.drawString(60, y, line)
                        y -= 14
                        line = word
                    else:
                        line = (line + " " + word).strip()
                if line:
                    p.drawString(60, y, line)
                    y -= 16

    # ---- Heart Health Score & Heart Age ----
    if y > 160:
        y -= 10
        p.setFont("Helvetica-Bold", 13)
        p.setFillColorRGB(0.92, 0.94, 1)
        p.drawString(40, y, "Heart Health Score & Estimated Heart Age")
        y -= 18
        p.setFont("Helvetica", 10)
        try:
            hs_val = int(float(heart_score))
            if hs_val >= 80:    hs_r,hs_g,hs_b = 0.13,0.77,0.37
            elif hs_val >= 65:  hs_r,hs_g,hs_b = 0.34,0.85,0.52
            elif hs_val >= 50:  hs_r,hs_g,hs_b = 0.96,0.62,0.04
            elif hs_val >= 35:  hs_r,hs_g,hs_b = 0.94,0.27,0.27
            else:               hs_r,hs_g,hs_b = 0.55,0.05,0.05
            p.setFillColorRGB(hs_r,hs_g,hs_b)
            p.roundRect(40, y-20, 140, 34, 8, fill=1, stroke=0)
            p.setFillColorRGB(1,1,1)
            p.setFont("Helvetica-Bold", 14)
            p.drawCentredString(110, y-6, f"Heart Score: {hs_val}/100")
            p.setFont("Helvetica", 9)
            p.drawCentredString(110, y-17, f"Grade: {hs_grade}")
        except:
            p.setFillColorRGB(0.92,0.94,1)
            p.drawString(40, y, f"Heart Score: {heart_score}/100  ({hs_grade})")
        try:
            ha_val = int(float(heart_age))
            aa_val = int(float(actual_age))
            diff = ha_val - aa_val
            p.setFillColorRGB(0.92,0.94,1)
            p.setFont("Helvetica-Bold", 10)
            p.drawString(200, y-3, f"Actual Age: {aa_val} yrs")
            p.drawString(200, y-18, f"Heart Age:  {ha_val} yrs  ({'+'+str(diff) if diff>0 else str(diff)} yrs)")
        except:
            pass
        y -= 48

    # ---- SHAP Explainability ----
    if shap_str and y > 100:
        y -= 6
        p.setFont("Helvetica-Bold", 12)
        p.setFillColorRGB(0.92, 0.94, 1)
        p.drawString(40, y, "Explainable AI — Top Risk Factors (SHAP)")
        y -= 16
        p.setFont("Helvetica", 9)
        for item in shap_str.split("|"):
            item = item.strip()
            if item and y > 60:
                try:
                    factor, delta = item.rsplit(":", 1)
                    delta_f = float(delta)
                    clr = (0.94,0.27,0.27) if delta_f > 0 else (0.13,0.77,0.37)
                    p.setFillColorRGB(*clr)
                    p.circle(50, y+3, 3, fill=1, stroke=0)
                    p.setFillColorRGB(0.92,0.94,1)
                    sign = "+" if delta_f > 0 else ""
                    p.drawString(60, y, f"{factor.strip()}  {sign}{delta_f}%")
                    y -= 14
                except:
                    p.drawString(60, y, item)
                    y -= 14

    # ---- Footer ----
    p.setStrokeColorRGB(0.3, 0.3, 0.5)
    p.setLineWidth(0.8)
    p.line(40, 55, W - 40, 55)
    p.setFont("Helvetica", 9)
    p.setFillColorRGB(0.55, 0.6, 0.85)
    p.drawString(40, 40, "HeartCare AI — Developed by Purushottam Kumar")
    p.drawRightString(W - 40, 40, "Disclaimer: Educational use only. Not a medical diagnosis.")

    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name=f"HeartCare_Report_{patient_name.replace(' ', '_')}.pdf",
                     mimetype="application/pdf")


# =========================
# AUTH ROUTES
# =========================
@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if request.method == "GET":
        return render_template("signup.html")
    username = request.form.get("username", "").strip()
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    confirm  = request.form.get("confirm", "")
    if not username or not password:
        return render_template("signup.html", error="Username and password are required.")
    if password != confirm:
        return render_template("signup.html", error="Passwords do not match.")
    if len(password) < 6:
        return render_template("signup.html", error="Password must be at least 6 characters.")
    uid, err = create_user(username, email, password)
    if err:
        return render_template("signup.html", error=err)
    session['user_id']  = uid
    session['username'] = username
    return redirect(url_for('profile_page'))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        return render_template("login.html")
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    user = verify_user(username, password)
    if not user:
        return render_template("login.html", error="Invalid username or password.")
    session['user_id']  = user['id']
    session['username'] = user['username']
    return redirect(url_for('profile_page'))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('home'))


# =========================
# PERSONAL PROFILE / TRENDS
# =========================
@app.route("/profile")
@login_required
def profile_page():
    uid  = session['user_id']
    user = get_user(uid)
    rows = get_user_predictions(uid, limit=30)

    # Build trend data (chronological order)
    records_rev = list(reversed(rows))
    trend_labels = [r["created_at"][:10] for r in records_rev]
    trend_risk   = [_safe_float(r["probability"]) for r in records_rev]
    trend_bmi    = [_safe_float(r["bmi"])         for r in records_rev]
    trend_ap_hi  = [_safe_float(r["ap_hi"])       for r in records_rev]
    trend_ap_lo  = [_safe_float(r["ap_lo"])        for r in records_rev]

    # Summary stats
    total = len(rows)
    avg_risk = round(sum(_safe_float(r["probability"]) for r in rows) / total, 1) if total else 0
    last = rows[0] if rows else None

    return render_template(
        "profile.html",
        user=user,
        records=rows[:10],
        total=total,
        avg_risk=avg_risk,
        last=last,
        trend_labels=json.dumps(trend_labels),
        trend_risk=json.dumps(trend_risk),
        trend_bmi=json.dumps(trend_bmi),
        trend_ap_hi=json.dumps(trend_ap_hi),
        trend_ap_lo=json.dumps(trend_ap_lo),
    )


# =========================
# WHAT-IF SIMULATION
# =========================
@app.route("/simulate", methods=["GET"])
def simulate():
    # Pre-fill from query params (passed from result page)
    pre = {k: request.args.get(k, "") for k in
           ['age','gender','height','weight','ap_hi','ap_lo',
            'cholesterol','gluc','smoke','alco','active']}
    return render_template("simulate.html", pre=pre)


@app.route("/simulate_api", methods=["POST"])
def simulate_api():
    """AJAX endpoint — returns JSON risk scores for two scenarios."""
    try:
        def _prob(a, ge, h, w, hi, lo, ch, gl, sm, al, ac):
            feats = _build_features_array(a, ge, h, w, hi, lo, ch, gl, sm, al, ac)
            return round(float(model.predict_proba(feats)[0][1]) * 100, 2)

        age    = float(request.form['age'])
        gender = int(request.form['gender'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        ap_hi  = float(request.form['ap_hi'])
        ap_lo  = float(request.form['ap_lo'])
        chol   = int(request.form['cholesterol'])
        gluc   = int(request.form['gluc'])
        smoke  = int(request.form['smoke'])
        alco   = int(request.form['alco'])
        active = int(request.form['active'])

        current = _prob(age, gender, height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active)

        # Modified scenario
        ap_hi2  = float(request.form.get('ap_hi2',  ap_hi))
        ap_lo2  = float(request.form.get('ap_lo2',  ap_lo))
        weight2 = float(request.form.get('weight2', weight))
        chol2   = int(request.form.get('cholesterol2', chol))
        smoke2  = int(request.form.get('smoke2', smoke))
        alco2   = int(request.form.get('alco2', alco))
        active2 = int(request.form.get('active2', active))

        modified = _prob(age, gender, height, weight2, ap_hi2, ap_lo2, chol2, gluc, smoke2, alco2, active2)

        bmi_current  = round(weight  / (height/100)**2, 1)
        bmi_modified = round(weight2 / (height/100)**2, 1)

        return jsonify(current=current, modified=modified,
                       bmi_current=bmi_current, bmi_modified=bmi_modified,
                       delta=round(modified - current, 1))
    except Exception as e:
        return jsonify(error=str(e)), 400


# =========================
# WEARABLE INTEGRATION (Concept)
# =========================
@app.route("/wearable")
def wearable():
    # Generate dummy wearable data (simulated 24-hour heart rate)
    import math
    hrs = []
    for i in range(144):   # every 10 min over 24 hours
        hour = (i * 10) / 60
        base = 72
        if 6 <= hour <= 8:   base = 90   # morning activity
        elif 12 <= hour <= 13: base = 85  # lunch walk
        elif 17 <= hour <= 18: base = 100  # exercise
        elif 22 <= hour <= 24 or hour < 6: base = 58  # sleep
        hr = base + math.sin(i * 0.4) * 5 + (hash(i) % 7 - 3)
        hrs.append(round(hr))

    labels = [f"{(i*10)//60:02d}:{(i*10)%60:02d}" for i in range(144)]
    steps_daily = [hash(d) % 4000 + 4000 for d in range(7)]
    step_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    return render_template(
        "wearable.html",
        hr_data=json.dumps(hrs),
        hr_labels=json.dumps(labels[::6]),   # every hour label
        hr_data_sparse=json.dumps(hrs[::6]),
        steps_data=json.dumps(steps_daily),
        step_labels=json.dumps(step_labels),
        avg_hr=round(sum(hrs)/len(hrs)),
        max_hr=max(hrs),
        min_hr=min(hrs),
    )


@app.route("/find_doctor")
def find_doctor():
    return render_template("find_doctor.html")


# =========================
# MODEL PERFORMANCE PAGE
# =========================
@app.route("/model_performance")
def model_performance():
    conn = sqlite3.connect("heartcare.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as c FROM predictions")
    total_preds = cursor.fetchone()["c"]
    cursor.execute("SELECT COUNT(*) as c FROM ecg_predictions")
    total_ecg = cursor.fetchone()["c"]
    conn.close()
    return render_template("model_performance.html",
                           total_preds=total_preds, total_ecg=total_ecg)


# =========================
# REST API ENDPOINTS
# =========================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API: Heart disease risk prediction.
    POST JSON: {age, gender, height, weight, ap_hi, ap_lo,
                cholesterol, gluc, smoke, alco, active, patient_name(optional)}
    Returns: {probability, risk_level, advice, bmi, map_value, pulse_pressure,
              risk_factors, health_tips, shap_data, emergency_alert}
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify(error="No JSON body provided"), 400

        patient_name = str(data.get("patient_name", "Anonymous")).strip() or "Anonymous"
        age         = float(data["age"])
        gender      = int(data["gender"])
        height      = float(data["height"])
        weight      = float(data["weight"])
        ap_hi       = float(data["ap_hi"])
        ap_lo       = float(data["ap_lo"])
        cholesterol = int(data["cholesterol"])
        gluc        = int(data["gluc"])
        smoke       = int(data["smoke"])
        alco        = int(data["alco"])
        active      = int(data["active"])

        errors = validate_inputs(age, height, weight, ap_hi, ap_lo,
                                 cholesterol, gluc, smoke, alco, active)
        if errors:
            return jsonify(error="Validation failed", details=errors), 422

        height_m = height / 100
        BMI  = weight / (height_m ** 2)
        MAP  = (2 * ap_lo + ap_hi) / 3
        PP   = ap_hi - ap_lo

        feats = _build_features_array(age, gender, height, weight, ap_hi, ap_lo,
                                      cholesterol, gluc, smoke, alco, active)
        prob  = float(model.predict_proba(feats)[0][1])
        level, advice = risk_level(prob)
        rf   = get_risk_factors(age, BMI, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
        tips = get_health_tips(BMI, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
        ea   = get_emergency_alert(ap_hi, ap_lo)
        shap = compute_shap_contributions(age, gender, height, weight, ap_hi, ap_lo,
                                          cholesterol, gluc, smoke, alco, active)

        prob_pct = round(prob * 100, 2)
        emergency_alert = ea
        if prob_pct >= 80 and not ea:
            emergency_alert = "HIGH RISK: Probability >= 80%. Immediate medical consultation strongly advised."

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_prediction({
            "patient_name": patient_name, "age": age, "gender": gender,
            "height": height, "weight": weight, "ap_hi": ap_hi, "ap_lo": ap_lo,
            "cholesterol": cholesterol, "gluc": gluc, "smoke": smoke,
            "alco": alco, "active": active,
            "bmi": round(BMI, 2), "map_value": round(MAP, 2),
            "pulse_pressure": round(PP, 2), "probability": prob_pct,
            "risk_level": level, "advice": advice,
            "created_at": generated_at, "user_id": session.get("user_id"),
        })

        return jsonify(
            patient_name=patient_name,
            probability=prob_pct,
            risk_level=level,
            advice=advice,
            bmi=round(BMI, 2),
            bmi_status=bmi_category(BMI),
            map_value=round(MAP, 2),
            pulse_pressure=round(PP, 2),
            risk_factors=rf,
            health_tips=tips,
            emergency_alert=emergency_alert,
            shap_data=[{"factor": f, "delta": d} for f, d in shap],
            generated_at=generated_at,
        )

    except KeyError as e:
        return jsonify(error=f"Missing required field: {e}"), 400
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/ecg", methods=["POST"])
def api_ecg():
    """
    REST API: ECG image analysis.
    POST multipart/form-data: ecg_image (file), patient_name (optional)
    Returns: {result_label, confidence, pattern, details, features, generated_at}
    """
    try:
        patient_name = request.form.get("patient_name", "Anonymous").strip() or "Anonymous"
        file = request.files.get("ecg_image")
        if not file or file.filename == "":
            return jsonify(error="No ECG image provided. Send as multipart 'ecg_image'."), 400
        if not allowed_file(file.filename):
            return jsonify(error="Invalid file type. Allowed: PNG, JPG, BMP, TIFF."), 400

        filename = secure_filename(file.filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = ts + filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.seek(0)
        file.save(save_path)

        file.seek(0)
        features, feature_info = extract_ecg_features_from_image(file)
        features = features.reshape(1, -1)
        proba = ecg_model.predict_proba(features)[0][1]

        if proba > 0.5:
            prediction   = 1
            confidence   = round(proba * 100, 2)
            result_label = "Possible Abnormal Pattern"
            pattern      = "abnormal"
            details      = ("The ECG waveform analysis suggests a possible cardiac abnormality. "
                            "This is a screening result only — please consult a cardiologist for clinical evaluation.")
        else:
            prediction   = 0
            confidence   = round((1 - proba) * 100, 2)
            result_label = "Normal Pattern"
            pattern      = "normal"
            details      = ("The ECG waveform appears within the normal screening range. "
                            "Regular cardiac checkups are still recommended.")

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_ecg_prediction({
            "patient_name": patient_name, "filename": filename,
            "result": str(prediction), "result_label": result_label,
            "confidence": confidence, "num_peaks": feature_info["num_peaks"],
            "rr_mean": feature_info["rr_mean"], "rr_std": feature_info["rr_std"],
            "mean_signal": feature_info["mean"], "std_signal": feature_info["std"],
            "created_at": generated_at,
        })

        return jsonify(
            patient_name=patient_name,
            result_label=result_label,
            pattern=pattern,
            confidence=confidence,
            details=details,
            features=feature_info,
            screening_note="This is an AI screening tool only, not a medical diagnosis.",
            generated_at=generated_at,
        )

    except Exception as e:
        return jsonify(error=str(e)), 500


def _row_to_dict(row):
    """Convert sqlite3.Row to a JSON-safe dict (bytes → str)."""
    d = dict(row)
    return {k: (v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v)
            for k, v in d.items()}


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    REST API: Fetch prediction history.
    GET params: limit (default 50), risk (filter by risk level), search (patient name)
    Returns: {records: [...], total, ecg_records: [...], ecg_total}
    """
    try:
        limit       = int(request.args.get("limit", 50))
        risk_filter = request.args.get("risk", "").strip()
        search      = request.args.get("search", "").strip()

        records  = get_all_predictions(search, risk_filter)[:limit]
        ecg_recs = get_all_ecg_predictions()[:limit]

        return jsonify(
            total=len(records),
            records=[_row_to_dict(r) for r in records],
            ecg_total=len(ecg_recs),
            ecg_records=[_row_to_dict(r) for r in ecg_recs],
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


def _safe_json(obj):
    """Recursively make an object JSON-safe (bytes → str, Row → dict)."""
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(i) for i in obj]
    return obj


@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    """REST API: Dashboard aggregate statistics."""
    try:
        data = get_dashboard_data()
        serializable = {k: v for k, v in data.items() if k != "recent_records"}
        return jsonify(_safe_json(serializable))
    except Exception as e:
        return jsonify(error=str(e)), 500


# =========================
# ADMIN PANEL
# =========================
@app.route("/admin")
@login_required
def admin_panel():
    conn = sqlite3.connect("heartcare.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as c FROM users")
    total_users = cursor.fetchone()["c"]
    cursor.execute("SELECT COUNT(*) as c FROM predictions")
    total_preds = cursor.fetchone()["c"]
    cursor.execute("SELECT COUNT(*) as c FROM predictions WHERE risk_level IN ('High Risk','Critical Risk')")
    high_risk_count = cursor.fetchone()["c"]
    cursor.execute("SELECT COUNT(*) as c FROM ecg_predictions")
    ecg_total = cursor.fetchone()["c"]
    cursor.execute("SELECT COUNT(*) as c FROM ecg_predictions WHERE result='1'")
    ecg_abnormal = cursor.fetchone()["c"]
    cursor.execute("SELECT ROUND(AVG(probability),1) as v FROM predictions")
    avg_risk = cursor.fetchone()["v"] or 0
    cursor.execute("SELECT COUNT(*) as c FROM predictions WHERE risk_level='Low Risk'")
    low = cursor.fetchone()["c"]
    cursor.execute("SELECT COUNT(*) as c FROM predictions WHERE risk_level='Moderate Risk'")
    moderate = cursor.fetchone()["c"]
    cursor.execute("SELECT COUNT(*) as c FROM predictions WHERE risk_level='High Risk'")
    high = cursor.fetchone()["c"]
    cursor.execute("SELECT COUNT(*) as c FROM predictions WHERE risk_level='Critical Risk'")
    critical = cursor.fetchone()["c"]

    # Most common risk factor
    cursor.execute("""
        SELECT
          SUM(CASE WHEN cholesterol>1 THEN 1 ELSE 0 END) as chol,
          SUM(CASE WHEN gluc>1 THEN 1 ELSE 0 END) as gluc,
          SUM(smoke) as smoke, SUM(alco) as alco,
          SUM(CASE WHEN active=0 THEN 1 ELSE 0 END) as inactive,
          SUM(CASE WHEN bmi>=30 THEN 1 ELSE 0 END) as obese,
          SUM(CASE WHEN ap_hi>=140 OR ap_lo>=90 THEN 1 ELSE 0 END) as hbp
        FROM predictions
    """)
    rf_row = cursor.fetchone()
    rf_counts = {
        "High Cholesterol": rf_row["chol"] or 0,
        "High Glucose":     rf_row["gluc"] or 0,
        "Smoking":          rf_row["smoke"] or 0,
        "Alcohol":          rf_row["alco"] or 0,
        "Inactivity":       rf_row["inactive"] or 0,
        "Obesity":          rf_row["obese"] or 0,
        "High BP":          rf_row["hbp"] or 0,
    }
    top_rf = max(rf_counts, key=rf_counts.get) if total_preds > 0 else "N/A"

    # Recent users
    cursor.execute("SELECT id, username, email, created_at FROM users ORDER BY id DESC LIMIT 10")
    recent_users = cursor.fetchall()

    # Weekly prediction trend
    cursor.execute("""
        SELECT DATE(created_at) as day, COUNT(*) as cnt
        FROM predictions
        GROUP BY DATE(created_at)
        ORDER BY day DESC LIMIT 7
    """)
    trend_rows = list(reversed(cursor.fetchall()))
    trend_labels = [r["day"] for r in trend_rows]
    trend_counts = [r["cnt"] for r in trend_rows]

    conn.close()
    return render_template("admin.html",
        total_users=total_users, total_preds=total_preds,
        high_risk_count=high_risk_count, ecg_total=ecg_total,
        ecg_abnormal=ecg_abnormal, avg_risk=avg_risk,
        low=low, moderate=moderate, high=high, critical=critical,
        top_rf=top_rf, rf_counts=rf_counts,
        recent_users=recent_users,
        trend_labels=json.dumps(trend_labels),
        trend_counts=json.dumps(trend_counts),
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
