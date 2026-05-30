"""
build_ecg_model.py  —  HeartCare AI
====================================
Trains a binary ECG classifier (0=Normal, 1=Abnormal) using synthetic ECG
signals that also simulate real-world photo/scan artifacts so the model
generalises to uploaded ECG printout images.

Run:
    python build_ecg_model.py

Output:
    ecg_binary_model.pkl   (replaces the old one)
"""

import numpy as np
import scipy.stats as sci_stats
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

FS = 250          # Simulated sampling frequency (Hz)
DURATION = 10     # Seconds per signal
N = FS * DURATION # Total samples


# ──────────────────────────────────────────────────────────────────────────────
# 1. SIGNAL GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_wave(t, center, duration, amplitude):
    """Add a single wave (P, Q, R, S, T) to the signal array."""
    return amplitude * np.exp(-0.5 * ((t - center) / (duration / 4)) ** 2)


def generate_normal_ecg(hr=None, n=N, fs=FS):
    """
    Synthesise a normal sinus rhythm ECG signal.
    HR defaults to a random value in 60–100 BPM.

    Key design principle:
      In a healthy heart, R-peak amplitude is CONSISTENT beat-to-beat (low
      peak_amp_std).  Each beat gets a base amplitude fixed for the whole
      recording, with only ±5 % physiological variation.  This is the single
      most important feature for distinguishing normal from abnormal.
    """
    if hr is None:
        hr = np.random.uniform(60, 100)

    t = np.linspace(0, n / fs, n)
    signal = np.zeros(n)
    beat_interval = fs * 60.0 / hr

    # Fix ONE base amplitude for the entire recording (realistic for normal)
    r_base = np.random.uniform(0.8, 1.5)
    p_base = r_base * np.random.uniform(0.12, 0.20)
    t_base = r_base * np.random.uniform(0.25, 0.35)

    pos = np.random.uniform(0, beat_interval)
    while pos < n:
        bt = pos / fs

        # Small ±5 % beat-to-beat variation only (normal sinus rhythm)
        r_amp = r_base * np.random.uniform(0.95, 1.05)

        signal += _gaussian_wave(t, bt + 0.09,  0.08,  p_base)
        signal -= _gaussian_wave(t, bt + 0.155, 0.02,  r_amp * 0.10)
        signal += _gaussian_wave(t, bt + 0.175, 0.025, r_amp)
        signal -= _gaussian_wave(t, bt + 0.20,  0.02,  r_amp * 0.20)
        signal += _gaussian_wave(t, bt + 0.32,  0.12,  t_base)

        pos += beat_interval * np.random.uniform(0.97, 1.03)   # mild HRV

    signal += 0.02 * np.sin(2 * np.pi * 0.15 * t)
    signal += np.random.normal(0, 0.02, n)
    return signal


def generate_abnormal_ecg(abnormality=None, n=N, fs=FS):
    """
    Synthesise an abnormal ECG signal.
    Supported abnormalities match real clinical patterns:
      tachycardia, bradycardia, afib, st_elevation, st_depression,
      lbbb, rbbb, pvcs, inverted_t, wide_qrs
    """
    if abnormality is None:
        abnormality = np.random.choice([
            'tachycardia', 'bradycardia', 'afib',
            'st_elevation', 'st_depression',
            'lbbb', 'rbbb', 'pvcs', 'inverted_t', 'wide_qrs'
        ])

    t = np.linspace(0, n / fs, n)
    signal = np.zeros(n)

    if abnormality == 'tachycardia':
        # HR > 100 BPM
        signal = generate_normal_ecg(hr=np.random.uniform(110, 170), n=n, fs=fs)

    elif abnormality == 'bradycardia':
        # HR < 60 BPM
        signal = generate_normal_ecg(hr=np.random.uniform(30, 55), n=n, fs=fs)

    elif abnormality == 'afib':
        # Atrial fibrillation: no P waves, irregularly irregular RR intervals,
        # AND highly variable R-peak amplitude → high peak_amp_std + high rr_cv
        pos = 0.0
        while pos < n:
            interval = np.random.uniform(0.35, 1.3) * fs
            bp = int(pos)
            r_dur = int(0.04 * fs)
            # Each beat has a very different amplitude (chaotic conduction)
            r_amp = np.random.uniform(0.3, 1.6)
            for i in range(r_dur):
                idx = bp + int(0.05 * fs) + i
                if 0 <= idx < n:
                    signal[idx] += r_amp * np.sin(np.pi * i / r_dur)
            pos += interval
        # Fibrillatory baseline
        signal += 0.12 * np.random.normal(0, 1, n)

    elif abnormality == 'st_elevation':
        signal = generate_normal_ecg(n=n, fs=fs)
        beat_interval = int(fs * 60.0 / np.random.uniform(60, 100))
        for bp in range(0, n, beat_interval):
            st_s = bp + int(0.20 * fs)
            st_e = bp + int(0.28 * fs)
            for i in range(st_s, min(st_e, n)):
                signal[i] += np.random.uniform(0.2, 0.6)   # raised ST

    elif abnormality == 'st_depression':
        signal = generate_normal_ecg(n=n, fs=fs)
        beat_interval = int(fs * 60.0 / np.random.uniform(60, 100))
        for bp in range(0, n, beat_interval):
            st_s = bp + int(0.20 * fs)
            st_e = bp + int(0.28 * fs)
            for i in range(st_s, min(st_e, n)):
                signal[i] -= np.random.uniform(0.1, 0.35)  # depressed ST

    elif abnormality == 'lbbb':
        # Left bundle branch block: wide QRS (>120 ms), notched (M-shape), inverted T
        beat_interval = int(fs * 60.0 / np.random.uniform(60, 100))
        for bp in range(0, n, beat_interval):
            r_dur = int(0.14 * fs)   # wide QRS
            r_amp = np.random.uniform(0.6, 1.2)
            for i in range(r_dur):
                idx = bp + int(0.14 * fs) + i
                if 0 <= idx < n:
                    signal[idx] += r_amp * (
                        np.sin(np.pi * i / r_dur) +
                        0.35 * np.sin(2 * np.pi * i / r_dur)
                    )
            # Inverted T
            t_dur = int(0.15 * fs)
            for i in range(t_dur):
                idx = bp + int(0.30 * fs) + i
                if 0 <= idx < n:
                    signal[idx] -= np.random.uniform(0.15, 0.35) * np.sin(np.pi * i / t_dur)
        signal += np.random.normal(0, 0.03, n)

    elif abnormality == 'rbbb':
        # Right bundle branch block: RSR' pattern in V1
        beat_interval = int(fs * 60.0 / np.random.uniform(60, 100))
        for bp in range(0, n, beat_interval):
            off = int(0.14 * fs)
            for i in range(int(0.04 * fs)):
                if 0 <= bp + off + i < n:
                    signal[bp + off + i] += 1.0 * np.sin(np.pi * i / int(0.04 * fs))
            for i in range(int(0.03 * fs)):
                if 0 <= bp + int(0.18 * fs) + i < n:
                    signal[bp + int(0.18 * fs) + i] -= 0.3
            for i in range(int(0.05 * fs)):
                if 0 <= bp + int(0.21 * fs) + i < n:
                    signal[bp + int(0.21 * fs) + i] += 0.55 * np.sin(np.pi * i / int(0.05 * fs))
        signal += np.random.normal(0, 0.03, n)

    elif abnormality == 'pvcs':
        # Premature ventricular contractions: large ectopic beats mixed with
        # normal ones → very high peak_amp_std (big height difference)
        signal = generate_normal_ecg(hr=np.random.uniform(60, 90), n=n, fs=fs)
        beat_interval = int(fs * 60.0 / np.random.uniform(60, 90))
        n_pvcs = np.random.randint(4, 10)
        locs = np.random.choice(range(beat_interval, n - beat_interval), n_pvcs, replace=False)
        for pp in locs:
            dur = int(0.14 * fs)   # wider + taller than normal
            pvc_amp = np.random.uniform(1.8, 2.8)
            for i in range(dur):
                if 0 <= pp + i < n:
                    signal[pp + i] += pvc_amp * np.sin(np.pi * i / dur) * (
                        -1 if i < dur // 2 else 1)

    elif abnormality == 'inverted_t':
        # T-wave inversion (exactly what this patient's ECG shows: V1/V2/V3)
        signal = generate_normal_ecg(n=n, fs=fs)
        beat_interval = int(fs * 60.0 / np.random.uniform(60, 100))
        for bp in range(0, n, beat_interval):
            t_dur = int(0.12 * fs)
            t_amp = np.random.uniform(0.25, 0.45)
            for i in range(t_dur):
                idx = bp + int(0.28 * fs) + i
                if 0 <= idx < n:
                    # Flip T wave from positive to negative
                    signal[idx] -= 2 * t_amp * np.sin(np.pi * i / t_dur)

    elif abnormality == 'wide_qrs':
        # Generic wide QRS (>120 ms) — common in ventricular conduction defects
        beat_interval = int(fs * 60.0 / np.random.uniform(60, 100))
        for bp in range(0, n, beat_interval):
            signal += _gaussian_wave(t, (bp + int(0.09 * fs)) / fs, 0.08,
                                     np.random.uniform(0.10, 0.20))
            r_dur = int(0.15 * fs)
            r_amp = np.random.uniform(0.7, 1.3)
            for i in range(r_dur):
                idx = bp + int(0.16 * fs) + i
                if 0 <= idx < n:
                    signal[idx] += r_amp * np.sin(np.pi * i / r_dur)
        signal += np.random.normal(0, 0.04, n)

    # Baseline wander + noise for all types
    signal += 0.02 * np.sin(2 * np.pi * 0.15 * t)
    signal += np.random.normal(0, 0.04, n)
    return signal


# ──────────────────────────────────────────────────────────────────────────────
# 2. SIMULATE PHOTO/SCAN ARTIFACTS
#    This bridges the domain gap between clean signals and real ECG images.
# ──────────────────────────────────────────────────────────────────────────────

def simulate_image_artifacts(signal, fs=FS):
    """
    Apply realistic distortions that occur when the feature-extraction code
    processes a real-world photograph or scan of an ECG printout.
    """
    n = len(signal)

    # (a) ECG grid noise — the regular grid paper pattern leaks into the signal
    grid_freq = np.random.uniform(4, 6)   # ~5 Hz grid harmonic
    signal = signal + np.random.uniform(0.02, 0.08) * np.sin(
        2 * np.pi * grid_freq * np.linspace(0, n / fs, n))

    # (b) Perspective / compression distortion — tilted paper
    stretch = np.random.uniform(0.93, 1.07)
    stretched_len = int(n * stretch)
    stretched = np.interp(
        np.linspace(0, n - 1, stretched_len),
        np.arange(n), signal)
    signal = np.interp(
        np.linspace(0, stretched_len - 1, n),
        np.arange(stretched_len), stretched)

    # (c) Camera / scanner noise
    signal += np.random.normal(0, np.random.uniform(0.03, 0.09), n)

    # (d) Baseline drift from uneven paper holding
    drift_amp = np.random.uniform(0.0, 0.15)
    signal += drift_amp * np.linspace(0, 1, n)

    return signal


# ──────────────────────────────────────────────────────────────────────────────
# 3. FEATURE EXTRACTION  (20 features — must match app.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(signal, fs=FS):
    """
    Extract 20 features from a 1-D ECG signal.
    The order and count must exactly match extract_ecg_features_from_image()
    in app.py.
    """
    # Normalise
    sig_mean = signal.mean()
    sig_std  = signal.std()
    if sig_std > 1e-6:
        sig = (signal - sig_mean) / sig_std
    else:
        sig = signal - sig_mean

    # ── Time-domain ──────────────────────────────────────────────────────────
    mean_val   = float(np.mean(sig))
    std_val    = float(np.std(sig))
    max_val    = float(np.max(sig))
    min_val    = float(np.min(sig))
    median_val = float(np.median(sig))
    range_val  = float(max_val - min_val)
    rms_val    = float(np.sqrt(np.mean(sig ** 2)))
    energy_val = float(np.sum(sig ** 2))
    skew_val   = float(sci_stats.skew(sig))
    kurt_val   = float(sci_stats.kurtosis(sig))

    # ── Peak / rhythm ─────────────────────────────────────────────────────────
    peaks, _ = find_peaks(sig, height=0.3, distance=max(5, len(sig) // 80))
    num_peaks = len(peaks)

    if len(peaks) > 1:
        rr = np.diff(peaks).astype(float)
        rr_mean = float(np.mean(rr))
        rr_std  = float(np.std(rr))
        rr_cv   = rr_std / (rr_mean + 1e-6)   # HRV: coefficient of variation
    else:
        rr_mean = 0.0
        rr_std  = 0.0
        rr_cv   = 0.0

    # ── Frequency-domain ─────────────────────────────────────────────────────
    fft_mag = np.abs(np.fft.rfft(sig))
    freqs   = np.fft.rfftfreq(len(sig), d=1.0 / fs)

    dom_freq = float(freqs[np.argmax(fft_mag)]) if len(fft_mag) > 0 else 0.0

    lf_mask = (freqs >= 0.5) & (freqs < 5.0)
    hf_mask = (freqs >= 5.0) & (freqs < 40.0)
    lf_power    = float(np.sum(fft_mag[lf_mask] ** 2)) if lf_mask.any() else 0.0
    hf_power    = float(np.sum(fft_mag[hf_mask] ** 2)) if hf_mask.any() else 0.0
    lf_hf_ratio = lf_power / (hf_power + 1e-6)

    psd      = fft_mag ** 2
    psd_norm = psd / (psd.sum() + 1e-6)
    spec_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-6)))

    # ── Amplitude variability of R-peaks ─────────────────────────────────────
    if num_peaks > 1:
        peak_heights = sig[peaks]
        peak_amp_std = float(np.std(peak_heights))
    else:
        peak_amp_std = 0.0

    return np.array([
        mean_val, std_val, max_val, min_val, median_val,
        range_val, rms_val, energy_val, skew_val, kurt_val,
        num_peaks, rr_mean, rr_std, rr_cv,
        dom_freq, lf_power, hf_power, lf_hf_ratio,
        spec_entropy, peak_amp_std
    ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# 4. DATASET GENERATION
# ──────────────────────────────────────────────────────────────────────────────

ABNORMALITIES = [
    'tachycardia', 'bradycardia', 'afib',
    'st_elevation', 'st_depression',
    'lbbb', 'rbbb', 'pvcs', 'inverted_t', 'wide_qrs'
]


def generate_non_ecg_signal(n=N):
    """
    Generate a 1-D signal that mimics what the image feature extractor produces
    when processing a NON-ECG image (receipt, photo, document, etc.).

    These are labelled 0 (Normal) during training because the validate_ecg_image()
    guard in app.py catches them first. However, training on their features makes
    the model assign lower confidence to borderline non-ECG inputs that slip
    through, acting as a safety net.

    Characteristics: random spiky noise, very irregular spacing, no true
    periodic rhythm — matching what a payment slip or photograph produces.
    """
    kind = np.random.choice(['text_doc', 'random_photo', 'uniform_region'])

    if kind == 'text_doc':
        # Simulates a document/receipt: sharp random spikes from text rows
        sig = np.random.normal(0, 0.1, n)
        n_spikes = np.random.randint(40, 120)
        locs = np.random.choice(n, n_spikes, replace=False)
        sig[locs] += np.random.choice([-1, 1], n_spikes) * np.random.uniform(0.5, 2.5, n_spikes)

    elif kind == 'random_photo':
        # Simulates a random photo: low-frequency trend + coloured noise
        t = np.linspace(0, 1, n)
        sig = (np.random.uniform(0.2, 1.0) * np.sin(2 * np.pi * np.random.uniform(0.5, 3) * t)
               + np.random.normal(0, 0.3, n))

    else:
        # Simulates a near-uniform region: almost flat with small random bumps
        sig = np.random.normal(0, 0.05, n)
        sig += np.random.uniform(-0.5, 0.5)

    return sig


def generate_dataset(n_normal=800, n_abnormal=800):
    X, y = [], []

    print(f"  Generating {n_normal} normal ECG samples …")
    for _ in range(n_normal):
        sig = generate_normal_ecg()
        sig = simulate_image_artifacts(sig)
        X.append(extract_features(sig))
        y.append(0)

    print(f"  Generating {n_abnormal} abnormal ECG samples …")
    for i in range(n_abnormal):
        abn = ABNORMALITIES[i % len(ABNORMALITIES)]
        sig = generate_abnormal_ecg(abnormality=abn)
        sig = simulate_image_artifacts(sig)
        X.append(extract_features(sig))
        y.append(1)

    # Non-ECG samples labelled Normal (0) — teaches the model these features
    # are NOT confidently abnormal, so confidence drops for junk images.
    n_non_ecg = n_normal // 5   # 20 % of normal count
    print(f"  Generating {n_non_ecg} non-ECG noise samples (safety net) …")
    for _ in range(n_non_ecg):
        sig = generate_non_ecg_signal()
        X.append(extract_features(sig))
        y.append(0)

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAIN & SAVE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HeartCare AI — ECG Binary Model Builder")
    print("=" * 60)

    print("\n[1/3] Generating synthetic dataset …")
    X, y = generate_dataset(n_normal=800, n_abnormal=800)
    print(f"      Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"      Normal: {(y==0).sum()}   Abnormal: {(y==1).sum()}")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4,
        ))
    ])

    print("\n[2/3] Cross-validating (5-fold) …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"      CV Accuracy : {scores.mean():.3f} ± {scores.std():.3f}")
    roc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"      CV ROC-AUC  : {roc.mean():.3f} ± {roc.std():.3f}")

    print("\n[3/3] Training final model on full dataset …")
    model.fit(X, y)

    out = 'ecg_binary_model.pkl'
    joblib.dump(model, out)
    print(f"\n  Saved: {out}")
    print("=" * 60)
    print("  Next: restart app.py -- the new model will be loaded automatically.")
    print("=" * 60)


if __name__ == "__main__":
    main()
