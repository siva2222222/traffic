"""MFCC + CNN siren classifier (as in the architecture diagram).
- features: MFCC (20) + delta, computed with numpy/scipy only (no librosa)
- model: small 4-block CNN (PyTorch) -> P(siren) for a 2-second window
- inference: `predict_windows` / `siren_intervals`; model file: models/siren_cnn.pt (train with train_siren.py)"""
import os
import numpy as np
from math import gcd
from scipy.fft import dct
from scipy.signal import resample_poly

SR, WIN, HOP_S = 16000, 2.0, 0.5
N_FFT, HOP, N_MELS, N_MFCC = 512, 160, 40, 20
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "siren_cnn.pt")

def resample(x, sr):
    if sr == SR:
        return x.astype(np.float32)
    g = gcd(int(sr), SR)
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)

def _hz2mel(f): return 2595 * np.log10(1 + f / 700.0)
def _mel2hz(m): return 700 * (10 ** (m / 2595.0) - 1)

_FB = None
def _filterbank():
    global _FB
    if _FB is None:
        mels = np.linspace(_hz2mel(0), _hz2mel(SR / 2), N_MELS + 2)
        bins = np.floor((N_FFT + 1) * _mel2hz(mels) / SR).astype(int)
        fb = np.zeros((N_MELS, N_FFT // 2 + 1), np.float32)
        for i in range(1, N_MELS + 1):
            l, c, r = bins[i - 1], bins[i], bins[i + 1]
            for k in range(l, c):
                fb[i - 1, k] = (k - l) / max(1, c - l)
            for k in range(c, r):
                fb[i - 1, k] = (r - k) / max(1, r - c)
        _FB = fb
    return _FB

def mfcc(x):
    """1-D float audio at 16 kHz -> (2, N_MFCC, T): MFCC and delta, mean-normalised over time."""
    x = np.asarray(x, np.float32)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * np.hanning(N_FFT).astype(np.float32)
    pw = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    mel = np.log(pw @ _filterbank().T + 1e-8)
    m = dct(mel, type=2, axis=1, norm="ortho")[:, :N_MFCC].T            # (N_MFCC, T)
    m = m - m.mean(1, keepdims=True)                                     # loudness / channel invariance
    d = np.gradient(m, axis=1)
    return np.stack([m, d]).astype(np.float32)

def windows(x, win=WIN, hop=HOP_S):
    n, h = int(win * SR), int(hop * SR)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    starts = list(range(0, len(x) - n + 1, h))
    return starts, [x[s:s + n] for s in starts]

def build_model():
    import torch.nn as nn
    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(2, 16), blk(16, 32), blk(32, 64), blk(64, 64),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

_MODEL = None
def load_model(path=MODEL_PATH):
    global _MODEL
    if _MODEL is None and os.path.exists(path):
        try:
            import torch
            m = build_model()
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            m.eval()
            _MODEL = m
        except Exception as e:
            print("siren CNN not loaded:", e)
    return _MODEL

def available():
    return load_model() is not None

def predict_windows(x, sr=SR):
    """Return (start_seconds[], prob[]) for 2 s windows (hop 0.5 s)."""
    import torch
    x = resample(np.asarray(x, np.float32), sr)
    starts, ws = windows(x)
    feats = np.stack([mfcc(w) for w in ws])
    with torch.no_grad():
        p = torch.sigmoid(load_model()(torch.from_numpy(feats))).squeeze(1).numpy()
    return np.array(starts) / SR, p

def siren_intervals(x, sr=SR, thr=0.5, min_windows=2):
    """Merged [(start_s, end_s)] where >= `min_windows` consecutive windows exceed `thr` (and audio is not silent)."""
    t, p = predict_windows(x, sr)
    x16 = resample(np.asarray(x, np.float32), sr)
    hit = []
    for s, pr in zip(t, p):
        seg = x16[int(s * SR):int((s + WIN) * SR)]
        hit.append(pr >= thr and np.sqrt(np.mean(seg ** 2)) > 0.003)
    out, i = [], 0
    while i < len(hit):
        if hit[i]:
            j = i
            while j + 1 < len(hit) and hit[j + 1]:
                j += 1
            if j - i + 1 >= min_windows:
                out.append((float(t[i]), float(t[j] + WIN)))
            i = j + 1
        else:
            i += 1
    return out
