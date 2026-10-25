"""Siren detection from audio (spectral analysis: strong tonal peak in 500-1800 Hz whose frequency sweeps/alternates).
No dataset needed. For higher accuracy, train an MFCC+CNN on UrbanSound8K/ESC-50 and swap in for `siren_windows`."""
import os, shutil, subprocess, tempfile, wave
import numpy as np
from scipy.signal import spectrogram

def extract_audio(path):
    """Any audio/video file -> 16 kHz mono wav (needs ffmpeg on PATH). Returns wav path or None."""
    if not shutil.which("ffmpeg"):
        return None
    out = tempfile.mktemp(suffix=".wav")
    r = subprocess.run(["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out],
                       capture_output=True)
    if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 2000:
        return out
    return None

def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("Need 16-bit PCM WAV")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr

def siren_windows(x, sr, win=1.5, hop=0.5):
    """Return merged [(start_s, end_s)] intervals where a siren-like sound is present."""
    if len(x) < sr * win:
        return []
    f, t, S = spectrogram(x, fs=sr, nperseg=1024, noverlap=768)
    band = (f >= 500) & (f <= 1800)
    fb, Sb = f[band], S[band]
    tot = S.sum(0) + 1e-12
    dom = fb[Sb.argmax(0)]
    prom = Sb.max(0) / (Sb.mean(0) + 1e-12)
    frac = Sb.sum(0) / tot
    ok = (prom > 8) & (frac > 0.35)
    hits = []
    for start in np.arange(0, t[-1] - win + hop, hop):
        m = (t >= start) & (t < start + win)
        if m.sum() < 5:
            continue
        seg = x[int(start * sr):int((start + win) * sr)]
        if np.sqrt(np.mean(seg ** 2)) < 0.005:
            continue
        if ok[m].mean() >= 0.6:
            d = dom[m][ok[m]]
            if np.percentile(d, 95) - np.percentile(d, 5) >= 100:   # sweeping/alternating tone (not a steady horn)
                hits.append((float(start), float(start + win)))
    merged = []
    for a, b in hits:
        if merged and a <= merged[-1][1] + 0.01:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged

def detector_name():
    try:
        import siren_model
        return "MFCC+CNN" if siren_model.available() else "spectral"
    except Exception:
        return "spectral"

def detect(x, sr):
    """Best available detector: MFCC+CNN (models/siren_cnn.pt) else spectral analysis."""
    try:
        import siren_model
        if siren_model.available():
            return siren_model.siren_intervals(x, sr)
    except Exception as e:
        print("CNN siren detector failed, using spectral:", e)
    return siren_windows(x, sr)

def analyse_file(path):
    """Path to audio/video -> (intervals, error). Used by the engine and the Source page."""
    wav = path if path.lower().endswith(".wav") else extract_audio(path)
    if not wav:
        return [], "No audio track / ffmpeg missing"
    try:
        x, sr = read_wav(wav)
        return detect(x, sr), None
    except Exception as e:
        return [], str(e)

class LiveSirenMonitor:
    """Live microphone / camera-audio siren monitor (needs `pip install sounddevice`).
    Every 0.5 s the last 2 s of audio is classified; `active` is True when >=2 of the last 3 checks say siren."""
    def __init__(self, device=None, sr=16000):
        import collections, threading
        self.device, self.sr = device, sr
        self.active, self.error = False, None
        self.hits = collections.deque(maxlen=3)
        self._buf = np.zeros(0, np.float32)
        self._stop = threading.Event()
        self._th = None

    def feed(self, chunk):
        """Add audio samples (float32 mono, self.sr). Returns self.active. Used by the stream callback and by tests."""
        self._buf = np.concatenate([self._buf, np.asarray(chunk, np.float32)])[-int(self.sr * 2.0):]
        return self.active

    def check(self):
        need = int(self.sr * 2.0)
        if len(self._buf) < need:
            return self.active
        x = self._buf[-need:]
        hit = False
        if np.sqrt(np.mean(x ** 2)) > 0.003:
            try:
                import siren_model
                if siren_model.available():                        # one 2 s window -> probability
                    hit = float(siren_model.predict_windows(x, self.sr)[1].max()) >= 0.5
                else:
                    hit = len(siren_windows(x, self.sr)) > 0
            except Exception:
                hit = len(siren_windows(x, self.sr)) > 0
        self.hits.append(hit)
        self.active = sum(self.hits) >= 2
        return self.active

    def start(self):
        import threading, time
        def run():
            try:
                import sounddevice as sd
                with sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", device=self.device,
                                    callback=lambda d, f, t, s: self.feed(d[:, 0])):
                    while not self._stop.is_set():
                        time.sleep(0.5)
                        self.check()
            except Exception as e:
                self.error = f"live audio unavailable: {e}"
        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
