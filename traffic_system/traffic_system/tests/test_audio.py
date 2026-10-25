"""Audio tests. Synthetic checks always run; CNN check runs if models/siren_cnn.pt exists;
set ESC50=/path/to/ESC-50-master to also run a real-data check."""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, audio
sr = 16000; rng = np.random.default_rng(0)
def noise(sec): return 0.02 * rng.standard_normal(int(sec * sr))
def wail(sec, f0=700, f1=1500, rate=0.5):
    t = np.arange(int(sec * sr)) / sr
    f = f0 + (f1 - f0) * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
    return 0.4 * np.sin(2 * np.pi * np.cumsum(f) / sr)
def tone(sec, f=1000):
    t = np.arange(int(sec * sr)) / sr; return 0.4 * np.sin(2 * np.pi * f * t)

# spectral fallback
x = np.concatenate([noise(3), wail(4) + noise(4), noise(3)]).astype(np.float32)
iv = audio.siren_windows(x, sr); assert iv and iv[0][0] >= 2.0 and iv[-1][1] <= 8.0, "siren not localised"
assert audio.siren_windows(noise(8).astype(np.float32), sr) == [], "false positive on noise"
assert audio.siren_windows((tone(6) + noise(6)).astype(np.float32), sr) == [], "false positive on steady horn"
print("spectral detector OK")

import siren_model as SM
f = SM.mfcc(noise(2)); assert f.shape[0] == 2 and f.shape[1] == SM.N_MFCC and np.isfinite(f).all(); print("MFCC features OK", f.shape)
if SM.available():
    _, p = SM.predict_windows(noise(6).astype(np.float32), sr); assert p.max() < 0.5, "CNN false alarm on white noise"
    print("CNN loaded, silent/noise negative OK; detector =", audio.detector_name())
    root = os.getenv("ESC50")
    if root:
        from scipy.io import wavfile
        rows = list(csv.DictReader(open(os.path.join(root, "meta", "esc50.csv"))))
        def clip(fn):
            s, x = wavfile.read(os.path.join(root, "audio", fn)); return SM.resample(x.astype(np.float32) / 32768, s)
        sir = [r["filename"] for r in rows if r["category"] == "siren"][:10]
        hit = sum(len(audio.detect(clip(f), SM.SR)) > 0 for f in sir)
        print(f"real ESC-50 siren clips detected: {hit}/{len(sir)} (these were in the training set)")
        mon = audio.LiveSirenMonitor(); s = clip(sir[0]); flags = []
        for i in range(0, len(s) - 8000, 8000):        # 0.5 s chunks like a live stream
            mon.feed(s[i:i + 8000]); flags.append(mon.check())
        print("live monitor flags over stream:", flags)
else:
    print("no CNN model file - skipped CNN checks")
mon = audio.LiveSirenMonitor()
for i in range(0, 16000 * 8, 8000):                    # live monitor stays quiet on noise
    mon.feed(noise(0.5).astype(np.float32)); mon.check()
assert not mon.active; print("AUDIO TESTS PASSED")
