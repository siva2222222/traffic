"""Train the MFCC+CNN siren detector on ESC-50 (https://github.com/karolpiczak/ESC-50, CC BY-NC 4.0).
  python train_siren.py --esc50 path/to/ESC-50-master --eval      # hold out fold 5 and report metrics
  python train_siren.py --esc50 path/to/ESC-50-master             # train on all folds and save models/siren_cnn.pt
Your own data: put extra siren clips in --extra_siren DIR and non-siren clips in --extra_other DIR (wav)."""
import os, argparse, csv, glob, random
import numpy as np
from scipy.io import wavfile
import torch, torch.nn as nn
import siren_model as SM

HARD_NEG = {"car_horn", "church_bells", "crying_baby", "rooster", "cat", "train", "hand_saw", "vacuum_cleaner",
            "engine", "helicopter", "airplane", "clock_alarm", "door_wood_knock", "crow", "hen", "chainsaw"}

def load_wav(p):
    sr, x = wavfile.read(p)
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    if x.ndim > 1:
        x = x.mean(1)
    return SM.resample(x, sr)

def crop(x, n, active=True):
    if len(x) <= n:
        return np.pad(x, (0, n - len(x)))
    for _ in range(12):
        s = random.randint(0, len(x) - n)
        c = x[s:s + n]
        if not active or np.sqrt(np.mean(c ** 2)) > 0.01:
            return c
    return c

def mix(sig, noise, snr_db):
    ps, pn = np.mean(sig ** 2) + 1e-9, np.mean(noise ** 2) + 1e-9
    return sig + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", required=True); ap.add_argument("--eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--out", default=SM.MODEL_PATH)
    ap.add_argument("--extra_siren"); ap.add_argument("--extra_other")
    a = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    meta = list(csv.DictReader(open(os.path.join(a.esc50, "meta", "esc50.csv"))))
    data = []
    for r in meta:
        data.append((load_wav(os.path.join(a.esc50, "audio", r["filename"])), r["category"], int(r["fold"])))
    for d, cat in ((a.extra_siren, "siren"), (a.extra_other, "other")):
        for p in (glob.glob(os.path.join(d, "*.wav")) if d else []):
            data.append((load_wav(p), cat, 0))
    val = [d for d in data if d[2] == 5] if a.eval else []
    tr = [d for d in data if d[2] != 5] if a.eval else data
    sir = [d[0] for d in tr if d[1] == "siren"]
    oth = [d[0] for d in tr if d[1] != "siren"]
    hard = [d[0] for d in tr if d[1] in HARD_NEG]
    print(f"train: {len(sir)} siren clips, {len(oth)} other clips ({len(hard)} hard negatives); val clips: {len(val)}")
    N = int(SM.WIN * SM.SR)

    def epoch_set():
        X, Y = [], []
        for c in sir:
            for _ in range(14):
                w = crop(c, N)
                if random.random() < 0.7:                                     # siren over background noise (traffic-like)
                    w = mix(w, crop(random.choice(oth), N, False), random.uniform(-3, 18))
                w = w * random.uniform(0.3, 1.5)
                X.append(SM.mfcc(w)); Y.append(1)
        for c in oth + random.sample(hard, min(len(hard), 400)):
            w = crop(c, N, False) * random.uniform(0.3, 1.5)
            X.append(SM.mfcc(w)); Y.append(0)
        return torch.from_numpy(np.stack(X)), torch.tensor(Y, dtype=torch.float32)

    model = SM.build_model()
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=a.epochs * 40)  # upper bound of steps
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
    step = 0
    for ep in range(a.epochs):
        X, Y = epoch_set(); perm = torch.randperm(len(X)); model.train(); tot = 0
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            opt.zero_grad(); loss = lossf(model(X[idx]).squeeze(1), Y[idx]); loss.backward(); opt.step()
            if step < a.epochs * 40 - 1: sched.step(); step += 1
            tot += loss.item() * len(idx)
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(X):.4f}", flush=True)
    model.eval()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(model.state_dict(), a.out); print("saved", a.out)

    if val:                                                                  # held-out evaluation on fold 5, clip level
        SM._MODEL = model
        for thr in (0.5, 0.7, 0.85):
            tp = fn = fp = tn = 0
            for x, cat, _ in val:
                pos = len(SM.siren_intervals(x, SM.SR, thr=thr, min_windows=2)) > 0
                if cat == "siren": tp += pos; fn += (not pos)
                else: fp += pos; tn += (not pos)
            print(f"[fold-5 clips] thr={thr}: siren detected {tp}/{tp + fn} | false alarms {fp}/{fp + tn} "
                  f"({100 * fp / max(1, fp + tn):.1f}%)")
        hn = [(x, c) for x, c, _ in val if c in ("car_horn", "engine", "train", "helicopter", "church_bells", "crying_baby")]
        fa = sum(len(SM.siren_intervals(x, SM.SR, thr=0.7)) > 0 for x, _ in hn)
        print(f"[fold-5 hard negatives: horn/engine/train/helicopter/bells/baby] false alarms {fa}/{len(hn)} at thr=0.7")

if __name__ == "__main__":
    main()
