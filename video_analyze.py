"""Scan an uploaded VIDEO for ambulances and fire engines, and report when each one appears.

This is the video counterpart of image_analyze.py. It is deliberately NOT the live dashboard
engine: no lane polygons, no stabiliser, no signal cycle, no violations. One job, done properly —
watch the video, find emergency vehicles, save a snapshot of each, report the timestamps.

Why a background job: CLIP costs ~60-120 ms per crop on CPU, so a one-minute clip takes a few
minutes to scan. The Flask route starts a job and returns immediately; the page polls for
progress. Nothing blocks.

Detection is per ByteTrack track and sticky — a vehicle is examined on up to CLIP_MAX_TRIES
separate frames as it approaches, and once flagged it is not re-checked. That is why video is
more reliable than a single still: one ambiguous look at distance does not lose the vehicle.

CLI:
    python video_analyze.py static/uploads/clip.mp4
"""
import os
import sys
import threading
import time
import uuid

import cv2
import numpy as np

from config import C
import emergency_vision as EV

BASE = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(BASE, "static", "evidence", "video_scan")

NAME_MAP = {"car": "car", "motorcycle": "bike", "motorbike": "bike", "bike": "bike",
            "bus": "bus", "truck": "truck", "ambulance": "ambulance",
            "fire_engine": "fire_engine", "fire engine": "fire_engine",
            "firetruck": "fire_engine", "fire truck": "fire_engine", "fire-truck": "fire_engine"}
EMERG_TYPES = ("ambulance", "fire_engine")
PRETTY = {"ambulance": "AMBULANCE", "fire_engine": "FIRE ENGINE"}
BOX_COLOR = {"ambulance": (0, 0, 255), "fire_engine": (0, 110, 255)}

_model = None
_ocr = None
_cnn = None


def resolve_path(path):
    """Accept a bare filename and find it where the videos actually live.

    'eng.mp4' typed from the project folder should work whether the file sits beside app.py
    or in static/uploads/, which is where the dashboard puts every upload.
    """
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for cand in (path, os.path.join(BASE, path), os.path.join(C.UPLOAD_DIR, os.path.basename(path))):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return path


def _open_error(path):
    """Say which of the two different failures this is: missing file, or unreadable codec."""
    if not os.path.exists(path):
        near = []
        for d in (BASE, C.UPLOAD_DIR):
            try:
                near += [os.path.join(os.path.basename(d), f) for f in os.listdir(d)
                         if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))]
            except OSError:
                pass
        hint = ("\n  Videos I can see:\n    " + "\n    ".join(sorted(set(near))[:12])) if near \
            else "\n  No video files found in the project folder or static/uploads/."
        return f"File not found: {path}{hint}"
    return (f"File exists but OpenCV cannot decode it: {path}\n"
            f"  Re-save it as H.264 MP4, or install a codec pack.")


def _yolo():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(C.IMAGE_YOLO_MODEL or C.YOLO_MODEL)
    return _model


def _cnn_predict(crop):
    global _cnn
    if _cnn is False:
        return None
    if _cnn is None:
        path = os.path.join(BASE, "models", "emergency_cnn.pt")
        if not os.path.exists(path):
            _cnn = False
            return None
        try:
            from emergency_classifier import EmergencyClassifier
            _cnn = EmergencyClassifier(path, min_conf=C.EMERG_CLF_MIN_CONF)
        except Exception:                                    # noqa: BLE001
            _cnn = False
            return None
    try:
        hit = _cnn.predict(crop)
        return (hit, C.EMERG_CLF_MIN_CONF, "cnn") if hit else None
    except Exception:                                        # noqa: BLE001
        return None


def _ocr_predict(crop):
    global _ocr
    if not C.ENABLE_OCR:
        return None
    try:
        if _ocr is None:
            import easyocr
            _ocr = easyocr.Reader(["en"], gpu=False)
        h, w = crop.shape[:2]
        scale = max(1.0, 240.0 / max(1, min(h, w)))
        img = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC) \
            if scale > 1.0 else crop
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        texts = []
        for v in (img, cv2.flip(img, 1), gray):
            texts += _ocr.readtext(v, detail=0)
        txt = " ".join(texts).upper()
        for etype, kws in C.OCR_KEYWORDS.items():
            for kw in kws:
                if kw in txt:
                    return etype, 1.0, f"ocr:{kw}"
        return None
    except Exception:                                        # noqa: BLE001
        return None


def _pad_crop(img, box, pad=0.08):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return img[int(max(0, y1 - bh * pad)):int(min(H, y2 + bh * pad)),
               int(max(0, x1 - bw * pad)):int(min(W, x2 + bw * pad))]


def _identify(crop, ycls, allow_clip=True, allow_ocr=True):
    """CLIP -> trained CNN -> OCR -> colour. First hit wins.

    allow_clip / allow_ocr come from the per-frame budgets: a track that only won an OCR slot
    must not also run CLIP, or the CLIP budget leaks and the scan slows down.
    """
    hit = EV.classify_crop(crop) if allow_clip else None
    hit = hit or _cnn_predict(crop)
    hit = hit or (_ocr_predict(crop) if allow_ocr else None)
    return hit or EV.color_hint(crop, ycls)


def _budget(dets, tracks, max_per_frame, min_h, max_tries, every_n, idx, key):
    """Largest eligible boxes first, capped — keeps a crowded frame from crawling on CPU.

    every_n/idx mirror engine.py's _budget_slots: without this stride gate, every eligible
    track ran CLIP (~60-120 ms/crop) and/or OCR (~300-800 ms/crop) on EVERY processed frame
    instead of every Nth one — a 6-8x unnecessary slowdown on video scans.
    """
    if max_per_frame <= 0 or idx % every_n:
        return set()
    cand = []
    for d in dets:
        if d["cls"] not in ("car", "bus", "truck"):
            continue
        tr = tracks.get(d["tid"])
        if tr and (tr.get("emerg") or tr.get(key, 0) >= max_tries):
            continue
        h = d["box"][3] - d["box"][1]
        if h > min_h:
            cand.append((h, d["tid"]))
    cand.sort(reverse=True)
    return {tid for _, tid in cand[:max_per_frame]}


def _snapshot(frame, box, label, conf, method, t, tid):
    """Save the exact frame a vehicle was flagged on, boxed and captioned. This is the evidence."""
    img = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    col = BOX_COLOR.get(label, (0, 0, 255))
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 4)
    tag = f"{PRETTY.get(label, label)} {conf:.2f} [{method}]"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = max(th + 8, y1)
    cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 10, ty), col, -1)
    cv2.putText(img, tag, (x1 + 5, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 42), (0, 0, 200), -1)
    cv2.putText(img, f"EMERGENCY VEHICLE at {int(t // 60):02d}:{t % 60:05.2f}  (track {tid})",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
    os.makedirs(SNAP_DIR, exist_ok=True)
    name = f"{label}_{tid}_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(os.path.join(SNAP_DIR, name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "evidence/video_scan/" + name


class ScanJob:
    """One background scan. Poll .snapshot() for progress and results."""

    def __init__(self, path, name="", on_detect=None, check_audio=True):
        self.id = uuid.uuid4().hex[:12]
        self.path = resolve_path(path)
        self.name = name or os.path.basename(self.path)
        self.on_detect = on_detect          # callback(detection dict) — used to write DB rows
        self.check_audio = check_audio
        self.lock = threading.Lock()
        self.state = "queued"
        self.error = None
        self.progress = 0.0
        self.frame_idx = 0
        self.total_frames = 0
        self.duration = 0.0
        self.detections = []
        self.siren_intervals = []
        self.vehicles_seen = 0
        self.started = time.time()
        self.finished = None
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop = True

    def snapshot(self):
        with self.lock:
            return {
                "id": self.id, "name": self.name, "state": self.state, "error": self.error,
                "progress": round(self.progress, 1), "frame": self.frame_idx,
                "total_frames": self.total_frames, "duration": round(self.duration, 1),
                "detections": list(self.detections),
                "emergency_types": sorted({d["type"] for d in self.detections}),
                "siren_intervals": [[round(a, 1), round(b, 1)] for a, b in self.siren_intervals],
                "vehicles_seen": self.vehicles_seen,
                "detector": EV.backend_name(),
                "elapsed": round((self.finished or time.time()) - self.started, 1),
            }

    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _run(self):
        try:
            self._set(state="loading model")
            model = _yolo()
            ids = [i for i, n in model.names.items() if n.lower() in NAME_MAP]

            if self.check_audio:
                self._set(state="analysing audio for sirens")
                try:
                    import audio as audiomod
                    iv, _err = audiomod.analyse_file(self.path)
                    self._set(siren_intervals=iv or [])
                except Exception:                            # noqa: BLE001
                    pass                                     # ffmpeg missing etc. — vision still runs

            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise RuntimeError(_open_error(self.path))
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) == 0:
                ok, _probe = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Opened '{self.path}' but could not read any frame — the file is "
                        f"empty or the codec is unsupported. Re-save it as H.264 MP4.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self._set(state="scanning", total_frames=total, duration=(total / fps) if total else 0.0)

            tracks = {}
            idx = 0
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if C.PROCESS_EVERY > 1 and idx % C.PROCESS_EVERY:
                    continue
                h0, w0 = frame.shape[:2]
                frame = cv2.resize(frame, (C.WORK_WIDTH, max(1, int(h0 * C.WORK_WIDTH / w0))))
                t = idx / fps

                res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=ids,
                                  conf=C.CONF, verbose=False)[0]
                dets = []
                if res.boxes is not None and res.boxes.id is not None:
                    for b, tid, c in zip(res.boxes.xyxy.cpu().numpy(),
                                         res.boxes.id.int().cpu().tolist(),
                                         res.boxes.cls.int().cpu().tolist()):
                        dets.append({"tid": int(tid), "box": tuple(float(v) for v in b),
                                     "cls": NAME_MAP[model.names[c].lower()]})

                clip_slots = _budget(dets, tracks, C.CLIP_MAX_PER_FRAME, C.CLIP_MIN_BOX_H,
                                     C.CLIP_MAX_TRIES, C.CLIP_EVERY_N, idx, "clip_tries") \
                    if (C.ENABLE_CLIP and EV.ready()) else set()
                ocr_slots = _budget(dets, tracks, C.OCR_MAX_PER_FRAME, C.OCR_MIN_BOX_H,
                                    C.OCR_MAX_TRIES, C.OCR_EVERY_N, idx, "ocr_tries") if C.ENABLE_OCR else set()

                for d in dets:
                    tr = tracks.setdefault(d["tid"], {"emerg": False, "clip_tries": 0, "ocr_tries": 0})
                    if tr["emerg"]:
                        continue
                    hit = None
                    if d["cls"] in EMERG_TYPES:                      # custom-trained YOLO class
                        hit = (d["cls"], 1.0, "yolo")
                    elif d["tid"] in clip_slots or d["tid"] in ocr_slots:
                        crop = _pad_crop(frame, d["box"])
                        if crop.size:
                            use_clip = d["tid"] in clip_slots
                            use_ocr = d["tid"] in ocr_slots
                            if use_clip:
                                tr["clip_tries"] += 1
                            if use_ocr:
                                tr["ocr_tries"] += 1
                            hit = _identify(crop, d["cls"], use_clip, use_ocr)
                    if hit and hit[0] in EMERG_TYPES:
                        label, conf, method = hit
                        tr["emerg"] = True
                        rel = _snapshot(frame, d["box"], label, float(conf), method, t, d["tid"])
                        det = {"t": round(t, 2),
                               "time": f"{int(t // 60):02d}:{t % 60:05.2f}",
                               "track": d["tid"], "type": label, "conf": round(float(conf), 3),
                               "method": method, "yolo_class": d["cls"], "snapshot": rel}
                        with self.lock:
                            self.detections.append(det)
                        if self.on_detect:
                            try:
                                self.on_detect(det)
                            except Exception:                        # noqa: BLE001
                                pass

                self._set(frame_idx=idx, vehicles_seen=len(tracks),
                          progress=(100.0 * idx / total) if total else 0.0)

            cap.release()
            self._set(state="stopped" if self._stop else "done", progress=100.0,
                      finished=time.time())
        except Exception as e:                                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._set(state="error", error=str(e), finished=time.time())


_jobs = {}
_jobs_lock = threading.Lock()


def start_job(path, name="", on_detect=None, check_audio=True):
    """Start a background scan.

    check_audio=False skips optional siren analysis to reduce processing time.
    Keep it True when the uploaded-video workflow also needs audio-based siren detection.
    """
    job = ScanJob(
        path,
        name=name,
        on_detect=on_detect,
        check_audio=check_audio,
    ).start()

    with _jobs_lock:
        _jobs[job.id] = job
        if len(_jobs) > 20:  # keep the last 20 scans only
            oldest = sorted(_jobs, key=lambda k: _jobs[k].started)[:-20]
            for k in oldest:
                _jobs.pop(k, None)
    return job


def get_job(job_id):
    with _jobs_lock:
        return _jobs.get(job_id)


def main():
    if len(sys.argv) < 2:
        print("Usage: python video_analyze.py <video>")
        sys.exit(1)
    job = start_job(sys.argv[1])
    last = -1
    while job.thread.is_alive():
        s = job.snapshot()
        if int(s["progress"]) != last:
            last = int(s["progress"])
            print(f"\r{s['state']}: {last}%  ({len(s['detections'])} emergency found)", end="", flush=True)
        time.sleep(0.5)
    s = job.snapshot()
    print(f"\n\nScanned {s['name']} in {s['elapsed']}s — detector CLIP={s['detector']}")
    print(f"Vehicles tracked: {s['vehicles_seen']}")
    if s["siren_intervals"]:
        print("Siren in audio at: " + ", ".join(f"{a}-{b}s" for a, b in s["siren_intervals"]))
    if not s["detections"]:
        print("\nNo ambulance or fire engine detected.")
        if s["detector"] == "none":
            print("NOTE: CLIP is not installed — run: pip install transformers")
        return
    print(f"\nEMERGENCY VEHICLES DETECTED: {len(s['detections'])}")
    for d in s["detections"]:
        print(f"  {d['time']}  track {d['track']:<4} {PRETTY[d['type']]:<12} "
              f"conf={d['conf']:<6} via {d['method']:<16} -> static/{d['snapshot']}")


if __name__ == "__main__":
    main()
