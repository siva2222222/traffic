"""Single-image analysis: upload a photo -> detect vehicles -> flag ambulance / fire engine.

The video engine gates its emergency checks on frame_idx stride and track history, which a
one-frame image never satisfies. This module runs the SAME detector cascade with all the
stride/retry gating removed, so every vehicle in a still image is checked once, properly:

    1. dedicated YOLO class        (only if YOLO_MODEL is custom-trained weights)
    2. CLIP zero-shot              (emergency_vision.py — no training needed)   <- main path
    3. trained CNN                 (models/emergency_cnn.pt, if you trained one)
    4. OCR keyword match           (easyocr, reads AMBULANCE / FIRE SERVICE / 108 ...)
    5. colour heuristic            (off unless ENABLE_COLOR_FALLBACK=1)

If YOLO finds no vehicle at all (common with tight close-up photos), the whole image is
passed to CLIP instead, so a cropped ambulance photo still gets an answer.

CLI:
    python image_analyze.py path/to/ambulance.jpg
    python image_analyze.py photo.jpg --save out.jpg --verbose
"""
import os
import sys
import time
import uuid

import cv2
import numpy as np

from config import C
import emergency_vision as EV

BASE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(C.UPLOAD_DIR, "analysis")

# Mirrors engine.NAME_MAP / EMERG_TYPES without importing the engine (keeps this usable
# headless, with no DB or Flask app around).
NAME_MAP = {"car": "car", "motorcycle": "bike", "motorbike": "bike", "bike": "bike",
            "bus": "bus", "truck": "truck", "ambulance": "ambulance",
            "fire_engine": "fire_engine", "fire engine": "fire_engine",
            "firetruck": "fire_engine", "fire truck": "fire_engine", "fire-truck": "fire_engine"}
EMERG_TYPES = ("ambulance", "fire_engine")
PRETTY = {"ambulance": "AMBULANCE", "fire_engine": "FIRE ENGINE", "car": "Car",
          "bus": "Bus", "truck": "Truck", "bike": "Bike"}
BOX_COLOR = {"ambulance": (0, 0, 255), "fire_engine": (0, 110, 255), "car": (80, 200, 80),
             "bike": (255, 160, 40), "bus": (60, 60, 230), "truck": (230, 80, 200)}

_model = None
_ocr = None
_cnn = None          # None = untried, False = unavailable, else EmergencyClassifier


def _yolo():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(C.IMAGE_YOLO_MODEL or C.YOLO_MODEL)
    return _model


def _cnn_predict(crop):
    """Optional trained MobileNet classifier (models/emergency_cnn.pt) — skipped if absent."""
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
        except Exception:                                # noqa: BLE001
            _cnn = False
            return None
    try:
        hit = _cnn.predict(crop)
        return (hit, C.EMERG_CLF_MIN_CONF, "cnn") if hit else None
    except Exception:                                    # noqa: BLE001
        return None


def _ocr_predict(crop):
    """EasyOCR keyword match against C.OCR_KEYWORDS. Returns (label, conf, 'ocr') or None."""
    global _ocr
    if not C.ENABLE_OCR:
        return None
    try:
        if _ocr is None:
            import easyocr
            _ocr = easyocr.Reader(["en"], gpu=False)
        h, w = crop.shape[:2]
        scale = max(1.0, 240.0 / max(1, min(h, w)))      # upscale: small text is unreadable otherwise
        img = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC) \
            if scale > 1.0 else crop
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        texts = []
        for v in (img, cv2.flip(img, 1), gray):          # mirrored: "AMBULANCE" is often reversed on the bonnet
            texts += _ocr.readtext(v, detail=0)
        txt = " ".join(texts).upper()
        for etype, kws in C.OCR_KEYWORDS.items():
            for kw in kws:
                if kw in txt:
                    return etype, 1.0, f"ocr:{kw}"
        return None
    except Exception as e:                               # noqa: BLE001
        print(f"[image_analyze] OCR unavailable: {e}")
        return None


def _pad_crop(img, box, pad=0.08):
    """Crop with a margin — CLIP reads light bars and roof markings that a tight box clips off."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * pad))
    y1 = int(max(0, y1 - bh * pad))
    x2 = int(min(W, x2 + bw * pad))
    y2 = int(min(H, y2 + bh * pad))
    return img[y1:y2, x1:x2]


def _identify(crop, yolo_cls):
    """Run the cascade on one crop. Returns (label, conf, method) or None."""
    hit = EV.classify_crop(crop)                         # 2. CLIP zero-shot
    if hit:
        return hit
    hit = _cnn_predict(crop)                             # 3. trained CNN, if present
    if hit:
        return hit
    hit = _ocr_predict(crop)                             # 4. OCR text
    if hit:
        return hit
    return EV.color_hint(crop, yolo_cls)                 # 5. colour, off by default


def analyse_image(path, save_dir=None, want_scores=True):
    """Analyse one image file. Returns a result dict (never raises for detection failures)."""
    t0 = time.time()
    if not os.path.isabs(path) and not os.path.exists(path):
        for cand in (os.path.join(BASE, path), os.path.join(C.UPLOAD_DIR, os.path.basename(path))):
            if os.path.exists(cand):
                path = cand
                break
    img = cv2.imread(path)
    if img is None:
        if not os.path.exists(path):
            return {"error": f"File not found: {path}  (looked in the project folder and "
                             f"static/uploads/)"}
        return {"error": f"File exists but is not a readable image: {path}"}

    h0, w0 = img.shape[:2]
    if w0 != C.WORK_WIDTH:                               # normalise scale so thresholds behave
        img = cv2.resize(img, (C.WORK_WIDTH, max(1, int(h0 * C.WORK_WIDTH / w0))))
    vis = img.copy()

    model = _yolo()
    ids = [i for i, n in model.names.items() if n.lower() in NAME_MAP]
    res = model.predict(img, classes=ids, conf=C.IMAGE_CONF, verbose=False)[0]

    vehicles, notes = [], []
    boxes = []
    if res.boxes is not None and len(res.boxes):
        for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(),
                            res.boxes.cls.int().cpu().tolist(),
                            res.boxes.conf.cpu().tolist()):
            boxes.append((tuple(float(v) for v in b), NAME_MAP[model.names[c].lower()], float(cf)))

    for i, (box, ycls, yconf) in enumerate(boxes, 1):
        crop = _pad_crop(img, box)
        if crop.size == 0:
            continue
        if ycls in EMERG_TYPES:                          # 1. custom YOLO weights already know
            label, conf, method = ycls, yconf, "yolo"
        else:
            hit = _identify(crop, ycls)
            label, conf, method = hit if hit else (ycls, yconf, "yolo")
        v = {"n": i, "box": [int(x) for x in box], "yolo_class": ycls, "yolo_conf": round(yconf, 3),
             "label": label, "conf": round(float(conf), 3), "method": method,
             "emergency": label in EMERG_TYPES}
        if want_scores:
            s = EV.raw_scores(crop)
            if s:
                v["scores"] = {k: round(val, 3) for k, val in s.items()}
        vehicles.append(v)

    # No vehicle box at all -> close-up photo. Ask CLIP about the whole frame.
    if not vehicles:
        hit = _identify(img, "car")
        if hit:
            label, conf, method = hit
            vehicles.append({"n": 1, "box": [2, 2, img.shape[1] - 2, img.shape[0] - 2],
                             "yolo_class": "-", "yolo_conf": None, "label": label,
                             "conf": round(float(conf), 3), "method": method + "+wholeimage",
                             "emergency": label in EMERG_TYPES,
                             "scores": ({k: round(v, 3) for k, v in EV.raw_scores(img).items()}
                                        if want_scores and EV.raw_scores(img) else None)})
            notes.append("YOLO found no vehicle box — classified the whole image instead "
                         "(normal for a tight close-up photo).")
        else:
            notes.append("No vehicle detected in this image.")

    # ---- annotate ----
    for v in vehicles:
        x1, y1, x2, y2 = v["box"]
        col = BOX_COLOR.get(v["label"], (200, 200, 200))
        thick = 4 if v["emergency"] else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, thick)
        tag = f"{PRETTY.get(v['label'], v['label'])} {v['conf']:.2f} [{v['method']}]"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(th + 6, y1)
        cv2.rectangle(vis, (x1, ty - th - 6), (x1 + tw + 8, ty), col, -1)
        cv2.putText(vis, tag, (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    emerg = [v for v in vehicles if v["emergency"]]
    if emerg:
        kinds = sorted({PRETTY[v["label"]] for v in emerg})
        banner = "EMERGENCY VEHICLE DETECTED: " + ", ".join(kinds)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 46), (0, 0, 200), -1)
        cv2.putText(vis, banner, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2)

    save_dir = save_dir or RESULT_DIR
    os.makedirs(save_dir, exist_ok=True)
    name = f"analysis_{uuid.uuid4().hex[:10]}.jpg"
    out_path = os.path.join(save_dir, name)
    cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

    if not EV.ready():
        notes.append("CLIP zero-shot is OFF (no backend installed) — only OCR text matching is "
                     "active, which needs large legible lettering. Run: pip install transformers")

    return {
        "ok": True,
        "image": path,
        "annotated_path": out_path,
        "annotated_rel": os.path.relpath(out_path, os.path.join(BASE, "static")).replace("\\", "/"),
        "vehicles": vehicles,
        "emergency": emerg,
        "emergency_types": sorted({v["label"] for v in emerg}),
        "counts": {k: sum(1 for v in vehicles if v["label"] == k)
                   for k in sorted({v["label"] for v in vehicles})},
        "detector": EV.backend_name(),
        "ocr": bool(C.ENABLE_OCR),
        "cnn": os.path.exists(os.path.join(BASE, "models", "emergency_cnn.pt")),
        "signal_action": ("PRE-EMPT: force GREEN for the emergency vehicle's approach"
                          if emerg else "No pre-emption — normal adaptive cycle"),
        "notes": notes,
        "ms": int((time.time() - t0) * 1000),
    }


def main():
    args = [a for a in sys.argv[1:]]
    if not args:
        print(__doc__)
        print("Usage: python image_analyze.py <image> [--save out.jpg] [--verbose]")
        sys.exit(1)
    verbose = "--verbose" in args
    save = None
    if "--save" in args:
        save = args[args.index("--save") + 1]
    path = args[0]

    r = analyse_image(path)
    if r.get("error"):
        print(r["error"])
        sys.exit(1)

    print(f"\nImage      : {path}")
    print(f"Detector   : CLIP={r['detector']}  OCR={r['ocr']}  trainedCNN={r['cnn']}")
    print(f"Time       : {r['ms']} ms\n")
    for v in r["vehicles"]:
        mark = "***" if v["emergency"] else "   "
        print(f" {mark} #{v['n']:<2} {PRETTY.get(v['label'], v['label']):<12} "
              f"conf={v['conf']:<6} via {v['method']:<16} (YOLO said {v['yolo_class']})")
        if verbose and v.get("scores"):
            print(f"        CLIP scores: {v['scores']}")
    print()
    if r["emergency"]:
        print(f"RESULT: EMERGENCY VEHICLE DETECTED -> {', '.join(PRETTY[t] for t in r['emergency_types'])}")
        print(f"SIGNAL: {r['signal_action']}")
    else:
        print("RESULT: no emergency vehicle detected")
    for n in r["notes"]:
        print(f"NOTE  : {n}")
    out = r["annotated_path"]
    if save:
        cv2.imwrite(save, cv2.imread(out))
        out = save
    print(f"\nAnnotated image saved to: {out}")


if __name__ == "__main__":
    main()
