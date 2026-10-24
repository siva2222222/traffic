"""Headless check: does YOLO + OCR/classifier ever flag ambulance/fire_engine in a video?
No dashboard needed — prints each relabel as it happens, then a summary. Useful for quickly
telling whether a "not detected" problem is YOLO/OCR/classifier-related without sitting
through the live dashboard.

Run:
    python check_emergency.py static/uploads/yourvideo.mp4
    python check_emergency.py 0            # webcam
"""
import os
import sys
from collections import Counter

import cv2
from ultralytics import YOLO

from config import C
import engine as E


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_emergency.py <video_path_or_camera_index>")
        sys.exit(1)
    source = sys.argv[1]
    source = int(source) if source.isdigit() else source

    print(f"Loading {C.YOLO_MODEL} ...")
    model = YOLO(C.YOLO_MODEL)
    ids = [i for i, n in model.names.items() if n.lower() in E.NAME_MAP]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open {source}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    clf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "emergency_cnn.pt")
    print(f"ENABLE_OCR={C.ENABLE_OCR}   trained classifier present={os.path.exists(clf_path)}")

    eng = E.TrafficEngine()
    eng.source_name = "check"
    cfg_loaded = False
    seen = Counter()
    events = []
    frame_idx = 0

    print("Processing... (Ctrl+C to stop early)\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            h0, w0 = frame.shape[:2]
            frame = cv2.resize(frame, (C.WORK_WIDTH, int(h0 * C.WORK_WIDTH / w0)))
            if not cfg_loaded:
                eng.load_cfg(frame.shape[1], frame.shape[0])
                cfg_loaded = True
            eng.frame_idx = frame_idx
            t = frame_idx / fps

            res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=ids,
                               conf=C.CONF, verbose=False)[0]
            dets = []
            if res.boxes is not None and res.boxes.id is not None:
                for b, tid, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist(),
                                      res.boxes.cls.int().cpu().tolist()):
                    dets.append({"tid": int(tid), "box": tuple(float(v) for v in b),
                                 "cls": E.NAME_MAP[model.names[c].lower()]})

            before = {tid: tr.get("emerg_type") for tid, tr in eng.tracks.items()}
            eng.process(frame, dets, t, False)
            for d in dets:
                seen[d["cls"]] += 1
            for tid, tr in eng.tracks.items():
                if tr.get("emerg_type") and before.get(tid) != tr["emerg_type"]:
                    events.append((round(t, 1), tid, tr["emerg_type"]))
                    print(f"  t={t:6.1f}s  track {tid:>3} -> {tr['emerg_type']}")
    except KeyboardInterrupt:
        pass

    print("\n--- summary ---")
    print("YOLO raw per-frame class counts:", dict(seen))
    print(f"Unique tracked vehicles: {len(eng.tracks)}")
    print(f"Emergency relabels: {len(events)}")
    if not events:
        print("No ambulance/fire_engine ever flagged. If one IS in this video: OCR needs legible text in "
              "frame (check ENABLE_OCR / OCR_KEYWORDS in config.py), and the visual classifier needs "
              "models/emergency_cnn.pt to exist (see TRAIN_EMERGENCY_MODEL.md) — right now it doesn't.")


if __name__ == "__main__":
    main()
