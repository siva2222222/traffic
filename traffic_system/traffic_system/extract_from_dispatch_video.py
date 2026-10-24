"""Extract labelled training crops for train_emergency.py from a real dispatch/traffic-cam video
(e.g. the Seattle Fire scanner-cam clip you shared) using known time windows.

You supply TIME WINDOWS saying "the fire engine/ambulance is visible between these seconds" —
get these by watching the video once (unit-ID overlays like E18/M18/L6, if your footage has
them, are a good guide to what each vehicle actually is). This script then:
  1. Samples frames inside each window (SAMPLE_EVERY seconds apart)
  2. Runs your project's own yolov8n.pt on each sampled frame
  3. Crops the LARGEST car/bus/truck box (assumed to be the responding unit) into that window's
     class folder
  4. Crops every OTHER car/bus/truck box in the same frame into dataset/<split>/normal/ — free
     negative examples from the same footage, no extra work

Add an entry to CLIPS below for each new video (key = filename, value = its windows), then:
    python extract_from_dispatch_video.py --video training_clips/yourclip.mp4 --out dataset

A couple minutes of one video gives you tens of correctly-labelled, domain-matched crops — a
useful supplement to a bigger public dataset (see TRAIN_EMERGENCY_MODEL.md), not a full dataset
by itself. Run this against each clip (same --out each time) to build the folder up — it only
adds files, never clears it.
"""
import argparse
import os

import cv2
from ultralytics import YOLO

import engine as E
from config import C

# ---- One entry per video: filename -> [(start_seconds, end_seconds, label), ...] ----
# label is 'ambulance' or 'fire_engine'. Add your own clips here the same way — watch the video
# once, note the seconds each vehicle is actually on screen (unit-ID overlays, if present, are
# a good guide to what each one really is).
CLIPS = {
    # Seattle Fire dispatch-cam clip, 30fps/~104s. Overlays read: E35 (0-9s), E18 (12-17s),
    # L8 (18-23s), M18 medic unit (30-35s), L6 (56-69s). Left out as too uncertain to
    # auto-label from the video alone: B4/B6 (Battalion Chief command vehicles — not fire
    # engines), RB1/RB5 (rehab unit, mostly off-screen here), SATT2 (safety/air truck, too
    # hazy/smoky to confirm) — verify and add those yourself if you want them included.
    "dispatch_clip_1.mp4": [
        (0, 9, "fire_engine"),     # E35
        (12, 17, "fire_engine"),   # E18
        (18, 23, "fire_engine"),   # L8
        (30, 35, "ambulance"),     # M18 medic unit
        (56, 69, "fire_engine"),   # L6
    ],
    # Intersection CCTV clip, 30fps/27s — car/motorcycle collision, a white European-pattern
    # (red/orange stripe, blue light bar) ambulance arrives at ~20s and is on screen to the end.
    # No fire engine in this one. Note: this livery is a different style than what you'll likely
    # see on your own camera footage (looked Baltic/European, not Indian) — still useful for
    # general shape/light-bar variety, but don't rely on it alone for livery-specific accuracy.
    "dispatch_clip_2.mp4": [
        (20, 27, "ambulance"),
    ],
}
SAMPLE_EVERY = 0.5      # seconds between sampled frames within a window
VAL_FRACTION = 0.15     # last ~15% of each window's samples go to val/ instead of train/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()

    key = os.path.basename(args.video)
    if key not in CLIPS:
        raise SystemExit(
            f"No windows defined for '{key}' in CLIPS at the top of this script. "
            f"Add an entry like CLIPS['{key}'] = [(start_s, end_s, 'ambulance'|'fire_engine'), ...]."
        )
    windows = CLIPS[key]

    print(f"Loading {C.YOLO_MODEL} ...")
    model = YOLO(C.YOLO_MODEL)
    ids = [i for i, n in model.names.items() if n.lower() in E.NAME_MAP]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")

    counts = {"ambulance": 0, "fire_engine": 0, "normal": 0}
    for start, end, label in windows:
        times = []
        t = start
        while t < end:
            times.append(t)
            t += SAMPLE_EVERY
        if not times:
            continue
        split_at = max(1, int(len(times) * (1 - VAL_FRACTION)))

        for i, t in enumerate(times):
            split = "train" if i < split_at else "val"
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            res = model.predict(frame, classes=ids, conf=C.CONF, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            order = areas.argsort()[::-1]   # largest first

            for rank, idx in enumerate(order):
                x1, y1, x2, y2 = [max(0, int(v)) for v in boxes[idx]]
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # largest box in frame = assumed to be the responding unit named in this window;
                # everything else on screen at the same moment is ordinary traffic
                bucket = label if rank == 0 else "normal"
                out_dir = os.path.join(args.out, split, bucket)
                os.makedirs(out_dir, exist_ok=True)
                fname = f"{os.path.splitext(os.path.basename(args.video))[0]}_{t:.1f}s_{rank}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), crop)
                counts[bucket] += 1

    print("Wrote crops:", counts)
    print(f"Combine with a bigger dataset (see TRAIN_EMERGENCY_MODEL.md) then:\n"
          f"  python train_emergency.py --data {args.out} --epochs 12")


if __name__ == "__main__":
    main()
