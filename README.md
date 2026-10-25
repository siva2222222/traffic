<<<<<<< HEAD
---
title: Traffic Monitoring System
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.21.0
python_version: 3.11
app_file: app.py
pinned: false
---
=======
# AI-Based Smart Traffic Management System

Implements the full architecture: **Input sources → AI processing layer → LLM layer → Public portal / Admin portal / Police app → Data layer → Outputs.**

| Diagram block | Where |
|---|---|
| 1 Vehicle detection (YOLOv8: car, truck, bus, bike, ambulance) | `engine.py` |
| 2 Tracking (ByteTrack IDs) | `engine.py` (Ultralytics `bytetrack.yaml`) |
| 3 Violations: red light, wrong way, stop line, illegal parking, speeding, lane violation | `engine.py` `process()` |
| 4 Lane assignment (any number of lanes/approaches, majority-vote, camera-motion compensated) | `engine.py`, `stabilizer.py`, `zones/lanes.json`, **Edit lanes on video** (Dashboard) |
| 5 Direction density: North / South / East / West panel + arrow on every vehicle | `engine.py` (`_hud`, `_compass`), Dashboard |
| 5 Density = vehicles / lane | `engine.py`, dashboard |
| 6 Emergency: visual (YOLO / trained classifier / EasyOCR "AMBULANCE") + siren (MFCC + CNN) | `engine.py`, `emergency_classifier.py`, `audio.py`, `siren_model.py` |
| Adaptive signal + emergency pre-emption | `SignalController` in `engine.py` |
| LLM layer: incident analysis, task priority, summaries, chatbot, reports | `llm.py`, `services.py` |
| Public portal / Admin portal / Police mobile app | `app.py`, `templates/` |
| Data layer (10 stores) + CSV/PDF | `database.py`, **Data** page |

## Run
```bash
python -m venv venv && venv\Scripts\activate       # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python app.py                                      # http://localhost:5000
```
Logins: `admin/admin123` · `police1/police123` (police2, police3) · `citizen/citizen123`. **Change before deploying.**
`yolov8n.pt` and the trained siren model (`models/siren_cnn.pt`) are included, so it runs offline.

## Workflow for ANY new video (everything on the Dashboard - no separate page)
1. Dashboard → **Upload & start** (or Camera). The live video shows: lane areas + density, an arrow on every vehicle (its direction of travel), and the **DIRECTION panel** (vehicles going North / South / East / West).
2. Click **✏️ Edit lanes on video**. The frame freezes on the same panel. Use the **North / South / East / West** buttons (or *Custom name*) to draw a lane area, then click its stop line (or *Skip stop line* if there is no signal). Rename with ✎, delete with ✕. Optional: no-parking zone, restricted zone, scale (speed), "North is up/right/down/left".
3. **Save & apply** - the running video updates immediately (no restart). No direction arrows to draw: each lane's normal flow is learned from the traffic.

`zones/lanes.json` holds the current layout (video5: 4 lanes named North/East/South/West, no signal). `zones/lanes_intersection_example.json` is a 4-way junction template. Zones are stored on the video's first frame; `stabilizer.py` follows camera zoom/pan/shake.

## LLM (optional)
`set LLM_PROVIDER=gemini` + `set GEMINI_API_KEY=...`  (or `anthropic` + `ANTHROPIC_API_KEY`, or `groq` + `GROQ_API_KEY`). Without a key: rule-based fallbacks.

## Emergency vehicle detection (ambulance / fire engine)
Stock `yolov8n.pt` has no `ambulance`/`fire_engine` class, so without help these show up as
`truck`. Two fallbacks run in order, both before/alongside the normal vehicle classes:
1. **Trained visual classifier** (`emergency_classifier.py`, no-op until you train one) — looks at
   how the vehicle looks, not text. See **TRAIN_EMERGENCY_MODEL.md** for the full how-to
   (dataset links, `crop_dataset.py`, `train_emergency.py`).
2. **OCR** (`easyocr`, in `requirements.txt`, **on by default** — `ENABLE_OCR=1`) — reads
   "AMBULANCE"/"FIRE ENGINE" text on the vehicle; edit `OCR_KEYWORDS` in `config.py` to match
   what's actually printed on your footage.

## Optional extras
* `pip install sounddevice` + tick *microphone* on Source page → live siren monitor for camera sources.
* ffmpeg on PATH → siren detection from the audio track of uploaded videos.
* Settings via env: `YOLO_MODEL` (e.g. yolov8m.pt or custom weights with an `ambulance` class), `SPEED_LIMIT`, `PROCESS_EVERY=2`, `CONF`.

## Siren model (MFCC + CNN)
Trained with `python train_siren.py --esc50 <ESC-50-master> [--eval]` (dataset: ESC-50, CC BY-NC 4.0, non-commercial use).
Held-out test (fold 5, never seen in training): 8/8 siren clips detected, 15/392 (3.8%) other clips falsely flagged at the default threshold.
Only 8 test sirens - treat as indicative. ESC-50 sirens are mostly US/EU; add local ambulance recordings with `--extra_siren DIR` for best results.

## Tests
`python tests/test_engine.py` · `test_audio.py` (set `ESC50=` for real-audio check) · `test_app.py` · `python tests/test_real_video.py video.mp4` (real YOLO through the web app).

## Known limits (be upfront in your report)
* Signal control is a **simulation** on the dashboard; it is not wired to a physical controller.
* Speed is an estimate (perspective scale, needs calibration). No licence-plate recognition, so fines are not linked to a registered owner.
* Detection quality depends on camera angle, resolution and lighting; `yolov8n` is the fastest, `yolov8m` more accurate. Rickshaws/bicycles are not separate classes.
* Lane direction is learned automatically from the first `DIR_LEARN_MIN` (6) vehicles of each lane (majority flow); Wrong Way is not checked in a lane until then. Each lane area must carry one-way traffic - draw opposite directions as separate lanes. You can still set `"direction"` in `zones/lanes.json` to fix it manually.
* Wrong-way / red-light / lane rules are only as good as the drawn zones.
* Uploaded media and evidence are served from `/static` - restrict access before public deployment.
>>>>>>> origin/master
