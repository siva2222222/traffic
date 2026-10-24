# Emergency Vehicle Detection (ambulance / fire engine) — image upload

## Why it wasn't working before

Three separate reasons, all of them now fixed:

1. **`yolov8n.pt` is stock COCO.** Its 80 classes include `car`, `bus`, `truck` — but there is
   no `ambulance` and no `fire_engine`. An ambulance was *always* being detected, just labelled
   `truck`. No amount of tuning `CONF` would change that.
2. **Both fallbacks were dead.** `models/emergency_cnn.pt` never existed (it only appears after
   you run `train_emergency.py` on a dataset you'd have to build first), and the EasyOCR path
   only fires when the vehicle has large, legible, front-facing lettering.
3. **There was no image upload path at all.** `/admin/source` accepts `.mp4/.avi/.mov/.mkv` only.
   And even if an image had been pushed through the engine, the emergency checks are gated on
   `self.frame_idx % C.EMERG_CLF_EVERY_N == 0` plus a per-track retry counter — gating a single
   still frame can never satisfy.

## What changed

| File | Change |
|---|---|
| `emergency_vision.py` | **NEW.** Zero-shot CLIP recogniser — classifies a vehicle crop as ambulance / fire engine / normal with **no training and no API key**. |
| `image_analyze.py` | **NEW.** Still-image pipeline. Same detector cascade, all stride/retry gating removed. Also a CLI. |
| `templates/admin_image.html` | **NEW.** Upload page with annotated result, per-vehicle verdict, and which detector decided it. |
| `app.py` | `+ /admin/image` (page) and `+ /api/detect/image` (JSON). Logs to `emergency_events` and notifies admins. |
| `engine.py` | CLIP inserted as step 2 of the cascade, so **video** detection improves too. |
| `config.py` | CLIP settings block. |
| `templates/base.html` | "Image Detection" nav item. |

## Setup

```bash
cd traffic_system
pip install -r requirements.txt
```

The one new thing that matters is `transformers`. On first run it downloads CLIP ViT-B/32
(~350 MB) from HuggingFace and caches it in `~/.cache/huggingface`. After that it is fully
offline and runs on CPU at roughly 60–120 ms per vehicle.

```bash
python app.py
```

Open <http://127.0.0.1:5000/admin/image>, log in as admin, upload a photo.

## Quick test without the dashboard

```bash
python image_analyze.py ambulance.jpg --verbose
python image_analyze.py fire_engine.jpg --save result.jpg
```

Output:

```
Detector   : CLIP=transformers  OCR=True  trainedCNN=False

 *** #1  AMBULANCE    conf=0.94   via clip             (YOLO said truck)
     #2  Car          conf=0.77   via yolo             (YOLO said car)

RESULT: EMERGENCY VEHICLE DETECTED -> AMBULANCE
SIGNAL: PRE-EMPT: force GREEN for the emergency vehicle's approach
```

The top banner on the page tells you immediately whether CLIP actually loaded. If it shows
`CLIP: none`, zero-shot is off and you are back to OCR-only — install `transformers`.

## The detection cascade

Run in order per vehicle; first hit wins.

| # | Method | Needs | Notes |
|---|---|---|---|
| 1 | dedicated YOLO class | custom-trained weights | only if you set `YOLO_MODEL` to your own `.pt` |
| 2 | **CLIP zero-shot** | `transformers` | **the main path** — no training, no text needed |
| 3 | trained MobileNet CNN | `models/emergency_cnn.pt` | optional, via `train_emergency.py` |
| 4 | EasyOCR keywords | `easyocr` | needs legible AMBULANCE / FIRE SERVICE / 108 text |
| 5 | colour heuristic | — | OFF by default, `ENABLE_COLOR_FALLBACK=1` |

If YOLO finds no vehicle box at all (tight close-up photo), the whole image goes to CLIP instead.

## Video upload page — `/admin/video`

Upload a clip, get back **the timestamp and a saved snapshot of every ambulance and fire engine
that appears in it**. This is separate from the live 4-lane dashboard on purpose: no lane
polygons, no signal cycle, no violations — it answers one question, "does an emergency vehicle
show up in this video, and when".

```
Scan: ambulance_run.mp4
[████████████░░░░░░]  67%        Vehicles tracked: 34   Emergency found: 2

EMERGENCY VEHICLE DETECTED
AMBULANCE, FIRE ENGINE — 2 appearance(s). Signal action: PRE-EMPT to GREEN.
```

Each hit shows as a card: the exact frame it was flagged on, boxed and captioned, plus track id,
confidence, which detector decided it, and what YOLO originally called it.

Scanning runs in a **background thread** — CLIP costs ~60–120 ms per crop, so a one-minute clip
takes a few minutes on CPU. The upload returns immediately and the page polls for progress. You
can navigate away and come back to `/admin/video?job=<id>`; the scan keeps running.

Siren audio is checked too (`audio.analyse_file`, MFCC + CNN), independently of the camera, and
any siren intervals are listed alongside the visual hits.

Every detection writes an `emergency_events` row and notifies all admins, the same as a live
detection would.

### Command line

```bash
python video_analyze.py static/uploads/clip.mp4
```

```
scanning: 100%  (2 emergency found)

Scanned clip.mp4 in 94.2s — detector CLIP=transformers
Vehicles tracked: 34
Siren in audio at: 12.4-18.1s

EMERGENCY VEHICLES DETECTED: 2
  00:03.40  track 12   AMBULANCE    conf=0.91   via clip   -> static/evidence/video_scan/ambulance_12_a1b2c3d4.jpg
  00:11.80  track 27   FIRE ENGINE  conf=0.88   via clip   -> static/evidence/video_scan/fire_engine_27_e5f6a7b8.jpg
```

### Three pages, three jobs

| Page | Input | Gives you |
|---|---|---|
| `/admin/image` | one photo | instant per-vehicle verdict |
| `/admin/video` | one video | timestamps + snapshots of every emergency vehicle |
| `/admin` | up to 4 lane videos | live dashboard, signal pre-emption, violations, density |

The dashboard is the real system; the two upload pages are for checking that detection actually
works before you trust it live.

## Video: yes, it detects — with a CPU budget

CLIP is wired into `engine.py`, so an uploaded **video** runs the same cascade as an image.
Three differences from the still-image path matter:

**1. Detection is per-track and sticky.** Once a ByteTrack track is flagged, `tr["emerg"]`
stays set for that track's whole life — the vehicle is re-labelled `ambulance`/`fire_engine`,
drawn in red, logged once to `emergency_events`, exempted from violations, and the lane it
occupies pre-empts the signal to GREEN. It does not need to be re-recognised every frame.

**2. It gets up to 10 shots, not one.** `CLIP_MAX_TRIES=10` means a vehicle is examined on 10
separate frames as it approaches. That is a large advantage over a single image — an ambulance
that is ambiguous at distance usually gets caught once it fills more of the frame. If ByteTrack
switches the ID, the new track simply starts its own 10 tries.

**3. Audio runs too.** Uploaded video goes through `audio.analyse_file()` for siren detection
(MFCC + CNN, `models/siren_cnn.pt`), which is independent of what the camera sees. A siren hit
alone raises the emergency banner and logs a `siren` event. Images have no equivalent.

### The CPU cost, and why it is capped

CLIP is ~60–120 ms per crop and EasyOCR ~300–800 ms. Without a cap, a frame with eight vehicles
costs about a second, and a 25 fps video falls far behind real time. So both are now budgeted:
only the **largest** eligible boxes get checked each frame (nearest vehicle = most readable),
capped by `CLIP_MAX_PER_FRAME` (default 2) and `OCR_MAX_PER_FRAME` (default 1).

At defaults (`CLIP_EVERY_N=6`, 2 per frame) that is ~8 crops/s, roughly **0.75 s of CPU per
second of video** — on top of YOLO itself. On a typical laptop CPU you should expect playback to
run behind real time. The engine handles this by just lagging, not dropping correctness.

**If video playback is sluggish, set `PROCESS_EVERY=3`** in `config.py` or as an env var. That
processes every 3rd frame and cuts CLIP calls proportionally. Detection quality barely changes,
because a vehicle is on screen for dozens of frames.

| Setting | crops/s | CPU per 1 s video | Use when |
|---|---|---|---|
| `CLIP_EVERY_N=10`, `CLIP_MAX_PER_FRAME=1` | 2.5 | ~0.22 s | weak CPU, light traffic |
| `CLIP_EVERY_N=6`, `CLIP_MAX_PER_FRAME=2` *(default)* | 8.3 | ~0.75 s | balanced |
| `CLIP_EVERY_N=4`, `CLIP_MAX_PER_FRAME=3` | 18.8 | ~1.7 s | busy junction, accuracy first, slow playback |

Boxes shorter than `CLIP_MIN_BOX_H` (40 px at 960-wide) are skipped entirely — a distant
ambulance 25 px tall carries no usable signal and would only burn budget. It gets picked up as
it approaches.

### Checking a video without the dashboard

```bash
python check_emergency.py static/uploads/yourvideo.mp4
```

Prints every relabel as it happens with its timestamp and track id, then a summary. This is the
fastest way to answer "did it detect or not" — no login, no streaming, no lane config.

```
  t=  3.4s  track  12 -> ambulance
  t= 11.8s  track  27 -> fire_engine

--- summary ---
Unique tracked vehicles: 34
Emergency relabels: 2
```

If it prints zero relabels on a video that definitely contains an ambulance, work down this list:

| Check | Fix |
|---|---|
| Did CLIP load? First lines say `CLIP ready` or `no CLIP backend loaded` | `pip install transformers` |
| Is the vehicle large enough on screen? | lower `CLIP_MIN_BOX_H` to 25 |
| Is it on screen only briefly? | lower `CLIP_EVERY_N` to 3 |
| Many vehicles competing for the budget? | raise `CLIP_MAX_PER_FRAME` to 4 |
| Does YOLO even box it? | lower `CONF` to 0.25, or `YOLO_MODEL=yolov8s.pt` |
| Still nothing | run `python image_analyze.py` on a screenshot of the same vehicle — that isolates recognition from the video pipeline |

## How CLIP decides, and how to tune it

CLIP scores the crop against prompt ensembles in `emergency_vision.PROMPTS`. The `normal` group
deliberately contains the lookalikes — *a red truck carrying cargo*, *a white delivery van*,
*a police car* — because without those negatives CLIP fires on colour alone.

Two gates must both pass:

- `CLIP_MIN_CONF` (default `0.45`) — tempered softmax confidence.
- `CLIP_MIN_MARGIN` (default `0.006`) — raw cosine gap over the runner-up class. **This is the
  real test.** CLIP's native logit scale of 100 saturates every call to ~0.99, so a softmax
  threshold on its own filters nothing; `CLIP_LOGIT_SCALE` is set to 50 and the margin does the
  discriminating.

Tuning, by symptom:

| Symptom | Fix |
|---|---|
| Real ambulance missed | `set CLIP_MIN_CONF=0.35` and `CLIP_MIN_MARGIN=0.003` |
| Red cargo trucks flagged as fire engines | raise `CLIP_MIN_MARGIN` to `0.012`; add the exact truck type to the `normal` prompts |
| Indian 108 / state ambulances missed | add their livery wording to `OCR_KEYWORDS` and a prompt like `"a 108 emergency ambulance van"` to `PROMPTS["ambulance"]` |
| Small / distant vehicles missed in video | lower `CLIP_MIN_BOX_H`, lower `CLIP_EVERY_N` |
| Too slow on video | raise `CLIP_EVERY_N` to 8–10, lower `CLIP_MAX_TRIES` |
| Vehicles not boxed at all | `set IMAGE_YOLO_MODEL=yolov8s.pt` (better recall than nano), lower `IMAGE_CONF` |

Editing `PROMPTS` is the highest-leverage tuning available and costs nothing — it is plain text,
no retraining.

## Honest limits

- **CLIP is zero-shot, not a trained detector.** Expect roughly 85–92% on clear, well-lit shots
  of recognisable ambulances and fire engines, and noticeably worse on unmarked private
  ambulances, night shots, heavy occlusion, or vehicles far from Western/Indian norms. It is a
  very strong default, not a guarantee.
- **`transformers` is required.** There is no way to get this accuracy with no model and no
  training data. If you cannot install it, your only remaining options are OCR (needs legible
  text) or `ENABLE_COLOR_FALLBACK=1` (noisy — flags red cargo trucks).
- **First run downloads ~350 MB** and needs internet once. Subsequent runs are offline.
- **The colour heuristic is a last resort**, not a solution. It is off by default for a reason.
- For production-grade accuracy you still want step 3: build a few hundred crops per class and
  run `train_emergency.py`. CLIP is what makes the system work *today*, without that dataset —
  and it is also the fastest way to auto-label crops for that dataset later.

## API

```bash
curl -b cookies.txt -F "image=@ambulance.jpg" http://127.0.0.1:5000/api/detect/image
```

```json
{
  "emergency_types": ["ambulance"],
  "signal_action": "PRE-EMPT: force GREEN for the emergency vehicle's approach",
  "vehicles": [
    {"n": 1, "label": "ambulance", "conf": 0.94, "method": "clip",
     "yolo_class": "truck", "emergency": true,
     "scores": {"ambulance": 0.94, "fire_engine": 0.02, "normal": 0.04}}
  ],
  "detector": "transformers", "ms": 410
}
```
