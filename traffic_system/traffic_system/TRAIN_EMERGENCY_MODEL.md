# Getting real ambulance / fire-engine detection working

## Why it currently shows "truck"
`yolov8n.pt` is stock COCO weights — 80 general classes (car, truck, bus, motorcycle...),
**no `ambulance` or `fire_engine` class**. So YOLO itself can never output those labels; an
ambulance just gets classified as whatever COCO class it looks closest to (almost always
`truck`). There's no way around this without either reading text on the vehicle (OCR) or
giving the system a model that's actually seen ambulances/fire engines labelled as such.

## What's already wired up in this zip (works right now, no training needed)
`config.py` → `ENABLE_OCR=1` by default now, `requirements.txt` has `easyocr`. When a
`car`/`bus`/`truck` box is large enough, `engine.py` reads the crop for text like
"AMBULANCE" / "FIRE ENGINE" / "108" (edit `OCR_KEYWORDS` in `config.py` to match what's
actually printed on your test footage) and relabels it. This genuinely works, but only when
the vehicle's markings are legible on camera — distance, angle and resolution all matter.

```bash
pip install -r requirements.txt    # now installs easyocr too
python app.py                      # OCR fallback is on by default
```

## The real fix: a trained visual classifier (recommended)
`engine.py` also checks for a trained classifier at `models/emergency_cnn.pt`
(`emergency_classifier.py`) **before** the OCR check — it looks at how the vehicle looks
(light bar, markings, shape) instead of needing readable text, so it keeps working at angles
and distances OCR can't. It's a no-op until that file exists. Here's how to produce it.

**Why a classifier and not a retrained YOLO detector:** fine-tuning `yolov8n.pt` itself on an
ambulance/fire-truck-only dataset resets its class head to that dataset's classes — it would
likely forget `car`/`bus`/`truck`/`motorcycle` in the process, since standard `model.train()`
doesn't preserve the original 80 COCO classes alongside new ones. A separate classifier that
just relabels boxes YOLO already found avoids that entirely, and is small enough to train on
a CPU.

### 1. Get images
Public datasets with ambulance + fire-truck classes (bounding-box labelled), free with a
Roboflow account:
- https://universe.roboflow.com/testing-iyut3/emergency-vehicle-ff3rh — classes: Ambulance, Fire_truck, News_van, Normal Vehicle, Police_car
- https://universe.roboflow.com/wogud/emergency-2-rmo7x — classes: Car, Ambulance, Firetruck

I haven't personally vetted label quality on either — skim a few sample images on the page
before committing to one. Open the dataset page → **Download Dataset** → format **YOLOv8** →
either download the zip or copy the generated `roboflow.download(...)` snippet (needs a free
API key from your Roboflow account settings).

### 2. Crop boxes into the 3-class folder layout
These datasets are bounding-box labelled for detection, not pre-cropped. Convert them:
```bash
pip install pyyaml opencv-python
python crop_dataset.py --src path/to/downloaded_dataset --out dataset
```
Edit `CLASS_MAP` at the top of `crop_dataset.py` first if the dataset's class names differ
from what's listed there (check with `python -c "import yaml; print(yaml.safe_load(open('path/to/downloaded_dataset/data.yaml'))['names'])"`).

### 3. Train
Locally (CPU is fine — backbone is frozen, only the last layer trains):
```bash
pip install torch torchvision
python train_emergency.py --data dataset --epochs 12 --out models/emergency_cnn.pt
```
Or paste into a free Colab GPU notebook (colab.research.google.com → new notebook) for a
bigger dataset / faster runs — upload `dataset/` (zip it first) and `train_emergency.py`,
then:
```python
!pip install torch torchvision -q
!python train_emergency.py --data dataset --epochs 12 --out emergency_cnn.pt
from google.colab import files
files.download("emergency_cnn.pt")
```

### 4. Drop it in
Put the resulting file at `models/emergency_cnn.pt` in this project. No code changes needed —
`engine.py` picks it up on the next run.

### Bonus: pulling training crops out of your own dispatch-cam footage
If you have real traffic/dispatch-cam clips, `extract_from_dispatch_video.py` auto-crops
labelled training images straight out of them — no manual box-drawing. You tell it which
seconds a fire engine/ambulance is on screen (watch the video once; unit-ID overlays like
`E18`/`M18`/`L6` are a good guide if your footage has them), it samples frames in that window,
runs your existing `yolov8n.pt` to find the vehicle, crops it into the right class folder, and
crops every *other* vehicle in the same frame as a free `normal` negative.
```bash
python extract_from_dispatch_video.py --video training_clips/dispatch_clip_1.mp4 --out dataset
python extract_from_dispatch_video.py --video training_clips/dispatch_clip_2.mp4 --out dataset
```
`CLIPS` at the top of that script already has entries for both bundled clips:
- `dispatch_clip_1.mp4` — Seattle Fire dispatch-cam, ~104s: E35/E18/L8/L6 (`fire_engine`), M18
  medic unit (`ambulance`). Left out: `B4`/`B6` (Battalion Chief command vehicles, not fire
  engines), `RB1`/`RB5` (rehab unit, mostly off-screen here), `SATT2` (safety/air truck, too
  hazy to confirm) — verify and add those yourself if you want them in.
- `dispatch_clip_2.mp4` — intersection CCTV, 27s, a car/motorcycle collision: a white
  European-liveried ambulance (red/orange stripe, blue light bar) arrives ~20s, on screen to
  the end. No fire engine in this one. Note: that livery style looked Baltic/European, not
  Indian — useful for general shape/light-bar variety, but don't rely on it alone for
  livery-specific accuracy in your deployment region.

Add your own clips to `training_clips/` and a matching `CLIPS` entry the same way. Two videos
only gets you ~45-75 crops total — run this against more clips (same `--out dataset` each time,
it only adds files) and merge with `crop_dataset.py`'s output from a public dataset for real
volume before training.

This footage is small-sample on its own, and I can't verify the licensing/rights on traffic-cam
clips shared over WhatsApp — worth checking you're clear to use them before relying on this
beyond a personal/academic project.

### Tuning
In `config.py`: `EMERG_CLF_MIN_CONF` (default 0.75) — raise it if you get false positives
(normal trucks flagged as ambulances), lower it if real ambulances aren't triggering.
`EMERG_CLF_MIN_BOX_H` / `EMERG_CLF_MAX_TRIES` / `EMERG_CLF_EVERY_N` control how large a box
has to be and how often it's rechecked per vehicle, same idea as the existing `OCR_*` knobs.

### Known limits of this approach, upfront
- Classifier accuracy depends entirely on your training data matching your real camera
  conditions (angle, lighting, local ambulance liveries) — the two Roboflow sets above are a
  starting point, not guaranteed to generalize to Indian traffic-cam footage. Add your own
  labelled crops (ambulance / fire_engine / normal folders) alongside them if accuracy is poor.
- It relabels a box YOLO already drew (as car/bus/truck) — if YOLO misses the vehicle
  entirely (unusual angle, heavy occlusion), there's nothing to classify.
- I could not train this for you inside this chat environment — no GPU here, and the dataset
  hosts above aren't reachable from this sandbox's network. The scripts are tested for syntax
  and reviewed carefully, but the actual training run (and real-world accuracy) is on you to
  verify.
