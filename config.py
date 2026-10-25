<<<<<<< HEAD
import os
=======

import os

>>>>>>> origin/master
BASE = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, ".env"))
except ImportError:
    pass

<<<<<<< HEAD
class C:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret")
=======

class C:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret")

>>>>>>> origin/master
    DB_PATH = os.path.join(BASE, "data", "traffic.db")
    UPLOAD_DIR = os.path.join(BASE, "static", "uploads")
    EVIDENCE_DIR = os.path.join(BASE, "static", "evidence")
    LANES_FILE = os.path.join(BASE, "zones", "lanes.json")

<<<<<<< HEAD
    # Detection
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")   # or your custom weights with 'ambulance' / 'fire_engine' classes
    CONF = float(os.getenv("CONF", "0.35"))
    WORK_WIDTH = 960                 # frames are resized to this width (calibration is relative to it)
    PROCESS_EVERY = int(os.getenv("PROCESS_EVERY", "2"))  # process every Nth frame of video files
    # Dashboard playback pace for uploaded video files, as a fraction of real-time: 1.0 = a 40s
    # clip takes ~40s, 0.65 = it takes ~40/0.65 =~ 62s. Lower = slower/easier to watch. This only
    # paces the on-screen playback speed; vehicle-speed/violation timing still uses the video's
    # own frame timestamps, so detection accuracy is unaffected.
    PLAYBACK_SPEED = float(os.getenv("PLAYBACK_SPEED", "0.3"))
    # EasyOCR text check for ambulance / fire engine markings. ON by default now — stock yolov8n.pt (COCO)
    # has no 'ambulance'/'fire_engine' classes, so without this (or custom-trained weights, see YOLO_MODEL
    # above / train_emergency.py) those vehicles only ever show up as 'car'/'bus'/'truck'.
    ENABLE_OCR = os.getenv("ENABLE_OCR", "1") == "1"
    OCR_MIN_BOX_H = int(os.getenv("OCR_MIN_BOX_H", "45"))   # lower = tries OCR on smaller/farther boxes too
    OCR_MAX_TRIES = int(os.getenv("OCR_MAX_TRIES", "6"))    # attempts per track before giving up
    OCR_EVERY_N = int(os.getenv("OCR_EVERY_N", "8"))        # frame-stride between OCR attempts
    OCR_MAX_PER_FRAME = int(os.getenv("OCR_MAX_PER_FRAME", "1"))  # EasyOCR is ~300-800 ms/crop on CPU
    # OCR text that marks a car/bus/truck box as that emergency type (checked when ENABLE_OCR=1 and no
    # dedicated 'ambulance'/'fire_engine' YOLO class fired). Stock COCO weights have no such classes, so this
    # keyword check (plus the visual class path below) is what flags plain yolov8n.pt detections as emergency.
    # Add any text actually printed on your test vehicles (state ambulance codes, "108", local fire-brigade
    # wording, etc.) — OCR can only match what's listed here.
    OCR_KEYWORDS = {
        "ambulance": ["AMBULAN", "EMERGENCY", "108", "102", "PARAMEDIC", "MEDICAL"],
        "fire_engine": ["FIRE ENGINE", "FIRE SERVICE", "FIRE BRIGADE", "FIRE TENDER", "FIRE FORCE",
                         "FIRE DEPT", "FIRE RESCUE", "FIRE DEPARTMENT", "FIRE", "DFES", "RESCUE"],
    }
    # Optional trained visual classifier (models/emergency_cnn.pt — see train_emergency.py / TRAIN_EMERGENCY_MODEL.md).
    # Looks at how the vehicle looks (light bar, markings, shape) instead of reading text; runs before the OCR
    # fallback above and is a no-op until that weights file exists.
    EMERG_CLF_MIN_CONF = float(os.getenv("EMERG_CLF_MIN_CONF", "0.75"))   # min softmax confidence to accept a label
    EMERG_CLF_MIN_BOX_H = int(os.getenv("EMERG_CLF_MIN_BOX_H", "50"))
    EMERG_CLF_MAX_TRIES = int(os.getenv("EMERG_CLF_MAX_TRIES", "8"))
    EMERG_CLF_EVERY_N = int(os.getenv("EMERG_CLF_EVERY_N", "5"))

    # ---- Zero-shot emergency-vehicle recognition (CLIP) -------------------------------
    # THIS is what makes an ambulance / fire engine detectable without training anything.
    # Stock yolov8n.pt (COCO) has no such classes, so without this they stay 'car'/'bus'/'truck'.
    # Needs:  pip install transformers      (weights download once, ~350 MB, then offline, CPU-fine)
    ENABLE_CLIP = os.getenv("ENABLE_CLIP", "1") == "1"
    CLIP_BACKEND = os.getenv("CLIP_BACKEND", "transformers,open_clip")   # tried in this order
    CLIP_DEBUG = os.getenv("CLIP_DEBUG", "0") == "1"            # 1 = full traceback when a backend fails
    CLIP_MODEL_HF = os.getenv("CLIP_MODEL_HF", "openai/clip-vit-base-patch32")
    CLIP_MODEL_OPEN = os.getenv("CLIP_MODEL_OPEN", "ViT-B-32")
    CLIP_PRETRAIN_OPEN = os.getenv("CLIP_PRETRAIN_OPEN", "laion2b_s34b_b79k")
    CLIP_MIN_CONF = float(os.getenv("CLIP_MIN_CONF", "0.45"))   # lower = more sensitive, more false alarms
    CLIP_LOGIT_SCALE = float(os.getenv("CLIP_LOGIT_SCALE", "50"))   # CLIP's own 100 saturates every call to ~0.99
    CLIP_MIN_MARGIN = float(os.getenv("CLIP_MIN_MARGIN", "0.006"))  # raw cosine gap over the runner-up class
    CLIP_MIN_BOX_H = int(os.getenv("CLIP_MIN_BOX_H", "40"))     # skip boxes shorter than this in VIDEO
    CLIP_EVERY_N = int(os.getenv("CLIP_EVERY_N", "6"))          # VIDEO only: frame stride between checks
    CLIP_MAX_TRIES = int(os.getenv("CLIP_MAX_TRIES", "10"))     # VIDEO only: attempts per track
    # CPU budget. CLIP is ~60-120 ms/crop, so without a cap a frame with 8 vehicles costs ~1 s and
    # the video falls far behind real time. Biggest box wins the slot (nearest = most readable).
    CLIP_MAX_PER_FRAME = int(os.getenv("CLIP_MAX_PER_FRAME", "2"))

    # Still-image analysis (/admin/image). No stride or retry gating here: every vehicle in the
    # image is checked once, which is why images behave differently from a single video frame.
    IMAGE_YOLO_MODEL = os.getenv("IMAGE_YOLO_MODEL", "")        # blank = use YOLO_MODEL; try yolov8s.pt for better recall
    IMAGE_CONF = float(os.getenv("IMAGE_CONF", "0.25"))         # lower than video CONF: stills get one shot only

    # Last-resort colour heuristic (fire engine = mostly red, ambulance = white + stripe).
    # OFF by default: a plain red cargo truck trips it. Only enable if you cannot install CLIP.
    ENABLE_COLOR_FALLBACK = os.getenv("ENABLE_COLOR_FALLBACK", "0") == "1"
    COLOR_RED_RATIO = float(os.getenv("COLOR_RED_RATIO", "0.42"))

    # Violation thresholds
    SPEED_LIMIT_KMH = float(os.getenv("SPEED_LIMIT", "40"))
    PARK_SECONDS = 10
    # Wrong-way: vehicle must move against its lane direction for WW_SUSTAIN s, covering >= WW_DIST px (reference frame)
    WW_WINDOW, WW_SPAN, WW_DIST, WW_SUSTAIN, WW_COS = 1.5, 1.0, 35, 0.5, -0.7
    HEADING_MIN_PX = 12    # min movement (px/s, reference frame) before a vehicle shows a heading arrow
    DIR_LEARN_MIN = 6      # vehicles needed to learn a lane's normal direction automatically
    STOP_MARGIN_PX = 60

    # Signal timing (seconds)
    MIN_GREEN, MAX_GREEN, PER_VEHICLE, YELLOW, ALL_RED = 8, 45, 2, 3, 1

    # Indicative fines (INR) - set according to your local rules
    FINES = {"Red Light Jump": 1000, "Stop Line Violation": 500, "Wrong Way": 1000,
             "Speeding": 1000, "Illegal Parking": 500, "Lane Violation": 500}

    # LLM: none | gemini | anthropic | groq
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
=======
    # --------------------------------------------------
    # YOLO VEHICLE DETECTION - PERFORMANCE
    # --------------------------------------------------
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
    CONF = float(os.getenv("CONF", "0.35"))

    # Processing resolution. Increase to 640 if needed.
    WORK_WIDTH = max(
        320, int(os.getenv("WORK_WIDTH", "480"))
    )

    # 1 = process every frame; 2 = every second frame.
    PROCESS_EVERY = max(
        1, int(os.getenv("PROCESS_EVERY", "1"))
    )

    # 0 = do not intentionally discard late video frames.
    SKIP_LATE_FRAMES = (
        os.getenv("SKIP_LATE_FRAMES", "0") == "1"
    )

    # 1.0 = normal playback speed.
    PLAYBACK_SPEED = max(
        0.05, float(os.getenv("PLAYBACK_SPEED", "1.0"))
    )

    # --------------------------------------------------
    # OCR - EMERGENCY VEHICLE TEXT RECOGNITION
    # --------------------------------------------------
    ENABLE_OCR = os.getenv("ENABLE_OCR", "1") == "1"

    OCR_MIN_BOX_H = int(os.getenv("OCR_MIN_BOX_H", "45"))
    OCR_MAX_TRIES = int(os.getenv("OCR_MAX_TRIES", "6"))
    OCR_EVERY_N = max(1, int(os.getenv("OCR_EVERY_N", "12")))
    OCR_MAX_PER_FRAME = max(
        0, int(os.getenv("OCR_MAX_PER_FRAME", "1"))
    )

    OCR_KEYWORDS = {
        "ambulance": [
            "AMBULAN", "EMERGENCY", "108", "102",
            "PARAMEDIC", "MEDICAL",
        ],
        "fire_engine": [
            "FIRE ENGINE", "FIRE SERVICE", "FIRE BRIGADE",
            "FIRE TENDER", "FIRE FORCE", "FIRE DEPT",
            "FIRE RESCUE", "FIRE DEPARTMENT", "FIRE",
            "DFES", "RESCUE",
        ],
    }

    # --------------------------------------------------
    # OPTIONAL EMERGENCY CLASSIFIER
    # --------------------------------------------------
    EMERG_CLF_MIN_CONF = float(
        os.getenv("EMERG_CLF_MIN_CONF", "0.75")
    )
    EMERG_CLF_MIN_BOX_H = int(
        os.getenv("EMERG_CLF_MIN_BOX_H", "50")
    )
    EMERG_CLF_MAX_TRIES = int(
        os.getenv("EMERG_CLF_MAX_TRIES", "8")
    )
    EMERG_CLF_EVERY_N = max(
        1, int(os.getenv("EMERG_CLF_EVERY_N", "5"))
    )

    # --------------------------------------------------
    # CLIP - ZERO-SHOT EMERGENCY VEHICLE RECOGNITION
    # --------------------------------------------------
    ENABLE_CLIP = os.getenv("ENABLE_CLIP", "1") == "1"

    CLIP_BACKEND = os.getenv(
        "CLIP_BACKEND", "transformers,open_clip"
    )
    CLIP_DEBUG = os.getenv("CLIP_DEBUG", "0") == "1"

    CLIP_MODEL_HF = os.getenv(
        "CLIP_MODEL_HF", "openai/clip-vit-base-patch32"
    )
    CLIP_MODEL_OPEN = os.getenv("CLIP_MODEL_OPEN", "ViT-B-32")
    CLIP_PRETRAIN_OPEN = os.getenv(
        "CLIP_PRETRAIN_OPEN", "laion2b_s34b_b79k"
    )

    CLIP_MIN_CONF = float(
        os.getenv("CLIP_MIN_CONF", "0.45")
    )
    CLIP_LOGIT_SCALE = float(
        os.getenv("CLIP_LOGIT_SCALE", "50")
    )
    CLIP_MIN_MARGIN = float(
        os.getenv("CLIP_MIN_MARGIN", "0.006")
    )
    CLIP_MIN_BOX_H = int(os.getenv("CLIP_MIN_BOX_H", "40"))

    CLIP_EVERY_N = max(
        1, int(os.getenv("CLIP_EVERY_N", "12"))
    )
    CLIP_MAX_TRIES = max(
        0, int(os.getenv("CLIP_MAX_TRIES", "10"))
    )
    CLIP_MAX_PER_FRAME = max(
        0, int(os.getenv("CLIP_MAX_PER_FRAME", "1"))
    )

    # --------------------------------------------------
    # STILL-IMAGE ANALYSIS
    # --------------------------------------------------
    IMAGE_YOLO_MODEL = os.getenv("IMAGE_YOLO_MODEL", "")
    IMAGE_CONF = float(os.getenv("IMAGE_CONF", "0.25"))

    # --------------------------------------------------
    # OPTIONAL COLOUR-BASED EMERGENCY DETECTION
    # --------------------------------------------------
    ENABLE_COLOR_FALLBACK = (
        os.getenv("ENABLE_COLOR_FALLBACK", "0") == "1"
    )
    COLOR_RED_RATIO = float(
        os.getenv("COLOR_RED_RATIO", "0.42")
    )

    # --------------------------------------------------
    # TRAFFIC VIOLATION SETTINGS
    # --------------------------------------------------
    SPEED_LIMIT_KMH = float(os.getenv("SPEED_LIMIT", "40"))
    PARK_SECONDS = 10

    WW_WINDOW = 1.5
    WW_SPAN = 1.0
    WW_DIST = 35
    WW_SUSTAIN = 0.5
    WW_COS = -0.7

    HEADING_MIN_PX = 12
    DIR_LEARN_MIN = 6
    STOP_MARGIN_PX = 60

    # --------------------------------------------------
    # TRAFFIC SIGNAL TIMING
    # --------------------------------------------------
    MIN_GREEN = 8
    MAX_GREEN = 45
    PER_VEHICLE = 2
    YELLOW = 3
    ALL_RED = 1

    # --------------------------------------------------
    # INDICATIVE FINES (INR)
    # --------------------------------------------------
    FINES = {
        "Red Light Jump": 1000,
        "Stop Line Violation": 500,
        "Wrong Way": 1000,
        "Speeding": 1000,
        "Illegal Parking": 500,
        "Lane Violation": 500,
    }

    # --------------------------------------------------
    # OPTIONAL LLM INTEGRATION
    # --------------------------------------------------
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL", "gemini-2.0-flash"
    )

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv(
        "ANTHROPIC_MODEL", "claude-sonnet-5"
    )

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv(
        "GROQ_MODEL", "openai/gpt-oss-120b"
    )
>>>>>>> origin/master
