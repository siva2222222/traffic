
import os

BASE = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, ".env"))
except ImportError:
    pass


class C:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret")
    DB_PATH = os.path.join(BASE, "data", "traffic.db")
    UPLOAD_DIR = os.path.join(BASE, "static", "uploads")
    EVIDENCE_DIR = os.path.join(BASE, "static", "evidence")
    LANES_FILE = os.path.join(BASE, "zones", "lanes.json")

    # --------------------------------------------------
    # YOLO DETECTION - OPTIMIZED FOR CPU / RENDER
    # --------------------------------------------------
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
    CONF = float(os.getenv("CONF", "0.35"))

    # Smaller frames reduce CPU usage.
    WORK_WIDTH = int(os.getenv("WORK_WIDTH", "640"))

    # Process one frame out of every three.
    PROCESS_EVERY = int(os.getenv("PROCESS_EVERY", "3"))

    # Playback speed for uploaded videos.
    PLAYBACK_SPEED = float(os.getenv("PLAYBACK_SPEED", "1.0"))

    # --------------------------------------------------
    # OCR - EMERGENCY VEHICLE TEXT RECOGNITION
    # --------------------------------------------------
    ENABLE_OCR = os.getenv("ENABLE_OCR", "1") == "1"

    OCR_MIN_BOX_H = int(os.getenv("OCR_MIN_BOX_H", "45"))
    OCR_MAX_TRIES = int(os.getenv("OCR_MAX_TRIES", "3"))
    OCR_EVERY_N = int(os.getenv("OCR_EVERY_N", "12"))
    OCR_MAX_PER_FRAME = int(os.getenv("OCR_MAX_PER_FRAME", "1"))

    OCR_KEYWORDS = {
        "ambulance": [
            "AMBULAN",
            "EMERGENCY",
            "108",
            "102",
            "PARAMEDIC",
            "MEDICAL",
        ],
        "fire_engine": [
            "FIRE ENGINE",
            "FIRE SERVICE",
            "FIRE BRIGADE",
            "FIRE TENDER",
            "FIRE FORCE",
            "FIRE DEPT",
            "FIRE RESCUE",
            "FIRE DEPARTMENT",
            "FIRE",
            "DFES",
            "RESCUE",
        ],
    }

    # --------------------------------------------------
    # OPTIONAL TRAINED EMERGENCY CLASSIFIER
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
    EMERG_CLF_EVERY_N = int(
        os.getenv("EMERG_CLF_EVERY_N", "5")
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
    CLIP_MODEL_OPEN = os.getenv(
        "CLIP_MODEL_OPEN", "ViT-B-32"
    )
    CLIP_PRETRAIN_OPEN = os.getenv(
        "CLIP_PRETRAIN_OPEN", "laion2b_s34b_b79k"
    )

    CLIP_MIN_CONF = float(os.getenv("CLIP_MIN_CONF", "0.45"))
    CLIP_LOGIT_SCALE = float(
        os.getenv("CLIP_LOGIT_SCALE", "50")
    )
    CLIP_MIN_MARGIN = float(
        os.getenv("CLIP_MIN_MARGIN", "0.006")
    )
    CLIP_MIN_BOX_H = int(os.getenv("CLIP_MIN_BOX_H", "40"))

    # Run CLIP less frequently to reduce CPU usage.
    CLIP_EVERY_N = int(os.getenv("CLIP_EVERY_N", "12"))
    CLIP_MAX_TRIES = int(os.getenv("CLIP_MAX_TRIES", "4"))
    CLIP_MAX_PER_FRAME = int(
        os.getenv("CLIP_MAX_PER_FRAME", "1")
    )

    # --------------------------------------------------
    # IMAGE ANALYSIS
    # --------------------------------------------------
    IMAGE_YOLO_MODEL = os.getenv("IMAGE_YOLO_MODEL", "")
    IMAGE_CONF = float(os.getenv("IMAGE_CONF", "0.25"))

    # --------------------------------------------------
    # COLOR FALLBACK
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

    # Wrong-way detection
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
    # OPTIONAL LLM SETTINGS
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
