"""Zero-shot emergency-vehicle recognition — works with NO training and NO API key.

This is the piece that was missing. Stock yolov8n.pt (COCO) has no 'ambulance' or
'fire_engine' class, so an ambulance/fire engine only ever came back as car/bus/truck.
The old fallbacks both dead-ended: models/emergency_cnn.pt had to be trained first, and
OCR only works when the lettering is large and legible.

This module uses CLIP (OpenAI's image-text model) in ZERO-SHOT mode: it compares the
vehicle crop against text prompts like "a photo of an ambulance" and "a red fire engine"
and picks the closest. No dataset, no training, no fine-tuning. Weights download once
(~350 MB for ViT-B/32) from HuggingFace and are cached; after that it is fully offline
and runs on CPU in roughly 60-120 ms per crop.

Backends are tried in order and the first one that imports wins:
    1. transformers  -> CLIPModel  ("openai/clip-vit-base-patch32")
    2. open_clip     -> ViT-B-32 / laion2b_s34b_b79k

If neither is installed the module stays silently disabled (classify_crop returns None)
and the rest of the app behaves exactly as before — nothing here is load-bearing.

    pip install transformers            # that is the whole setup step
"""
import os
import threading

import cv2
import numpy as np

from config import C

# Must match engine.EMERG_TYPES for the two emergency labels.
EMERG_LABELS = ("ambulance", "fire_engine")

# Prompt ensembles. Score for a class = best similarity across its prompts, so adding
# more phrasings only helps. 'normal' deliberately lists the things most often confused
# with an ambulance (white vans, minibuses, police cars) and with a fire engine (any red
# truck) — without these negatives CLIP over-fires on colour alone.
PROMPTS = {
    "ambulance": [
        "a photo of an ambulance",
        "an ambulance with a red cross and emergency light bar",
        "a hospital emergency ambulance van on the road",
        "a white and red ambulance with the word AMBULANCE written on it",
        "an emergency medical services vehicle",
        "a paramedic ambulance responding to an emergency",
    ],
    "fire_engine": [
        "a photo of a fire engine",
        "a red fire truck with ladders and hoses",
        "a fire brigade rescue truck with emergency lights",
        "a fire department engine responding to a fire",
        "a fire tender with a water cannon",
        "a firefighting vehicle",
    ],
    "normal": [
        "a photo of an ordinary car",
        "a white delivery van",
        "a passenger bus",
        "a cargo truck carrying goods",
        "a red car",
        "a red truck carrying cargo",
        "a police car",
        "a taxi",
        "a motorcycle",
        "a parked vehicle on a street",
        "an empty road",
    ],
}


class _ClipBackend:
    """Thin wrapper so transformers and open_clip look identical to the caller."""

    def __init__(self):
        self.kind = None
        self.model = None
        self.text_feats = None      # (n_prompts, dim) L2-normalised
        self.owner = []             # class label for each row of text_feats
        self._load()

    # ---------- loading ----------
    def _load(self):
        order = [b.strip() for b in C.CLIP_BACKEND.split(",") if b.strip()]
        for b in order:
            try:
                if b == "transformers" and self._load_transformers():
                    return
                if b == "open_clip" and self._load_open_clip():
                    return
            except Exception as e:                       # noqa: BLE001 - optional dependency
                print(f"[emergency_vision] backend '{b}' FAILED: {type(e).__name__}: {e}")
                if C.CLIP_DEBUG:
                    import traceback
                    traceback.print_exc()
        print("[emergency_vision] no CLIP backend loaded — zero-shot detection is OFF. "
              "Run:  pip install transformers")

    def _load_transformers(self):
        """Load CLIP across transformers v4 and v5.

        v4 exposed CLIPModel + CLIPProcessor and get_*_features returned a plain tensor.
        v5 reorganised both: Auto* resolves the classes on either version, and get_*_features
        can return a BaseModelOutputWithPooling instead of a tensor. _encode_text/_encode_image
        below handle that, falling back to the full model forward (text_embeds / image_embeds),
        which is the one API that has been stable across both.
        """
        import torch
        from PIL import Image
        name = C.CLIP_MODEL_HF
        self.torch = torch
        model = proc = None
        errors = []
        try:
            from transformers import AutoModel, AutoProcessor
            model = AutoModel.from_pretrained(name)
            proc = AutoProcessor.from_pretrained(name)
        except Exception as e:                               # noqa: BLE001
            errors.append(f"Auto* path: {type(e).__name__}: {e}")
            try:
                from transformers import CLIPModel, CLIPProcessor
                model = CLIPModel.from_pretrained(name)
                proc = CLIPProcessor.from_pretrained(name)
            except Exception as e2:                          # noqa: BLE001
                errors.append(f"CLIP* path: {type(e2).__name__}: {e2}")
                raise RuntimeError(" | ".join(errors))
        self.model, self.proc = model, proc
        self.model.eval()
        self._dummy_img = Image.new("RGB", (224, 224), (0, 0, 0))
        self._text_route = self._img_route = None            # decided on first use, then cached

        prompts, owner = [], []
        for label, ps in PROMPTS.items():
            prompts += ps
            owner += [label] * len(ps)
        with torch.no_grad():
            tf = self._encode_text(prompts)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        self.text_feats, self.owner, self.kind = tf, owner, "transformers"
        print(f"[emergency_vision] CLIP ready (transformers / {name}, "
              f"text route={self._text_route}, image route={self._img_route or 'pending'})")
        return True

    def _as_tensor(self, out):
        """get_*_features returns a tensor on v4 and may return an output object on v5."""
        if self.torch.is_tensor(out):
            return out
        for attr in ("text_embeds", "image_embeds", "embeds"):
            v = getattr(out, attr, None)
            if v is not None and self.torch.is_tensor(v):
                return v
        return None

    def _encode_text(self, prompts):
        """Prompts -> (n, dim) embedding tensor, whichever API this transformers version offers."""
        tok = self.proc(text=prompts, return_tensors="pt", padding=True)
        if self._text_route in (None, "features"):
            try:
                t = self._as_tensor(self.model.get_text_features(**tok))
                if t is not None:
                    self._text_route = "features"
                    return t
            except Exception:                                # noqa: BLE001
                pass
        # Fallback: full forward. text_embeds is independent of the image, so one dummy is enough.
        inp = self.proc(text=prompts, images=self._dummy_img, return_tensors="pt", padding=True)
        out = self.model(**inp)
        t = getattr(out, "text_embeds", None)
        if t is None:
            raise RuntimeError(f"cannot get text embeddings from {type(self.model).__name__} "
                               f"(forward returned {type(out).__name__})")
        self._text_route = "forward"
        return t

    def _encode_image(self, pil_img):
        """PIL image -> (1, dim) embedding tensor."""
        if self._img_route in (None, "features"):
            try:
                inp = self.proc(images=pil_img, return_tensors="pt")
                t = self._as_tensor(self.model.get_image_features(**inp))
                if t is not None:
                    self._img_route = "features"
                    return t
            except Exception:                                # noqa: BLE001
                pass
        inp = self.proc(text=["a photo"], images=pil_img, return_tensors="pt", padding=True)
        out = self.model(**inp)
        t = getattr(out, "image_embeds", None)
        if t is None:
            raise RuntimeError(f"cannot get image embeddings from {type(self.model).__name__} "
                               f"(forward returned {type(out).__name__})")
        self._img_route = "forward"
        return t

    def _load_open_clip(self):
        import torch
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            C.CLIP_MODEL_OPEN, pretrained=C.CLIP_PRETRAIN_OPEN)
        tokenizer = open_clip.get_tokenizer(C.CLIP_MODEL_OPEN)
        self.torch, self.model, self.preprocess = torch, model, preprocess
        model.eval()
        prompts, owner = [], []
        for label, ps in PROMPTS.items():
            prompts += ps
            owner += [label] * len(ps)
        with torch.no_grad():
            tf = model.encode_text(tokenizer(prompts))
            tf = tf / tf.norm(dim=-1, keepdim=True)
        self.text_feats, self.owner, self.kind = tf, owner, "open_clip"
        print(f"[emergency_vision] CLIP ready (open_clip / {C.CLIP_MODEL_OPEN})")
        return True

    # ---------- inference ----------
    def evaluate(self, crop_bgr):
        """BGR crop -> (probs, sims).

        sims  = raw best cosine similarity per class (what actually discriminates).
        probs = tempered softmax over sims, for a readable confidence number.

        CLIP's own logit scale is 100, which saturates every call to ~0.99 and makes a
        confidence threshold meaningless. C.CLIP_LOGIT_SCALE (default 50) keeps the number
        informative, and the caller additionally requires a raw similarity MARGIN over the
        best non-emergency class — that margin, not the softmax value, is the real test.
        """
        from PIL import Image
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        torch = self.torch
        with torch.no_grad():
            if self.kind == "transformers":
                f = self._encode_image(img)
            else:
                f = self.model.encode_image(self.preprocess(img).unsqueeze(0))
            f = f / f.norm(dim=-1, keepdim=True)
            sim = (f @ self.text_feats.T)[0]             # cosine similarity per prompt
        sims = {}
        for label in PROMPTS:
            idx = [i for i, o in enumerate(self.owner) if o == label]
            sims[label] = float(sim[idx].max())          # best-matching prompt for that class
        vals = np.array([sims[k] for k in PROMPTS]) * C.CLIP_LOGIT_SCALE
        vals = np.exp(vals - vals.max())
        vals = vals / vals.sum()
        return {k: float(v) for k, v in zip(PROMPTS, vals)}, sims

    def scores(self, crop_bgr):
        return self.evaluate(crop_bgr)[0]


_backend = None
_lock = threading.Lock()


def get_backend():
    """Lazy singleton — the model loads on first use, not at import."""
    global _backend
    if _backend is None:
        with _lock:
            if _backend is None:
                _backend = _ClipBackend() if C.ENABLE_CLIP else False
    return _backend or None


def ready():
    b = get_backend()
    return bool(b and b.kind)


def backend_name():
    b = get_backend()
    return b.kind if (b and b.kind) else "none"


def classify_crop(crop_bgr, min_conf=None):
    """Vehicle crop (BGR) -> (label, confidence, 'clip') or None.

    label is 'ambulance' or 'fire_engine'. A confident 'normal' call, a low-confidence
    call, or no CLIP backend all return None.
    """
    b = get_backend()
    if not (b and b.kind) or crop_bgr is None or crop_bgr.size == 0:
        return None
    if min_conf is None:
        min_conf = C.CLIP_MIN_CONF
    try:
        probs, sims = b.evaluate(crop_bgr)
    except Exception as e:                               # noqa: BLE001
        print(f"[emergency_vision] inference failed: {e}")
        return None
    label = max(probs, key=probs.get)
    if label == "normal" or probs[label] < min_conf:
        return None
    # Raw-similarity margin over the best competing class. Without this, two nearly tied
    # classes still produce a high softmax value and a confident-looking false alarm.
    rival = max((k for k in sims if k != label), key=lambda k: sims[k])
    if sims[label] - sims[rival] < C.CLIP_MIN_MARGIN:
        return None
    return label, probs[label], "clip"


def raw_scores(crop_bgr):
    """Full probability dict, for debugging / the 'why' column in the image report."""
    b = get_backend()
    if not (b and b.kind) or crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        return b.scores(crop_bgr)
    except Exception:                                    # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Last-resort colour heuristic (OFF by default — set ENABLE_COLOR_FALLBACK=1).
# A fire engine is overwhelmingly red; an ambulance is usually white with a red or blue
# stripe plus a roof light bar. This flags "suspected" only, never a confident call,
# because a plain red truck trips it. Use it only when CLIP cannot be installed.
# ---------------------------------------------------------------------------
def color_hint(crop_bgr, yolo_cls):
    if not C.ENABLE_COLOR_FALLBACK or crop_bgr is None or crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > 90) & (v > 60)
    red = (((h <= 10) | (h >= 170)) & sat).mean()
    white = ((s < 45) & (v > 170)).mean()
    blue = (((h >= 100) & (h <= 130)) & sat).mean()
    if yolo_cls in ("truck", "bus") and red > C.COLOR_RED_RATIO:
        return "fire_engine", float(red), "color"
    if yolo_cls in ("car", "truck", "bus") and white > 0.45 and (red > 0.06 or blue > 0.06):
        return "ambulance", float(white), "color"
    return None
