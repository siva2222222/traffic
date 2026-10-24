"""Optional trained image classifier for emergency-vehicle re-labelling.

Companion to the OCR fallback in engine.py: given a cropped car/bus/truck box,
predicts 'ambulance' / 'fire_engine' / None (normal vehicle) from how the vehicle
LOOKS — light bar, markings, livery, shape — rather than needing legible text.
This is a classifier (one label per crop), not an object detector, so it's small
and quick to train, and it doesn't touch yolov8n.pt or risk it forgetting its
normal car/bus/truck/motorcycle classes.

Train it with train_emergency.py (steps in TRAIN_EMERGENCY_MODEL.md) and drop the
resulting weights at models/emergency_cnn.pt. engine.py picks it up automatically
on the next run and silently stays off (always returns None) if that file isn't
there yet — nothing else in the app depends on it.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

# ImageFolder (used in train_emergency.py) sorts class subfolders alphabetically,
# so this order must stay exactly ambulance, fire_engine, normal.
CLASSES = ["ambulance", "fire_engine", "normal"]

_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, len(CLASSES))
    return m


class EmergencyClassifier:
    def __init__(self, weights_path, min_conf=0.75, device="cpu"):
        self.device = device
        self.min_conf = min_conf
        self.model = _build_model()
        state = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval().to(device)

    @torch.no_grad()
    def predict(self, crop_bgr):
        """crop_bgr: OpenCV BGR numpy crop. Returns 'ambulance' / 'fire_engine' / None
        (None covers both a confident 'normal' call and a low-confidence call of any kind)."""
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        x = _TFM(img).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        label = CLASSES[idx.item()]
        if label == "normal" or conf.item() < self.min_conf:
            return None
        return label
