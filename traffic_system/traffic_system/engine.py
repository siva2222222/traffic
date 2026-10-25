"""Vision engine: YOLOv8 detection + ByteTrack tracking -> lane density, violations, adaptive signal, emergency logic.
`TrafficEngine.process()` holds all rules and is independent of YOLO, so it can be unit-tested with fake detections."""
import os, json, time, math, threading, traceback
from collections import deque, defaultdict
import cv2
import numpy as np
from config import C
import database as db
import audio as audiomod
from stabilizer import CameraStabilizer, apply, apply_poly
from collections import Counter
try:
    from emergency_classifier import EmergencyClassifier       # optional trained visual classifier (see train_emergency.py)
except Exception:
    EmergencyClassifier = None
try:
    import emergency_vision as EV                              # zero-shot CLIP recogniser - no training needed
except Exception:
    EV = None

LANE_ORDER = ["North", "East", "South", "West"]
NAME_MAP = {"car": "car", "motorcycle": "bike", "motorbike": "bike", "bike": "bike", "bus": "bus",
            "truck": "truck", "ambulance": "ambulance",
            # recognised spellings for a custom-trained fire-engine class (yolov8n.pt itself has none of these -
            # see the OCR fallback below for plain COCO weights)
            "fire_engine": "fire_engine", "fire engine": "fire_engine", "firetruck": "fire_engine",
            "fire truck": "fire_engine", "fire-truck": "fire_engine"}
COLORS = {"car": (80, 200, 80), "bike": (255, 160, 40), "bus": (60, 60, 230), "truck": (230, 80, 200),
          "ambulance": (0, 0, 255), "fire_engine": (0, 100, 255)}
EMERG_TYPES = ("ambulance", "fire_engine")
DIR_COLOR = {"North": (214, 111, 47), "South": (11, 158, 245), "East": (37, 48, 217), "West": (75, 156, 26)}
SIG_COLOR = {"green": (0, 200, 0), "yellow": (0, 220, 255), "red": (0, 0, 230)}

def _cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

def _window(hist, secs):
    """(span, dx, dy) between latest point and oldest point within `secs`."""
    t1, x1, y1 = hist[-1]
    for t0, x0, y0 in hist:
        if t1 - t0 <= secs:
            return t1 - t0, x1 - x0, y1 - y0
    return 0.0, 0.0, 0.0

class SignalController:
    """Adaptive control: next green goes to the lane with highest (density + waiting bonus);
    green time scales with queue length; emergency lane pre-empts."""
    def __init__(self, lanes=None):
        self.lanes = list(lanes or LANE_ORDER)
        self.reset()

    def set_lanes(self, lanes):
        self.lanes = list(lanes)
        self.reset()

    def reset(self):
        self.green, self.phase, self.phase_end = None, "ALLRED", 0.0
        self.wait = {l: 0.0 for l in self.lanes}
        self.last_t, self.reason = None, "starting"
        self.green_duration = 0

    def states(self):
        return {l: ("green" if l == self.green and self.phase == "GREEN" else
                    "yellow" if l == self.green and self.phase == "YELLOW" else "red") for l in self.lanes}

    def remaining(self, t):
        return max(0.0, self.phase_end - t)

    def update(self, t, counts, emerg_lane=None):
        dt = 0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        for l in self.lanes:
            self.wait[l] = 0.0 if (l == self.green and self.phase == "GREEN") else self.wait[l] + dt
        if emerg_lane in self.lanes and self.phase == "GREEN":
            if self.green != emerg_lane:
                self.phase_end = min(self.phase_end, t)          # cut green short -> yellow
                self.reason = f"Emergency vehicle in {emerg_lane}: pre-empting {self.green}"
            else:
                self.phase_end = max(self.phase_end, t + 3)     # hold green while the emergency vehicle is there
        if t >= self.phase_end:
            if self.phase == "GREEN":
                self.phase, self.phase_end = "YELLOW", t + C.YELLOW
            elif self.phase == "YELLOW":
                self.phase, self.phase_end = "ALLRED", t + C.ALL_RED
            else:
                self._pick(t, counts, emerg_lane)

    def _pick(self, t, counts, emerg_lane):
        if emerg_lane in self.lanes:
            nxt, why = emerg_lane, f"Emergency priority to {emerg_lane}"
        elif sum(counts.values()) == 0:
            i = (self.lanes.index(self.green) + 1) % len(self.lanes) if self.green else 0
            nxt, why = self.lanes[i], "No traffic: cyclic rotation"
        else:
            score = {l: counts.get(l, 0) + self.wait[l] / 10.0 for l in self.lanes}
            nxt = max(score, key=score.get)
            why = f"{nxt}: highest demand (density {counts.get(nxt, 0)}, waited {self.wait[nxt]:.0f}s)"
        dur = max(C.MIN_GREEN, min(C.MAX_GREEN, C.MIN_GREEN + C.PER_VEHICLE * counts.get(nxt, 0)))
        self.green, self.phase, self.phase_end, self.reason, self.green_duration = nxt, "GREEN", t + dur, why, dur
        try:
            db.x("INSERT INTO signal_log(ts,lane,state,duration,reason) VALUES(?,?,?,?,?)",
                 (db.now(), nxt, "green", dur, why))
        except Exception:
            pass

class TrafficEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.running, self.thread = False, None
        self.paused, self.auto_paused = False, False
        self.signal = SignalController()
        self.stab = CameraStabilizer()
        self.lane_names = list(LANE_ORDER)
        self.raw, self.raw_M, self._reload = None, np.eye(3), False
        self.cfg = None
        self.source_name = ""
        self._reader = None
        self._emerg_clf = None        # lazy: None=not tried, False=tried/no weights file, else EmergencyClassifier
        self.jpg = self._placeholder("No source running - open Source page")
        self.metrics = {"status": "idle"}
        self._reset_state()

    # ---------- config ----------
    def load_cfg(self, w, h):
        raw = json.load(open(C.LANES_FILE))
        sc = np.array([w, h], dtype=np.float64)
        lanes = {}
        for name, L in raw["lanes"].items():
            d = None                                      # direction is OPTIONAL: learned from traffic if not given
            if L.get("direction"):
                d = np.array(L["direction"], dtype=np.float64)
                d /= (np.linalg.norm(d) or 1)
            poly = (np.array(L["polygon"]) * sc).astype(np.int32)
            e = {"poly": poly, "dir": d, "n": max(1, L.get("num_lanes", 1)), "learn": np.zeros(2), "learn_n": 0}
            if L.get("stop_line"):
                p1, p2 = np.array(L["stop_line"][0]) * sc, np.array(L["stop_line"][1]) * sc
                if d is not None:
                    past = math.copysign(1, _cross(p1, p2, (p1 + p2) / 2 + d * 40)) or 1
                else:                                     # "past the line" = the side away from the lane area
                    past = -(math.copysign(1, _cross(p1, p2, poly.mean(0))) or 1)
                e.update(p1=p1, p2=p2, len=float(np.linalg.norm(p2 - p1)) or 1.0, past=past)
            lanes[name] = e
        self.lane_names = list(lanes)
        self.signal.set_lanes(self.lane_names)
        pp = raw.get("perspective")           # ground scale: metres/pixel = k / (y - y0)
        nv = np.array(raw.get("north_direction", [0, -1]), dtype=np.float64)     # image direction that is "North" (default: up)
        nv /= (np.linalg.norm(nv) or 1)
        self.cfg = {"lanes": lanes, "north": nv, "mpp": raw.get("meters_per_pixel", 0.05),
                    "persp": (pp["k"] * w / 960.0, pp["y0"] * h) if pp else None,
                    "signalized": raw.get("signalized", True), "stabilize": raw.get("stabilize", True),
                    "speed_limit": raw.get("speed_limit_kmh", C.SPEED_LIMIT_KMH),
                    "nopark": [(np.array(z) * sc).astype(np.int32) for z in raw.get("no_parking_zones", [])],
                    "restricted": [(np.array(z) * sc).astype(np.int32) for z in raw.get("restricted_zones", [])]}

    def _compass(self, dx, dy):
        """Image-space motion (dx, dy) -> North / South / East / West (top of image is North unless `north_direction` is set)."""
        n = self.cfg["north"]
        east = np.array([-n[1], n[0]])
        a, b = dx * n[0] + dy * n[1], dx * east[0] + dy * east[1]
        if abs(a) >= abs(b):
            return "North" if a > 0 else "South"
        return "East" if b > 0 else "West"

    def _meters(self, x0, y0, x1, y1):
        p = self.cfg["persp"]
        if p:
            k, yh = p
            a, b = max(y0 - yh, 30.0), max(y1 - yh, 30.0)
            return math.hypot(k * math.log(b / a), (x1 - x0) * k / ((a + b) / 2))
        return math.hypot(x1 - x0, y1 - y0) * self.cfg["mpp"]

    def _reset_state(self):
        self.tracks = {}
        self.uniq = defaultdict(set)          # (lane, class) -> ids
        self.totals = defaultdict(int)        # violation totals
        self.recent = deque(maxlen=15)
        self.emerg_logged = set()
        self.siren_logged = False
        self.last_log_t = -999
        self.signal.reset()
        self.frame_idx = 0

    # ---------- helpers ----------
    def _placeholder(self, text):
        img = np.zeros((540, 960, 3), np.uint8) + 25
        cv2.putText(img, text, (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        return cv2.imencode(".jpg", img)[1].tobytes()

    def get_frame(self):
        with self.lock:
            return self.jpg

    def snapshot(self):
        """(clean frame, current->reference transform) of the latest processed frame, or None."""
        with self.lock:
            return None if self.raw is None else (self.raw.copy(), self.raw_M.copy())

    def reload_cfg(self):
        """Apply newly saved zones to the running engine (no restart needed)."""
        self._reload = True

    def _apply_reload(self, w, h):
        self._reload = False
        self.load_cfg(w, h)
        for tr in self.tracks.values():
            tr["home"] = tr["lane"] = tr["before"] = None
            tr["votes"].clear()
        self.uniq.clear()

    def get_metrics(self):
        with self.lock:
            return dict(self.metrics)

    def _lane_of(self, pt):
        for name, L in self.cfg["lanes"].items():
            if cv2.pointPolygonTest(L["poly"], (float(pt[0]), float(pt[1])), False) >= 0:
                return name
        return None

    def _in_any(self, zones, pt):
        return any(cv2.pointPolygonTest(z, (float(pt[0]), float(pt[1])), False) >= 0 for z in zones)

    def _budget_slots(self, dets, max_per_frame, min_box_h, max_tries, every_n, tries_key):
        """Track ids allowed to run an expensive check this frame: eligible boxes, biggest first,
        capped at max_per_frame. Keeps a crowded frame from stalling the video on CPU."""
        if max_per_frame <= 0 or self.frame_idx % every_n:
            return set()
        cand = []
        for d in dets:
            if d["cls"] not in ("car", "bus", "truck"):
                continue
            tr = self.tracks.get(d["tid"])
            if tr and (tr.get("emerg") or tr.get(tries_key, 0) >= max_tries):
                continue
            h = d["box"][3] - d["box"][1]
            if h > min_box_h:
                cand.append((h, d["tid"]))
        cand.sort(reverse=True)
        return {tid for _, tid in cand[:max_per_frame]}

    def _clip_emergency_type(self, frame, box):
        """Zero-shot CLIP on the crop -> 'ambulance' / 'fire_engine' / None. See emergency_vision.py.
        Returns None (and costs nothing after the first call) if no CLIP backend is installed."""
        if EV is None:
            return None
        try:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            bw, bh = x2 - x1, y2 - y1
            x1, y1 = max(0, x1 - bw // 12), max(0, y1 - bh // 12)      # pad: roof light bars matter
            x2, y2 = min(frame.shape[1], x2 + bw // 12), min(frame.shape[0], y2 + bh // 12)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            hit = EV.classify_crop(crop)
            return hit[0] if hit else None
        except Exception:
            return None

    def _visual_emergency_type(self, frame, box):
        """Trained classifier (models/emergency_cnn.pt, see train_emergency.py) on the crop -> 'ambulance' /
        'fire_engine' / None. Silently stays off (always None) until that weights file exists, so the app
        runs fine without it — OCR (below) is the fallback either way."""
        if EmergencyClassifier is None:
            return None
        if self._emerg_clf is False:
            return None
        if self._emerg_clf is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "emergency_cnn.pt")
            if not os.path.exists(path):
                self._emerg_clf = False
                return None
            try:
                self._emerg_clf = EmergencyClassifier(path, min_conf=C.EMERG_CLF_MIN_CONF)
            except Exception:
                self._emerg_clf = False
                return None
        try:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            return self._emerg_clf.predict(crop)
        except Exception:
            return None

    def _ocr_emergency_type(self, frame, box):
        """OCR the crop for emergency-vehicle markings (C.OCR_KEYWORDS). Returns 'ambulance' / 'fire_engine' / None."""
        try:
            if self._reader is None:
                import easyocr
                self._reader = easyocr.Reader(["en"], gpu=False)
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            # small/far boxes have tiny text EasyOCR otherwise misses — upscale so the shortest side is >= 200px
            h, w = crop.shape[:2]
            scale = max(1.0, 200.0 / max(1, min(h, w)))
            if scale > 1.0:
                crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)   # boost contrast for faded/sun-washed text
            variants = [crop, cv2.flip(crop, 1), gray]
            texts = []
            for v in variants:
                texts += self._reader.readtext(v, detail=0)
            txt = " ".join(texts).upper()
            for etype, kws in C.OCR_KEYWORDS.items():
                if any(kw in txt for kw in kws):
                    return etype
            return None
        except Exception:
            return None

    def _violate(self, frame, tid, tr, vtype, t, speed=None):
        if vtype in tr["flags"]:
            return
        tr["flags"].add(vtype)
        img = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in tr["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"{vtype} ID{tid}", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, time.strftime("%Y-%m-%d %H:%M:%S"), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        name = f"{vtype.replace(' ', '_')}_{tid}_{int(time.time() * 1000)}.jpg"
        os.makedirs(C.EVIDENCE_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(C.EVIDENCE_DIR, name), img)
        lane = tr.get("home")
        db.x("INSERT INTO violations(vtype,track_id,vehicle_class,lane,speed_kmh,fine,evidence,ts,source) VALUES(?,?,?,?,?,?,?,?,?)",
             (vtype, tid, tr["cls"], lane, speed, C.FINES.get(vtype, 0), "evidence/" + name, db.now(), self.source_name))
        self.totals[vtype] += 1
        for adm in db.q("SELECT id FROM users WHERE role='admin'"):          # violation alert (fine / notification)
            db.x("INSERT INTO notifications(user_id,msg,created_at) VALUES(?,?,?)",
                 (adm["id"], f"{vtype} in {lane or '-'} by {tr['cls']} #{tid} - fine INR {C.FINES.get(vtype, 0)}", db.now()))
        self.recent.appendleft({"type": vtype, "lane": lane, "cls": tr["cls"], "id": tid,
                                "fine": C.FINES.get(vtype, 0), "evidence": "evidence/" + name,
                                "time": time.strftime("%H:%M:%S")})
        tr["flash"] = (vtype, t + 3)

    # ---------- core rules ----------
    def process(self, frame, dets, t, siren=False, M=None):
        """dets: [{'tid':int,'cls':str,'box':(x1,y1,x2,y2),'emerg':bool?}]. Draws on `frame`. Returns metrics dict."""
        cfg, lanes = self.cfg, self.cfg["lanes"]
        M = np.eye(3) if M is None else M           # current-frame -> reference-frame (camera motion compensation)
        lane_ids = defaultdict(list)
        # CPU budget: CLIP is ~60-120 ms/crop and EasyOCR ~300-800 ms/crop. With 6-10 vehicles on
        # screen, running them on every eligible box would stall the video far below real time.
        # Cap how many crops each may examine per frame, largest box first (nearest = most legible).
        clip_slots = self._budget_slots(dets, C.CLIP_MAX_PER_FRAME, C.CLIP_MIN_BOX_H,
                                        C.CLIP_MAX_TRIES, C.CLIP_EVERY_N, "clip_tries") \
            if (EV is not None and C.ENABLE_CLIP) else set()
        ocr_slots = self._budget_slots(dets, C.OCR_MAX_PER_FRAME, C.OCR_MIN_BOX_H,
                                       C.OCR_MAX_TRIES, C.OCR_EVERY_N, "ocr_tries") if C.ENABLE_OCR else set()
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            gp = apply(M, ((x1 + x2) / 2, y2))                # ground point in reference (zone) coordinates
            tr = self.tracks.setdefault(d["tid"], {"hist": deque(maxlen=400), "flags": set(), "home": None,
                                                   "before": None, "stat_since": None, "over_since": None, "ww_since": None, "votes": deque(maxlen=9), "logged": False, "dir_counted": False, "heading": None, "compass": None,
                                                   "emerg": False, "emerg_type": None, "ocr_tries": 0, "clf_tries": 0, "clip_tries": 0, "speed": None})
            tr.update(cls=d["cls"], box=d["box"], gp=gp, seen=t)
            tr["hist"].append((t, gp[0], gp[1]))
            if d.get("emerg_type") in EMERG_TYPES:                           # explicit type passed in (e.g. from a test)
                tr["emerg"], tr["emerg_type"] = True, d["emerg_type"]
            elif d.get("emerg") or d["cls"] in EMERG_TYPES:                  # dedicated YOLO class fired
                tr["emerg"], tr["emerg_type"] = True, d["cls"] if d["cls"] in EMERG_TYPES else "ambulance"
            if not tr["emerg"] and d["cls"] in ("car", "bus", "truck"):
                # zero-shot CLIP first: works with stock yolov8n.pt and needs no training or legible text
                if d["tid"] in clip_slots:
                    tr["clip_tries"] += 1
                    hit = self._clip_emergency_type(frame, d["box"])
                    if hit:
                        tr["emerg"], tr["emerg_type"] = True, hit
                if not tr["emerg"] and tr["clf_tries"] < C.EMERG_CLF_MAX_TRIES and (y2 - y1) > C.EMERG_CLF_MIN_BOX_H \
                        and self.frame_idx % C.EMERG_CLF_EVERY_N == 0:
                    tr["clf_tries"] += 1
                    hit = self._visual_emergency_type(frame, d["box"])       # trained classifier, 'ambulance'/'fire_engine'/None
                    if hit:
                        tr["emerg"], tr["emerg_type"] = True, hit
                if not tr["emerg"] and d["tid"] in ocr_slots:
                    tr["ocr_tries"] += 1
                    hit = self._ocr_emergency_type(frame, d["box"])          # text-reading fallback, 'ambulance'/'fire_engine'/None
                    if hit:
                        tr["emerg"], tr["emerg_type"] = True, hit
            if tr["emerg"]:
                tr["cls"] = tr["emerg_type"] or "ambulance"
            hspan, hx, hy = _window(tr["hist"], 1.0)                        # direction of travel (reference coordinates)
            if hspan >= 0.5 and math.hypot(hx, hy) >= C.HEADING_MIN_PX:
                tr["heading"], tr["compass"] = (hx, hy), self._compass(hx, hy)
            tr["votes"].append(self._lane_of(gp))
            top = Counter(v for v in tr["votes"] if v).most_common(1)      # majority vote: no flicker at lane lines
            tr["lane"] = top[0][0] if top and top[0][1] >= 3 else None
            if tr["lane"] and not tr["logged"]:                              # detection log: one row per vehicle
                tr["logged"] = True
                db.x("INSERT INTO detection_logs(ts,track_id,vclass,lane,source) VALUES(?,?,?,?,?)",
                     (db.now(), d["tid"], tr["cls"], tr["lane"], self.source_name))
            if tr["lane"]:
                tr["home"] = tr["lane"]
                lane_ids[tr["lane"]].append(d["tid"])
                self.uniq[(tr["lane"], tr["cls"])].add(d["tid"])
        active = {d["tid"] for d in dets}

        density = {l: round(len(lane_ids[l]) / lanes[l]["n"], 1) for l in self.lane_names}
        counts = {l: len(lane_ids[l]) for l in self.lane_names}
        emerg_lane = next((tr["home"] for tid, tr in self.tracks.items()
                           if tid in active and tr["emerg"] and tr["home"]), None)
        for tid, tr in self.tracks.items():
            if tid in active and tr["emerg"] and tid not in self.emerg_logged:
                self.emerg_logged.add(tid)
                etype = tr["emerg_type"] or "ambulance"
                db.x("INSERT INTO emergency_events(ts,kind,lane,note) VALUES(?,?,?,?)",
                     (db.now(), f"visual-{etype}", tr["home"], f"{etype.replace('_', ' ').title()} track {tid}"))
        if siren and not self.siren_logged:
            self.siren_logged = True
            db.x("INSERT INTO emergency_events(ts,kind,lane,note) VALUES(?,?,?,?)", (db.now(), "siren", emerg_lane, "Siren detected in audio"))
        if not siren:
            self.siren_logged = False

        if cfg["signalized"]:
            self.signal.update(t, counts, emerg_lane)
            sig = self.signal.states()
            sig_reason, sig_left = self.signal.reason, round(self.signal.remaining(t), 1)
        else:                                   # road without a signal: monitoring only
            sig = {l: "green" for l in self.lane_names}
            sig_reason, sig_left = "No traffic signal at this location (monitoring only)", 0

        for tid in active:
            tr = self.tracks[tid]
            hist, gp, home, lane = tr["hist"], tr["gp"], tr["home"], tr["lane"]
            if tr["emerg"]:
                continue                                          # emergency vehicles are exempt
            # red light / stop line
            if home and cfg["signalized"] and "p1" in lanes[home]:
                L = lanes[home]
                s = _cross(L["p1"], L["p2"], gp) * L["past"]
                before = bool(s <= 0)
                span, dx, dy = _window(hist, 1.5)
                still = span >= 1.5 and math.hypot(dx, dy) < 4
                if tr["before"] is True and not before and sig[home] == "red":
                    self._violate(frame, tid, tr, "Red Light Jump", t)
                elif sig[home] == "red" and 0 < s / L["len"] <= C.STOP_MARGIN_PX and still \
                        and "Red Light Jump" not in tr["flags"]:
                    self._violate(frame, tid, tr, "Stop Line Violation", t)
                tr["before"] = before
            # wrong way + speed (only inside a lane polygon)
            if lane:
                L = lanes[lane]
                span, dx, dy = _window(hist, C.WW_WINDOW)
                dist = math.hypot(dx, dy)
                if L["dir"] is None:                                     # learn the normal flow of this lane (majority)
                    if not tr["dir_counted"] and span >= C.WW_SPAN and dist >= C.WW_DIST:
                        tr["dir_counted"] = True
                        L["learn"] += np.array([dx, dy]) / dist
                        L["learn_n"] += 1
                        if L["learn_n"] >= C.DIR_LEARN_MIN:
                            m = L["learn"] / L["learn_n"]
                            if np.linalg.norm(m) >= 0.6:                 # clearly one-way flow -> lock it in
                                L["dir"] = m / np.linalg.norm(m)
                elif span >= C.WW_SPAN and dist >= C.WW_DIST and (dx * L["dir"][0] + dy * L["dir"][1]) / dist < C.WW_COS:
                    tr["ww_since"] = t if tr["ww_since"] is None else tr["ww_since"]
                    if t - tr["ww_since"] >= C.WW_SUSTAIN:                       # must persist, not a one-frame glitch
                        self._violate(frame, tid, tr, "Wrong Way", t)
                else:
                    tr["ww_since"] = None
                span1, dx1, dy1 = _window(hist, 1.0)
                if span1 >= 0.8:
                    kmh = self._meters(gp[0] - dx1, gp[1] - dy1, gp[0], gp[1]) / span1 * 3.6
                    tr["speed"] = kmh
                    if cfg["speed_limit"] < kmh < 220:
                        tr["over_since"] = tr["over_since"] or t
                        if t - tr["over_since"] >= 1.0:
                            self._violate(frame, tid, tr, "Speeding", t, round(kmh, 1))
                    else:
                        tr["over_since"] = None
            # stationary tracking + illegal parking
            span, dx, dy = _window(hist, 1.5)
            if span >= 1.2 and math.hypot(dx, dy) < 6:
                tr["stat_since"] = tr["stat_since"] if tr["stat_since"] is not None else t
            else:
                tr["stat_since"] = None
            if tr["stat_since"] is not None and t - tr["stat_since"] >= C.PARK_SECONDS and self._in_any(cfg["nopark"], gp):
                self._violate(frame, tid, tr, "Illegal Parking", t)
            # restricted-lane / divider entry
            if self._in_any(cfg["restricted"], gp) and math.hypot(dx, dy) > 8:
                self._violate(frame, tid, tr, "Lane Violation", t)

        # periodic DB logging
        if t - self.last_log_t >= 5:
            self.last_log_t = t
            ts = db.now()
            dv = [density.get(n, 0) for n in self.lane_names[:4]] + [0] * 4      # columns are positional (lane 1..4)
            db.x("INSERT INTO density_log(ts,north,east,south,west) VALUES(?,?,?,?,?)", (ts, dv[0], dv[1], dv[2], dv[3]))
            for (ln, cl), ids in self.uniq.items():
                db.x("INSERT INTO vehicle_counts(ts,lane,vclass,count) VALUES(?,?,?,?)", (ts, ln, cl, len(ids)))

        dcount = Counter(self.tracks[d["tid"]]["compass"] for d in dets if self.tracks[d["tid"]]["compass"])
        lane_compass = {n: (self._compass(*L["dir"]) if L["dir"] is not None else None) for n, L in lanes.items()}
        ddens = {}
        for dname in ("North", "South", "East", "West"):                       # density = vehicles / lane (lanes carrying that direction)
            nl = sum(L["n"] for n, L in lanes.items() if lane_compass[n] == dname) or 1
            ddens[dname] = round(dcount.get(dname, 0) / nl, 1)
        dcount = {k: dcount.get(k, 0) for k in ("North", "South", "East", "West")}
        self._draw(frame, dets, density, sig, t, siren, emerg_lane, M, dcount, lane_compass)
        cls_now = defaultdict(int)
        for d in dets:
            cls_now[self.tracks[d["tid"]]["cls"]] += 1
        lane_counts = defaultdict(dict)
        for (ln, cl), ids in self.uniq.items():
            lane_counts[ln][cl] = len(ids)
        return {"density": density, "counts": counts, "lane_counts": dict(lane_counts),
                "direction_counts": dcount, "direction_density": ddens, "lane_compass": lane_compass, "classes": dict(cls_now), "signal": sig,
                "signal_remaining": sig_left, "signal_reason": sig_reason,
                "lane_dirs": {n: ([round(float(v), 2) for v in L["dir"]] if L["dir"] is not None else None) for n, L in lanes.items()},
                "emergency_lane": emerg_lane, "siren": siren, "siren_detector": audiomod.detector_name(), "totals": dict(self.totals),
                "recent": list(self.recent), "video_time": round(t, 1)}

    def _draw(self, frame, dets, density, sig, t, siren, emerg_lane, M, dcount, lane_compass):
        if not np.all(np.isfinite(M)) or abs(np.linalg.det(M)) < 1e-6:
            M = np.eye(3)                                          # bad/singular stabilizer transform: fall back to identity
        Minv = np.linalg.inv(M)
        lin = Minv[:2, :2]
        for name, L in self.cfg["lanes"].items():
            poly = apply_poly(Minv, L["poly"])
            cv2.polylines(frame, [poly], True, (255, 200, 0), 1)
            if "p1" in L:
                a, b = apply(Minv, L["p1"]), apply(Minv, L["p2"])
                cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), SIG_COLOR[sig[name]], 4)
            c = poly.mean(0).astype(int)
            sfx = f" ({lane_compass[name][0]})" if lane_compass.get(name) and name.lower() not in ("north", "south", "east", "west") else ""
            lab = f"{name}{sfx}: {density.get(name, 0)}"
            cv2.putText(frame, lab, (int(c[0]) - 40, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if L["dir"] is not None:                                               # lane's normal flow arrow
                v = lin @ L["dir"]; v = v / (np.linalg.norm(v) or 1)
                cv2.arrowedLine(frame, (int(c[0]), int(c[1]) + 14), (int(c[0] + v[0] * 45), int(c[1] + 14 + v[1] * 45)), (255, 255, 0), 2, tipLength=0.35)
        for z in self.cfg["nopark"]:
            cv2.polylines(frame, [apply_poly(Minv, z)], True, (0, 165, 255), 2)
        for z in self.cfg["restricted"]:
            cv2.polylines(frame, [apply_poly(Minv, z)], True, (0, 0, 200), 2)
        for d in dets:
            tr = self.tracks[d["tid"]]
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            col = COLORS.get(tr["cls"], (200, 200, 200))
            if tr.get("flash") and tr["flash"][1] > t:
                col = (0, 0, 255)
                cv2.putText(frame, tr["flash"][0], (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            sp = f" {tr['speed']:.0f}km/h" if tr.get("speed") else ""
            if tr.get("heading") is not None:                                      # arrow = this vehicle's direction of travel
                v = lin @ np.array(tr["heading"]); v = v / (np.linalg.norm(v) or 1)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.arrowedLine(frame, (cx, cy), (int(cx + v[0] * 32), int(cy + v[1] * 32)), DIR_COLOR[tr["compass"]], 3, tipLength=0.4)
            cv2.putText(frame, f"{tr['cls']} #{d['tid']} {tr['lane'] or ''}{sp}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        self._hud(frame, dcount, lin)
        if emerg_lane or siren:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)
            cv2.putText(frame, "EMERGENCY" + (f" - {emerg_lane}" if emerg_lane else " (siren)"), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    def _hud(self, frame, dcount, lin):
        """Top-left panel like the diagram's North / South / East / West density box."""
        x0, y0, w, h = 8, 8, 175, 104
        ov = frame.copy()
        cv2.rectangle(ov, (x0, y0), (x0 + w, y0 + h), (25, 25, 25), -1)
        frame[:] = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "DIRECTION (vehicles)", (x0 + 6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        n = self.cfg["north"]; east = np.array([-n[1], n[0]])
        vec = {"North": n, "South": -n, "East": east, "West": -east}
        for i, dname in enumerate(("North", "South", "East", "West")):
            y = y0 + 36 + i * 22
            v = lin @ vec[dname]; v = v / (np.linalg.norm(v) or 1)
            cv2.arrowedLine(frame, (x0 + 22, y - 5), (int(x0 + 22 + v[0] * 12), int(y - 5 + v[1] * 12)), DIR_COLOR[dname], 3, tipLength=0.5)
            cv2.putText(frame, f"{dname:<6}{dcount.get(dname, 0):>3}", (x0 + 44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DIR_COLOR[dname], 2)

    # ---------- runtime ----------
    def start(self, source, loop=False, name="", mic=False):
        self.stop()
        self.mic = mic
        self.source_name = name or str(source)
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(source, loop), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.paused = self.auto_paused = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def pause(self, auto=False):
        """Freeze playback in place (video position/tracks kept) so `resume()` continues from here.
        `auto=True` marks a pause triggered by another lane's emergency-vehicle detection, so the
        dashboard can show why and the admin can resume it manually once it's safe."""
        if self.running:
            self.paused, self.auto_paused = True, auto
            self._set(status="paused", auto_paused=auto)

    def resume(self):
        if self.running and self.paused:
            self.paused = self.auto_paused = False
            self._set(status="running", auto_paused=False)

    def _set(self, **kw):
        with self.lock:
            self.metrics.update(kw)

    def _run(self, source, loop):
        try:
            self._set(status="loading model")
            from ultralytics import YOLO
            model = YOLO(C.YOLO_MODEL)
            ids = [i for i, n in model.names.items() if n.lower() in NAME_MAP]
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open source: {source}")
            is_file = isinstance(source, str) and os.path.exists(source)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            siren_iv, monitor = [], None
            if not is_file and getattr(self, "mic", False):
                monitor = audiomod.LiveSirenMonitor()
                monitor.start()
            if is_file:
                self._set(status="analysing audio")
                siren_iv, err = audiomod.analyse_file(source)
            self._reset_state()
            self.stab.reset()
            self.cfg = None
            self._reload = False
            start_wall, next_due, n_proc = time.time(), time.time(), 0
            while self.running:
                if self.paused:
                    # Don't grab/decode while paused, so the video stays at this exact frame and
                    # `resume()` continues forward rather than restarting or skipping ahead.
                    time.sleep(0.2)
                    start_wall += 0.2         # keep the real-time pacing clock from "catching up" during the pause
                    next_due += 0.2
                    continue
                ok = cap.grab()
                self.frame_idx += 1
                if not ok:
                    if loop and is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._reset_state()
                        self.stab.reset()
                        start_wall = next_due = time.time()
                        self.frame_idx, n_proc = 0, 0
                        continue
                    break
                if is_file:
                    # Catch-up: with several lanes running at once the CPU falls behind the
                    # video's own clock. Rather than decoding + running full YOLO/CLIP/OCR on
                    # every frame regardless (which is what silently turned a 40s clip into a
                    # 5-minute run), skip frames outright (grab only, no decode) until back on
                    # schedule, so total runtime tracks the video's actual length.
                    behind = (time.time() - start_wall) - (self.frame_idx / fps) / C.PLAYBACK_SPEED
                    if behind > (C.PROCESS_EVERY / fps) / C.PLAYBACK_SPEED:
                        continue
                    if C.PROCESS_EVERY > 1 and self.frame_idx % C.PROCESS_EVERY:
                        continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                h0, w0 = frame.shape[:2]
                frame = cv2.resize(frame, (C.WORK_WIDTH, int(h0 * C.WORK_WIDTH / w0)))
                if self.cfg is None:
                    self.load_cfg(frame.shape[1], frame.shape[0])
                t = (self.frame_idx / fps) if is_file else (time.time() - start_wall)
                res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=ids,
                                  conf=C.CONF, verbose=False)[0]
                dets = []
                if res.boxes is not None and res.boxes.id is not None:
                    for b, tid, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist(),
                                         res.boxes.cls.int().cpu().tolist()):
                        dets.append({"tid": int(tid), "box": tuple(float(v) for v in b),
                                     "cls": NAME_MAP[model.names[c].lower()]})
                siren = monitor.active if monitor else any(a <= t <= b for a, b in siren_iv)
                M = self.stab.update(frame) if self.cfg["stabilize"] else None
                if self._reload:
                    self._apply_reload(frame.shape[1], frame.shape[0])
                clean = frame.copy()
                m = self.process(frame, dets, t, siren, M)
                n_proc += 1
                m.update(status="running", source=self.source_name, fps=round(n_proc / max(1e-6, time.time() - start_wall), 1),
                         progress=round(100 * self.frame_idx / total, 1) if (is_file and total) else None)
                jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                with self.lock:
                    # pause() may have fired while this frame was mid-processing (YOLO takes a
                    # while); self.paused is the source of truth, so re-check it here rather than
                    # blindly publishing "running" and clobbering the "paused" pause() just wrote.
                    if self.paused:
                        m.update(status="paused", auto_paused=self.auto_paused)
                    self.jpg, self.metrics = jpg, m
                    self.raw, self.raw_M = clean, (np.eye(3) if M is None else M)
                if is_file:                                       # pace to C.PLAYBACK_SPEED x real-time
                    next_due += (C.PROCESS_EVERY / fps) / C.PLAYBACK_SPEED
                    time.sleep(max(0, next_due - time.time()))
            cap.release()
            if monitor:
                monitor.stop()
            self._set(status="finished" if self.running else "stopped")
        except Exception as e:
            traceback.print_exc()
            self._set(status=f"error: {e}")
        finally:
            self.running = False

engine = TrafficEngine()
