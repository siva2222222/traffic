"""Synthetic test of all violation rules (no YOLO needed). Run: python tests/test_engine.py"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import C
C.DB_PATH = tempfile.mktemp(suffix=".db"); C.EVIDENCE_DIR = tempfile.mkdtemp(); C.LANES_FILE = os.path.join(os.path.dirname(C.LANES_FILE), "lanes_intersection_example.json")
import numpy as np, database as db
db.init_db()
from engine import TrafficEngine, LANE_ORDER
import engine as E

e = TrafficEngine(); e.load_cfg(960, 540)
frame = np.zeros((540, 960, 3), np.uint8)
FPS = 25

def run(tracks, seconds, force_signal=None, siren=False):
    """tracks: {tid: fn(t)->(cx, ground_y, cls)}"""
    e._reset_state(); e.source_name = "test"
    for f in range(int(seconds * FPS)):
        t = f / FPS; e.frame_idx = f
        dets = []
        for tid, fn in tracks.items():
            cx, gy, cls = fn(t)
            dets.append({"tid": tid, "cls": cls, "box": (cx - 15, gy - 30, cx + 15, gy)})
        if force_signal:
            e.signal.green, e.signal.phase, e.signal.phase_end = force_signal, "GREEN", 1e9
            e.signal.last_t = t
        e.process(frame.copy(), dets, t, siren)
    return e.totals

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); 
    if not cond: sys.exit(1)

# North lane: x 0.36-0.50 (345-480px), moving down (+y), stop line y=216
# 1) Red-light jump: East is green, North is red; vehicle crosses North stop line at ~120 px/s
tot = run({1: lambda t: (410, 100 + 60 * t, "car")}, 4, force_signal="East")
check("red light jump detected", tot.get("Red Light Jump", 0) == 1)

# 2) Same crossing while North is green -> no violation
tot = run({1: lambda t: (410, 100 + 60 * t, "car")}, 4, force_signal="North")
check("no red-light on green", tot.get("Red Light Jump", 0) == 0)

# 3) Wrong way: North lane, moving UP
tot = run({2: lambda t: (410, 200 - 40 * t, "car")}, 3, force_signal="North")
check("wrong way detected", tot.get("Wrong Way", 0) == 1)

# 4) Speeding: set 0.1 m/px -> limit 40 km/h = 111 px/s. East lane, 150 px/s = 54 km/h
e.cfg["mpp"] = 0.1
tot = run({3: lambda t: (950 - 150 * t, 240, "car")}, 2.5, force_signal="East")
check("speeding detected", tot.get("Speeding", 0) == 1)

# 5) Legal speed: 80 px/s = 29 km/h
tot = run({3: lambda t: (950 - 80 * t, 240, "car")}, 4, force_signal="East")
check("no speeding at 29km/h", tot.get("Speeding", 0) == 0)

e.cfg["mpp"] = 0.05
# 6) Illegal parking: stationary 12 s inside no-parking zone (x 19-269, y 27-119)
tot = run({4: lambda t: (100, 90, "car")}, 13, force_signal="North")
check("illegal parking detected", tot.get("Illegal Parking", 0) == 1)

# 7) Stationary outside zone -> none
tot = run({4: lambda t: (700, 500, "car")}, 13, force_signal="North")
check("no parking flag outside zone", tot.get("Illegal Parking", 0) == 0)

# 8) Lane violation: moving through restricted zone (x 499-557, y 11-184)
tot = run({5: lambda t: (528, 20 + 40 * t, "bike")}, 3, force_signal="North")
check("lane violation detected", tot.get("Lane Violation", 0) == 1)

# 9) Stop-line violation: red, vehicle halted 30px past stop line
tot = run({6: lambda t: (410, 100 + 60 * t if t < 2.0 else 220, "car")} , 6, force_signal="East")
print("   totals:", dict(tot))

# 10) Ambulance exempt + priority
tot = run({7: lambda t: (410, 100 + 60 * t, "ambulance")}, 4, force_signal="East")
check("ambulance exempt from red light", tot.get("Red Light Jump", 0) == 0)

# 10a) Fire engine exempt + priority (same rule, different visual class)
tot = run({8: lambda t: (410, 100 + 60 * t, "fire_engine")}, 4, force_signal="East")
check("fire engine exempt from red light", tot.get("Red Light Jump", 0) == 0)

# 10b) Lane WITHOUT direction: learns flow from traffic, then flags a wrong-way vehicle
import json, copy
raw = json.load(open(C.LANES_FILE)); nd = copy.deepcopy(raw)
for L in nd["lanes"].values(): L.pop("direction", None)
_f = tempfile.mktemp(suffix=".json"); json.dump(nd, open(_f, "w")); _old = C.LANES_FILE; C.LANES_FILE = _f
e3 = TrafficEngine(); e3.load_cfg(960, 540); C.LANES_FILE = _old; e3.source_name = "learn"
e3.signal.green, e3.signal.phase, e3.signal.phase_end = "North", "GREEN", 1e9
def drive(eng, tracks, seconds):
    for f in range(int(seconds * FPS)):
        t = f / FPS; eng.frame_idx = f; dets = []
        for tid, fn in tracks.items():
            cx, gy = fn(t); dets.append({"tid": tid, "cls": "car", "box": (cx - 15, gy - 30, cx + 15, gy)})
        eng.signal.last_t = t; eng.process(frame.copy(), dets, t, False)
drive(e3, {30 + i: (lambda t, i=i: (380 + 8 * i, 20 + 45 * t)) for i in range(7)}, 3)     # 7 cars moving DOWN
check("lane direction learned automatically (down)", e3.cfg["lanes"]["North"]["dir"] is not None and e3.cfg["lanes"]["North"]["dir"][1] > 0.9)
drive(e3, {99: lambda t: (400, 200 - 40 * t)}, 3)                                            # one car moving UP
check("wrong way flagged against learned flow", e3.totals.get("Wrong Way", 0) == 1)
# red light without a direction: 'past the line' derived from the lane area
e4 = TrafficEngine(); C.LANES_FILE = _f; e4.load_cfg(960, 540); C.LANES_FILE = _old; e4.source_name = "nodir"
e4.signal.green, e4.signal.phase, e4.signal.phase_end = "East", "GREEN", 1e9
drive(e4, {5: lambda t: (410, 100 + 60 * t)}, 4)
check("red light works without direction", e4.totals.get("Red Light Jump", 0) == 1)

# 10c) Direction of travel: compass names, per-direction counts, custom North
check("compass: down=South up=North right=East left=West", [e._compass(0, 9), e._compass(0, -9), e._compass(9, 0), e._compass(-9, 0)] == ["South", "North", "East", "West"])
e.cfg["north"] = np.array([1.0, 0.0]); check("north can point right", e._compass(9, 0) == "North" and e._compass(0, 9) == "East"); e.cfg["north"] = np.array([0.0, -1.0])
e5 = TrafficEngine(); e5.load_cfg(960, 540); e5.source_name = "dirs"
e5.signal.green, e5.signal.phase, e5.signal.phase_end = "North", "GREEN", 1e9
seen = {}
for f in range(60):
    tt = f / FPS; e5.frame_idx = f; e5.signal.last_t = tt
    dets = [{"tid": 1, "cls": "car", "box": (365, 20 + 60 * tt - 30, 395, 20 + 60 * tt)},          # moving DOWN
            {"tid": 2, "cls": "car", "box": (900 - 90 * tt, 200, 930 - 90 * tt, 230)}]            # moving LEFT (west)
    seen = e5.process(frame.copy(), dets, tt, False)
check("direction counts: 1 South + 1 West", seen["direction_counts"] == {"North": 0, "South": 1, "East": 0, "West": 1})
check("direction density reported", set(seen["direction_density"]) == {"North", "South", "East", "West"})

# 10d) hot reload of zones on a running engine (rename lanes, no restart)
ren = json.load(open(C.LANES_FILE)); ren["lanes"] = {"Gate A": ren["lanes"]["North"]}
_g = tempfile.mktemp(suffix=".json"); json.dump(ren, open(_g, "w")); _keep = C.LANES_FILE; C.LANES_FILE = _g
e5._apply_reload(960, 540); C.LANES_FILE = _keep
e5.frame_idx += 1; m5 = e5.process(frame.copy(), [{"tid": 1, "cls": "car", "box": (365, 100, 395, 130)}], 3.0, False)
check("zones hot-reloaded (new lane names, tracks re-assigned)", e5.lane_names == ["Gate A"] and list(m5["density"]) == ["Gate A"])

# 11) Adaptive signal: heavy East traffic should win next green; emergency pre-empts
s = E.SignalController()
t = 0.0
s.update(t, {"North": 1, "East": 12, "South": 2, "West": 0})
check("adaptive picks busiest lane (East)", s.green == "East" and s.phase == "GREEN")
check("green time scales with queue", s.phase_end - t == min(C.MAX_GREEN, C.MIN_GREEN + C.PER_VEHICLE * 12))
t = 5.0
s.update(t, {"North": 1, "East": 12, "South": 2, "West": 0}, emerg_lane="South")
t = 5.1; s.update(t, {}, "South"); 
check("emergency cuts green to yellow", s.phase == "YELLOW")
t += C.YELLOW + 0.1; s.update(t, {}, "South"); t += C.ALL_RED + 0.1; s.update(t, {}, "South")
check("emergency lane gets green", s.green == "South" and s.phase == "GREEN")

ok_ = db.q("SELECT COUNT(*) n FROM detection_logs", one=True)["n"] > 0
check("detection logs written (one row per vehicle)", ok_)
check("violation alerts sent to admin", db.q("SELECT COUNT(*) n FROM notifications WHERE user_id=1 AND msg LIKE '%fine%'", one=True)["n"] > 0)
rows = db.q("SELECT vtype, COUNT(*) n FROM violations GROUP BY vtype")
print("DB violations:", rows)
print("ALL ENGINE TESTS PASSED")

# ---- optional: EasyOCR ambulance-text detection (runs only if easyocr is installed) ----
try:
    import easyocr, cv2
    C.ENABLE_OCR = True
    e2 = TrafficEngine(); e2.load_cfg(960, 540); e2.source_name = "ocr"
    frame = np.full((540, 960, 3), 120, np.uint8)
    frame[100:200, 300:470] = 245                                                     # white van body
    cv2.putText(frame, "AMBULANCE", (305, 160), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 200), 2)
    for fi in (15, 16, 17):                                                           # lane needs 3 frames of votes
        e2.frame_idx = fi
        m = e2.process(frame, [{"tid": 9, "cls": "car", "box": (300.0, 100.0, 470.0, 200.0)}], fi / 25, False)
    check("OCR reads AMBULANCE on vehicle -> emergency in its lane", m["emergency_lane"] == "North" and e2.tracks[9]["cls"] == "ambulance")

    frame2 = np.full((540, 960, 3), 120, np.uint8)
    frame2[100:200, 300:470] = 0                                                      # red fire-engine body
    cv2.putText(frame2, "FIRE ENGINE", (300, 160), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    for fi in (30, 31, 32):
        e2.frame_idx = fi
        m2 = e2.process(frame2, [{"tid": 10, "cls": "truck", "box": (300.0, 100.0, 470.0, 200.0)}], fi / 25, False)
    check("OCR reads FIRE ENGINE on vehicle -> emergency in its lane", m2["emergency_lane"] == "North" and e2.tracks[10]["cls"] == "fire_engine")
except ImportError:
    print("skip OCR test (easyocr not installed)")
