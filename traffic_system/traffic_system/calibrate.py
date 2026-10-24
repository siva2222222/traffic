"""Draw your intersection geometry on a video frame -> writes zones/lanes.json (normalized).
Usage: python calibrate.py your_video.mp4
Mouse: left-click adds a point. Keys: ENTER = finish item, BACKSPACE = undo point, S = skip optional item, Q = quit.
Order: for each lane -> lane polygon, stop line (2 clicks, S to skip). Direction is NOT needed (learned from traffic).
Then: no-parking zone (optional), restricted zone (optional), scale (2 clicks + real distance in metres in the terminal)."""
import sys, json, math, cv2, numpy as np
from config import C

W = C.WORK_WIDTH
cap = cv2.VideoCapture(sys.argv[1]) if len(sys.argv) > 1 else None
if cap is None or not cap.isOpened():
    sys.exit("Usage: python calibrate.py video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # zones are drawn on the FIRST frame (engine reference)
ok, frame = cap.read()
if not ok:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, frame = cap.read()
h0, w0 = frame.shape[:2]
frame = cv2.resize(frame, (W, int(h0 * W / w0))); H = frame.shape[0]
pts, done_items = [], []

def on_mouse(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
cv2.namedWindow("calibrate"); cv2.setMouseCallback("calibrate", on_mouse)

def collect(prompt, need=None, optional=False):
    global pts
    pts = []
    while True:
        img = frame.copy()
        for d in done_items:
            cv2.polylines(img, [np.array(d, np.int32)], False, (0, 255, 0), 2)
        for p in pts:
            cv2.circle(img, p, 4, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 200, 255), 2)
        cv2.putText(img, prompt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("calibrate", img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"): sys.exit("aborted")
        if k == 8 and pts: pts.pop()
        if optional and k == ord("s"): return None
        if need and len(pts) == need: break
        if not need and k == 13 and len(pts) >= 3: break
    done_items.append(list(pts)); return list(pts)

norm = lambda p: [round(p[0] / W, 4), round(p[1] / H, 4)]
out = {"meters_per_pixel": 0.05, "lanes": {}, "no_parking_zones": [], "restricted_zones": []}
for lane in ["North", "East", "South", "West"]:
    poly = collect(f"{lane}: click lane polygon, ENTER when done")
    stop = collect(f"{lane}: click 2 points of the STOP LINE (S = no signal / skip)", 2, optional=True)
    nl = input(f"Number of lanes on {lane} approach [2]: ").strip() or "2"
    out["lanes"][lane] = {"polygon": [norm(p) for p in poly], "num_lanes": int(nl)}
    if stop:
        out["lanes"][lane]["stop_line"] = [norm(p) for p in stop]
    else:
        out["signalized"] = False
z = collect("No-parking zone polygon (ENTER) or S to skip", optional=True)
if z: out["no_parking_zones"].append([norm(p) for p in z])
z = collect("Restricted zone / divider polygon (ENTER) or S to skip", optional=True)
if z: out["restricted_zones"].append([norm(p) for p in z])
s = collect("SCALE: click 2 points a known distance apart (e.g. lane width, 3.5 m)", 2)
m = float(input("Real distance between those points in metres: "))
out["meters_per_pixel"] = round(m / (math.dist(s[0], s[1]) or 1), 5)
json.dump(out, open(C.LANES_FILE, "w"), indent=2)
print("Saved", C.LANES_FILE, "- meters_per_pixel =", out["meters_per_pixel"])
