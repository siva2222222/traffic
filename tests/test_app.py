"""End-to-end test of portals via Flask test client. Run: python tests/test_app.py"""
import sys, os, tempfile, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import C
import shutil
C.DB_PATH = tempfile.mktemp(suffix=".db"); C.UPLOAD_DIR = tempfile.mkdtemp(); C.EVIDENCE_DIR = tempfile.mkdtemp()
_z = tempfile.mktemp(suffix=".json"); shutil.copy(C.LANES_FILE, _z); C.LANES_FILE = _z
import app as A, database as db
c = A.app.test_client()
def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: sys.exit(1)
def login(u, p): c.get("/logout"); return c.post("/login", data={"username": u, "password": p}, follow_redirects=True)

ok("public pages", c.get("/").status_code == 200 and c.get("/login").status_code == 200)
ok("admin blocked when anonymous", c.get("/admin").status_code == 302)
ok("register citizen", c.post("/register", data={"name": "Test", "username": "t1", "password": "secret1"}, follow_redirects=True).status_code == 200)
r = c.post("/report", data={"type": "Accident", "description": "Two vehicles collided, one injured", "location": "MG Road",
        "lat": "12.9750", "lng": "77.6060", "photo": (io.BytesIO(b"\xff\xd8\xff fake"), "a.jpg")}, content_type="multipart/form-data", follow_redirects=True)
ok("citizen report submitted", b"submitted" in r.data)
rep = db.q("SELECT * FROM reports", one=True); task = db.q("SELECT * FROM tasks", one=True)
ok("AI classified high priority", rep["priority"] == "High")
ok("auto-assigned to nearest officer", task is not None and task["officer_id"] == 2 and task["task_code"].startswith("TRF-"))
ok("my reports shows status", b"Assigned" in c.get("/my-reports").data)
ok("citizen cannot open admin", c.get("/admin").status_code == 403)
login("police1", "police123")
ok("police sees task", b"TRF-" in c.get("/police").data)
ok("police in-progress", c.post(f"/police/task/{task['id']}/update", data={"status": "In Progress", "remarks": "on the way"}, follow_redirects=True).status_code == 200)
r = c.post(f"/police/task/{task['id']}/update", data={"status": "Completed", "remarks": "cleared", "evidence": (io.BytesIO(b"x"), "e.jpg")}, content_type="multipart/form-data", follow_redirects=True)
ok("police completed with evidence", db.q("SELECT status,evidence FROM tasks", one=True)["status"] == "Completed" and db.q("SELECT evidence FROM tasks", one=True)["evidence"])
ok("police cannot update others' task", login("police2", "police123") and c.post(f"/police/task/{task['id']}/update", data={"status": "Completed"}).status_code == 404)
login("t1", "secret1")
ok("citizen notified", b"Completed" in c.get("/notifications").data)
login("admin", "admin123")
for p in ["/admin", "/admin/reports", "/admin/tasks", "/admin/violations", "/admin/analytics", "/admin/assistant", "/admin/source", "/api/live", "/api/analytics", "/api/summary"]:
    ok("admin page " + p, c.get(p).status_code == 200)
ok("task time-taken shown", b"min" in c.get("/admin/tasks").data)
db.x("INSERT INTO violations(vtype,track_id,vehicle_class,lane,fine,evidence,ts,source) VALUES('Red Light Jump',1,'car','North',1000,'evidence/x.jpg',?,'t')", (db.now(),))
ok("assistant answers", b"1" in c.post("/api/assistant", json={"q": "how many violations?"}).data)
for p in ["daily.csv", "daily.pdf", "weekly.pdf", "monthly.pdf"]:
    r = c.get("/admin/export/" + p); ok("export " + p, r.status_code == 200 and len(r.data) > 100)
ok("pdf valid", c.get("/admin/export/daily.pdf").data[:4] == b"%PDF")
ok("manual assign", c.post("/admin/reports/1/assign", data={"officer_id": "3"}, follow_redirects=True).status_code == 200)
r = c.get("/video_feed", buffered=False); chunk = next(iter(r.response)); ok("mjpeg stream yields jpeg", b"image/jpeg" in chunk); r.close()
# ---- new: data layer, zone editor, media log ----
import cv2, numpy as np, json
A.engine.start = lambda *a, **k: setattr(A.engine, "_started", (a, k))     # don't launch YOLO in the web test
vp = os.path.join(tempfile.mkdtemp(), "t.mp4")
w = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
for i in range(10): w.write(np.full((360, 640, 3), 90, np.uint8))
w.release()
r = c.post("/admin/source", data={"action": "video", "video": (open(vp, "rb"), "t.mp4")}, content_type="multipart/form-data")
ok("video upload starts engine + logged in media_files", r.status_code == 302 and db.q("SELECT kind FROM media_files")[0]["kind"] == "video" and A.engine._started)
r = c.post("/admin/source", data={"action": "camera", "camera": "0", "mic": "on"})
ok("camera with mic flag", A.engine._started[1].get("mic") is True)
ok("snapshot 404 before any video runs", c.get("/api/live/snapshot").status_code == 404)
A.engine.raw, A.engine.raw_M = np.full((540, 960, 3), 80, np.uint8), np.eye(3)
r = c.get("/api/live/snapshot"); j = r.get_json()
ok("live snapshot returns frame + transform for on-video editor", r.status_code == 200 and j["image"].startswith("data:image/jpeg;base64,") and len(j["M"]) == 2 and j["W"] == 960)
dash = c.get("/admin").data
ok("editor is ON the dashboard video (no separate page)", b"Edit lanes on video" in dash and b"b_save" in dash and b"Direction density" in dash and c.get("/admin/zones").status_code == 404)
good = json.load(open("/tmp/zcfg.json")) if os.path.exists("/tmp/zcfg.json") else json.load(open(C.LANES_FILE))
A.engine._reload = False
ok("valid zone config saved", c.post("/api/zones/config", json=good).status_code == 200 and json.load(open(C.LANES_FILE)) == good)
ok("saving zones hot-reloads the running engine", A.engine._reload is True)
ok("backup of previous zones kept", os.path.exists(C.LANES_FILE + ".bak"))
nodir = {**good, "lanes": {n: {k: v for k, v in L.items() if k != "direction"} for n, L in good["lanes"].items()}}
ok("zone config WITHOUT directions accepted", c.post("/api/zones/config", json=nodir).status_code == 200)
ok("north_direction validated", c.post("/api/zones/config", json={**good, "north_direction": [0, 0]}).status_code == 400)
bad = {"lanes": {"X": {"polygon": [[0, 0], [1, 1]], "direction": [0, 0]}}}
r = c.post("/api/zones/config", json=bad); ok("invalid zone config rejected", r.status_code == 400 and len(r.get_json()["errors"]) >= 2)
ok("data layer page lists all stores", all(s in c.get("/admin/data").data for s in (b"Signal Timing History", b"Detection Logs", b"Task Management", b"Video Storage")))
r = c.get("/admin/export/table/users.csv"); ok("users CSV excludes password hashes", r.status_code == 200 and b"password_hash" not in r.data and b"pbkdf2" not in r.data and b"scrypt" not in r.data)
ok("unknown table export blocked", c.get("/admin/export/table/sqlite_master.csv").status_code == 404)
ok("key info stored on report", db.q("SELECT ai_key_info FROM reports", one=True)["ai_key_info"] is not None)
login("police1", "police123"); ok("police page has AI assistant", b"Ask AI assistant" in c.get("/police").data)
print("ALL APP TESTS PASSED")
