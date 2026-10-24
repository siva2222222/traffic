"""REAL end-to-end test: YOLOv8 + ByteTrack + all rules on a video, through the Flask app.
   python tests/test_real_video.py path/to/video.mp4"""
import sys, os, time, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import C
C.DB_PATH = tempfile.mktemp(suffix=".db"); C.UPLOAD_DIR = tempfile.mkdtemp(); C.EVIDENCE_DIR = tempfile.mkdtemp()
import app as A, database as db
vid = sys.argv[1]
c = A.app.test_client(); c.post("/login", data={"username": "admin", "password": "admin123"})
r = c.post("/admin/source", data={"action": "video", "video": (open(vid, "rb"), os.path.basename(vid))}, content_type="multipart/form-data")
t0 = time.time(); seen = set()
while time.time() - t0 < 180:
    time.sleep(2); m = c.get("/api/live").get_json(); seen |= set((m.get("classes") or {}))
    if m.get("status") == "finished" or str(m.get("status", "")).startswith("error"): break
print("status:", m["status"], "| fps:", m.get("fps"), "| lanes:", list((m.get("density") or {})))
print("vehicle classes seen live:", sorted(seen))
print("lane-wise unique vehicles:", m.get("lane_counts"))
print("detection_logs:", db.q("SELECT COUNT(*) n FROM detection_logs", one=True)["n"], "| density_log rows:", db.q("SELECT COUNT(*) n FROM density_log", one=True)["n"])
print("violations:", db.q("SELECT vtype,lane,vehicle_class,speed_kmh FROM violations"))
assert m["status"] == "finished", m["status"]
assert c.get("/admin/export/daily.pdf").data[:4] == b"%PDF"
assert c.get("/admin/data").status_code == 200
print("REAL VIDEO TEST DONE")
