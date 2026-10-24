import os, io, csv, uuid, time
import cv2
import numpy as np
from functools import wraps
from datetime import datetime
from flask import (Flask, render_template, request, redirect, url_for, session, flash, jsonify,
                   Response, send_file, abort)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from config import C
import database as db, services as sv, llm, audio
import image_analyze
import video_analyze
import emergency_vision as EV
from engine import engine, TrafficEngine

# Four independent video-processing engines: one uploaded video per dashboard lane.
LANE_NAMES = ["North", "East", "West", "South"]
lane_engines = {name: TrafficEngine() for name in LANE_NAMES}

_emergency_episode = {"lane": None, "last_seen": 0.0, "handled": set()}
EMERGENCY_CLEAR_GRACE = 3.0   # seconds with no detection before an episode is considered over

def enforce_emergency_lock(panels):
    """If any lane currently shows an ambulance/fire_engine (or siren), auto-pause every other
    running lane so the emergency vehicle has priority with no distracting cross-traffic feeds.
    Paused-for-this-reason lanes stay paused until the admin clicks Resume (see /admin/source/<lane>
    action=resume) - they don't auto-resume just because the emergency cleared on its own.

    Detection (CLIP confidence / siren audio) flickers frame to frame, so "emergency_lane" can
    drop out and reappear within the same real-world emergency. EMERGENCY_CLEAR_GRACE debounces
    that: the episode only ends after the emergency is absent for a few seconds straight. Lanes
    already auto-paused this episode are tracked in "handled" so a lane the admin resumes stays
    resumed even while detection keeps flickering - it's only eligible to be auto-paused again
    once this episode truly ends and a new one begins."""
    # A paused lane's emergency_lane/siren fields are frozen at whatever they last were (it stops
    # processing new frames), so they must not count as a live signal - otherwise a paused lane can
    # permanently squat the "emergency source" slot and spuriously re-pause lanes that are fine.
    emergency_lane = next((name for name, m in panels.items()
                           if not m.get("paused") and (m.get("emergency_lane") or m.get("siren"))), None)
    ep, now = _emergency_episode, time.time()
    if emergency_lane:
        if emergency_lane != ep["lane"]:
            ep["lane"], ep["handled"] = emergency_lane, set()
        ep["last_seen"] = now
    elif ep["lane"] and now - ep["last_seen"] > EMERGENCY_CLEAR_GRACE:
        ep["lane"], ep["handled"] = None, set()
    if not ep["lane"]:
        return
    for name, eng in lane_engines.items():
        if name != ep["lane"] and name not in ep["handled"] and eng.running and not eng.paused:
            eng.pause(auto=True)
            ep["handled"].add(name)

def pause_other_lanes(active_lane):
    """Only one lane may be actively running at a time. Call this after starting/resuming
    active_lane so every other lane auto-pauses (manual pause/resume on those lanes still
    works normally - this just enforces the "one runner at a time" rule each time a lane
    is (re)started)."""
    for name, eng in lane_engines.items():
        if name != active_lane and eng.running and not eng.paused:
            eng.pause(auto=True)

app = Flask(__name__)
app.secret_key = C.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 800 * 1024 * 1024
for d in (C.UPLOAD_DIR, C.EVIDENCE_DIR):
    os.makedirs(d, exist_ok=True)
db.init_db()

REPORT_TYPES = ["Accident", "Vehicle Breakdown", "Illegal Parking", "Road Obstruction", "Traffic Violation", "Other"]

def cur_user():
    uid = session.get("uid")
    return db.q("SELECT * FROM users WHERE id=?", (uid,), one=True) if uid else None

def need(*roles):
    def deco(f):
        @wraps(f)
        def w(*a, **k):
            u = cur_user()
            if not u:
                return redirect(url_for("login", next=request.path))
            if roles and u["role"] not in roles:
                abort(403)
            return f(*a, **k)
        return w
    return deco

@app.context_processor
def inject():
    u = cur_user()
    unread = db.q("SELECT COUNT(*) n FROM notifications WHERE user_id=? AND is_read=0", (u["id"],), one=True)["n"] if u else 0
    return {"user": u, "unread": unread}

def save_upload(f, exts):
    if not f or not f.filename:
        return None, None
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in exts:
        return None, None
    name = f"{uuid.uuid4().hex[:12]}{ext}"
    path = os.path.join(C.UPLOAD_DIR, name)
    f.save(path)
    return "uploads/" + name, path

IMG = {".jpg", ".jpeg", ".png", ".webp"}

def log_media(kind, rel, original):
    db.x("INSERT INTO media_files(kind,filename,original,uploaded_by,ts) VALUES(?,?,?,?,?)",
         (kind, rel, original, session.get("uid"), db.now()))

# ---------------- auth ----------------
@app.route("/")
def index():
    u = cur_user()
    if u:
        return redirect({"admin": "/admin", "police": "/police", "citizen": "/report"}[u["role"]])
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db.q("SELECT * FROM users WHERE username=?", (request.form["username"].strip(),), one=True)
        if u and check_password_hash(u["password_hash"], request.form["password"]):
            session["uid"] = u["id"]
            return redirect(request.args.get("next") or url_for("index"))
        flash("Invalid username or password")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        f = request.form
        if len(f["password"]) < 6:
            flash("Password must be at least 6 characters")
        elif db.q("SELECT id FROM users WHERE username=?", (f["username"].strip(),), one=True):
            flash("Username already taken")
        else:
            uid = db.x("INSERT INTO users(name,username,password_hash,role,phone) VALUES(?,?,?,?,?)",
                       (f["name"].strip(), f["username"].strip(), generate_password_hash(f["password"]), "citizen", f.get("phone")))
            session["uid"] = uid
            return redirect("/report")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/notifications")
@need()
def notifications():
    u = cur_user()
    rows = db.q("SELECT * FROM notifications WHERE user_id=? ORDER BY id DESC LIMIT 50", (u["id"],))
    db.x("UPDATE notifications SET is_read=1 WHERE user_id=?", (u["id"],))
    return render_template("notifications.html", rows=rows)

# ---------------- public portal ----------------
@app.route("/report", methods=["GET", "POST"])
@need("citizen")
def report():
    if request.method == "POST":
        f = request.form
        photo, _ = save_upload(request.files.get("photo"), IMG | {".mp4", ".mov"})
        lat = float(f["lat"]) if f.get("lat") else None
        lng = float(f["lng"]) if f.get("lng") else None
        rid = sv.create_report(session["uid"], f["type"], f["description"], f["location"], lat, lng, photo)
        flash(f"Report #{rid} submitted. Track it under 'My Reports'.")
        return redirect("/my-reports")
    return render_template("report.html", types=REPORT_TYPES)

@app.route("/my-reports")
@need("citizen")
def my_reports():
    rows = db.q("""SELECT r.*, t.task_code, u.name officer FROM reports r LEFT JOIN tasks t ON t.report_id=r.id
                   LEFT JOIN users u ON u.id=t.officer_id WHERE r.citizen_id=? ORDER BY r.id DESC""", (session["uid"],))
    return render_template("my_reports.html", rows=rows)

# ---------------- admin portal ----------------
@app.route("/admin")
@need("admin")
def admin():
    return render_template("admin_dashboard.html")

@app.route("/api/live")
@need("admin")
def api_live():
    """Return live data for all four independently uploaded videos."""
    panels = {}
    for name, eng in lane_engines.items():
        m = eng.get_metrics()
        m["paused"], m["auto_paused"] = eng.paused, eng.auto_paused
        panels[name] = m
    enforce_emergency_lock(panels)
    # Keep old fields for existing pages while adding the four-panel data.
    active = next((m for m in panels.values() if m.get("status") in ("running", "loading model", "analysing audio")), {})
    out = dict(active)
    out["panels"] = panels
    out["sources"] = {k: v.get("source", "") for k,v in panels.items()}
    out["pending"] = db.q("SELECT COUNT(*) n FROM tasks WHERE status IN('Assigned','In Progress')", one=True)["n"]
    out["new_reports"] = db.q("SELECT COUNT(*) n FROM reports WHERE status='Pending'", one=True)["n"]
    out["violations_today"] = sv.get_stats(1)["violations_total"]
    return jsonify(out)

@app.route("/api/live/<lane>")
@need("admin")
def api_live_lane(lane):
    lane = lane.title()
    if lane not in lane_engines:
        abort(404)
    return jsonify(lane_engines[lane].get_metrics())

@app.route("/api/summary")
@need("admin")
def api_summary():
    combined = {k: e.get_metrics() for k,e in lane_engines.items()}
    return jsonify({"text": llm.summarize_situation(combined)})

@app.route("/video_feed")
@need("admin")
def video_feed():
    # Backward compatible default feed: North panel.
    return lane_video_feed("North")

@app.route("/video_feed/<lane>")
@need("admin")
def lane_video_feed(lane):
    """Each dashboard panel streams its own independently uploaded video."""
    lane = lane.title()
    if lane not in lane_engines:
        abort(404)
    eng = lane_engines[lane]
    def gen():
        while True:
            try:
                jpg = eng.get_frame()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            except Exception:
                pass
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/admin/source/<lane>", methods=["POST"])
@need("admin")
def source_lane(lane):
    lane = lane.title()
    if lane not in lane_engines:
        abort(404)
    eng = lane_engines[lane]
    act = request.form.get("action", "video")
    if act == "video":
        f = request.files.get("video")
        rel, path = save_upload(f, {".mp4", ".avi", ".mov", ".mkv"})
        if not path:
            flash(f"Upload a valid video for {lane} (.mp4/.avi/.mov/.mkv)")
            return redirect("/admin")
        log_media("video", rel, f.filename)
        eng.start(path, loop=bool(request.form.get("loop", "1")), name=f"{lane}: {os.path.basename(path)}")
        pause_other_lanes(lane)
        flash(f"{lane} video started")
    elif act == "stop":
        eng.stop()
        flash(f"{lane} video stopped")
    elif act == "pause":
        for name, other in lane_engines.items():
            if other.running and not other.paused:
                other.pause(auto=(name != lane))
        flash(f"{lane} video paused")
    elif act == "resume":
        eng.resume()
        pause_other_lanes(lane)
        flash(f"{lane} video resumed")
    return redirect("/admin")

@app.route("/admin/source", methods=["GET", "POST"])
@need("admin")
def source():
    audio_result = None
    if request.method == "POST":
        rel, path = save_upload(request.files.get("audio"), {".wav", ".mp3", ".ogg", ".m4a", ".mp4", ".flac"})
        if path:
            log_media("audio", rel, request.files["audio"].filename)
            iv, err = audio.analyse_file(path)
            audio_result = {"intervals": iv, "error": err}
            if iv:
                db.x("INSERT INTO emergency_events(ts,kind,lane,note) VALUES(?,?,?,?)", (db.now(), "siren", None, "Siren in uploaded audio"))
        else:
            flash("Unsupported audio file")
    return render_template("admin_source.html", audio_result=audio_result, detector=audio.detector_name())

@app.route("/api/detect/image", methods=["POST"])
@need("admin", "police")
def api_detect_image():
    """JSON API: POST multipart 'image' -> detection result. For the police app / external calls."""
    rel, path = save_upload(request.files.get("image"), IMG)
    if not path:
        return jsonify(error="send an image file as form field 'image'"), 400
    r = image_analyze.analyse_image(path, want_scores=True)
    r.pop("annotated_path", None)
    return jsonify(r)

@app.route("/admin/video", methods=["GET", "POST"])
@need("admin")
def admin_video():
    """Upload a video -> background scan for ambulance / fire engine -> timestamps + snapshots.

    Separate from the live dashboard engine: this one just answers "does an emergency vehicle
    appear in this clip, and when". Scanning runs in a thread; the page polls /api/video/status.
    """
    job_id = request.args.get("job")
    if request.method == "POST":
        f = request.files.get("video")
        rel, path = save_upload(f, {".mp4", ".avi", ".mov", ".mkv", ".webm"})
        if not path:
            flash("Upload a .mp4 / .avi / .mov / .mkv video")
            return redirect("/admin/video")
        log_media("video", rel, f.filename)
        admins = [a["id"] for a in db.q("SELECT id FROM users WHERE role='admin'")]
        src = os.path.basename(path)

        def on_detect(d):
            """Called from the scan thread each time a vehicle is flagged."""
            note = (f"{d['type'].replace('_', ' ')} at {d['time']} in {src} "
                    f"(conf {d['conf']}, via {d['method']})")
            db.x("INSERT INTO emergency_events(ts,kind,lane,note) VALUES(?,?,?,?)",
                 (db.now(), d["type"], None, note))
            for uid in admins:
                db.x("INSERT INTO notifications(user_id,msg,created_at) VALUES(?,?,?)",
                     (uid, "Emergency vehicle in uploaded video: " + note, db.now()))

        job = video_analyze.start_job(path, name=f.filename, on_detect=on_detect)
        return redirect(url_for("admin_video", job=job.id))

    status = {"clip": EV.backend_name(), "ocr": bool(C.ENABLE_OCR)}
    return render_template("admin_video.html", job_id=job_id, status=status)

@app.route("/api/video/status/<job_id>")
@need("admin", "police")
def api_video_status(job_id):
    job = video_analyze.get_job(job_id)
    if not job:
        return jsonify(error="unknown or expired scan job"), 404
    return jsonify(job.snapshot())

@app.route("/api/video/stop/<job_id>", methods=["POST"])
@need("admin")
def api_video_stop(job_id):
    job = video_analyze.get_job(job_id)
    if not job:
        return jsonify(error="unknown scan job"), 404
    job.stop()
    return jsonify(ok=True)

@app.route("/admin/reports")
@need("admin")
def admin_reports():
    rows = db.q("""SELECT r.*, c.name citizen, t.id task_id, t.task_code, o.name officer FROM reports r
                   JOIN users c ON c.id=r.citizen_id LEFT JOIN tasks t ON t.report_id=r.id
                   LEFT JOIN users o ON o.id=t.officer_id ORDER BY r.id DESC""")
    officers = db.q("SELECT id,name FROM users WHERE role='police'")
    return render_template("admin_reports.html", rows=rows, officers=officers)

@app.route("/admin/reports/delete-all", methods=["POST"])
@need("admin")
def admin_reports_delete_all():
    db.x("DELETE FROM tasks")
    db.x("DELETE FROM reports")
    flash("All reports deleted")
    return redirect("/admin/reports")

@app.route("/admin/reports/<int:rid>/assign", methods=["POST"])
@need("admin")
def assign(rid):
    oid = request.form.get("officer_id")
    old = db.q("SELECT id FROM tasks WHERE report_id=? AND status!='Completed'", (rid,))
    for t in old:
        db.x("DELETE FROM tasks WHERE id=?", (t["id"],))
    sv.assign_report(rid, int(oid) if oid else None, "manual" if oid else "auto")
    flash("Task assigned")
    return redirect("/admin/reports")

@app.route("/admin/tasks/<int:tid>/complete", methods=["POST"])
@need("admin")
def admin_complete(tid):
    sv.set_task_status(tid, "Completed", "Closed by admin")
    return redirect(request.referrer or "/admin/tasks")

@app.route("/admin/tasks")
@need("admin")
def admin_tasks():
    return render_template("admin_tasks.html", rows=sv.task_rows())

@app.route("/admin/tasks/delete-all", methods=["POST"])
@need("admin")
def admin_tasks_delete_all():
    db.x("DELETE FROM tasks")
    flash("All tasks deleted")
    return redirect("/admin/tasks")

@app.route("/admin/violations")
@need("admin")
def admin_violations():
    vt = request.args.get("type")
    rows = db.q("SELECT * FROM violations WHERE (?1 IS NULL OR vtype=?1) ORDER BY id DESC LIMIT 300", (vt,))
    types = [r["vtype"] for r in db.q("SELECT DISTINCT vtype FROM violations")]
    return render_template("admin_violations.html", rows=rows, types=types, sel=vt)

@app.route("/admin/violations/<int:vid>/delete", methods=["POST"])
@need("admin")
def delete_violation(vid):
    db.x("DELETE FROM violations WHERE id=?", (vid,))
    flash("Violation deleted")
    return redirect(request.referrer or "/admin/violations")

@app.route("/admin/violations/delete_all", methods=["POST"])
@need("admin")
def delete_all_violations():
    vt = request.form.get("type") or None
    if vt:
        db.x("DELETE FROM violations WHERE vtype=?", (vt,))
        flash(f"All '{vt}' violations deleted")
    else:
        db.x("DELETE FROM violations")
        flash("All violations deleted")
    return redirect(request.referrer or "/admin/violations")

@app.route("/admin/analytics")
@need("admin")
def analytics():
    return render_template("admin_analytics.html")

@app.route("/api/analytics")
@need("admin")
def api_analytics():
    dens = db.q("SELECT ts,north,east,south,west FROM density_log ORDER BY id DESC LIMIT 60")[::-1]
    return jsonify({"stats": sv.get_stats(7), "density": dens, "lane_names": sv.lane_names(),
                    "classes": db.q("SELECT vclass, MAX(count) n FROM vehicle_counts GROUP BY vclass")})

@app.route("/admin/assistant")
@need("admin")
def assistant_page():
    return render_template("admin_assistant.html")

@app.route("/api/assistant", methods=["POST"])
@need("admin", "police")
def api_assistant():
    q = (request.json or {}).get("q", "")
    return jsonify({"answer": llm.assistant(q, sv.assistant_context())})

PERIODS = {"daily": 1, "weekly": 7, "monthly": 30}

@app.route("/admin/export/<period>.<fmt>")
@need("admin")
def export(period, fmt):
    if period not in PERIODS or fmt not in ("csv", "pdf"):
        abort(404)
    stats = sv.get_stats(PERIODS[period])
    text = llm.narrative(period, stats)
    viol = db.q("SELECT ts,vtype,vehicle_class,lane,speed_kmh,fine FROM violations WHERE ts>=? ORDER BY id DESC", (stats["since"],))
    name = f"traffic_{period}_{datetime.now():%Y%m%d}"
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Traffic report", period, "since", stats["since"]])
        for k in ("violations_total", "fines_total", "reports_total", "tasks_completed", "avg_completion_min", "busiest_lane", "emergency_events"):
            w.writerow([k, stats[k]])
        w.writerow([]); w.writerow(["Summary", text]); w.writerow([])
        w.writerow(["time", "violation", "vehicle", "lane", "speed_kmh", "fine"])
        for r in viol:
            w.writerow([r["ts"], r["vtype"], r["vehicle_class"], r["lane"], r["speed_kmh"], r["fine"]])
        return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={name}.csv"})
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    buf = io.BytesIO()
    st = getSampleStyleSheet()
    els = [Paragraph(f"Smart Traffic Management - {period.title()} Report", st["Title"]),
           Paragraph(f"Period since {stats['since']}", st["Normal"]), Spacer(1, 10), Paragraph(text.replace("\n", "<br/>"), st["BodyText"]), Spacer(1, 10)]
    kp = [["Metric", "Value"]] + [[k.replace("_", " ").title(), str(stats[k])] for k in
          ("violations_total", "fines_total", "reports_total", "tasks_completed", "avg_completion_min", "busiest_lane", "emergency_events")]
    kp += [[f"Violations: {k}", str(v)] for k, v in stats["violations_by_type"].items()]
    t = Table(kp, colWidths=[260, 200])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3c7a")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
    els += [t, Spacer(1, 12), Paragraph("Recent violations", st["Heading3"])]
    vt = Table([["Time", "Type", "Vehicle", "Lane", "Fine"]] + [[r["ts"], r["vtype"], r["vehicle_class"], r["lane"] or "-", r["fine"]] for r in viol[:40]])
    vt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    els.append(vt)
    SimpleDocTemplate(buf, pagesize=A4).build(els)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=name + ".pdf")

# ---------------- data layer ----------------
STORES = [  # (title, table, timestamp column, description)
    ("Video Storage (uploaded / live)", "media_files", "ts", "Uploaded videos and audio files"),
    ("Detection Logs (vehicles)", "detection_logs", "ts", "One row per tracked vehicle: class + lane"),
    ("Detection Logs (violations)", "violations", "ts", "Violations with fine and evidence image"),
    ("Vehicle Counts (lane-wise)", "vehicle_counts", "ts", "Cumulative unique vehicles per lane and class"),
    ("Incident Reports (public / system)", "reports", "created_at", "Citizen reports with AI priority"),
    ("Task Management (assigned / completed)", "tasks", "assigned_at", "Police tasks and their status"),
    ("Signal Timing History", "signal_log", "ts", "Every green phase with reason"),
    ("Traffic Statistics (analytics data)", "density_log", "ts", "Lane density over time"),
    ("Emergency Events", "emergency_events", "ts", "Visual / siren emergency detections"),
    ("User Data (citizens / police)", "users", None, "Accounts (passwords are never exported)"),
]
EXPORTABLE = {t: ts for _, t, ts, _ in STORES}

@app.route("/admin/data")
@need("admin")
def data_layer():
    rows = []
    for title, table, ts, desc in STORES:
        n = db.q(f"SELECT COUNT(*) n FROM {table}", one=True)["n"]
        last = db.q(f"SELECT MAX({ts}) m FROM {table}", one=True)["m"] if ts else None
        rows.append({"title": title, "table": table, "count": n, "last": last, "desc": desc})
    media = db.q("SELECT kind, COUNT(*) n FROM media_files GROUP BY kind")
    return render_template("admin_data.html", rows=rows, media={m["kind"]: m["n"] for m in media})

@app.route("/admin/export/table/<name>.csv")
@need("admin")
def export_table(name):
    if name not in EXPORTABLE:
        abort(404)
    cols = [c["name"] for c in db.q(f"PRAGMA table_info({name})") if c["name"] != "password_hash"]
    data = db.q(f"SELECT {','.join(cols)} FROM {name}")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in data:
        w.writerow([r[c] for c in cols])
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={name}.csv"})

# ---------------- zone editor (draw lanes on a video frame) ----------------
def _validate_zones(cfg):
    errs = []
    lanes = cfg.get("lanes")
    if not isinstance(lanes, dict) or not lanes:
        return ["Add at least one lane"]
    def pt(p): return isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) and -0.5 <= v <= 1.5 for v in p)
    nd = cfg.get("north_direction")
    if nd is not None and not (isinstance(nd, list) and len(nd) == 2 and any(abs(v) > 1e-6 for v in nd)):
        errs.append("north_direction must be a 2-number vector")
    for n, L in lanes.items():
        if not (isinstance(L.get("polygon"), list) and len(L["polygon"]) >= 3 and all(pt(p) for p in L["polygon"])):
            errs.append(f"Lane {n}: polygon needs 3+ points")
        d = L.get("direction")                                   # optional (learned automatically when absent)
        if d is not None and not (isinstance(d, list) and len(d) == 2 and any(abs(v) > 1e-6 for v in d)):
            errs.append(f"Lane {n}: invalid direction")
        sl = L.get("stop_line")
        if sl is not None and not (len(sl) == 2 and all(pt(p) for p in sl)):
            errs.append(f"Lane {n}: stop line needs 2 points")
    return errs

@app.route("/api/live/snapshot")
@need("admin")
def live_snapshot():
    """Freeze the current live frame for the on-video lane editor (frame + camera-motion transform)."""
    import cv2, base64
    snap = engine.snapshot()
    if snap is None:
        return jsonify(error="No video running yet - start a source first"), 404
    frame, M = snap
    jpg = cv2.imencode(".jpg", frame)[1].tobytes()
    return jsonify(image="data:image/jpeg;base64," + base64.b64encode(jpg).decode(), M=M[:2].tolist(),
                   W=frame.shape[1], H=frame.shape[0])

@app.route("/api/zones/config", methods=["GET", "POST"])
@need("admin")
def zones_config():
    if request.method == "GET":
        return send_file(C.LANES_FILE, mimetype="application/json")
    cfg = request.get_json(force=True, silent=True) or {}
    errs = _validate_zones(cfg)
    if errs:
        return jsonify(ok=False, errors=errs), 400
    if os.path.exists(C.LANES_FILE):
        import shutil
        shutil.copy(C.LANES_FILE, C.LANES_FILE + ".bak")
    import json
    with open(C.LANES_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    engine.reload_cfg()                      # applies to the running video immediately
    return jsonify(ok=True, lanes=list(cfg["lanes"]))

# ---------------- police mobile app ----------------
@app.route("/police")
@need("police")
def police():
    u = cur_user()
    active = sv.task_rows("t.officer_id=? AND t.status!='Completed'", (u["id"],))
    history = sv.task_rows("t.officer_id=? AND t.status='Completed'", (u["id"],))
    return render_template("police.html", active=active, history=history)

@app.route("/police/task/<int:tid>/update", methods=["POST"])
@need("police")
def police_update(tid):
    t = db.q("SELECT * FROM tasks WHERE id=? AND officer_id=?", (tid, session["uid"]), one=True)
    if not t:
        abort(404)
    rel, _ = save_upload(request.files.get("evidence"), IMG)
    status = request.form.get("status") or t["status"]
    if status not in ("Assigned", "In Progress", "Completed"):
        abort(400)
    sv.set_task_status(tid, status, request.form.get("remarks", ""), rel)
    flash(f"Task {t['task_code']} -> {status}")
    return redirect("/police")

@app.route("/police/location", methods=["POST"])
@need("police")
def police_location():
    d = request.json or {}
    db.x("UPDATE users SET lat=?, lng=? WHERE id=?", (d.get("lat"), d.get("lng"), session["uid"]))
    return jsonify(ok=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False, threaded=True)
