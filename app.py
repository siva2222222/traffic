import os
import io
import csv
import uuid
import time
import json
import base64
import shutil
from functools import wraps
from datetime import datetime

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash,
    jsonify, Response, send_file, abort
)
from werkzeug.security import check_password_hash, generate_password_hash

from config import C
import database as db
import services as sv
import llm
import audio
import image_analyze
import video_analyze
import emergency_vision as EV
from engine import TrafficEngine

# Four independent video-processing engines: one uploaded video per dashboard lane.
LANE_NAMES = ["North", "East", "West", "South"]
lane_engines = {name: TrafficEngine() for name in LANE_NAMES}

_emergency_episode = {"lane": None, "last_seen": 0.0, "handled": set()}
EMERGENCY_CLEAR_GRACE = 3.0


def enforce_emergency_lock(panels):
    """Pause other running lanes when a live lane detects an emergency vehicle/siren.

    Paused lanes have frozen metrics, so they are not considered live emergency sources.
    A lane paused automatically for an episode stays paused until an admin resumes it.
    """
    emergency_lane = next(
        (
            name for name, metrics in panels.items()
            if not metrics.get("paused")
            and (metrics.get("emergency_lane") or metrics.get("siren"))
        ),
        None,
    )
    episode = _emergency_episode
    now = time.time()

    if emergency_lane:
        if emergency_lane != episode["lane"]:
            episode["lane"] = emergency_lane
            episode["handled"] = set()
        episode["last_seen"] = now
    elif episode["lane"] and now - episode["last_seen"] > EMERGENCY_CLEAR_GRACE:
        episode["lane"] = None
        episode["handled"] = set()

    if not episode["lane"]:
        return

    for name, eng in lane_engines.items():
        if (
            name != episode["lane"]
            and name not in episode["handled"]
            and eng.running
            and not eng.paused
        ):
            eng.pause(auto=True)
            episode["handled"].add(name)


def pause_other_lanes(active_lane):
    """Enforce one active video runner at a time."""
    for name, eng in lane_engines.items():
        if name != active_lane and eng.running and not eng.paused:
            eng.pause(auto=True)


app = Flask(__name__)
app.secret_key = C.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 800 * 1024 * 1024

for directory in (C.UPLOAD_DIR, C.EVIDENCE_DIR):
    os.makedirs(directory, exist_ok=True)

db.init_db()

REPORT_TYPES = [
    "Accident", "Vehicle Breakdown", "Illegal Parking",
    "Road Obstruction", "Traffic Violation", "Other",
]


def cur_user():
    uid = session.get("uid")
    return db.q("SELECT * FROM users WHERE id=?", (uid,), one=True) if uid else None


def need(*roles):
    def deco(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            user = cur_user()
            if not user:
                return redirect(url_for("login", next=request.path))
            if roles and user["role"] not in roles:
                abort(403)
            return func(*args, **kwargs)
        return wrapped
    return deco


@app.context_processor
def inject():
    user = cur_user()
    unread = (
        db.q(
            "SELECT COUNT(*) n FROM notifications WHERE user_id=? AND is_read=0",
            (user["id"],),
            one=True,
        )["n"]
        if user else 0
    )
    return {"user": user, "unread": unread}


def save_upload(file_storage, extensions):
    """Save an uploaded file using a generated safe filename.

    Returns (relative_path, absolute_path); returns (None, None) for invalid uploads.
    """
    if not file_storage or not file_storage.filename:
        return None, None

    ext = os.path.splitext(file_storage.filename)[1].lower()
    if ext not in extensions:
        return None, None

    filename = f"{uuid.uuid4().hex[:12]}{ext}"
    path = os.path.join(C.UPLOAD_DIR, filename)
    file_storage.save(path)
    return "uploads/" + filename, path


IMG = {".jpg", ".jpeg", ".png", ".webp"}


def log_media(kind, relative_path, original_name):
    db.x(
        "INSERT INTO media_files(kind,filename,original,uploaded_by,ts) VALUES(?,?,?,?,?)",
        (kind, relative_path, original_name, session.get("uid"), db.now()),
    )


# ---------------- authentication ----------------
@app.route("/")
def index():
    user = cur_user()
    if user:
        destinations = {
            "admin": "/admin",
            "police": "/police",
            "citizen": "/report",
        }
        return redirect(destinations.get(user["role"], "/"))
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = db.q(
            "SELECT * FROM users WHERE username=?",
            (username,),
            one=True,
        )
        if user and check_password_hash(user["password_hash"], password):
            session["uid"] = user["id"]
            return redirect(request.args.get("next") or url_for("index"))
        flash("Invalid username or password")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        form = request.form
        name = form.get("name", "").strip()
        username = form.get("username", "").strip()
        password = form.get("password", "")

        if not name or not username:
            flash("Name and username are required")
        elif len(password) < 6:
            flash("Password must be at least 6 characters")
        elif db.q(
            "SELECT id FROM users WHERE username=?",
            (username,),
            one=True,
        ):
            flash("Username already taken")
        else:
            uid = db.x(
                "INSERT INTO users(name,username,password_hash,role,phone) VALUES(?,?,?,?,?)",
                (
                    name,
                    username,
                    generate_password_hash(password),
                    "citizen",
                    form.get("phone"),
                ),
            )
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
    user = cur_user()
    rows = db.q(
        "SELECT * FROM notifications WHERE user_id=? ORDER BY id DESC LIMIT 50",
        (user["id"],),
    )
    db.x("UPDATE notifications SET is_read=1 WHERE user_id=?", (user["id"],))
    return render_template("notifications.html", rows=rows)


# ---------------- citizen portal ----------------
@app.route("/report", methods=["GET", "POST"])
@need("citizen")
def report():
    if request.method == "POST":
        form = request.form
        photo, _ = save_upload(
            request.files.get("photo"),
            IMG | {".mp4", ".mov"},
        )
        try:
            lat = float(form["lat"]) if form.get("lat") else None
            lng = float(form["lng"]) if form.get("lng") else None
        except (TypeError, ValueError):
            flash("Invalid location coordinates")
            return redirect("/report")

        report_type = form.get("type", "Other")
        if report_type not in REPORT_TYPES:
            abort(400)

        report_id = sv.create_report(
            session["uid"],
            report_type,
            form.get("description", ""),
            form.get("location", ""),
            lat,
            lng,
            photo,
        )
        flash(f"Report #{report_id} submitted. Track it under 'My Reports'.")
        return redirect("/my-reports")
    return render_template("report.html", types=REPORT_TYPES)


@app.route("/my-reports")
@need("citizen")
def my_reports():
    rows = db.q(
        """SELECT r.*, t.task_code, u.name officer
           FROM reports r
           LEFT JOIN tasks t ON t.report_id=r.id
           LEFT JOIN users u ON u.id=t.officer_id
           WHERE r.citizen_id=?
           ORDER BY r.id DESC""",
        (session["uid"],),
    )
    return render_template("my_reports.html", rows=rows)


# ---------------- admin dashboard ----------------
@app.route("/admin")
@need("admin")
def admin():
    return render_template("admin_dashboard.html")


@app.route("/api/live")
@need("admin")
def api_live():
    """Return live metrics for all four independently uploaded dashboard videos."""
    panels = {}
    for name, eng in lane_engines.items():
        metrics = eng.get_metrics()
        metrics["paused"] = eng.paused
        metrics["auto_paused"] = eng.auto_paused
        panels[name] = metrics

    enforce_emergency_lock(panels)

    active = next(
        (
            metrics for metrics in panels.values()
            if metrics.get("status") in (
                "running", "loading model", "analysing audio"
            )
        ),
        {},
    )
    output = dict(active)
    output["panels"] = panels
    output["sources"] = {
        name: metrics.get("source", "")
        for name, metrics in panels.items()
    }
    output["pending"] = db.q(
        "SELECT COUNT(*) n FROM tasks WHERE status IN('Assigned','In Progress')",
        one=True,
    )["n"]
    output["new_reports"] = db.q(
        "SELECT COUNT(*) n FROM reports WHERE status='Pending'",
        one=True,
    )["n"]
    output["violations_today"] = sv.get_stats(1)["violations_total"]
    return jsonify(output)


@app.route("/api/live/<lane>")
@need("admin")
def api_live_lane(lane):
    lane = lane.title()
    if lane not in lane_engines:
        abort(404)
    metrics = lane_engines[lane].get_metrics()
    metrics["paused"] = lane_engines[lane].paused
    metrics["auto_paused"] = lane_engines[lane].auto_paused
    return jsonify(metrics)


@app.route("/api/summary")
@need("admin")
def api_summary():
    combined = {name: eng.get_metrics() for name, eng in lane_engines.items()}
    return jsonify({"text": llm.summarize_situation(combined)})


@app.route("/video_feed")
@need("admin")
def video_feed():
    return lane_video_feed("North")


@app.route("/video_feed/<lane>")
@need("admin")
def lane_video_feed(lane):
    """Stream the most recent annotated frame from the selected lane engine."""
    lane = lane.title()
    if lane not in lane_engines:
        abort(404)
    eng = lane_engines[lane]

    def generate():
        while True:
            try:
                jpg = eng.get_frame()
                if jpg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpg + b"\r\n"
                    )
            except GeneratorExit:
                break
            except Exception:
                app.logger.exception("Video feed error for lane %s", lane)
                time.sleep(0.2)
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/admin/source/<lane>", methods=["POST"])
@need("admin")
def source_lane(lane):
    lane = lane.title()
    if lane not in lane_engines:
        abort(404)

    eng = lane_engines[lane]
    action = request.form.get("action", "video")

    if action == "video":
        file_storage = request.files.get("video")
        relative_path, path = save_upload(
            file_storage,
            {".mp4", ".avi", ".mov", ".mkv"},
        )
        if not path:
            flash(f"Upload a valid video for {lane} (.mp4/.avi/.mov/.mkv)")
            return redirect("/admin")

        log_media("video", relative_path, file_storage.filename)
        eng.start(
            path,
            loop=bool(request.form.get("loop", "1")),
            name=f"{lane}: {os.path.basename(path)}",
        )
        pause_other_lanes(lane)
        flash(f"{lane} video started")

    elif action == "stop":
        eng.stop()
        flash(f"{lane} video stopped")

    elif action == "pause":
        for name, other in lane_engines.items():
            if other.running and not other.paused:
                other.pause(auto=(name != lane))
        flash(f"{lane} video paused")

    elif action == "resume":
        eng.resume()
        pause_other_lanes(lane)
        flash(f"{lane} video resumed")
    else:
        abort(400)

    return redirect("/admin")


@app.route("/admin/source", methods=["GET", "POST"])
@need("admin")
def source():
    audio_result = None
    if request.method == "POST":
        file_storage = request.files.get("audio")
        relative_path, path = save_upload(
            file_storage,
            {".wav", ".mp3", ".ogg", ".m4a", ".mp4", ".flac"},
        )
        if path:
            log_media("audio", relative_path, file_storage.filename)
            intervals, error = audio.analyse_file(path)
            audio_result = {"intervals": intervals, "error": error}
            if intervals:
                db.x(
                    "INSERT INTO emergency_events(ts,kind,lane,note) VALUES(?,?,?,?)",
                    (db.now(), "siren", None, "Siren in uploaded audio"),
                )
        else:
            flash("Unsupported audio file")
    return render_template(
        "admin_source.html",
        audio_result=audio_result,
        detector=audio.detector_name(),
    )


@app.route("/api/detect/image", methods=["POST"])
@need("admin", "police")
def api_detect_image():
    """POST multipart field 'image' and return image-analysis JSON."""
    relative_path, path = save_upload(request.files.get("image"), IMG)
    if not path:
        return jsonify(error="send an image file as form field 'image'"), 400

    result = image_analyze.analyse_image(path, want_scores=True)
    result.pop("annotated_path", None)
    return jsonify(result)


@app.route("/admin/video", methods=["GET", "POST"])
@need("admin")
def admin_video():
    """Start an asynchronous scan for emergency vehicles in an uploaded video."""
    job_id = request.args.get("job")
    if request.method == "POST":
        file_storage = request.files.get("video")
        relative_path, path = save_upload(
            file_storage,
            {".mp4", ".avi", ".mov", ".mkv", ".webm"},
        )
        if not path:
            flash("Upload a .mp4 / .avi / .mov / .mkv video")
            return redirect("/admin/video")

        log_media("video", relative_path, file_storage.filename)
        admins = [
            row["id"]
            for row in db.q("SELECT id FROM users WHERE role='admin'")
        ]
        source_name = os.path.basename(path)

        def on_detect(detection):
            note = (
                f"{detection['type'].replace('_', ' ')} at {detection['time']} "
                f"in {source_name} (conf {detection['conf']}, "
                f"via {detection['method']})"
            )
            db.x(
                "INSERT INTO emergency_events(ts,kind,lane,note) VALUES(?,?,?,?)",
                (db.now(), detection["type"], None, note),
            )
            for uid in admins:
                db.x(
                    "INSERT INTO notifications(user_id,msg,created_at) VALUES(?,?,?)",
                    (
                        uid,
                        "Emergency vehicle in uploaded video: " + note,
                        db.now(),
                    ),
                )

        job = video_analyze.start_job(
            path,
            name=file_storage.filename,
            on_detect=on_detect,
        )
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
    rows = db.q(
        """SELECT r.*, c.name citizen, t.id task_id, t.task_code, o.name officer
           FROM reports r
           JOIN users c ON c.id=r.citizen_id
           LEFT JOIN tasks t ON t.report_id=r.id
           LEFT JOIN users o ON o.id=t.officer_id
           ORDER BY r.id DESC"""
    )
    officers = db.q("SELECT id,name FROM users WHERE role='police'")
    return render_template(
        "admin_reports.html",
        rows=rows,
        officers=officers,
    )


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
    officer_id = request.form.get("officer_id")
    old_tasks = db.q(
        "SELECT id FROM tasks WHERE report_id=? AND status!='Completed'",
        (rid,),
    )
    for task in old_tasks:
        db.x("DELETE FROM tasks WHERE id=?", (task["id"],))

    sv.assign_report(
        rid,
        int(officer_id) if officer_id else None,
        "manual" if officer_id else "auto",
    )
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
    violation_type = request.args.get("type")
    rows = db.q(
        "SELECT * FROM violations WHERE (?1 IS NULL OR vtype=?1) ORDER BY id DESC LIMIT 300",
        (violation_type,),
    )
    types = [
        row["vtype"]
        for row in db.q("SELECT DISTINCT vtype FROM violations")
    ]
    return render_template(
        "admin_violations.html",
        rows=rows,
        types=types,
        sel=violation_type,
    )


@app.route("/admin/violations/<int:vid>/delete", methods=["POST"])
@need("admin")
def delete_violation(vid):
    db.x("DELETE FROM violations WHERE id=?", (vid,))
    flash("Violation deleted")
    return redirect(request.referrer or "/admin/violations")


@app.route("/admin/violations/delete_all", methods=["POST"])
@need("admin")
def delete_all_violations():
    violation_type = request.form.get("type") or None
    if violation_type:
        db.x("DELETE FROM violations WHERE vtype=?", (violation_type,))
        flash(f"All '{violation_type}' violations deleted")
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
    density = db.q(
        "SELECT ts,north,east,south,west FROM density_log ORDER BY id DESC LIMIT 60"
    )[::-1]
    return jsonify(
        {
            "stats": sv.get_stats(7),
            "density": density,
            "lane_names": sv.lane_names(),
            "classes": db.q(
                "SELECT vclass, MAX(count) n FROM vehicle_counts GROUP BY vclass"
            ),
        }
    )


@app.route("/admin/assistant")
@need("admin")
def assistant_page():
    return render_template("admin_assistant.html")


@app.route("/api/assistant", methods=["POST"])
@need("admin", "police")
def api_assistant():
    payload = request.get_json(silent=True) or {}
    question = payload.get("q", "")
    return jsonify({"answer": llm.assistant(question, sv.assistant_context())})


PERIODS = {"daily": 1, "weekly": 7, "monthly": 30}


@app.route("/admin/export/<period>.<fmt>")
@need("admin")
def export(period, fmt):
    if period not in PERIODS or fmt not in ("csv", "pdf"):
        abort(404)

    stats = sv.get_stats(PERIODS[period])
    narrative = llm.narrative(period, stats)
    violations = db.q(
        """SELECT ts,vtype,vehicle_class,lane,speed_kmh,fine
           FROM violations WHERE ts>=? ORDER BY id DESC""",
        (stats["since"],),
    )
    filename = f"traffic_{period}_{datetime.now():%Y%m%d}"

    if fmt == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Traffic report", period, "since", stats["since"]])
        for key in (
            "violations_total", "fines_total", "reports_total",
            "tasks_completed", "avg_completion_min", "busiest_lane",
            "emergency_events",
        ):
            writer.writerow([key, stats[key]])
        writer.writerow([])
        writer.writerow(["Summary", narrative])
        writer.writerow([])
        writer.writerow(
            ["time", "violation", "vehicle", "lane", "speed_kmh", "fine"]
        )
        for row in violations:
            writer.writerow(
                [
                    row["ts"], row["vtype"], row["vehicle_class"],
                    row["lane"], row["speed_kmh"], row["fine"],
                ]
            )
        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}.csv"
            },
        )

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from xml.sax.saxutils import escape

    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    safe_narrative = escape(str(narrative)).replace("\n", "<br/>")
    elements = [
        Paragraph(
            f"Smart Traffic Management - {period.title()} Report",
            styles["Title"],
        ),
        Paragraph(f"Period since {escape(str(stats['since']))}", styles["Normal"]),
        Spacer(1, 10),
        Paragraph(safe_narrative, styles["BodyText"]),
        Spacer(1, 10),
    ]
    key_metrics = [["Metric", "Value"]] + [
        [key.replace("_", " ").title(), str(stats[key])]
        for key in (
            "violations_total", "fines_total", "reports_total",
            "tasks_completed", "avg_completion_min", "busiest_lane",
            "emergency_events",
        )
    ]
    key_metrics += [
        [f"Violations: {escape(str(key))}", str(value)]
        for key, value in stats["violations_by_type"].items()
    ]
    metrics_table = Table(key_metrics, colWidths=[260, 200])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3c7a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ]
        )
    )
    elements.extend(
        [
            metrics_table,
            Spacer(1, 12),
            Paragraph("Recent violations", styles["Heading3"]),
        ]
    )
    violation_rows = [["Time", "Type", "Vehicle", "Lane", "Fine"]]
    violation_rows += [
        [
            str(row["ts"]), str(row["vtype"]), str(row["vehicle_class"]),
            str(row["lane"] or "-"), str(row["fine"]),
        ]
        for row in violations[:40]
    ]
    violation_table = Table(violation_rows, repeatRows=1)
    violation_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(violation_table)
    SimpleDocTemplate(buffer, pagesize=A4).build(elements)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename + ".pdf",
    )


# ---------------- data layer ----------------
STORES = [
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
EXPORTABLE = {table: timestamp for _, table, timestamp, _ in STORES}


@app.route("/admin/data")
@need("admin")
def data_layer():
    rows = []
    for title, table, timestamp_column, description in STORES:
        count = db.q(f"SELECT COUNT(*) n FROM {table}", one=True)["n"]
        last = (
            db.q(
                f"SELECT MAX({timestamp_column}) m FROM {table}",
                one=True,
            )["m"]
            if timestamp_column else None
        )
        rows.append(
            {
                "title": title,
                "table": table,
                "count": count,
                "last": last,
                "desc": description,
            }
        )
    media_counts = db.q("SELECT kind, COUNT(*) n FROM media_files GROUP BY kind")
    return render_template(
        "admin_data.html",
        rows=rows,
        media={row["kind"]: row["n"] for row in media_counts},
    )


@app.route("/admin/export/table/<name>.csv")
@need("admin")
def export_table(name):
    if name not in EXPORTABLE:
        abort(404)

    columns = [
        column["name"]
        for column in db.q(f"PRAGMA table_info({name})")
        if column["name"] != "password_hash"
    ]
    if not columns:
        abort(404)

    rows = db.q(f"SELECT {','.join(columns)} FROM {name}")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[column] for column in columns])

    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={name}.csv"},
    )


# ---------------- lane-zone editor ----------------
def _validate_zones(cfg):
    errors = []
    lanes = cfg.get("lanes")
    if not isinstance(lanes, dict) or not lanes:
        return ["Add at least one lane"]

    def valid_point(point):
        return (
            isinstance(point, list)
            and len(point) == 2
            and all(
                isinstance(value, (int, float)) and -0.5 <= value <= 1.5
                for value in point
            )
        )

    north_direction = cfg.get("north_direction")
    if north_direction is not None and not (
        isinstance(north_direction, list)
        and len(north_direction) == 2
        and any(abs(value) > 1e-6 for value in north_direction)
    ):
        errors.append("north_direction must be a 2-number vector")

    for lane_name, lane_cfg in lanes.items():
        if not isinstance(lane_cfg, dict):
            errors.append(f"Lane {lane_name}: configuration must be an object")
            continue

        polygon = lane_cfg.get("polygon")
        if not (
            isinstance(polygon, list)
            and len(polygon) >= 3
            and all(valid_point(point) for point in polygon)
        ):
            errors.append(f"Lane {lane_name}: polygon needs 3+ points")

        direction = lane_cfg.get("direction")
        if direction is not None and not (
            isinstance(direction, list)
            and len(direction) == 2
            and all(isinstance(value, (int, float)) for value in direction)
            and any(abs(value) > 1e-6 for value in direction)
        ):
            errors.append(f"Lane {lane_name}: invalid direction")

        stop_line = lane_cfg.get("stop_line")
        if stop_line is not None and not (
            isinstance(stop_line, list)
            and len(stop_line) == 2
            and all(valid_point(point) for point in stop_line)
        ):
            errors.append(f"Lane {lane_name}: stop line needs 2 points")

    return errors


def _active_lane_engine():
    """Return a running dashboard engine for zone-editor snapshots.

    The editor previously used a separate global engine that the four-lane dashboard
    did not start. Prefer the currently running lane engine, then any available lane.
    """
    for eng in lane_engines.values():
        if eng.running and not eng.paused:
            return eng
    for eng in lane_engines.values():
        if eng.running:
            return eng
    return None


@app.route("/api/live/snapshot")
@need("admin")
def live_snapshot():
    """Freeze a dashboard lane's frame for the zone editor."""
    eng = _active_lane_engine()
    if eng is None:
        return jsonify(error="No lane video running yet - start a source first"), 404

    snapshot = eng.snapshot()
    if snapshot is None:
        return jsonify(error="No video frame available yet"), 404

    frame, transform = snapshot
    encoded = cv2.imencode(".jpg", frame)
    if not encoded[0]:
        return jsonify(error="Could not encode the current frame"), 500

    return jsonify(
        image="data:image/jpeg;base64," + base64.b64encode(encoded[1].tobytes()).decode(),
        M=transform[:2].tolist(),
        W=frame.shape[1],
        H=frame.shape[0],
    )


@app.route("/api/zones/config", methods=["GET", "POST"])
@need("admin")
def zones_config():
    if request.method == "GET":
        if not os.path.exists(C.LANES_FILE):
            return jsonify(error="Lane configuration file does not exist"), 404
        return send_file(C.LANES_FILE, mimetype="application/json")

    cfg = request.get_json(force=True, silent=True) or {}
    errors = _validate_zones(cfg)
    if errors:
        return jsonify(ok=False, errors=errors), 400

    if os.path.exists(C.LANES_FILE):
        shutil.copy2(C.LANES_FILE, C.LANES_FILE + ".bak")

    with open(C.LANES_FILE, "w", encoding="utf-8") as file:
        json.dump(cfg, file, indent=2)

    # Reload every dashboard engine so each lane sees the updated zone configuration.
    reload_errors = []
    for name, eng in lane_engines.items():
        try:
            eng.reload_cfg()
        except Exception:
            app.logger.exception("Failed to reload lane config for %s", name)
            reload_errors.append(name)

    if reload_errors:
        return jsonify(
            ok=True,
            lanes=list(cfg["lanes"]),
            warning="Configuration saved, but reload failed for: " + ", ".join(reload_errors),
        ), 200

    return jsonify(ok=True, lanes=list(cfg["lanes"]))


# ---------------- police portal ----------------
@app.route("/police")
@need("police")
def police():
    user = cur_user()
    active = sv.task_rows(
        "t.officer_id=? AND t.status!='Completed'",
        (user["id"],),
    )
    history = sv.task_rows(
        "t.officer_id=? AND t.status='Completed'",
        (user["id"],),
    )
    return render_template("police.html", active=active, history=history)


@app.route("/police/task/<int:tid>/update", methods=["POST"])
@need("police")
def police_update(tid):
    task = db.q(
        "SELECT * FROM tasks WHERE id=? AND officer_id=?",
        (tid, session["uid"]),
        one=True,
    )
    if not task:
        abort(404)

    evidence, _ = save_upload(request.files.get("evidence"), IMG)
    status = request.form.get("status") or task["status"]
    if status not in ("Assigned", "In Progress", "Completed"):
        abort(400)

    sv.set_task_status(
        tid,
        status,
        request.form.get("remarks", ""),
        evidence,
    )
    flash(f"Task {task['task_code']} -> {status}")
    return redirect("/police")


@app.route("/police/location", methods=["POST"])
@need("police")
def police_location():
    payload = request.get_json(silent=True) or {}
    try:
        latitude = float(payload["lat"]) if payload.get("lat") is not None else None
        longitude = float(payload["lng"]) if payload.get("lng") is not None else None
    except (TypeError, ValueError):
        return jsonify(ok=False, error="Invalid coordinates"), 400

    db.x(
        "UPDATE users SET lat=?, lng=? WHERE id=?",
        (latitude, longitude, session["uid"]),
    )
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        debug=False,
        threaded=True,
    )
