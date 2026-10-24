import math
from datetime import datetime, timedelta
import json
import database as db
import llm

def notify(uid, msg):
    if uid:
        db.x("INSERT INTO notifications(user_id,msg,created_at) VALUES(?,?,?)", (uid, msg, db.now()))

def haversine(a, b, c, d):
    R = 6371.0
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = p2 - p1, math.radians(d - b)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))

def officer_load(oid):
    return db.q("SELECT COUNT(*) n FROM tasks WHERE officer_id=? AND status IN('Assigned','In Progress')", (oid,), one=True)["n"]

def pick_officer(report):
    officers = db.q("SELECT * FROM users WHERE role='police'")
    if not officers:
        return None
    def score(o):
        load = officer_load(o["id"])
        if report.get("lat") is not None and o.get("lat") is not None:
            return haversine(report["lat"], report["lng"], o["lat"], o["lng"]) + load * 2.0  # km + workload penalty
        return 1000 + load
    return min(officers, key=score)["id"]

def assign_report(report_id, officer_id=None, by="auto"):
    rep = db.q("SELECT * FROM reports WHERE id=?", (report_id,), one=True)
    if not rep:
        return None
    officer_id = officer_id or pick_officer(rep)
    if not officer_id:
        return None
    tid = db.x("INSERT INTO tasks(report_id,officer_id,assigned_at,status,assigned_by) VALUES(?,?,?,?,?)",
               (report_id, officer_id, db.now(), "Assigned", by))
    db.x("UPDATE tasks SET task_code=? WHERE id=?", (f"TRF-{1000 + tid}", tid))
    db.x("UPDATE reports SET status='Assigned' WHERE id=?", (report_id,))
    notify(officer_id, f"New task TRF-{1000+tid}: {rep['type']} at {rep['location']} [{rep['priority']}]")
    notify(rep["citizen_id"], f"Your report #{report_id} was assigned to an officer.")
    return tid

def create_report(citizen_id, rtype, desc, loc, lat, lng, photo):
    ai = llm.classify_report(rtype, desc)
    rid = db.x("INSERT INTO reports(citizen_id,type,description,location,lat,lng,photo,priority,ai_summary,ai_action,ai_key_info,created_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
               (citizen_id, ai.get("category", rtype), desc, loc, lat, lng, photo,
                ai["priority"], ai.get("summary", ""), ai.get("action", ""),
                json.dumps(ai.get("key_info") or {}), db.now()))
    assign_report(rid)
    return rid

def set_task_status(task_id, status, remarks="", evidence=None):
    t = db.q("SELECT * FROM tasks WHERE id=?", (task_id,), one=True)
    if not t:
        return False
    now = db.now()
    if status == "In Progress" and not t["started_at"]:
        db.x("UPDATE tasks SET started_at=? WHERE id=?", (now, task_id))
    if status == "Completed":
        db.x("UPDATE tasks SET completed_at=? WHERE id=?", (now, task_id))
    db.x("UPDATE tasks SET status=?, remarks=COALESCE(NULLIF(?,''),remarks), evidence=COALESCE(?,evidence) WHERE id=?",
         (status, remarks, evidence, task_id))
    db.x("UPDATE reports SET status=? WHERE id=?", (status, t["report_id"]))
    rep = db.q("SELECT * FROM reports WHERE id=?", (t["report_id"],), one=True)
    notify(rep["citizen_id"], f"Report #{rep['id']} status: {status}")
    return True

def task_rows(where="1=1", args=()):
    rows = db.q(f"""SELECT t.*, r.type, r.location, r.priority, r.description, r.photo, u.name officer
                    FROM tasks t JOIN reports r ON r.id=t.report_id LEFT JOIN users u ON u.id=t.officer_id
                    WHERE {where} ORDER BY t.id DESC""", args)
    for r in rows:
        r["time_taken"] = None
        if r["completed_at"]:
            a = datetime.strptime(r["assigned_at"], "%Y-%m-%d %H:%M:%S")
            b = datetime.strptime(r["completed_at"], "%Y-%m-%d %H:%M:%S")
            r["time_taken"] = round((b - a).total_seconds() / 60, 1)
    return rows

def lane_names():
    import config
    try:
        return list(json.load(open(config.C.LANES_FILE))["lanes"])[:4]
    except Exception:
        return ["North", "East", "South", "West"]

def get_stats(days=1):
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    v = db.q("SELECT vtype, COUNT(*) n, SUM(fine) f FROM violations WHERE ts>=? GROUP BY vtype", (since,))
    by_type = {r["vtype"]: r["n"] for r in v}
    rs = db.q("SELECT status, COUNT(*) n FROM reports WHERE created_at>=? GROUP BY status", (since,))
    dens = db.q("SELECT AVG(north) n, AVG(east) e, AVG(south) s, AVG(west) w FROM density_log WHERE ts>=?", (since,), one=True)
    avg = {}
    if dens and dens["n"] is not None:
        vals = [dens["n"], dens["e"], dens["s"], dens["w"]]      # DB columns are positional (lane 1..4)
        avg = {n: round(v, 1) for n, v in zip(lane_names(), vals) if v is not None}
    done = task_rows("t.status='Completed' AND t.completed_at>=?", (since,))
    times = [r["time_taken"] for r in done if r["time_taken"] is not None]
    return {
        "since": since,
        "violations_total": sum(by_type.values()),
        "violations_by_type": by_type,
        "top_violation": max(by_type, key=by_type.get) if by_type else None,
        "fines_total": sum((r["f"] or 0) for r in v),
        "reports_total": sum(r["n"] for r in rs),
        "reports_by_status": {r["status"]: r["n"] for r in rs},
        "tasks_completed": len(done),
        "avg_completion_min": round(sum(times) / len(times), 1) if times else None,
        "avg_density": avg,
        "busiest_lane": max(avg, key=avg.get) if avg else None,
        "emergency_events": db.q("SELECT COUNT(*) n FROM emergency_events WHERE ts>=?", (since,), one=True)["n"],
    }

def assistant_context():
    return {"day": get_stats(1), "week": get_stats(7),
            "active_tasks": db.q("SELECT COUNT(*) n FROM tasks WHERE status IN('Assigned','In Progress')", one=True)["n"],
            "recent_violations": db.q("SELECT vtype,lane,ts FROM violations ORDER BY id DESC LIMIT 5")}
