import sqlite3, os
from datetime import datetime
from werkzeug.security import generate_password_hash
from config import C

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def conn():
    os.makedirs(os.path.dirname(C.DB_PATH), exist_ok=True)
    c = sqlite3.connect(C.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c

def q(sql, args=(), one=False):
    c = conn()
    try:
        rows = [dict(r) for r in c.execute(sql, args).fetchall()]
    finally:
        c.close()
    return (rows[0] if rows else None) if one else rows

def x(sql, args=()):
    c = conn()
    try:
        cur = c.execute(sql, args)
        c.commit()
        return cur.lastrowid
    finally:
        c.close()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, username TEXT UNIQUE, password_hash TEXT,
  role TEXT, phone TEXT, lat REAL, lng REAL);
CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY, citizen_id INTEGER, type TEXT, description TEXT,
  location TEXT, lat REAL, lng REAL, photo TEXT, priority TEXT, ai_summary TEXT, ai_action TEXT, ai_key_info TEXT,
  status TEXT DEFAULT 'Pending', created_at TEXT);
CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, task_code TEXT, report_id INTEGER, officer_id INTEGER,
  assigned_at TEXT, started_at TEXT, completed_at TEXT, status TEXT, remarks TEXT, evidence TEXT, assigned_by TEXT);
CREATE TABLE IF NOT EXISTS violations(id INTEGER PRIMARY KEY, vtype TEXT, track_id INTEGER, vehicle_class TEXT,
  lane TEXT, speed_kmh REAL, fine INTEGER, evidence TEXT, ts TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS density_log(id INTEGER PRIMARY KEY, ts TEXT, north REAL, east REAL, south REAL, west REAL);
CREATE TABLE IF NOT EXISTS vehicle_counts(id INTEGER PRIMARY KEY, ts TEXT, lane TEXT, vclass TEXT, count INTEGER);
CREATE TABLE IF NOT EXISTS signal_log(id INTEGER PRIMARY KEY, ts TEXT, lane TEXT, state TEXT, duration REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS emergency_events(id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, lane TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS detection_logs(id INTEGER PRIMARY KEY, ts TEXT, track_id INTEGER, vclass TEXT, lane TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS media_files(id INTEGER PRIMARY KEY, kind TEXT, filename TEXT, original TEXT, uploaded_by INTEGER, ts TEXT);
CREATE TABLE IF NOT EXISTS notifications(id INTEGER PRIMARY KEY, user_id INTEGER, msg TEXT, created_at TEXT, is_read INTEGER DEFAULT 0);
"""

def init_db():
    c = conn()
    c.executescript(SCHEMA)
    try:                                                   # upgrade older databases
        c.execute("ALTER TABLE reports ADD COLUMN ai_key_info TEXT")
    except sqlite3.OperationalError:
        pass
    c.commit()
    c.close()
    if not q("SELECT id FROM users LIMIT 1"):
        seed = [("Admin", "admin", "admin123", "admin", None, None),
                ("Officer Ravi", "police1", "police123", "police", 12.9716, 77.5946),
                ("Officer Meena", "police2", "police123", "police", 12.9352, 77.6245),
                ("Officer Arun", "police3", "police123", "police", 13.0358, 77.5970),
                ("Demo Citizen", "citizen", "citizen123", "citizen", None, None)]
        for n, u, p, r, la, lo in seed:
            x("INSERT INTO users(name,username,password_hash,role,lat,lng) VALUES(?,?,?,?,?,?)",
              (n, u, generate_password_hash(p), r, la, lo))
