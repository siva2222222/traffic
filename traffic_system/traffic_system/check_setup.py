"""Verify the install before you waste time on a video. Run:  python check_setup.py

Checks every piece the emergency-vehicle detection depends on and tells you exactly which
command fixes whatever is missing. Nothing here downloads a large model except the optional
--download flag, which pre-fetches CLIP so your first real scan isn't slow.
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "
problems = []


def line(tag, what, detail=""):
    print(f"[{tag}] {what}" + (f"  -  {detail}" if detail else ""))


def check_python():
    v = sys.version_info
    if v[:2] == (3, 11):
        line(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    elif (3, 9) <= v[:2] <= (3, 12):
        line(WARN, f"Python {v.major}.{v.minor}", "3.11 is the tested version, but this should work")
    else:
        line(BAD, f"Python {v.major}.{v.minor}", "use Python 3.11")
        problems.append("Install Python 3.11 and recreate the venv")


def check_pkg(mod, label, fix, critical=True):
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "")
        line(OK, label, ver)
        return True
    except Exception as e:                                   # noqa: BLE001
        line(BAD if critical else WARN, label, str(e)[:60])
        problems.append(fix)
        return False


def check_file(rel, label, fix, critical=True):
    p = os.path.join(BASE, rel)
    if os.path.exists(p):
        n = os.path.getsize(p)
        line(OK, label, f"{n / 1e6:.1f} MB" if n > 1e6 else f"{n / 1e3:.0f} KB")
        return True
    line(BAD if critical else WARN, label, "missing")
    if fix:
        problems.append(fix)
    return False


def main():
    print("\n=== Core ===")
    check_python()
    check_pkg("flask", "flask", "pip install flask")
    check_pkg("cv2", "opencv-python", "pip install opencv-python")
    check_pkg("numpy", "numpy", "pip install numpy")
    check_pkg("torch", "torch", "pip install torch")
    check_pkg("ultralytics", "ultralytics (YOLOv8)", "pip install ultralytics")
    try:
        import lap            # noqa: F401
        line(OK, "lap (ByteTrack)")
    except Exception:         # noqa: BLE001
        try:
            import lapx       # noqa: F401
            line(OK, "lapx (ByteTrack)")
        except Exception:     # noqa: BLE001
            line(BAD, "lap / lapx (ByteTrack)", "missing")
            problems.append("pip install lapx")

    print("\n=== Emergency vehicle detection ===")
    check_pkg("transformers", "transformers (CLIP zero-shot)", "pip install transformers")
    check_pkg("PIL", "pillow", "pip install pillow")
    check_pkg("easyocr", "easyocr (text fallback)", "pip install easyocr", critical=False)

    try:
        import emergency_vision as EV
        name = EV.backend_name()
        if name != "none":
            line(OK, "CLIP backend loaded", name)
        else:
            line(BAD, "CLIP backend", "not loaded - emergency detection will NOT work")
            problems.append("pip install transformers")
    except Exception as e:                                   # noqa: BLE001
        line(BAD, "emergency_vision import", str(e)[:70])

    print("\n=== Model files ===")
    check_file("yolov8n.pt", "yolov8n.pt (vehicle detector)",
               "Delete nothing - ultralytics re-downloads it automatically on first run", False)
    check_file("models/siren_cnn.pt", "models/siren_cnn.pt (siren audio)", "", False)
    if not check_file("models/emergency_cnn.pt", "models/emergency_cnn.pt (optional trained CNN)",
                      "", False):
        line(WARN, "", "optional - CLIP covers this. Train later with train_emergency.py")

    print("\n=== Project files ===")
    for f in ("app.py", "engine.py", "image_analyze.py", "video_analyze.py",
              "emergency_vision.py", "config.py", "data/traffic.db"):
        check_file(f, f, f"{f} is missing - re-extract the zip")

    print("\n=== Optional ===")
    from shutil import which
    if which("ffmpeg"):
        line(OK, "ffmpeg", "siren detection works on mp4/mp3")
    else:
        line(WARN, "ffmpeg", "not on PATH - siren audio only works on .wav. Vision is unaffected")

    print("\n" + "=" * 62)
    if problems:
        print("NOT READY. Fix these, then run this check again:\n")
        for p in dict.fromkeys(problems):
            print("   " + p)
    else:
        print("ALL GOOD. Start the app with:   python app.py")
        print("Then open:  http://127.0.0.1:5000/admin/image")
    print("=" * 62 + "\n")

    if "--download" in sys.argv:
        print("Pre-downloading CLIP weights (~350 MB, one time)...")
        try:
            import emergency_vision as EV
            import numpy as np
            EV.classify_crop(np.zeros((200, 200, 3), dtype="uint8"))
            print("Done -", EV.backend_name())
        except Exception as e:                               # noqa: BLE001
            print("Download failed:", e)


if __name__ == "__main__":
    main()
