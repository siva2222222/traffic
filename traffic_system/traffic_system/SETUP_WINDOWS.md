# Setup — Windows + Python 3.11

## What you download, and from where

| What | Size | How you get it |
|---|---|---|
| `traffic_system.zip` | 50 MB | the file I gave you — extract it |
| Python 3.11 | 25 MB | <https://www.python.org/downloads/release/python-3119/> — only if `py -3.11` fails |
| Python packages | ~1.5 GB | `pip install -r requirements.txt` (automatic) |
| CLIP weights | ~350 MB | downloads itself on first detection, then cached offline |
| YOLO weights | 6.5 MB | **already inside the zip** (`yolov8n.pt`) |
| Siren model | 0.3 MB | **already inside the zip** (`models/siren_cnn.pt`) |
| ffmpeg *(optional)* | 100 MB | <https://www.gyan.dev/ffmpeg/builds/> — only for siren audio on mp4 |

So: **one zip, plus Python 3.11 if you don't have it.** Everything else installs itself.

---

## Fastest path

Extract the zip, open the `traffic_system` folder, double-click:

```
setup.bat          <- once, takes 10-20 minutes
run.bat            <- every time after that
```

Then open <http://127.0.0.1:5000/admin/image>.

---

## Or do it by hand

Open **Command Prompt** in the `traffic_system` folder (type `cmd` in the address bar of
File Explorer and press Enter).

```bat
py -3.11 --version
```

Should print `Python 3.11.x`. If it says it's not recognised, install Python 3.11 from the link
above and tick **"Add python.exe to PATH"** during install.

```bat
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

That last line is the slow one — about 1.5 GB, mostly `torch`. Leave it running.

Check everything landed:

```bat
python check_setup.py
```

It prints a line per dependency and, if something is missing, the exact `pip install` command to
fix it. Do not move on until it says **ALL GOOD**.

Pre-download the CLIP weights so your first scan isn't slow:

```bat
python check_setup.py --download
```

Start it:

```bat
python app.py
```

Open <http://127.0.0.1:5000/admin/image>.

---

## Every time after the first

```bat
cd traffic_system
venv\Scripts\activate
python app.py
```

Or just double-click `run.bat`. The `activate` step is the one people forget — without it
Python can't see any of the installed packages.

---

## Testing without the dashboard

```bat
venv\Scripts\activate
python image_analyze.py ambulance.jpg --verbose
python video_analyze.py clip.mp4
python check_emergency.py clip.mp4
```

---

## When it breaks

| Error | Fix |
|---|---|
| `'py' is not recognized` | Python isn't on PATH — reinstall Python 3.11 with "Add python.exe to PATH" ticked |
| `ModuleNotFoundError: flask` (or any module) | you forgot `venv\Scripts\activate` |
| `error: Microsoft Visual C++ 14.0 is required` (building `lap`) | `pip install lapx` — prebuilt, no compiler needed |
| `CLIP: none` on the page banner | `pip install transformers`, then restart `app.py` |
| Hangs on first detection | that's the 350 MB CLIP download — wait it out once |
| `Cannot open video` | path has spaces: wrap in quotes, e.g. `python video_analyze.py "my clip.mp4"` |
| `Port 5000 in use` | `set PORT=5001` then `python app.py`, or close whatever has 5000 |
| Video scan very slow | expected on CPU. Set `PROCESS_EVERY=3` in `config.py` |
| `ffmpeg not found` | only affects siren audio on mp4. Vision detection is unaffected — ignore it |

If `pip install -r requirements.txt` fails partway, re-running it is safe; it skips what's
already installed.

---

## Disk and time

- Download: ~1.9 GB total (packages + CLIP weights)
- Disk after install: ~4 GB
- One-time setup: 10–20 minutes on a normal connection
- After that: startup is a few seconds, and nothing downloads again
