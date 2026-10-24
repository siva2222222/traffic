# AI Traffic Management System — Picture-Based Video Upload Dashboard

## Main dashboard
The admin dashboard is designed to match the supplied reference image:
- Left Upload Video sidebar
- Uploaded MP4 as the primary source
- Siren Detection (MFCC Audio AI)
- Start Analysis / Stop
- Signal Phase Guide
- North, East, West, South lane panels in a 2x2 grid
- YOLO vehicle boxes and ByteTrack IDs
- Vehicle count, density, signal and green time
- Emergency/siren alert

## Video architecture
One uploaded video is processed by one YOLO + ByteTrack engine. The four lane feeds are generated from the configured lane polygons in `zones/lanes.json` and show the corresponding crop of the processed frame.

## Run
```powershell
cd traffic_system
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000/admin`.

Upload an MP4 from the left sidebar and press Upload & start.

## Lane configuration
Edit `zones/lanes.json` to change the North/East/West/South polygons for your uploaded camera/video view. The lane video endpoints are:
- `/video_feed/north`
- `/video_feed/east`
- `/video_feed/west`
- `/video_feed/south`

## Notes
The ZIP intentionally does not contain the old large sample videos. Upload your own video through the dashboard. The YOLO model and siren model are included.
