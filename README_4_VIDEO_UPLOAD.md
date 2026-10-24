# AI Traffic Management System - 4 Independent Video Upload Dashboard

This version implements the requested architecture:

- North: independent uploaded video + YOLO + ByteTrack engine
- East: independent uploaded video + YOLO + ByteTrack engine
- West: independent uploaded video + YOLO + ByteTrack engine
- South: independent uploaded video + YOLO + ByteTrack engine

Uploading one video does NOT replace another lane's video.

## Dashboard
Open:
http://127.0.0.1:5000/admin

Each lane has its own:
- Upload button
- Stop button
- Live video feed
- Vehicle count
- Density
- Signal state
- Green time
- Source filename

## Backend routes
POST /admin/source/north
POST /admin/source/east
POST /admin/source/west
POST /admin/source/south

GET /video_feed/north
GET /video_feed/east
GET /video_feed/west
GET /video_feed/south

GET /api/live
GET /api/live/north
GET /api/live/east
GET /api/live/west
GET /api/live/south

Each engine creates its own TrafficEngine instance and therefore has an independent ByteTrack state and video processing thread.

## Run
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python app.py

## Note on hardware
Four simultaneous YOLO engines are more demanding than one engine. A GPU is recommended. On CPU, reduce C.PROCESS_EVERY or WORK_WIDTH in config.py if processing is slow.
