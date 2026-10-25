AI TRAFFIC MANAGEMENT SYSTEM - VIDEO UPLOAD DASHBOARD

This version matches the supplied reference screenshot.

MAIN FLOW
1. Login as admin.
2. Open /admin.
3. Choose one traffic video in the left Upload Video panel.
4. Click Upload & start.
5. YOLO + ByteTrack processes the uploaded video.
6. The dashboard displays four lane panels: North, East, West, South.
7. Each panel receives a lane-specific crop from the same uploaded video using zones/lanes.json.
8. Counts, density, signal, green time, vehicle classes and emergency status update every second.

IMPORTANT
- The four panels are not four independent cameras. They are four configured lane regions from one uploaded video.
- Edit zones/lanes.json or use the existing lane editor to match your actual intersection.
- For four independent cameras later, create four TrafficEngine workers.

RUN
python app.py

Open:
http://127.0.0.1:5000/admin

Default user credentials depend on the users table in data/traffic.db.
Do not overwrite the database if you already have users/reports/tasks.
