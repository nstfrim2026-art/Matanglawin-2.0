# Running MATANGLAWIN where the camera is

MATANGLAWIN receives drone video via RTMP (through MediaMTX) and runs
AI inference server-side. It then streams the annotated video to the
browser. That means it must run **on the same machine as MediaMTX** or
on a machine with network access to the camera source. This is a local/
field monitoring tool, not a public demo website.

There are two practical ways to run it:

## Option A -- Run directly on the operator's laptop (simplest)

This is the normal way to use it day-to-day.

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` on that same laptop.

### DJI Neo 2 via MediaMTX (production workflow)

1. Install and start MediaMTX on the same PC.
2. Start MatanglaWIN (`python app.py`).
3. Note the PC IP displayed in the Network Status panel.
4. In DJI Fly, set RTMP address: `rtmp://<PC-IP>:1935` and stream key: `matanglawin`.
5. Start the live stream from DJI Fly.

The AI backend receives the stream from `rtsp://localhost:8554/matanglawin`
and the browser receives the live POV from `http://localhost:8889/matanglawin`
via WebRTC.

### IP Camera or webcam

To use a phone or network camera instead, set the IP camera URL:

```bash
export IP_CAMERA_URL="http://<camera-ip>:4747/video"
python app.py
```

Then select **IP Camera** in the dashboard.

## Option B -- One-click executable (no Python needed to run it)

Build once with `./build_exe.sh` (or `build_exe.bat` on Windows), then
hand the resulting `dist/Matanglawin` (or `.exe`) file to the operator.
They double-click it; no Python install required on their machine. See
the main [README.md](README.md) for details.

## Option C -- Docker, on the same machine/network as the camera

Still local -- Docker here is just for consistent deployment (e.g. onto
a small field PC or edge box that sits next to the drone ground station),
not for public internet hosting.

```bash
docker build -t matanglawin .
docker run -p 5000:5000 --device=/dev/video0 matanglawin
```

- `--device=/dev/video0` passes the host's local webcam into the
  container (Linux only; omit it if you are only using a network camera
  source or MediaMTX RTMP).
- Open `http://localhost:5000` from a browser on that machine, or from
  another device on the same local network using the host machine's LAN
  IP (e.g. `http://<lan-ip>:5000`).

## Why not host it on Hugging Face Spaces / a public cloud server?

A cloud server has no route to your webcam, your phone's IP camera
stream, or a drone's local video downlink. Those only exist on your
local network. Public hosting would only make sense if the app received
video *from* the browser (e.g. WebRTC ingest) instead of opening the
camera itself server-side. That is a different architecture than what was
requested here (a continuous server-side OpenCV/YOLO pipeline), so for
now this app is designed to run locally, next to the camera.
