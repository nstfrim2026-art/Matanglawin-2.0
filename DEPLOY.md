# Running MATANGLAWIN where the camera is

MATANGLAWIN reads a live camera feed directly on the server side (via
OpenCV), then streams the annotated video to the browser. That means it
must run **on or near the camera** — a public cloud host has no way to
see a webcam sitting on your desk or a phone on your Wi-Fi network. This
is a local/field monitoring tool, not a public demo website.

There are two practical ways to run it:

## Option A — Run directly on the operator's laptop (simplest)

This is the normal way to use it day-to-day.

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` on that same laptop. The webcam source
uses the laptop's built-in/USB camera. To use a phone camera instead,
install [DroidCam](https://www.dev47apps.com/) on the phone, connect it
to the same Wi-Fi as the laptop, and set:

```bash
export DROIDCAM_URL="http://<phone-ip>:4747/video"
python app.py
```

then select **DroidCam** in the dashboard's camera selector.

## Option B — One-click executable (no Python needed to run it)

Build once with `./build_exe.sh` (or `build_exe.bat` on Windows), then
hand the resulting `dist/Matanglawin` (or `.exe`) file to the operator.
They double-click it; no Python install required on their machine. See
the main [README.md](README.md) for details.

## Option C — Docker, on the same machine/network as the camera

Still local — Docker here is just for consistent deployment (e.g. onto
a small field PC or edge box that sits next to the drone ground station),
not for public internet hosting.

```bash
docker build -t matanglawin .
docker run -p 5000:5000 --device=/dev/video0 matanglawin
```

- `--device=/dev/video0` passes the host's local webcam into the
  container (Linux only; omit it if you're only using a network camera
  source like DroidCam or the future drone feed).
- Open `http://localhost:5000` from a browser on that machine, or from
  another device on the same local network using the host machine's LAN
  IP (e.g. `http://192.168.1.20:5000`).

## Why not host it on Hugging Face Spaces / a public cloud server?

A cloud server has no route to your webcam, your phone's DroidCam
stream, or a drone's local video downlink — those only exist on your
local network. Public hosting would only make sense if the app received
video *from* the browser (e.g. WebRTC) instead of opening the camera
itself server-side. That's a different architecture than what was
requested here (a continuous server-side OpenCV/YOLO pipeline), so for
now this app is designed to run locally, next to the camera.
