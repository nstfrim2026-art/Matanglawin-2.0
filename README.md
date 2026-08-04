# MATANGLAWIN — Real-Time AI-Assisted Drone Visual Crack Detection System

A real-time monitoring dashboard that continuously reads a live camera
feed (laptop webcam, a phone running DroidCam, or in the future a
drone's video downlink), runs every frame through the trained YOLO11-seg
model (`best.pt`), and highlights visible cracks directly on the live
video with a semi-transparent red overlay — no bounding boxes, no
measurements, no severity scores. It only detects and visually
highlights cracks; it does not perform any structural analysis.

When a crack is visible, the dashboard shows a blinking red alert banner
and plays an alert sound once (not continuously). When no crack is
visible, it simply shows "Monitoring...".

## Important: this is a local monitoring tool, not a public website

Because the app reads a camera directly (via OpenCV) rather than through
the visitor's browser, **it must run on a machine that has network
access to the camera** — your own laptop's webcam, or a phone/drone on
the same local network. It is not meant to be deployed as a public cloud
demo that lets arbitrary internet visitors see your camera. Run it on
the operator's laptop or a local server in the field/control room, and
open it from a browser on that same machine or local network.

## Architecture

Kept intentionally separated per concern, so the video source can be
swapped without ever touching the detection pipeline:

- `video_source.py` — configurable camera abstraction (webcam / DroidCam
  / future drone feed), all opened the same way via OpenCV.
- `inference_core.py` — the YOLO11-seg model + mask-overlay drawing
  logic, shared by the live dashboard, the original CLI script
  (`infer_overlay.py`), and `detector.py`.
- `detector.py` — a background worker thread that continuously pulls
  frames from the selected video source, runs them through
  `inference_core.annotate_frame`, and keeps the latest annotated JPEG
  + status (`camera_connected`, `crack_present`) available for Flask to
  serve, without blocking the video loop.
- `app.py` — Flask routes only (dashboard page, MJPEG video stream,
  status polling, camera-source switching, health check).
- `templates/dashboard.html`, `static/` — the dashboard UI (HTML/CSS/
  vanilla JS only — no frameworks).

## Run it

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# Recommended: install the CPU-only build of torch first (much smaller
# than the default CUDA build):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

# Linux only, if you hit "libGL.so.1: cannot open shared object file":
#   Debian/Ubuntu: sudo apt-get install -y libgl1
#   Fedora/Amazon Linux: sudo dnf install -y mesa-libGL

python app.py
```

Open http://localhost:5000 — you'll see the loading screen, then the
live dashboard with your webcam feed and crack highlighting.

### Camera sources

Use the **Camera Source** selector at the bottom of the dashboard to
switch between:

- **Webcam** — your machine's default local camera (device index 0)
- **DroidCam** — a phone running the DroidCam app, streamed over your
  local Wi-Fi (default MJPEG URL: `http://<phone-ip>:4747/video`)
- **Drone Camera** — a placeholder slot for the future drone video feed

Configure the network camera URLs with environment variables before
starting the app:

```bash
export DROIDCAM_URL="http://192.168.1.50:4747/video"
export DRONE_URL="http://192.168.1.60:8080/video"
python app.py
```

### Other environment variables

- `WEIGHTS` — path to a different `.pt` weights file (default: `best.pt`)
- `HOST` — host to bind to (default: `127.0.0.1`)
- `PORT` — preferred port; a free one is chosen automatically if busy (default: `5000`)
- `CONF` — detection confidence threshold (default: `0.25`)
- `TARGET_FPS` — cap on the inference loop rate (default: `8`)
- `DEFAULT_SOURCE` — `webcam` | `droidcam` | `drone` (default: `webcam`)
- `AUTO_OPEN` — set to `1` to auto-open the browser on startup in dev mode

## One-click executable (no Python required to *run* it)

Bundles Python, the dashboard, and `best.pt` into a single executable
for the operator's machine — no install needed to run it, only to build it.

```bash
./build_exe.sh      # Linux/macOS
build_exe.bat        # Windows
```

This creates `dist/Matanglawin` (or `.exe`), ~300-350 MB. Double-click it
— it opens a console and your browser automatically. Build separately
per OS (PyInstaller does not cross-compile).

## Running in Docker (same machine/local network as the camera)

```bash
docker build -t matanglawin .
docker run -p 5000:5000 --device=/dev/video0 matanglawin
# open http://localhost:5000
```

`--device=/dev/video0` passes the host's webcam into the container (Linux
only). If you're only using a network camera source (DroidCam/drone
URL), you can drop that flag. See [DEPLOY.md](DEPLOY.md) for more detail.

## Files

- `app.py` — Flask routes (dashboard, `/video_feed`, `/status`, `/set_source`, `/sources`, `/health`)
- `detector.py` — background continuous-inference worker
- `video_source.py` — configurable camera source abstraction
- `inference_core.py` — shared YOLO11-seg inference + overlay-drawing logic
- `infer_overlay.py` — original standalone single-image CLI script (unchanged)
- `templates/dashboard.html`, `static/` — dashboard UI (HTML/CSS/vanilla JS, logo, alert sound)
- `matanglawin.spec`, `build_exe.sh` / `build_exe.bat` — one-click executable build
- `Dockerfile` — containerized deployment
