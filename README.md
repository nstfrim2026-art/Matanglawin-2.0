# MATANGLAWIN — Real-Time AI-Assisted Drone Visual Crack Detection System

A real-time monitoring dashboard that receives a live drone video feed
via RTMP/MediaMTX, runs every frame through the trained YOLO11-seg
model (`best.pt`), and highlights visible cracks directly on the live
video with a semi-transparent red overlay. It only detects and visually
highlights cracks; it does not perform any structural analysis.

When a crack is visible, the dashboard shows a blinking red alert banner
and plays an alert sound once (not continuously). When no crack is
visible, it simply shows "Monitoring...".

## Important: this is a local monitoring tool, not a public website

The app receives drone video via RTMP on the local network and runs
AI inference server-side. **It must run on the same machine as MediaMTX**
(or a machine with network access to the camera source). Run it on the
operator's laptop or a local server in the field/control room, and open
it from a browser on that same machine or local network.

## Architecture

```
DJI NEO 2
    |
    | DJI Fly (RTMP publish)
    v
MediaMTX (rtmp://<PC-IP>:1935/matanglawin)
    |
    +-- RTSP (rtsp://localhost:8554/matanglawin) --> YOLO11-seg --> Crack Detection
    |                                                                  |
    |                                                                  v
    |                                                         Auto Capture + Inspection DB
    |                                                                  |
    |                                                                  v
    |                                                            PDF Report
    |
    +-- WebRTC (http://localhost:8889/matanglawin) --> Browser (Live Drone POV)
```

**Key design principle:** The PC's LAN IP is auto-detected at runtime.
No source code changes are needed when switching networks.

### Module responsibilities

- `network_config.py` — dynamic LAN IP detection, RTMP address generation,
  MediaMTX reachability checks. Removes the need for hardcoded IP addresses.
- `video_source.py` — configurable camera abstraction (webcam / IP Camera
  / drone via MediaMTX RTSP), all opened the same way via OpenCV.
- `inference_core.py` — the YOLO11-seg model + mask-overlay drawing
  logic, shared by the live dashboard, the original CLI script
  (`infer_overlay.py`), and `detector.py`.
- `detector.py` — a background worker thread that continuously pulls
  frames from the selected video source, runs them through
  `inference_core.annotate_frame`, and keeps the latest annotated JPEG
  + status (`camera_connected`, `crack_present`) available for Flask to
  serve, without blocking the video loop.
- `capture_manager.py` — automatic crack capture state machine with
  temporal verification, cooldown, and cropped image extraction.
- `gps_provider.py` — GPS data association for inspection records (when
  available from DJI flight logs or other sources).
- `inspection_db.py` — SQLite database for storing inspection records.
- `report_generator.py` — PDF and CSV report generation with GPS data.
- `app.py` — Flask routes (dashboard page, MJPEG video stream,
  status polling, network API, inspection CRUD, health check).
- `templates/dashboard.html`, `static/` — the dashboard UI (HTML/CSS/
  vanilla JS only, no frameworks).

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

Open http://localhost:5000 -- you will see the loading screen, then the
live dashboard with crack highlighting.

### Camera sources (MediaMTX architecture)

The primary production workflow uses the DJI Neo 2 drone streaming via
RTMP to a local MediaMTX server:

1. **DJI Neo 2 via MediaMTX** (production) -- DJI Fly publishes RTMP to
   `rtmp://<PC-IP>:1935/matanglawin`. MediaMTX converts this to RTSP
   for the AI backend (`rtsp://localhost:8554/matanglawin`) and WebRTC
   for the browser dashboard.
2. **Webcam** -- your machine's default local camera (device index 0),
   useful for development and testing.
3. **IP Camera** -- a phone or network camera serving an MJPEG/RTSP stream.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATANGLAWIN_HOST_IP` | auto-detect | Override the auto-detected LAN/Wi-Fi IPv4 address. Set this if auto-detection picks the wrong interface. |
| `MATANGLAWIN_STREAM_KEY` | `matanglawin` | Stream key/path used by MediaMTX and DJI Fly. |
| `MEDIAMTX_RTSP_URL` | `rtsp://localhost:8554/matanglawin` | Full RTSP URL override for the AI backend input. |
| `MEDIAMTX_URL` | `http://localhost:8889` | MediaMTX WebRTC player base URL. |
| `IP_CAMERA_URL` | (empty) | URL for an IP camera source (e.g., `http://192.168.1.50:4747/video`). |
| `WEIGHTS` | `best.pt` | Path to the YOLO model weights file. |
| `HOST` | `127.0.0.1` | Host to bind the Flask server to. |
| `PORT` | `5000` | Preferred port (a free one is chosen automatically if busy). |
| `CONF` | `0.40` | Detection confidence threshold. |
| `TARGET_FPS` | `8` | Cap on the inference loop rate. |
| `DEFAULT_SOURCE` | `webcam` | Default camera source: `webcam`, `ip_camera`, or `drone`. |
| `AUTO_OPEN` | `0` | Set to `1` to auto-open the browser on startup in dev mode. |

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
only). If you're only using a network camera source (IP Camera/drone
URL), you can drop that flag. See [DEPLOY.md](DEPLOY.md) for more detail.

## Files

- `app.py` -- Flask routes (dashboard, `/video_feed`, `/status`, `/api/network`, `/sources`, `/health`, inspection CRUD)
- `network_config.py` -- dynamic LAN IP detection, RTMP address generation, MediaMTX reachability
- `detector.py` -- background continuous-inference worker
- `video_source.py` -- configurable camera source abstraction
- `inference_core.py` -- shared YOLO11-seg inference + overlay-drawing logic
- `capture_manager.py` -- automatic crack capture with temporal verification and cropping
- `gps_provider.py` -- GPS data association for inspection records
- `inspection_db.py` -- SQLite database for inspection storage
- `report_generator.py` -- PDF and CSV report generation
- `image_quality.py` -- frame quality validation (brightness, blur, exposure)
- `crack_validator.py` -- crack detection validation filters (DJI UI zone, aspect ratio, area)
- `infer_overlay.py` -- original standalone single-image CLI script (unchanged)
- `templates/dashboard.html`, `templates/inspections.html` -- dashboard and inspections UI
- `static/` -- CSS, JS, images, alert sounds
- `matanglawin.spec`, `build_exe.sh` / `build_exe.bat` -- one-click executable build
- `Dockerfile` -- containerized deployment
- `tests/test_integration.py` -- comprehensive integration test suite

## Hardware Test Procedure

The following procedure documents how to test the system with actual DJI Neo 2 hardware.
This has **not** been verified with real hardware yet -- it documents the intended workflow.

### Prerequisites

- DJI Neo 2 drone with DJI Fly app configured
- MediaMTX installed and running on the PC
- MatanglaWIN application running
- PC connected to a Wi-Fi/LAN network accessible from DJI Fly

### Steps

1. **Connect PC to Wi-Fi** -- ensure the PC has a valid LAN IP address.

2. **Start MediaMTX** -- run the MediaMTX binary on the PC. It should listen on
   ports 1935 (RTMP), 8554 (RTSP), and 8889 (WebRTC).

3. **Start MatanglaWIN** -- run `python app.py`. Open the dashboard at
   `http://localhost:5000`.

4. **Check displayed PC IP** -- the Network Status panel on the dashboard shows
   the current PC IP and the DJI RTMP address. Example: `172.20.10.3`.

5. **Configure DJI Fly** -- in the DJI Fly app, set the RTMP live streaming
   destination:
   - RTMP Address: `rtmp://<DISPLAYED-PC-IP>:1935`
   - Stream Key: `matanglawin`

6. **Start DJI live transmission** -- begin the live stream from DJI Fly.

7. **Verify MediaMTX** -- check that MediaMTX reports the stream is available
   and online. The dashboard should show stream status as LIVE.

8. **Verify live drone POV** -- the dashboard should display the live video
   from the drone via WebRTC.

9. **Verify YOLO receives frames** -- the AI backend receives frames from
   `rtsp://localhost:8554/matanglawin` and processes them with YOLO11-seg.

10. **Test crack detection** -- fly/point the drone camera toward a surface
    with cracks. Verify detection triggers (alert banner, confidence display).

11. **Verify auto capture** -- after sustained crack detection (5 frames over
    1.5 seconds), an automatic cropped crack image should be captured.

12. **Verify inspection record** -- check the Inspections page for the new
    record with timestamp, confidence, and image.

13. **Verify GPS** -- if GPS data is available (from DJI flight log import),
    verify coordinates appear in the inspection record.

14. **Generate PDF** -- export a PDF report and verify it includes all
    inspection data and the cropped crack image.

### IP Change Test

1. Disconnect from the current network.
2. Connect to a different Wi-Fi network.
3. Verify the dashboard automatically shows the new PC IP and updated
   DJI RTMP address (no code changes required).
4. Update only the RTMP address in DJI Fly to match the new IP.
5. Stream key remains `matanglawin`.

