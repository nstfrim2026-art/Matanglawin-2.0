# MatanglaWIN 2.0

Professional UAV structural crack-inspection system for the DJI Neo 2.

MatanglaWIN keeps a **clean live drone point-of-view** for monitoring and
runs **automatic crack segmentation on each captured still photo** — press
the shutter on the controller and the photo is transferred, analyzed with
the trained YOLO11-seg model (`best.pt`), and shown on the dashboard with
no manual upload and no Analyze button.

The web UI is a wide, desktop-first dark dashboard that preserves the
established MatanglaWIN look (dark navy surface, red accent, panelled
cards) rather than a narrow mobile stack.

---

## Two independent pipelines

```
Pipeline A — LIVE DRONE POV (display only)
    DJI Neo 2 → DJI Fly → RTMP → MediaMTX → MatanglaWIN → clean live POV
    (no YOLO, no masks, no boxes, no auto-capture — monitoring only)

Pipeline B — DJI STILL PHOTO INSPECTION (automatic)
    DJI Neo 2 → press PHOTO → automatic transfer → watch folder
    → InspectionService → best.pt / YOLO segmentation
    → CRACK DETECTED / NO CRACK DETECTED + original + red-highlighted
    → automatic website update → inspection history
```

The two are fully independent: the live stream being down never blocks
photo inspection, and running an inspection never interrupts the live POV.

---

## Logo

Place the official logo at:

```
static/img/matanglawin-logo.png
```

The header, favicon, and branding load that PNG automatically and fall
back to the bundled `static/img/matanglawin-logo.svg` placeholder only if
the PNG is missing. The PNG's aspect ratio is preserved (it is scaled by
height in the header). No other logo is generated or substituted.

---

## Live drone POV (Pipeline A) — MediaMTX

1. **Run MediaMTX** (a standalone binary, download from its releases page)
   on this PC. From the running app you can download a ready-made config:

   ```
   http://localhost:5000/api/mediamtx/config    →  mediamtx.yml
   ```

   Then start it with `mediamtx mediamtx.yml`. The config is generated from
   the current network settings, so it always matches your ports/stream key.

2. **Point DJI Fly at this PC.** Open the dashboard — the *Live Feed
   Connection* panel shows the exact values to enter in DJI Fly's
   live-stream (RTMP) setting:

   - **DJI RTMP address**, e.g. `rtmp://<your-PC-IP>:1935`
   - **Stream key**, e.g. `matanglawin`

   These follow the PC's current network automatically (see *Dynamic IP*).

3. The dashboard embeds the MediaMTX **WebRTC** feed directly in the
   browser. The backend never reads or annotates the video — the POV is
   always clean.

---

## Automatic DJI photo inspection (Pipeline B)

When the operator presses the shutter, the DJI Neo 2 saves the original
still. MatanglaWIN then receives and analyzes that **exact original JPEG**
automatically — no browse/select/drag/Upload/Analyze step, ever.

### Honest note on what DJI provides

DJI Fly / the DJI Neo 2 do **not** expose a webhook or a documented local
API that pushes a freshly captured photo into a third-party app. After the
shutter, the original ends up in a phone/PC **album folder**. So the
"phone → Windows" hop is completed by one of the two bridges below. Both
deliver the original bytes and feed the **same** `InspectionService`; both
are de-duplicated so a photo is analyzed only once.

### Option 1 — Companion uploader (`dji_photo_bridge.py`) — direct LAN push

A small stdlib-only watcher that monitors the DJI album folder and POSTs
each new original to MatanglaWIN's `POST /api/import`. Run it on Android
via **Termux** (watch the DJI album, target the PC's LAN address), or on
the PC itself watching a folder that a sync app mirrors from the phone.

```
python dji_photo_bridge.py --server http://<PC-LAN-IP>:5000 --watch-dir "<DJI album folder>"
```

Environment fallbacks: `MATANGLAWIN_BRIDGE_SERVER`,
`MATANGLAWIN_BRIDGE_WATCH_DIR`, `MATANGLAWIN_BRIDGE_INTERVAL`. The PC address
is **configured here, never hardcoded** — change Wi-Fi and just point
`--server` at the PC's current LAN IP (shown on the dashboard). The uploader:

- waits until each file finishes downloading/copying (size stable),
- uploads the **raw original bytes** (no quality loss, no re-encode),
- **retries with backoff** when the PC is temporarily unavailable,
- **never uploads the same photo twice** (SHA-256, persisted across restarts),
- sends a heartbeat so the dashboard shows the bridge as connected,
- requires no manual action after setup — you only press the shutter.

### Option 2 — Folder sync into the watch folder — zero extra code

Point any sync tool at the import folder MatanglaWIN already watches:

```
MATANGLAWIN_IMPORT_DIR = C:\path\to\your\import\folder   (default: matanglawin_data\import)
```

Then mirror the phone's DJI album into that folder with, e.g.:

- **Syncthing** (LAN-only, no cloud — recommended), or
- a cloud client (Google Photos/Drive, OneDrive, Dropbox), or
- DJI Assistant / QuickTransfer download destination, or USB/SD copy.

The watch-folder importer is **partial-copy safe**, **de-duplicates** by
content hash (survives restarts), and **tolerates corrupt files** (skips
them without stopping).

> Prerequisite for both options: in DJI Fly, enable saving/downloading the
> full-resolution **original** photo to the phone album (not just a cache
> thumbnail), so a real original reaches the folder.

Either way, the dashboard's *Latest Inspection* panel updates on its own —
no page refresh, no Analyze button.

### Photo bridge status + logging

The dashboard shows a small **PHOTO BRIDGE** indicator (`READY` /
`WAITING FOR PHOTO` / `OFFLINE`) driven by `GET /api/bridge/status` from the
companion uploader's heartbeat. Both the app and the uploader log the flow:

```
[PHOTO BRIDGE] New DJI photo detected: DJI_0001.JPG
[PHOTO BRIDGE] Upload complete
[MATANGLAWIN] Inspection started
[MATANGLAWIN] CRACK DETECTED (inspection #1)
[PHOTO BRIDGE] PC unavailable - will retry     (on failure)
```

No technical metrics appear in the user-facing inspection result.

---

## What a result shows (and never shows)

Each inspection shows only:

- **CRACK DETECTED** or **NO CRACK DETECTED**
- the **original** untouched photo
- a **red-highlighted** photo (segmentation masks painted red) — only when
  a crack is found

It never shows confidence/score, FPS, IoU, precision/recall, frame counts,
thresholds, bounding boxes, or any crack-only / cropped / separate-mask
image.

---

## Tuning detection sensitivity (thin cracks — no retraining)

High-resolution DJI stills shrink ~6× when analyzed at a small inference
size, so a hairline crack can drop below one pixel and vanish before the
model ever sees it. To recover thin/faint cracks **without retraining**,
analysis runs with recall-tuned defaults and **tiled (sliced) inference**:
the photo is cut into overlapping tiles, each analyzed at near-native
resolution, and the masks are stitched back together. A light,
detection-only contrast boost (CLAHE + unsharp) helps faint cracks stand
out — applied only to the model's input, never to the stored/displayed
original, and never to the live POV.

All knobs are environment variables (no code edits, nothing shown in the
UI):

```
MATANGLAWIN_CONF=0.15          # detection threshold (lower = more sensitive)
MATANGLAWIN_IMGSZ=1280         # whole-image inference size
MATANGLAWIN_MIN_AREA_PX=40     # noise floor (mask pixels)
MATANGLAWIN_TILED=1            # tiled/sliced inference on/off (biggest win)
MATANGLAWIN_TILE=1024          # tile size in px (try 640 for very thin cracks)
MATANGLAWIN_TILE_OVERLAP=0.2   # tile overlap fraction
MATANGLAWIN_ENHANCE=1          # CLAHE+unsharp on the analysis copy only
MATANGLAWIN_AUGMENT=0          # test-time augmentation (extra recall, slower)
```

Measured on the real `best.pt` with a faint ~1px crack in a 1600×2400
image: the old whole-image path (conf 0.25, imgsz 640) detected **0** crack
pixels, and even imgsz 1280 + enhancement still missed it — while tiled +
enhancement recovered it (~13k crack pixels). Tiling was the decisive
factor. Smaller tiles (e.g. `MATANGLAWIN_TILE=640`) recover the thinnest
cracks best, at the cost of more tiles/time per photo.

Trade-offs: lower `conf` and tiling raise recall but can add occasional
false positives on plain wall texture, shadows, or edges — tune `conf` on a
few of your own photos. Tiling costs seconds per photo; that is fine here
because it is **still-photo** analysis and never touches the live POV.
Use the **Manual Upload** page to re-run one photo and compare settings
quickly. If thin cracks are *still* missed after tuning, that is when more
training data (native-resolution hairline-crack crops) becomes the answer.

---

## Dynamic IP (works on any network)

The PC's LAN IP is never hardcoded. On every request the app either uses
your override or auto-detects the active LAN IPv4:

```
MATANGLAWIN_HOST_IP = 172.20.10.3     # optional manual override; wins if set
```

If unset, the current Wi-Fi/Ethernet/hotspot IP is detected automatically,
so everything keeps working when you switch networks. Consumer-side URLs
(RTSP/WebRTC) always use `localhost`; only the DJI-facing RTMP address uses
the LAN IP.

Other optional variables: `MATANGLAWIN_STREAM_KEY`, `MATANGLAWIN_RTMP_PORT`,
`MATANGLAWIN_RTSP_PORT`, `MATANGLAWIN_WEBRTC_PORT`, `MATANGLAWIN_API_PORT`.

---

## Manual upload (fallback / testing only)

The **Manual Upload** page (`/upload`) is a fallback for testing or when no
drone transfer is available. It runs the exact same `InspectionService` as
the automatic path. DJI photos never need it.

---

## Run from source

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# Recommended: CPU-only torch (much smaller than the CUDA build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Linux only, if you hit "libGL.so.1: cannot open shared object file":
#   Debian/Ubuntu: sudo apt-get install -y libgl1

python app.py                       # open http://localhost:5000
```

Optional: `HOST`, `PORT`, `WEIGHTS`, `AUTO_OPEN=1` (auto-open browser).

## One-click executable

```bash
./build_exe.sh        # Linux/macOS
build_exe.bat         # Windows
```

Produces `dist/MatanglaWIN(.exe)`. It bundles the site and `best.pt`;
runtime data (photos, results, DB) is written to a `matanglawin_data/`
folder next to the executable. Build separately per OS (PyInstaller does
not cross-compile).

---

## Tests

```bash
pip install pytest
pytest -q
```

Covers dynamic IP config + override, no hardcoded IPs, no live-video
inference / no live YOLO, automatic photo ingestion (watch folder **and**
`/api/import` bridge), partial-copy safety, duplicate protection across
retries, invalid-file handling, original preservation, red-highlighted
output, CRACK/NO-CRACK status, no confidence/box/crack-only in the
user-facing output, bridge status transitions, manual-upload fallback, and
the app routes.

---

## Real DJI Neo 2 hardware test

Prove that pressing PHOTO makes the actual original arrive and appear in
MatanglaWIN with **no manual upload**.

1. Start MediaMTX (for the live POV) and `python app.py`. Note the PC's LAN
   IP from the dashboard's *Live Feed Connection* panel.
2. In DJI Fly, enable saving the full-resolution **original** photo to the
   phone album, and configure the live-stream (RTMP) values shown on the
   dashboard.
3. Start the bridge:
   - Companion uploader: `python dji_photo_bridge.py --server http://<PC-LAN-IP>:5000 --watch-dir "<DJI album>"` (Termux on the phone, or on the PC watching a synced folder), **or**
   - Folder sync: point Syncthing/cloud at `MATANGLAWIN_IMPORT_DIR`.
   Confirm the dashboard shows **PHOTO BRIDGE: READY / WAITING FOR PHOTO**.
4. Connect the DJI Neo 2, open DJI Fly, confirm the **live POV** shows clean
   video (no boxes/masks).
5. **Isolation check:** stop MediaMTX so there is no live stream, aim at a
   known crack, and press the **PHOTO/shutter button once**. Do not upload
   or copy anything.
6. Confirm, hands-off: the original lands in the watch folder / is POSTed,
   the log shows `[MATANGLAWIN] Inspection started` then `CRACK DETECTED` /
   `NO CRACK DETECTED`, and the dashboard's *Latest Inspection* updates by
   itself with the original + red-highlighted image.
7. **Prove it's the actual DJI still (not a stream frame):**
   - The companion uploader's `POST /api/import` response includes a
     `sha256` of the exact bytes sent. Compare it to the file on the phone/PC:
     `certutil -hashfile "DJI_0001.JPG" SHA256` (Windows) — they must match.
   - Confirm the stored original's resolution equals the DJI capture's
     resolution, and its timestamp/filename correspond to your capture.
   - Because MediaMTX was stopped in step 5, the result provably cannot have
     come from the live stream.
8. Restart MediaMTX; confirm the POV plays again, that merely streaming
   creates **no** inspection, and that pressing PHOTO still produces one
   while the POV keeps playing.
9. Press PHOTO twice, and also re-copy an already-processed file: confirm
   duplicates are **not** re-analyzed. Briefly stop the app mid-capture:
   confirm the uploader retries and the photo still arrives once the app is
   back.

Automated stand-in for this flow (no drone needed): the test suite exercises
the uploader → `/api/import` → analysis → status path, and a byte-identity
check confirms the exact original bytes reach the analysis input.

---

## Files

- `app.py` — Flask app: dashboard, result, inspections, `/api/import`
  bridge ingest, bridge status, manual-upload fallback, JSON/image APIs.
- `inspection_service.py` — the single central analysis pipeline (manual +
  automatic converge here).
- `photo_import.py` — automatic DJI photo watch-folder bridge (Option 2).
- `dji_photo_bridge.py` — companion uploader (Option 1): watch DJI album →
  POST original to `/api/import`. Stdlib only; runs on Termux or the PC.
- `import_ledger.py` — SHA-256 de-dup for `/api/import` (retry-safe).
- `bridge_status.py` — PHOTO BRIDGE liveness (READY/WAITING/OFFLINE).
- `detector.py` / `inference_core.py` — YOLO11-seg segmentation + red
  overlay drawing (shared, no duplication). Includes tiled/sliced inference
  and detection-only contrast enhancement for thin-crack recall (see
  "Tuning detection sensitivity").
- `inspection_db.py` — SQLite inspection history.
- `network_config.py` — dynamic IP + MediaMTX config generation.
- `infer_overlay.py` — original standalone CLI (unchanged behavior).
- `templates/`, `static/` — the desktop-first dark UI.
- `tests/` — the full test suite.
- `matanglawin.spec`, `build_exe.sh`, `build_exe.bat` — one-click build.
