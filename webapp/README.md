# Matanglawin Website

A small full-stack website around the Matanglawin YOLO11-seg crack detection
model (`best.pt`). Upload a photo in the browser and get back the same
"exact contour" overlay that `infer_overlay.py` produces locally — a
semi-transparent fill plus a crisp outline following the true crack shape,
with a live count of detected crack instances.

## Structure

```
webapp/
├── backend/
│   └── main.py        # FastAPI app: /api/detect, /api/health, serves frontend
├── frontend/
│   ├── index.html      # Upload UI, controls, result display
│   ├── style.css
│   └── app.js           # Fetches /api/detect, renders the result
├── requirements.txt
├── setup.sh             # Creates a venv and installs deps (handles opencv/libGL)
└── run_server.py        # Launches uvicorn on 0.0.0.0:8000
```

## Setup

```bash
cd webapp
./setup.sh
```

This creates `../webapp_venv`, installs `requirements.txt`, and swaps the
GUI `opencv-python` (pulled in by `ultralytics`) for `opencv-python-headless`
so it runs on headless servers without `libGL`.

## Run

```bash
cd webapp
../webapp_venv/bin/python run_server.py
```

Then open **http://localhost:8000** in a browser. The backend loads
`../best.pt` (the repo's trained weights) once at startup.

## API

- `GET /api/health` — `{"status": "ok", "weights_found": true}`
- `POST /api/detect` — multipart form fields:
  - `file` (required): the image
  - `conf` (default 0.25): confidence threshold
  - `imgsz` (default 640): inference size
  - `alpha` (default 0.5): overlay opacity
  - `color` (default `255,0,0`, R,G,B): fill/outline color
  - `outline` (default 2): outline thickness in px, `0` disables

  Returns `{"detections": N, "width": W, "height": H, "image": "data:image/png;base64,..."}`.

## Notes

- The detection logic mirrors `infer_overlay.py`: masks from all detected
  instances are unioned into a single binary mask, then rendered as a
  semi-transparent fill (`cv2.addWeighted`) plus `cv2.drawContours` on the
  true contour — no bounding boxes.
- The model is loaded once and cached in memory (`get_model()` in
  `backend/main.py`) for fast repeated requests.
