# Matanglawin-2.0
Second Version

## Web App

A small Flask website (`app.py`) that lets you upload a photo of a
surface (road, wall, pipe, etc.) in the browser and see the trained
YOLO11-seg model (`best.pt`) highlight every crack it detects, using the
same overlay logic as `infer_overlay.py` (exact mask contour, no
bounding boxes), refactored into `inference_core.py` so both the CLI
script and the website share it.

### Run it locally

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# Linux only, if you hit "libGL.so.1: cannot open shared object file":
#   Debian/Ubuntu: sudo apt-get install -y libgl1
#   Fedora/Amazon Linux: sudo dnf install -y mesa-libGL

python app.py
```

Then open http://localhost:5000 in your browser, upload an image, and
click **Detect Cracks**. You can tweak confidence threshold, mask
opacity, and outline thickness under "Advanced options" before
detecting.

Optional environment variables:

- `WEIGHTS` - path to a different `.pt` weights file (default: `best.pt`)
- `PORT` - port to listen on (default: `5000`)

### Files

- `app.py` - Flask web app (upload form, `/detect` route, `/health` check)
- `inference_core.py` - shared inference + overlay-drawing logic
- `infer_overlay.py` - original standalone CLI script (unchanged)
- `templates/`, `static/` - website HTML/CSS and uploaded/result images
