#!/usr/bin/env bash
# setup.sh - Create a virtualenv and install backend dependencies.
#
# ultralytics depends on "opencv-python" (the GUI build, which needs
# libGL.so.1). On headless servers that library usually isn't installed,
# so this script swaps it for "opencv-python-headless" after install.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="../webapp_venv"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r requirements.txt -q

# Swap GUI opencv for the headless build (no system libGL required).
"$VENV_DIR/bin/pip" uninstall -y opencv-python -q || true
"$VENV_DIR/bin/pip" install --force-reinstall --no-deps opencv-python-headless -q

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
