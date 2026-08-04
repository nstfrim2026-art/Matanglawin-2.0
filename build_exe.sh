#!/usr/bin/env bash
# build_exe.sh - Build the one-click Matanglawin executable (Linux/macOS).
#
# Usage:
#   ./build_exe.sh
#
# Produces dist/Matanglawin - a single file you can double-click (or run
# with `./dist/Matanglawin`) with no Python installation required. It
# starts a local web server and opens your browser to the crack-detection
# website automatically.
#
# On Windows, run the equivalent commands from build_exe.bat instead.

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "==> Creating virtual environment (venv/)..."
  python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Installing CPU-only torch (keeps the executable small)..."
pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q

echo "==> Installing remaining dependencies..."
pip install -r requirements.txt -q
pip install pyinstaller -q

echo "==> Building executable with PyInstaller..."
pyinstaller --clean -y matanglawin.spec

echo
echo "==> Done! Your executable is at: dist/Matanglawin"
echo "    Run it with: ./dist/Matanglawin"
echo "    (or copy dist/Matanglawin anywhere and double-click it)"
