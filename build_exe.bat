@echo off
REM build_exe.bat - Build the one-click Matanglawin executable (Windows).
REM
REM Usage: double-click this file, or run it from a command prompt.
REM
REM Produces dist\Matanglawin.exe - a single file you can double-click
REM with no Python installation required. It starts a local web server
REM and opens your browser to the real-time crack-detection dashboard
REM automatically.

cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing CPU-only torch (keeps the executable small)...
pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q

echo Installing remaining dependencies...
pip install -r requirements.txt -q
pip install pyinstaller -q

echo Building executable with PyInstaller...
pyinstaller --clean -y matanglawin.spec

echo.
echo Done! Your executable is at: dist\Matanglawin.exe
echo Double-click it to run, or copy it anywhere you like.
pause
