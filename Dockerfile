# Dockerfile - run the MATANGLAWIN real-time crack-detection dashboard.
#
# This must run on the same machine/local network as the camera (see
# DEPLOY.md) - it is not meant for public cloud hosting, since the app
# opens the camera directly via OpenCV on the server side.
#
# Build & run locally:
#   docker build -t matanglawin .
#   docker run -p 5000:5000 --device=/dev/video0 matanglawin
#   -> open http://localhost:5000
#
# Drop --device=/dev/video0 if you're only using a network camera
# source (DroidCam / future drone feed) instead of a local webcam.

FROM python:3.11-slim

# System library OpenCV needs at runtime (libGL) plus libglib for good measure.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=5000 \
    # Keep image small: CPU-only inference doesn't need CUDA.
    PIP_NO_CACHE_DIR=1

WORKDIR /home/user/app

# Install CPU-only torch first (much smaller than the default CUDA build),
# then the rest of the dependencies.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application code and model weights.
COPY --chown=user . .

# Point libraries that expect a writable config/cache dir at /tmp, so they
# work regardless of the home directory's permissions on the host platform.
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 5000

CMD ["python", "app.py"]
