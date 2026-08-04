# Dockerfile - deploy the Matanglawin crack-detection website.
#
# Works on any Docker host, and is set up to run directly on
# Hugging Face Spaces (Docker SDK), which serves on port 7860.
#
# Build & run locally:
#   docker build -t matanglawin .
#   docker run -p 7860:7860 matanglawin
#   -> open http://localhost:7860

FROM python:3.11-slim

# System library OpenCV needs at runtime (libGL) plus libglib for good measure.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (Hugging Face Spaces requires this).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=7860 \
    # Keep image small: single-image CPU inference doesn't need CUDA.
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
# work regardless of the home directory's permissions on the host platform
# (Hugging Face Spaces, etc.).
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 7860

CMD ["python", "app.py"]
