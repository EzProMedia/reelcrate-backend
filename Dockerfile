FROM python:3.11-slim AS base

# System deps: ffmpeg for rendering, fonts for the drawtext overlays
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-lato \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache deps before copying code
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Engine + API (flat layout: main.py + analyze.py + render.py all in /app)
COPY *.py ./

# Data folder mounted as a volume on Railway / fly.io / etc.
ENV REELCRATE_DATA=/data
RUN mkdir -p /data

EXPOSE 8080

# Railway/Render inject $PORT; default to 8080 locally
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
