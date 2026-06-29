# Reelcrate Backend

FastAPI service that turns a DJ set upload into 5–8 ready-to-post 9:16 clips.

Wraps `analyze.py` + `render.py` (audio analysis via librosa + render via ffmpeg).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/healthz` | Liveness check |
| POST | `/api/upload` | Upload a set, returns `job_id` |
| GET  | `/api/jobs/{job_id}` | Job status + progress + clip URLs |
| GET  | `/api/clips/{job_id}/{file}.mp4` | Serve a rendered clip |

### Upload form fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file (audio/video) | required | Up to 2 GB, any format ffmpeg reads |
| `genre` | string | `all` | One of `GENRE_BPM_RANGES` (see `engine/analyze.py`) |
| `visualizer` | string | `freq_bars` | One of `VISUALIZER_STYLES` (see `engine/render.py`) |
| `num_clips` | int | 5 | 1–12 |
| `clip_length` | int | 30 | 10–90 (seconds) |
| `watermark` | string | `@realdjez1` | Bottom-left overlay text |

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

## Container

```bash
docker build -t reelcrate-backend .
docker run -p 8080:8080 -v $PWD/data:/data reelcrate-backend
```

## Railway deploy

1. Push this repo to GitHub
2. Railway → New Project → Deploy from GitHub repo → select this repo
3. Railway auto-detects the Dockerfile and builds
4. Set env var `REELCRATE_DATA=/data` and attach a Volume mounted at `/data` for persistent clip storage
5. Get the public URL (e.g. `https://reelcrate-backend.up.railway.app`)
6. Point the frontend at that URL (set `BACKEND_URL` in the frontend JS)

## Clip lifetime

Jobs and their clips auto-delete after 24 hours. Adjust `CLIP_TTL_HOURS` in `main.py`.
