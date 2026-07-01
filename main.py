"""
Reelcrate backend — FastAPI wrapper around the analyze + render engine.

Endpoints:
  POST   /api/upload            Upload a DJ set, returns job_id
  GET    /api/jobs/{job_id}     Get processing status + progress + clip URLs
  GET    /api/clips/{job_id}/{filename}   Serve a clip MP4
  GET    /healthz               Health check

Run locally:
  uvicorn main:app --reload --port 8080

Container:
  docker build -t reelcrate-backend .
  docker run -p 8080:8080 reelcrate-backend
"""

import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Engine modules live alongside this file (flat layout for simpler deploys).
sys.path.insert(0, str(Path(__file__).parent))
from analyze import detect_drops, assign_hooks, GENRE_BPM_RANGES  # type: ignore
from render import render_clip, VISUALIZER_STYLES  # type: ignore
from auth import router as auth_router, current_user  # type: ignore
from billing import router as billing_router, is_paying  # type: ignore


# -------------------- Configuration --------------------

DATA_ROOT = Path(os.environ.get("REELCRATE_DATA", "/tmp/reelcrate"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)
JOBS_DIR = DATA_ROOT / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
CLIP_TTL_HOURS = 6                           # auto-delete after 6 hours (was 24, saves volume space)
FREE_SPACE_MIN_MB = 500                      # if less than this free, run aggressive cleanup
DEFAULT_NUM_CLIPS = 5
DEFAULT_CLIP_LENGTH = 30

# CORS — allow the production landing/app + localhost for dev.
ALLOWED_ORIGINS = [
    "https://reelcrate.app",
    "https://www.reelcrate.app",
    "https://reelcrate.netlify.app",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
]


# -------------------- App setup --------------------

app = FastAPI(title="Reelcrate API", version="0.4.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(billing_router)


# -------------------- Job state helpers --------------------

def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def write_state(job_id: str, state: dict) -> None:
    """Persist job state to disk so workers and HTTP handlers stay in sync."""
    p = job_dir(job_id) / "state.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(p)


def read_state(job_id: str) -> Optional[dict]:
    p = job_dir(job_id) / "state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def cleanup_old_jobs() -> None:
    """Best-effort: delete job dirs older than CLIP_TTL_HOURS."""
    now = time.time()
    cutoff = now - (CLIP_TTL_HOURS * 3600)
    for d in JOBS_DIR.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            continue


def _free_mb() -> int:
    """Free space on the data volume, in MB."""
    try:
        s = shutil.disk_usage(str(DATA_ROOT))
        return int(s.free / 1024 / 1024)
    except Exception:
        return 999999


def emergency_cleanup() -> None:
    """When free space is tight, delete oldest jobs first regardless of TTL,
    until we have at least 2 GB free (enough for a 3-hour set upload + 1 GB slack)."""
    if _free_mb() >= 2048:
        return
    # Sort jobs oldest first
    try:
        jobs = sorted(JOBS_DIR.iterdir(),
                      key=lambda p: p.stat().st_mtime if p.exists() else 0)
    except Exception:
        return
    for d in jobs:
        if _free_mb() >= 2048:
            break
        if d.is_dir():
            print(f"[cleanup] emergency delete {d.name} (free={_free_mb()} MB)")
            shutil.rmtree(d, ignore_errors=True)


# -------------------- Background processing --------------------

async def process_job(job_id: str, source_path: Path, genre: str,
                      visualizer: str, num_clips: int, clip_length: int,
                      watermark: str) -> None:
    """Run analyze + render in a worker thread. Updates job state as it goes."""
    out_dir = job_dir(job_id)
    state = read_state(job_id) or {}

    try:
        # --- Step 1: analyze ---
        state.update({"status": "analyzing", "progress": 10,
                      "message": "Finding the moments…"})
        write_state(job_id, state)

        loop = asyncio.get_event_loop()
        clips = await loop.run_in_executor(
            None,
            lambda: detect_drops(str(source_path), num_clips=num_clips,
                                 clip_len_sec=clip_length, genre=genre),
        )
        clips = assign_hooks(clips)

        manifest = {
            "source": str(source_path),
            "num_clips": len(clips),
            "clip_length_sec": clip_length,
            "genre": genre,
            "visualizer": visualizer,
            "clips": clips,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # --- Step 2: render each clip; progress 20 → 95 ---
        state.update({"status": "rendering", "progress": 20,
                      "message": f"Rendering {len(clips)} clips…"})
        write_state(job_id, state)

        rendered = []
        for i, c in enumerate(clips):
            out_path = out_dir / f"clip_{c['rank']:02d}.mp4"
            ok = await loop.run_in_executor(
                None,
                lambda c=c, p=str(out_path): render_clip(
                    str(source_path), c, p, watermark, visualizer
                ),
            )
            if not ok:
                continue
            rendered.append({
                "rank": c["rank"],
                "tag": c["tag"],
                "hook": c["hook"],
                "bpm": c.get("bpm"),
                "local_bpm": c.get("local_bpm"),
                "duration_sec": clip_length,
                "url": f"/api/clips/{job_id}/clip_{c['rank']:02d}.mp4",
            })
            pct = 20 + int(75 * (i + 1) / len(clips))
            state.update({"progress": pct,
                          "message": f"Rendered {i + 1}/{len(clips)} clips"})
            write_state(job_id, state)

        # --- Done ---
        state.update({
            "status": "done",
            "progress": 100,
            "message": "Ready",
            "clips": rendered,
            "finished_at": time.time(),
        })
        write_state(job_id, state)

    except Exception as e:
        state.update({"status": "failed", "progress": 100,
                      "message": f"Engine error: {type(e).__name__}: {e}"})
        write_state(job_id, state)
        # On failure, also delete the source so we don't hold onto GB of nothing.
        try: source_path.unlink()
        except Exception: pass

    finally:
        # Source file is huge and we don't need it after render — remove it.
        # Keep the clip MP4s (they're small and the user needs to download them).
        try:
            if source_path.exists():
                source_path.unlink()
        except Exception:
            pass


# -------------------- Endpoints --------------------

@app.get("/healthz")
async def healthz():
    return {"ok": True, "version": app.version}


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    genre: str = Form("all"),
    visualizer: str = Form("freq_bars"),
    num_clips: int = Form(DEFAULT_NUM_CLIPS),
    clip_length: int = Form(DEFAULT_CLIP_LENGTH),
    watermark: str = Form("@realdjez1"),
    user_email: str = Depends(current_user),     # gated: sign-in required
):
    # Verified-email gate (signup is allowed but upload requires verification).
    import json as _json
    from auth import USERS_FILE
    try:
        users = _json.loads(USERS_FILE.read_text())
    except Exception:
        users = {}
    if not users.get(user_email, {}).get("verified"):
        raise HTTPException(403, "Please verify your email before uploading. Check your inbox.")
    if not is_paying(user_email):
        raise HTTPException(402, "Start your free trial to upload sets — reelcrate.app/app → Upgrade")
    if genre not in GENRE_BPM_RANGES:
        raise HTTPException(400, f"unknown genre '{genre}'")
    if visualizer not in VISUALIZER_STYLES:
        raise HTTPException(400, f"unknown visualizer '{visualizer}'")
    if not (1 <= num_clips <= 12):
        raise HTTPException(400, "num_clips out of range (1-12)")
    if not (10 <= clip_length <= 90):
        raise HTTPException(400, "clip_length out of range (10-90 sec)")

    # Run cleanup FIRST so we have room. Then check we actually have space.
    cleanup_old_jobs()
    emergency_cleanup()
    free_mb = _free_mb()
    if free_mb < 800:  # need at least 800 MB free for a modest upload
        raise HTTPException(507, f"Server disk almost full ({free_mb} MB free) — try again in a few minutes")

    # Create job dir; stream the upload to disk so we can handle big files.
    job_id = str(uuid.uuid4())
    out_dir = job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "set.mp4").suffix or ".mp4"
    source_path = out_dir / f"source{suffix}"
    size = 0
    try:
        with open(source_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    f.close()
                    shutil.rmtree(out_dir, ignore_errors=True)
                    raise HTTPException(413, "file too large (max 2 GB)")
                f.write(chunk)
    except OSError as e:
        # Disk-full or write error — clean up partial file and bail with a clean message.
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(507, f"Server ran out of storage while receiving your upload: {e}")

    # Persist initial state.
    state = {
        "job_id": job_id,
        "status": "queued",
        "progress": 5,
        "message": "Uploaded — queued for processing",
        "filename": file.filename,
        "size_bytes": size,
        "genre": genre,
        "visualizer": visualizer,
        "num_clips": num_clips,
        "clip_length": clip_length,
        "started_at": time.time(),
        "owner_email": user_email,
    }
    write_state(job_id, state)

    # Cleanup old jobs in the background (best-effort, never blocks).
    asyncio.create_task(asyncio.to_thread(cleanup_old_jobs))

    # Kick off processing in the background. We don't await it.
    asyncio.create_task(process_job(
        job_id, source_path, genre, visualizer, num_clips, clip_length, watermark
    ))

    return JSONResponse({"job_id": job_id, "status_url": f"/api/jobs/{job_id}"})


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    state = read_state(job_id)
    if not state:
        raise HTTPException(404, "job not found (may have expired)")
    return state


@app.get("/api/clips/{job_id}/{filename}")
async def get_clip(job_id: str, filename: str):
    # Trim filename to basename to prevent path traversal.
    safe = Path(filename).name
    if not safe.endswith(".mp4"):
        raise HTTPException(400, "only mp4 supported")
    p = job_dir(job_id) / safe
    if not p.exists():
        raise HTTPException(404, "clip not found")
    return FileResponse(p, media_type="video/mp4",
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/")
async def root():
    return {
        "service": "Reelcrate API",
        "version": app.version,
        "docs": "/docs",
        "health": "/healthz",
    }


@app.get("/api/admin/disk")
async def admin_disk():
    """Public disk-usage probe so I can tell if the volume is full."""
    try:
        s = shutil.disk_usage(str(DATA_ROOT))
        return {
            "total_mb": int(s.total / 1024 / 1024),
            "used_mb":  int(s.used / 1024 / 1024),
            "free_mb":  int(s.free / 1024 / 1024),
            "n_jobs":   sum(1 for _ in JOBS_DIR.iterdir() if _.is_dir()),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/admin/cleanup")
async def admin_cleanup(token: str = ""):
    """Force cleanup. Gate with the JWT secret so randos can't wipe our data."""
    from auth import JWT_SECRET
    if token != JWT_SECRET:
        raise HTTPException(401, "bad token")
    before = _free_mb()
    for d in list(JOBS_DIR.iterdir()):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "before_mb": before, "after_mb": _free_mb()}
