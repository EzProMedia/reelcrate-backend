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
import re
from contextlib import asynccontextmanager
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Header, Request, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from reliability import start_mail_worker, queue_email, rate_limit
from pydantic import BaseModel

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

@asynccontextmanager
async def lifespan(app):
    # Jobs interrupted by a previous process cannot still be running.
    for directory in JOBS_DIR.iterdir():
        if directory.is_dir():
            state = read_state(directory.name)
            if state and state.get('status') not in ('done', 'failed'):
                state.update(status='failed', message='Server restarted. Please upload again.')
                write_state(directory.name, state)
    stop, worker = start_mail_worker()
    async def scheduled_cleanup():
        while True:
            await asyncio.to_thread(cleanup_old_jobs)
            await asyncio.sleep(60)
    cleanup_task = asyncio.create_task(scheduled_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        stop.set()
        await asyncio.to_thread(worker.join, 12)

app = FastAPI(title="Reelcrate API", version="0.6.0", lifespan=lifespan)

@app.middleware('http')
async def limit_auth_attempts(request: Request, call_next):
    if request.method == 'POST' and request.url.path in {
        '/api/auth/signup', '/api/auth/signin', '/api/auth/forgot',
        '/api/auth/reset', '/api/auth/resend', '/api/auth/mfa/verify',
        '/api/auth/mfa/confirm', '/api/auth/mfa/disable', '/api/waitlist'}:
        try:
            rate_limit('ip:' + (request.client.host if request.client else 'unknown') + ':' + request.url.path, 30)
        except HTTPException as exc:
            return JSONResponse({'detail': exc.detail}, status_code=exc.status_code, headers=exc.headers)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response

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
    try:
        if str(uuid.UUID(job_id)) != job_id:
            raise ValueError()
    except ValueError:
        raise HTTPException(404, 'Job not found')
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


def completed_job(directory):
    state = read_state(directory.name)
    return bool(state and state.get('status') in ('done', 'failed'))


def cleanup_old_jobs() -> None:
    """Best-effort: delete job dirs older than CLIP_TTL_HOURS."""
    now = time.time()
    cutoff = now - (CLIP_TTL_HOURS * 3600)
    for d in JOBS_DIR.iterdir():
        try:
            if d.is_dir() and completed_job(d) and d.stat().st_mtime < cutoff:
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
        if d.is_dir() and completed_job(d):
            print(f"[cleanup] emergency delete {d.name} (free={_free_mb()} MB)")
            shutil.rmtree(d, ignore_errors=True)


# -------------------- Background processing --------------------

async def process_job(job_id: str, source_path: Path, genre: str,
                      visualizer: str, num_clips: int, clip_length: int,
                      watermark: str, variation_seed: int = 0,
                      bpm_min: float = 0, bpm_max: float = 0,
                      hide_logo: bool = False) -> None:
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
                                 clip_len_sec=clip_length, genre=genre,
                                 variation_seed=variation_seed,
                                 bpm_min=bpm_min, bpm_max=bpm_max),
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
        import subprocess
        for i, c in enumerate(clips):
            out_path = out_dir / f"clip_{c['rank']:02d}.mp4"
            # Cut a small MP4 of just this clip window — VIDEO + AUDIO, not
            # audio-only. This is what render_clip uses as its source, so when
            # the DJ uploads a video mix we KEEP their footage in the output
            # (the caption + bars overlay on top). Also lets us re-render
            # later with a different caption without needing the full source.
            clip_src = out_dir / f"clip_{c['rank']:02d}_src.mp4"
            # Keep a backwards-compatible audio-only WAV for jobs whose src
            # doesn't have video (e.g. mp3 mixes) OR for the rerender endpoint
            # to fall back to on older jobs.
            clip_audio = out_dir / f"clip_{c['rank']:02d}.wav"
            cache_ok = False
            try:
                # Re-encode the 30-sec window: cheap, avoids keyframe seeking
                # gotchas, and keeps file size small (~15–25 MB per clip).
                await asyncio.to_thread(subprocess.run,
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-ss", str(c["start_sec"]),
                     "-i", str(source_path),
                     "-t",  str(c["end_sec"] - c["start_sec"]),
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                     "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                     "-movflags", "+faststart",
                     str(clip_src)],
                    check=True,
                )
                cache_ok = True
            except Exception as e:
                print(f"[warn] could not cache clip source for rank {c['rank']}: {e}")

            # Fallback audio-only cache — used only if the MP4 cut failed above.
            if not cache_ok:
                try:
                    await asyncio.to_thread(subprocess.run,
                        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                         "-ss", str(c["start_sec"]),
                         "-i", str(source_path),
                         "-t",  str(c["end_sec"] - c["start_sec"]),
                         "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le",
                         str(clip_audio)],
                        check=True,
                    )
                except Exception as e2:
                    print(f"[warn] could not cache audio-only either: {e2}")

            # Timestamps in `c` reference the ORIGINAL source. For the render
            # shift them to zero because we're feeding in the pre-cut clip.
            local_clip = {**c, "start_sec": 0.0,
                          "end_sec": c["end_sec"] - c["start_sec"],
                          "peak_sec": max(0.0, c["peak_sec"] - c["start_sec"])}

            # Prefer the video-preserving MP4 cut; fall back to WAV cut; fall
            # back to full source only if both caches failed.
            def _pick_source(cs=str(clip_src), ca=str(clip_audio),
                             src=str(source_path)):
                if os.path.exists(cs):
                    return cs, True   # per-clip mp4 → local_clip (t shifted)
                if os.path.exists(ca):
                    return ca, True   # per-clip wav → local_clip (t shifted)
                return src, False     # full source → original c (t not shifted)

            ok = await loop.run_in_executor(
                None,
                lambda lc=local_clip, p=str(out_path): (
                    lambda picked=_pick_source(): render_clip(
                        picked[0], lc if picked[1] else c,
                        p, watermark, visualizer,
                        hide_logo=hide_logo,
                    )
                )(),
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

        if not rendered:
            raise RuntimeError('No clips could be rendered. Try a different file.')
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
    variation_seed: int = Form(0),               # 0 = deterministic; >0 = randomize picks
    bpm_min: float = Form(0),                    # explicit BPM range (preferred over genre)
    bpm_max: float = Form(0),
    hide_logo: bool = Form(False),               # paid-tier-only; enforced below
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

    # Watermark toggle gate: hide_logo is a paid-tier-only perk. Trial users
    # (subscription_status == "trialing") always get the REEL/CRATE watermark
    # baked in as free-tier marketing exposure. Only "active" subscribers can
    # remove it. We silently coerce rather than 4xx so the frontend can send
    # the flag optimistically and let the backend do the enforcement.
    if hide_logo:
        _sub_status = (users.get(user_email, {}) or {}).get("subscription_status")
        if _sub_status != "active":
            hide_logo = False
    # The frontend can send an explicit bpm_min/bpm_max range and any string
    # for genre (used as a display label). Only reject an unknown genre if
    # NO explicit BPM range was provided AND the genre isn't a known preset.
    if genre not in GENRE_BPM_RANGES and not (bpm_min > 0 and bpm_max > bpm_min):
        raise HTTPException(400, f"unknown genre '{genre}' and no bpm range")
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

    if any(read_state(d.name).get('status') not in ('done', 'failed') for d in JOBS_DIR.iterdir()
           if d.is_dir() and read_state(d.name)):
        raise HTTPException(429, 'Another set is processing. Please try again shortly.')

    # Create job dir; stream the upload to disk so we can handle big files.
    job_id = str(uuid.uuid4())
    out_dir = job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_state(job_id, {'job_id': job_id, 'status': 'uploading', 'owner_email': user_email})
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
                if _free_mb() < 500:
                    raise OSError('Not enough room to finish this upload')
                f.write(chunk)
    except asyncio.CancelledError:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    except OSError as e:
        # Disk-full or write error — clean up partial file and bail with a clean message.
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(507, f"Server ran out of storage while receiving your upload: {e}")

    if size == 0:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(400, 'The uploaded file is empty')
    # Persist initial state.
    state = {
        "job_id": job_id,
        "status": "queued",
        "progress": 5,
        "message": "Uploaded — queued for processing",
        "filename": file.filename,
        "size_bytes": size,
        "genre": genre,
        "bpm_min": bpm_min,
        "bpm_max": bpm_max,
        "visualizer": visualizer,
        "num_clips": num_clips,
        "clip_length": clip_length,
        "hide_logo": hide_logo,
        "watermark": watermark,
        "started_at": time.time(),
        "owner_email": user_email,
    }
    write_state(job_id, state)

    # Cleanup old jobs in the background (best-effort, never blocks).
    asyncio.create_task(asyncio.to_thread(cleanup_old_jobs))

    # Kick off processing in the background. We don't await it.
    asyncio.create_task(process_job(
        job_id, source_path, genre, visualizer, num_clips, clip_length, watermark,
        variation_seed=variation_seed,
        bpm_min=bpm_min, bpm_max=bpm_max,
        hide_logo=hide_logo,
    ))

    return JSONResponse({"job_id": job_id, "status_url": f"/api/jobs/{job_id}"})


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str, user_email: str = Depends(current_user)):
    state = read_state(job_id)
    if not state:
        raise HTTPException(404, "job not found (may have expired)")
    if state.get('owner_email') != user_email:
        raise HTTPException(403, 'Not your job')
    from auth import make_token
    state.pop('owner_email', None)
    for clip in state.get('clips', []):
        filename = f"clip_{int(clip['rank']):02d}.mp4"
        ticket = make_token(job_id + '/' + filename, ttl=6*3600, kind='clip')
        clip['url'] = f'/api/clips/{job_id}/{filename}?ticket={ticket}'
    return state


class RerenderReq(BaseModel):
    caption: str
    visualizer: Optional[str] = None
    watermark: Optional[str] = None
    hide_logo: Optional[bool] = None   # paid-tier only — enforced below


@app.post("/api/clips/{job_id}/{rank}/rerender")
async def rerender_clip(job_id: str, rank: int, req: RerenderReq,
                        user_email: str = Depends(current_user)):
    """Re-render an existing clip with a new caption. Fast — uses the cached
    per-clip audio window, not the full source."""
    state = read_state(job_id)
    if not state:
        raise HTTPException(404, "job not found (may have expired)")
    if state.get("owner_email") != user_email:
        raise HTTPException(403, "not your job")

    if state.get("status") != "done":
        raise HTTPException(409, "This job is still processing. Try again shortly.")

    clip_meta = next((c for c in state.get("clips", []) if int(c.get("rank", 0)) == rank), None)
    if not clip_meta:
        raise HTTPException(404, f"clip {rank} not found")

    out_dir = job_dir(job_id)
    clip_src   = out_dir / f"clip_{rank:02d}_src.mp4"   # preserves video+audio
    clip_audio = out_dir / f"clip_{rank:02d}.wav"       # audio-only fallback
    out_mp4    = out_dir / f"clip_{rank:02d}.mp4"

    # Prefer MP4 cache (keeps user's uploaded video). Fall back to WAV cache
    # for jobs cached under the pre-MP4-cache scheme.
    if clip_src.exists():
        source_for_render = str(clip_src)
    elif clip_audio.exists():
        source_for_render = str(clip_audio)
    else:
        raise HTTPException(410,
            "Clip source no longer cached — the job is older than the retention "
            "window. Re-upload the set to edit captions."
        )

    # Build a minimal clip dict for render_clip. Use the caption text as the
    # custom_title override (render.py picks it up via clip.get('custom_title')).
    duration = float(clip_meta.get("duration_sec") or 30)
    local_clip = {
        "rank": rank,
        "start_sec": 0.0,
        "end_sec":   duration,
        "peak_sec":  min(duration * 0.2, 6.0),
        "hook":      clip_meta.get("hook", ""),
        "custom_title": (req.caption or "").strip() or clip_meta.get("hook", ""),
        "tag":       clip_meta.get("tag", "MOMENT"),
        "bpm":       clip_meta.get("bpm") or 0,
        "local_bpm": clip_meta.get("local_bpm") or 0,
        "bpm_confidence":       clip_meta.get("bpm_confidence") or 0,
        "local_bpm_confidence": clip_meta.get("local_bpm_confidence") or 0,
    }

    wm = req.watermark or state.get("watermark") or "@realdjez1"
    viz = req.visualizer or state.get("visualizer") or "freq_bars"

    # Watermark toggle: default to whatever the original job used, override
    # with req.hide_logo if the caller sent one. Then enforce the paid gate —
    # only "active" subscribers can hide the REEL/CRATE logo.
    hide_logo_req = req.hide_logo if req.hide_logo is not None else bool(state.get("hide_logo"))
    if hide_logo_req:
        import json as _json
        from auth import USERS_FILE
        try:
            _users = _json.loads(USERS_FILE.read_text())
        except Exception:
            _users = {}
        if (_users.get(user_email, {}) or {}).get("subscription_status") != "active":
            hide_logo_req = False

    loop = asyncio.get_event_loop()
    temporary = out_dir / f"clip_{rank:02d}_editing.mp4"
    state['status'] = 'rerendering'
    write_state(job_id, state)
    try:
        ok = await loop.run_in_executor(
            None,
            lambda: render_clip(source_for_render, local_clip, str(temporary), wm, viz,
                                hide_logo=hide_logo_req),
        )
        if not ok:
            raise HTTPException(500, 'Re-render failed; your previous clip is still available')
        temporary.replace(out_mp4)
    finally:
        state['status'] = 'done'
        write_state(job_id, state)

    # Update the persisted state so /jobs returns the new caption.
    for c in state.get("clips", []):
        if int(c.get("rank", 0)) == rank:
            c["hook"] = local_clip["custom_title"]
    write_state(job_id, state)

    # Cache-bust query so the browser refetches the new bytes.
    from auth import make_token
    ticket = make_token(f'{job_id}/clip_{rank:02d}.mp4', ttl=6*3600, kind='clip')
    return {"ok": True,
            "url": f"/api/clips/{job_id}/clip_{rank:02d}.mp4?ticket={ticket}&v={int(time.time())}"}


@app.get("/api/clips/{job_id}/{filename}")
async def get_clip(job_id: str, filename: str, request: Request, ticket: str = '', download: bool = False):
    from auth import verify_token
    if not re.fullmatch(r'clip_\d{2}\.mp4', filename):
        raise HTTPException(404, 'Clip not found')
    if verify_token(ticket, expect_kind='clip') != job_id + '/' + filename:
        raise HTTPException(401, 'Open your clips again to refresh this download link')
    p = job_dir(job_id) / filename
    if not p.exists():
        raise HTTPException(404, 'Clip expired or not found')
    size = p.stat().st_size
    start, end, status = 0, size - 1, 200
    headers = {'Accept-Ranges': 'bytes', 'Cache-Control': 'private, no-store'}
    if download:
        headers['Content-Disposition'] = f'attachment; filename="reelcrate-{filename}"'
    value = request.headers.get('range')
    if value:
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
        if not match or not any(match.groups()):
            raise HTTPException(416, headers={'Content-Range': f'bytes */{size}'})
        left, right = match.groups()
        if left:
            start = int(left)
            end = min(int(right), size-1) if right else size-1
        else:
            start = max(0, size-int(right))
        if start > end or start >= size:
            raise HTTPException(416, headers={'Content-Range': f'bytes */{size}'})
        status = 206
        headers['Content-Range'] = f'bytes {start}-{end}/{size}'
    headers['Content-Length'] = str(end-start+1)
    def chunks():
        with p.open('rb') as source:
            source.seek(start)
            remaining = end-start+1
            while remaining:
                chunk = source.read(min(256*1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
    return StreamingResponse(chunks(), status_code=status, media_type='video/mp4', headers=headers)


@app.get("/")
async def root():
    return {
        "service": "Reelcrate API",
        "version": app.version,
        "docs": "/docs",
        "health": "/healthz",
    }


# -------------------- Waitlist (landing page CLAIM SPOT form) --------------------

WAITLIST_FILE = DATA_ROOT / "waitlist.json"
ADMIN_EMAIL   = os.environ.get("ADMIN_EMAIL", "ezanaberhe@gmail.com")
ADMIN_TOKEN   = os.environ.get("ADMIN_TOKEN", "")


def _require_admin(authorization: Optional[str] = Header(None)) -> None:
    import secrets
    if len(ADMIN_TOKEN.encode()) < 32:
        raise HTTPException(503, 'Admin API disabled; configure ADMIN_TOKEN')
    supplied = (authorization or '')
    if not supplied.startswith('Bearer ') or not secrets.compare_digest(supplied[7:].encode(), ADMIN_TOKEN.encode()):
        raise HTTPException(401, 'Admin authorization required')


class WaitlistReq(BaseModel):
    email: str
    name: Optional[str] = ""
    handle: Optional[str] = ""
    source: Optional[str] = "reelcrate.app"


def _load_waitlist() -> list:
    if not WAITLIST_FILE.exists():
        return []
    try:
        return json.loads(WAITLIST_FILE.read_text())
    except Exception:
        return []


def _save_waitlist(entries: list) -> None:
    try:
        WAITLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = WAITLIST_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2))
        tmp.replace(WAITLIST_FILE)
    except Exception as e:
        print(f"[waitlist] save failed: {e}")


@app.post("/api/waitlist")
async def join_waitlist(req: WaitlistReq):
    """Landing-page CLAIM SPOT form posts here. Records the entry, sends the
    submitter a welcome email so they know they made it onto the list, and
    fires an alert to the founder inbox."""
    from email_service import send_waitlist_welcome, send_waitlist_alert
    email = (req.email or "").strip().lower()
    if "@" not in email or "." not in email:
        raise HTTPException(400, "invalid email")

    entries = _load_waitlist()
    # De-dup by email — resubmit still fires the confirmation, but we don't
    # store the same row twice.
    already = any((e.get("email") or "").lower() == email for e in entries)
    if not already:
        entries.append({
            "email": email,
            "name":   (req.name or "").strip(),
            "handle": (req.handle or "").strip(),
            "source": req.source or "reelcrate.app",
            "joined_at": int(time.time()),
        })
        _save_waitlist(entries)

    if not already:
        queue_email(send_waitlist_welcome, email, (req.name or '').strip())
        queue_email(send_waitlist_alert, ADMIN_EMAIL, email, (req.name or '').strip())

    return {"ok": True, "already_on_list": already, "total": len(entries)}


@app.get("/api/admin/waitlist")
async def admin_waitlist(_=Depends(_require_admin)):
    """Peek at the current waitlist. Requires the dedicated admin bearer token."""
    entries = _load_waitlist()
    return {"count": len(entries), "entries": entries}


@app.get("/api/admin/disk")
async def admin_disk(_=Depends(_require_admin)):
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
async def admin_cleanup(_=Depends(_require_admin)):
    """Force cleanup. Requires the dedicated admin bearer token."""
    before = _free_mb()
    for d in list(JOBS_DIR.iterdir()):
        if d.is_dir() and completed_job(d):
            shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "before_mb": before, "after_mb": _free_mb()}

# Optional same-origin frontend: works on Railway and a local phone preview.
FRONTEND = Path(__file__).parent / 'frontend'
@app.get('/app/config.js')
async def app_config():
    from fastapi.responses import Response
    return Response('window.REELCRATE_BACKEND = "";', media_type='application/javascript', headers={'Cache-Control': 'no-store'})

if FRONTEND.exists():
    app.mount('/app', StaticFiles(directory=FRONTEND / 'app', html=True), name='phone-app')
    app.mount('/assets', StaticFiles(directory=FRONTEND), name='assets')
