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
import math
import tempfile
import zipfile
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
from pydantic import BaseModel, Field, EmailStr
from starlette.background import BackgroundTask

# Engine modules live alongside this file (flat layout for simpler deploys).
sys.path.insert(0, str(Path(__file__).parent))
from analyze import detect_drops, assign_hooks, GENRE_BPM_RANGES  # type: ignore
from transitions import detect_transitions
from render import render_clip, VISUALIZER_STYLES  # type: ignore
from auth import router as auth_router, current_user  # type: ignore
from billing import router as billing_router, is_paying, stripe, stripe_error_handler  # type: ignore


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

app = FastAPI(title="Reelcrate API", version="0.8.0", lifespan=lifespan)
app.add_exception_handler(stripe.error.StripeError, stripe_error_handler)

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
                      hide_logo: bool = False, show_caption: bool = True,
                      framing: str = "fit", selection_mode: str = "transitions") -> None:
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
            lambda: (detect_transitions if selection_mode == "transitions" else detect_drops)(str(source_path), num_clips=num_clips,
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
            "selection_mode": selection_mode,
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
                clip_src.unlink(missing_ok=True)
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
                    clip_audio.unlink(missing_ok=True)
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
                        hide_logo=hide_logo, show_caption=show_caption, framing=framing,
                        logo_path=str(out_dir / "dj-logo.png") if (out_dir / "dj-logo.png").exists() else None,
                    )
                )(),
            )
            if not ok:
                out_path.unlink(missing_ok=True)
                continue
            rendered.append({
                "rank": c["rank"],
                "tag": c["tag"],
                **{key: c[key] for key in ("selection_mode", "transition_start_sec", "transition_end_sec", "transition_evidence", "tempo_before", "tempo_after") if key in c},
                "hook": c["hook"],
                "bpm": c.get("bpm"),
                "local_bpm": c.get("local_bpm"),
                "bpm_confidence": c.get("bpm_confidence", 0),
                "local_bpm_confidence": c.get("local_bpm_confidence", 0),
                "start_sec": c["start_sec"],
                "end_sec": c["end_sec"],
                "genre_match": c.get("genre_match", True),
                "bpm_octave_adjusted": c.get("bpm_octave_adjusted", False),
                "show_caption": show_caption,
                "framing": local_clip.get("framing_applied", c.get("framing_applied", framing)),
                "framing_requested": framing,
                "tracking": local_clip.get("tracking", c.get("tracking", {})),
                "visualizer": visualizer,
                "watermark": watermark,
                "hide_logo": hide_logo,
                "duration_sec": round(c["end_sec"] - c["start_sec"], 2),
                "url": f"/api/clips/{job_id}/clip_{c['rank']:02d}.mp4",
            })
            pct = 20 + int(75 * (i + 1) / len(clips))
            state.update({"progress": pct,
                          "message": f"Rendered {i + 1}/{len(clips)} clips"})
            write_state(job_id, state)

        if not rendered:
            raise RuntimeError('No clips could be rendered. Try a different file.')
        warnings = []
        protected = sum(c.get("framing_requested") == "crop" and c.get("framing") == "fit" for c in rendered)
        if protected:
            warnings.append(f"Kept the full shot in {protected} clips because a reliable close-up could not be maintained.")
        if len(rendered) < num_clips:
            warnings.append(f"Made {len(rendered)} of {num_clips} requested clips. " + ("Only clear transitions that fit your length and BPM range are included; ordinary peaks were not added. Try 60 seconds or a wider BPM range." if selection_mode == "transitions" else "Try fewer clips or a wider BPM range. A short set or a failed render can also reduce the total."))
        if any(not c.get("genre_match", True) for c in rendered):
            warnings.append("Some clips are outside your BPM range because there were not enough confident matches. Tempo is an estimate.")
        # --- Done ---
        state.update({
            "status": "done",
            "progress": 100,
            "message": "Ready",
            "warnings": warnings,
            "clips": rendered,
            "finished_at": time.time(),
        })
        write_state(job_id, state)

    except Exception as e:
        state.update({"status": "failed", "progress": 100,
                      "message": str(e) if isinstance(e, ValueError) else "We could not process this recording. Please try a different audio or video file."})
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
    logo: Optional[UploadFile] = File(None),
    genre: str = Form("all"),
    visualizer: str = Form("freq_bars"),
    num_clips: int = Form(DEFAULT_NUM_CLIPS),
    clip_length: int = Form(DEFAULT_CLIP_LENGTH),
    watermark: str = Form(""),
    show_caption: bool = Form(True),
    framing: str = Form("fit"),
    selection_mode: str = Form("transitions"),
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

    if selection_mode not in ("transitions", "energy"):
        raise HTTPException(400, "Choose Music transitions or Energy moments")
    if selection_mode == "transitions" and clip_length < 30:
        raise HTTPException(400, "Music transitions need 30 or 60 seconds so both songs have room")
    if framing not in ("fit", "crop"):
        raise HTTPException(400, "Choose Keep full frame or Fill screen")
    if len(watermark) > 40 or any(ord(c) < 32 for c in watermark):
        raise HTTPException(400, "DJ name must be 40 characters or fewer, on one line")
    if not (math.isfinite(bpm_min) and math.isfinite(bpm_max)) or not (
        (bpm_min == 0 and bpm_max == 0) or 30 <= bpm_min < bpm_max <= 220
    ):
        raise HTTPException(400, "Choose a BPM range between 30 and 220")

    # Run cleanup FIRST so we have room. Then check we actually have space.
    cleanup_old_jobs()
    emergency_cleanup()
    free_mb = _free_mb()
    if free_mb < 800:  # need at least 800 MB free for a modest upload
        raise HTTPException(507, f"Server disk almost full ({free_mb} MB free) — try again in a few minutes")

    if any(read_state(d.name).get('status') not in ('done', 'failed') for d in JOBS_DIR.iterdir()
           if d.is_dir() and read_state(d.name)):
        raise HTTPException(429, 'Another set is processing. Please try again shortly.')

    # Decode logos before starting the job. Re-encoding removes metadata and
    # rejects malformed/oversized images; no remote URLs or SVG scripts accepted.
    logo_image = None
    if logo is not None and logo.filename:
        from PIL import Image, UnidentifiedImageError
        import io
        data = await logo.read(5 * 1024 * 1024 + 1)
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(400, 'Logo must be smaller than 5 MB')
        try:
            candidate = Image.open(io.BytesIO(data))
            if candidate.format not in ('PNG', 'JPEG', 'WEBP') or max(candidate.size) > 4096:
                raise ValueError('Unsupported logo')
            candidate.load()
            logo_image = candidate.convert('RGBA')
            logo_image.thumbnail((800, 800))
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(400, 'Choose a PNG, JPG or WebP logo up to 4096 pixels wide and tall')

    # Create job dir; stream the upload to disk so we can handle big files.
    job_id = str(uuid.uuid4())
    out_dir = job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_state(job_id, {'job_id': job_id, 'status': 'uploading', 'owner_email': user_email})
    if logo_image is not None:
        logo_image.save(out_dir / "dj-logo.png")
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
        "watermark": watermark.strip(),
        "show_caption": show_caption,
        "framing": framing,
        "selection_mode": selection_mode,
        "has_custom_logo": logo_image is not None,
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
        hide_logo=hide_logo, show_caption=show_caption, framing=framing, selection_mode=selection_mode,
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
    archive_ticket = make_token(job_id + '/clips.zip', ttl=6*3600, kind='clip')
    state['download_url'] = f'/api/downloads/{job_id}?ticket={archive_ticket}'
    return state


@app.get("/api/downloads/{job_id}")
async def download_all(job_id: str, ticket: str = ''):
    from auth import verify_token
    if verify_token(ticket, expect_kind='clip') != job_id + '/clips.zip':
        raise HTTPException(401, 'Open your clips again to refresh this download link')
    state = read_state(job_id)
    if not state:
        raise HTTPException(404, 'These clips have expired. Upload your set again.')
    if state.get('status') != 'done':
        raise HTTPException(409, 'Please wait for your clips to finish')
    def make_archive():
        # Stored MP4s are already compressed. A disk file keeps RAM bounded.
        fd, name = tempfile.mkstemp(suffix='.zip', prefix='reelcrate-')
        os.close(fd)
        archive = Path(name)
        try:
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
                for clip in state.get('clips', []):
                    path = job_dir(job_id) / f"clip_{int(clip['rank']):02d}.mp4"
                    bundle.write(path, path.name)
            return archive
        except Exception:
            archive.unlink(missing_ok=True)
            raise
    archive = await asyncio.to_thread(make_archive)
    return FileResponse(archive, filename='reelcrate-clips.zip', media_type='application/zip',
                        headers={'Cache-Control': 'private, no-store'},
                        background=BackgroundTask(archive.unlink, missing_ok=True))


class RerenderReq(BaseModel):
    caption: str = Field(max_length=120)
    framing: Optional[str] = None
    show_caption: Optional[bool] = None
    visualizer: Optional[str] = None
    watermark: Optional[str] = Field(default=None, max_length=40)
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

    if req.visualizer is not None and req.visualizer not in VISUALIZER_STYLES:
        raise HTTPException(400, 'Unknown visualizer')
    if req.framing is not None and req.framing not in ('fit', 'crop'):
        raise HTTPException(400, 'Unknown framing')
    if req.watermark is not None and any(ord(c) < 32 for c in req.watermark):
        raise HTTPException(400, 'DJ name must be on one line')
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
        "custom_title": req.caption.strip(),
        "tag":       clip_meta.get("tag", "MOMENT"),
        "bpm":       clip_meta.get("bpm") or 0,
        "local_bpm": clip_meta.get("local_bpm") or 0,
        "bpm_confidence":       clip_meta.get("bpm_confidence") or 0,
        "local_bpm_confidence": clip_meta.get("local_bpm_confidence") or 0,
    }

    wm = req.watermark if req.watermark is not None else clip_meta.get("watermark", state.get("watermark", ""))
    viz = req.visualizer or clip_meta.get("visualizer", state.get("visualizer", "freq_bars"))
    framing = req.framing or clip_meta.get("framing_requested", clip_meta.get("framing", state.get("framing", "crop")))
    show_caption = req.show_caption if req.show_caption is not None else bool(req.caption.strip())

    # Watermark toggle: default to whatever the original job used, override
    # with req.hide_logo if the caller sent one. Then enforce the paid gate —
    # only "active" subscribers can hide the REEL/CRATE logo.
    hide_logo_req = req.hide_logo if req.hide_logo is not None else bool(clip_meta.get("hide_logo", state.get("hide_logo")))
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
                                hide_logo=hide_logo_req, show_caption=show_caption, framing=framing,
                                logo_path=str(out_dir / "dj-logo.png") if (out_dir / "dj-logo.png").exists() else None),
        )
        if not ok:
            raise HTTPException(500, 'Re-render failed; your previous clip is still available')
        temporary.replace(out_mp4)
    finally:
        temporary.unlink(missing_ok=True)
        state['status'] = 'done'
        write_state(job_id, state)

    # Update the persisted state so /jobs returns the new caption.
    for c in state.get("clips", []):
        if int(c.get("rank", 0)) == rank:
            c["hook"] = local_clip["custom_title"]
            c.update(watermark=wm, visualizer=viz,
                     framing=local_clip.get("framing_applied", framing),
                     framing_requested=framing, tracking=local_clip.get("tracking", {}),
                     show_caption=show_caption, hide_logo=hide_logo_req)
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
    email: EmailStr
    name: Optional[str] = Field(default="", max_length=120)
    handle: Optional[str] = Field(default="", max_length=80)
    source: Optional[str] = Field(default="reelcrate.app", max_length=120)
    bot_field: str = Field(default="", max_length=500)


def _load_waitlist() -> list:
    if not WAITLIST_FILE.exists():
        return []
    try:
        return json.loads(WAITLIST_FILE.read_text())
    except Exception as exc:
        raise RuntimeError('Signup records cannot be read; refusing to overwrite them') from exc


def _save_waitlist(entries: list) -> None:
    try:
        WAITLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = WAITLIST_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2))
        tmp.replace(WAITLIST_FILE)
    except Exception as exc:
        raise RuntimeError('Signup records could not be saved') from exc


def _signup_roster():
    """One DJ per normalized email across the form and verified app accounts."""
    from auth import _load_users
    people = {}
    for entry in _load_waitlist():
        email = (entry.get('email') or '').strip().lower()
        if not email:
            continue
        row = people.setdefault(email, {'email':email, 'name':entry.get('name',''),
            'joined_at':entry.get('joined_at',0), 'sources':[]})
        source = entry.get('source') or 'landing page'
        if source not in row['sources']:
            row['sources'].append(source)
        row['joined_at'] = min(row['joined_at'] or entry.get('joined_at',0), entry.get('joined_at',0) or row['joined_at'])
    for email, user in _load_users().items():
        if not user.get('verified'):
            continue
        email = email.strip().lower()
        row = people.setdefault(email, {'email':email,'name':user.get('name',''),
            'joined_at':user.get('created_at',0),'sources':[]})
        row['sources'].append('verified app account')
    internal = {ADMIN_EMAIL.strip().lower(), 'support@reelcrate.app', 'hello@reelcrate.app'}
    for row in people.values():
        email = row['email']
        base = email.split('@')[0].split('+')[0] + '@' + email.split('@')[-1]
        row['counts_as_dj'] = base not in internal and not email.endswith(('.test','.invalid','@example.com'))
    return sorted(people.values(), key=lambda row:(row['joined_at'], row['email']))


def _founder_counts(roster=None):
    roster = _signup_roster() if roster is None else roster
    total = sum(row['counts_as_dj'] for row in roster)
    return {'capacity':40, 'signed_up':total, 'remaining':max(0,40-total)}


@app.get('/api/founder-spots')
async def founder_spots():
    # Only aggregates are public. Never expose names, addresses or account state.
    try:
        return JSONResponse(_founder_counts(), headers={'Cache-Control':'no-store'})
    except Exception as exc:
        raise HTTPException(503, 'Signup count is temporarily unavailable') from exc


async def _require_owner(email: str = Depends(current_user)):
    from auth import _load_users
    if email.strip().lower() != ADMIN_EMAIL.strip().lower() or not _load_users().get(email, {}).get('verified'):
        raise HTTPException(403, 'Only the Reelcrate owner can open signup records')
    return email


@app.get('/api/admin/signups')
async def owner_signups(_=Depends(_require_owner)):
    rows = _signup_roster()
    return JSONResponse({**_founder_counts(rows), 'entries':rows}, headers={'Cache-Control':'no-store'})


class SignupImportEntry(BaseModel):
    email: EmailStr
    name: str = Field(default='', max_length=120)
    joined_at: int = Field(ge=0)


class SignupImport(BaseModel):
    entries: list[SignupImportEntry] = Field(max_length=1000)


@app.post('/api/admin/signups/import')
async def import_signups(req: SignupImport, _=Depends(_require_owner)):
    entries = _load_waitlist()
    existing = {(e.get('email') or '').strip().lower():e for e in entries}
    added = 0
    for item in req.entries:
        email = str(item.email).strip().lower()
        if email in existing:
            old = existing[email]
            old['joined_at'] = min(old.get('joined_at') or item.joined_at, item.joined_at)
            continue
        entries.append({'email':email,'name':item.name,'joined_at':item.joined_at,'source':'Netlify claimspot import'})
        existing[email] = entries[-1]
        added += 1
    _save_waitlist(entries)
    return {'ok':True,'imported':added,**_founder_counts()}


@app.post("/api/waitlist")
async def join_waitlist(req: WaitlistReq):
    """Landing-page CLAIM SPOT form posts here. Records the entry, sends the
    submitter a welcome email so they know they made it onto the list, and
    fires an alert to the founder inbox."""
    from email_service import send_waitlist_welcome, send_waitlist_alert
    if req.bot_field:
        return {'ok':True}
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

    roster = _signup_roster()
    founders = [row['email'] for row in roster if row['counts_as_dj']][:40]
    founder = email in founders
    if not already:
        if founder:
            queue_email(send_waitlist_welcome, email, (req.name or '').strip())
        queue_email(send_waitlist_alert, ADMIN_EMAIL, email, (req.name or '').strip())

    return {"ok": True, "founder":founder, **_founder_counts(roster)}


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
