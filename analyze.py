#!/usr/bin/env python3
"""
Reelcrate audio analysis engine.

Takes a DJ set audio/video file → finds the N best clipping moments based on
energy + spectral flux (proxy for "drops"), avoiding silences and overlap.

Outputs a JSON manifest of {start_sec, end_sec, score, reason} segments.

Usage:
    python3 analyze.py <input_file> [--num-clips N] [--clip-length 30] [--genre dnb]
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import find_peaks, butter, sosfiltfilt
import random


# ---------------- Genre → BPM range mapping ----------------
# Local BPM around each peak must fall in this range (or its half/double for
# tempo-detection ambiguity). Set "all" to skip filtering.
# Ranges reflect real-world style spans, not textbook BPM. Room-mic recordings
# have wobbly tempo estimates so being generous here rescues real matches.
GENRE_BPM_RANGES = {
    "hiphop":    (70, 105),    # trap-tinged hip-hop drifts to 100+
    "trap":      (125, 175),   # often double-time of half-time
    "reggae":    (60, 92),
    "dancehall": (85, 112),
    "afrobeats": (90, 120),    # was 95-115 — too narrow, missed real matches
    "afrohouse": (110, 128),   # was 115-125
    "amapiano":  (100, 122),   # was 108-116
    "reggaeton": (85, 105),
    "house":     (118, 130),   # was 120-128
    "techhouse": (120, 132),
    "deephouse": (115, 128),
    "disco":     (105, 125),
    "techno":    (125, 145),
    "trance":    (128, 148),
    "hardstyle": (140, 165),
    "dnb":       (155, 190),
    "dubstep":   (135, 150),
    "all":       None,         # no filter
}
GENRE_PRETTY = {
    "hiphop": "Hip-Hop", "trap": "Trap", "reggae": "Reggae",
    "dancehall": "Dancehall", "afrobeats": "Afrobeats",
    "afrohouse": "Afrohouse", "amapiano": "Amapiano", "reggaeton": "Reggaeton",
    "house": "House", "techhouse": "Tech House", "deephouse": "Deep House",
    "disco": "Disco", "techno": "Techno", "trance": "Trance",
    "hardstyle": "Hardstyle", "dnb": "Drum & Bass", "dubstep": "Dubstep",
    "all": "All Genres",
}


# ---------------- I/O ----------------

def ensure_wav(input_path: str) -> tuple[str, bool]:
    """If input is video or non-wav audio, ffmpeg-extract to a temp wav.
    Returns (wav_path, is_temp)."""
    ext = Path(input_path).suffix.lower()
    if ext in (".wav",):
        return input_path, False
    # Extract mono 22050 Hz wav from anything (audio or video)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "22050",
        "-vn", "-loglevel", "error",
        tmp.name,
    ]
    subprocess.run(cmd, check=True)
    return tmp.name, True


# ---------------- Analysis ----------------

def _bass_filter(y: np.ndarray, sr: int, cutoff_hz: float = 250.0) -> np.ndarray:
    """Low-pass filter to isolate kick + bassline. Room-mic recordings have lots
    of high-freq chatter that confuses tempo detection; killing everything above
    250 Hz keeps just the rhythm section."""
    try:
        sos = butter(4, cutoff_hz, btype="low", fs=sr, output="sos")
        return sosfiltfilt(sos, y)
    except Exception:
        return y


def _stable_bpm(y_bass: np.ndarray, sr: int) -> tuple[float, float]:
    """Detect BPM on bass-filtered audio. Returns (bpm, confidence).
    Confidence is 0..1: higher = more consistent across short windows."""
    try:
        # Aggregate=None gives per-frame tempo estimates → check stability
        tempo_curve = librosa.feature.tempo(y=y_bass, sr=sr, aggregate=None)
        if tempo_curve is None or len(tempo_curve) == 0:
            return 0.0, 0.0
        median_bpm = float(np.median(tempo_curve))
        # Confidence = fraction of estimates within ±5 BPM of median
        agreement = float(np.mean(np.abs(tempo_curve - median_bpm) <= 5))
        return median_bpm, agreement
    except Exception:
        return 0.0, 0.0


def _local_bpm(y, sr, peak_sec: float, window_sec: float = 15.0) -> tuple[float, float]:
    """Estimate BPM (and confidence) in a window centered on the given peak.
    Uses bass-isolated signal for accuracy on room-mic / phone-mic recordings."""
    half = window_sec / 2.0
    start = max(0, int((peak_sec - half) * sr))
    end = min(len(y), int((peak_sec + half) * sr))
    window = y[start:end]
    if len(window) < sr:  # less than 1 sec — too short
        return 0.0, 0.0
    bass = _bass_filter(window, sr)
    return _stable_bpm(bass, sr)


def _bpm_matches_genre(bpm: float, genre: str) -> bool:
    """True if local bpm (or its half/double) falls in the genre's range."""
    rng = GENRE_BPM_RANGES.get(genre)
    if rng is None:
        return True
    lo, hi = rng
    for candidate in (bpm, bpm * 2, bpm / 2):
        if lo <= candidate <= hi:
            return True
    return False


def _bpm_in_range(bpm: float, lo: float, hi: float) -> bool:
    """True if `bpm` or its half/double falls in [lo, hi]. Handles tempo-detection
    octave ambiguity so a 60 BPM detection of a 120 BPM track still matches
    the (100, 130) range."""
    if lo <= 0 or hi <= 0 or hi <= lo:
        return True   # unset = no filter
    for candidate in (bpm, bpm * 2, bpm / 2):
        if lo <= candidate <= hi:
            return True
    return False


def _ffmpeg_predecode(input_path: str, target_sr: int = 11025) -> tuple[str, str]:
    """
    Convert any input format to two small mono WAV files: main + bass-filtered.
    ffmpeg streams the input so we never hold the full source file in memory.

    Returns (main_wav_path, bass_wav_path). Files live in a temp dir next to
    the input so they get cleaned up with the job.
    """
    import subprocess, tempfile
    src_dir = os.path.dirname(os.path.abspath(input_path)) or tempfile.gettempdir()
    main_wav = os.path.join(src_dir, "_analyze_main.wav")
    bass_wav = os.path.join(src_dir, "_analyze_bass.wav")

    # Main: mono, target_sr, 16-bit PCM. Tiny disk footprint (~22 MB per hour).
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", input_path,
         "-ac", "1", "-ar", str(target_sr),
         "-c:a", "pcm_s16le", main_wav],
        check=True,
    )
    # Bass: low-pass at 250 Hz — used for stable BPM detection on the kick drum.
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", main_wav, "-af", "lowpass=f=250",
         "-c:a", "pcm_s16le", bass_wav],
        check=True,
    )
    return main_wav, bass_wav


def _read_wav_window(wav_path: str, start_sec: float, dur_sec: float) -> tuple[np.ndarray, int]:
    """Read a small window from a WAV file (used for BPM around each peak)."""
    import soundfile as sf
    with sf.SoundFile(wav_path) as f:
        sr = f.samplerate
        f.seek(max(0, int(start_sec * sr)))
        y = f.read(int(dur_sec * sr), dtype="float32", always_2d=False)
    return y, sr


def _stream_rms_and_flux(wav_path: str, hop: int = 256, n_fft: int = 1024,
                         chunk_sec: float = 300.0) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Stream the WAV in `chunk_sec` chunks, accumulating full-length RMS + spectral
    flux arrays. Peak memory ~40 MB per chunk regardless of set length.

    Returns (rms, flux, sr, duration_sec).
    """
    import soundfile as sf, gc

    rms_parts = []
    flux_parts = []
    prev_last_col = None  # carry-over for spectral flux continuity across chunks

    with sf.SoundFile(wav_path) as f:
        sr = f.samplerate
        total_frames = f.frames
        chunk_frames = int(chunk_sec * sr)

        for offset in range(0, total_frames, chunk_frames):
            f.seek(offset)
            chunk = f.read(chunk_frames, dtype="float32", always_2d=False)
            if chunk.size == 0:
                break

            # RMS on this chunk
            r = librosa.feature.rms(y=chunk, hop_length=hop)[0]
            rms_parts.append(r)

            # STFT + spectral flux on this chunk. Prepend previous last column so
            # the diff between chunks doesn't produce a bogus spike at the seam.
            S = np.abs(librosa.stft(chunk, hop_length=hop, n_fft=n_fft))
            if prev_last_col is not None:
                S = np.hstack([prev_last_col[:, None], S])
            prev_last_col = S[:, -1].copy()
            fl = np.maximum(0, np.diff(S, axis=1)).sum(axis=0)
            flux_parts.append(fl)

            del chunk, S, r, fl
            gc.collect()

        duration = total_frames / sr

    rms = np.concatenate(rms_parts) if rms_parts else np.zeros(0, dtype=np.float32)
    flux = np.concatenate(flux_parts) if flux_parts else np.zeros(0, dtype=np.float32)
    # Match original 'flux' shape (prepend 0 to align with rms length)
    if flux.size < rms.size:
        flux = np.concatenate([np.zeros(rms.size - flux.size, dtype=np.float32), flux])
    return rms, flux, sr, duration


def detect_drops(audio_path: str, num_clips: int = 5, clip_len_sec: float = 30.0,
                 lead_in_sec: float = 6.0, genre: str = "all",
                 variation_seed: int = 0,
                 bpm_min: float = 0, bpm_max: float = 0) -> list[dict]:
    """
    Returns a list of clip dicts:
      [{start_sec, end_sec, score, peak_sec, energy, flux, bpm, local_bpm, genre_match}, ...]
    sorted by start_sec ascending.

    v2 memory strategy (fits under 512 MB even for 3-hour sets):
      - ffmpeg pre-decodes to a tiny mono 11025 Hz WAV on disk (no full-file load)
      - Streaming RMS + spectral flux in 5-minute chunks
      - BPM windows re-read directly from the WAV (never hold full audio in RAM)
    """
    import gc

    # Resolve the effective BPM filter range. Explicit bpm_min/bpm_max wins.
    # Otherwise fall back to the genre → range mapping (backwards compat with
    # older frontends). A range of (0, 0) means "no filter".
    if bpm_min > 0 and bpm_max > bpm_min:
        eff_lo, eff_hi = float(bpm_min), float(bpm_max)
        filter_active = True
    else:
        rng = GENRE_BPM_RANGES.get(genre)
        if rng is None:
            eff_lo, eff_hi = 0.0, 0.0
            filter_active = False
        else:
            eff_lo, eff_hi = float(rng[0]), float(rng[1])
            filter_active = True

    # Step 1: ffmpeg pre-decode → small on-disk WAV files
    main_wav, bass_wav = _ffmpeg_predecode(audio_path, target_sr=11025)

    hop = 256
    n_fft = 1024

    # Step 2: streaming analysis → full-length RMS + flux arrays
    rms, flux, sr, duration = _stream_rms_and_flux(main_wav, hop=hop, n_fft=n_fft)
    frame_sec = hop / sr

    if duration < clip_len_sec * 2:
        print(f"[warn] track is short ({duration:.1f}s); reducing num_clips")
        num_clips = max(1, min(num_clips, int(duration // clip_len_sec)))

    # 1) Smooth RMS over ~2 sec
    win = max(1, int(2.0 / frame_sec))
    rms_smooth = librosa.util.normalize(
        np.convolve(rms, np.ones(win) / win, mode="same")
    )

    # 2) Smooth flux over ~1 sec
    win_f = max(1, int(1.0 / frame_sec))
    flux_smooth = librosa.util.normalize(
        np.convolve(flux, np.ones(win_f) / win_f, mode="same")
    )
    del rms, flux; gc.collect()

    # 3) Combined score — drops have both high sustained energy AND a flux spike
    score = 0.55 * rms_smooth + 0.45 * flux_smooth

    # Dead-air guard: zero out frames where rms is below noise floor
    noise_floor = np.percentile(rms_smooth, 15)
    score[rms_smooth < noise_floor] = 0

    # 4) Edge guard: don't pick peaks too close to start/end
    edge_frames = int(clip_len_sec / frame_sec)
    score[:edge_frames] = 0
    score[-edge_frames:] = 0

    # 5) Find peaks with min separation = clip_len_sec (non-overlapping clips).
    # Find MORE candidates than num_clips so genre filter has room to work.
    min_distance_frames = int(clip_len_sec / frame_sec)
    candidate_pool = max(num_clips * 4, 15)  # find at least 15 candidates
    peaks_idx, props = find_peaks(
        score,
        distance=min_distance_frames,
        prominence=score.std() * 0.4,
    )

    if len(peaks_idx) == 0:
        # Fallback: just take the top-N highest values, spaced out
        ordered = np.argsort(score)[::-1]
        picked = []
        for idx in ordered:
            if all(abs(idx - p) >= min_distance_frames for p in picked):
                picked.append(idx)
                if len(picked) >= candidate_pool:
                    break
        peaks_idx = np.array(picked)

    # 6) Rank candidate peaks by score
    peak_scores = score[peaks_idx]
    ranked_candidates = sorted(zip(peaks_idx, peak_scores), key=lambda x: -x[1])[:candidate_pool]

    # 7) Global BPM: sample three 30s windows from the bass WAV and pick the
    # most confident tempo. Cheap (~1 MB per window) and avoids loading the
    # full bass-filtered audio into memory.
    def _global_bpm() -> tuple[float, float]:
        try:
            probe_offsets = [duration * 0.25, duration * 0.50, duration * 0.75]
            best_bpm, best_conf = 0.0, 0.0
            for off in probe_offsets:
                y_win, wsr = _read_wav_window(bass_wav, max(0.0, off - 15.0), 30.0)
                if y_win.size < wsr * 5:
                    continue
                b, c = _stable_bpm(y_win, wsr)
                if c > best_conf:
                    best_bpm, best_conf = b, c
                del y_win; gc.collect()
            return best_bpm, best_conf
        except Exception:
            return 0.0, 0.0

    global_bpm, global_conf = _global_bpm()

    # 8) Compute local BPM around each candidate peak (window read directly
    # from the bass WAV — never holds the full track in RAM).
    def _local_bpm_streamed(peak_sec: float, window_sec: float = 15.0) -> tuple[float, float]:
        start = max(0.0, peak_sec - window_sec / 2)
        y_win, wsr = _read_wav_window(bass_wav, start, window_sec)
        if y_win.size < wsr * 2:
            return 0.0, 0.0
        try:
            return _stable_bpm(y_win, wsr)
        finally:
            del y_win; gc.collect()

    # Confidence gate: below this we don't trust the BPM enough to call it a
    # genre match. Room-mic recordings often produce wobbly BPMs; we'd rather
    # demote a low-confidence peak to the "secondary" pool than let a bogus
    # BPM either pass or fail the genre check erroneously.
    GENRE_MATCH_MIN_CONF = 0.30

    # Frames per second (of analysis grid) used below for flux windowing.
    frames_per_sec = 1.0 / frame_sec

    candidates_with_meta = []
    for frame, sc in ranked_candidates:
        peak_sec = float(frame * frame_sec)
        local_bpm, local_conf = _local_bpm_streamed(peak_sec)
        used_global = False
        if local_bpm == 0.0:
            local_bpm, local_conf = global_bpm, global_conf
            used_global = True

        # A "trusted" BPM has to (a) come from the local window with at least
        # some agreement, or (b) fall back to the global tempo which we already
        # verified against three set-wide probes.
        trusted = (local_conf >= GENRE_MATCH_MIN_CONF) or (used_global and global_conf >= 0.45)
        if not filter_active:
            matches = True
        elif not trusted:
            matches = False   # low-confidence BPMs never count as matches
        else:
            matches = _bpm_in_range(local_bpm, eff_lo, eff_hi)

        # Flux sustain — real DJ transitions have SUSTAINED spectral change
        # over 5–10s (a mix window). In-track drops spike then decay in <2s.
        # Compare flux around the peak to flux 8–15s before it: if elevated
        # persists, it's a mix; if it's a spike, it's a drop.
        pre_lo  = max(0, frame - int(15 * frames_per_sec))
        pre_hi  = max(0, frame - int(8  * frames_per_sec))
        mid_lo  = max(0, frame - int(3  * frames_per_sec))
        mid_hi  = min(len(flux_smooth), frame + int(7 * frames_per_sec))
        try:
            baseline = float(np.mean(flux_smooth[pre_lo:pre_hi])) if pre_hi > pre_lo else 0.0
            sustain  = float(np.mean(flux_smooth[mid_lo:mid_hi])) if mid_hi > mid_lo else 0.0
            flux_sustain = max(0.0, sustain - baseline)
        except Exception:
            flux_sustain = 0.0

        candidates_with_meta.append({
            "frame": frame, "score": sc, "peak_sec": peak_sec,
            "local_bpm": local_bpm, "local_bpm_confidence": local_conf,
            "flux_sustain": flux_sustain,
            "genre_match": matches,
        })

    # 9) Split candidates into "matches the selected genre" vs "doesn't", each
    #    sorted by score. We ALWAYS prefer matching clips — even if a higher-
    #    scoring off-genre moment exists. Only fill remaining slots with
    #    non-matching peaks when we don't have num_clips matches. Fallback to
    #    "all" only when ZERO clips match (previously the fallback discarded
    #    genre priority by re-sorting everything together, which is why
    #    picking Afrobeats returned R&B clips from mixed sets).
    matching     = sorted([c for c in candidates_with_meta if c["genre_match"]],
                          key=lambda c: -c["score"])
    non_matching = sorted([c for c in candidates_with_meta if not c["genre_match"]],
                          key=lambda c: -c["score"])
    fallback_used = False

    if not filter_active:
        primary_pool   = matching + non_matching  # no filter → everything is a match
        secondary_pool = []
    elif len(matching) == 0:
        # No matches at all — genuine fallback so user still gets clips.
        print(f"[warn] no moments in BPM range [{eff_lo:.0f}, {eff_hi:.0f}]. "
              f"Falling back to full set.")
        primary_pool   = non_matching
        secondary_pool = []
        fallback_used  = True
    else:
        # Matching pool has priority. Non-matching only fills leftover slots.
        primary_pool   = matching
        secondary_pool = non_matching
        if len(matching) < num_clips:
            print(f"[warn] only {len(matching)} moments in [{eff_lo:.0f}, "
                  f"{eff_hi:.0f}] BPM; padding with {num_clips - len(matching)} best overall.")
            fallback_used = True

    # Minimum spacing between picked clips — bumped from 90s to 180s because
    # 90s wasn't enough on dense sets (multiple drops per section clustered).
    MIN_PICK_GAP_SEC = 180.0

    # -------- Zone-based spread --------
    # Divide the whole set into num_clips equal zones and prefer picking ONE
    # candidate per zone. This guarantees clips are spread across the set even
    # when energy is heavily front-loaded or back-loaded. Fall back to relaxed
    # picking if a zone is empty.
    def _zone_of(peak_sec: float, n_zones: int, total_dur: float) -> int:
        if total_dur <= 0 or n_zones <= 0:
            return 0
        z = int((peak_sec / total_dur) * n_zones)
        return max(0, min(n_zones - 1, z))

    def _pick_by_zones(pool: list[dict], k: int, total_dur: float, seed: int,
                       already_picked: list[dict] | None = None) -> list[dict]:
        """Split the timeline into k zones. In each zone, pick the top candidate
        (or a weighted-random one if seed is set). Zones with no candidate get
        filled at the end from the pool's remaining best."""
        if k <= 0 or not pool:
            return []
        picked = list(already_picked or [])
        picked_keys = {id(c) for c in picked}
        by_zone: dict[int, list[dict]] = {}
        for c in pool:
            if id(c) in picked_keys:
                continue
            z = _zone_of(c["peak_sec"], k, total_dur)
            by_zone.setdefault(z, []).append(c)

        out: list[dict] = []
        rng = None
        if seed:
            import random as _random
            rng = _random.Random(int(seed))

        for z in range(k):
            candidates = sorted(by_zone.get(z, []), key=lambda c: -c["score"])
            if not candidates:
                continue
            if rng and len(candidates) > 1:
                top = candidates[:max(3, min(len(candidates), 5))]
                weights = [max(0.01, float(c["score"])) for c in top]
                pick = rng.choices(top, weights=weights, k=1)[0]
            else:
                pick = candidates[0]
            out.append(pick)

        # Fill leftover slots from zones that had extras (top of any zone).
        remaining = [c for c in pool if id(c) not in {id(x) for x in out + picked}]
        remaining.sort(key=lambda c: -c["score"])
        for c in remaining:
            if len(out) >= k:
                break
            out.append(c)

        return out

    def _pick_from_pool(pool: list[dict], k: int, seed: int,
                        already_picked: list[dict] | None = None) -> list[dict]:
        """Pick up to k clips from `pool`. Enforces MIN_PICK_GAP_SEC spacing
        between picks (relative to each other AND to already_picked). With a
        seed, weighted-random from a wider top-K subpool so re-uploads produce
        a different mix. Without a seed, deterministic top-k by score."""
        if k <= 0 or not pool:
            return []
        picked = list(already_picked or [])

        def _far_enough(cand):
            for p in picked:
                if abs(cand["peak_sec"] - p["peak_sec"]) < MIN_PICK_GAP_SEC:
                    return False
            return True

        # Wider pool so the spacing filter has room to work.
        pool_size = min(len(pool), max(k * 4, k + 8))
        sub = list(pool[:pool_size])

        if seed:
            import random as _random
            rng = _random.Random(int(seed))
            weights = [max(0.01, float(c["score"])) for c in sub]
            out = []
            while len(out) < k and sub:
                # Try to pick something that respects spacing. If nothing
                # respects it, relax and take the top of what's left.
                candidates = [(c, w) for c, w in zip(sub, weights) if _far_enough(c)]
                if not candidates:
                    # Relaxed pick: strongest remaining candidate.
                    pick = max(zip(sub, weights), key=lambda x: x[1])[0]
                else:
                    cands, ws = zip(*candidates)
                    pick = rng.choices(list(cands), weights=list(ws), k=1)[0]
                out.append(pick); picked.append(pick)
                idx = sub.index(pick)
                sub.pop(idx); weights.pop(idx)
            return out
        else:
            # Deterministic: walk down by score, keep only if far enough.
            out = []
            for c in sub:
                if len(out) >= k:
                    break
                if _far_enough(c):
                    out.append(c); picked.append(c)
            # If we didn't fill k slots, take the best remaining without gap.
            if len(out) < k:
                for c in sub:
                    if len(out) >= k:
                        break
                    if c not in out:
                        out.append(c)
            return out

    # Prefer zone-based picking on the primary pool — this GUARANTEES clips
    # come from different sections of the set. Only fall back to the
    # score-based picker (still with 180s spacing) if we don't have enough
    # zones populated.
    selected = _pick_by_zones(primary_pool, num_clips, duration, variation_seed)
    remaining_slots = num_clips - len(selected)
    if remaining_slots > 0 and secondary_pool:
        # Fill leftover slots from the secondary pool, spread across zones too.
        selected += _pick_by_zones(
            secondary_pool, remaining_slots, duration,
            variation_seed + 1 if variation_seed else 0,
            already_picked=selected,
        )
    # Final safety net — if zones can't fill the target (e.g. tiny set), fall
    # back to the classic weighted picker with 180s spacing.
    remaining_slots = num_clips - len(selected)
    if remaining_slots > 0:
        pool_all = primary_pool + secondary_pool
        selected += _pick_from_pool(
            pool_all, remaining_slots,
            variation_seed + 2 if variation_seed else 0,
            already_picked=selected,
        )
    selected.sort(key=lambda c: c["frame"])

    # Clean up the temporary pre-decoded WAV files (best-effort)
    for tmp in (main_wav, bass_wav):
        try: os.remove(tmp)
        except Exception: pass

    # 10) Build clip segments — center peak with lead_in_sec from clip start
    clips = []
    for rank, c in enumerate(selected):
        frame = c["frame"]
        peak_sec = c["peak_sec"]
        start = max(0.0, peak_sec - lead_in_sec)
        end = min(duration, start + clip_len_sec)
        if end - start < clip_len_sec:
            start = max(0.0, end - clip_len_sec)
        clips.append({
            "rank": rank + 1,
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "peak_sec": round(peak_sec, 2),
            "score": round(float(c["score"]), 4),
            "energy": round(float(rms_smooth[frame]), 4),
            "flux": round(float(flux_smooth[frame]), 4),
            "flux_sustain": round(float(c.get("flux_sustain") or 0), 4),
            "bpm": round(global_bpm, 1),
            "bpm_confidence": round(global_conf, 2),
            "local_bpm": round(c["local_bpm"], 1),
            "local_bpm_confidence": round(c["local_bpm_confidence"], 2),
            "genre_match": bool(c["genre_match"]),
            "genre_requested": genre,
            "genre_fallback_used": fallback_used,
        })

    return clips


# ---------------- Tagging ----------------

HOOK_POOL = {
    "PEAK": [
        "this drop hits different",
        "when the bass took over the room",
        "watch the energy change at the drop",
        "this is the moment everyone screamed",
        "drop of the night",
        "the second the room went off",
        "they didn't see this drop coming",
        "this one stays in the head all week",
    ],
    "TRANSITION": [
        "this transition is illegal",
        "name a smoother blend",
        "watch this blend land",
        "the way these tracks fit together",
        "this mix is a crime scene",
        "transition you have to rewind",
        "two tracks, one perfect blend",
        "this is the kind of mix you save",
    ],
    "BUILD": [
        "feel that build coming",
        "the tension before the drop",
        "wait for it",
        "you can feel the room hold its breath",
        "this build was made for the floor",
        "the calm before the chaos",
        "patience pays off in three… two…",
        "the moment before the moment",
    ],
    "MOMENT": [
        "this moment though",
        "vibes locked in",
        "the feel of this section",
        "this is what the booth sees",
        "energy you can't fake",
        "this is the part you replay",
        "captured the vibe right here",
        "this stretch though",
    ],
    "VIBE": [
        "pure vibe right here",
        "just let it ride",
        "this is the cruise mode",
        "set this on loop",
        "tap in for a second",
        "this stretch felt right",
        "press play and float",
        "this loop is what summer sounds like",
    ],
}


def _resolve_tag(c: dict) -> str:
    """Assign a tag that actually reflects what's happening in the audio.

    Prior version tagged anything with a flux spike as TRANSITION, so 90 % of
    in-track drops got mislabeled. Real DJ transitions have SUSTAINED spectral
    change over a mix window (5-10s) — captured in `flux_sustain`, which is
    the elevation of flux at the peak relative to 8-15s earlier.
    """
    energy   = c.get("energy", 0) or 0
    flux     = c.get("flux", 0) or 0
    sustain  = c.get("flux_sustain", 0) or 0

    # Sustain >= 0.15 means the spectrum stayed noticeably different from a
    # baseline 8-15s earlier — that's a real mix, not a snare hit.
    if sustain >= 0.15 and flux >= 0.55:
        return "TRANSITION"
    # PEAK = high energy + high flux at the moment. In-track drops land here.
    if flux > 0.75 and energy > 0.65:
        return "PEAK"
    if energy > 0.75:
        return "BUILD"
    if energy > 0.5:
        return "MOMENT"
    return "VIBE"


def assign_hooks(clips: list[dict]) -> list[dict]:
    """Tag each clip and assign a UNIQUE hook within each tag bucket.
    With 8 hooks per tag and ~8 clips per upload, we get guaranteed uniqueness."""
    # First pass: tag everything
    for c in clips:
        c["tag"] = _resolve_tag(c)

    # Group clips by tag, shuffle the pool deterministically per upload,
    # then assign hooks in order so no two clips of the same tag share one.
    tag_to_clips = {}
    for c in clips:
        tag_to_clips.setdefault(c["tag"], []).append(c)

    for tag, group in tag_to_clips.items():
        pool = list(HOOK_POOL[tag])
        # Deterministic shuffle based on first clip's peak time → reproducible
        seed = int(group[0].get("peak_sec", 0) * 13) + len(group)
        random.Random(seed).shuffle(pool)
        for i, c in enumerate(group):
            c["hook"] = pool[i % len(pool)]
    return clips


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input audio or video file (mp3, wav, m4a, mp4, mov, ...)")
    ap.add_argument("--num-clips", type=int, default=5)
    ap.add_argument("--clip-length", type=float, default=30.0)
    ap.add_argument("--genre", default="all",
                    choices=list(GENRE_BPM_RANGES.keys()),
                    help="Filter peaks to a genre's BPM range. Use 'all' for no filter.")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent / "reelcrate_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] decoding {args.input} ...")
    print(f"[analyze] genre filter: {GENRE_PRETTY.get(args.genre, args.genre)}")
    wav_path, is_temp = ensure_wav(args.input)
    try:
        clips = detect_drops(wav_path, num_clips=args.num_clips,
                             clip_len_sec=args.clip_length, genre=args.genre)
    finally:
        if is_temp:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    clips = assign_hooks(clips)

    manifest = {
        "source": str(Path(args.input).resolve()),
        "num_clips": len(clips),
        "clip_length_sec": args.clip_length,
        "clips": clips,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n[analyze] found {len(clips)} moments. Manifest -> {manifest_path}")
    for c in clips:
        match_mark = "✓" if c.get("genre_match") else "·"
        print(f"  #{c['rank']} {c['tag']:<11} {match_mark}  "
              f"start {c['start_sec']:>7.2f}s  peak {c['peak_sec']:>7.2f}s  "
              f"score {c['score']:.3f}  local_bpm {c.get('local_bpm', 0):.1f}  "
              f"— {c['hook']}")
    if clips and clips[0].get("genre_fallback_used"):
        print(f"\n[note] genre filter '{args.genre}' didn't match enough peaks; "
              f"results include best overall moments as fallback.")


if __name__ == "__main__":
    main()
