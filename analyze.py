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
GENRE_BPM_RANGES = {
    "hiphop":    (75, 100),
    "trap":      (130, 175),   # often double-time of half-time
    "reggae":    (60, 90),
    "dancehall": (88, 110),
    "afrobeats": (95, 115),
    "afrohouse": (115, 125),
    "amapiano":  (108, 116),
    "reggaeton": (88, 100),
    "house":     (120, 128),
    "techhouse": (122, 130),
    "deephouse": (118, 125),
    "disco":     (110, 125),
    "techno":    (128, 140),
    "trance":    (130, 145),
    "hardstyle": (145, 165),
    "dnb":       (160, 185),
    "dubstep":   (138, 145),
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


def detect_drops(audio_path: str, num_clips: int = 5, clip_len_sec: float = 30.0,
                 lead_in_sec: float = 6.0, genre: str = "all") -> list[dict]:
    """
    Returns a list of clip dicts:
      [{start_sec, end_sec, score, peak_sec, energy, flux, bpm, local_bpm, genre_match}, ...]
    sorted by start_sec ascending.

    Strategy:
      - Energy (RMS) finds loud sections.
      - Spectral flux spikes find "drop" moments where the mix suddenly changes.
      - Combine via weighted sum, then pick top-N non-overlapping peaks.
      - When genre is set, compute local BPM near each peak and filter to peaks
        whose tempo matches the genre's BPM range (or half/double).
      - Each clip is positioned so the peak sits at lead_in_sec from clip start
        (build-up → drop pattern works better for Reels).
    """
    # Use 11025 Hz mono — plenty for drop/onset detection, halves memory vs 22050
    # so we can analyze 2+ hour sets without OOM kills in constrained environments.
    SR = 11025
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(y) / sr
    if duration < clip_len_sec * 2:
        print(f"[warn] track is short ({duration:.1f}s); reducing num_clips")
        num_clips = max(1, min(num_clips, int(duration // clip_len_sec)))

    hop = 256  # half hop preserves time resolution at lower sample rate
    frame_sec = hop / sr

    # 1) RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_smooth = librosa.util.normalize(
        np.convolve(rms, np.ones(int(2.0 / frame_sec)) / int(2.0 / frame_sec), mode="same")
    )

    # 2) Spectral flux (sum of positive frame-to-frame magnitude changes)
    # n_fft=1024 cuts STFT memory in half vs default 2048
    S = np.abs(librosa.stft(y, hop_length=hop, n_fft=1024))
    flux = np.maximum(0, np.diff(S, axis=1)).sum(axis=0)
    flux = np.concatenate([[0], flux])
    del S  # free memory before next allocations
    flux_smooth = librosa.util.normalize(
        np.convolve(flux, np.ones(int(1.0 / frame_sec)) / int(1.0 / frame_sec), mode="same")
    )

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

    # 7) Global BPM on bass-filtered signal (used as fallback / metadata)
    try:
        y_bass_global = _bass_filter(y, sr)
        global_bpm, global_conf = _stable_bpm(y_bass_global, sr)
    except Exception:
        global_bpm, global_conf = 0.0, 0.0

    # 8) Compute local BPM around each candidate peak; apply genre filter
    candidates_with_meta = []
    for frame, sc in ranked_candidates:
        peak_sec = float(frame * frame_sec)
        local_bpm, local_conf = _local_bpm(y, sr, peak_sec)
        if local_bpm == 0.0:
            local_bpm, local_conf = global_bpm, global_conf
        matches = _bpm_matches_genre(local_bpm, genre)
        candidates_with_meta.append({
            "frame": frame, "score": sc, "peak_sec": peak_sec,
            "local_bpm": local_bpm, "local_bpm_confidence": local_conf,
            "genre_match": matches,
        })

    # 9) Apply genre filter — but fall back to unfiltered if too few match
    filtered = [c for c in candidates_with_meta if c["genre_match"]]
    fallback_used = False
    if genre != "all" and len(filtered) < num_clips:
        if len(filtered) == 0:
            print(f"[warn] no moments matched genre '{genre}'. Falling back to all genres.")
            filtered = candidates_with_meta
            fallback_used = True
        else:
            # Partial match: fill remaining slots with best non-matching peaks
            print(f"[warn] only {len(filtered)} {genre} moments found; padding with best overall.")
            remaining = [c for c in candidates_with_meta if not c["genre_match"]]
            filtered = filtered + remaining
            fallback_used = True

    # Take top-N from filtered, then re-sort to time order
    selected = sorted(filtered, key=lambda c: -c["score"])[:num_clips]
    selected.sort(key=lambda c: c["frame"])

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
    energy = c["energy"]
    flux = c["flux"]
    if flux > 0.75 and energy > 0.65:
        return "PEAK"
    if flux > 0.65:
        return "TRANSITION"
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
