"""Conservative, persistent musical-boundary detection for continuous DJ sets.

Unlike the energy picker, this compares harmony and timbre on both sides of a
boundary. Loudness is not a transition feature. Repeated material across a
boundary, brief fills, silence, and changes that do not fit the requested clip
are rejected. It is still a signal-based estimate, not a song-ID service.
"""
import os
import random
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

FEATURE_STEP = .5


def extract_features(wav_path):
    """Bounded-memory half-second features, aligned across minute boundaries."""
    chroma, timbre, rms = [], [], []
    with sf.SoundFile(wav_path) as f:
        sr = f.samplerate
        duration = f.frames / sr
        for second in range(0, int(duration) + 1, 60):
            f.seek(second * sr)
            y = f.read(sr * 60, dtype='float32')
            if len(y) < 2048:
                break
            power = np.abs(librosa.stft(y, n_fft=2048, hop_length=256)) ** 2
            ch = librosa.feature.chroma_stft(S=power, sr=sr, n_fft=2048,
                                           hop_length=256, tuning=0)
            mel = librosa.feature.melspectrogram(S=power, sr=sr, n_fft=2048, n_mels=40)
            log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)
            # Discard coefficient zero (overall gain).
            mf = librosa.feature.mfcc(S=log_mel, n_mfcc=14)[1:]
            times = librosa.frames_to_time(np.arange(power.shape[1]), sr=sr, hop_length=256)
            for start in np.arange(0, len(y) / sr - .25, FEATURE_STEP):
                mask = (times >= start) & (times < start + FEATURE_STEP)
                chroma.append(ch[:, mask].mean(axis=1))
                timbre.append(mf[:, mask].mean(axis=1))
                samples = y[int(start * sr):int((start + FEATURE_STEP) * sr)]
                rms.append(float(np.sqrt(np.mean(samples ** 2))))
    return dict(chroma=np.array(chroma), mfcc=np.array(timbre), rms=np.array(rms),
                duration=duration)


def _cosine(a, b):
    return float(np.clip(1 - np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9), 0, 2))


def find_boundaries(features):
    ch = np.asarray(features['chroma'], dtype=float)
    mf = np.asarray(features['mfcc'], dtype=float)
    rms = np.asarray(features['rms'], dtype=float)
    if len(ch) < 130 or not np.isfinite(ch).all() or float(np.max(rms)) < .001:
        return []
    ch /= np.maximum(np.linalg.norm(ch, axis=1, keepdims=True), 1e-8)
    mf = (mf - np.median(mf, axis=0)) / np.maximum(np.std(mf, axis=0), 3)
    def avg(x, a, b):
        return x[max(0, a):min(len(x), b)].mean(axis=0)
    def distance(a, ta, b, tb):
        return _cosine(a, b) * 10 + float(np.mean((ta - tb) ** 2)) * .4

    rows = []
    for i in range(64, len(ch) - 64):
        a, b = avg(ch, i-48, i-12), avg(ch, i+12, i+48)
        ta, tb = avg(mf, i-48, i-12), avg(mf, i+12, i+48)
        tonal = _cosine(a, b)
        texture = float(np.mean((ta-tb) ** 2))
        within = (_cosine(avg(ch, i-48, i-30), avg(ch, i-30, i-12)) +
                  _cosine(avg(ch, i+12, i+30), avg(ch, i+30, i+48))) / 2
        farther = _cosine(avg(ch, i-64, i-40), avg(ch, i+40, i+64))
        score = (tonal * 10 + texture * .4) * max(0, 1 - within / (tonal + 1e-6))
        rows.append(dict(frame=i, tonal=tonal, timbre=texture, within=within,
                         farther=farther, score=score))
    scores = gaussian_filter1d(np.array([r['score'] for r in rows]), 2)
    peaks, _ = find_peaks(scores, distance=60, prominence=.12)
    result = []
    for p in peaks:
        r = rows[p]
        i = r['frame']
        # Two independent cues and persistent separation are required. A single
        # loud hit, key flourish, or instrument entrance cannot fill the quota.
        if (r['tonal'] < .035 or r['timbre'] < .18 or r['score'] < .65 or
                r['within'] > r['tonal'] * .2 or r['farther'] < r['tonal'] * .35):
            continue
        if min(np.median(rms[i-48:i-12]), np.median(rms[i+12:i+48])) < .003:
            continue
        a, b = avg(ch, i-48, i-12), avg(ch, i+12, i+48)
        ta, tb = avg(mf, i-48, i-12), avg(mf, i+12, i+48)
        cross = distance(a, ta, b, tb)
        prior, future, recent_prior = [], [], []
        for k in range(16, 181, 8):
            if i-k-24 >= 0:
                similarity = distance(avg(ch, i-k-24, i-k), avg(mf, i-k-24, i-k), b, tb)
                prior.append(similarity)
                if k <= 80:
                    recent_prior.append(similarity)
            if i+k+24 < len(ch):
                future.append(distance(avg(ch, i+k, i+k+24), avg(mf, i+k, i+k+24), a, ta))
        # Verse/chorus changes often return to the same musical material on
        # BOTH sides. This also rejects a brief effect followed by the old song.
        if prior and future and max(min(prior), min(future)) < cross * .35:
            continue
        # A chorus can look like a new song even when its musical profile
        # occurred only seconds ago. Reject that return without requiring the
        # outgoing verse to appear again after the candidate boundary.
        if recent_prior and future and min(recent_prior) < cross * .4 and min(future) < cross * .8:
            continue
        # Refine the handover using a smoothed trajectory between the two
        # musical profiles, instead of cutting around the loudest frame.
        trajectory = []
        indices = range(i-40, i+41)
        for j in indices:
            c, t = avg(ch, j-3, j+4), avg(mf, j-3, j+4)
            da, db = distance(c, t, a, ta), distance(c, t, b, tb)
            trajectory.append(da / max(da + db, 1e-8))
        trajectory = gaussian_filter1d(trajectory, 1)
        crossings = [k for k in range(1, len(trajectory))
                     if trajectory[k-1] < .5 <= trajectory[k]]
        if not crossings:
            continue
        mid = min(crossings, key=lambda k: abs(k-40))
        left = [k for k in range(mid) if trajectory[k] <= .3]
        right = [k for k in range(mid+1, len(trajectory)) if trajectory[k] >= .7]
        if not left or not right:
            continue
        lo, hi = left[-1], right[0]
        # Require stable incoming and outgoing music around the handover.
        if np.mean(trajectory[max(0, lo-6):lo+1]) > .35 or np.mean(trajectory[hi:hi+7]) < .65:
            continue
        # The feature midpoint can lag the first incoming vocal or an echo-out.
        # Preserve the coarse change region too, not just its steepest instant.
        # A timbre-dominated change may be the end of a spoken intro or a
        # stripped-down blend, rather than its beginning. Keep the preceding
        # phrase too; these longer handovers deliberately do not fit 30s clips.
        pre_roll = 28 if r['timbre']*.4 > r['tonal']*20 else 6
        result.append(dict(transition_start_sec=min((i-40+lo)*FEATURE_STEP, i*FEATURE_STEP-pre_roll),
                           transition_end_sec=max((i-40+hi)*FEATURE_STEP, i*FEATURE_STEP+6),
                           peak_sec=(i-40+mid)*FEATURE_STEP,
                           score=round(r['score'], 4),
                           transition_evidence={k:round(r[k], 4) for k in ('tonal','timbre','within','farther')},
                           selection_mode='transitions'))
    return sorted(result, key=lambda c: -c['score'])


def select_transition_clips(boundaries, duration, num_clips, clip_len_sec, variation_seed=0):
    """Keep the entire estimated handover plus >=4s of each side. No filler."""
    context = max(4, min(10, clip_len_sec * .2))
    candidates = list(boundaries)
    if variation_seed:
        rng = random.Random(variation_seed)
        candidates.sort(key=lambda c: -(c['score'] * rng.uniform(.9, 1.1)))
    selected = []
    for c in candidates:
        lo, hi = c['transition_start_sec'], c['transition_end_sec']
        if hi-lo + context*2 > clip_len_sec:
            continue
        start = max(0, min(duration-clip_len_sec, (lo+hi-clip_len_sec)/2))
        end = start+clip_len_sec
        if start > lo-context or end < hi+context:
            continue
        if any(start < x['end_sec'] and end > x['start_sec'] for x in selected):
            continue
        selected.append({**c, 'start_sec':round(start, 2), 'end_sec':round(end, 2)})
        if len(selected) >= num_clips:
            break
    for rank, c in enumerate(sorted(selected, key=lambda c:c['start_sec']), 1):
        c['rank'] = rank
    return sorted(selected, key=lambda c:c['rank'])


def detect_transitions(audio_path, num_clips=5, clip_len_sec=30, genre='all',
                       variation_seed=0, bpm_min=0, bpm_max=0):
    from analyze import GENRE_BPM_RANGES, _stable_bpm, _bass_filter, _bpm_for_range
    if clip_len_sec < 30:
        raise ValueError('Choose 30 or 60 seconds for music transitions so both songs have room. Use Energy moments for 15-second clips.')
    bpm_range = (bpm_min, bpm_max) if bpm_min > 0 and bpm_max > bpm_min else GENRE_BPM_RANGES.get(genre)
    with tempfile.TemporaryDirectory(prefix='reelcrate-transition-') as directory:
        wav = str(Path(directory)/'audio.wav')
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(audio_path),'-vn','-ac','1',
                        '-ar','11025','-c:a','pcm_s16le',wav], check=True)
        features = extract_features(wav)
        boundaries = find_boundaries(features)
        # Estimate tempo separately on each side. A crossfade is precisely the
        # place where a single whole-clip BPM may be misleading.
        for c in boundaries:
            tempos = []
            with sf.SoundFile(wav) as f:
                for start in (c['transition_start_sec']-20, c['transition_end_sec']+2):
                    f.seek(max(0,int(start*f.samplerate)))
                    y = f.read(int(18*f.samplerate),dtype='float32')
                    bpm, confidence = _stable_bpm(_bass_filter(y, f.samplerate), f.samplerate)
                    adjusted = _bpm_for_range(bpm,*bpm_range) if bpm_range else bpm
                    tempos.append((adjusted, confidence, bpm))
            c['tempo_before'], c['tempo_after'] = [dict(bpm=round(b,1) if cf>=.55 and b else None,
                confidence=round(cf,2)) for b,cf,_ in tempos]
            if bpm_range and any(cf < .55 or not b for b,cf,_ in tempos):
                c['genre_match'] = False
                continue
            # Only show one number when both sides support that same tempo.
            a,b=tempos
            same = a[0] and b[0] and min(a[1],b[1])>=.55 and abs(a[0]-b[0])<=3
            c.update(local_bpm=round((a[0]+b[0])/2,1) if same else 0,
                     local_bpm_confidence=round(min(a[1],b[1]),2) if same else 0,
                     bpm=0,bpm_confidence=0,genre_match=True,genre_requested=genre,
                     bpm_octave_adjusted=any(abs(x[0]-x[2])>1 for x in tempos))
        matched = [c for c in boundaries if c.get('genre_match')]
        clips = select_transition_clips(matched, features['duration'], num_clips, clip_len_sec, variation_seed)
        if not clips:
            raise ValueError('No clear music transitions fit these settings. Try 30 or 60 seconds, a wider BPM range, or Energy moments. We did not substitute ordinary peaks for transitions.')
        return clips
