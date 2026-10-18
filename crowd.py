"""Crowd-facing video: distributed motion after compensating camera movement.

An activity estimate, not emotion, applause or person recognition. The user
selects this mode for audience footage. No ordinary audio peaks are substituted.
"""
import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


def validate_region(region):
    if region is None: return None
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise ValueError('Choose a rectangular crowd area.')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in region):
        raise ValueError('Crowd area coordinates must be finite numbers.')
    x,y,w,h=map(float,region)
    if x<0 or y<0 or w<.15 or h<.15 or x+w>1.000001 or y+h>1.000001:
        raise ValueError('Keep the crowd area inside the video and at least 15% wide and high.')
    return [x,y,w,h]


def region_filter(region):
    if region is None: return ''
    x,y,w,h=validate_region(region)
    return f'crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},'


def motion_evidence(previous, current):
    # Match local features then remove camera translation/rotation/zoom.
    points = cv2.goodFeaturesToTrack(previous, 240, .015, 5)
    if points is None or len(points) < 12:
        return 0.0, 0.0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if nxt is None:
        return 0.0, 0.0
    a, b = points[status.ravel() == 1].reshape(-1, 2), nxt[status.ravel() == 1].reshape(-1, 2)
    if len(a) < 12:
        return 0.0, 0.0
    transform, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if transform is None or inliers.mean() < .35:
        return 0.0, 0.0  # cuts, unstable frames, or no reliable camera reference
    aligned = cv2.warpAffine(previous, transform, (current.shape[1], current.shape[0]))
    # Ignore image edges exposed by camera motion. Equalization reduces lighting changes.
    flow = cv2.calcOpticalFlowFarneback(aligned, current, None, .5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.linalg.norm(flow, axis=2)[12:-12, 12:-12]
    tiles = [np.mean(tile > 1.2) for row in np.array_split(magnitude, 3) for tile in np.array_split(row, 4, axis=1)]
    active = np.array(tiles) > .12
    coverage = float(np.mean(active))
    # A single moving object, or a flash with no tracked texture, is not enough.
    score = float(np.mean(np.minimum(magnitude, 8))) * coverage if active.sum() >= 3 else 0.0
    return score, coverage


def scan_video(path, region=None):
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration:stream=codec_type','-of','json',str(path)]))
    if not any(s.get('codec_type') == 'video' for s in probe.get('streams', [])):
        raise ValueError('Crowd reactions needs a video pointed at the audience. Choose Energy moments for audio files.')
    duration = float(probe['format']['duration'])
    # Fixed 320x180 analysis canvas. Sequential decode; bounded memory even for long sets.
    cmd=['ffmpeg','-v','error','-i',str(path),'-an','-vf',region_filter(region)+'fps=3,scale=320:180',
         '-pix_fmt','gray','-f','rawvideo','pipe:1']
    scores, coverage = [], []
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors)
        previous = None
        try:
            while True:
                data = process.stdout.read(320*180)
                if not data:
                    process.wait()
                    break
                if len(data) != 320*180: raise ValueError('Could not decode the full crowd video.')
                gray = np.frombuffer(data, np.uint8).reshape(180,320)
                gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(gray)
                score, area = motion_evidence(previous, gray) if previous is not None else (0.,0.)
                scores.append(score); coverage.append(area); previous = gray
        finally:
            process.stdout.close()
            if process.poll() is None: process.terminate()
            process.wait()
        if process.returncode: raise ValueError('Could not decode the crowd video. Try exporting it as MP4.')
    return dict(scores=scores, coverage=coverage, duration=duration, fps=3)


def select_reactions(features, num_clips, clip_len_sec, all_candidates=False):
    fps=features['fps']; values=np.asarray(features['scores'])
    if len(values)<fps*4: return []
    smooth=median_filter(values, size=5)
    threshold=max(.12, float(np.median(smooth)+1.5*np.median(np.abs(smooth-np.median(smooth)))))
    peaks,_=find_peaks(smooth, height=threshold, prominence=.06, distance=fps*5)
    candidates=[]
    for p in peaks:
        # Require sustained distributed activity, not one isolated frame.
        if np.mean(smooth[max(0,p-3):p+4] >= threshold*.7)<.7: continue
        peak=p/fps; start=peak-clip_len_sec*.4; end=start+clip_len_sec
        if start<0 or end>features['duration']: continue
        candidates.append(dict(start_sec=round(start,3),end_sec=round(end,3),peak_sec=round(peak,3),
            score=float(smooth[p]),selection_mode='crowd',
            crowd_evidence={'movement_score':round(float(smooth[p]),3),'frame_coverage':round(features['coverage'][p],3)},
            energy=0,flux=0,bpm=0,bpm_confidence=0,local_bpm=0,local_bpm_confidence=0,genre_match=True))
    if all_candidates: return candidates
    selected=[]
    for c in sorted(candidates,key=lambda c:-c['score']):
        if all(c['end_sec']<=s['start_sec'] or c['start_sec']>=s['end_sec'] for s in selected):
            selected.append(c)
            if len(selected)>=num_clips: break
    selected.sort(key=lambda c:c['start_sec'])
    for rank,c in enumerate(selected,1): c['rank']=rank
    return selected


def detect_crowd(path,num_clips=5,clip_len_sec=30,genre='all',variation_seed=0,bpm_min=0,bpm_max=0,crowd_region=None):
    from analyze import GENRE_BPM_RANGES, _stable_bpm, _bass_filter, _bpm_for_range
    import soundfile as sf
    crowd_region=validate_region(crowd_region)
    features=scan_video(path,crowd_region)
    # Apply tempo filters before selecting/counting, so weaker eligible moments can fill the request.
    candidates=select_reactions(features, num_clips,clip_len_sec,all_candidates=True)
    bpm_range=(bpm_min,bpm_max) if bpm_min>0 and bpm_max>bpm_min else GENRE_BPM_RANGES.get(genre)
    accepted=[]
    with tempfile.TemporaryDirectory(prefix='reelcrate-crowd-') as directory:
        for c in sorted(candidates,key=lambda c:-c['score']):
            if any(c['start_sec']<s['end_sec'] and c['end_sec']>s['start_sec'] for s in accepted): continue
            c['crowd_region']=crowd_region
            wav=str(Path(directory)/'tempo.wav')
            subprocess.run(['ffmpeg','-y','-v','error','-ss',str(max(0,c['peak_sec']-9)),
                '-i',str(path),'-t','18','-vn','-ac','1','-ar','11025',wav],check=True)
            y,sr=sf.read(wav,dtype='float32')
            bpm,confidence=_stable_bpm(_bass_filter(y,sr),sr)
            adjusted=_bpm_for_range(bpm,*bpm_range) if bpm_range else bpm
            if bpm_range and (confidence<.55 or not adjusted): continue
            c.update(local_bpm=round(adjusted,1) if confidence>=.55 else 0,
                     local_bpm_confidence=round(confidence,2),genre_requested=genre,
                     bpm_octave_adjusted=bool(adjusted and abs(adjusted-bpm)>1))
            accepted.append(c)
            if len(accepted)>=num_clips: break
    accepted.sort(key=lambda c:c['start_sec'])
    for rank,c in enumerate(accepted,1): c['rank']=rank
    if not accepted:
        raise ValueError('No clear crowd movement fits these settings. Try a wider BPM range or footage with more visible audience activity. Music peaks were not substituted.')
    return accepted
