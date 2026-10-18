#!/usr/bin/env python3
"""
Reelcrate clip renderer.

Reads a manifest.json produced by analyze.py and renders each clip into a
ready-to-post 9:16 1080x1920 MP4 with:
  - original stereo audio
  - animated waveform strip (yellow)
  - bold caption hook (centered, bottom third)
  - REEL/CRATE watermark (top-left)
  - optional DJ attribution
  - dark club-style overlay

Usage:
    python3 render.py <manifest.json> [--source <override_input>] [--watermark @handle]
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


FONT_BOLD = next((p for p in [os.environ.get("REELCRATE_FONT", ""),
    "/usr/share/fonts/truetype/lato/Lato-Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if Path(p).is_file()), "")
FONT_TXT = "/usr/share/fonts/truetype/lato/Lato-Heavy.ttf"
YELLOW = "0xFFE600"
WHITE = "0xFFFFFF"

W, H = 1080, 1920  # 9:16 vertical
FPS = 30
TRACK_CLOCK = f"setpts=PTS-STARTPTS,fps={FPS}"


# ----------------- Face-aware smart crop ---------------------------------
#
# When we cut a 9:16 window out of a landscape video, a naive center crop
# slices off whoever is standing to one side of frame — DJs behind the
# booth, dancers in the corner, reactions on the flank. This helper
# follows the foreground face throughout each shot. Camera cuts reset the
# track; missing evidence preserves the full shot instead of guessing.

FACE_MODEL = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")


def _crop_track(observations, duration, crop_width, cuts=()):
    """Build a bounded, smooth path; never interpolate across camera cuts.

    Observations are (seconds, center, face width), including None for misses.
    A long blind interval or a face wider than the crop requires a wide view.
    """
    import numpy as np
    boundaries = [0.] + sorted(t for t in cuts if 0 < t < duration) + [duration]
    points = []
    half = crop_width / 2
    for begin, end in zip(boundaries, boundaries[1:]):
        samples = [o for o in observations if begin <= o[0] < end]
        known = [o for o in samples if o[1] is not None]
        if not known or len(known) < len(samples) * .6:
            return None
        times = [begin] + [o[0] for o in known] + [end]
        if max(b-a for a,b in zip(times,times[1:])) > 1.5:
            return None
        centers = []
        for i, (t, cx, width) in enumerate(known):
            if width + .06 > crop_width:
                return None
            # Small median smoothing suppresses detection jitter. Clamp it to
            # a safe interval so smoothing cannot push the face off screen.
            smooth = float(np.median([o[1] for o in known[max(0,i-1):i+2]]))
            lo = max(half, cx + width/2 + .03 - half)
            hi = min(1-half, cx - width/2 - .03 + half)
            if lo > hi:
                return None
            centers.append((t, max(lo,min(hi,smooth))))
        points.append((begin, centers[0][1]))
        points.extend((t,x) for t,x in centers if begin < t < end)
        # A tiny interval gives an immediate cut instead of a pan through
        # unrelated scenery; it is shorter than one output frame.
        points.append((max(begin,end-.0001), centers[-1][1]))
    return points


def _crop_commands(points, scaled_width, duration, path):
    """Use bounded numeric commands, not an expression growing with duration.

    Long crop expressions are rejected by some production FFmpeg builds.
    sendcmd applies the interpolated position on each frame without that limit.
    """
    import numpy as np
    times=np.arange(max(1,int(duration*FPS)+1))/FPS
    centers=np.interp(times,[t for t,_ in points],[x for _,x in points])
    offsets=np.clip(centers*scaled_width-W/2,0,max(0,scaled_width-W))
    # Round safely before the intended frame. Rounding 11/30 upward to
    # 0.366667 would otherwise delay the camera-cut command by one frame.
    path.write_text("\n".join(f"{max(0,t-.000001):.6f} crop@follow x {int(x)};" for t,x in zip(times,offsets)))
    return int(offsets[0])


def _track_shot(samples, frame_width, prefer_center=True):
    """Anchor on the clearest foreground face in the shot, then track both ways.

    Starting at the first frame is unsafe: a camera may cut while the DJ is
    looking down, leaving only a background face detectable in that frame.
    """
    import numpy as np
    # A nearby spectator at the edge can have a larger face than the DJ.
    # Prefer a prominent central subject when choosing the shot anchor;
    # subsequent tracking follows that subject even as they move off-center.
    largest_area = max((float(f[2]*f[3]) for _,_,faces in samples for f in faces), default=0)
    choices=[(float(f[2]*f[3]) *
              (1 if not prefer_center or .15 <= float((f[0]+f[2]/2)/frame_width) <= .85 else .25),i,f)
             for i,(_,_,faces) in enumerate(samples) for f in faces
             if float(f[2]*f[3]) >= largest_area * .65]
    if not choices:
        return [(t,None,0.) for t,_,_ in samples]
    _,seed,seed_box=max(choices,key=lambda item:item[0])
    observations={seed:(samples[seed][0],float((seed_box[0]+seed_box[2]/2)/frame_width),float(seed_box[2]/frame_width))}
    for direction in (-1,1):
        last_box=np.array(seed_box,dtype=np.float32)
        last_gray=samples[seed][1];last_seen=samples[seed][0]
        for i in range(seed+direction,len(samples) if direction==1 else -1,direction):
            t,gray,faces=samples[i]
            predicted=_flow_face_box(last_gray,gray,last_box)
            anchor=predicted if predicted is not None else last_box
            chosen=_select_foreground_face(faces,float((anchor[0]+anchor[2]/2)/frame_width),
                                           frame_width,float(anchor[2]*anchor[3]))
            if chosen is not None:
                last_seen=t
            elif predicted is not None and abs(t-last_seen)<=4.:
                chosen=predicted
            elif abs(t-last_seen)<=.75:
                chosen=last_box
            if chosen is None:
                observations[i]=(t,None,0.)
                if predicted is not None:
                    last_box=predicted
            else:
                x,y,w,h=chosen
                observations[i]=(t,float((x+w/2)/frame_width),float(w/frame_width))
                last_box=np.array(chosen,dtype=np.float32)
            last_gray=gray
    return [observations[i] for i in range(len(samples))]


def _face_crop_plan(source, start, duration, src_w, src_h):
    """Check every frame for shot changes; detect and track faces at 4 Hz.

    Decode through a temporary small video. Retain only sparse gray frames
    (about 55 MB for 60 seconds of landscape footage); all work stays local.
    """
    if src_w * H <= src_h * W:
        return [(0., .5)], {"coverage": 1., "shots": 1}
    import cv2
    import numpy as np
    detector=cv2.FaceDetectorYN.create(str(FACE_MODEL),"",(640,360),.8,.3,1000)
    observations,cuts,shots=[],[],[[]]
    with tempfile.TemporaryDirectory(prefix="reelcrate-track-") as folder:
        sample=str(Path(folder)/"tracking.mp4")
        subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-ss",str(start),
                        "-t",str(duration),"-i",source,"-an","-vf",f"{TRACK_CLOCK},scale=640:-2,setsar=1",
                        "-c:v","libx264","-preset","ultrafast","-crf","18",sample],
                       check=True,capture_output=True,timeout=max(45,duration*2))
        capture=cv2.VideoCapture(sample)
        try:
            fps=capture.get(cv2.CAP_PROP_FPS) or 30
            previous=previous_hist=None;last_detection=-1.;index=0
            while True:
                ok,frame=capture.read()
                if not ok:
                    break
                t=index/fps;index+=1
                if t>=duration:
                    break
                small=cv2.resize(frame,(64,36))
                hist=cv2.calcHist([small],[0,1,2],None,[8,8,8],[0,256]*3)
                cv2.normalize(hist,hist)
                cut=previous is not None and (
                    float(np.mean(cv2.absdiff(small,previous)))>40 or
                    (float(np.mean(cv2.absdiff(small,previous)))>32 and
                     cv2.compareHist(previous_hist,hist,cv2.HISTCMP_BHATTACHARYYA)>.35))
                previous,previous_hist=small,hist
                if cut:
                    cuts.append(t);shots.append([])
                if not cut and t-last_detection<.25 and t<duration-1/fps:
                    continue
                last_detection=t
                fh,fw=frame.shape[:2];detector.setInputSize((fw,fh))
                _,detected=detector.detect(frame)
                faces=[] if detected is None else list(detected[:,:4])
                shots[-1].append((t,cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),faces))
        finally:
            capture.release()
    for shot in shots:
        if shot:
            observations.extend(_track_shot(shot,640))
    coverage=sum(o[1] is not None for o in observations)/max(1,len(observations))
    plan=_crop_track(observations,duration,min(1.,src_h*W/H/src_w),cuts)
    if plan is None:
        # Centrality is only a preference: a DJ can legitimately be near an
        # edge. Try the prominent foreground anchor before abandoning crop.
        alternate = [o for shot in shots if shot for o in _track_shot(shot,640,False)]
        alternate_plan = _crop_track(alternate,duration,min(1.,src_h*W/H/src_w),cuts)
        if alternate_plan is not None:
            observations,plan = alternate,alternate_plan
            coverage=sum(o[1] is not None for o in observations)/max(1,len(observations))
    info={"coverage":round(coverage,3),"shots":len(cuts)+1,"samples":len(observations),
          "moving":bool(plan and max(x for _,x in plan)-min(x for _,x in plan)>.01)}
    print(f"    [smart-crop] {'tracked' if plan else 'wide fallback'}: {info}")
    return plan,info


def _flow_face_box(previous_gray, gray, box):
    """Follow an occluded head briefly using consistent local pixel motion."""
    if previous_gray is None:
        return None
    import cv2
    import numpy as np
    x,y,w,h=box;fh,fw=gray.shape
    mask=np.zeros_like(gray)
    mask[max(0,int(y-h*.3)):min(fh,int(y+h*1.5)),
         max(0,int(x-w*.2)):min(fw,int(x+w*1.2))]=255
    points=cv2.goodFeaturesToTrack(previous_gray,60,.015,5,mask=mask)
    if points is None or len(points)<6:
        return None
    moved,status,error=cv2.calcOpticalFlowPyrLK(previous_gray,gray,points,None,winSize=(21,21),maxLevel=3)
    if moved is None:
        return None
    returned,back_status,_=cv2.calcOpticalFlowPyrLK(gray,previous_gray,moved,None,winSize=(21,21),maxLevel=3)
    if returned is None:
        return None
    valid=(status.ravel()==1)&(back_status.ravel()==1)&(error.ravel()<30)&(np.linalg.norm(returned-points,axis=2).ravel()<1.5)
    if valid.sum()<6:
        return None
    delta=(moved-points).reshape(-1,2)[valid]
    dx,dy=np.median(delta,axis=0)
    if np.median(np.linalg.norm(delta-[dx,dy],axis=1))>10 or abs(dx)>fw*.18 or abs(dy)>fh*.18:
        return None
    return np.array([max(0,min(fw-w,x+dx)),max(0,min(fh-h,y+dy)),w,h],dtype=np.float32)


def _select_foreground_face(faces, previous_center, frame_width, previous_area=None):
    """Track one prominent subject; never average them with the crowd."""
    if previous_area is not None:
        faces=[f for f in faces if previous_area*.45 <= f[2]*f[3] <= previous_area*2.5
               and abs((f[0]+f[2]/2)/frame_width-previous_center)<.15]
    if not faces:
        return None
    largest = max(faces, key=lambda f: f[2]*f[3])
    eligible = [f for f in faces if f[2]*f[3] >= largest[2]*largest[3]*.65]
    return largest if previous_center is None else min(
        eligible, key=lambda f: abs((f[0]+f[2]/2)/frame_width-previous_center))


# Visualizer styles, cycled by clip rank so each clip in a batch gets a different look.
# Each style returns an ffmpeg filter chain that produces a [wave] labeled output,
# 1080 wide x configurable height, ready to overlay on the video background.
VISUALIZER_STYLES = ["clean_waves", "freq_bars", "spectrum", "dual_waves", "cqt_rainbow"]

def _visualizer_filter(style: str, w: int, yellow: str) -> tuple[str, int]:
    """Returns (filter_chain, height_used). Filter outputs to [wave] label."""
    if style == "freq_bars":
        # Vertical frequency bars — log scale so bass shows clearly.
        # Taller + more transparent per DJ feedback: sit over video without
        # blocking what's behind. showfreqs ignores user colors in some builds;
        # force-tint to brand yellow via channel mixer (zero blue, dim green).
        ht = 320
        f = (
            f"[amain]showfreqs=s={w}x{ht}:mode=bar:fscale=log:ascale=log:"
            f"win_size=2048:cmode=combined,"
            f"format=yuva420p,colorchannelmixer=rr=1:rg=1:rb=1:gr=0.9:gg=0.9:gb=0.9:"
            f"br=0:bg=0:bb=0:aa=0.65[wave]"
        )
        return f, ht
    if style == "spectrum":
        # Scrolling spectrogram strip, fire-style ramp (we recolor to brand yellow)
        ht = 180
        f = (
            f"[amain]showspectrum=s={w}x{ht}:mode=combined:slide=scroll:scale=cbrt:"
            f"color=intensity:fps={FPS},format=yuva420p,"
            f"colorchannelmixer=rr=1:gg=0.85:bb=0:aa=0.85[wave]"
        )
        return f, ht
    if style == "dual_waves":
        # Two-tone wave: yellow + white outline — more visual depth
        ht = 240
        f = (
            f"[amain]showwaves=s={w}x{ht}:mode=line:colors={yellow}|0xFFFFFF:rate={FPS}:"
            f"draw=full:split_channels=1:scale=sqrt,format=yuva420p,colorchannelmixer=aa=0.85[wave]"
        )
        return f, ht
    if style == "cqt_rainbow":
        # Constant-Q transform — looks like a frequency rainbow that pulses
        ht = 260
        f = (
            f"[amain]showcqt=s={w}x{ht}:fps={FPS}:basefreq=40:endfreq=4000:count=16:"
            f"text=0:bar_g=2:tlength=0.25,format=yuva420p,"
            f"colorchannelmixer=rr=1:gg=0.85:bb=0:aa=0.82[wave]"
        )
        return f, ht
    # default: clean_waves — thin single line, modern minimal look
    ht = 200
    f = (
        f"[amain]showwaves=s={w}x{ht}:mode=line:colors={yellow}:rate={FPS}:"
        f"draw=full:scale=sqrt,format=yuva420p,colorchannelmixer=aa=0.92[wave]"
    )
    return f, ht


def ff_escape_text(s: str) -> str:
    """Escape text for ffmpeg drawtext filter.
    Apostrophes go to the curly typographic variant — visually identical, but
    doesn't break ffmpeg's single-quoted argument syntax."""
    return (
        s.replace("\\", "\\\\")
         .replace("'", "’")  # curly apostrophe — safe inside ffmpeg quotes
         .replace(":", "\\:")
         .replace(",", "\\,")
         .replace("%", "\\%")
    )


def wrap_caption(text: str, size: int, max_width: int = 850) -> list[str]:
    """Wrap by rendered width, including long handles and wide capitals."""
    from PIL import ImageFont
    font = ImageFont.truetype(FONT_BOLD, size)
    lines, line = [], ""
    for word in text.split():
        if line and font.getlength(line + " " + word) <= max_width:
            line += " " + word
            continue
        if line:
            lines.append(line)
        line = ""
        for character in word:
            if line and font.getlength(line + character) > max_width:
                lines.append(line)
                line = ""
            line += character
    if line:
        lines.append(line)
    return lines


def render_clip(source: str, clip: dict, out_path: str, watermark: str = "",
                visualizer: str = "freq_bars", hide_logo: bool = False,
                show_caption: bool = True, framing: str = "fit",
                logo_path: str | None = None) -> bool:
    """Render a phone-compatible clip. Text files keep DJ text out of filter syntax."""
    with tempfile.TemporaryDirectory(prefix="reelcrate-text-") as text_dir:
        return _render_clip(source, clip, out_path, watermark, visualizer,
                            hide_logo, show_caption, framing, Path(text_dir), logo_path)


def _render_clip(source, clip, out_path, watermark, visualizer,
                 hide_logo, show_caption, framing, text_dir, logo_path):
    start = clip["start_sec"]
    end = clip["end_sec"]
    duration = end - start
    hook = clip.get("hook", "")
    tag = clip.get("tag", "")
    bpm = clip.get("bpm", 0)
    bpm_conf = clip.get("bpm_confidence", clip.get("local_bpm_confidence", 0))
    custom_title = clip.get("custom_title")  # optional DJ-supplied title override
    rank = int(clip.get("rank", 1))
    peak_offset = max(0.0, clip["peak_sec"] - start)  # seconds into the clip

    # Visualizer style. Default freq_bars (per DJ preference); can be overridden
    # via the visualizer arg (driven from UI selector) or per-clip in the manifest.
    style = clip.get("visualizer") or visualizer or "freq_bars"
    if style not in VISUALIZER_STYLES:
        style = "freq_bars"
    wave_filter, wave_h = _visualizer_filter(style, W, YELLOW)

    # Leave room for the platform's bottom caption and right-side buttons.
    # Break long words too; captions never silently lose their final words.
    display_text = str(custom_title if custom_title is not None else hook)
    display_text = " ".join(display_text.split())[:120]
    hook_lines = wrap_caption(display_text.upper(), 52) if show_caption else []
    caption_fs = 52 if len(hook_lines) <= 3 else 42
    if caption_fs == 42:
        hook_lines = wrap_caption(display_text.upper(), caption_fs)
    line_h = caption_fs + 16
    caption_top_y = H - 370 - line_h * len(hook_lines)
    text_filters = []

    def draw_text(text, size, x, y, extra=""):
        path = text_dir / f"text_{len(text_filters)}.txt"
        path.write_text(text, encoding="utf-8")
        text_filters.append(
            f"drawtext=fontfile='{FONT_BOLD}':textfile='{path}':expansion=none:"
            f"fontcolor=white:fontsize={size}:x={x}:y={y}:"
            f"shadowcolor=black@0.85:shadowx=2:shadowy=2{extra}"
        )

    if not hide_logo:
        draw_text("REEL/CRATE", 28, 70, 110)
    if watermark.strip():
        draw_text(watermark.strip()[:40], 28, 70, H - 290)
    for i, line in enumerate(hook_lines):
        draw_text(line, caption_fs, "(w-tw)/2-35", caption_top_y + i*line_h,
                  ":box=1:boxcolor=black@0.45:boxborderw=12")

    # ---------- Construct video base ----------
    # Probe for video stream presence AND get the source dimensions so we can
    # ask OpenCV to find faces at the right coordinates.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=codec_type,width,height",
         "-of", "default=nw=1", source],
        capture_output=True, text=True
    )
    has_video = "video" in probe.stdout
    src_w = src_h = 0
    for line in (probe.stdout or "").splitlines():
        if line.startswith("width="):  src_w = int(line.split("=", 1)[1] or 0)
        if line.startswith("height="): src_h = int(line.split("=", 1)[1] or 0)

    # Follow movement and restart the track when the shot changes.
    crop_plan = None
    face_x_frac = 0.5  # 0..1 horizontal center of interest in source frame
    if clip.get("selection_mode") == "crowd":
        framing = "fit"  # Never follow one face out of a crowd, including on re-render.

    if has_video and framing == "crop" and src_w > 0 and src_h > 0:
        try:
            crop_plan, tracking = _face_crop_plan(source, start, duration, src_w, src_h)
            clip["tracking"] = tracking
            face_x_frac = crop_plan[0][1] if crop_plan else None
        except Exception as e:
            print(f"    [smart-crop] face detection failed, keeping full frame: {e}")
            face_x_frac = None
        if face_x_frac is None:
            framing = "fit"
            face_x_frac = .5
    clip["framing_applied"] = framing

    # Always split the audio stream: one copy for the foreground waveform,
    # one copy for the fullscreen backdrop (or drained, if the source has video).
    audio_split = "[0:a]asplit=2[abg][amain];"

    if has_video:
        audio_split = "[0:a]apad=pad_dur=1[amain];"

    if has_video and clip.get("selection_mode") == "crowd":
        from crowd import region_filter
        crowd_crop = region_filter(clip.get("crowd_region"))
        region = clip.get("crowd_region") or [0,0,1,1]
        if show_caption and src_h and src_w * region[2] / (src_h * region[3]) > 1.2:
            draw_text("FROM THE", 42, 70, 340)
            draw_text("DANCE FLOOR", 96, 70, 410)
        video_chain = (f"{audio_split}[0:v]{crowd_crop}scale={W}:{H}:force_original_aspect_ratio=decrease:"
                       f"force_divisible_by=2,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[bg]")
    elif has_video and framing == "fit":
        video_chain = (
            f"{audio_split}"
            f"[0:v]split=2[blur_src][fit_src];"
            f"[blur_src]scale=270:480:force_original_aspect_ratio=increase,"
            f"crop=270:480,gblur=sigma=18,scale={W}:{H},"
            f"eq=brightness=-0.15[blur];"
            f"[fit_src]scale={W}:1200:force_original_aspect_ratio=decrease:"
            f"force_divisible_by=2,setsar=1[fit];"
            f"[blur][fit]overlay=(W-w)/2:(H-h)/2-170,setsar=1[bg]"
        )
    elif has_video:
        # Scale to fill the phone frame and follow the time-varying crop.
        motion_filter = ""
        if src_w > 0 and src_h > 0 and src_w * H > src_h * W:
            # Landscape: scale=-2:H (height fixed at H, width scales up).
            scaled_w = int(round(src_w * H / src_h))
            if scaled_w % 2:                # -2 rounds to even
                scaled_w -= 1
            target_x  = int(scaled_w * face_x_frac)
            crop_x_px = max(0,min(scaled_w-W,target_x-W//2))
            if crop_plan:
                commands_path = text_dir / "crop-motion.txt"
                crop_x_px = _crop_commands(crop_plan,scaled_w,duration,commands_path)
                motion_filter = f"{TRACK_CLOCK},sendcmd=f='{commands_path}',"
        else:
            # Portrait / square: scale to W wide → no horizontal shift possible.
            crop_x_px = 0
        video_chain = (
            f"{audio_split}"
            f"[0:v]scale=w='if(gt(a,{W}/{H}),-2,{W})':h='if(gt(a,{W}/{H}),{H},-2)',"
            f"{motion_filter}crop@follow={W}:{H}:x={crop_x_px}:y=(ih-{H})/2,"
            f"setsar=1,"
            f"eq=brightness=-0.05:saturation=1.1[bg]"
            # [abg] is unused in the has_video branch — anullsink drains it.

        )
    else:
        # No source video — fill the frame with a full-screen CQT visualizer
        # painted over a warm dark base. showcqt draws colored bars across the
        # WHOLE width every frame (unlike showspectrum which scrolls in over
        # time and leaves 90 % of a short clip black). Guaranteed content in
        # every part of the frame from t=0.
        video_chain = (
            f"{audio_split}"
            f"color=c=0x1a1005:s={W}x{H}:r={FPS}:d={duration},format=yuv420p[bgbase];"
            f"[abg]showcqt=s={W}x{H}:fps={FPS}:basefreq=30:endfreq=8000:count=8:"
            f"bar_g=2:tlength=0.25:text=0,format=yuva420p,"
            f"colorchannelmixer=rr=1:gg=0.72:bb=0.18:aa=0.80[bgcqt];"
            f"[bgbase][bgcqt]overlay=0:0,format=yuv420p[bg]"
        )

    # Compose: bg → overlay waveform near bottom → drawtext stack
    text_chain = ",".join(text_filters) or "null"

    # Waveform strip sits just ABOVE the caption block, matching the
    # composition in the app preview (bars low, caption underneath).
    wave_gap  = 40
    wave_y    = caption_top_y - wave_h - wave_gap

    logo_filter = ""
    base_label = "[vbg]"
    if logo_path:
        logo_filter = (f"[1:v]scale=280:180:force_original_aspect_ratio=decrease,format=rgba[logo];"
                       f"[vbg][logo]overlay=x=70:y=190:eof_action=repeat[vlogo];")
        base_label = "[vlogo]"
    full_filter = (
        f"{video_chain};"
        f"{wave_filter};"
        f"[bg][wave]overlay=0:{wave_y}:shortest=1[vbg];"
        f"{logo_filter}{base_label}{text_chain}[vout]"
    )
    inputs = ["-ss", str(start), "-t", str(duration), "-i", source]
    if logo_path:
        inputs += ["-i", logo_path]
    map_args = ["-map", "[vout]", "-map", "0:a"]

    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-filter_complex_threads", "2",
        *inputs,
        "-filter_complex", full_filter,
        *map_args,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-r", str(FPS),
        "-movflags", "+faststart",
        "-t", str(duration),
        out_path,
    ]

    print(f"    rendering -> {Path(out_path).name}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=max(180, duration * 20))
    except subprocess.TimeoutExpired:
        print("    [ffmpeg error] rendering timed out")
        return False
    if res.returncode != 0:
        print(f"    [ffmpeg error]\n{res.stderr[:1500]}\n{res.stderr[-1500:] if len(res.stderr)>1500 else ''}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="manifest.json from analyze.py")
    ap.add_argument("--source", help="Override source file (default: from manifest)")
    ap.add_argument("--watermark", default="")
    ap.add_argument("--visualizer", default="freq_bars",
                    choices=VISUALIZER_STYLES,
                    help=f"Visualizer style for all clips. Options: {', '.join(VISUALIZER_STYLES)}")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    source = args.source or manifest["source"]
    if not os.path.exists(source):
        print(f"error: source not found: {source}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.manifest).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] {len(manifest['clips'])} clips from {source}")
    successes = 0
    for c in manifest["clips"]:
        out_path = out_dir / f"clip_{c['rank']:02d}.mp4"
        if render_clip(source, c, str(out_path), args.watermark, args.visualizer):
            successes += 1

    print(f"\n[render] done: {successes}/{len(manifest['clips'])} clips rendered to {out_dir}")


if __name__ == "__main__":
    main()
