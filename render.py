#!/usr/bin/env python3
"""
Reelcrate clip renderer.

Reads a manifest.json produced by analyze.py and renders each clip into a
ready-to-post 9:16 1080x1920 MP4 with:
  - source audio (volume-normalized)
  - animated waveform strip (yellow)
  - bold caption hook (centered, bottom third)
  - REEL/CRATE watermark (top-left)
  - @realdjez1 attribution (bottom-right)
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
from pathlib import Path


FONT_BOLD = "/usr/share/fonts/truetype/lato/Lato-Black.ttf"
FONT_TXT = "/usr/share/fonts/truetype/lato/Lato-Heavy.ttf"
YELLOW = "0xFFE600"
WHITE = "0xFFFFFF"

W, H = 1080, 1920  # 9:16 vertical
FPS = 30

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
            f"draw=full:n=80,format=yuva420p,colorchannelmixer=aa=0.85[wave]"
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
        f"draw=full:n=120,format=yuva420p,colorchannelmixer=aa=0.92[wave]"
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


def render_clip(source: str, clip: dict, out_path: str, watermark: str,
                visualizer: str = "freq_bars") -> bool:
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

    # ---------- text overlays ----------
    # Use the custom_title if provided (lets the DJ override the auto-generated hook
    # per clip — wired up from the upload UI later). Otherwise use the hook pool.
    display_text = custom_title if custom_title else hook
    safe_hook = ff_escape_text(display_text.upper())
    safe_tag = ff_escape_text(tag)
    safe_wm = ff_escape_text(watermark)

    # BPM display only if detector is confident enough — bad room-mic BPMs
    # were misleading, so we suppress them rather than show a lie.
    show_bpm = bool(bpm) and bpm_conf >= 0.55
    bpm_text = ff_escape_text(f"{int(bpm)} BPM") if show_bpm else ""

    text_filters = []

    # Logo wordmark (top-left) — smaller, less dominating
    text_filters.append(
        f"drawtext=fontfile={FONT_BOLD}:text='REEL/CRATE':fontcolor={WHITE}:"
        f"fontsize=30:x=40:y=44:shadowcolor=black@0.6:shadowx=2:shadowy=2:"
        f"alpha=0.85"
    )

    # Tag chip (top-right) — slightly smaller too, balanced with the logo
    text_filters.append(
        f"drawtext=fontfile={FONT_BOLD}:text='{safe_tag}':fontcolor=black:"
        f"fontsize=32:x=w-tw-40:y=46:box=1:boxcolor={YELLOW}:boxborderw=12"
    )

    # BPM (bottom-right) — small, above the caption area so it doesn't collide
    if bpm_text:
        text_filters.append(
            f"drawtext=fontfile={FONT_BOLD}:text='{bpm_text}':fontcolor={YELLOW}:"
            f"fontsize=26:x=w-tw-40:y=44:shadowcolor=black@0.6:shadowx=2:shadowy=2"
        )

    # Watermark (bottom-left) — very bottom, safely below the caption block
    text_filters.append(
        f"drawtext=fontfile={FONT_BOLD}:text='{safe_wm}':fontcolor={WHITE}:"
        f"fontsize=26:x=40:y=h-56:shadowcolor=black@0.6:shadowx=2:shadowy=2:alpha=0.75"
    )

    # Caption sits at the VERY BOTTOM — matches the app's clip-preview layout.
    # Wrap long hooks into ~18-char lines.
    def wrap(text, width=18):
        words = text.split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= width:
                cur = (cur + " " + w).strip()
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    hook_lines = wrap(display_text.upper(), width=18)[:3]
    # Slightly smaller than before so 3-line captions still fit under the bars.
    line_h    = 76
    caption_fs = 62
    block_h    = line_h * len(hook_lines)

    # Caption block anchored ~120 px from bottom of frame.
    caption_bottom_margin = 120
    caption_top_y = H - caption_bottom_margin - block_h

    for i, ln in enumerate(hook_lines):
        safe_ln = ff_escape_text(ln)
        text_filters.append(
            f"drawtext=fontfile={FONT_BOLD}:text='{safe_ln}':fontcolor=white:"
            f"fontsize={caption_fs}:x=(w-tw)/2:y={caption_top_y + i*line_h}:"
            f"shadowcolor=black@0.85:shadowx=3:shadowy=3:"
            f"box=1:boxcolor=black@0.35:boxborderw=14"
        )

    # ---------- Construct video base ----------
    # Probe for video stream presence
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=codec_type", "-of", "default=nw=1:nk=1", source],
        capture_output=True, text=True
    )
    has_video = "video" in probe.stdout

    # Always split the audio stream: one copy for the foreground waveform,
    # one copy for the fullscreen backdrop (or drained, if the source has video).
    audio_split = "[0:a]asplit=2[abg][amain];"

    if has_video:
        # Scale-and-crop the source video to 9:16. Use scale=-2:H to size by height
        # (works for landscape sources where we'd otherwise underfill), force_original
        # ensures we always have enough pixels in both directions, then crop center.
        video_chain = (
            f"{audio_split}"
            f"[0:v]scale=w='if(gt(a,{W}/{H}),-2,{W})':h='if(gt(a,{W}/{H}),{H},-2)',"
            f"crop={W}:{H}:(iw-{W})/2:(ih-{H})/2,"
            f"setsar=1,"
            f"eq=brightness=-0.05:saturation=1.1[bg];"
            # [abg] is unused in the has_video branch — anullsink drains it.
            f"[abg]anullsink"
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
    text_chain = ",".join(text_filters)

    # Waveform strip sits just ABOVE the caption block, matching the
    # composition in the app preview (bars low, caption underneath).
    wave_gap  = 40
    wave_y    = caption_top_y - wave_h - wave_gap

    full_filter = (
        f"{video_chain};"
        f"{wave_filter};"
        f"[bg][wave]overlay=0:{wave_y}[vbg];"
        f"[vbg]{text_chain}[vout]"
    )
    inputs = ["-ss", str(start), "-t", str(duration), "-i", source]
    map_args = ["-map", "[vout]", "-map", "0:a"]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
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
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"    [ffmpeg error]\n{res.stderr[-1500:]}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="manifest.json from analyze.py")
    ap.add_argument("--source", help="Override source file (default: from manifest)")
    ap.add_argument("--watermark", default="@realdjez1")
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
