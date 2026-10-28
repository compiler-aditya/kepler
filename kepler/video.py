"""Video helpers: probe frame rate, convert to MP4, read frames.

Slow-motion clips are the main trap. iPhone slow-motion MOVs store every
captured frame but carry an edit list that stretches playback, and some
Android phones export slow motion re-timed to 30 fps. Kepler always works in
capture time: frame index / capture fps.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int  # as displayed, after any rotation flag
    height: int
    codec: str
    stream_fps: float  # what the file says
    n_frames: int
    duration_s: float
    rotation: int
    has_audio: bool
    has_edit_list_hint: bool

    def summary(self) -> str:
        lines = [
            f"{self.path}",
            f"  {self.width}x{self.height} {self.codec}, {self.n_frames} frames, "
            f"{self.stream_fps:.3f} fps in the file, {self.duration_s:.2f} s",
        ]
        if self.rotation:
            lines.append(f"  rotation flag: {self.rotation} degrees")
        if self.stream_fps < 100:
            lines.append(
                "  NOTE: under 100 fps. If you filmed slow motion, the phone may have "
                "re-timed it. Export the original, or pass --capture-fps 240."
            )
        if self.has_edit_list_hint:
            lines.append(
                "  NOTE: this looks like a phone slow-motion file with playback edits. "
                "Run `kepler prep` first; it keeps every captured frame at capture speed."
            )
        return "\n".join(lines)


def _need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise SystemExit(f"{tool} not found. Install ffmpeg (macOS: brew install ffmpeg).")
    return path


def probe(path: str | Path) -> VideoInfo:
    path = str(path)
    out = subprocess.run(
        [_need("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-count_packets",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    meta = json.loads(out)
    vstreams = [s for s in meta["streams"] if s.get("codec_type") == "video"]
    if not vstreams:
        raise SystemExit(f"{path}: no video stream")
    v = vstreams[0]
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(v.get("tags", {}).get("rotate", rot) or rot)
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    rate = v.get("r_frame_rate") or v.get("avg_frame_rate") or "0/1"
    fps = float(Fraction(rate)) if rate != "0/0" else 0.0
    n = int(v.get("nb_read_packets") or v.get("nb_frames") or 0)
    dur = float(meta.get("format", {}).get("duration") or v.get("duration") or 0)
    tags = {k.lower(): str(val).lower() for k, val in (meta.get("format", {}).get("tags") or {}).items()}
    slowmo_hint = any("slow" in k or "slow" in val for k, val in tags.items())
    avg = v.get("avg_frame_rate") or rate
    avg_fps = float(Fraction(avg)) if avg not in ("0/0", "") else fps
    edit_hint = slowmo_hint or (fps > 0 and avg_fps > 0 and abs(avg_fps - fps) / fps > 0.2)
    return VideoInfo(
        path=path, width=w, height=h, codec=v.get("codec_name", "?"), stream_fps=fps,
        n_frames=n, duration_s=dur, rotation=rot,
        has_audio=any(s.get("codec_type") == "audio" for s in meta["streams"]),
        has_edit_list_hint=edit_hint,
    )


def prep(src: str | Path, dst: str | Path) -> Path:
    """Convert to H.264 MP4 with faststart, keeping every captured frame.

    -ignore_editlist drops phone slow-motion playback edits so frames come out
    evenly spaced; fps_mode passthrough keeps every frame, no duplicates or drops.
    SAM 3.1 accepts MP4/MOV, but Muse video understanding accepts MP4 only.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _need("ffmpeg"), "-y", "-ignore_editlist", "1", "-i", str(src),
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-fps_mode", "passthrough", "-c:a", "aac", "-movflags", "+faststart", str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


def read_frame(path: str | Path, index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {index} of {path}")
    return frame


def iter_frames(path: str | Path):
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()
