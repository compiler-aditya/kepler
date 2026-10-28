"""Debug overlay: the tracked path drawn over the clip, to catch bad frames by eye."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .trajectory import Sample
from .video import iter_frames


def render(video, samples: list[Sample], out: Path, *, corners_px=None, fall_frames=None,
           label: str = "", fps_out: float = 30.0) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    by = {s.frame: s for s in samples}
    trail: list[tuple[int, int]] = []
    writer = None
    ff = shutil.which("ffmpeg")
    for i, frame in iter_frames(video):
        if writer is None:
            h, w = frame.shape[:2]
            if ff:
                writer = subprocess.Popen(
                    [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", f"{w}x{h}", "-r", str(fps_out), "-i", "-", "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(out)],
                    stdin=subprocess.PIPE)
            else:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))
        if corners_px is not None:
            cv2.polylines(frame, [np.asarray(corners_px, np.int32)], True, (255, 200, 0), 2)
        s = by.get(i)
        if s is not None:
            trail.append((int(round(s.x_px)), int(round(s.y_px))))
        for k in range(1, len(trail)):
            cv2.line(frame, trail[k - 1], trail[k], (0, 220, 255), 2)
        if s is not None:
            in_fall = fall_frames is not None and fall_frames[0] <= i <= fall_frames[1]
            color = (0, 255, 0) if in_fall else (0, 0, 255)
            cv2.circle(frame, trail[-1], max(4, int(s.size_px / 2)), color, 2)
        cv2.putText(frame, f"frame {i}  {label}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        if ff:
            writer.stdin.write(frame.tobytes())
        else:
            writer.write(frame)
    if writer is not None:
        if ff:
            writer.stdin.close()
            writer.wait()
        else:
            writer.release()
    return out


def draw_plan(frame: np.ndarray, H_px_to_mm: np.ndarray, geom, cup: np.ndarray, rim_diameter: float,
              marks: list[dict], ramp_top_s: float, scale: float = 1.0) -> np.ndarray:
    """The plan over a frame: ramp, cup rim, and per mark its release tick and predicted path.

    geom: phases.RampGeometry; cup: rim centre; marks: dicts with "label", "s" (metres up the
    ramp), "path" (N x 2, release to rim height) and "color" (BGR). Positions are in the ball's
    plane; `scale` is the gravity correction that took sheet metres there.
    """
    from .sheet import mm_to_px

    img = frame.copy()

    def px(P):
        P = np.atleast_2d(np.asarray(P, float)) / scale
        x, y = mm_to_px(H_px_to_mm, P[:, 0] * 1000, P[:, 1] * 1000)
        return np.stack([x, y], 1)

    top = geom.launch - ramp_top_s * geom.d
    cv2.line(img, tuple(px(top)[0].round().astype(int)), tuple(px(geom.launch)[0].round().astype(int)),
             (200, 200, 200), 1, cv2.LINE_AA)
    rim = px([cup - 0.5 * rim_diameter * geom.across, cup + 0.5 * rim_diameter * geom.across]).round().astype(int)
    cv2.line(img, tuple(rim[0]), tuple(rim[1]), (255, 200, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "cup", tuple(rim[1] + [8, 5]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
    nrm = np.array([-geom.d[1], geom.d[0]])
    for k, m in enumerate(marks):
        pts = px(m["path"]).round().astype(np.int32)
        cv2.polylines(img, [pts], False, m["color"], 2, cv2.LINE_AA)
        p = geom.launch - m["s"] * geom.d
        tick = px([p - 0.015 * nrm, p + 0.015 * nrm]).round().astype(int)
        cv2.line(img, tuple(tick[0]), tuple(tick[1]), m["color"], 3, cv2.LINE_AA)
        y = img.shape[0] - 24 - 30 * (len(marks) - 1 - k)  # bottom-left, clear of the ramp's top
        cv2.putText(img, m["label"], (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, m["color"], 2, cv2.LINE_AA)
    return img
