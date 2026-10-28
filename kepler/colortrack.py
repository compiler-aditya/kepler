"""Offline colour tracker.

Used for tests and as a fallback when there's no Meta API key yet. For real
clips use SAM 3.1 (`--tracker sam`); this one only works for a strongly
coloured ball on a plain background.
"""

from __future__ import annotations

import cv2
import numpy as np

from .trajectory import Sample
from .video import iter_frames

NAMED = {
    "red": [((0, 110, 60), (10, 255, 255)), ((170, 110, 60), (180, 255, 255))],
    "orange": [((8, 120, 80), (22, 255, 255))],
    "yellow": [((22, 100, 100), (35, 255, 255))],
    "green": [((40, 80, 50), (85, 255, 255))],
    "blue": [((95, 100, 50), (130, 255, 255))],
}


def _ranges(color: str):
    color = color.lower()
    if color in NAMED:
        return NAMED[color]
    if color.startswith("#") and len(color) == 7:
        bgr = np.uint8([[[int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16)]]])
        h = int(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 0])
        lo, hi = h - 10, h + 10
        if lo < 0:
            return [((0, 80, 50), (hi, 255, 255)), ((180 + lo, 80, 50), (180, 255, 255))]
        if hi > 180:
            return [((lo, 80, 50), (180, 255, 255)), ((0, 80, 50), (hi - 180, 255, 255))]
        return [((lo, 80, 50), (hi, 255, 255))]
    raise SystemExit(f"unknown colour {color!r}; use one of {sorted(NAMED)} or #rrggbb")


def track_color(video, color: str = "red", min_area: int = 12) -> list[Sample]:
    ranges = _ranges(color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out: list[Sample] = []
    for i, frame in iter_frames(video):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        m = cv2.moments((labels == k).astype(np.uint8), binaryImage=True)
        out.append(Sample(frame=i, x_px=m["m10"] / m["m00"], y_px=m["m01"] / m["m00"],
                          size_px=float(np.sqrt(area / np.pi) * 2), confidence=1.0))
    return out
