"""Per-frame positions, in pixels and in metres, and the trajectory.json file.

trajectory.json is the hand-off to the theory agent (days 3-4): it reads
only this file, never the video.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    frame: int
    x_px: float
    y_px: float
    size_px: float  # box/disc diameter in pixels
    confidence: float | None


def samples_from_sam(records_by_frame: dict) -> list[Sample]:
    out = []
    for f in sorted(records_by_frame):
        r = records_by_frame[f]
        out.append(Sample(frame=f, x_px=r.box.cx, y_px=r.box.cy,
                          size_px=float((r.box.width + r.box.height) / 2), confidence=r.confidence))
    return out


def write_csv(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(samples[0]).keys()) if samples else ["frame"])
        w.writeheader()
        for s in samples:
            w.writerow(asdict(s))


@dataclass
class Trajectory:
    clip: str
    capture_fps: float
    frame_size: tuple[int, int]
    tracker: str
    target: str
    sheet: dict
    frames: np.ndarray  # int
    t: np.ndarray  # seconds, capture time
    x: np.ndarray  # metres, along the sheet's tl->tr edge
    y: np.ndarray  # metres, along the sheet's tl->bl edge
    confidence: np.ndarray
    gravity: dict | None = None
    size: np.ndarray | None = None  # metres: ball diameter seen on each frame (box or disc)

    def to_json(self) -> dict:
        size = self.size if self.size is not None else np.full(len(self.t), np.nan)
        return {
            "version": 1,
            "clip": self.clip,
            "capture_fps": self.capture_fps,
            "frame_size": list(self.frame_size),
            "tracker": self.tracker,
            "target": self.target,
            "units": "m",
            "axes": "x along the A4 sheet's top edge, y along its left edge (as seen in frame 0)",
            "sheet": self.sheet,
            "gravity_check": self.gravity,
            "samples": [
                {"frame": int(f), "t": round(float(t), 6), "x": round(float(x), 6),
                 "y": round(float(y), 6),
                 "confidence": None if np.isnan(c) else round(float(c), 4),
                 "size": None if np.isnan(sz) else round(float(sz), 6)}
                for f, t, x, y, c, sz in zip(self.frames, self.t, self.x, self.y, self.confidence, size)
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "Trajectory":
        d = json.loads(Path(path).read_text())
        s = d["samples"]
        return cls(
            clip=d["clip"], capture_fps=d["capture_fps"], frame_size=tuple(d["frame_size"]),
            tracker=d["tracker"], target=d["target"], sheet=d["sheet"],
            frames=np.array([e["frame"] for e in s]), t=np.array([e["t"] for e in s]),
            x=np.array([e["x"] for e in s]), y=np.array([e["y"] for e in s]),
            confidence=np.array([np.nan if e["confidence"] is None else e["confidence"] for e in s]),
            gravity=d.get("gravity_check"),
            size=np.array([np.nan if e.get("size") is None else e["size"] for e in s], float),
        )
