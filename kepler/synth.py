"""Render synthetic test clips with known physics.

Two scenes, both seen by a keystoned camera with an A4 sheet taped a few
degrees off level in the motion plane:

- "fall": a red ball rolls along a table, leaves the edge, falls under
  g = 9.81 m/s^2 and bounces once. The pipeline should recover g within 2%.
- "ramp": a red ball is released from rest on a straight ramp, rolls down it in
  a groove (contact radius smaller than its own, plus rolling resistance),
  leaves the lower end and flies to the floor. The ramp-to-cup demo, with a
  known answer.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

G = 9.81
R_BALL = 0.011

VIEW_WORLD = np.float32([[-0.36, -0.12], [0.62, -0.12], [0.62, 0.62], [-0.36, 0.62]])
VIEW_IMG = np.float32([[30, 40], [1255, 18], [1268, 708], [12, 690]])  # keystone


def ball_path(t: np.ndarray, v0=1.15, x_edge=0.0, floor=0.55, e=0.55, r=R_BALL):
    """World plane coords in metres, y pointing down. Ball centre starts at y=0 on the table."""
    x = np.empty_like(t)
    y = np.empty_like(t)
    t_edge = 0.25  # rolls 0.29 m on the table first
    x0 = x_edge - v0 * t_edge
    t_land = t_edge + np.sqrt(2 * (floor - r) / G)
    vy_land = G * (t_land - t_edge)
    for i, ti in enumerate(t):
        x[i] = x0 + v0 * ti
        if ti < t_edge:
            y[i] = 0.0
        elif ti < t_land:
            dt = ti - t_edge
            y[i] = 0.5 * G * dt * dt
        else:
            dt = ti - t_land
            y[i] = (floor - r) - e * vy_land * dt + 0.5 * G * dt * dt
            y[i] = min(y[i], floor - r)
    return x, y


def ramp_accel(theta_deg: float, r_eff_ratio: float = 0.75, c_rr: float = 0.01) -> float:
    """Groove rolling: a = g (sin θ - c_rr cos θ) / (1 + (2/5) R^2 / r_eff^2)."""
    th = np.radians(theta_deg)
    return float(G * (np.sin(th) - c_rr * np.cos(th)) / (1 + 0.4 / r_eff_ratio ** 2))


def ramp_path(t: np.ndarray, theta_deg=18.0, s_release=0.25, t_release=0.15, r_eff_ratio=0.75,
              c_rr=0.01, floor=0.55, r=R_BALL):
    """Ball centre in world plane coords (metres, y down). The ramp's lower end, where the
    ball centre leaves it, is at (0, 0); the ball starts s_release up the ramp, at rest."""
    th = np.radians(theta_deg)
    d = np.array([np.cos(th), np.sin(th)])  # down the ramp
    a = ramp_accel(theta_deg, r_eff_ratio, c_rr)
    tau_end = np.sqrt(2 * s_release / a)
    v_end = a * tau_end
    vy = v_end * np.sin(th)
    t_flight = (-vy + np.sqrt(vy ** 2 + 2 * G * (floor - r))) / G
    tau = np.asarray(t, float) - t_release
    q = 0.5 * a * np.clip(tau, 0, tau_end) ** 2  # distance travelled down the ramp
    on_ramp = (-s_release + q)[:, None] * d[None, :]
    tf = np.clip(tau - tau_end, 0, t_flight)[:, None]
    flying = v_end * d[None, :] * tf + 0.5 * G * np.array([0.0, 1.0])[None, :] * tf ** 2
    pos = np.where((tau <= tau_end)[:, None], on_ramp, flying)
    truth = {"theta_deg": theta_deg, "s_release": s_release, "t_release": t_release,
             "r_eff_ratio": r_eff_ratio, "c_rr": c_rr, "accel": a, "v_launch": float(v_end),
             "t_launch": float(t_release + tau_end), "x_land": float(v_end * np.cos(th) * t_flight),
             "y_land": float(floor - r), "ball_radius": r}
    return pos[:, 0], pos[:, 1], truth


def ramp_trajectory(fps: float = 240.0, noise_m: float = 0.0003, sheet_angle_deg: float = 3.0,
                    seed: int = 0, **kw):
    """The ramp scene as trajectory.json would hold it, without rendering video: true positions
    seen in an A4 frame rotated by sheet_angle_deg, plus Gaussian position noise. Returns
    (Trajectory, truth) with the gravity check already run, as `kepler track` would."""
    from . import gravity
    from .trajectory import Trajectory

    rng = np.random.default_rng(seed)
    _, _, phys = ramp_path(np.zeros(1), **kw)
    seconds = phys["t_launch"] + phys["x_land"] / (phys["v_launch"] * np.cos(np.radians(phys["theta_deg"])))
    t = np.arange(int(seconds * fps)) / fps
    x, y, _ = ramp_path(t, **kw)
    c, s = np.cos(np.radians(sheet_angle_deg)), np.sin(np.radians(sheet_angle_deg))
    origin = np.array([0.25, 0.10])
    P = (np.stack([x, y], 1) - origin) @ np.array([[c, -s], [s, c]])  # into the sheet's frame
    P += rng.normal(0, noise_m, P.shape)
    size = np.full(len(t), 2 * R_BALL) + rng.normal(0, 0.0003, len(t))
    tr = Trajectory(clip="synthetic", capture_fps=fps, frame_size=(1280, 720), tracker="synthetic",
                    target="ball", sheet={}, frames=np.arange(len(t)), t=t, x=P[:, 0], y=P[:, 1],
                    confidence=np.ones(len(t)), size=size)
    tr.gravity = gravity.check(t, tr.x, tr.y, fps)
    phys["launch_sheet_m"] = ((np.zeros(2) - origin) @ np.array([[c, -s], [s, c]])).tolist()
    phys["sheet_origin_world"], phys["sheet_angle_deg"] = origin.tolist(), sheet_angle_deg
    return tr, phys


def _projector():
    Hw = cv2.getPerspectiveTransform(VIEW_WORLD, VIEW_IMG).astype(np.float64)

    def proj(px, py):
        return cv2.perspectiveTransform(np.array([[[px, py]]], np.float64), Hw)[0, 0]

    return proj


def _sheet(origin, angle_deg) -> np.ndarray:
    """A4 sheet corners in world coords, landscape, rotated by angle_deg."""
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    sheet = np.array([[0, 0], [0.297, 0], [0.297, 0.210], [0, 0.210]])
    return sheet @ np.array([[c, s], [-s, c]]) + origin


def _record(out: Path, base: np.ndarray, path_fn, proj, n: int, fps: float, rate: float, rng,
            r: float = R_BALL) -> list:
    """Draw the ball over `base` for n frames and pipe them to ffmpeg. Returns true positions."""
    H, W = base.shape[:2]
    sub = 4  # sub-frames for a short shutter (1/1000 s) blur
    shutter = 1 / 1000
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(rate), "-i", "-", "-c:v", "libx264", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    truth = []
    ring = np.stack([np.cos(np.linspace(0, 2 * np.pi, 48, endpoint=False)),
                     np.sin(np.linspace(0, 2 * np.pi, 48, endpoint=False))], 1) * r
    for ti in np.arange(n) / fps:
        acc = np.zeros((H, W, 3), np.float32)
        for k in range(sub):
            bx, by = path_fn(np.array([ti + shutter * k / sub]))
            # The ball's outline is a circle in the world plane; the keystoned view stretches it.
            outline = np.array([proj(bx[0] + dx, by[0] + dy) for dx, dy in ring])
            img = base.copy()
            cv2.fillPoly(img, [np.round(outline * 16).astype(np.int32)], (30, 30, 210),
                         lineType=cv2.LINE_AA, shift=4)
            acc += img
        frame = (acc / sub + rng.normal(0, 2.0, acc.shape)).clip(0, 255).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
        bx, by = path_fn(np.array([ti + shutter / 2]))
        truth.append([float(bx[0]), float(by[0])])
    proc.stdin.close()
    proc.wait()
    return truth


def _base(rng, size) -> np.ndarray:
    W, H = size
    base = np.full((H, W, 3), (62, 58, 55), np.uint8)
    return (base + rng.normal(0, 3, base.shape)).clip(0, 255).astype(np.uint8)


def _poly(img, proj, pts, color) -> None:
    cv2.fillPoly(img, [np.array([proj(*p) for p in pts]).astype(np.int32)], color)


def _draw_sheet(img, proj, sheet) -> None:
    sheet_img = np.array([proj(*p) for p in sheet])
    cv2.fillPoly(img, [np.round(sheet_img * 16).astype(np.int32)], (238, 240, 242),
                 lineType=cv2.LINE_AA, shift=4)


def render(out: Path, fps: float = 240.0, seconds: float = 0.85, export_fps: float | None = None,
           size=(1280, 720), seed: int = 7) -> dict:
    """The "fall" scene."""
    rng = np.random.default_rng(seed)
    proj = _projector()
    sheet = _sheet([0.10, 0.06], 4)
    base = _base(rng, size)
    _poly(base, proj, [(-0.40, 0.011), (0.0, 0.011), (0.0, 0.08), (-0.40, 0.08)], (40, 70, 110))
    _poly(base, proj, [(-0.40, 0.55), (0.65, 0.55), (0.65, 0.66), (-0.40, 0.66)], (45, 45, 45))
    _draw_sheet(base, proj, sheet)
    n = int(round(seconds * fps))
    rate = export_fps or fps
    truth = _record(Path(out), base, ball_path, proj, n, fps, rate, rng)
    meta = {"capture_fps": fps, "file_fps": rate, "g": G, "frames": n,
            "sheet_world_m": sheet.tolist(), "ball_world_m": truth}
    Path(out).with_suffix(".truth.json").write_text(json.dumps(meta))
    return meta


def render_ramp(out: Path, fps: float = 240.0, theta_deg: float = 18.0, s_release: float = 0.25,
                r_eff_ratio: float = 0.75, c_rr: float = 0.01, export_fps: float | None = None,
                size=(1280, 720), seed: int = 11) -> dict:
    """The "ramp" scene: release, groove rolling, launch off the lower end, flight to the floor."""
    rng = np.random.default_rng(seed)
    proj = _projector()
    floor = 0.55
    kw = dict(theta_deg=theta_deg, s_release=s_release, r_eff_ratio=r_eff_ratio, c_rr=c_rr, floor=floor)
    _, _, phys = ramp_path(np.zeros(1), **kw)
    seconds = phys["t_launch"] + (phys["x_land"] / (phys["v_launch"] * np.cos(np.radians(theta_deg)))) + 0.12
    sheet = _sheet([0.25, 0.10], 3)
    base = _base(rng, size)
    th = np.radians(theta_deg)
    d = np.array([np.cos(th), np.sin(th)])
    nrm = np.array([-np.sin(th), np.cos(th)])  # into the ramp, below the ball
    top_len = min(s_release + 0.05, 0.36 / np.cos(th))
    surf_end = nrm * R_BALL
    surf_top = surf_end - d * top_len
    # Slate blue: a brown ramp would pass the colour tracker's "red" test and merge with the ball.
    _poly(base, proj, [surf_top, surf_end, surf_end + nrm * 0.015, surf_top + nrm * 0.015], (95, 80, 60))
    _poly(base, proj, [(-0.40, floor), (0.65, floor), (0.65, 0.66), (-0.40, 0.66)], (45, 45, 45))
    _draw_sheet(base, proj, sheet)
    n = int(round(seconds * fps))
    rate = export_fps or fps

    def path_fn(t):
        x, y, _ = ramp_path(t, **kw)
        return x, y

    truth = _record(Path(out), base, path_fn, proj, n, fps, rate, rng)
    meta = {"scene": "ramp", "capture_fps": fps, "file_fps": rate, "g": G, "frames": n,
            "sheet_world_m": sheet.tolist(), "ball_world_m": truth, "physics": phys}
    Path(out).with_suffix(".truth.json").write_text(json.dumps(meta))
    return meta
