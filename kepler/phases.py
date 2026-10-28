"""Split a ramp-and-flight trajectory into phases and find the ramp's geometry.

A ball released from rest rolls down a straight ramp, leaves its lower end and
flies. From trajectory.json (sheet coordinates, metres) this finds:

- the down direction, from the free-flight parabola (so a crooked sheet is fine),
- the ramp line and its angle below horizontal,
- the launch point: where the flight parabola runs parallel to the ramp, which
  for a ball leaving a straight ramp is the moment it leaves,
- which samples are on the ramp (rest before release included) and which fly.

Distances along the ramp, s, are measured up the ramp from the launch point, so
a release mark is "this far up from the end".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gravity
from .trajectory import Trajectory


@dataclass
class RampGeometry:
    down: np.ndarray  # unit vector, sheet coords
    across: np.ndarray  # unit horizontal vector, the way the ball travels
    d: np.ndarray  # unit vector down the ramp
    theta: float  # ramp angle below horizontal, radians
    theta_sd: float  # its uncertainty, radians: ramp-line fit, down direction, sheet calibration
    launch: np.ndarray  # ball centre at launch, sheet coords, metres
    t_launch: float
    t_release_guess: float  # first clear movement; the fit refines the true release time
    ramp: np.ndarray  # sample indices used for the ramp fit (from just before release to launch)
    flight: np.ndarray  # sample indices in the free-flight window
    v_launch_flight: float  # launch speed from the flight parabola alone, m/s
    launch_gap_mm: float  # how far the flight parabola passes from the ramp line at launch
    noise_mm: float  # position noise while at rest, RMS
    ball_diameter_mm: float | None  # measured on the frames at rest, if sizes were recorded

    def s(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Distance up the ramp from the launch point, metres."""
        return (self.launch[None, :] - np.stack([x, y], 1)) @ self.d

    def to_json(self) -> dict:
        r = lambda v, n=6: [round(float(e), n) for e in v]
        return {
            "down": r(self.down), "across": r(self.across), "ramp_dir": r(self.d),
            "theta_deg": round(float(np.degrees(self.theta)), 4),
            "theta_sd_deg": round(float(np.degrees(self.theta_sd)), 4),
            "launch_m": r(self.launch), "t_launch": round(self.t_launch, 5),
            "t_release_guess": round(self.t_release_guess, 5),
            "ramp_samples": [int(self.ramp[0]), int(self.ramp[-1])],
            "flight_samples": [int(self.flight[0]), int(self.flight[-1])],
            "v_launch_flight": round(self.v_launch_flight, 4),
            "launch_gap_mm": round(self.launch_gap_mm, 3), "noise_mm": round(self.noise_mm, 3),
            "ball_diameter_mm": None if self.ball_diameter_mm is None else round(self.ball_diameter_mm, 2),
        }

    @classmethod
    def from_json(cls, d: dict) -> "RampGeometry":
        a = lambda k: np.array(d[k], float)
        return cls(down=a("down"), across=a("across"), d=a("ramp_dir"), theta=np.radians(d["theta_deg"]),
                   theta_sd=np.radians(d["theta_sd_deg"]), launch=a("launch_m"), t_launch=d["t_launch"], t_release_guess=d["t_release_guess"],
                   ramp=np.arange(d["ramp_samples"][0], d["ramp_samples"][1] + 1),
                   flight=np.arange(d["flight_samples"][0], d["flight_samples"][1] + 1),
                   v_launch_flight=d["v_launch_flight"], launch_gap_mm=d["launch_gap_mm"],
                   noise_mm=d["noise_mm"], ball_diameter_mm=d["ball_diameter_mm"])


def _line(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line through points: (centroid, unit direction)."""
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[0]


CALIBRATION_SD = np.radians(0.03)  # residual shear from the sheet's corners, a floor on θ's error


def segment(tr: Trajectory, rest_s: float = 0.04) -> RampGeometry:
    t, x, y = tr.t, tr.x, tr.y
    fps = tr.capture_fps
    P = np.stack([x, y], 1)
    g = tr.gravity if tr.gravity and "parabola" in tr.gravity else gravity.check(t, x, y, fps)
    par = g["parabola"]
    p0, v0, acc = (np.array(par[k], float) for k in ("p0", "v0", "a"))
    tf0 = par["t0"]
    w0, w1 = g["window_s"]
    flight = np.where((t >= w0 - 1e-9) & (t <= w1 + 1e-9))[0]
    down = acc / np.linalg.norm(acc)

    pre = np.where(t < tf0)[0]
    if len(pre) < 20:
        raise ValueError("fewer than 20 tracked frames before the free fall: no ramp to fit")

    # Release: the first sample clearly away from where the ball sat at the start.
    head = P[pre[:8]]
    rest = np.median(head, 0)
    noise = float(np.sqrt(np.mean(np.sum((head - rest) ** 2, 1))))
    moved = np.linalg.norm(P[pre] - rest, axis=1) > max(1.5e-3, 6 * noise)
    if not moved.any():
        raise ValueError("the ball never moves before the free fall")
    i_rel = int(pre[np.argmax(moved)])

    # The ramp line, from the samples that are surely on it (skip the last stretch before
    # the flight window, which may already be airborne), then keep what stays on it.
    cand = pre[pre >= i_rel]
    core = cand[: max(8, int(0.8 * len(cand)))]
    c, d = _line(P[core])
    if d @ down < 0:
        d = -d
    nrm = np.array([-d[1], d[0]])
    tol = max(1.0e-3, 4 * noise)
    off = np.abs((P[cand] - c) @ nrm) > tol
    # Leaving the line means 3 samples in a row off it; a single noisy one doesn't count.
    run = off[:-2] & off[1:-1] & off[2:] if len(off) > 2 else off
    keep = cand[: int(np.argmax(run)) if run.any() else len(cand)]
    if len(keep) < 8:
        raise ValueError("too few samples lie on a straight ramp")
    c, d = _line(P[keep])
    if d @ down < 0:
        d = -d
    nrm = np.array([-d[1], d[0]])
    # θ's error: the line's direction (slope error of the fit) and the down direction (the
    # parabola's acceleration, whose angle error is about its relative magnitude error).
    along = (P[keep] - c) @ d
    perp = (P[keep] - c) @ nrm
    sd_line = float(np.std(perp) / np.sqrt(np.sum((along - along.mean()) ** 2)))
    sd_down = float(g.get("g_std_error", 0.0) / max(g.get("g_measured", 9.81), 1e-9))
    theta_sd = float(np.sqrt(sd_line ** 2 + sd_down ** 2 + CALIBRATION_SD ** 2))

    # Launch: the flight velocity is parallel to the ramp there, i.e. n·v(τ) = 0.
    n_a = float(nrm @ acc)
    if abs(n_a) < 1e-6:
        raise ValueError("the flight doesn't curve away from the ramp line; is the ramp vertical?")
    tau_l = -float(nrm @ v0) / n_a
    launch = p0 + v0 * tau_l + 0.5 * acc * tau_l ** 2
    v_l = v0 + acc * tau_l
    gap = float(abs((launch - c) @ nrm)) * 1000
    t_launch = tf0 + tau_l
    if not (t[keep[0]] < t_launch <= tf0 + 0.05):
        raise ValueError(f"launch time {t_launch:.3f} s is outside the ramp-to-flight transition")

    t_rel = float(t[i_rel])
    start = t_rel - rest_s
    ramp = np.where((t >= start) & (t < t_launch - 0.5 / fps))[0]
    # Drop anything that drifted off the line before release (a nudge, a hand).
    ramp = ramp[np.abs((P[ramp] - c) @ nrm) <= tol]

    across = d - (d @ down) * down
    across /= np.linalg.norm(across)
    theta = float(np.arcsin(np.clip(d @ down, -1, 1)))

    diam = None
    if tr.size is not None:
        at_rest = pre[pre < i_rel]
        sizes = tr.size[at_rest] if len(at_rest) else np.array([])
        sizes = sizes[np.isfinite(sizes)]
        if len(sizes) >= 3:
            diam = float(np.median(sizes) * 1000)

    return RampGeometry(down=down, across=across, d=d, theta=theta, theta_sd=theta_sd, launch=launch,
                        t_launch=float(t_launch), t_release_guess=t_rel, ramp=ramp, flight=flight,
                        v_launch_flight=float(np.linalg.norm(v_l)), launch_gap_mm=gap,
                        noise_mm=noise * 1000, ball_diameter_mm=diam)
