"""Predict where the ball lands, and plan the release mark that puts it in a cup.

Geometry comes from one clip (phases.RampGeometry): the launch point, the ramp
direction and the down direction, all in sheet coordinates. A cup is the centre
of its rim, C. The ball goes in if its centre crosses the rim's height within
(rim radius - ball radius) of C.

`drop` is how far the rim sits below the launch point and `out` how far it sits
from it horizontally, both in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .harness.fit import FitResult
from .phases import RampGeometry
from .theory import Theory, by_id

G = 9.81


def cup_offsets(geom: RampGeometry, cup: np.ndarray) -> tuple[float, float]:
    """(drop, out) of a rim centre given in sheet coordinates."""
    v = np.asarray(cup, float) - geom.launch
    return float(v @ geom.down), float(v @ geom.across)


def launch_speed(theory: Theory, p: dict, theta: float, known: dict, s_release: float) -> float:
    tau = theory.time_to_travel(p, s_release, theta, known)
    return float(theory.speed(p, tau, theta, known)) if np.isfinite(tau) else 0.0


def landing_out(v: float, theta: float, drop: float, g: float = G) -> float:
    """Horizontal distance from the launch point where the centre falls through `drop`."""
    vx, vz = v * np.cos(theta), v * np.sin(theta)  # out, down
    tf = (-vz + np.sqrt(vz * vz + 2 * g * drop)) / g
    return float(vx * tf)


def flight_points(geom: RampGeometry, v: float, drop: float, n: int = 40, g: float = G) -> np.ndarray:
    """The predicted flight in sheet coordinates, from launch down to the rim's height."""
    vz = v * np.sin(geom.theta)
    tf = (-vz + np.sqrt(vz * vz + 2 * g * drop)) / g
    tau = np.linspace(0, tf, n)[:, None]
    return geom.launch[None, :] + v * geom.d[None, :] * tau + 0.5 * g * geom.down[None, :] * tau ** 2


def _solve(f, lo: float, hi: float) -> float | None:
    a, b = f(lo), f(hi)
    if not (np.isfinite(a) and np.isfinite(b)) or a > 0 or b < 0:
        return None
    return float(brentq(f, lo, hi, xtol=1e-6))


@dataclass
class Plan:
    label: str
    s_release: float | None  # metres up the ramp from its end; None if the cup is out of reach
    s_interval: tuple[float, float] | None  # 5th-95th percentile over the theory's uncertainty
    hit_probability: float | None
    landing_out: float | None  # where this theory says the ball crosses rim height, metres
    note: str = ""

    def to_json(self) -> dict:
        r = lambda v: None if v is None else round(float(v), 5)
        return {"label": self.label, "s_release_m": r(self.s_release),
                "s_interval_m": None if self.s_interval is None else [r(v) for v in self.s_interval],
                "hit_probability": r(self.hit_probability), "landing_out_m": r(self.landing_out),
                "note": self.note}


def plan_release(fit: FitResult, theta: float, known: dict, drop: float, out: float,
                 rim_diameter: float, s_range: tuple[float, float], place_sd: float = 0.001,
                 cup_sd: float = 0.001, seed: int = 0, label: str | None = None) -> Plan:
    """Release mark for one fitted theory, with its uncertainty and the chance of a hit.

    place_sd: how precisely a hand sets the ball on the mark; cup_sd: how precisely the
    cup's position is known. Both are in metres and both count against the hit chance."""
    th = by_id(fit.theory)
    lo, hi = s_range
    land = lambda p, s: landing_out(launch_speed(th, p, theta, known, s), theta, drop)
    s_best = _solve(lambda s: land(fit.params, s) - out, lo, hi)
    if s_best is None:
        return Plan(label or fit.theory, None, None, None, None,
                    note=f"the cup is out of reach for releases {lo * 100:.0f}-{hi * 100:.0f} cm up the ramp")
    draws = fit.param_draws()
    per_draw = [_solve(lambda s, p=p: land(p, s) - out, lo, hi) for p in draws]
    per_draw = np.array([v for v in per_draw if v is not None])
    interval = tuple(np.percentile(per_draw, [5, 95])) if len(per_draw) else (s_best, s_best)

    rng = np.random.default_rng(seed)
    tol = rim_diameter / 2 - known["R"]
    k = max(len(draws), 200)
    picks = [draws[i % len(draws)] for i in range(k)]
    s_hand = s_best + rng.normal(0, place_sd, k)
    miss = np.array([land(p, s) for p, s in zip(picks, s_hand)]) - (out + rng.normal(0, cup_sd, k))
    return Plan(label or fit.theory, s_best, (float(interval[0]), float(interval[1])),
                float(np.mean(np.abs(miss) <= tol)), float(land(fit.params, s_best)))
