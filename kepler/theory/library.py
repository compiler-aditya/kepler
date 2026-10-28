"""The standard library of ramp theories.

Every experiment is tested against these, so a vague or wrong proposal from the
agent can never leave Kepler with nothing. Each theory gives the ball's distance
travelled down the ramp, q, at time tau after release from rest. All of them are
closed-form, so the harness can fit and bootstrap them quickly.

On one ramp angle every theory here except rolling_ideal predicts constant
acceleration, so they fit one clip equally well: they disagree only about how
the acceleration changes with the angle. That is why a tie is the normal
outcome of a single clip, and why the next run changes the angle.

Rolling resistance sits inside the inertia factor:

    m a (1 + I / (m r^2)) = m g sin θ - c_rr m g cos θ

so c_rr is the physical coefficient (force / normal force), the number that
goes into MuJoCo (as c_rr * R, a length; see export/mjcf.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

G = 9.81


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    unit: str
    guess: float


@dataclass(frozen=True)
class Theory:
    id: str
    summary: str
    law: str
    params: tuple[Param, ...]
    # Constant driving acceleration A(params, θ, known); the motion is dv/dt = A - b v.
    accel: Callable[[dict, float, dict], float]
    drag: str | None = None  # name of the linear-drag parameter b (1/s), if any
    rolls: bool = True  # False: the object slides without turning (affects the MuJoCo export)

    def _ab(self, p: dict, theta: float, known: dict) -> tuple[float, float]:
        return max(float(self.accel(p, theta, known)), 0.0), (p[self.drag] if self.drag else 0.0)

    def distance(self, p: dict, tau: np.ndarray, theta: float, known: dict) -> np.ndarray:
        """Distance travelled down the ramp, metres, tau seconds after release (0 before)."""
        A, b = self._ab(p, theta, known)
        tau = np.maximum(np.asarray(tau, float), 0.0)
        if b < 1e-6:
            return 0.5 * A * tau ** 2
        return A / b * (tau - (1 - np.exp(-b * tau)) / b)

    def speed(self, p: dict, tau: np.ndarray, theta: float, known: dict) -> np.ndarray:
        A, b = self._ab(p, theta, known)
        tau = np.maximum(np.asarray(tau, float), 0.0)
        if b < 1e-6:
            return A * tau
        return A / b * (1 - np.exp(-b * tau))

    def time_to_travel(self, p: dict, dist: float, theta: float, known: dict) -> float:
        """Time after release to cover `dist` down the ramp (inf if it never gets there)."""
        A, b = self._ab(p, theta, known)
        if A <= 0:
            return float("inf")
        if b < 1e-6:
            return float(np.sqrt(2 * dist / A))
        from scipy.optimize import brentq
        hi = np.sqrt(2 * dist / A) + 1.0
        while self.distance(p, hi, theta, known) < dist:
            hi *= 2
            if hi > 1e4:
                return float("inf")
        return float(brentq(lambda tau: self.distance(p, tau, theta, known) - dist, 0.0, hi, xtol=1e-9))


def _rolling_factor(ratio: float) -> float:
    """1 + I / (m r^2) for a solid ball touching at r = ratio * R."""
    return 1 + 0.4 / ratio ** 2


_c_rr = Param("c_rr", 0.0, 0.1, "", 0.01)

LIBRARY: tuple[Theory, ...] = (
    Theory(
        "rolling_ideal",
        "Solid ball rolls without slipping and loses nothing: the textbook case, and what a "
        "simulator does with its default contact settings",
        "a = (5/7) g sin θ",
        (),
        lambda p, th, k: G * np.sin(th) / _rolling_factor(1.0),
    ),
    Theory(
        "rolling",
        "Rolls without slipping on a flat surface, with rolling resistance",
        "a = g (sin θ - c_rr cos θ) / (7/5)",
        (_c_rr,),
        lambda p, th, k: G * (np.sin(th) - p["c_rr"] * np.cos(th)) / _rolling_factor(1.0),
    ),
    Theory(
        "rolling_groove",
        "Rolls in a groove or on two rails: it touches at a radius r_eff smaller than its own, "
        "so more of its energy goes into spin",
        "a = g (sin θ - c_rr cos θ) / (1 + (2/5) R²/r_eff²)",
        (Param("r_eff_ratio", 0.3, 1.0, "of R", 0.8), _c_rr),
        lambda p, th, k: G * (np.sin(th) - p["c_rr"] * np.cos(th)) / _rolling_factor(p["r_eff_ratio"]),
    ),
    Theory(
        "sliding",
        "Slides without turning, with Coulomb friction",
        "a = g (sin θ - μ cos θ)",
        (Param("mu", 0.0, 1.5, "", 0.2),),
        lambda p, th, k: G * (np.sin(th) - p["mu"] * np.cos(th)),
        rolls=False,
    ),
    Theory(
        "rolling_drag",
        "Rolls on a flat surface with rolling resistance and linear air drag",
        "dv/dt = g (sin θ - c_rr cos θ) / (7/5) - b v",
        (_c_rr, Param("b", 0.0, 5.0, "1/s", 0.1)),
        lambda p, th, k: G * (np.sin(th) - p["c_rr"] * np.cos(th)) / _rolling_factor(1.0),
        drag="b",
    ),
)


def by_id(theory_id: str) -> Theory:
    for th in LIBRARY:
        if th.id == theory_id:
            return th
    raise KeyError(f"no theory {theory_id!r}; known: {[t.id for t in LIBRARY]}")
