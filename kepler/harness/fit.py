"""Fit theories to ramp clips, score them on frames they never saw, and pick a winner.

The language model writes physics; this trusted code does the statistics.

- Time split: the first 80% of each clip's ramp motion is used for fitting and
  the last 20% is held out, so a theory is judged on forecasting, not
  interpolation.
- Physical parameters are shared by every clip of the experiment; the release
  time and release point are fitted per clip.
- Error bars come from the Jacobian, and a moving-block bootstrap over the
  residuals (which are correlated in time) gives the spread that the verdict
  and the predictions use. Each bootstrap draw also redraws every clip's ramp
  angle from its measurement error: c_rr and r_eff are sensitive to a tenth of
  a degree, and position noise alone would hide that.
- Verdict: the best held-out theory wins only if the runner-up is at least 15%
  worse and the gap beats both bootstrap spreads. Otherwise it's a tie.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..phases import RampGeometry
from ..theory import LIBRARY, Theory
from ..trajectory import Trajectory


@dataclass
class ClipData:
    name: str
    t: np.ndarray  # seconds
    s: np.ndarray  # metres up the ramp from the launch point
    theta: float
    t_release_guess: float
    s_release_guess: float
    split: float  # samples after this time are held out
    theta_sd: float = 0.0

    @classmethod
    def from_trajectory(cls, name: str, tr: Trajectory, geom: RampGeometry,
                        holdout: float = 0.2) -> "ClipData":
        idx = geom.ramp
        t = tr.t[idx]
        s = geom.s(tr.x[idx], tr.y[idx])
        t_rel = geom.t_release_guess
        split = t_rel + (1 - holdout) * (geom.t_launch - t_rel)
        at_rest = t < t_rel
        s0 = float(np.median(s[at_rest])) if at_rest.sum() >= 3 else float(s.max())
        return cls(name, t, s, geom.theta, t_rel, s0, split, geom.theta_sd)


@dataclass
class FitResult:
    theory: str
    params: dict[str, float]
    std: dict[str, float]  # Jacobian standard errors
    spread: dict[str, list[float]]  # bootstrap 5th-95th percentile
    identified: dict[str, bool]  # False when the data can't pin the parameter down
    ics: list[dict]  # per clip: release time t0 and release point s_release
    train_rms_mm: float
    holdout_rms_mm: float
    holdout_rms_mm_by_clip: list[float]
    holdout_spread_mm: float  # bootstrap standard deviation of holdout_rms_mm
    bic: float
    n_params: int
    draws: np.ndarray = field(repr=False)  # bootstrap draws of the physical parameters, B x k

    def to_json(self) -> dict:
        r = lambda v: round(float(v), 6)
        return {
            "theory": self.theory, "params": {k: r(v) for k, v in self.params.items()},
            "std": {k: (r(v) if np.isfinite(v) else None) for k, v in self.std.items()},
            "spread_5_95": {k: [r(a), r(b)] for k, (a, b) in self.spread.items()},
            "identified": self.identified,
            "ics": [{k: r(v) for k, v in ic.items()} for ic in self.ics],
            "train_rms_mm": r(self.train_rms_mm), "holdout_rms_mm": r(self.holdout_rms_mm),
            "holdout_rms_mm_by_clip": [r(v) for v in self.holdout_rms_mm_by_clip],
            "holdout_spread_mm": r(self.holdout_spread_mm), "bic": r(self.bic),
            "n_params": self.n_params, "draws": np.round(self.draws, 7).tolist(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "FitResult":
        k = len(d["params"])
        std = {k: (float("inf") if v is None else v) for k, v in d["std"].items()}
        return cls(theory=d["theory"], params=d["params"], std=std, spread=d["spread_5_95"],
                   identified=d["identified"], ics=d["ics"], train_rms_mm=d["train_rms_mm"],
                   holdout_rms_mm=d["holdout_rms_mm"], holdout_rms_mm_by_clip=d["holdout_rms_mm_by_clip"],
                   holdout_spread_mm=d["holdout_spread_mm"], bic=d["bic"], n_params=d["n_params"],
                   draws=np.array(d["draws"], float).reshape(len(d["draws"]), k))

    def param_draws(self) -> list[dict]:
        names = list(self.params)
        return [dict(zip(names, row)) for row in self.draws] or [dict(self.params)]


def _unpack(theory: Theory, x: np.ndarray, n_clips: int) -> tuple[dict, list[tuple[float, float]]]:
    k = len(theory.params)
    p = {pr.name: float(x[i]) for i, pr in enumerate(theory.params)}
    ics = [(float(x[k + 2 * j]), float(x[k + 2 * j + 1])) for j in range(n_clips)]
    return p, ics


def _model(theory: Theory, p: dict, ic: tuple[float, float], clip: ClipData, t: np.ndarray,
           known: dict, theta: float | None = None) -> np.ndarray:
    t0, s0 = ic
    return s0 - theory.distance(p, t - t0, clip.theta if theta is None else theta, known)


def fit_theory(theory: Theory, clips: list[ClipData], known: dict, bootstrap: int = 200,
               block: int = 12, seed: int = 0) -> FitResult:
    k = len(theory.params)
    x0 = [pr.guess for pr in theory.params]
    lo = [pr.lo for pr in theory.params]
    hi = [pr.hi for pr in theory.params]
    for c in clips:
        x0 += [c.t_release_guess - 0.03, c.s_release_guess]
        lo += [c.t_release_guess - 0.3, c.s_release_guess - 0.02]
        hi += [c.t_release_guess + 0.02, c.s_release_guess + 0.02]
    x0, lo, hi = (np.array(v, float) for v in (x0, lo, hi))
    train = [c.t <= c.split for c in clips]

    def resid(x, ys=None, thetas=None):
        p, ics = _unpack(theory, x, len(clips))
        return np.concatenate([
            _model(theory, p, ics[j], c, c.t[train[j]], known, None if thetas is None else thetas[j])
            - (c.s[train[j]] if ys is None else ys[j])
            for j, c in enumerate(clips)])

    def holdout(x) -> tuple[float, list[float]]:
        p, ics = _unpack(theory, x, len(clips))
        errs = [_model(theory, p, ics[j], c, c.t[~train[j]], known) - c.s[~train[j]]
                for j, c in enumerate(clips)]
        by_clip = [float(np.sqrt(np.mean(e ** 2)) * 1000) if len(e) else float("nan") for e in errs]
        allv = np.concatenate(errs)
        return float(np.sqrt(np.mean(allv ** 2)) * 1000), by_clip

    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", method="trf")
    r = sol.fun
    n, K = len(r), len(sol.x)
    rss = float(r @ r)
    p, ics = _unpack(theory, sol.x, len(clips))

    J = sol.jac
    s2 = rss / max(1, n - K)
    cov = np.linalg.pinv(J.T @ J) * s2
    std = {pr.name: float(np.sqrt(max(cov[i, i], 0))) for i, pr in enumerate(theory.params)}
    # Exactly degenerate parameters (two that only matter through one combination, like
    # r_eff and c_rr on a single angle) leave a flat valley the bootstrap never explores,
    # because it restarts at the solution. Find that valley in the Jacobian instead.
    scale = np.maximum(hi - lo, 1e-12)
    _, sv, vt = np.linalg.svd(J * scale[None, :], full_matrices=False)
    flat = vt[sv < 1e-7 * sv.max()] if sv.size and sv.max() > 0 else np.zeros((0, K))
    in_valley = {pr.name: bool(flat.size and np.abs(flat[:, i]).max() > 0.1)
                 for i, pr in enumerate(theory.params)}

    # Moving-block bootstrap: refit to (fit + resampled residual blocks), per clip.
    rng = np.random.default_rng(seed)
    fitted = []
    res_by_clip = []
    for j, c in enumerate(clips):
        f = _model(theory, p, ics[j], c, c.t[train[j]], known)
        fitted.append(f)
        res_by_clip.append(c.s[train[j]] - f)
    draws, hold_draws = [], []
    for _ in range(bootstrap):
        ys = []
        for f, e in zip(fitted, res_by_clip):
            m = len(e)
            L = min(block, m)
            starts = rng.integers(0, m - L + 1, int(np.ceil(m / L)))
            ys.append(f + np.concatenate([e[s:s + L] for s in starts])[:m])
        thetas = [c.theta + rng.normal(0, c.theta_sd) for c in clips]
        sb = least_squares(lambda x: resid(x, ys, thetas), sol.x, bounds=(lo, hi), x_scale="jac",
                           method="trf", max_nfev=200)
        draws.append(sb.x[:k])
        hold_draws.append(holdout(sb.x)[0])
    draws = np.array(draws, float).reshape(len(draws), k)
    spread, identified = {}, {}
    for i, pr in enumerate(theory.params):
        q5, q95 = (np.percentile(draws[:, i], [5, 95]) if len(draws) else (p[pr.name], p[pr.name]))
        spread[pr.name] = [float(q5), float(q95)]
        identified[pr.name] = bool((q95 - q5) < 0.5 * (pr.hi - pr.lo)) and not in_valley[pr.name]
        if in_valley[pr.name]:
            std[pr.name] = float("inf")

    h, h_by = holdout(sol.x)
    return FitResult(
        theory=theory.id, params=p, std=std, spread=spread, identified=identified,
        ics=[{"t0": t0, "s_release": s0} for t0, s0 in ics],
        train_rms_mm=float(np.sqrt(rss / n) * 1000), holdout_rms_mm=h, holdout_rms_mm_by_clip=h_by,
        holdout_spread_mm=float(np.std(hold_draws)) if hold_draws else 0.0,
        bic=float(n * np.log(rss / n) + K * np.log(n)), n_params=k, draws=draws,
    )


def fit_all(clips: list[ClipData], known: dict, theories=LIBRARY, **kw) -> list[FitResult]:
    return [fit_theory(th, clips, known, **kw) for th in theories]


def verdict(results: list[FitResult], margin: float = 0.15) -> dict:
    """Rank by held-out error (BIC breaks exact ties, so simpler theories come first)."""
    ranked = sorted(results, key=lambda r: (round(r.holdout_rms_mm, 3), r.bic))
    best = ranked[0]
    tied = [best]
    rejected = []
    for r in ranked[1:]:
        gap = r.holdout_rms_mm - best.holdout_rms_mm
        clear = (r.holdout_rms_mm >= best.holdout_rms_mm * (1 + margin)
                 and gap > max(best.holdout_spread_mm, r.holdout_spread_mm))
        (rejected if clear else tied).append(r)
    return {
        "winner": best.theory if len(tied) == 1 else None,
        "tied": [r.theory for r in tied],
        "rejected": [r.theory for r in rejected],
        "ranking": [r.theory for r in ranked],
    }
