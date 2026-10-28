"""Run the ramp scene in MuJoCo: release at a mark, report where the ball crosses the rim's height.

The same planner as predict.py, but with the simulator as the model. With the
default scene that's the plan a robotics team would make out of the box; with
Kepler's scene it should agree with Kepler's own prediction, which checks the
export.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _model(xml: str):
    import mujoco
    return mujoco, mujoco.MjModel.from_xml_string(xml)


def run(xml: str, theta: float, s_release: float, drop: float, t_max: float = 4.0,
        keep_path: bool = False):
    """Returns (out, path): horizontal distance from the launch point where the ball's centre
    falls through `drop` below it (nan if it never does), and optionally the path (N x 2, x z)."""
    mujoco, m = _model(xml)
    d = mujoco.MjData(m)
    d.qpos[:3] = [-s_release * np.cos(theta), 0.0, s_release * np.sin(theta)]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    path = [d.qpos[[0, 2]].copy()] if keep_path else None
    prev = d.qpos[[0, 2]].copy()
    steps = int(t_max / m.opt.timestep)
    for i in range(steps):
        mujoco.mj_step(m, d)
        cur = d.qpos[[0, 2]].copy()
        if keep_path and i % 4 == 0:
            path.append(cur)
        if cur[1] <= -drop and prev[1] > -drop:
            w = (prev[1] + drop) / (prev[1] - cur[1])
            out = float(prev[0] + w * (cur[0] - prev[0]))
            if keep_path:
                path.append(np.array([out, -drop]))
            return out, (np.array(path) if keep_path else None)
        prev = cur
    return float("nan"), (np.array(path) if keep_path else None)


def plan(xml: str, theta: float, drop: float, out: float, s_range: tuple[float, float]) -> float | None:
    """Release mark (metres up the ramp from its end) that lands the simulated ball at `out`."""
    f = lambda s: run(xml, theta, s, drop)[0] - out
    lo, hi = s_range
    a, b = f(lo), f(hi)
    if not (np.isfinite(a) and np.isfinite(b)) or a > 0 or b < 0:
        return None
    return float(brentq(f, lo, hi, xtol=2e-5))
