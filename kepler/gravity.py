"""The day 2 gate: measure g from the free-fall part of the path.

A free-falling ball has constant acceleration of magnitude g. Fitting a
parabola to x(t) and y(t) gives that acceleration as a vector, so the check
works even if the A4 sheet was taped crooked. One number tests the frame
rate, the scale and the tracking together.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

G_REF = 9.81
COMMON_FPS = [24, 25, 30, 48, 50, 60, 90, 100, 120, 180, 200, 240, 250, 480, 500, 960, 1000]


def find_free_fall(t: np.ndarray, x: np.ndarray, y: np.ndarray, fps: float) -> tuple[int, int]:
    """Return (i0, i1) sample indices, inclusive, of the longest constant-acceleration run
    with the strongest acceleration (the fall). Raises ValueError if none is found."""
    # The right smoothing window depends on the true frame rate, which may be the
    # thing that's wrong, so try several and keep the longest clean run.
    best: tuple[int, int] | None = None
    for win in (7, 11, 15, 21, 31, 41):
        if len(t) < win + 10:
            break
        try:
            i0, i1 = _run_for_window(t, x, y, win)
        except ValueError:
            continue
        if best is None or (i1 - i0) > (best[1] - best[0]):
            best = (i0, i1)
    if best is None:
        raise ValueError("no clean free-fall run found; pass --fall START,END in seconds")
    return best


def _run_for_window(t: np.ndarray, x: np.ndarray, y: np.ndarray, win: int) -> tuple[int, int]:
    n = len(t)
    dt = float(np.median(np.diff(t)))
    ax = savgol_filter(x, win, 2, deriv=2, delta=dt)
    ay = savgol_filter(y, win, 2, deriv=2, delta=dt)
    amag = np.hypot(ax, ay)
    level = np.percentile(amag, 90)
    if level <= 0:
        raise ValueError("no acceleration found")
    on = amag > 0.6 * level
    # Direction must stay steady within the run (separates a ramp from a fall).
    ang = np.arctan2(ay, ax)
    best = (0, -1)
    i = 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        ref = np.angle(np.mean(np.exp(1j * ang[i:min(n, i + win)])))
        while j + 1 < n and on[j + 1] and abs(np.angle(np.exp(1j * (ang[j + 1] - ref)))) < np.radians(25):
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j + 1
    i0, i1 = best
    # The smoothing window smears the run's edges; trim half a window each side.
    i0, i1 = i0 + win // 2, i1 - win // 2
    if i1 - i0 < 12:
        raise ValueError("free-fall run is too short; pass --fall START,END in seconds")
    # Also require the frames to be contiguous in time (no tracking gaps inside).
    gaps = np.where(np.diff(t[i0:i1 + 1]) > 1.5 * dt)[0]
    if len(gaps):
        seg = np.split(np.arange(i0, i1 + 1), gaps + 1)
        longest = max(seg, key=len)
        i0, i1 = int(longest[0]), int(longest[-1])
    return i0, i1


def check(t: np.ndarray, x: np.ndarray, y: np.ndarray, fps: float,
          window_s: tuple[float, float] | None = None, tol_pct: float = 2.0) -> dict:
    """tol_pct: how far g may be from 9.81 to pass. A sheet 1 cm off the ball's plane, filmed
    from 1 m, already costs 1%; rolling shutter and blur add more, so real clips may need 3."""
    if window_s:
        sel = np.where((t >= window_s[0]) & (t <= window_s[1]))[0]
        if len(sel) < 12:
            raise ValueError("the --fall window holds fewer than 12 tracked frames")
        i0, i1 = int(sel[0]), int(sel[-1])
    else:
        i0, i1 = find_free_fall(t, x, y, fps)
    tt = t[i0:i1 + 1] - t[i0]
    px, cx = np.polyfit(tt, x[i0:i1 + 1], 2, cov=True)
    py, cy = np.polyfit(tt, y[i0:i1 + 1], 2, cov=True)
    ax, ay = 2 * px[0], 2 * py[0]
    sax, say = 2 * np.sqrt(cx[0, 0]), 2 * np.sqrt(cy[0, 0])
    g = float(np.hypot(ax, ay))
    g_err = float(np.sqrt((ax * sax) ** 2 + (ay * say) ** 2) / g) if g > 0 else float("nan")
    res = np.hypot(x[i0:i1 + 1] - np.polyval(px, tt), y[i0:i1 + 1] - np.polyval(py, tt))
    err_pct = (g / G_REF - 1) * 100
    out = {
        "g_measured": round(g, 4),
        "g_std_error": round(g_err, 4),
        "error_pct": round(err_pct, 2),
        "passed": bool(abs(err_pct) <= tol_pct),
        "tolerance_pct": tol_pct,
        "down_direction_deg": round(float(np.degrees(np.arctan2(ay, ax))), 2),
        "window_s": [round(float(t[i0]), 4), round(float(t[i1]), 4)],
        "frames_used": int(i1 - i0 + 1),
        "fit_rms_mm": round(float(np.sqrt(np.mean(res ** 2)) * 1000), 3),
        # The flight parabola itself, for the ramp's launch point: p(τ) = p0 + v0 τ + ½ a τ², τ = t - t0.
        "parabola": {"t0": round(float(t[i0]), 6), "p0": [float(px[2]), float(py[2])],
                     "v0": [float(px[1]), float(py[1])], "a": [float(ax), float(ay)]},
    }
    if abs(err_pct) > 10 and g > 0:
        implied = fps * np.sqrt(G_REF / g)
        nearest = min(COMMON_FPS, key=lambda f: abs(f - implied))
        out["hint"] = (
            f"g is far off. If the clip's timing is wrong, its true capture rate is about "
            f"{implied:.0f} fps (nearest common: {nearest}). Re-run with --capture-fps {nearest}."
        )
    elif not out["passed"]:
        out["hint"] = ("Check, in order: the A4 sheet sits in the ball's plane, the 1x lens was used, "
                       "the phone is square to the motion, and there's enough light to cut blur.")
    return out


def describe(res: dict) -> str:
    mark = "PASS" if res["passed"] else "FAIL"
    s = (f"[{mark}] g = {res['g_measured']:.3f} ± {res['g_std_error']:.3f} m/s² "
         f"({res['error_pct']:+.2f}% vs 9.81, limit ±{res.get('tolerance_pct', 2.0):g}%) "
         f"from {res['frames_used']} frames, "
         f"{res['window_s'][0]:.3f}-{res['window_s'][1]:.3f} s, fit RMS {res['fit_rms_mm']:.2f} mm")
    if "hint" in res:
        s += "\n       " + res["hint"]
    return s
