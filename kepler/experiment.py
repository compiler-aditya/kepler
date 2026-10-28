"""One experiment: one ramp and ball, filmed in one or more clips from a fixed camera.

    fit:  trajectories -> scale from gravity -> ramp geometry per clip -> every
          theory fitted jointly, scored on held-out frames -> verdict. Writes fit.json.
    plan: a cup's position -> the release mark under MuJoCo's default physics and
          under Kepler's, the chance each lands in the cup, and MuJoCo scenes for
          both. Writes plan.json, scene_default.xml, scene_kepler.xml,
          physics_card.json (and plan.png when the clip is on disk).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import gravity, sim
from .export.mjcf import kepler_contact, scene_xml
from .harness import ClipData, FitResult, fit_all, verdict
from .phases import RampGeometry, segment
from .predict import cup_offsets, flight_points, landing_out, launch_speed, plan_release
from .theory import by_id
from .trajectory import Trajectory


def _trajectory_path(p: Path) -> Path:
    p = Path(p)
    return p / "trajectory.json" if p.is_dir() else p


def rescale_by_gravity(tr: Trajectory, max_pct: float = 10.0) -> float:
    """Gravity is the ruler: scale the clip so its free fall measures 9.81 m/s^2.

    The A4 sheet fixes the plane's perspective, but if the sheet sits a little in front
    of or behind the ball's plane every distance is off by (gap / camera distance). The
    flight phase measures that error directly, since frame times are exact, so remove it.
    Errors over max_pct mean wrong timing, not wrong scale (the gravity gate says so); those
    clips are left alone. Returns the factor applied."""
    g = tr.gravity
    if not g or abs(g["error_pct"]) > max_pct:
        return 1.0
    k = gravity.G_REF / g["g_measured"]
    tr.x, tr.y = tr.x * k, tr.y * k
    if tr.size is not None:
        tr.size = tr.size * k
    tr.gravity = gravity.check(tr.t, tr.x, tr.y, tr.capture_fps, tuple(g["window_s"]),
                               g.get("tolerance_pct", 2.0))
    return float(k)


def fit(runs: list[Path], out: Path, ball_diameter_mm: float | None = None, holdout: float = 0.2,
        bootstrap: int = 200, trajectories: list[Trajectory] | None = None) -> dict:
    """runs: run directories (or trajectory.json paths). `trajectories` skips loading (tests)."""
    trs = trajectories or [Trajectory.load(_trajectory_path(r)) for r in runs]
    names = [str(r) for r in runs] if runs else [f"clip{i}" for i in range(len(trs))]
    g_before = [tr.gravity["g_measured"] if tr.gravity else None for tr in trs]
    scales = [rescale_by_gravity(tr) for tr in trs]
    geoms = [segment(tr) for tr in trs]
    measured = [g.ball_diameter_mm for g in geoms if g.ball_diameter_mm]
    diam = ball_diameter_mm or (float(np.median(measured)) if measured else None)
    if not diam:
        raise SystemExit("ball size unknown: pass --ball-diameter MM")
    known = {"R": diam / 2000}
    clips = [ClipData.from_trajectory(n, tr, g, holdout) for n, tr, g in zip(names, trs, geoms)]
    results = fit_all(clips, known, bootstrap=bootstrap)
    v = verdict(results)
    best = next(r for r in results if r.theory == v["ranking"][0])
    th = by_id(best.theory)
    checks = []
    for c, g, ic in zip(clips, geoms, best.ics):
        v_ramp = launch_speed(th, best.params, c.theta, known, ic["s_release"])
        checks.append({"v_launch_ramp_fit": round(v_ramp, 4), "v_launch_flight_fit": round(g.v_launch_flight, 4),
                       "diff_pct": round((v_ramp / g.v_launch_flight - 1) * 100, 2)})
    doc = {
        "version": 1, "known": known, "ball_diameter_mm": diam,
        "ball_diameter_measured_mm": float(np.median(measured)) if measured else None,
        "holdout": holdout, "bootstrap": bootstrap,
        "clips": [{"name": n, "trajectory": str(_trajectory_path(r)) if runs else None,
                   "g_measured": gb, "scale_from_gravity": round(k, 6),
                   "geometry": g.to_json(), "launch_speed_check": ck}
                  for n, r, gb, k, g, ck in zip(names, runs or [None] * len(trs), g_before, scales, geoms, checks)],
        "results": [r.to_json() for r in results],
        "verdict": v,
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fit.json").write_text(json.dumps(doc, indent=1))
    return doc


def plan(exp: Path, cup_m: np.ndarray | None = None, cup_px: tuple[float, float] | None = None,
         rim_diameter: float = 0.075, clip_index: int = -1, ramp_length: float | None = None,
         theory: str | None = None, frame_index: int = 0, place_sd: float = 0.001) -> dict:
    exp = Path(exp)
    doc = json.loads((exp / "fit.json").read_text())
    known = doc["known"]
    R = known["R"]
    clip = doc["clips"][clip_index]
    geom = RampGeometry.from_json(clip["geometry"])
    kappa = clip.get("scale_from_gravity", 1.0)  # sheet metres -> metres in the ball's plane
    tr = Trajectory.load(Path(clip["trajectory"])) if clip.get("trajectory") else None
    H = np.array(tr.sheet["homography_px_to_mm"]) if tr and tr.sheet else None
    if cup_m is None:
        if cup_px is None or H is None:
            raise SystemExit("give the cup as --cup-m X,Y (sheet metres) or --cup-px X,Y (needs a tracked clip)")
        from .sheet import px_to_mm
        cx, cy = px_to_mm(H, np.array([cup_px[0]]), np.array([cup_px[1]]))
        cup_m = np.array([cx[0], cy[0]]) / 1000 * kappa
    cup_m = np.asarray(cup_m, float)
    drop, out = cup_offsets(geom, cup_m)

    results = {r["theory"]: FitResult.from_json(r) for r in doc["results"]}
    v = doc["verdict"]
    chosen = theory or v["winner"] or v["tied"][0]
    fit_r = results[chosen]
    released = [ic["s_release"] for ic in fit_r.ics]
    top = ramp_length or max(0.6, 1.5 * max(released))
    s_range = (0.01, top - R)

    kepler = plan_release(fit_r, geom.theta, known, drop, out, rim_diameter, s_range, place_sd=place_sd)
    if kepler.s_release is not None and not (min(released) - 0.02 <= kepler.s_release <= max(released) + 0.02):
        kepler.note = (f"outside the releases filmed ({min(released) * 100:.1f}-{max(released) * 100:.1f} cm): "
                       f"the physics carries over, but film one release near this mark before the shot")
    tied_marks = {}
    for tid in v["tied"]:
        if tid != chosen:
            p = plan_release(results[tid], geom.theta, known, drop, out, rim_diameter, s_range, place_sd=place_sd)
            tied_marks[tid] = p.s_release

    default_xml = scene_xml(geom.theta, R, top)
    contact = kepler_contact(chosen, fit_r.params, R)
    kepler_xml = scene_xml(geom.theta, R, top, contact)
    s_default = sim.plan(default_xml, geom.theta, drop, out, s_range)

    th = by_id(chosen)
    land = lambda s: landing_out(launch_speed(th, fit_r.params, geom.theta, known, s), geom.theta, drop)
    tol = rim_diameter / 2 - R
    default_miss = None if s_default is None else land(s_default) - out
    export_check = None
    if kepler.s_release is not None:
        x_sim, _ = sim.run(kepler_xml, geom.theta, kepler.s_release, drop)
        export_check = (x_sim - land(kepler.s_release)) * 1000

    res = {
        "cup_m": cup_m.tolist(), "cup_drop_m": round(drop, 5), "cup_out_m": round(out, 5),
        "rim_diameter_m": rim_diameter, "hit_tolerance_m": round(tol, 5), "theory": chosen,
        "kepler": kepler.to_json(),
        "tied_theories_release_m": {k: None if s is None else round(s, 5) for k, s in tied_marks.items()},
        "default": {"s_release_m": None if s_default is None else round(s_default, 5),
                    "kepler_predicted_miss_m": None if default_miss is None else round(default_miss, 5),
                    "kepler_predicts_hit": None if default_miss is None else bool(abs(default_miss) <= tol)},
        "export_check_mm": None if export_check is None else round(export_check, 2),
        "ramp_length_m": top,
    }
    (exp / "scene_default.xml").write_text(default_xml)
    (exp / "scene_kepler.xml").write_text(kepler_xml)
    (exp / "physics_card.json").write_text(json.dumps(physics_card(doc, fit_r, contact, export_check), indent=1))
    (exp / "plan.json").write_text(json.dumps(res, indent=1))

    if H is not None and tr is not None and Path(tr.clip).exists():
        import cv2
        from .overlay import draw_plan
        from .video import read_frame
        marks = []
        for label, s, color in (("default", s_default, (60, 60, 230)), ("Kepler", kepler.s_release, (80, 220, 80))):
            if s is None:
                continue
            vl = launch_speed(th, fit_r.params, geom.theta, known, s)
            path = np.vstack([geom.launch - s * geom.d, flight_points(geom, vl, drop)])
            marks.append({"label": f"{label}: release {s * 100:.1f} cm up the ramp", "s": s,
                          "path": path, "color": color})
        img = draw_plan(read_frame(tr.clip, frame_index), H, geom, cup_m, rim_diameter, marks, top, kappa)
        cv2.imwrite(str(exp / "plan.png"), img)
        res["image"] = str(exp / "plan.png")
    return res


def physics_card(doc: dict, fit_r: FitResult, contact: dict, export_check: float | None) -> dict:
    th = by_id(fit_r.theory)
    return {
        "object": "ball", "diameter_mm": doc["ball_diameter_mm"],
        "theory": th.id, "summary": th.summary, "law": th.law,
        "parameters": {p.name: {"value": fit_r.params[p.name],
                                "std": None if not np.isfinite(fit_r.std[p.name]) else fit_r.std[p.name],
                                "spread_5_95": fit_r.spread[p.name], "unit": p.unit,
                                "identified": fit_r.identified[p.name]} for p in th.params},
        "measured_on": [c["name"] for c in doc["clips"]],
        "ramp_angles_deg": [c["geometry"]["theta_deg"] for c in doc["clips"]],
        "holdout_rms_mm": fit_r.holdout_rms_mm,
        "mujoco": {"condim": 6, "friction_pair": [contact["slide"], contact["slide"], contact["spin"],
                                                  contact["roll"], contact["roll"]],
                   "inertia_scale": contact["inertia_scale"], "note": contact["note"],
                   "export_check_mm": None if export_check is None else round(export_check, 2)},
    }
