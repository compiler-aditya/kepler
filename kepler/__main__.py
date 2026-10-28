"""Kepler command line.

    python -m kepler probe CLIP
    python -m kepler prep CLIP -o clip.mp4
    python -m kepler track clip.mp4 --object "steel ball" -o runs/ramp1
    python -m kepler check runs/ramp1/trajectory.json [--fall 0.31,0.52]
    python -m kepler fit runs/ramp1 runs/ramp2 --ball-diameter 22 -o exp/ramp
    python -m kepler plan exp/ramp --cup-px 812,640 --rim-diameter 75
    python -m kepler synth -o synth/fall.mp4 [--scene ramp]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_env() -> None:
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _floats(s: str, n: int) -> list[float]:
    vals = [float(v) for v in s.split(",")]
    if len(vals) != n:
        raise SystemExit(f"expected {n} comma-separated numbers, got {s!r}")
    return vals


def cmd_probe(a) -> None:
    from .video import probe
    print(probe(a.clip).summary())


def cmd_prep(a) -> None:
    from .video import prep, probe
    out = prep(a.clip, a.out)
    print(f"wrote {out}")
    print(probe(out).summary())


def cmd_synth(a) -> None:
    from .synth import render, render_ramp
    if a.scene == "ramp":
        meta = render_ramp(Path(a.out), fps=a.fps, theta_deg=a.theta, s_release=a.release / 100,
                           export_fps=a.export_fps)
        p = meta["physics"]
        print(f"wrote {a.out}: {meta['frames']} frames at {meta['capture_fps']} fps; ramp {p['theta_deg']}°, "
              f"released {p['s_release'] * 100:.1f} cm up, groove r_eff = {p['r_eff_ratio']} R, "
              f"c_rr = {p['c_rr']}, a = {p['accel']:.4f} m/s²")
        return
    meta = render(Path(a.out), fps=a.fps, export_fps=a.export_fps)
    print(f"wrote {a.out}: {meta['frames']} frames captured at {meta['capture_fps']} fps, "
          f"file says {meta['file_fps']} fps, true g = {meta['g']}")


def _fmt_param(r: dict, name: str, unit: str) -> str:
    v = r["params"][name]
    if not r["identified"][name]:
        return f"{name} = {v:.4g} (not pinned down)"
    lo, hi = r["spread_5_95"][name]
    return f"{name} = {v:.4g} [{lo:.4g}, {hi:.4g}]{(' ' + unit) if unit else ''}"


def cmd_fit(a) -> None:
    from . import experiment
    from .theory import by_id
    doc = experiment.fit([Path(r) for r in a.runs], Path(a.out), ball_diameter_mm=a.ball_diameter,
                         holdout=a.holdout, bootstrap=a.bootstrap)
    for c in doc["clips"]:
        g, ck = c["geometry"], c["launch_speed_check"]
        print(f"{c['name']}: ramp {g['theta_deg']:.2f}° ± {g['theta_sd_deg']:.2f}° below horizontal, "
              f"{g['ramp_samples'][1] - g['ramp_samples'][0] + 1} ramp frames")
        if c["scale_from_gravity"] != 1.0:
            print(f"  scale from gravity: x{c['scale_from_gravity']:.4f} (the free fall read {c['g_measured']:.3f} m/s², "
                  f"so the ball's plane is {(c['scale_from_gravity'] - 1) * 100:+.2f}% from the sheet's)")
        print(f"  launch speed {ck['v_launch_ramp_fit']:.3f} m/s from the ramp fit vs "
              f"{ck['v_launch_flight_fit']:.3f} from the flight ({ck['diff_pct']:+.2f}%): two independent measurements")
    if doc["ball_diameter_measured_mm"] and a.ball_diameter:
        dm = doc["ball_diameter_measured_mm"]
        pct = (dm / a.ball_diameter - 1) * 100
        # A coarse check: blur and edge pixels make tracked outlines run a few percent large,
        # but a wrong lens or a sheet far off the ball's plane shows up as 20% or more.
        mark = "PASS" if abs(pct) <= 10 else "CHECK"
        print(f"[{mark}] ball diameter on the frames at rest: {dm:.2f} mm vs {a.ball_diameter:.2f} mm stated "
              f"({pct:+.1f}%): a scale check that doesn't depend on the A4 sheet or on gravity")
    print(f"\n{'theory':16s}{'train':>8s}{'held-out':>10s}{'± spread':>10s}{'BIC':>9s}  parameters")
    for r in sorted(doc["results"], key=lambda r: r["holdout_rms_mm"]):
        th = by_id(r["theory"])
        ps = "; ".join(_fmt_param(r, p.name, p.unit) for p in th.params) or "-"
        print(f"{r['theory']:16s}{r['train_rms_mm']:7.2f} {r['holdout_rms_mm']:8.2f}mm {r['holdout_spread_mm']:8.2f}"
              f"{r['bic']:10.0f}  {ps}")
    v = doc["verdict"]
    if v["winner"]:
        print(f"\nverdict: {v['winner']} wins. Rejected: {', '.join(v['rejected']) or 'none'}.")
    else:
        print(f"\nverdict: tie between {', '.join(v['tied'])}: they predict the same motion on "
              f"{'this ramp' if len(doc['clips']) == 1 else 'these ramps'}. Rejected: {', '.join(v['rejected']) or 'none'}."
              f"\n  To tell them apart, film one more run with the ramp at a clearly different angle "
              f"(10° or more steeper or shallower).")
    print(f"wrote {Path(a.out) / 'fit.json'}")


def cmd_plan(a) -> None:
    from . import experiment
    res = experiment.plan(Path(a.exp), cup_m=np.array(_floats(a.cup_m, 2)) if a.cup_m else None,
                          cup_px=tuple(_floats(a.cup_px, 2)) if a.cup_px else None,
                          rim_diameter=a.rim_diameter / 1000, clip_index=a.clip,
                          ramp_length=a.ramp_length / 100 if a.ramp_length else None, theory=a.theory,
                          frame_index=a.frame, place_sd=a.place_sd / 1000)
    k, d = res["kepler"], res["default"]
    print(f"cup rim: {res['cup_out_m'] * 100:.1f} cm out and {res['cup_drop_m'] * 100:.1f} cm below the ramp's end; "
          f"the ball goes in within ±{res['hit_tolerance_m'] * 100:.1f} cm")
    if d["s_release_m"] is None:
        print("MuJoCo default physics: the cup is out of reach on this ramp")
    else:
        verdict = "lands in the cup" if d["kepler_predicts_hit"] else "misses"
        print(f"MuJoCo default physics: release {d['s_release_m'] * 100:.1f} cm up the ramp. Kepler predicts that "
              f"{verdict}: {d['kepler_predicted_miss_m'] * 100:+.1f} cm from the rim's centre")
    if k["s_release_m"] is None:
        print(f"Kepler ({res['theory']}): {k['note']}")
    else:
        lo, hi = k["s_interval_m"]
        print(f"Kepler ({res['theory']}): release {k['s_release_m'] * 100:.1f} cm up the ramp "
              f"(90%: {lo * 100:.1f}-{hi * 100:.1f}), chance of landing in the cup {k['hit_probability'] * 100:.0f}%"
              f"{'  [' + k['note'] + ']' if k['note'] else ''}")
        if res["tied_theories_release_m"]:
            others = ", ".join(f"{t} {s * 100:.1f} cm" for t, s in res["tied_theories_release_m"].items() if s)
            print(f"  tied theories put the mark at: {others}")
        print(f"  export check: MuJoCo with Kepler's scene lands {res['export_check_mm']:+.1f} mm from Kepler's own prediction")
    print(f"wrote plan.json, scene_default.xml, scene_kepler.xml, physics_card.json"
          f"{', plan.png' if res.get('image') else ''} in {a.exp}")


def cmd_track(a) -> None:
    from . import gravity, overlay, sheet
    from .trajectory import Trajectory, samples_from_sam, write_csv
    from .video import probe, read_frame

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = probe(a.clip)
    print(info.summary())
    fps = a.capture_fps or info.stream_fps
    if not fps:
        raise SystemExit("could not read the frame rate; pass --capture-fps")

    # 1. Track the ball.
    if a.tracker == "sam":
        from . import sam
        from .samparse import by_frame
        if a.box:
            x1, y1, x2, y2 = (int(v) for v in _floats(a.box, 4))
            recs = sam.track_video_box(a.clip, (x1, y1, x2, y2), (info.width, info.height),
                                       a.box_frame, raw_log=out / "sam_ball_raw.txt")
            per_frame = by_frame(recs, object_id=1)
        else:
            recs = sam.track_video(a.clip, a.object, raw_log=out / "sam_ball_raw.txt")
            per_frame = by_frame(recs)
        samples = samples_from_sam(per_frame)
        target = a.object if not a.box else f"box {a.box} on frame {a.box_frame}"
    else:
        from .colortrack import track_color
        samples = track_color(a.clip, a.color)
        target = f"colour {a.color}"
    if not samples:
        raise SystemExit("the ball was not found on any frame")
    write_csv(samples, out / "positions_px.csv")
    found = len(samples) / max(1, info.n_frames) * 100
    print(f"tracked on {len(samples)} of {info.n_frames} frames ({found:.1f}%)")

    # Jitter while the ball sits still before release: needs --still START,END (s).
    if a.still:
        s0, s1 = _floats(a.still, 2)
        sel = [s for s in samples if s0 <= s.frame / fps <= s1]
        if len(sel) >= 5:
            xs = np.array([s.x_px for s in sel]); ys = np.array([s.y_px for s in sel])
            jit = float(np.sqrt(np.mean((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)))
            print(f"{'[PASS]' if jit <= 1 else '[FAIL]'} jitter at rest: {jit:.2f} px RMS (target <= 1 px)")

    # 2. Find the A4 sheet and build the homography.
    frame0 = read_frame(a.clip, a.sheet_frame)
    if a.sheet and a.sheet not in ("auto", "sam"):
        corners = np.array(_floats(a.sheet, 8)).reshape(4, 2)
    else:
        roi = None
        if a.sheet == "sam" or (a.sheet == "auto" and a.tracker == "sam"):
            from . import sam
            srec = sam.segment_image(frame0, a.sheet_prompt, raw_log=out / "sam_sheet_raw.txt")
            if srec:
                b = max(srec, key=lambda r: r.box.width * r.box.height).box
                roi = (b.x1, b.y1, b.x2, b.y2)
            else:
                print("SAM found no sheet; falling back to brightness detection on the whole frame")
        corners = sheet.detect_corners(frame0, roi)
    fit = sheet.fit(frame0, corners)
    print(f"A4 sheet: corners {fit.corners_px.round(1).tolist()}, edge straightness "
          f"{fit.edge_rms_mm:.2f} mm RMS (target <= 1 mm)")

    # 3. Pixels -> metres.
    xs_px = np.array([s.x_px for s in samples]); ys_px = np.array([s.y_px for s in samples])
    x_mm, y_mm = sheet.px_to_mm(fit.H, xs_px, ys_px)
    frames = np.array([s.frame for s in samples])
    t = frames / fps
    conf = np.array([np.nan if s.confidence is None else s.confidence for s in samples], float)
    size_m = np.array([s.size_px for s in samples]) * sheet.mm_per_px(fit.H, xs_px, ys_px) / 1000
    traj = Trajectory(clip=str(a.clip), capture_fps=fps, frame_size=(info.width, info.height),
                      tracker=a.tracker, target=target, sheet=fit.to_json(), frames=frames, t=t,
                      x=x_mm / 1000, y=y_mm / 1000, confidence=conf, size=size_m)

    # 4. The gravity gate.
    try:
        res = gravity.check(t, traj.x, traj.y, fps, _floats(a.fall, 2) if a.fall else None, a.g_tol)
        traj.gravity = res
        print(gravity.describe(res))
        fall_frames = (int(round(res["window_s"][0] * fps)), int(round(res["window_s"][1] * fps)))
    except ValueError as e:
        print(f"[SKIP] gravity check: {e}")
        fall_frames = None
    traj.save(out / "trajectory.json")
    print(f"wrote {out / 'trajectory.json'}")

    # 5. Debug overlay.
    if not a.no_overlay:
        label = f"g={traj.gravity['g_measured']:.2f}" if traj.gravity else ""
        path = overlay.render(a.clip, samples, out / "overlay.mp4", corners_px=fit.corners_px,
                              fall_frames=fall_frames, label=label)
        print(f"wrote {path}")


def cmd_check(a) -> None:
    from . import gravity
    from .trajectory import Trajectory
    tr = Trajectory.load(Path(a.trajectory))
    fps = a.capture_fps or tr.capture_fps
    t = tr.frames / fps
    res = gravity.check(t, tr.x, tr.y, fps, _floats(a.fall, 2) if a.fall else None, a.g_tol)
    print(gravity.describe(res))
    if a.save:
        tr.gravity = res
        tr.save(Path(a.trajectory))
        print(f"saved the new gravity check (and its fall window) into {a.trajectory}")
    if a.json:
        print(json.dumps(res, indent=1))


def main(argv=None) -> None:
    _load_env()
    p = argparse.ArgumentParser(prog="kepler", description="Kepler: phone video to verified physics")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("probe", help="show a clip's real frame rate, size and slow-motion warnings")
    s.add_argument("clip"); s.set_defaults(fn=cmd_probe)

    s = sp.add_parser("prep", help="convert to H.264 MP4 (faststart), keeping every captured frame")
    s.add_argument("clip"); s.add_argument("-o", "--out", required=True); s.set_defaults(fn=cmd_prep)

    s = sp.add_parser("track", help="track the ball, convert to metres, run the gravity check")
    s.add_argument("clip")
    s.add_argument("-o", "--out", default="runs/latest")
    s.add_argument("--tracker", choices=["sam", "color"], default="sam")
    s.add_argument("--object", default="steel ball", help="SAM concept prompt for the ball")
    s.add_argument("--box", help="x1,y1,x2,y2 box around the ball (SAM box prompt instead of a phrase)")
    s.add_argument("--box-frame", type=int, default=0)
    s.add_argument("--color", default="red", help="colour tracker: red/orange/yellow/green/blue or #rrggbb")
    s.add_argument("--sheet", default="auto", help="auto | sam | 8 numbers: tl,tr,br,bl corners in pixels")
    s.add_argument("--sheet-prompt", default="white paper sheet")
    s.add_argument("--sheet-frame", type=int, default=0)
    s.add_argument("--capture-fps", type=float, help="true capture rate if the file is re-timed (e.g. 240)")
    s.add_argument("--fall", help="START,END seconds of the free fall, if auto-detection picks wrong")
    s.add_argument("--still", help="START,END seconds while the ball is at rest, for the jitter check")
    s.add_argument("--g-tol", type=float, default=2.0, help="gravity gate limit, percent (real clips: try 3)")
    s.add_argument("--no-overlay", action="store_true")
    s.set_defaults(fn=cmd_track)

    s = sp.add_parser("check", help="re-run the gravity check on a saved trajectory.json")
    s.add_argument("trajectory"); s.add_argument("--fall"); s.add_argument("--capture-fps", type=float)
    s.add_argument("--g-tol", type=float, default=2.0)
    s.add_argument("--save", action="store_true", help="write the result back into trajectory.json")
    s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_check)

    s = sp.add_parser("fit", help="fit every theory to one or more ramp clips and pick a winner")
    s.add_argument("runs", nargs="+", help="run directories from `kepler track` (or trajectory.json files)")
    s.add_argument("-o", "--out", default="exp/latest")
    s.add_argument("--ball-diameter", type=float, help="mm; also checks the scale against the ball's size")
    s.add_argument("--holdout", type=float, default=0.2, help="fraction of each ramp run held out (the end)")
    s.add_argument("--bootstrap", type=int, default=200)
    s.set_defaults(fn=cmd_fit)

    s = sp.add_parser("plan", help="release marks that land the ball in a cup: default physics vs Kepler")
    s.add_argument("exp", help="experiment directory from `kepler fit`")
    s.add_argument("--cup-px", help="X,Y pixel of the cup rim's centre, in the same camera view as the clips")
    s.add_argument("--cup-m", help="X,Y of the cup rim's centre in sheet metres")
    s.add_argument("--rim-diameter", type=float, default=75.0, help="inner rim diameter, mm")
    s.add_argument("--clip", type=int, default=-1, help="which clip's ramp setup you'll shoot with (default: last)")
    s.add_argument("--frame", type=int, default=0, help="frame of that clip to draw plan.png on")
    s.add_argument("--ramp-length", type=float, help="cm from the ramp's end to its top (limits the marks)")
    s.add_argument("--theory", help="plan with this theory instead of the verdict's")
    s.add_argument("--place-sd", type=float, default=1.0, help="how precisely you set the ball on a mark, mm")
    s.set_defaults(fn=cmd_plan)

    s = sp.add_parser("synth", help="render a synthetic test clip with known physics")
    s.add_argument("-o", "--out", default="synth/fall.mp4")
    s.add_argument("--scene", choices=["fall", "ramp"], default="fall")
    s.add_argument("--theta", type=float, default=18.0, help="ramp scene: angle, degrees")
    s.add_argument("--release", type=float, default=25.0, help="ramp scene: cm up the ramp")
    s.add_argument("--fps", type=float, default=240.0)
    s.add_argument("--export-fps", type=float, help="write the file at this rate (mimics a re-timed export)")
    s.set_defaults(fn=cmd_synth)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
