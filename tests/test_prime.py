"""Kepler Prime: ramp -> theories -> verdict -> cup plan, default physics vs Kepler's.

Run with:  python -m unittest discover -s tests -v

Most tests use trajectories made straight from the known physics (fast); one
renders ramp clips and drives the command line end to end.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from kepler import experiment
from kepler.export.mjcf import kepler_contact, scene_xml
from kepler.harness import ClipData, fit_all, verdict
from kepler.phases import segment
from kepler.synth import ramp_path, ramp_trajectory

R = 0.011
KNOWN = {"R": R}
_TMP = Path(tempfile.mkdtemp(prefix="kepler-prime-"))


def _sheet_point(world_xy, truth) -> np.ndarray:
    """A world-plane point in the synthetic sheet's frame, as ramp_trajectory sees it."""
    c, s = np.cos(np.radians(truth["sheet_angle_deg"])), np.sin(np.radians(truth["sheet_angle_deg"]))
    return (np.asarray(world_xy) - truth["sheet_origin_world"]) @ np.array([[c, -s], [s, c]])


def _clips(specs):
    out = []
    for i, (th, s) in enumerate(specs):
        tr, truth = ramp_trajectory(theta_deg=th, s_release=s, seed=i)
        out.append((tr, truth, segment(tr)))
    return out


class GeometryTests(unittest.TestCase):
    def test_segment_finds_ramp_and_launch(self):
        (tr, truth, g), = _clips([(18.0, 0.25)])
        self.assertAlmostEqual(np.degrees(g.theta), 18.0, delta=0.2)
        self.assertLess(np.linalg.norm(g.launch - truth["launch_sheet_m"]) * 1000, 1.0)
        self.assertAlmostEqual(g.t_launch, truth["t_launch"], delta=0.002)
        self.assertAlmostEqual(g.v_launch_flight, truth["v_launch"], delta=0.005 * truth["v_launch"])
        self.assertAlmostEqual(g.ball_diameter_mm, 22.0, delta=0.7)
        # No flight samples leak into the ramp fit.
        self.assertTrue(np.all(tr.t[g.ramp] < truth["t_launch"]))


class TheoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clips = _clips([(18.0, 0.25), (28.0, 0.20)])
        cls.data = [ClipData.from_trajectory(f"c{i}", tr, g) for i, (tr, _, g) in enumerate(cls.clips)]

    def test_one_angle_is_a_tie_and_the_textbook_fails(self):
        res = fit_all(self.data[:1], KNOWN, bootstrap=40)
        v = verdict(res)
        self.assertIsNone(v["winner"])
        for t in ("rolling", "sliding", "rolling_groove"):
            self.assertIn(t, v["tied"])
        self.assertIn("rolling_ideal", v["rejected"])
        groove = next(r for r in res if r.theory == "rolling_groove")
        self.assertFalse(groove.identified["r_eff_ratio"])  # one angle can't separate r_eff from c_rr

    def test_two_angles_pick_the_groove(self):
        res = fit_all(self.data, KNOWN, bootstrap=60)
        v = verdict(res)
        self.assertEqual(v["winner"], "rolling_groove")
        g = next(r for r in res if r.theory == "rolling_groove")
        self.assertTrue(g.identified["r_eff_ratio"])
        self.assertAlmostEqual(g.params["r_eff_ratio"], 0.75, delta=0.03)
        lo, hi = g.spread["c_rr"]
        self.assertTrue(lo - 0.005 <= 0.01 <= hi + 0.005, (lo, hi))


class PlanTests(unittest.TestCase):
    """Fit both angles, then put a cup where the true physics lands a 17 cm release at 28°."""

    @classmethod
    def setUpClass(cls):
        clips = _clips([(18.0, 0.25), (28.0, 0.20)])
        cls.exp = _TMP / "exp"
        experiment.fit([], cls.exp, ball_diameter_mm=22.0, bootstrap=60, trajectories=[c[0] for c in clips])
        _, _, true_shot = ramp_path(np.zeros(1), theta_deg=28.0, s_release=0.17)
        cup = _sheet_point([true_shot["x_land"], true_shot["y_land"]], clips[1][1])
        cls.res = experiment.plan(cls.exp, cup_m=cup, rim_diameter=0.075)

    def test_kepler_finds_the_true_release_mark(self):
        k = self.res["kepler"]
        self.assertEqual(self.res["theory"], "rolling_groove")
        self.assertAlmostEqual(k["s_release_m"], 0.17, delta=0.003)
        self.assertGreater(k["hit_probability"], 0.9)

    def test_default_physics_plans_differently_and_falls_short(self):
        d = self.res["default"]
        self.assertLess(d["s_release_m"], self.res["kepler"]["s_release_m"] - 0.02)  # thinks it rolls faster
        self.assertLess(d["kepler_predicted_miss_m"], -0.015)

    def test_mujoco_export_matches_keplers_prediction(self):
        self.assertLess(abs(self.res["export_check_mm"]), 2.0)
        card = json.loads((self.exp / "physics_card.json").read_text())
        self.assertEqual(card["mujoco"]["condim"], 6)
        self.assertAlmostEqual(card["mujoco"]["friction_pair"][3],
                               card["parameters"]["c_rr"]["value"] * R, places=6)

    def test_scene_xml_uses_an_explicit_pair_only_for_kepler(self):
        kep = scene_xml(0.3, R, 0.4, kepler_contact("rolling", {"c_rr": 0.02}, R))
        self.assertIn('condim="6"', kep)
        self.assertIn('contype="0"', kep)
        self.assertIn(f"{0.02 * R:.7g}", kep)
        self.assertNotIn("<pair", scene_xml(0.3, R, 0.4))


class EndToEndTests(unittest.TestCase):
    def test_render_track_fit_plan(self):
        from kepler.__main__ import main
        from kepler.synth import render_ramp
        runs = []
        for th, s in ((18.0, 0.25), (28.0, 0.20)):
            clip = _TMP / f"ramp{int(th)}.mp4"
            render_ramp(clip, theta_deg=th, s_release=s)
            out = _TMP / f"run{int(th)}"
            with redirect_stdout(io.StringIO()):
                main(["track", str(clip), "--tracker", "color", "--no-overlay", "-o", str(out)])
            runs.append(str(out))
        exp = _TMP / "e2e"
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["fit", *runs, "--ball-diameter", "22", "--bootstrap", "40", "-o", str(exp)])
        self.assertIn("rolling_groove wins", buf.getvalue(), buf.getvalue())
        # Cup at the true landing for a 17 cm release on the 28° ramp, in that clip's pixels.
        from kepler.synth import _projector
        _, _, shot = ramp_path(np.zeros(1), theta_deg=28.0, s_release=0.17)
        px = _projector()(shot["x_land"], shot["y_land"])
        with redirect_stdout(io.StringIO()):
            main(["plan", str(exp), "--cup-px", f"{px[0]:.2f},{px[1]:.2f}"])
        plan = json.loads((exp / "plan.json").read_text())
        self.assertAlmostEqual(plan["kepler"]["s_release_m"], 0.17, delta=0.005)
        for f in ("plan.png", "scene_default.xml", "scene_kepler.xml", "physics_card.json"):
            self.assertTrue((exp / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
