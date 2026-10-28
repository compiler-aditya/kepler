"""Run with:  python -m unittest discover -s tests -v

No API key needed: the SAM path is tested with a simulated SAM 3.1 response
built from the synthetic clip, in the documented wire format.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from kepler import samparse

DOC_VIDEO_LINE = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,60,"
    "~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>,1<|box;x1=155;y1=228;x2=202;"
    "y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>"
)
DOC_CONF_LINE = (
    "<0f>1<|box;x1=444;y1=625;x2=679;y2=850;w=1800;h=1200;c=0.9609375|>"
    "<|mask;x=0;y=0;c=0.9609375;data=226,236,!ab|c|>de|>"
)

_TMP = Path(tempfile.mkdtemp(prefix="kepler-test-"))
_CLIP = _TMP / "fall.mp4"


def setUpModule():
    from kepler.synth import render
    render(_CLIP)


class ParseTests(unittest.TestCase):
    def test_documented_video_line(self):
        recs = samparse.parse_output(DOC_VIDEO_LINE)
        self.assertEqual([r.object_id for r in recs], [0, 1])
        self.assertEqual((recs[0].box.cx, recs[0].box.cy), (240.5, 241.0))
        self.assertEqual((recs[0].mask_rows, recs[0].mask_cols), (27, 60))
        self.assertEqual(recs[0].mask_encoding, "lossless")

    def test_confidence_and_payload_with_pipes(self):
        (r,) = samparse.parse_output(DOC_CONF_LINE)
        self.assertAlmostEqual(r.confidence, 0.9609375)
        self.assertEqual(r.mask_encoding, "one_bit")
        self.assertEqual(r.mask_payload, "ab|c|>de")

    def test_frames_out_of_order_and_sparse(self):
        text = DOC_CONF_LINE.replace("<0f>", "<7f>") + "\n" + DOC_CONF_LINE.replace("<0f>", "<2f>")
        frames = sorted(samparse.by_frame(samparse.parse_output(text)))
        self.assertEqual(frames, [2, 7])


def _track(*args) -> tuple[dict, str]:
    from kepler.__main__ import main
    out = _TMP / ("run-" + "-".join(a.strip("-") for a in args[:2]))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["track", str(_CLIP), "--no-overlay", "-o", str(out), *args])
    return json.loads((out / "trajectory.json").read_text()), buf.getvalue()


class PipelineTests(unittest.TestCase):
    def test_color_tracker_passes_gravity_gate(self):
        traj, log = _track("--tracker", "color")
        g = traj["gravity_check"]
        self.assertTrue(g["passed"], log)
        self.assertLess(abs(g["error_pct"]), 2.0)
        self.assertLess(traj["sheet"]["edge_rms_mm"], 1.0)

    def test_sam_path_with_simulated_response(self):
        """Feed the pipeline SAM-format text built from the colour tracker's boxes."""
        from kepler.colortrack import track_color
        samples = track_color(_CLIP, "red")
        lines = []
        for s in samples:
            r = s.size_px / 2
            x1, y1 = round(s.x_px - r), round(s.y_px - r)
            x2, y2 = round(s.x_px + r), round(s.y_px + r)
            lines.append(f"<{s.frame}f>0<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=1280;h=720;c=0.97|>"
                         f"<|mask;x=0;y=0;c=0.97;data={y2 - y1 + 1},{x2 - x1 + 1},!xyz|>")
        ball_recs = samparse.parse_output("\n".join(lines))
        sheet_recs = samparse.parse_output(
            "<0f>0<|box;x1=560;y1=185;x2=968;y2=400;w=1280;h=720|><|mask;x=0;y=0;data=216,409,!x|>")
        with mock.patch("kepler.sam.track_video", return_value=ball_recs), \
             mock.patch("kepler.sam.segment_image", return_value=sheet_recs):
            traj, log = _track("--tracker", "sam", "--object", "red ball")
        g = traj["gravity_check"]
        # Box centres are whole-pixel, so allow a little more noise than mask centroids.
        self.assertTrue(g["passed"], log)
        self.assertEqual(traj["tracker"], "sam")

    def test_retimed_export_is_flagged(self):
        from kepler.gravity import check
        from kepler.trajectory import Trajectory
        traj, _ = _track("--tracker", "color")
        path = _TMP / "t.json"
        path.write_text(json.dumps(traj))
        tr = Trajectory.load(path)
        res = check(tr.frames / 30.0, tr.x, tr.y, 30.0)
        self.assertFalse(res["passed"])
        self.assertIn("--capture-fps 240", res["hint"])


if __name__ == "__main__":
    unittest.main()
