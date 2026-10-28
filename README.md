# Kepler

Film anything that moves and get a working, verified simulation of it.
Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026). Everything in this repo was written during the event.

This is the **day 1–2 slice**: a phone clip goes in, and the ball's position in metres comes out for every frame, checked against gravity.

```
clip ──▶ probe/prep ──▶ SAM 3.1 tracks the ball ──▶ A4 sheet ▶ homography ──▶ metres ──▶ gravity check (g within 2%)
                                                                                  └──▶ trajectory.json (input to the theory agent)
```

## Setup (macOS)

```bash
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your Meta Model API key into .env
python -m unittest discover -s tests -v   # 14 tests, no API key needed, ~20 s
```

## Day 1: film and track

1. **Film.** 240 fps slow motion, phone on a tripod, main 1× lens, square to the motion. Tape an A4 sheet **in the plane the ball's centre moves in**, fully visible. Use a plain background and bright light. Release the ball from rest.
2. **Check the file.** `python -m kepler probe IMG_0001.MOV` shows the real frame rate. If it says 30 fps for a slow-motion clip, the phone re-timed it: export the original, or add `--capture-fps 240` later.
3. **Convert.** `python -m kepler prep IMG_0001.MOV -o clips/ramp1.mp4` writes an H.264 MP4 with every captured frame kept and phone slow-motion edits dropped.
4. **Track.**
   ```bash
   python -m kepler track clips/ramp1.mp4 --object "steel ball" --still 0,0.2 -o runs/ramp1
   ```
   - `--object` is the SAM 3.1 concept prompt: a short noun phrase, one object.
   - If SAM picks the wrong thing, mark the ball with a box instead: `--box x1,y1,x2,y2 --box-frame 0`.
   - `--still 0,0.2` gives the seconds when the ball sits at rest, for the jitter check.
   - If the sheet isn't found, pass its corners: `--sheet tlx,tly,trx,try,brx,bry,blx,bly`.

## Day 2: the gravity gate

`track` already prints the result:

```
[PASS] g = 9.764 ± 0.009 m/s² (-0.47% vs 9.81) from 63 frames, 0.279-0.537 s, fit RMS 0.19 mm
```

- **FAIL with g far off** (say 0.15): timing is wrong. The message names the true frame rate; rerun with `--capture-fps`.
- **FAIL by a few percent:** check, in order, the sheet's plane, the 1× lens, the phone being square to the motion, and motion blur.
- **Wrong window picked:** set it by hand with `python -m kepler check runs/ramp1/trajectory.json --fall 0.31,0.52`.

Open `runs/ramp1/overlay.mp4` to see the tracked path drawn over the clip (green circle = frames used for the fit).

## Day 3: theories, and the cup (Kepler Prime)

The ball rolls down a straight ramp, leaves its end and flies. `fit` finds the ramp's angle and the launch point, fits every theory in the library to every clip at once (physical parameters shared, release time and point per clip), and scores each on the last 20% of the ramp run, which it never saw.

```bash
python -m kepler fit runs/a1 runs/b1 --ball-diameter 22 -o exp/rails
```

- **Gravity is the ruler.** Each clip is rescaled so its free fall reads 9.81 m/s², which removes the error from a sheet that isn't quite in the ball's plane.
- **Two independent launch speeds** (ramp fit vs flight parabola) should agree within about 1%.
- **One angle is a tie** between rolling, sliding and groove rolling: they all predict constant acceleration there. Film a second angle 10° or more away and the winner emerges.

```bash
python -m kepler plan exp/rails --cup-px 812,640 --rim-diameter 70
```

`plan` gives two release marks for a cup: one from MuJoCo with default physics, one from Kepler's measured physics. It also gives Kepler's chance of a hit, where Kepler says the default mark really lands, and a check that MuJoCo with Kepler's exported scene agrees with Kepler (about 1 mm). Output: `plan.png`, `plan.json`, `scene_default.xml`, `scene_kepler.xml`, `physics_card.json`.

`python -m kepler synth --scene ramp --theta 18 --release 25` renders a test clip with known groove physics. The rig, the day-by-day plan and the demo outline are in [docs/PLAN.md](docs/PLAN.md).

## Outputs (per run)

| File | What it is |
| --- | --- |
| `trajectory.json` | Per-frame `t, x, y` in metres, the sheet homography, the gravity result. The agent reads only this. |
| `positions_px.csv` | Raw per-frame pixel positions and confidence |
| `overlay.mp4` | Debug video |
| `sam_ball_raw.txt`, `sam_sheet_raw.txt` | SAM 3.1's raw output, for debugging |

## Notes

- **Cost:** SAM 3.1 video is $0.20 per 1,000 frames, minimum 50 frames per request. A 4 s clip at 240 fps is about $0.19.
- **Positions** come from SAM's box centres (accurate to about half a pixel for a ball). Mask centroids via Meta's `meta-sam-parser` are the planned upgrade.
- **Offline testing:** `--tracker color --color red` tracks a coloured ball without the API. `python -m kepler synth` renders a test clip with known g.

## Layout

```
kepler/
  video.py       probe, prep, frame reading
  sam.py         SAM 3.1 calls (Responses API, streaming, Files API upload)
  samparse.py    parser for SAM's box/mask token output
  colortrack.py  offline colour tracker (tests, fallback)
  sheet.py       A4 corner detection (sub-pixel edge fit) and homography
  trajectory.py  Sample/Trajectory types, trajectory.json
  gravity.py     free-fall detection and the g check
  overlay.py     debug video, plan picture
  synth.py       synthetic clips with known physics (fall, ramp)
  phases.py      ramp line, angle, launch point, release
  theory/        the standard library of theories
  harness/       fit, held-out scoring, bootstrap, verdict
  predict.py     landing point, release mark, hit probability
  export/mjcf.py MuJoCo scenes: default, or Kepler's measured physics
  sim.py         MuJoCo runs and the default-physics plan
  experiment.py  fit and plan for one experiment
tests/test_kepler.py, tests/test_prime.py
docs/PLAN.md     rig, schedule to Nov 4, demo outline
```
