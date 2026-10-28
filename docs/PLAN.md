# Kepler Prime: plan to the deadline

Submission: **Wed Nov 4, 9:00 PM ET** (DST ends Sun Nov 1, so that's EST). Aim to submit by 6 PM ET.
Needs: public repo, 3-minute video, all team members listed. Only work done during the event counts.

## What Kepler Prime adds

Kepler proves its physics in the real world, on camera:

1. Film the ball on the ramp. Kepler fits the theories; one clip ties, so it asks for a second angle and the winner emerges.
2. Put the cup somewhere new.
3. Plan the release twice: MuJoCo with **default** physics, and with **Kepler's** measured physics.
4. Release from both marks. Default misses; Kepler lands in the cup.

That shows a simulator tuned by Kepler matches reality and a stock one doesn't, which is the sim-to-real problem, with no robot needed.

## Day by day

| Day | Goal | Done when |
|---|---|---|
| **Wed Oct 28** ✅ | Deterministic backbone: phases, theory library, fit harness and verdict, cup planner, MuJoCo A/B, MJCF export, tests | `python -m unittest` passes (14 tests); synthetic ramp-to-cup finds the true release mark within 1 mm |
| **Thu Oct 29** | Build the rig (below). Put the API key in `.env`. `git init` and a public GitHub repo. Film 3 releases at ~15° and 2 at ~25°. Track with SAM, fit. Fix whatever real footage breaks: a hand at release, the rails' end, blur | Gravity within 3% before correction, launch-speed check within 1%, `rolling_groove` (or a clear winner) from the two angles |
| **Fri Oct 30** | First real A/B rehearsal: `kepler plan`, place the cup, release from both marks, film it. Fireworks workshop at 12 PM (Spark Open). Then **Muse Wonder** `watch.py`: SceneSpec (objects, SAM prompts, phases, stated quantities) as validated JSON; feed its SAM prompts into `track`. Voice Transcribe for spoken notes | Kepler's mark lands in the cup at least 4 times out of 5; default misses |
| **Sat Oct 31** | **Theory agent.** Wonder proposes TheorySpecs beyond the library. SymPy compiles their equations into a reference `simulate()`. **Spark Open on Baseten** writes code for what SymPy can't easily do (ramp, then edge, then flight, then bounce, with event detection); the harness checks agent code against the SymPy version on random inputs. Sandbox runner (Modal Sandbox, local subprocess fallback) | An agent-proposed theory goes through fit and verdict like the library ones; agent code that disagrees with its SymPy reference is rejected |
| **Sun Nov 1** | **Next-run design** (`design.py`): separation score over candidate angles and releases, Monte Carlo from the bootstrap draws, phrased by Spark 1.3. **Sim-vs-real metric**: leave one clip out, predict it with MuJoCo default vs Kepler's scene, report the error in cm | A table of default vs Kepler error on at least 3 real clips |
| **Mon Nov 2** | Web app: FastAPI on Modal (upload, SSE progress, results), thin Next.js on Vercel (upload, theory table, overlay, click the cup, plan, export). "Where will it land if...?" parsed by Spark 1.3 | Full flow in the browser on a real clip |
| **Tue Nov 3** | Film the final A/B shots (several takes) and screen recordings. README with results and the prior-art table. Draft the X post. Stretch: sliding box, bank shot | Every shot in the video outline exists |
| **Wed Nov 4** | Edit the 3-minute video, clean the repo, submit by 6 PM ET. Post on X (viral engagement counts until Nov 8) | Submitted |

Cut order if behind: pendulum → voice → bank shot → sliding box → web polish. Never cut the A/B shot or the sim-vs-real table.

## The rig

The A/B only shows something if default physics is really wrong for this ramp. On a flat ramp it's nearly right (under 1 cm off), so **the ball must roll on two rails**:

| Ramp | r_eff / R | Default plan's miss, cup 65 cm below |
|---|---|---|
| Flat board | 1.0 | under 1 cm: no visible difference |
| Rails, contact points 0.66 × ball diameter apart | 0.75 | about 3 cm |
| **Rails, contact points 0.8 × ball diameter apart** | **0.6** | **about 5 cm: a clear miss** |

(From `predict.py`, 15° ramp, 30 cm release, c_rr = 0.01.) On rails the ball touches at a smaller radius, so more of its energy goes into spin. Default physics doesn't know that; Kepler measures it.

- **Ball:** a steel ball, about 22 mm. Measure it and pass `--ball-diameter`.
- **Ramp:** two straight rails (aluminium angle edges, square rods or dowels), contact points about 0.8 × the ball's diameter apart (about 17-18 mm for 22 mm). The ramp's lower end at the table edge.
- **Marks:** a ruler or tape along the rails, every cm from the end. Release marks are measured from there.
- **Cup:** rim 60-75 mm, on the floor, rim about 65 cm below the rails' end, in the plane of motion.
- **A4 sheet:** in the ball's plane. Kepler now corrects a small offset using gravity, but keep it close.
- **Camera:** 240 fps, tripod, 1× lens, square to the motion, bright light. **Don't move it** between clips and the shot.

## Commands

```bash
python -m kepler track clips/a1.mp4 --object "steel ball" -o runs/a1 --g-tol 3
python -m kepler fit runs/a1 runs/a2 runs/b1 --ball-diameter 22 -o exp/rails
python -m kepler plan exp/rails --cup-px 812,640 --rim-diameter 70   # cup rim centre, same camera view
```

`plan` prints both marks, Kepler's chance of a hit, where Kepler predicts the default mark lands, and the export check. It writes `plan.png`, `plan.json`, `scene_default.xml`, `scene_kepler.xml` and `physics_card.json`.

## Design changes folded in

| # | Change | Status | Where |
|---|---|---|---|
| 1 | Prove impact in the real world: default vs Kepler cup shot | ✅ planner built | `experiment.plan`, `sim.py` |
| 1b | Sim-vs-real error on held-out real clips | Nov 1 | `design.py` / report |
| 2 | One ramp angle can't separate rolling from sliding: tie by design, second angle resolves it | ✅ tested | `harness/fit.py`, `tests/test_prime.py` |
| 2b | Stripe on the ball to measure spin directly | stretch | |
| 3 | Rolling resistance inside the inertia factor | ✅ | `theory/library.py` |
| 4 | MJCF: `condim="6"`, rolling friction = c_rr × R, explicit `<pair>` with the ball's automatic contacts off, groove as effective inertia (R/r_eff)² | ✅ MuJoCo matches the theory within about 1 mm | `export/mjcf.py` |
| 5 | SymPy reference for each theory; agent code only for hybrid phases, checked against it | Oct 31 | `theory/`, `agent/` |
| 6 | Spark Open: set `reasoning_effort` low/medium (default xhigh), model `meta-models/muse-spark-open-1026`, stream | Oct 31 | `agent/` |
| 7 | Gravity gate: `--g-tol`; **gravity as the ruler** (rescale so the fall reads 9.81, which removes the sheet-plane error); ball-diameter check | ✅ | `experiment.rescale_by_gravity`, `phases.py` |
| 8 | Prior art: add Swim2Real, Multi-Object System Identification from Videos (ICLR 2026), gradSim. Novelty is "chooses which physics applies, phone only, validated on unseen frames". Next-run design applies model discrimination (Box & Hill, 1967) | Nov 3 | README |
| 9 | Scope: ramp-to-cup first | ✅ | |

Also new: each bootstrap draw redraws every clip's ramp angle from its measurement error (c_rr and r_eff are sensitive to 0.1°); a launch-speed cross-check (ramp fit vs flight fit, two independent measurements); hit probability that includes hand placement (±1 mm) and cup position (±1 mm).

## The 3-minute video

| Time | Shot |
|---|---|
| 0:00 | "Simulators get real objects wrong." Robot sim-to-real failure, one line |
| 0:15 | Film, track, the theory table; one clip ties, Kepler asks for a steeper ramp; the second clip settles it |
| 0:45 | **The A/B: the default mark misses, Kepler's lands in the cup** |
| 1:30 | How: SAM 3.1 tracks, gravity as the ruler, Wonder proposes theories, the Spark Open agent writes code in a sandbox, held-out scoring |
| 2:10 | Sim-vs-real error on 3 objects, default vs Kepler, in cm. The exported MuJoCo file |
| 2:40 | Public link: film anything |
