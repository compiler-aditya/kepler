"""MuJoCo scenes of the ramp: out-of-the-box physics, or the physics Kepler measured.

World frame: x points the way the ball travels (horizontal), z points up, and the
origin is the ball's centre at the moment it leaves the ramp.

"Default" is what a robotics team gets without measuring anything: MuJoCo's
default contact (condim 3, friction 1 / 0.005 / 0.0001) and a solid sphere's
inertia. That's rolling without slipping and no losses, the textbook case.

Three ways a naive export silently gives the wrong simulation, all handled here:

1. Rolling friction only acts with condim="6". The default, 3, ignores it.
2. MuJoCo's rolling-friction coefficient is a length (torque = coefficient x
   normal force). Kepler's c_rr is dimensionless (force / normal force), so it
   goes in as c_rr * R.
3. Two touching geoms combine friction by taking the larger value, so setting it
   on the ball alone does nothing if the ramp's is higher. Kepler writes an
   explicit <pair>, which sets the contact's parameters directly, and turns off
   the ball's automatic contacts so the pair is the only contact.

A groove can't be drawn on a flat ramp, but its effect can: on a flat surface,
inertia I_eff = (2/5) m R^2 (R / r_eff)^2 gives exactly the groove's acceleration
g (sin θ - c_rr cos θ) / (1 + (2/5) R^2 / r_eff^2).
"""

from __future__ import annotations

import numpy as np

BALL_MASS = 0.044  # kg, a 22 mm steel ball; none of the theories depend on it


def kepler_contact(theory: str, params: dict, R: float) -> dict:
    """The MuJoCo settings that reproduce a fitted theory, with the reasoning written down."""
    if theory == "sliding":
        mu = params["mu"]
        # Rolling friction above mu * R stops the ball from turning, so it slides like a block.
        # MuJoCo's soft contacts still leak some rolling at 2 mu R (9 mm long at 27 cm); 10 is enough.
        return {"slide": mu, "spin": 0.005, "roll": 10 * mu * R, "inertia_scale": 1.0,
                "note": "sliding: rolling suppressed with roll = 10 mu R"}
    ratio = params.get("r_eff_ratio", 1.0)
    c_rr = params.get("c_rr", 0.0)
    return {"slide": 1.0, "spin": 0.005, "roll": c_rr * R, "inertia_scale": 1 / ratio ** 2,
            "note": f"roll = c_rr x R = {c_rr:.4f} x {R * 1000:.1f} mm; "
                    f"inertia x (R/r_eff)^2 = x{1 / ratio ** 2:.3f} for the groove"}


def scene_xml(theta: float, R: float, ramp_length: float, contact: dict | None = None,
              timestep: float = 0.0005, mass: float = BALL_MASS) -> str:
    """contact=None: MuJoCo defaults. Otherwise the dict from kepler_contact()."""
    th = float(theta)
    d = np.array([np.cos(th), 0.0, -np.sin(th)])  # down the ramp
    n_up = np.array([np.sin(th), 0.0, np.cos(th)])  # the ramp surface's normal
    half_t = 0.005
    edge = -R * n_up  # contact point at the moment of launch
    centre = edge - 0.5 * ramp_length * d - half_t * n_up
    f = lambda v: " ".join(f"{x:.7g}" for x in v)
    i_solid = 0.4 * mass * R * R
    if contact is None:
        ball = f'<geom name="ball" type="sphere" size="{R:.7g}" mass="{mass:.7g}" rgba="0.8 0.2 0.2 1"/>'
        pair = ""
        option = f'<option timestep="{timestep}"/>'
    else:
        I = i_solid * contact["inertia_scale"]
        ball = (f'<inertial pos="0 0 0" mass="{mass:.7g}" diaginertia="{I:.7g} {I:.7g} {I:.7g}"/>'
                f'<geom name="ball" type="sphere" size="{R:.7g}" contype="0" conaffinity="0" '
                f'rgba="0.2 0.7 0.3 1"/>')
        fr = f'{contact["slide"]:.7g} {contact["slide"]:.7g} {contact["spin"]:.7g} {contact["roll"]:.7g} {contact["roll"]:.7g}'
        pair = (f'<contact><pair geom1="ball" geom2="ramp" condim="6" friction="{fr}"/></contact>'
                f'<!-- {contact["note"]} -->')
        option = f'<option timestep="{timestep}" cone="elliptic" noslip_iterations="20"/>'
    return f"""<mujoco model="kepler_ramp">
  <compiler angle="radian"/>
  {option}
  <worldbody>
    <light pos="0 -1 1" dir="0 1 -1"/>
    <geom name="ramp" type="box" size="{0.5 * ramp_length:.7g} 0.03 {half_t}" pos="{f(centre)}"
          euler="0 {th:.7g} 0" rgba="0.55 0.4 0.25 1"/>
    <body name="ball_body" pos="0 0 0">
      <freejoint name="ball_free"/>
      {ball}
    </body>
  </worldbody>
  {pair}
</mujoco>
"""
