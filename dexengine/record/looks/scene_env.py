"""Build-time scene inputs of a process that draws robot frames: the env of the dumps a record session replays (record
stage-2 render, the camera reference render, the E5 closed-loop sessions). The env cfg reads these before the USD
stage exists, so the one look function (looks.look_d.apply) cannot set them; everything else about the look is
look_d's.

  DEXENGINE_KEY_LIGHT        3200  the distant key light prim of the env cfg (look_d.apply zeroes it with the dome)
  DEXENGINE_CAMERA_FOCAL_MM  12    head camera focal length of a drawing scene (at the hardware 8 mm the head housing
                                   enters the frame bottom; training and judging keep 8 mm and draw nothing)

Who reads it: dexengine.rl.launch.modes.scene_env (the render and replay dump modes of the RL launcher, written only
for a record session). A key light the caller already set (non-empty) is kept; the focal length is always this one.
Standard library only.
"""

from __future__ import annotations

KEY_LIGHT, KEY_LIGHT_VALUE = "DEXENGINE_KEY_LIGHT", "3200"
FOCAL, FOCAL_MM = "DEXENGINE_CAMERA_FOCAL_MM", "12"


def env(environ) -> dict[str, str]:
    """The scene env of a drawing process: the variables whose value differs from `environ`."""
    out = {} if environ.get(KEY_LIGHT) else {KEY_LIGHT: KEY_LIGHT_VALUE}
    if environ.get(FOCAL) != FOCAL_MM:
        out[FOCAL] = FOCAL_MM
    return out
