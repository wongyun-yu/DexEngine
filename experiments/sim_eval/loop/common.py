"""Shared constants and helpers of the closed-loop session: marker, frame clock, image size, mode names, recorder
version, hold window."""

from __future__ import annotations

import os
from pathlib import Path

MARK = "[E5_EVAL]"


FPS = 30


POLICY_IMG_HW = (180, 320)


def _mkdir(p) -> Path:
    """Create a session output folder (parents too) and return its real path."""
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(os.path.realpath(p))


def _say(msg: str) -> None:
    print(f"{MARK} {msg}", flush=True)


def frame_rows(n_rows: int, step_dt: float, fps: int = FPS) -> list[int]:
    """Video frame j → state row round(j / fps / step_dt), within n_rows (the record frame_steps rule)."""
    from dexengine.record.render import episode as CR

    return CR.frame_steps(n_rows, step_dt, fps)


def box_down(img, hw=POLICY_IMG_HW):
    """1280×720 → 320×180 box mean (integer factor 4 — the box of ffmpeg scale flags=area; no colour-space round
    trip).
    """
    import numpy as np

    h, w = img.shape[:2]
    fy, fx = h // hw[0], w // hw[1]
    if fy * hw[0] != h or fx * hw[1] != w:
        raise SystemExit(f"{MARK} image {w}×{h} is not an integer multiple of {hw[1]}×{hw[0]}")
    x = img[: hw[0] * fy, : hw[1] * fx].reshape(hw[0], fy, hw[1], fx, -1).astype(np.float32).mean(axis=(1, 3))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


ACT_REPLAY = (
    "act_replay",
    "act_replay0",
    # action replay: the stage-1 recorded action column → canonical action term · act_replay0 = action 0 (negative
    # test)
)


#: Strict hold window on the data: write the last stage-1 recorded row
#: (position + the last two rows' finite-difference velocity), keep the data's last joint target and simulate
#: hold_rows() rows (HOLD_WINDOW_S) — the same _hold as the policy rollout.
DATA_HOLD = "data_hold"


PD_REPLAY = (
    "pd_replay"  # plain PD replay: only the per-physics-step joint targets recorded by an action replay (rec_sub=1)
)


CTRLS = (
    "hold",
    "interp",
    # evaluation controller: hold = the row target for every substep · interp = linear from the previous to this target
    # over the substeps
)


#: Episode recorder version — 2 = meta.held names entities (the objects of force rows hand_obj_force[:, k]), held_sides
#: names sides. The batch runner does not count a record below this version as done.
RECORDER = 2


#: Hold window: after the last action, simulate HOLD_WINDOW_S with the last action kept. Success = within this window
#: the object stays within ε and still for the hold length (judge rule put.hold_s = 0.5 s) — settling time
#: = HOLD_WINDOW_S − put.hold_s (1.0 s). Data (data_hold), policy and replay use the same window. This constant is the
#: one owner.
HOLD_WINDOW_S = 1.5


ROW_HZ = 50  # record row period (step_dt 0.02 — checked by _closed)


def hold_rows() -> int:
    """Hold-window rows = round(HOLD_WINDOW_S × ROW_HZ) = 75."""
    return int(round(HOLD_WINDOW_S * ROW_HZ))
