"""Evaluation options of one closed-loop session: the "eval" dict of job_run.json (`$E5_JOB`, written by run_scene).

Values are strings, as the former environment flags were: "1" switches a flag on, a missing key means off/default.
Keys (all optional unless noted): profile ("official", required by the closed loop), spawn_f0, extra_f0,
policy_until_t, policy_job_text, policy_oracle_keys, policy_img_codec ("dataset"), place_marker (e.g. "pad-red"),
start_override (SPIDER start JSON), obj_color, obj_keep, arm_lead ("self"), physics_dr ("on"), epstate ("off"),
warm_counter, episode_seed, rec_obj_pair, rec_obj_desk, rec_sub, ep_no_frames, pd_src_batch, pd_src_root, pd_hold.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MARK = "[E5_EVAL]"
_CFG: dict | None = None


def value(name: str, default=None):
    """Option `name` (None/`default` when absent)."""
    global _CFG
    if _CFG is None:
        _CFG = dict(json.loads(Path(os.environ["E5_JOB"]).read_text()).get("eval") or {})
    v = _CFG.get(name)
    return default if v is None else v


def flag01(name: str) -> bool:
    """A 0/1 option (absent = 0); any other value dies."""
    v = str(value(name, "0"))
    if v not in ("0", "1"):
        raise SystemExit(f"{MARK} {name}={v!r} — 0 or 1")
    return v == "1"
