"""Record session stage ``record_kin`` of the kinematic-data experiments (E5): records the selected trajectories
kinematically into one .npz (--out) without rendering. Enable with
DEXENGINE_RECORD_PLUGINS=experiments.policies.kin_data.plugin."""

from __future__ import annotations

import os

from dexengine.record import hooks
from dexengine.record.session.runner import HOOK as SESSION_HOOK
from dexengine.record.session.runner import set_arg


def _prepare(a, argv: list[str], env: dict):
    from dexengine.record.render import cameras as CC
    from experiments.policies.kin_data import record_kin as CKN

    if not a.out.endswith(".npz"):
        raise SystemExit("record_kin: --out must be the record file (.npz)")
    if not a.seq_list:
        raise SystemExit("record_kin: --seq-list (originals + variants) is required")
    out_dir = os.path.dirname(a.out)
    set_arg(argv, "--videos_out_dir", os.path.join(out_dir, "_videos"))
    env["COLLECT_OUT"] = out_dir
    env[CKN.OUT_ENV] = os.path.abspath(a.out)
    env["DEXENGINE_SEQ_LIST"] = os.path.abspath(a.seq_list)
    print(f"[COLLECT_EXO] {'on' if CC.exo_on() else 'off'} · videos {CC.videos()}", flush=True)
    return CKN.record_kin, "collect_kin.record_kin"


def _log_problems(text: str) -> list[str]:
    from experiments.policies.kin_data import record_kin as CKN

    return CKN.log_problems(text)


hooks.register(
    "session_stage",
    "record_kin",
    hooks.SessionStage(
        name="record_kin",
        prepare=_prepare,
        log_problems=_log_problems,
        timeout_s=1800,
        hook_line=f"{SESSION_HOOK} clip_video._run_per_clip_video_render",
        progress="[TRAJ_REC] ",
    ),
)
