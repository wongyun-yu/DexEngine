"""Hyperextension clip table — the window that calls the IK clipping rule as is from outside the IK model (the
training-asset generator, the reference finger ranges).

The rule is not restated here: which joints are clipped in which order is `prior.hyperext_clip_targets`, how they are
clipped is `prior.hyperext_clip_range`, the chains (joint names, sign, neutral, clip_roles) are `prior.build_prior_conf`
and the scene's side set is `prior.emb_sides` (the hand tracker imports the same functions). This file only exports
those four as JSON. It does not import the solver — the bake and judge interpreter (the Isaac venv on the execution
machine) has no solver runtime (loguru, tyro, viewer). No graphics either (callers run it with `MUJOCO_GL=disable`).

Invocation (from the repository root, the same resolution the stage-1 batch uses for the hand tracker)::

    <python> -m dexengine.retarget.ik.hyperext_limits <spec dir> <emb> < ranges.json

Input (stdin): ``{"<joint>": [lo, hi], …}`` — the ranges the asset (URDF) has, rad. A joint to clip that is missing from
the input is an error.
Output (stdout, one JSON): ``rule_root`` · ``rule_file`` (path of the rule module `prior.py` — the caller checks the
tree) · ``prior``/``prior_sha256`` · ``meta`` · ``marker`` (= `prior_marker`) · ``rows`` =
``[{joint, side, sign, neutral, lo, hi, new_lo, new_hi, rig_lo, rig_hi, rig_new_lo, rig_new_hi}]``
(order = `hyperext_clip_targets`). ``side`` is the side the joint belongs to — read from each side's target list of
`build_prior_conf(…, (side,))`. ``rig_*`` is the same rule applied to the ranges of the IK rig xml
(`<spec dir>/<emb>.xml`) — the caller checks that the asset and the IK model end with the same limits.
A hand whose prior is `enabled: false` or has `limit_hyperextension` off is not clipped by the IK, so there is nothing
to clip — an error. The third argument `--off-ok` returns that case as a value instead (empty ``rows`` + ``off`` =
why, rc 0); the range window of the reference repair (`wrist_shift.finger_ranges`) uses it ("an unclipped hand keeps the
asset ranges").
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import mujoco

from dexengine.core.paths import hand_rig_dir
from dexengine.retarget.ik import prior


def _rig_ranges(spec: str, emb: str) -> dict:
    m = mujoco.MjModel.from_xml_path(str(hand_rig_dir(spec) / f"{emb}.xml"))
    return {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): (float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1]))
        for j in range(m.njnt)
        if m.jnt_limited[j]
    }


def _side_of(path: str, spec: str, emb: str) -> dict:
    """Joint → side. The chains are built per side (`build_prior_conf(…, (side,))`) and that side's clip targets
    collected — target selection is again `hyperext_clip_targets` alone. A joint on both sides is an error."""
    out = {}
    for side in prior.emb_sides(emb):
        conf_s, _ = prior.build_prior_conf(path, spec, (side,))
        for joint, _, _ in prior.hyperext_clip_targets(conf_s):
            if joint in out:
                raise SystemExit(f"[hyperext_limits] {spec}: {joint!r} is on both sides ({out[joint]}, {side})")
            out[joint] = side
    return out


def table(spec: str, emb: str, ranges: dict, off_ok: bool = False) -> dict:
    path = str(hand_rig_dir(spec) / "retarget_prior.yaml")
    conf, meta = prior.build_prior_conf(path, spec, prior.emb_sides(emb))
    if conf is None or not conf["limit_hyperextension"]:
        why = "enabled: false" if conf is None else "limit_hyperextension: false"
        if off_ok:
            return {"meta": meta, "marker": prior.prior_marker(meta), "rows": [], "off": why}
        raise SystemExit(
            f"[hyperext_limits] {spec}/{emb}: the prior says {why} — the IK does not clip "
            "hyperextension, so there is no asset limit to clip (a switch without a consumer)"
        )
    rig, side_of = _rig_ranges(spec, emb), _side_of(path, spec, emb)
    rows = []
    for joint, sign, neutral in prior.hyperext_clip_targets(conf):
        for have, what in ((ranges, "the asset range input"), (rig, "the IK rig"), (side_of, "the per-side targets")):
            if joint not in have:
                raise SystemExit(
                    f"[hyperext_limits] {spec}/{emb}: clip target {joint!r} is not in {what} — "
                    "the joint names diverged (skipping silently would apply only half of the clip)"
                )
        lo, hi = (float(v) for v in ranges[joint])
        new_lo, new_hi = prior.hyperext_clip_range(lo, hi, sign, neutral)
        rig_new_lo, rig_new_hi = prior.hyperext_clip_range(*rig[joint], sign, neutral)
        rows.append(
            {
                "joint": joint,
                "side": side_of[joint],
                "sign": sign,
                "neutral": neutral,
                "lo": lo,
                "hi": hi,
                "new_lo": new_lo,
                "new_hi": new_hi,
                "rig_lo": rig[joint][0],
                "rig_hi": rig[joint][1],
                "rig_new_lo": rig_new_lo,
                "rig_new_hi": rig_new_hi,
            }
        )
    if len(rows) != len(side_of):
        raise SystemExit(f"[hyperext_limits] {spec}/{emb}: {len(rows)} targets ≠ {len(side_of)} per-side targets")
    return {"meta": meta, "marker": prior.prior_marker(meta), "rows": rows}


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3) or (len(argv) == 3 and argv[2] != "--off-ok"):
        raise SystemExit("usage: -m dexengine.retarget.ik.hyperext_limits <spec dir> <emb> [--off-ok] < ranges.json")
    spec, emb = argv[:2]
    ranges = json.load(sys.stdin)
    path = str(hand_rig_dir(spec) / "retarget_prior.yaml")
    with open(path, "rb") as f:
        prior_sha = hashlib.sha256(f.read()).hexdigest()
    out = {
        "rule_root": os.path.dirname(os.path.abspath(prior.__file__)),
        "rule_file": prior.__file__,
        "prior": path,
        "prior_sha256": prior_sha,
        "emb": emb,
        **table(spec, emb, ranges, off_ok=len(argv) == 3),
    }
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
