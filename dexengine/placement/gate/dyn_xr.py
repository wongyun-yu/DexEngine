"""Gate J-gate-dynpairs-xr — S-dyn-pairs on the rows after cross repair (the rule and its reading are
data/objects/dyn_pairs.after_cross_repair's; this file reads the tree's three row sets and states the gate's reason).

Rows: before cross repair = its input copy (wrist_shift backup, tag cross_repair.XR_TAG); after it on the same time
axis = the copy retiming keeps (core.locations.names.IK_SIDECAR_SFX), or the tree's rows when the tree carries no
retiming (no retime.TAU_KEY in its npz); final = the tree's rows (what the scene reads). A clip whose pair cross repair
turned off or deepened has a visual overlap deeper than the true-shape overlap on the final rows fails both sides
(reason REASON_HEAD:<pairs>). A tree without a cross-repair report (cross repair off) is not read and is counted
(`no_xr`); a report without its input copy, or a retimed tree without the copy before retiming, raises — the gate
stops on it (no clip is ever skipped).
Counts (gate_context `n_dxr`; the chunk's `[DYN_PAIRS_XR]` line, run_setup.write_gate_table): trees · no_xr · read
(trees with a pair read again) · turned_off · deepened · drop · freed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REASON_HEAD = "dyn_pair_xr"  # gate-table reason head of a drop (the pairs follow)
COUNTS = ("trees", "no_xr", "read", "turned_off", "deepened", "drop", "freed")


def rows(tree) -> tuple | None:
    """(q_pre, q_post, q_final) of one baked tree, or None when cross repair did not run (no report)."""
    from dexengine.core.locations.names import IK_SIDECAR_SFX  # noqa: PLC0415 — the copy before retiming
    from dexengine.placement.repair import cross_repair as XR  # noqa: PLC0415 — the report and the input-copy tag
    from dexengine.placement.repair import wrist_shift as WS  # noqa: PLC0415 — the backup naming
    from dexengine.placement.retime import retime as RT  # noqa: PLC0415 — the retimed tree's key

    tree = Path(tree)
    if not (tree / XR.REPORT_REL).is_file():
        return None
    kin = tree / "0" / "trajectory_kinematic.npz"
    pre = WS._bk(kin, XR.XR_TAG)[0]
    if not pre.is_file():
        raise FileNotFoundError(f"[DYN_PAIRS_XR] {tree.name}: a cross-repair report without its input copy {pre.name}")
    z = np.load(kin)
    q_final = np.asarray(z["qpos"], float)
    if RT.TAU_KEY in z.files:
        post = kin.with_suffix(IK_SIDECAR_SFX)
        if not post.is_file():
            raise FileNotFoundError(f"[DYN_PAIRS_XR] {tree.name}: a retimed tree without the copy before retiming")
        q_post = np.asarray(np.load(post)["qpos"], float)
    else:
        q_post = q_final
    return np.asarray(np.load(pre)["qpos"], float), q_post, q_final


def reason(tree, n_dxr: dict) -> str | None:
    """The gate reason of one tree (None = no drop); adds to the counts `n_dxr` (COUNTS)."""
    from dexengine.data.objects import dyn_pairs as DP  # noqa: PLC0415 — the rule

    n_dxr["trees"] += 1
    got = rows(tree)
    if got is None:
        n_dxr["no_xr"] += 1
        return None
    ti = json.loads((Path(tree) / "task_info.json").read_text())
    st = DP.after_cross_repair(Path(tree), ti, *got)
    n_dxr["read"] += bool(st["read"])
    n_dxr["turned_off"] += sum(p["why"] == DP.XR_TURNED_OFF for p in st["read"])
    n_dxr["deepened"] += sum(p["why"] == DP.XR_DEEPENED for p in st["read"])
    n_dxr["freed"] += len(st["freed"])
    if not st["drop"]:
        return None
    n_dxr["drop"] += 1
    cells = ",".join(
        f"{p['a']}x{p['b']}({p['visual_m'] * 1e3:.1f}mm@{p['visual_frame']})"
        for p in st["read"]
        if p["verdict"] == DP.DROP
    )
    return f"{REASON_HEAD}:{cells}"


def line(n_dxr: dict) -> str:
    """The chunk's count line."""
    from dexengine.data.objects import dyn_pairs as DP  # noqa: PLC0415
    from dexengine.data.objects import scene_requirement as SR  # noqa: PLC0415

    return (
        f"[DYN_PAIRS_XR] trees {n_dxr['trees']} · no cross repair {n_dxr['no_xr']} · read again {n_dxr['read']} "
        f"(turned off {n_dxr['turned_off']} · deepened {n_dxr['deepened']}) · dropped {n_dxr['drop']} (visual > "
        f"{SR.TRUE_SHAPE_OVERLAP_M * 1e3:g} mm) · freed {n_dxr['freed']} · deeper than {DP.XR_DEEPER_TOL_M * 1e3:g} mm"
    )
