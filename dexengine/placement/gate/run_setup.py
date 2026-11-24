"""Final gate run frame: the CLI parser, the clip-independent run context (arm speed limits, the deciding ruler's arm
contexts, the finger budget) and the gate table writer. The per-clip verdict lives in gate/core.py.
"""

from __future__ import annotations

import argparse

import numpy as np

from dexengine.core.paths import gate_table
from dexengine.placement import ledger_states as LS
from dexengine.placement.collision import hand_surface as HSF
from dexengine.placement.collision import support_surface as SSF
from dexengine.placement.gate import dyn_xr as DXR  # J-gate-dynpairs-xr counts
from dexengine.placement.judge import input_record as _JFP


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--root", required=True, help="gate meta root (the dataset variant's gate folder) — no default")
    ap.add_argument(
        "--base-root",
        required=True,
        help="reference universe (children = base clip directories) — source of the original length "
        "T_orig (stretch factor). Owner = paths.ref_base_dir; stage 2 passes the value resolved with "
        "`--what ref-base --dataset/--hand/--emb/--src-layer`. No default",
    )
    # What is measured = this chunk's bake-OK list + the output locations (paths.bake_outputs). The row's anchor
    # column = --tag.
    ap.add_argument("--clips", required=True, help="this chunk's bake-OK list (stage 2's baked_ok.txt)")
    ap.add_argument("--tree-root", required=True, help="baked tree folder (stage 2's TROOT)")
    ap.add_argument("--robots-dir", required=True, help="placement and arm IK root (stage 2's W2 = robots/<combo>)")
    ap.add_argument(
        "--tag", required=True, help="chunk tag — the table's anchor column and file name (paths.gate_table)"
    )
    ap.add_argument(
        "--dataset", required=True, help="dataset of the list — clip name normalisation (paths.read_clip_list)"
    )
    ap.add_argument(
        "--judge-root",
        required=True,
        help="this chunk's judge root (stage 2's OUT) — the judge input fingerprint's height is checked "
        "against the stamp",
    )
    ap.add_argument(
        "--assets-root",
        required=True,
        help="object shape root (stage 2's ASSETS_ROOT = parent of the assets_objects layer) — frame-0 "
        "robot <-> object penetration (J-gate-f0pen)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="clip-level fork workers: the CPU share (capacity caps it by memory) — stage 2 passes the lane's CPU "
        "budget DEXENGINE_BAKE_NPAR. 1 = one process. The table and the log keep clip order either way",
    )
    return ap


def write_gate_table(a, combo, rows, n_height, n_f0, n_sup, n_dxr):
    gt = gate_table(a.root, combo, a.tag)
    gt.parent.mkdir(parents=True, exist_ok=True)
    tmp = gt.with_name(gt.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write("\t".join(LS.GATE_COLUMNS) + "\n")
        for r_ in rows:
            fh.write("\t".join(str(x) for x in r_) + "\n")
    tmp.replace(gt)
    print(
        f"[GATE_HEIGHT] aerial tree heights checked {n_height['measured']} · off {n_height['bad']} · "
        f"no fingerprint {n_height['no_fingerprint']} · unmeasurable {n_height['unmeasurable']} "
        f"(tolerance {1000 * 10.0**-_JFP.SETUP_ROUND:.1f} mm)",
        flush=True,
    )
    print(
        f"[F0_PEN] measured {n_f0['measured']} · violations {n_f0['violation']} (object-object "
        f"{n_f0.get('violation_3_obj_obj', 0)} · arm-object {n_f0.get('violation_5_arm_obj', 0)}) · unmeasurable "
        f"{n_f0['unmeasurable']} · hand-own-object deeper than arm-object {n_f0['hand_report']} (reported only)",
        flush=True,
    )
    from dexengine.placement.collision import core as _SCL

    print(
        f"[SUPPORT_GUARD] clips with a support object {n_sup['clips']} · sides failed on arm-support penetration "
        f"{n_sup['arm_fail']} · sides with hand-support > {_SCL.DESK_SURF_PEN_MM:g} mm at frame 0 "
        f"{n_sup['hand_report']} (reported only — support_surface.HAND_SUPPORT_MODE={SSF.HAND_SUPPORT_MODE})",
        flush=True,
    )
    xr = [float(v) for v in {r_[0]: r_[LS.GATE_COLUMNS.index("xr_obj_rel_mm")] for r_ in rows}.values() if v != "N/A"]
    print(
        f"[XR_OBJ_REL_GATE] clips with a cross-repair report {len(xr)} · max {max(xr, default=0.0):.1f} mm · without "
        f"{len({r_[0] for r_ in rows}) - len(xr)} (N/A — cross repair off) (reported only — not a verdict)",
        flush=True,
    )
    print(DXR.line(n_dxr), flush=True)
    npass = sum(1 for r_ in rows if r_[LS.GATE_COLUMNS.index("verdict")] == "PASS")
    print(f"[v5gate:{combo}] rows {len(rows)} · PASS {npass} -> {gt}", flush=True)
    return 0


def gate_context(a) -> dict:
    """The clip-independent context of one gate run: the arm speed limits (URDF velocityLimit), the deciding ruler's
    two arm contexts (mesh ruler) and the applied finger budget — built once, read by every clip."""
    from dexengine.placement import arm_ik as AIP
    from dexengine.placement.collision import arm_proxy as AP
    from dexengine.placement.collision import core as SCL
    from dexengine.robots import urdf_load

    LIM = {}
    for side in ("right", "left"):
        urdf, ee, joints, bx, _ = AIP.resolve_combo(a.arm, a.hand, side)
        model = urdf_load.build_model(str(urdf))
        idx = [model.joints[model.getJointId(n)].idx_v for n in joints]
        LIM[side] = np.asarray(model.velocityLimit)[idx].astype(np.float64)
    with AP.mesh_ruler():  # the gate is a deciding ruler: arm_proxy.GUARD_KIND (the original mesh), never the proxy
        PEN_CTX = {s2: SCL.build_arm_ctx(a.arm, a.hand, s2) for s2 in ("right", "left")}
    FBUD = HSF.finger_budget(a.hand)  # applied finger limit (HandSpec) × stretch.FRAME_DT — unknown hand dies here
    return {
        "a": a,
        "combo": f"{a.arm}_{a.hand}",
        "LIM": LIM,
        "PEN_CTX": PEN_CTX,
        "FBUD": FBUD,
        "n_height": {"measured": 0, "unmeasurable": 0, "bad": 0, "no_fingerprint": 0},  # height_reason
        "n_f0": {"measured": 0, "violation": 0, "unmeasurable": 0, "hand_report": 0},
        "n_fr": {"measured": 0, "unmeasurable": 0},  # J-gate-fingerrange (ledger_states.measured)
        "n_art": {"measured": 0, "unmeasurable": 0},  # a bake-OK clip's artifact defect (the gate's one rule)
        "n_sup": {"clips": 0, "arm_fail": 0, "hand_report": 0},
        "n_dxr": dict.fromkeys(DXR.COUNTS, 0),  # J-gate-dynpairs-xr (dyn_xr.reason)
    }
