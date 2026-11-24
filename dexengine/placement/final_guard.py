#!/usr/bin/env python3
"""Final hard constraints of the bake — checks, after retiming, the exact bytes the gate reads.

The IK stage's guards look at the joints before retiming. Retiming lengthens the time axis and turns poses on the
chord between frames into real data, and those poses can cross a threshold; such violations are only caught here.

The check function, thresholds and inputs are the same as the judge's and the gate's: all three call
`collision.pen_cross_guard()` (the gate calls its two steps side_guard and cross_guard). This is the second layer of
the two-layer guard; the first runs at judge adoption, before retiming. The ruler is the sim collider surface
(arm_surface):
  desk    surface penetration <= collision.DESK_SURF_PEN_MM
  torso   arm / palm / finger <-> torso surface overlap <= dexengine.placement.collision.torso.TORSO_OVERLAP_MAX_MM
  (exact ruler; the
          fingers are the tree's finger values, on the retimed tree and on the pre-retiming tree `*.pre_rt.npz`)
  cross   signed distance between the two arm surfaces >= -collision.ARM_OVERLAP_MAX_MM
The thresholds are constants. When the pre-retiming joints exist they are checked with the same function, and the
verdict line says whether the violation was there before retiming or was created by it.

Verdict line: one `[final_guard] <sequence>: <head> — <text>` line (written by verdict_line, read by verdict_head in
ledger.bake_fail_status; the bake driver copies these lines into the bake ledger). Three heads: reject (HEAD_REJECT —
penetration / cross violation after retiming, rc 4, the physical rejection GUARD_FAIL), missing input (HEAD_INPUT —
no IK npz, rc 2) and fault (HEAD_FAULT — the line is printed and the exception re-raised, rc 1). Missing input and
fault are re-bake cases (BAKE_ERROR). The head strings and the verdict-line texts are ledger data: keep their bytes.
The robot height (robot_z_off) comes in as --base-z-off (read strictly from the adopted JSON by the caller).

Return 4 = violation (the chain dies and the bake driver retries the next anchor) · 2 = missing input.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from dexengine.core.paths import IK_SIDECAR_SFX

#: Tag and the three heads of the verdict line — only the reject head is GUARD_FAIL. Ledger data: keep the bytes.
TAG = "[final_guard]"
HEAD_REJECT, HEAD_INPUT, HEAD_FAULT = "violation after retiming", "missing input", "exception"
HEADS = (HEAD_REJECT, HEAD_INPUT, HEAD_FAULT)


def verdict_line(seq: str, head: str, text: str) -> str:
    """One verdict line — the owner of its form (writer side)."""
    if head not in HEADS:
        raise ValueError(f"[final_guard] unknown verdict head {head!r} — {HEADS}")
    return f"{TAG} {seq}: {head} — {text}"


def verdict_head(line: str) -> str | None:
    """Verdict line -> its head (one of HEADS), or None if it is not a verdict line.

    Parses the verdict_line form by position (tag, then after the first ': ' and before ' — '), not by searching for
    words: a sequence name containing ': ' shifts the head and gives None (= not a reject, the safe side)."""
    if not line.startswith(TAG + " "):
        return None
    head, sep, _ = line.partition(": ")[2].partition(" — ")
    return head if sep and head in HEADS else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--ik-right", required=True)
    ap.add_argument("--ik-left", required=True)
    ap.add_argument(
        "--tree",
        required=True,
        help="tree directory — the torso guard's finger values (qpos hand blocks) and the desk box "
        "(desk_footprint_raw)",
    )
    ap.add_argument(
        "--base-z-off",
        type=float,
        required=True,
        help="robot height = robot_z_off of the adopted JSON (the caller reads it strictly)",
    )
    a = ap.parse_args()
    try:
        return _guard(a)
    except Exception as e:  # print the fault head and re-raise (rc 1) — without a verdict line the ledger reason
        # would be an earlier stage's line
        msg = " ".join(str(e).split()).replace(";", ",").replace("|", "/")[:120]
        print(verdict_line(a.sequence, HEAD_FAULT, f"{type(e).__name__}: {msg} — failed"), flush=True)
        raise


def check(
    arm: str, hand: str, tree, q: dict, z_off: float, *, qpos=None, fingers: dict | None = None
) -> tuple[bool, str]:
    """One hard-constraint check -> (ok, first violation reason); calls the same `collision.pen_cross_guard` as the
    judge and the gate.

    q = {side: (T, 7) arm joints} · z_off = robot_z_off · qpos = the tree qpos to read the fingers from (default = the
    tree's trajectory_kinematic; the attribution check passes the pre-retiming one) · fingers = finger values given
    directly ({side: (T, n)} in hand_surface.guard_finger_q order, e.g. executed fingers of a rollout; giving both
    fingers and qpos raises)."""
    from dexengine.placement.collision import core as SCL  # noqa: PLC0415
    from dexengine.placement.collision import support_surface as SSF  # noqa: PLC0415

    with _tree_scene(tree):
        ctx, desk_xy, fingers = _inputs(arm, hand, tree, q, qpos=qpos, fingers=fingers)
        return SCL.pen_cross_guard(ctx, q, z_off, desk_xy, fingers=fingers, supports=SSF.scene_supports(tree))


def values(arm: str, hand: str, tree, q: dict, z_off: float, *, qpos=None, fingers: dict | None = None) -> dict:
    """Per-term maxima (mm) for the same inputs as `check` — collision.term_values (does not stop at the first
    violation)."""
    from dexengine.placement.collision import core as SCL  # noqa: PLC0415
    from dexengine.placement.collision import support_surface as SSF  # noqa: PLC0415

    with _tree_scene(tree):
        ctx, desk_xy, fingers = _inputs(arm, hand, tree, q, qpos=qpos, fingers=fingers)
        return SCL.term_values(ctx, q, z_off, desk_xy, fingers=fingers, supports=SSF.scene_supports(tree))


_SCENE_SAID: set = set()


def _tree_scene(tree):
    """The guard's scene = the tree's stamp (coords.scene_mode(tree) — dies there if the env disagrees; without a stamp,
    the env or table), entered as an env context.

    arm_surface's desk term calls scene_mode() without a tree and reads only the env, so a caller without the env (the
    augmentation collision gate) would otherwise measure an aerial tree against a desk that does not exist. Marker
    [GUARD_SCENE], once per scene."""
    from dexengine.placement.coords import core as VC  # noqa: PLC0415

    sc = VC.scene_mode(tree)
    if sc not in _SCENE_SAID:
        _SCENE_SAID.add(sc)
        print(
            f"[GUARD_SCENE] guard scene {sc} (tree stamp · env {os.environ.get('DEXENGINE_SCENE') or 'unset'})",
            flush=True,
        )
    return VC.scene_env(sc)


def _inputs(arm: str, hand: str, tree, q: dict, *, qpos=None, fingers: dict | None = None) -> tuple:
    """Input assembly of check and values (arm contexts, desk box, fingers) -> (ctx, desk_xy, fingers). The supports
    are passed at the guard call itself (support_surface.scene_supports)."""
    from dexengine.placement.collision import core as SCL  # noqa: PLC0415
    from dexengine.placement.collision import hand_surface as HSF  # noqa: PLC0415
    from dexengine.placement.collision import idle_arm as IA  # noqa: PLC0415
    from dexengine.placement.coords import core as _vc  # noqa: PLC0415

    if fingers is not None and qpos is not None:
        raise ValueError("[final_guard] check: give fingers or qpos, not both")
    if fingers is None:
        fingers = HSF.guard_finger_q(tree, hand, idle=IA.idle_sides_of_tree(tree), qpos=qpos)
    return (
        {side: SCL.build_arm_ctx(arm, hand, side) for side in q},
        _vc.desk_bounds_scene_for(tree, margin=0.0),
        fingers,
    )


def _guard(a) -> int:
    zoff = a.base_z_off
    q, qpre = {}, {}
    for side, p in (("right", a.ik_right), ("left", a.ik_left)):  # inputs first (before the heavy imports)
        if not Path(p).exists():
            print(verdict_line(a.sequence, HEAD_INPUT, f"{side} IK npz missing ({Path(p).name}) — failed"), flush=True)
            return 2
        q[side] = np.asarray(np.load(p)["arm_joint_positions_all"], np.float64)
        pre = Path(p).with_suffix(IK_SIDECAR_SFX)  # snapshot right before retiming (when present)
        if pre.exists():
            qpre[side] = np.asarray(np.load(pre)["arm_joint_positions_all"], np.float64)

    ok, why = check(a.arm, a.hand, a.tree, q, zoff)
    if ok:
        print(f"[final_guard] {a.sequence}: penetration and cross checks pass after retiming", flush=True)
        return 0

    # Attribution: the pre-retiming joints are checked with the same function to tell a violation that was already
    # there from one the retiming created. The words below are ledger text (keep their bytes).
    pre_verdict, attrib = "not available", "unknown"
    kin_pre = Path(a.tree) / f"0/trajectory_kinematic{IK_SIDECAR_SFX}"
    if len(qpre) == 2 and kin_pre.is_file():
        pok, pwhy = check(a.arm, a.hand, a.tree, qpre, zoff, qpos=np.load(kin_pre)["qpos"])
        pre_verdict = "pass" if pok else f"violation({pwhy})"
        attrib = "created by retiming" if pok else "present before retiming"
    # The verdict line comes last — the bake ledger reads a reject only when it is the reason's last line
    # (ledger.bake_fail_status).
    print(
        verdict_line(
            a.sequence,
            HEAD_REJECT,
            f"{why} (before retiming {pre_verdict} · attributed to {attrib}, the driver tries the next anchor)",
        ),
        flush=True,
    )
    return 4


if __name__ == "__main__":
    sys.exit(main())
