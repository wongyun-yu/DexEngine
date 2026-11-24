#!/usr/bin/env python3
"""Final per-side gate — recomputed directly from the baked artifacts (npz, tree).

The retiming numbers in the chain are values as of the warp moment, so they cannot see a seam teleport in one side's
final artifact. Here the check is made per side on the final joint trajectory itself:

  r_p99 <= GATE_SPEED_P99_MAX  and  r_max <= GATE_SPEED_MAX_MAX   (joint speed / limit per frame, the worst of the
      side's arm joints (URDF velocityLimit) and that side's finger joints (HandSpec applied limit) —
      stretch.frame_ratio)
  wrist err p99                       (tracking — diagnostic column only)
  grasp: share of err > GATE_GRASP_ERR_MM frames among contact-and-object-moving frames <= GATE_GRASP_BAD_FRAC
  (the values are the module constants below; the rule table core/datasets.py reads them from this file)

Output: the per-chunk table paths.gate_table(<root>, <combo>, <tag>) — one row per clip and side (columns =
ledger_states.GATE_COLUMNS, read by name), the verdict and reason, then per-side depths reported for comparison
(hand <-> own object and hand <-> support at the spawn frame; N/A = nothing to measure) and the cross repair's object
relative shift (cross_repair.obj_rel_shift_reported; N/A = no cross-repair report). The table is the gate's only output.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from dexengine.core.paths import bake_outputs, canonical_clip, read_clip_list
from dexengine.placement import ledger_states as LS  # table header GATE_COLUMNS · verdict words · non-verdict rule
from dexengine.placement.collision import hand_surface as HSF  # the tree's finger values · the applied finger limit
from dexengine.placement.collision import support_surface as SSF  # the clip's support objects
from dexengine.placement.coords import core as _VC  # object block addresses (owned by the tree)
from dexengine.placement.gate import dyn_xr as DXR  # J-gate-dynpairs-xr (S-dyn-pairs after cross repair)
from dexengine.placement.gate.checks import (  # noqa: F401 — re-exported: the gate's per-clip helpers live there
    GATE_RC_SYSTEMATIC,
    _err,
    _f0_reason,
    _fail_both,
    _joint_model,
    _support_report,
    _term_gate,
    _verdict,
    chunk_rc,
    height_reason,
    speed_ratio,
    systematic_rc,
)
from dexengine.placement.gate.run_setup import build_parser, gate_context, write_gate_table
from dexengine.placement.judge import input_record as _JFP  # the judge's per-clip input statement (its scene height)
from dexengine.placement.retime import stretch as VST  # the speed-ratio definition (frame_ratio)

#: Gate thresholds — the rule table (J-gate-* cards in core/datasets.py) reads them from this file.
GATE_SPEED_P99_MAX = 1.0  # joint speed / speed limit (arm URDF · finger HandSpec applied) — cap on the p99 of the ratio
GATE_SPEED_MAX_MAX = 3.0  # cap on the max of the same ratio
GATE_GRASP_ERR_MM = 70  # a grasp-and-move frame whose wrist error exceeds this is a "slipped frame"
GATE_GRASP_BAD_FRAC = 0.05  # cap on the slipped-frame share
GATE_GRASP_MIN_FRAMES = 5  # with fewer grasp-and-move frames than this the grasp share is not measured (left at 0)
GATE_OBJ_MOVING_MPS = 0.01  # an object-moving frame = object xy speed > this (m/s)
GATE_FINGER_RANGE_TOL_DEG = (
    0.5  # how far (deg) a finger angle may exceed the reference range (retargeting cut) — else FAIL
)
_SAID_FR: list = []  # [FINGER_RANGE_GATE] once per process


def clip_rows(meas: dict, t_tree, n_art: dict) -> dict:
    """One clip's per-side verdict inputs from its measurements (``meas[side]`` = the side's numbers · None = the side
    has no arm IK; ``t_tree`` = the tree's frame count · None = no tree trajectory). The verdict tests (speed · grasp ·
    the arm guard — the judge's and the bake's function) set ok/why; err_p99 is a diagnostic column only. A bake-OK
    clip's artifact defects are not data — they go to err (ERROR, see gate.checks), never ok False: no arm IK · a
    finger column of another length (speed_ratio's reason) · no contact label (the contact reward rejects the tree; a
    silent grasp 0 would PASS it) · a generation mismatch (a single-side re-bake leaves ik_right and ik_left — or the
    tree — at different frame counts; the repair is a re-bake of both sides under the same warp). The clip counts once
    in n_art (chunk_rc's artifact-defect term). The reason strings are gate-table data: keep their bytes."""
    res = {}
    for side, m in meas.items():
        if m is None:
            res[side] = {
                "ok": True,
                "why": "",
                "err": "no IK",
                "T": 0,
                "st": np.nan,
                "rp": np.nan,
                "rm": np.nan,
                "ep": np.nan,
                "gb": np.nan,
            }
            continue
        rp, rm, gb, (gok, gwhy), pen = m["rp"], m["rm"], m["gb"], m["guard"], m["pen"]
        ok = (
            (bool(m["sp_why"]) or (rp <= GATE_SPEED_P99_MAX and rm <= GATE_SPEED_MAX_MAX))
            and gb <= GATE_GRASP_BAD_FRAC
            and gok
        )
        why = (
            ""
            if ok
            else ";".join(
                ([f"r_p99={rp:.2f}{m['labs'][0]}"] if rp > GATE_SPEED_P99_MAX else [])
                + ([f"r_max={rm:.1f}{m['labs'][1]}"] if rm > GATE_SPEED_MAX_MAX else [])
                + ([f"grasp{gb * 100:.0f}%"] if gb > GATE_GRASP_BAD_FRAC else [])
                + (["penetration N/A (no colliders · explicitly allowed)"] if pen != pen else ([gwhy] if gwhy else []))
            )
        )
        err = ";".join(
            ([m["sp_why"]] if m["sp_why"] else [])
            + (["no contact labels (contact_derive_combo not run)"] if m["contact_missing"] else [])
        )
        res[side] = {
            "ok": ok,
            "why": why,
            "err": err,
            "T": m["T"],
            "st": m["st"],
            "rp": rp,
            "rm": rm,
            "ep": m["ep"],
            "gb": gb,
            "pen": pen,
            "hsup": m["hsup"],
        }
    ts = {s: r["T"] for s, r in res.items() if r["T"]}
    if len(ts) == 2 and (len(set(ts.values())) > 1 or (t_tree is not None and set(ts.values()) != {t_tree})):
        msg = (
            f"generation_mismatch(ik_right={ts['right']},ik_left={ts['left']}"
            + (f",tree={t_tree}" if t_tree is not None else "")
            + ")"
        )
        for side in res:
            _err(res[side], msg)
    n_art["unmeasurable" if any(r["err"] for r in res.values()) else "measured"] += 1
    return res


def finger_range_reason(tree, hand: str, n_fr: dict) -> str | None:
    """J-gate-fingerrange: every finger angle of the baked reference — every frame, the f0 pre-pad and the cross-repair
    frames included — lies within the reference range (wrist_shift.scene_finger_ranges: the vendor range the
    retargeting IK cut) + GATE_FINGER_RANGE_TOL_DEG.

    An independent oracle of the two repairs that write fingers: each joint's qpos column is the compiled scene's own
    (MuJoCo jnt_qposadr of _joint_model), not the layout readers the repairs use; only the range is read from its
    owner. A tree it cannot measure has no declared data property here — every exception is GateUnmeasured (both
    sides ERROR), never a reason or a silent pass."""
    if not Path(tree).is_dir():
        return None  # no tree: the gate's per-clip reads already die on it
    import mujoco as mj

    from dexengine.placement.repair import wrist_shift as WS

    def _read():
        m = _joint_model(Path(tree) / "scene.xml")
        emb = json.loads((Path(tree) / "task_info.json").read_text())["embodiment_type"]
        ref = WS.scene_finger_ranges(hand, m, emb)
        q = np.load(Path(tree) / "0/trajectory_kinematic.npz")["qpos"]
        return ref, q, {n: int(m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)]) for n in ref}

    got, why = LS.measured(n_fr, "finger_range", (), _read)
    if why:
        return why
    ref, q, col = got
    if not _SAID_FR:
        _SAID_FR.append(True)
        print(
            f"[FINGER_RANGE_GATE] tolerance {GATE_FINGER_RANGE_TOL_DEG:g}° · range = wrist_shift.scene_finger_ranges · "
            f"columns = compiled scene jnt_qposadr · joints {len(ref)}",
            flush=True,
        )
    over = sorted(
        ((float(np.degrees(np.maximum(lo - q[:, col[n]], q[:, col[n]] - hi).max())), n) for n, (lo, hi) in ref.items()),
        reverse=True,
    )
    bad = [(d, n) for d, n in over if d > GATE_FINGER_RANGE_TOL_DEG]
    return (
        "finger_range:" + ",".join(f"{n}+{d:.1f}°" for d, n in bad[:3]) + (f"(+{len(bad) - 3})" if len(bad) > 3 else "")
        if bad
        else None
    )


COUNTERS = ("n_height", "n_f0", "n_fr", "n_art", "n_sup", "n_dxr")  # the chunk's per-term counts (each clip adds)
#: The gate's clip pool (capacity.NEEDS key): every worker holds one clip's collision model and guard arrays at a time;
#: the lane's CPU budget (stage2 passes --workers = DEXENGINE_BAKE_NPAR) is the CPU share, memory and the cap decide.
STEP = "dexengine.placement.gate.core"
_G: dict = {}  # the context a forked worker reads (set before the pool forks)


def _measure_side(
    f, _qs, side, LIM, _fv, FBUD, out, VC_, tree, PEN_CTX, AP, SCL, _fq, _sup, clip, n_sup, qk, qk_p, T_orig, meas
):
    """One side's gate measurements (speed, wrist error, penetration/guard, contact-band, stretch) into meas[side]."""
    z = np.load(f)
    q = z["arm_joint_positions_all"].astype(np.float64)
    _qs[side] = q
    rp, rm, _fp, _fm, _sp_why = speed_ratio(q, LIM[side] * VST.FRAME_DT, _fv[side], FBUD)
    e = z["wrist_pos_error_all"].astype(np.float64) * 1000
    ep = float(np.percentile(e, 99))
    # The placement file must exist — it is the source of the clip enumeration. A read or key failure does not
    # fall back to z_off=0, it dies: the bake reads the same key strictly.
    _adp = out["placement"]
    _zoff = float(json.loads(_adp.read_text())["robot_z_off"])
    _dxy = VC_.desk_bounds_scene_for(tree, margin=0.0)
    # The verdict is the judge's own step (collision.side_guard: desk, support objects, then the torso); the
    # penetration column is the desk surface penetration it measured. No colliders (explicitly allowed) -> N/A.
    pen_mm, _hs = float("nan"), None
    _gok, _gwhy = True, ""
    if PEN_CTX[side]["surf"]["links"]:
        with AP.mesh_ruler():
            pen_mm = (
                1000.0
                * SCL.desk_penetration(PEN_CTX[side], q, z_off=_zoff, want_components=True, desk_xy=_dxy, torso=False)[
                    3
                ]
            )
        _gok, _gwhy = SCL.side_guard(
            PEN_CTX[side], q, side, z_off=_zoff, desk_xy=_dxy, fingers=_fq[side], supports=_sup
        )
        _hs = _support_report(clip, side, PEN_CTX[side], q, _fq[side], _zoff, _sup, _gwhy, n_sup)
    gb = 0.0
    cf = tree / f"0/contact_mano_{side}.npz"
    has_cf = cf.exists()  # none = an artifact defect (clip_rows)
    if has_cf and qk is not None:
        cz = np.load(cf)
        cm = cz["contact"].any(axis=(1, 2))
        n = min(len(cm), len(qk), len(e))
        # The object-motion test uses the object blocks the tree declares (fixed columns would read the wrong
        # columns on 1-object or 3+-object trees).
        _sl = _VC.tree_object_slots(qk_p.parent.parent, qk.shape[1])
        mv = np.zeros(n, bool)
        for _s0, _ in _sl:
            if _s0 + 2 > qk.shape[1]:
                continue
            _sp = np.linalg.norm(np.diff(qk[:n, _s0 : _s0 + 2], axis=0), axis=1) / VST.FRAME_DT
            mv |= np.concatenate([[False], _sp > GATE_OBJ_MOVING_MPS])
        mask = cm[:n] & mv[:n]
        if mask.sum() >= GATE_GRASP_MIN_FRAMES:
            gb = float((e[:n][mask] > GATE_GRASP_ERR_MM).mean())
    st = (len(q) / T_orig) if T_orig else np.nan
    meas[side] = {
        "T": len(q),
        "st": st,
        "rp": rp,
        "rm": rm,
        "labs": (_fp, _fm),
        "sp_why": _sp_why,
        "gb": gb,
        "guard": (_gok, _gwhy),
        "pen": pen_mm,
        "contact_missing": not has_cf,
        "ep": ep,
        "hsup": _hs,
    }


def _table_rows(res, rows, clip, anc, _xr):
    """The two table rows of the clip (verdict and every measured term)."""
    for side in ("right", "left"):
        d, (v, vwhy) = res[side], _verdict(res[side])
        rows.append(
            (
                clip,
                anc,
                side,
                d["T"],
                f"{d['st']:.2f}",
                f"{d['rp']:.3f}",
                f"{d['rm']:.2f}",
                f"{d['ep']:.1f}",
                f"{d['gb']:.3f}",
                (
                    "N/A"
                    if d.get("pen", float("nan")) != d.get("pen", float("nan"))  # NaN = no colliders (allowed)
                    else f"{d['pen']:+.1f}"
                ),
                "N/A" if _xr is None else f"{_xr:.1f}",
                v,
                vwhy,
                "N/A" if d.get("f0own") is None else f"{d['f0own']:.1f}",  # the two depth columns
                "N/A" if d.get("hsup") is None else f"{d['hsup']:.1f}",
            )
        )


def _clip_trajectories(clip, a):
    """The clip's bake outputs, its baked kinematic trajectory and the original length (None when missing)."""
    out = bake_outputs(clip, a.tree_root, a.robots_dir)
    tree = out["tree"]
    qk_p = tree / "0/trajectory_kinematic.npz"
    T_orig = None
    qk = None
    if qk_p.exists():
        qk = np.load(qk_p)["qpos"]
    oq = Path(a.base_root) / clip / "0/trajectory_kinematic.npz"
    if oq.exists():
        T_orig = len(np.load(oq)["qpos"])
    return out, tree, qk_p, T_orig, qk


def _clip(G: dict, clip: str) -> list:
    """One clip's two table rows (right, left) — every term of the gate on that clip; adds to the counters in G."""
    from dexengine.placement.collision import arm_proxy as AP
    from dexengine.placement.collision import core as SCL
    from dexengine.placement.collision import idle_arm as IA
    from dexengine.placement.repair import cross_repair as XR  # the cross-repair report's audit value (table column)

    a, LIM, PEN_CTX, FBUD, jfp = G["a"], G["LIM"], G["PEN_CTX"], G["FBUD"], G["jfp"]
    n_height, n_f0, n_fr, n_art, n_sup, n_dxr = (G[k] for k in COUNTERS)
    VC_, anc, rows = _VC, a.tag, []  # VC_: desk box from the tree footprint (else the canonical constant)
    out, tree, qk_p, T_orig, qk = _clip_trajectories(clip, a)
    meas = {}
    _qs = {}
    # Torso guard hand = this tree's finger values (idle side · absent hand = hand_surface.rest_finger_q) — the
    # same function as the judge, the bake and the final guard (hand_surface.guard_finger_q).
    _fq = HSF.guard_finger_q(tree, a.hand, idle=IA.idle_sides_of_tree(tree), qpos=qk)
    # The speed gate's fingers = every hand block the retiming floored (idle=() — an idle side too): an
    # independent re-measurement of the final artifact against the current HandSpec limit.
    _fv = HSF.guard_finger_q(tree, a.hand, idle=(), qpos=qk)
    _sup = SSF.scene_supports(tree)  # the guard's support term — the same function as judge · bake
    n_sup["clips"] += bool(_sup)
    _xr = XR.obj_rel_shift_reported(tree)  # a clip value (both sides) · None = no xr report
    for side in ("right", "left"):
        f = out[f"ik_{side}"]
        if not f.exists():  # no arm IK — clip_rows
            meas[side] = None
            continue
        _measure_side(
            f,
            _qs,
            side,
            LIM,
            _fv,
            FBUD,
            out,
            VC_,
            tree,
            PEN_CTX,
            AP,
            SCL,
            _fq,
            _sup,
            clip,
            n_sup,
            qk,
            qk_p,
            T_orig,
            meas,
        )
    res = clip_rows(meas, len(qk) if qk is not None else None, n_art)
    # Start support: a main object the hand does not hold at the start must stand on a support that vanishes only
    # once a hand holds it. The bake stamped the verdict (coords.start_support); a tree property, so both sides fail
    # together. The "D40:" prefix is gate-table data.
    if tree.is_dir():
        _d40 = VC_.start_support_verdict(tree)
        if _d40:
            _msg = "D40:" + ",".join(f"{x['why']}({x['obj']})" for x in _d40)
            for side in ("right", "left"):
                res[side]["ok"] = False
                res[side]["why"] = ((res[side]["why"] + ";") if res[side]["why"] else "") + _msg
    _term_gate(res, lambda: _f0_reason(res, tree, out, a.assets_root, n_f0))  # J-gate-f0pen
    _term_gate(res, lambda: finger_range_reason(tree, a.hand, n_fr))  # J-gate-fingerrange
    _term_gate(res, lambda: height_reason(tree, jfp.get(clip), n_height))  # height contract
    # S-dyn-pairs on the rows after cross repair (J-gate-dynpairs-xr): a pair cross repair turned off or deepened is
    # read again on the final rows; a missing input copy raises (the gate stops), never a pass on nothing.
    _fail_both(res, DXR.reason(tree, n_dxr) if tree.is_dir() else None)
    # Two-arm crossing — a pair property, so it sits outside the per-side loop: the judge's own step
    # (collision.cross_guard, the last step of pen_cross_guard) — the case check = gate contract.
    if "right" in _qs and "left" in _qs:
        _xok, _xwhy = SCL.cross_guard(PEN_CTX["right"], PEN_CTX["left"], _qs["right"], _qs["left"])
        if not _xok:
            for side in ("right", "left"):
                if res[side]["ok"]:
                    res[side]["ok"] = False
                    res[side]["why"] = ((res[side]["why"] + ";") if res[side]["why"] else "") + _xwhy
    _table_rows(res, rows, clip, anc, _xr)
    return rows


def _merge(G: dict, delta: dict) -> None:
    """Add one worker clip's counter increments to G (a flag — e.g. a marker said — is or-ed)."""
    for k, d in delta.items():
        for x, v in d.items():
            G[k][x] = (bool(G[k].get(x)) or v) if isinstance(v, bool) else G[k].get(x, 0) + v


def _clip_worker(clip: str):
    """A forked worker's clip: (rows, counter increments, its fd 1 + fd 2 output, traceback text or '', peak RSS kB)."""
    import resource
    import tempfile
    import traceback

    G, err, rws = _G, "", []
    before = {k: dict(G[k]) for k in COUNTERS}
    with tempfile.TemporaryFile() as tf:
        sys.stdout.flush()
        sys.stderr.flush()
        keep = (os.dup(1), os.dup(2))
        os.dup2(tf.fileno(), 1)
        os.dup2(tf.fileno(), 2)
        try:
            rws = _clip(G, clip)
        except BaseException:  # noqa: BLE001 — carried to the parent, which dies on it (never a skipped clip)
            err = traceback.format_exc()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(keep[0], 1)
            os.dup2(keep[1], 2)
            os.close(keep[0])
            os.close(keep[1])
        tf.seek(0)
        text = tf.read()
    delta = {k: {x: (v if isinstance(v, bool) else v - before[k].get(x, 0)) for x, v in G[k].items()} for k in COUNTERS}
    return rws, delta, text, err, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _run_clips(G: dict, clips: list, workers: int) -> list:
    """Every clip's rows in clip order. One worker (or fewer than three clips): clip by clip in this process. Else the
    first clip runs here — its once-per-process markers and the clip-independent caches it builds (the URDF shapes ·
    the finger-range marker) exist before the fork — and the rest go to forked workers that share the built context;
    each returns its rows, counter increments and captured output, written here in clip order (the table and the log
    keep the serial order). A worker's failure — its exception or its death — kills the gate (rc != 0) after its
    output is written; no clip is ever skipped."""
    if len(clips) < 3:
        return [r for c in clips for r in _clip(G, c)]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from dexengine.core import capacity

    with capacity.sized(STEP, len(clips) - 1, share=int(workers)) as cap:
        if cap.workers <= 1:
            return [r for c in clips for r in _clip(G, c)]
        rows = _clip(G, clips[0])
        print(
            f"[GATE_WORKERS] {cap.workers} (the [CPU] line) · clips {len(clips)} — first clip in this "
            "process, the rest in fork workers · table and log in clip order",
            flush=True,
        )
        sys.stderr.flush()
        _G.clear()
        _G.update(G)
        rss = []
        with ProcessPoolExecutor(
            cap.workers, mp_context=mp.get_context("fork"), initializer=capacity.worker_init, initargs=cap.initargs
        ) as ex:
            for clip, (rws, delta, text, err, kb) in zip(clips[1:], ex.map(_clip_worker, clips[1:]), strict=True):
                sys.stdout.flush()
                while text:
                    text = text[os.write(1, text) :]
                if err:
                    raise RuntimeError(f"[GATE_WORKERS] the worker of clip {clip} died — stopping the gate:\n{err}")
                rows += rws
                _merge(G, delta)
                rss.append(kb)
        print(f"[GATE_WORKERS] done · clips {len(clips)} · worker peak RSS {max(rss) / 1024:.0f} MB", flush=True)
        return rows


def main() -> int:
    a = build_parser().parse_args()
    G = gate_context(a)
    listed = read_clip_list(a.clips, dataset=a.dataset, label="gate clips", quiet=True)
    # Height contract: an aerial tree's stamped scene height must be the height the judge solved the robot against
    # (the setup field of its input fingerprint). Every clip is checked — no sampling. Tolerance = one unit of the
    # fingerprint's rounding.
    G["jfp"] = _JFP.read_files([a.judge_root], key=lambda c: canonical_clip(c, a.dataset))
    # only clips with a placement record (a bake-OK clip has one)
    clips = sorted({c for c in listed if bake_outputs(c, a.tree_root, a.robots_dir)["placement"].is_file()})
    rows = _run_clips(G, clips, a.workers)
    # One table per chunk, published by rename (a reader never sees half a table). Its reason column carries every
    # failure reason, so no separate failures-only file is needed.
    return write_gate_table(a, G["combo"], rows, G["n_height"], G["n_f0"], G["n_sup"], G["n_dxr"]) or chunk_rc(
        G["n_f0"], G["n_fr"], G["n_height"], G["n_art"]
    )


if __name__ == "__main__":
    sys.exit(main())
