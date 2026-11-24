"""D25 (b) arctic joint-value alignment: restore the qpos joint columns of trees and base to the **source**
`qpos_obj_arti_<side>`.

Why: the hand tracker overwrites the hinge column with the source every frame, smooths all columns in
time, and at the end restores **only the free-joint (object pose) columns** to the source.
The hinge is not a free joint, so it is left out of that list and stays **smoothed**: the qpos column is a 3-frame
moving
average of the separate array (residual 0.0002–0.018 rad, measured), so within one tree the body pose (source) and the
joint value (smoothed) disagree (up to 17.6°). The env, judge and render read the qpos column, so the source is imposed
again here. The separate array stays as it is (the two records become equal).

What changes: `qpos[:, joint column]` of `0/trajectory_kinematic.npz` ← `qpos_obj_arti_<side>[:,0]`, and that column's
`qvel` is rebuilt from the same source by backward differences (the hand tracker's convention for object columns).
task_info `articulation.max_abs_diff_vs_qpos` is updated to the new value (0) and the marker `rules.D25` is added.
Idempotent: skipped when `rules.D25` exists. Usage (hand from DEXENGINE_HAND, no default):
  DEXENGINE_HAND=<hand> python3 dexengine/data/artic/arctic_align.py --emb bimanual            # dry-run (measure only)
  DEXENGINE_HAND=<hand> python3 dexengine/data/artic/arctic_align.py --emb bimanual --write    # the form stage 1 step
      2c calls
  … --roots <layer1>,<layer2>   # only the given layers (e.g. old trees). Default = this hand's arctic base layer
The default `--roots` is the active hand's layer only (other hands' trees are never rewritten). Called by stage 1
(data.stages 2c).
The audit json goes to the shared location only for the axis hand (data.rules.audit_out). The dry-run audit json is
`arctic_artic_align.dryrun.json` (paths.mode_record) and never overwrites the record of a write run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from dexengine.core.paths import IK_SIDECAR_SFX, main_root, mode_record

#: States that finished normally (first word); anything else counts as an error.
_D25_OK = ("aligned", "would_align", "skip_done", "no_artic")
DATA = str(main_root())  # root of layer paths = the data root (env DEXENGINE_MAIN_ROOT, else this code tree)
from dexengine.core.paths import (
    qpos_layout as QL,  # noqa: E402  (canonical path that imports only the layout reader, without isaaclab)
)
from dexengine.data import rules as RR  # noqa: E402  — owner of the hand axis

VERSION = "D25b-2026-09-18"


def _qvel_col(tree: str, nq: int, qcol: int) -> int:
    """qvel address of a joint column, from the owner qpos_layout.qvel_blocks (no arithmetic)."""
    lay = QL.read(tree, nq=nq)
    for kind, s, _w, v, _wv in lay.qvel_blocks():
        if kind == "joint" and s == qcol:
            return v
    raise ValueError(f"{tree}: no qvel address for joint column {qcol} in the layout")


def align(tree: str, write: bool) -> dict:
    """Restore one tree's qpos joint column to the source sidecar (measure only unless `write`); returns its row."""
    ti_p = os.path.join(tree, "task_info.json")
    with open(ti_p) as fh:
        ti = json.load(fh)
    arts = [o for o in (ti.get("objects") or []) if o.get("articulation")]
    if not arts:
        return {"tree": tree, "status": "no_artic"}
    if "D25" in (ti.get("rules") or {}):
        return {"tree": tree, "status": "skip_done"}
    if len(arts) != 1:
        return {"tree": tree, "status": f"artic_objects={len(arts)}"}
    art = arts[0]["articulation"]
    key = art.get("npz_key")
    qcol = int(art["qpos_index"])
    npzs = glob.glob(os.path.join(tree, "*", "trajectory_kinematic.npz")) or glob.glob(
        os.path.join(tree, "trajectory_kinematic.npz")
    )
    if not npzs:
        return {"tree": tree, "status": "no_npz"}
    z = dict(np.load(npzs[0]))
    q = z["qpos"]
    if key not in z:
        return {"tree": tree, "status": f"no_{key}"}
    sep = np.asarray(z[key], float).reshape(-1)
    note = ""
    if len(sep) != len(q):
        got = _retimed_source(tree, z, q, qcol, sep)
        if isinstance(got, dict):
            return got
        sep, note = got
    before = float(np.abs(q[:, qcol] - sep).max())
    dt = (
        float(1.0 / z["frequency"]) if "frequency" in z and float(z["frequency"]) > 0 else float(ti.get("ref_dt", 0.02))
    )
    if write:
        after = _write_aligned(tree, ti, ti_p, art, z, q, npzs, key, qcol, sep, before, dt)
    else:
        after = before
    return {
        "tree": tree,
        "status": ("aligned" if write else "would_align") + (f" {note}" if note else ""),
        "before": round(before, 6),
        "after": round(after, 8),
    }


def _retimed_source(tree: str, z: dict, q: np.ndarray, qcol: int, sep: np.ndarray) -> dict | tuple[np.ndarray, str]:
    """The source joint values on a retimed tree's timeline, with a note; or the skip record when the time map is
    missing or fails its check.

    Retimed trees: the sidecar keeps its pre-retiming length (missed upstream). The retiming source is
    `trajectory_kinematic.pre_rt.npz` (with f0 padding) and the time map is `retime_src_idx`; the source is padded the
    same way (first value held for the leading P frames) and resampled with the same map.
    Oracle: resampling the **smoothed column** with the same map must equal the current qpos column (otherwise
    untouched)."""
    pre = os.path.join(tree, "0", f"trajectory_kinematic{IK_SIDECAR_SFX}")
    if "retime_src_idx" not in z or not os.path.isfile(pre):
        return {"tree": tree, "status": f"len {len(sep)}!={len(q)} (no time map)"}
    zp = np.load(pre)
    si = np.asarray(z["retime_src_idx"], float).reshape(-1)
    qp = zp["qpos"]
    if len(si) != len(q):
        return {"tree": tree, "status": f"src_idx {len(si)}!={len(q)}"}
    chk = np.interp(np.clip(si, 0, len(qp) - 1), np.arange(len(qp)), qp[:, qcol])
    if float(np.abs(chk - q[:, qcol]).max()) > 1e-4:
        return {"tree": tree, "status": "time map check failed"}
    P = len(qp) - len(sep)
    if P < 0:
        return {"tree": tree, "status": f"pre_rt {len(qp)} < sidecar {len(sep)}"}
    rawp = np.concatenate([np.repeat(sep[:1], P), sep]) if P else sep
    sep = np.interp(np.clip(si, 0, len(rawp) - 1), np.arange(len(rawp)), rawp)
    note = f"retimed(P={P})"
    return sep, note


def _write_aligned(tree, ti, ti_p, art, z, q, npzs, key, qcol, sep, before, dt) -> float:
    """Write the source into the qpos joint column (qvel by backward differences, the sidecar on the tree timeline),
    mark task_info with rules.D25, and return the mismatch read back from the written npz."""
    q[:, qcol] = sep
    if "qvel" in z and z["qvel"].shape[0] >= 1:
        v = _qvel_col(tree, q.shape[1], qcol)
        qv = z["qvel"]
        n = min(len(qv), len(sep) - 1)
        qv[1 : n + 1, v] = (
            sep[1 : n + 1] - sep[:n]
        ) / dt  # backward difference (row 0 as is, the hand tracker's object-column convention)
        if n >= 1:
            qv[0, v] = qv[1, v] if len(qv) > 1 else 0.0
        z["qvel"] = qv
    z["qpos"] = q
    z[key] = sep.reshape(-1, 1).astype(
        np.asarray(z[key]).dtype
    )  # sidecar on the tree timeline too (both records equal in length and value)
    tmp = npzs[0] + ".tmp.npz"
    np.savez(tmp, **z)
    os.replace(tmp, npzs[0])
    art["max_abs_diff_vs_qpos"] = 0.0
    ti.setdefault("rules", {})["D25"] = {"version": VERSION, "max_abs_diff_before": round(before, 6), "source": key}
    t2 = ti_p + ".tmp"
    with open(t2, "w") as fh:
        json.dump(ti, fh, ensure_ascii=False, indent=1)
    os.replace(t2, ti_p)
    after = float(np.abs(np.load(npzs[0])["qpos"][:, qcol] - sep).max())
    return after


def _align_or_error(tree: str, write: bool) -> dict:
    """align, keeping a tree's exception as its error row: one tree's defect does not stop the rest."""
    try:
        return align(tree, write)
    except Exception as e:  # noqa: BLE001 — one tree's defect does not stop the rest (the reason is kept)
        return {"tree": tree, "status": f"error:{type(e).__name__}", "why": str(e)[:160]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--emb", default="", help="emb of this hand's arctic base layer (required without --roots)")
    ap.add_argument(
        "--roots",
        default="",
        help="comma-separated layer paths (relative to the data root, DEXENGINE_MAIN_ROOT in a worktree); only those. "
        "Default = this hand's base layer",
    )
    a = ap.parse_args()
    hand = RR.active_hand()  # no default; the audit location and the layers follow the same hand
    if not a.roots:
        if not a.emb:
            raise SystemExit("[D25] --emb or --roots is required; the layer to fix is never guessed")
        from dexengine.core.paths import layer_path  # noqa: PLC0415 — owner of layer paths

        a.roots = os.path.relpath(layer_path("base", "arctic", hand=RR.hand_dir(hand), emb=a.emb), DATA)
    print(f"[D25] hand={hand} roots={a.roots} write={a.write} data_root={DATA}", flush=True)
    trees = []
    for r in a.roots.split(","):
        p = os.path.join(DATA, r.strip())  # layers are relative to the data root (not the code tree)
        trees += [d for d in glob.glob(os.path.join(p, "*")) if os.path.isfile(os.path.join(d, "task_info.json"))]
        trees += [d for d in glob.glob(os.path.join(p, "*", "*")) if os.path.isfile(os.path.join(d, "task_info.json"))]
    rows = [_align_or_error(t, a.write) for t in sorted(set(trees))]
    import collections

    c = collections.Counter(r["status"].split(" ")[0] for r in rows)
    ok = [r for r in rows if "before" in r]
    bef = [r["before"] for r in ok]
    aft = [r["after"] for r in ok]
    # a dry run does not overwrite the write run's audit record (mode suffix, paths.mode_record)
    out = str(mode_record(RR.audit_out("arctic_artic_align.json", hand), None if a.write else "dryrun"))
    with open(out, "w") as fh:
        json.dump(
            {
                "generated": time.strftime("%Y-%m-%d %H:%M"),
                "version": VERSION,
                "write": a.write,
                "counts": dict(c),
                "before_max": (max(bef) if bef else None),
                "after_max": (max(aft) if aft else None),
                "before_gt_0.05": int(sum(1 for x in bef if x > 0.05)),
                "rows": rows[:5],
            },
            fh,
            ensure_ascii=False,
            indent=1,
        )
    print("states:", dict(c))
    print(
        f"max mismatch before {max(bef) if bef else 0:.4f} rad(>0.05: {sum(1 for x in bef if x > 0.05)}) → "
        f"max after {max(aft) if aft else 0:.2e} rad · {'WRITE' if a.write else 'dry-run'} · {out}"
    )
    # Done is judged by counts and errors: zero trees (layer not found) or per-tree exceptions give rc≠0, so the stage
    # driver never mistakes it for a run. Only decided states (aligned · already done · no articulation · articulation
    # count outside the rule) are normal; anything else (input npz, column or time map missing or mismatched,
    # exceptions, unknown new states) is an error.
    n_err = sum(v for k, v in c.items() if k not in _D25_OK and not k.startswith("artic_objects="))
    print(f"[D25_SUMMARY] hand={hand} trees={len(rows)} errors={n_err} counts={dict(c)}", flush=True)
    return 3 if (not rows or n_err) else 0


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.artic.arctic_align")
    raise SystemExit(main())
