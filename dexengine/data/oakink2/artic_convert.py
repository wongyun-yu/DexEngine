"""OakInk2 rigid parts → **articulated objects and rigid joints** (runs D1-g). The conversion rule lives here.

What
----
When a base-layer tree holds two parts of the same instance (official part tree `oakink2.parts`, D1-d) as separate
free-joint bodies, the two pose sequences are fitted as one joint (`oakink2.artic_fit.fit_pair`, thresholds rotation 5°
·
translation 1 cm); revolute or prismatic results are rewritten **in the form of arctic articulated trees**:
  scene.xml   inside the base body `<body name="<base>_top" pos quat>` + `<joint name="<base>_hinge|_slide" type pos
      axis range>` + the moving part geoms (free joint removed)
  qpos        the moving part's 7 free columns removed + 1 joint-value column θ(t); the new model is compiled and
      addresses are resolved by joint name
  npz         qpos·qpos_rollout rearranged, qvel recomputed (mujoco), `qpos_obj_arti_<side>` = θ(t)
  task_info   moving part removed from objects[] · `articulation`{type, joint, axis, pos, range, qpos_index, npz_key,
      parts, max_abs_diff_vs_qpos} on the base object ·
              qpos_layout rearranged + joint entry · nq · slot · part-group record = the remaining object itself ·
                  rules.D1g marker
Canonicalisation (default): the joint statement is copied as is, by asset id, from the canonical statement of the
**committed axis-hand asset spec** (artic_ranges.json) · range = spec (lo, hi) · θ = s·α + θ_off is set by
dexengine.data.artic.ranges.reexpress alone (including failures beyond 2π or outside the window). Outside the D1-g
thresholds in canonical coordinates → free · assets missing from the spec fail. --write calls the spec stage
(artic.spec_stage.ensure) before converting; tree statements (per_tree_record · per_tree_rest · per_tree_fixed) are
built in memory only. Marker [D1G_CANON] assets · restated · free · spec sha.
Verification (per tree): the new model's FK rebuilds the world pose of the moving part body and compares it with the
source pose: unless position ≤ 1 cm and rotation ≤ 5°, the tree is **not converted** (stays free, reason recorded).
Required follow-up: objects[] assignments change, so regenerate the canonical contact sidecar with
`python -m dexengine.data.contact.sidecar --dataset <ds>` (else contact.derive fails with FAIL_ASSIGN).
End: trees the spec judged unconvertible (row `exclude` reason word) go to the exclusion ledger
(data.rules.retarget_excluded_record) with rc 0 · exceptions, broken input and skips without a reason give rc 3 ·
marker [ARTIC_EXCLUDE] · count check convert_version.tally.
Idempotent: a tree whose rules.D1g is current (convert_version.D1G_VERSION) is skipped; another version (old, or
the
old per-tree version D1G_PER_TREE) is an error (merged trees no longer have the source part poses; rerun after
regenerating base (2)). No backup copies (the base layer can be regenerated with base_adapter).

Why
---
OakInk2 delivers articulated objects as two rigid bodies. Fits (D1-g) that come out revolute or prismatic are converted;
screw (not one joint), rigid (no joint needed) and free are not. arctic sources are already articulated and come in this
form; only oakink2 arrives as two rigid bodies.

Usage:
  python3 dexengine/data/oakink2/artic_convert.py --emb <emb> --limit 3  # dry run (no writes · audit = .dryrun.json)
  python3 dexengine/data/oakink2/artic_convert.py --emb <emb> --write    # all trees →
      records/oakink2_artic_convert.json + changed-clip TSV
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

from dexengine.core import paths as _P20  # noqa: E402 — owner of the clip-list entry (read_clip_list)
from dexengine.core.paths import (
    main_root as _data_root,  # noqa: E402 — owner of the data root (env DEXENGINE_MAIN_ROOT)
)
from dexengine.data import rules as RR  # noqa: E402 — owner of the hand axis
from dexengine.data.artic import (
    convert_version as D1G,  # noqa: E402 — owner of the version and the fate-count check (tally); dependency-free module
)
from dexengine.data.artic import ranges as AR  # noqa: E402 — owner of the canonical joint spec and reexpress
from dexengine.data.artic.convert_version import (
    D1G_PER_TREE,  # noqa: E402 — owner of the version (dependency-free module; the 2b cell counts it)
)
from dexengine.data.artic.convert_version import D1G_VERSION as VERSION  # noqa: E402
from dexengine.data.oakink2 import (
    fixed_merge as FM,  # noqa: E402 — owner of the rigid-joint output (merged mesh asset, pose assignment, oracle)
)
from dexengine.data.oakink2 import multi_instance as MI  # noqa: E402 — owner of scenes with two or more instances
from dexengine.data.oakink2.artic_decide import (  # noqa: E402,F401  (re-exported)
    _adr,
    _decide_one,
    _instance_decisions,
    _tree_joint,
    canon_joint,
    decide,
    per_tree_fixed,
    per_tree_record,
    per_tree_rest,
)
from dexengine.data.oakink2.artic_steps import HAND, _artic_step, _fixed_step
from dexengine.data.oakink2.artic_xml import (  # noqa: E402,F401  (re-exported)
    JOINT_ATTR,
    _child_pose,
    _eq,
    _find_body,
    _fixed_task_info,
    _move_children,
    _parent_of,
    _parts_record,
    _pose_attr,
    _rewrite_xml,
    _rewrite_xml_fixed,
    _self_group,
    repair_parts,
)
from dexengine.data.objects import (
    scene_parts as SP,  # noqa: E402 — scene convex parts = current decomposition (checked before rewriting in place)
)


def spec() -> tuple[dict, str]:
    """Committed axis-hand articulation spec (artic_ranges.json, the file the bake objects.usd_bake and the surface
    points
    dexengine.data.artic.surface read): asset table (objects) · first 12 chars of the file content sha · rigid-joint
    asset table (fixed section)."""
    raw = RR.axis_output(RR.ARTIC_SPEC).read_bytes()
    d = json.loads(raw)
    return d["objects"], hashlib.sha256(raw).hexdigest()[:12], d.get(FM.SECTION) or {}


def convert_tree(tree, write=False, canon=None):
    """canon = (spec asset table, spec sha) for canonicalisation (default) · None = this tree's own fit (its first
    frame is joint 0; no CLI flag selects it)."""
    ti_p = os.path.join(tree, "task_info.json")
    with open(ti_p) as fh:
        ti = json.load(fh)
    if r := D1G.pre_row(
        tree, ti
    ):  # rows decided without measuring (already converted · old version · inherited exclusion)
        return r
    objs = ti.get("objects") or []
    clip = os.path.basename(tree)
    npz_p = glob.glob(os.path.join(tree, "*", "trajectory_kinematic.npz"))[0]
    z = dict(np.load(npz_p))
    q = z["qpos"]
    dec = decide(objs, q, ti["dataset_name"], canon, lever := D1G.lever_for(ti))  # lever pin request
    plans = dec.get("multi", [dec])
    D1G.lever_used(clip, lever, plans)
    if "status" in dec:
        return {"clip": clip, **dec}
    if "multi" in dec:  # marker
        print(MI.mark(clip, dec), flush=True)
    SP.verify_dir(tree)  # convex parts of the rewritten scene = current decomposition; otherwise base is stale
    eq_p = os.path.join(tree, "scene_eq.xml")
    with open(os.path.join(tree, "scene.xml")) as fh:
        st = {"xml": fh.read(), "eq": None, "z": z, "ti": ti}
    if os.path.isfile(eq_p):
        with open(eq_p) as fh:
            st["eq"] = fh.read()
    done = []
    for p in plans:  # order = MI.instances (joint last)
        r = (_fixed_step if "fixed" in p else _artic_step)(clip, st, q, p, write, canon)
        if "static_artic" in p:  # static-articulation marker · tree record rules.D1g.static_artic
            r["static_artic"] = p["static_artic"]
            print(D1G.static_mark(clip, p, r["status"]), flush=True)
        if r["status"] not in ("converted", "would_convert"):
            return r
        done.append(r)
    res = MI.merged(done)
    if not write:
        return res
    ti.setdefault("rules", {})["D1g"] = {"version": VERSION if canon is not None else D1G_PER_TREE, **MI.stamp(res)}
    tmp = npz_p + ".tmp.npz"  # atomic write
    np.savez(tmp, **z)
    os.replace(tmp, npz_p)
    texts = [
        (os.path.join(tree, "scene.xml"), st["xml"]),
        (eq_p, st["eq"]),
        (os.path.join(tree, "task_info.json"), json.dumps(ti, ensure_ascii=False, indent=1)),
    ]
    for path, text in texts:
        if text is not None:
            with open(path + ".tmp", "w") as fh:
                fh.write(text)
            os.replace(path + ".tmp", path)
    return res


def main() -> int:
    a = _parse_args()
    if a.write and not a.repair_parts:
        from dexengine.data.artic import (
            spec_stage as ASS,  # noqa: PLC0415 — owner of the spec update and asset bake (2b calls it first)
        )

        ASS.ensure("oakink2")
    canon = spec()
    root, _cl, trees = _trees(a)
    rows, n, n_no_parts = _convert_all(trees, a, canon)
    import collections

    c = collections.Counter(r["status"].split(":")[0].split("=")[0] for r in rows)
    out = _write_audit(a, _cl, rows)
    print("status:", dict(c))
    print("mode:", "WRITE" if a.write else "dry-run", "·", out)
    for r in rows[:6]:
        print("  ", {k: r[k] for k in r if k in ("clip", "status", "kind", "fk_pos_max_m", "fk_rot_max_deg", "range")})
    seen, n_trees, tl = _tally(a, rows, trees, n, n_no_parts)
    n_err = len(tl["errors"])
    print(
        f"[D1G_SUMMARY] hand={RR.active_hand()} root={root} trees={n_trees} errors={n_err} counts={dict(c)}", flush=True
    )
    for r in tl["errors"][:8]:  # unexpected errors
        print(f"[D1G_ERROR] {r['clip']}: {r['status']} {r.get('why', '')}", flush=True)
    _report_exclusions(a, tl, seen, n_err, n_no_parts, n_trees)
    if canon is not None:
        _canon_summary(rows, canon)
    return tl["rc"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OakInk2 part pairs → articulated objects (D1-g)")
    ap.add_argument("--emb", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clips", default="")
    ap.add_argument("--write", action="store_true")
    ap.add_argument(
        "--repair-parts", action="store_true", help="only repair parts paths and child_pose of converted trees"
    )
    a = ap.parse_args()
    return a


def _trees(a) -> tuple[str, list, list]:
    """This hand's base layer of the embodiment and its trees (all, or the --clips list)."""
    root = os.path.join(str(_data_root()), "data", "oakink2", "base", HAND, a.emb)
    # names from outside go through the single entry point (P.read_clip_list)
    _cl = [c for c in a.clips.split(",") if c]
    trees = (
        sorted(glob.glob(os.path.join(root, "*")))
        if not _cl
        else [os.path.join(root, c) for c in _P20.read_clip_list(_cl, dataset="oakink2", quiet=True)]
    )
    return root, _cl, trees


def _convert_all(trees, a, canon) -> tuple[list, int, int]:
    """Convert (or repair) every tree: result rows, converted count and trees that are not a target (no instance
    with two or more parts). A tree's exception becomes its error row; --limit stops after that many conversions."""
    rows = []
    n = n_no_parts = 0
    for t in trees:
        if not os.path.isfile(os.path.join(t, "task_info.json")):
            continue
        try:
            r = repair_parts(t, write=a.write) if a.repair_parts else convert_tree(t, write=a.write, canon=canon)
        except Exception as e:  # noqa: BLE001 — failures are kept with their reason
            r = {"clip": os.path.basename(t), "status": f"error:{type(e).__name__}:{str(e)[:80]}"}
        if r["status"] == "skip_no_parts":  # no instance with two or more parts: not a target of this rule
            n_no_parts += 1
            continue
        rows.append(r)
        n += r["status"] in ("converted", "would_convert")
        if a.limit and n >= a.limit:
            break
    return rows, n, n_no_parts


def _write_audit(a, _cl, rows) -> str:
    """Audit json of the run (and the converted-instance tsv of a write run); returns the json path."""
    # Audit outputs of the axis hand go to the shared location; other hands (stage 1 step 2b) to `<name>__<hand>`
    # (data.rules.audit_out). A dry run does not overwrite the write run's audit (mode suffix, paths.mode_record);
    # a scoped run (--clips) does not overwrite the full-dataset record (scope suffix, paths.clips_record).
    rec = _P20.clips_record(RR.audit_out("oakink2_artic_convert.json", RR.active_hand()), bool(_cl))
    out = str(_P20.mode_record(rec, None if a.write else "dryrun"))
    with open(out, "w") as fh:
        json.dump(
            {"generated": time.strftime("%Y-%m-%d %H:%M"), "version": VERSION, "write": a.write, "rows": rows},
            fh,
            ensure_ascii=False,
            indent=1,
        )
    if a.write:
        tsv = _P20.clips_record(RR.audit_out("oakink2_artic_converted.tsv", RR.active_hand()), bool(_cl))
        with open(tsv, "w") as fh:
            # one line per instance
            cols = ("clip", "kind", "base", "moving", "fk_pos_max_m", "fk_rot_max_deg")
            fh.write(
                "\t".join(cols)
                + "\n"
                + "".join(
                    "\t".join(str(r[k]) for k in cols) + "\n"
                    for r in MI.instance_rows(rows)
                    if r["status"] == "converted"
                )
            )
    return out


def _tally(a, rows, trees, n, n_no_parts) -> tuple[list, int, dict]:
    """Trees seen, their count and the run tally (repair mode has its own vocabulary)."""
    # Done = counts and errors: zero trees (layer not found) or per-tree exceptions give rc≠0, so the stage 1 driver
    # does not take it as run.
    seen = [os.path.basename(t) for t in trees if os.path.isfile(os.path.join(t, "task_info.json"))]
    n_trees = len(seen)
    if a.repair_parts:  # repair mode has its own status vocabulary; only exceptions are errors
        tl = {
            "done": n,
            "excluded": [],
            "by_reason": {},
            "ok": True,
            "errors": [r for r in rows if r["status"].startswith("error")],
        }
        tl["rc"] = 3 if (n_trees == 0 or tl["errors"]) else 0
    else:
        tl = D1G.tally(rows, n_trees, n_no_parts, bool(a.limit))
    return seen, n_trees, tl


def _report_exclusions(a, tl, seen, n_err, n_no_parts, n_trees) -> None:
    """Explicit exclusions go to the exclusion ledger on a write run; marker lines of inherited and new exclusions."""
    led = "dry-run"
    if (
        a.write and not a.repair_parts
        # explicit exclusions go to the exclusion ledger with their reason (only the lines of clips seen this run
        # change)
    ):
        led = str(
            RR.retarget_excluded_record(
                "oakink2", D1G.EXCLUDE_SRC, {r["clip"]: (r["exclude"], r.get("why", "")) for r in tl["excluded"]}, seen
            )["path"]
        )
    print(f"[D1G_INHERIT] inherited exclusions {sum(bool(r.get('inherited')) for r in tl['excluded'])}", flush=True)
    print(
        f"[ARTIC_EXCLUDE] n={len(tl['excluded'])} by_reason={json.dumps(tl['by_reason'], ensure_ascii=False)} · "
        f"converted {tl['done']} · errors {n_err} · not a target {n_no_parts} · input {n_trees} · "
        f"total {'match' if tl['ok'] else 'mismatch'} · ledger {led}",
        flush=True,
    )


def _canon_summary(rows, canon) -> None:
    """[D1G_CANON] marker of a canonicalised run and the spec-stage marker of its converted instances."""
    done = [r for r in MI.instance_rows(rows) if r["status"] in ("converted", "would_convert")]
    print(
        f"[D1G_CANON] assets {len({r['asset'] for r in done})} · restated {sum(r['restated'] for r in done)} · "
        f"rigid joints {sum(r.get('kind') == 'fixed' for r in done)} · static articulated (out of band) "
        f"{sum('skip' not in r.get('static_artic', {'skip': 0}) for r in done)}"
        f"({sum('skip' in r.get('static_artic', {}) for r in done)}) · "
        f"free {sum(r['status'] == 'skip_canon_misfit' for r in rows)} · spec sha {canon[1]}",
        flush=True,
    )
    AR.mark(done)


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.oakink2.artic_convert")
    raise SystemExit(main())
