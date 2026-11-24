#!/usr/bin/env python3
"""E5 kinematic replay baseline — build trees that use **only the hand side** of our pipeline, with every arm-side
selection removed.

Meaning: hands = the retargeting as is (base layer, URDF position limits only) · arm = per-frame IK (human wrist pose;
no
  penetration, speed or retiming checks) · robot = one fixed default pose (no placement search; robot height = the dz
  policy's canonical TACO layer) · desk = the source capture height (desk_z, the scene z rule coords.scene_z_offset) ·
  time = human time (no retiming) · objects = the human object trajectory (trajectory_kinematic.npz). Placement =
  offsets.clip_placement (clip move: start wrist centre → c_robot, rotation ψ_ds) — the scene (desk, objects, hands)
  moves, the robot pose stays. Not decided here: the tree layout (the bake dexengine.placement.bake.combo is called with
  that placement), support stamping (coords.stamp_scene), the idle arm (stage_dp's D16 rest pose — a constant pose
  without a judge), body FK and the wrist cache (canonical stage scripts). New here are only the placement record
  (adopted.json, in
  place of the judge) and per-frame IK (arm_ik's solvers; target z by the scene z rule; no penetration, speed or
  continuity rejection — unreachable frames keep their error).

Roots: --root = the shadow data root (DATA_SHADOW.json use e5_kin; base --main = the machine's canonical data root, read
  only). Layers kept as real folders: base (input copy, hard links from --inputs), trees, precompute/<ds>/robots/<combo>
  —
  they start empty (never mixed with our trees and caches of the same name). Everything else (object USD, assets …) is
  a
  symlink to the base.
Markers [KIN_PREP] …, per clip [KIN_PREP] <tree> OK ik_pos_mm(mean/max) …, last line [KIN_PREP] DONE ok=<n> fail=<n>
total=<n>.
  python -m experiments.policies.kin_data.prep --rows <tsv: ds emb clip> --root <shadow> --main <canonical root>
      --inputs <input root>       (Isaac python, PYTHONPATH = the code tree)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

from experiments.policies.kin_data.prep_clip import prep_clip, prep_variants
from experiments.policies.kin_data.prep_shadow import setup_root
from experiments.policies.kin_data.prep_tree import (
    MARK,
)


def _add_run_args(ap: argparse.ArgumentParser) -> None:
    """Inputs, roots, placement, output and the parallel/resume switches."""
    ap.add_argument("--rows", required=True, help="tsv with header ds · emb · clip")
    ap.add_argument("--root", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument(
        "--offsets",
        required=True,
        help="offsets table (experiments/policies/kin_data/offsets.py output JSON — c_robot · ψ_ds)",
    )
    ap.add_argument(
        "--placement",
        required=True,
        help="pipeline = our stage-1 placement (adopted), else kin_opt | clipside (superseded) — "
        "kin_offsets.PLACEMENTS",
    )
    ap.add_argument("--out", required=True, help="per-clip result jsonl")
    ap.add_argument(
        "--setup-only",
        action="store_true",
        help="build the shadow root and its written layers, then stop (once before a parallel run)",
    )
    ap.add_argument("--part", default="0/1", help="k/n — only the k-th share of rows (parallel)")
    ap.add_argument(
        "--done",
        default="",
        help="resume — result jsonl glob of earlier runs (ok · same offsets sha · placement · cache present = skipped)",
    )
    ap.add_argument(
        "--variants",
        default="",
        help="variant branch — kin_variants distribution json (the source trees must already be in this root)",
    )


def _add_source_args(ap: argparse.ArgumentParser) -> None:
    """Optional sources replacing the plain kinematic tree: our trees, seeds, reference arm and variants, SPIDER."""
    ap.add_argument(
        "--our-trees",
        action="store_true",
        help="source = copy of our tree from the base canonical root, no bake (retiming · f0 fix as is · "
        "default off = bake)",
    )
    ap.add_argument(
        "--arm-seed",
        default="",
        help="our arm seed folder <ds>/<tree>.npz (names · q · scene) — default none = plain IK",
    )
    ap.add_argument(
        "--ref-arm",
        action="store_true",
        help="arm = copy of our reference arm cache from the base canonical root (no IK · default off) — with "
        "--our-trees",
    )
    ap.add_argument(
        "--idle-from",
        default="",
        help="our idle-hand table (dataset · tree · our_idle) — default none = active-hand rule",
    )
    ap.add_argument(
        "--ref-var-root",
        default="",
        help="variant = copy of tree · placement · arm · wrist cache from this root (canonical root layout ·"
        " our variant reference picked by piece id; no make_variant · no IK · default none = build variants)"
        " — with --ref-arm · --variants (exact placement)",
    )
    ap.add_argument(
        "--spider-root",
        default="",
        help="third E5 source: SPIDER processed root (<root>/<ds>/wuji2/<emb>/<clip>/0/trajectory_mjwp.npz) — "
        "variant hand · object = SPIDER · arm = --spider-arm · command = SPIDER ctrl (default none = unchanged)",
    )
    ap.add_argument("--spider-arm", default="", help="required with --spider-root: a6 (SPIDER paper A.6 rollout IK)")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_run_args(ap)
    _add_source_args(ap)
    return ap.parse_args()


def _check_args(a: argparse.Namespace) -> None:
    """Validate the source switches against each other and load the idle-hand table into a._idle_tab."""
    a._idle_tab = {}
    if a.spider_root or a.spider_arm:
        from experiments.policies.spider_arm import arm_targets as SE  # noqa: PLC0415
        from experiments.policies.spider_arm import spider_load as SL  # noqa: PLC0415

        if not (a.spider_root and a.spider_arm):
            raise SystemExit(f"{MARK} --spider-root and --spider-arm must be given together")
        if a.spider_arm not in SL.ARM_MODES:
            raise SystemExit(f"{MARK} --spider-arm {a.spider_arm!r} unknown — {SL.ARM_MODES}")
        if not (a.ref_var_root and a.ref_arm and a.our_trees and a.variants):
            raise SystemExit(
                f"{MARK} --spider-root needs --our-trees --ref-arm --ref-var-root --variants (on top of "
                "the kin pair only)"
            )
        if not Path(a.spider_root).is_dir():
            raise SystemExit(f"{MARK} --spider-root {a.spider_root} missing")
        print(f"{MARK} [SPIDER] on — root {a.spider_root} · arm {a.spider_arm} · A.6 constants {SE.A6}", flush=True)
    if a.ref_arm and not a.our_trees:
        raise SystemExit(f"{MARK} --ref-arm needs --our-trees (the reference arm belongs to our tree)")
    if a.ref_var_root and not (a.ref_arm and a.variants):
        raise SystemExit(f"{MARK} --ref-var-root needs --ref-arm and --variants (variant reference replay)")
    if a.arm_seed and not a.idle_from:
        raise SystemExit(f"{MARK} --arm-seed needs --idle-from (pair scene table)")
    if a.idle_from:
        with open(a.idle_from) as fh:
            a._idle_tab = {
                (r["dataset"], r["tree"]): set(filter(None, r["our_idle"].split(",")))
                for r in csv.DictReader(fh, delimiter="\t")
            }


def _skip_done(a: argparse.Namespace, rows: list[dict]) -> list[dict]:
    """Resume: drop clips whose ok row in an earlier jsonl has the same move table, placement and IK cache."""
    import glob as _g  # noqa: PLC0415
    import hashlib  # noqa: PLC0415

    from dexengine.core import paths as P  # noqa: PLC0415

    osha = hashlib.sha256(Path(a.offsets).read_bytes()).hexdigest()[:16]
    prev = {}
    for f in sorted(_g.glob(a.done)):
        with open(f) as fh:
            for ln in fh:
                if ln.strip():
                    d = json.loads(ln)
                    if (
                        d.get("ok")
                        and d.get("offsets_sha256") == osha
                        and d.get("placement") == a.placement
                        and d.get("mode", "orig") == ("variants" if a.variants else "orig")
                    ):
                        prev[(d["ds"], d["emb"], d["clip"])] = d
    keep, skip = [], 0
    for r in rows:
        d = prev.get((r["ds"], r["emb"], r["clip"]))
        names = d.get("kept", []) if d and a.variants else ([d["tree"]] if d else [])
        if d and all((P.ik_dir(d["combo"], s_, r["ds"]) / f"{t_}.npz").is_file() for t_ in names for s_ in d["active"]):
            skip += 1
            continue
        keep.append(r)
    print(
        f"{MARK} resume: {skip} ok from earlier runs skipped · remaining {len(keep)}/{len(rows)} · "
        f"offsets table {osha}",
        flush=True,
    )
    return keep


def _print_variants_ok(res: dict) -> None:
    print(
        f"{MARK} {res['tree']} variants OK {res['s']} s · kept {res['n_kept']}/{res['need']} · draws "
        f"{res['n_draws']} · off the desk (kept) {res['desk_last_fail_kept']}",
        flush=True,
    )


def _print_clip_ok(res: dict) -> None:
    ikt = " · ".join(
        f"{s} pos {d['pos_mm_mean']:.1f}/{d['pos_mm_max']:.1f} mm rot {d['rot_deg_mean']:.1f}/{d['rot_deg_max']:.1f}°"
        for s, d in res["ik"].items()
    )
    print(f"{MARK} {res['tree']} OK {res['s']} s · {ikt} · idle {res['idle']}", flush=True)


def _fail_row(r: dict, e: Exception) -> dict:
    """Result row of a failed clip (printed); called inside the except block (the traceback is the active one)."""
    res = {
        "ds": r["ds"],
        "emb": r["emb"],
        "clip": r["clip"],
        "ok": False,
        "err": f"{type(e).__name__}: {e}"[:600],
        "tb": traceback.format_exc()[-6000:],
    }
    print(f"{MARK} {r['clip']} FAIL {res['err'][:300]}", flush=True)
    return res


def main() -> int:
    a = _parse_args()
    _check_args(a)
    if os.environ.get("DEXENGINE_MAIN_ROOT") != a.root:
        raise SystemExit(
            f"{MARK} DEXENGINE_MAIN_ROOT must equal --root ({os.environ.get('DEXENGINE_MAIN_ROOT')} ≠ {a.root})"
        )
    with open(a.rows) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    print(
        f"{MARK} [KIN_IK] naive — DLS from the previous solution + restarts (nearest solution within "
        "tolerance) + divergence guard · no velocity box · "
        f"jumps are measured only",
        flush=True,
    )
    setup_root(Path(a.root), Path(a.main), rows, variants=bool(a.variants))
    if a.setup_only:
        return 0
    k, n = (int(x) for x in a.part.split("/"))
    rows = rows[k::n]
    if a.done:
        rows = _skip_done(a, rows)
    dist = None
    if a.variants:
        from experiments.policies.kin_data import variants as KV  # noqa: PLC0415

        dist = KV.load(a.variants)
    ok = fail = 0
    t0 = time.time()
    with open(a.out, "a") as fh:
        for r in rows:
            t1 = time.time()
            try:
                if dist is not None:
                    res = prep_variants(a, r["ds"], r["emb"], r["clip"], dist)
                    res.update(ok=True, s=round(time.time() - t1, 1))
                    ok += 1
                    _print_variants_ok(res)
                else:
                    res = prep_clip(a, r["ds"], r["emb"], r["clip"])
                    res.update(ok=True, s=round(time.time() - t1, 1))
                    ok += 1
                    _print_clip_ok(res)
            except Exception as e:  # noqa: BLE001 — a failed clip is recorded and the next clip runs (the last line's fail count says it)
                fail += 1
                res = _fail_row(r, e)
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"{MARK} DONE ok={ok} fail={fail} total={len(rows)} · {time.time() - t0:.0f} s", flush=True)
    return 0 if fail == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
