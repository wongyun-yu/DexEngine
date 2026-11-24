#!/usr/bin/env python3
"""Placement augment: collision gates (pre and post). **No new rule**: the hard constraint the judge, bake and gate
share, final_guard.check (= collision.pen_cross_guard: desk surface · support · arm/palm/finger <-> torso · both arms
crossing, thresholds DESK_SURF_PEN_MM · TORSO_OVERLAP_MAX_MM · ARM_OVERLAP_MAX_MM), is called as it is. The result is
the first violation reason.

Pre gate (pre): the variant's arm IK cache (shadow precompute) + the shadow tree (fingers, desk box, supports) ->
  check. Two controls: canonical (canonical tree + canonical IK cache; passing is normal, otherwise that is a signal
  itself) · pv00 (identity; a byte copy of the canonical clip's tree arrays and arm IK, augment.place_src
  canon_variant, so it gets the canonical verdict; a different verdict is a signal).
Post gate (post): the executed rollout states (rollouts.npz: from the articulation columns arm joints = the IK cache
  arm_joint_names · fingers = hand_surface.finger_names, matched by name, missing dies) -> max per term with the same
  inputs (final_guard.values = collision.term_values; does not stop at the first violation) -> rule policy.post_ok
  (per term <= max(judge threshold, the same clip's original placement value + margin)). Canonical RL physics
  (canon_r2r3off) turns R1 · R2 (right <-> left hand/arm) · R3 (arm/hand <-> torso) collisions off, so their contact
  forces are always 0 and contact sensors or the force partner decomposition cannot see them; hence only the
  geometric check of the executed poses.
Hand-hand (recorded only): dexengine.placement.collision.cross_sets.CrossSetModel (1_cross) hand-hand pair minimum
(hh_min_mm, every
  HH_STRIDE frames).
Markers [AUG_COLL] … · end line [AUG_COLL] DONE <mode> …
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dexengine.core.arms import active_arm_name  # the recorded arm: DEXENGINE_ARM, set by the augment step
from dexengine.core.errors import InputMissingError
from dexengine.core.repo import REPO_ROOT
from dexengine.record.augment.coll_check import HH_STRIDE, MARK, post_one, pre_one

HERE = Path(__file__).resolve().parent
STEP = "dexengine.record.augment.coll_gate"  # capacity.NEEDS key (pre and post pools)
WT = REPO_ROOT  # tree root

from dexengine.core import capacity  # noqa: E402
from dexengine.core import paths as P  # noqa: E402
from dexengine.record.augment import place_src as APS


def gate_ctx(src) -> dict:
    return {
        "arm": src.arm,
        "hand": src.hand,
        "ds": src.ds,
        "combo": src.combo,
        "trees_rel": src.trees_rel,
        "ik_rel": src.ik_rel,
        "tree_name": src.tree_name,
        "zoff": float(src.adopted["robot_z_off"]),
        "main": str(src.main),
    }


def _ctx_meta(m: dict, shadow: Path, no_ref: bool = False) -> dict:
    """Post gate context from the generation record meta + the shadow's variant placement files alone (pv00 = copy of
    the original), even on machines without the canonical source tree. With no_ref (off-screen canonical cell dropped,
    --no-ref) pv00 was dropped and may have no placement file (augment.place_gen writes placement files only at the end
    of writing); then the passing variants' placement files are read: the value read, robot_z_off, is a copy of the
    original placement (src.adopted) in every variant (only sequence and aug_place change). Marker
    [AUG_FOVOUT_CANON]."""
    ds, combo, emb = m["dataset"], m["combo"], m["emb"]
    trees_rel, _w, ik_rel, pl_rel = APS.rel_layers(ds, combo, emb)
    v0 = next(v for v in m["variants"] if v["k"] == 0)["name"]
    f0 = shadow / pl_rel / f"{v0}{P.PLACEMENT_SFX}"
    if no_ref and not f0.is_file():
        # passing variants (in an extension run record, first-run passes = phys_done)
        ok = [v["name"] for v in m["variants"] if v["k"] != 0 and v["status"] in ("ok", "phys_done")]
        got = [f for n in ok if (f := shadow / pl_rel / f"{n}{P.PLACEMENT_SFX}").is_file()]
        if not got:
            raise InputMissingError(f"{MARK} --no-ref: no placement file for pv00 or any passing variant {f0}")
        zs = {float(json.loads(f.read_text())["robot_z_off"]) for f in got}
        if len(zs) != 1:
            raise SystemExit(
                f"{MARK} --no-ref: robot_z_off of the passing variant placements differ {sorted(zs)}: "
                "not copies of the original"
            )
        print(
            f"{MARK} [AUG_FOVOUT_CANON] pv00 placement file missing (dropped) -> robot_z_off {zs.pop()} of "
            f"{len(got)} passing variants (equal)",
            flush=True,
        )
        f0 = got[0]
    ad = json.loads(f0.read_text())
    return {
        "arm": active_arm_name(),
        "hand": m["hand"],
        "ds": ds,
        "combo": combo,
        "trees_rel": trees_rel,
        "ik_rel": ik_rel,
        "tree_name": m["tree"],
        "zoff": float(ad["robot_z_off"]),
    }


def cmd_pre(a) -> int:
    m = json.loads(a.variants.read_text())
    src = APS.Src(a.main, m["hand"], m["dataset"], m["emb"], m["clip"], arm=active_arm_name())
    c = gate_ctx(src)
    todo = [(c, str(a.shadow), src.tree_name, "canon")]
    todo += [(c, str(a.shadow), v["name"], "variant") for v in m["variants"] if v["status"] == "ok"]
    t0 = time.time()
    with (
        capacity.sized(STEP, len(todo), a.procs, share=_pool_post()) as cap,
        ProcessPoolExecutor(cap.workers, initializer=capacity.worker_init, initargs=cap.initargs) as ex,
    ):
        res = list(ex.map(pre_one, todo))
    canon = res[0]
    out = {
        "clip": m["clip"],
        "hand": m["hand"],
        "dataset": m["dataset"],
        "tree": str(WT),
        "commit": _commit(),
        "canon": canon,
        "variants": {r["name"]: r for r in res[1:]},
        "hh_stride": HH_STRIDE,
        "sec": round(time.time() - t0, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    nok = sum(r["ok"] for r in res[1:])
    pv0 = next((r for r in res[1:] if "_pv00" in r["name"]), None)
    print(
        f"{MARK} pre {m['hand']} {m['clip']}: canonical {'pass' if canon['ok'] else 'violation ' + canon['why']} · "
        f"pv00 {'—' if pv0 is None else ('pass' if pv0['ok'] else 'violation ' + pv0['why'])} · "
        f"variants passing {nok}/{len(res) - 1} · {out['sec']} s",
        flush=True,
    )
    return 0


def gate_apply(rec: dict, r: dict) -> dict:
    """Pre gate result r (pre_one) -> variant record: adds the coll field; a previously passing variant with a
    violation gets status reject · why collision · detail = the first violation reason. Shared by the generator
    (augment.place_gen) and apply."""
    rec = dict(rec, coll={k: r[k] for k in ("ok", "why", "hh_min_mm", "sec")})
    if rec["status"] == "ok" and not r["ok"]:
        rec.update(status="reject", why="collision", detail=r["why"])
    return rec


def cmd_apply(a) -> int:
    """Existing generation record (variants_<clip>.json) + pre gate result (pre_<clip>.json) -> gated generation
    record (same form)."""
    m = json.loads(a.variants.read_text())
    pre = json.loads(a.pre.read_text())
    if pre["clip"] != m["clip"]:
        raise SystemExit(f"{MARK} apply: clip {pre['clip']} != {m['clip']}")
    out = []
    for v in m["variants"]:
        if v["status"] == "ok" and v["name"] not in pre["variants"]:
            raise InputMissingError(f"{MARK} apply: no pre gate result for passing variant {v['name']}")
        out.append(gate_apply(v, pre["variants"][v["name"]]) if v["name"] in pre["variants"] else v)
    meta = dict(m, variants=out, coll_canon=pre["canon"], coll_commit=pre.get("commit", ""))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    n0 = sum(v["status"] == "ok" for v in m["variants"])
    n1 = sum(v["status"] == "ok" for v in out)
    print(f"{MARK} apply {m['hand']} {m['clip']}: passing {n0} -> {n1}", flush=True)
    return 0


def floors_mm() -> dict:
    """Judge threshold (mm) per post gate term; value owners are the collision and dexengine.placement.collision.torso
    constants (read only)."""
    from dexengine.placement.collision import core as SCL
    from dexengine.placement.collision import torso as TS

    return {
        "desk": SCL.DESK_SURF_PEN_MM,
        "support": SCL.DESK_SURF_PEN_MM,
        "torso": TS.TORSO_OVERLAP_MAX_MM,
        "cross": SCL.ARM_OVERLAP_MAX_MM,
    }


def apply_post_rule(kept: list[dict], floors: dict, ref_kept: list[dict] | None = None) -> dict:
    """Kept successful rollouts -> post_ok · why per rollout (policy.post_ok) · original placement reference. ref_kept =
    the record of the other run when the original placement (pv00) rollouts are in another run (IK repair rerun F)."""
    from dexengine.record.augment import policy as AP

    pv = [r["values"] for r in (ref_kept if ref_kept is not None else kept) if "_pv00" in r["name"]]

    ref = AP.post_ref(pv)
    for r in kept:
        r["post_ok"], r["why"] = AP.post_ok(r["values"], ref, floors)
    return {"ref": ref, "n_ref": len(pv)}


def own_rows(names: list[str], m: dict) -> list[int]:
    """Kept rollout names of a run -> only this clip's (generation record m) variant rows; cell batch sessions hold
    several clips in one run. A one-clip run gives all of them (the same result)."""
    mine = {P.tree_name(v["name"]) for v in m["variants"]}
    return [j for j, n in enumerate(names) if P.tree_name(n) in mine]


def _post_all(todo, names, rank, a, c, fl, ref_kept, m, meta, nk, t0):
    """Post-checks of the own rows: pv00 rows, then the first rank, then the rest; the result file and the summary
    line."""
    pv = [j for j in todo if "_pv00" in names[j]]
    rest = [j for j in todo if "_pv00" not in names[j]]
    ranks = sorted({rank[j] for j in rest})
    # the first rank is always measured in full (done below is empty)
    first = [j for j in rest if ranks and rank[j] == ranks[0]]
    with (
        capacity.sized(STEP, len(todo), a.procs, share=_pool_post()) as cap,
        ProcessPoolExecutor(cap.workers, initializer=capacity.worker_init, initargs=cap.initargs) as ex,
    ):
        # the original placement (reference) and the first rank go out at once: post_one measures each rollout alone and
        # only apply_post_rule uses the reference, so the result is the same (rows measured, order, verdict). Waiting
        # for the original placement was the post gate's wall time.
        both = list(ex.map(post_one, [(c, str(a.shadow), str(a.run), j) for j in pv + first]))
        res, head = both[: len(pv)], both[len(pv) :]
        rule = apply_post_rule(res, fl, ref_kept)
        # variants in rank order: a variant with one passing rollout is not measured further (salvaged = passes >= 1,
        # same result; --all measures all)
        done = set()
        for rk in ranks:
            batch = [j for j in rest if rank[j] == rk and (a.all or names[j] not in done)]
            got = head if rk == ranks[0] else list(ex.map(post_one, [(c, str(a.shadow), str(a.run), j) for j in batch]))
            apply_post_rule(got, fl, ref_kept if ref_kept is not None else res)
            res += got
            done |= {r["name"] for r in got if r["post_ok"]}
    out = {
        "clip": m["clip"],
        "hand": m["hand"],
        "run": str(a.run),
        "k_keep": meta.get("k_keep"),
        "tree": str(WT),
        "commit": _commit(),
        "floors_mm": fl,
        "rule": rule,
        "ref_from": str(a.ref or ""),
        "hh_stride": HH_STRIDE,
        "kept": res,
        "n_kept": nk,
        "pre_filter": str(a.pre or ""),
        "pre_sel": a.pre_sel if a.pre else "",
        "sec": round(time.time() - t0, 1),
    }
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    bad = [r for r in res if not r["post_ok"]]
    print(
        f"{MARK} post {m['hand']} {m['clip']} {a.run.name}: kept successful rollouts {len(res)}/{nk} · "
        f"caught by the post gate {len(bad)} · "
        f"original placement reference {rule['ref']} (pv00 rollouts {rule['n_ref']}) · {out['sec']} s",
        flush=True,
    )
    return 0


def cmd_post(a) -> int:
    m = json.loads(a.variants.read_text())
    c = dict(_ctx_meta(m, a.shadow, bool(getattr(a, "no_ref", False))), main=str(a.main))
    z = np.load(a.run / "rollouts.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    names = [str(z["clip_names"][int(ci)]) for ci in z["kept_clip"]]
    mine = {P.tree_name(v["name"]) for v in m["variants"]}
    todo = own_rows(names, m)
    nk = len(todo)
    # only rollouts of pre-passing variants (+ the pv00 control); variants caught by the pre gate drop out anyway (saves
    # time)
    if a.pre:
        pre = json.loads(a.pre.read_text())["variants"]
        if a.pre_sel == "pass":
            todo = [j for j in todo if "_pv00" in names[j] or pre.get(names[j], {}).get("ok")]
        # calibration: was the real execution of variants caught by the pre gate clean (is the pre gate too
        # conservative)?
        else:
            todo = [j for j in todo if names[j] in pre and not pre[names[j]]["ok"]]
    t0 = time.time()
    fl = floors_mm()
    ref_kept = json.loads(a.ref.read_text())["kept"] if a.ref else None
    rank = [int(r) for r in z["kept_rank"]]
    has_pv = any("_pv00" in str(n) and P.tree_name(str(n)) in mine for n in z["clip_names"])
    if getattr(
        a,
        "no_ref",
        # off-screen canonical cell dropped (chain FOVOUT_CANON_RULE): no reference = judge thresholds only
        False,
    ):
        if has_pv or ref_kept is not None:
            raise SystemExit(
                f"{MARK} --no-ref with original placement (pv00) rollouts or --ref present: one of the two"
            )
        print(
            f"{MARK} [AUG_FOVOUT_CANON] --no-ref · no original placement (pv00) reference -> judge thresholds only",
            flush=True,
        )
    elif ref_kept is None and not has_pv:
        raise InputMissingError(
            f"{MARK} this run has no original placement (pv00): give the record of another run of "
            "the same clip with --ref"
        )
    # empty with 0 successes -> no reference = judge thresholds only (policy)
    return _post_all(todo, names, rank, a, c, fl, ref_kept, m, meta, nk, t0)


def _commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _pool_post() -> int:
    from dexengine.record.augment import prefetch

    return prefetch.pool_sizes()["post"]


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--variants", required=True, type=Path)
    p.add_argument("--pre", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    for nm in ("pre", "post"):
        p = sub.add_parser(nm)
        p.add_argument("--variants", required=True, type=Path, help="gen/main/variants_<clip>.json")
        p.add_argument("--main", required=True, type=Path, help="canonical data root (trees · IK · placement)")
        p.add_argument("--shadow", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--procs", type=int, default=None, help="pool size (default: capacity.plan, lane share)")
        if nm == "post":
            p.add_argument("--run", required=True, type=Path, help="runs/<scene>/<mode> (rollouts.npz)")
            p.add_argument(
                "--pre",
                type=Path,
                default=None,
                help="pre gate result: when given, only passing variants + pv00 rollouts",
            )
            p.add_argument(
                "--all",
                action="store_true",
                help="also measure the remaining rollouts of passing variants (default: stop at the first pass)",
            )
            p.add_argument(
                "--ref",
                type=Path,
                default=None,
                help="another post gate result to read the original placement (pv00) rollout values from "
                "(IK repair rerun F takes A's)",
            )
            p.add_argument(
                "--no-ref",
                action="store_true",
                help="without an original placement (pv00) reference (off-screen canonical cell dropped, "
                "chain FOVOUT_CANON_RULE): judge thresholds only",
            )
            p.add_argument(
                "--pre-sel",
                choices=("pass", "fail"),
                default="pass",
                help="with --pre: pass = passing variants + pv00 · fail = caught variants only (calibration)",
            )
    a = ap.parse_args()
    if a.cmd == "apply":
        return cmd_apply(a)
    # the guard's assets (arm and hand meshes, hand stage) are read from the canonical data root; another root in the
    # env than --main dies
    have = os.environ.get("DEXENGINE_MAIN_ROOT")
    if have and Path(have).resolve() != a.main.resolve():
        raise SystemExit(f"{MARK} DEXENGINE_MAIN_ROOT={have} ≠ --main {a.main}")
    os.environ["DEXENGINE_MAIN_ROOT"] = str(a.main)
    return cmd_pre(a) if a.cmd == "pre" else cmd_post(a)


if __name__ == "__main__":
    from dexengine.robots import urdf_load

    urdf_load.install()
    raise SystemExit(main())
