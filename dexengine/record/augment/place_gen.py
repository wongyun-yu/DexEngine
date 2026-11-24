#!/usr/bin/env python3
"""Placement augment: bakes the 'scene placement variant' clips of one clip into the shadow data root.

Meaning: the scene (all objects + the reference trajectory of the active hands) is moved rigidly on the desk plane
(planar shift dx, dy + yaw, pivot = the frame-0 xy centre of the active objects). Object-object and hand-object
relations stay, and so do the robot torso, the desk and idle hands (rest pose). The transform is the canonical bake's
(placement.retime_variants rigid_pl, the same function the combo pipeline's _placed_arrays uses), called as it is
(no second implementation).
Candidates (judged-cell contract v2): this clip's list in the judged cell bundle
  (--judged = paths.run_path("run_aug_cells", …)): the canonical cell (k 0 = pv00 · identity · control) + cells chosen
  from the region (policy.pick_cells). Only cells whose judge mode L row is stage pass become variants (absolute cell
  -> variant coordinates = policy.abs_to_variant). Reading and checks = augment.judged.
pv00 = the canonical clip as it is: tree arrays, arm IK and wrist caches copied byte for
  byte from the canonical root (augment.place_src canon_variant), not baked again — the arm solution the judge and the
  bake passed (a bake again from the judge seed re-solved the arm on another machine and could cross the desk).
Arm trajectory and time axis of k > 0: **the same path as the bake** (augment.bake): variant tree = the tree the
  canonical tree's IK saw (augment.bake base_kin: before retiming and cross repair), moved rigidly -> the bake chain
  steps after bake (body -> wrist -> stage_dp (seed = that cell's judge q) -> xr -> retime -> body -> wrist) through
  the same resident worker -> each variant retimed on its own (both hands and fingers on one clock).

Shadow root (a new directory used as DEXENGINE_MAIN_ROOT): the top level of the canonical data root as symlinks, only
the layer directories to overwrite as real directories + per-entry symlinks + variant entries. Nothing is written to
the canonical root.

Variant name = <clip>_pv<NN><TREE_SFX> (NN = cell list order k; no new `__`). pv00 = the canonical cell (identity,
  canonical copy). Home of the form = paths.variant_name (back to the source clip = paths.variant_source).

What a variant writes (clip V):
  this file      data/<ds>/trees/<combo>/<emb>/V/{task_info.json (+ aug_place record) · scene*.xml (symlink) ·
                 F0_REPAIR.json}
                 0/trajectory_kinematic.npz: rigid_pl of the tree the IK saw (qpos · qpos_rollout · qvel ·
                 contact_pos) · idle hands original
                 0/trajectory_ikrollout.npz: the transformed qpos_rollout (retiming writes it again)
                 precompute/<ds>/robots/<combo>/placement/V.adopted.json: copy of the original (sequence name only)
  augment.bake   0/body_positions.npz · wrist_poses_<side>/V.npz · arm_ik_poses_<side>_cont25/V.npz · retiming · xr
                 report, built by the bake steps · 0/contact_mano_<side>.npz: canonical contacts on the variant time
                 axis (augment.bake_post remap_contacts)
  pv00           the same files, copies of the canonical clip's (augment.place_src canon_pairs) · contacts as above
Filters (in order): judge drop (row stage · reason) -> on the desk at the last frame (object last frame xy outside the
  desk box − margin; off for aerial scenes, coords.desk_check_on alone; aerial support discs move with the same
  transform) -> bake steps (stage_dp refusal · stretch cap · cross repair failure · length; the same rc as the chain)
  -> gate speed (augment.speed_gate) -> copy check (pv00 only, augment.bake_post canon_check: sha256 = canonical) ->
  collision (augment.coll_gate: chain final_guard.check: arm-desk · arm-support · arm/hand <-> torso ·
  both arms crossing) -> placement (chain place_rule.baked_check) -> physics cap (policy.phys_set; passing ones over
  the cap = status spare) -> recorded with the reason in the variant record (variants.json) and left out of the list.
Markers: [AUG_GEN] … · end line [AUG_GEN] DONE ok=<n> reject=<n> total=<n>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from dexengine.core.arms import active_arm_name  # the recorded arm: DEXENGINE_ARM, set by the augment step
from dexengine.core.errors import InputMissingError
from dexengine.core.repo import REPO_ROOT
from dexengine.record.augment import place_src as APS
from dexengine.record.augment.place_src import MARK, Src, add_roster_rows, make_variant

HERE = Path(__file__).resolve().parent
WT = REPO_ROOT  # tree root
STEP = "dexengine.record.augment.place_gen"  # capacity.NEEDS key (the collision pre-check pool)

from dexengine.core import capacity  # noqa: E402
from dexengine.core import paths as P  # noqa: E402
from dexengine.record.augment import bake as AB  # noqa: E402
from dexengine.record.augment import bake_post as ABP  # noqa: E402
from dexengine.record.augment import bake_run as ABR  # noqa: E402
from dexengine.record.augment import judged as AJ  # noqa: E402
from dexengine.record.augment import policy as AP  # noqa: E402
from dexengine.record.augment import speed_gate as SPG  # noqa: E402  (gate speed rule of variants)
from dexengine.record.augment.scene import (
    init_shadow,
    shadow_dir,  # noqa: E402
)


def _load_mcv():
    from dexengine.placement import retime_variants as m

    return m


def after_bake(src: Src, shadow: Path, recs: list) -> None:
    """Variants past the bake (status baked; pv00 = the canonical copy): contact time axis (augment.bake_post
    remap_contacts) -> gate speed (augment.speed_gate) -> pv00 check (augment.bake_post canon_check)."""
    troot = Path(shadow) / src.trees_rel
    for r in recs:
        if r["status"] != "baked":
            continue
        r["contacts"] = ABP.remap_contacts(src, troot / r["name"])
        qs = {
            s: np.asarray(
                np.load(Path(shadow) / src.ik_rel[s] / f"{r['name']}.npz")["arm_joint_positions_all"], np.float64
            )
            for s in src.active
        }
        ok, why, r["speed"] = SPG.speed(src, troot / r["name"], qs)
        r.update({"status": "ok"} if ok else {"status": "reject", "why": "speed", "detail": why})
        # the one pv00 check (the copy check; replacing this call replaces the check)
        if r["k"] == 0 and r["status"] == "ok":
            ABP.canon_check(src, shadow, r)


def _commit(full: bool = False) -> str:
    return subprocess.run(
        ["git", "-C", str(WT), "rev-parse", *([] if full else ["--short"]), "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def _judge_rec(src: Src, r: dict, psi: float, ds) -> dict:
    """Record of a judge-dropped cell (no variant written)."""
    k = int(r["k"])
    return {
        "k": k,
        "name": P.tree_name(P.variant_name(src.tree_name.removesuffix(P.TREE_SFX), k)),
        "clip": src.tree_name.removesuffix(P.TREE_SFX),
        "dx": float(ds[0]),
        "dy": float(ds[1]),
        "yaw": float(psi),
        "r": float(np.hypot(*ds)),
        "status": "reject",
        "why": "judge",
        "detail": f"{r['stage']} {r['reason']}",
    }


def _work_line(a, src, checks: int) -> None:
    """The work line the parent's generation cap reads (augment.subrun.GenBound): checks x T / workers."""
    from dexengine.record.augment import subrun as SR

    jobs = a.jobs if a.jobs is not None else capacity.plan(STEP, checks).workers
    print(SR.work_line(checks, len(src.kin["qpos"]), jobs), flush=True)


def bake_step(a, src, recs) -> dict:
    """Bake step: augment.bake, or with --bake-from the plugin registered as hook ("bake", "bake_from")."""
    blog = a.out / f"bake_{src.tree_name}"
    blog.mkdir(parents=True, exist_ok=True)
    if os.environ.get("AUG_V4") == "1" and not a.bake_from:
        raise InputMissingError(f"{MARK} AUG_V4=1 without --bake-from — v4 reuses the v2 bake files")
    if a.bake_from:
        from dexengine.record import hooks

        hooks.load_plugins()
        adopt = hooks.get("bake", "bake_from")
        if adopt is None:
            raise SystemExit(
                f"{MARK} --bake-from needs a plugin that registers hook ('bake', 'bake_from') "
                f"(name it in {hooks.PLUGINS_ENV})"
            )
        return adopt(src, a.shadow, recs, a.bake_from, a.main, blog)
    return ABR.bake_all(src, a.shadow, recs, a.judged, a.jobs, blog)


def _parser():
    """Command line of the placement variant generator."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", required=True, type=Path, help="canonical data root (read only)")
    ap.add_argument("--shadow", required=True, type=Path, help="shadow data root (written)")
    ap.add_argument("--hand", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument(
        "--judged",
        required=True,
        type=Path,
        help="judged cell bundle folder (paths.run_path('run_aug_cells', …): region · cells · judged · ikc)",
    )
    ap.add_argument("--only", default="", help="comma list of k, for tests")
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--out", required=True, type=Path, help="folder for variants.json")
    ap.add_argument("--aerial", action="store_true", help="aerial clip (scene=aerial): support discs move too")
    ap.add_argument(
        "--bake-from",
        type=Path,
        default=None,
        help="v4 only: instead of baking, the bake files of this v2 shadow as they are (experiments.real_kit.v4.bake; "
        "required with AUG_V4=1)",
    )
    return ap


def _gate_and_write(src, a, recs, bake, J):
    """Collision gate of every variant (canonical first; one progress line per check), place gate, physics cap, v0
    check, variants_<clip>.json."""
    from concurrent.futures import ProcessPoolExecutor

    from dexengine.record.augment import coll_check as ACK
    from dexengine.record.augment import coll_gate as AC
    from dexengine.record.augment import subrun as SR

    cg = AC.gate_ctx(src)
    gate = [(cg, str(a.shadow), src.tree_name, "canon")] + [
        (cg, str(a.shadow), r["name"], "variant") for r in recs if r["status"] == "ok"
    ]
    with (
        capacity.sized(STEP, len(gate), a.jobs) as cap,
        ProcessPoolExecutor(cap.workers, initializer=capacity.worker_init, initargs=cap.initargs) as ex,
    ):
        gres = SR.pool_map(ex, ACK.pre_one, gate, MARK)
    coll_canon, gby = gres[0], {r["name"]: r for r in gres[1:]}
    recs = [AC.gate_apply(r, gby[r["name"]]) if r["name"] in gby else r for r in recs]
    print(
        f"{MARK} collision gate canonical {'pass' if coll_canon['ok'] else 'violation ' + coll_canon['why']} · "
        f"variants {sum(r['ok'] for r in gres[1:])}/{len(gres) - 1} pass",
        flush=True,
    )
    ABP.place_gate(src, a.shadow, recs, bake)
    phys = set(AP.fovx_phys_set(recs, AJ.fovx_offset(a.judged, a.clip)))  # off-screen clips start from fovx cells
    for r in recs:  # physics cap (front of the sample order); passing ones over it are spare (no rollout)
        if r["status"] == "ok" and r["k"] != 0 and r["name"] not in phys:
            r.update(status="spare", why="phys_cap", detail=f"> {AP.SAMPLE['phys_max']}")
    v0 = next((r for r in recs if r["k"] == 0), None)
    # no control: the record passes this clip with a shortage reason (canon_problem)
    if v0 is None or v0["status"] != "ok":
        w0 = v0 or {}
        print(f"{MARK} canonical cell dropped pv00 {w0.get('why', 'none')} {w0.get('detail', '')}: signal", flush=True)
    for r in recs:
        b = r.get("bake") or {}
        print(
            f"{MARK} pv{r['k']:02d} dx={r['dx']:+.3f} dy={r['dy']:+.3f} yaw={r['yaw']:+.1f} → {r['status']} "
            f"{r.get('why', '')} {r.get('detail', '')} T={b.get('T', '-')} stretch={b.get('stretch', '-')} "
            f"solver={b.get('solver', '-')} {b.get('sec', '-')}s",
            flush=True,
        )
    meta = {
        "clip": a.clip,
        "tree": src.tree_name,
        "hand": a.hand,
        "dataset": a.dataset,
        "emb": a.emb,
        "combo": src.combo,
        "active": src.active,
        "idle": src.idle,
        "seed": J["seed"],
        "judged": str(a.judged),
        "judge_fp": J["judge_fp"],
        "scale": J["scale"],
        "cell_rule": AP.CELL_RULE,
        "n_region": J["n_region"],
        "variants": recs,
        "coll_canon": coll_canon,
        "bake": bake,
        "commit": _commit(full=True),
    }
    (a.out / f"variants_{a.clip}.json").write_text(json.dumps(meta, indent=1))
    ok = sum(r["status"] == "ok" for r in recs)
    print(f"{MARK} DONE ok={ok} reject={len(recs) - ok} total={len(recs)}", flush=True)
    return 0


def main() -> int:
    ap = _parser()
    a = ap.parse_args()
    init_shadow(a.shadow, a.main)
    src = Src(a.main, a.hand, a.dataset, a.emb, a.clip, arm=active_arm_name(), aerial=a.aerial)
    mcv = _load_mcv()
    rp = AJ.refused_path(a.out, a.clip)
    rp.unlink(missing_ok=True)
    try:
        J = AJ.load(a.judged, a.clip, a.dataset, src)
    # a judge input defect concerns this clip alone: record and die (the caller holds the clip only)
    except SystemExit as e:
        a.out.mkdir(parents=True, exist_ok=True)
        rec = {"clip": a.clip, "kind": getattr(e, "kind", AP.HOLD_INPUT), "why": str(e)}  # kind = aug_policy.HOLD_*
        rp.write_text(json.dumps(rec, ensure_ascii=False))
        raise
    q_xy = src.object_xy_frame0()
    only = {int(x) for x in a.only.split(",") if x.strip()} if a.only else None
    a.out.mkdir(parents=True, exist_ok=True)
    print(
        f"{MARK} {src.tree_name} hand={a.hand} ds={a.dataset} emb={a.emb} active={src.active} idle={src.idle} "
        f"H={src.H} judge cells {len(J['rows'])} (passed {sum(bool(r['ok']) for r in J['rows'])}) · region "
        f"{J['n_region']} · "
        f"judge_fp={J['judge_fp']} · shadow {a.shadow} · tree={WT} "
        f"commit={_commit()}",
        flush=True,
    )
    base, xr, base_name = AB.base_kin(src)
    recs = []
    for r in J["rows"]:
        k = int(r["k"])
        if only is not None and k not in only:
            continue
        psi, ds = AP.abs_to_variant(r["theta_abs_deg"], r["d_abs"], J["canon"], q_xy)
        jd = {"stage": r["stage"], "reason": r["reason"], "met": r.get("met"), "cell": r["cell"]}
        # canonical cell = identity (control); residuals of the judge cell values (plan row rounding) recorded only
        if k == 0:
            jd["canon_residual"] = [round(psi, 6), round(float(np.hypot(*ds)), 7)]
            psi, ds = 0.0, (0.0, 0.0)
        if not r["ok"]:
            rec = _judge_rec(src, r, psi, ds)
        elif k == 0:  # the canonical cell = the canonical clip as it is (module doc)
            rec = APS.canon_variant(src, a.shadow)
        else:
            rec = make_variant(src, mcv, a.shadow, k, ds, psi, base, base_name)
        rec["judge"] = jd
        recs.append(rec)
    made = [r["name"] for r in recs if r["status"] in ("written", "baked")]  # baked = pv00 (canonical copy)
    todo = [r for r in recs if r["status"] == "written"]
    # the generation cap's work (bake and collision gate): the variants to bake + the canonical control + pv00 (copied,
    # gated only — counted as a whole check, so the cap errs loose)
    _work_line(a, src, len(todo) + 2)
    print(
        f"{MARK} list source {add_roster_rows(src, a.shadow, made)} · bake input {base_name} · xr={xr}",
        flush=True,
    )
    for s_ in ("right", "left"):  # cache layer shadows (the bake steps write variant entries)
        for rel in (src.ik_rel[s_], src.wrist_rel[s_]):
            if (a.main / rel).is_dir():
                shadow_dir(a.shadow, a.main, rel)
    bake = bake_step(a, src, recs)
    after_bake(src, a.shadow, recs)
    # collision gate (chain final_guard via augment.coll_gate): variants past bake and speed + the canonical control
    return _gate_and_write(src, a, recs, bake, J)


if __name__ == "__main__":
    from dexengine.robots import urdf_load

    urdf_load.install()
    sys.exit(main())
