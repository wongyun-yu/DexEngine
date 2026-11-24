#!/usr/bin/env python3
"""Contact sidecar (combo-independent canonical): one `precompute/<ds>/contact_mano_base/<clip>_<side>.npz` per base
tree × hand.

Why a separate file: the hand tracker's trajectory_kinematic `contact` (T,10) is an aggregate whose 10 columns are
equal (no per-finger information; see the cross_repair comment). TACO used MANO vertex (778) penetration, but
deliveries of external datasets have no MANO parameters, only **keypoints** (wrist · 5 fingertips). So here
  per-finger contact comes from the **nearest surface distance** between human fingertip keypoints
  (mano/…/trajectory_keypoints.npz qpos_finger_<side>, thumb…pinky) and the base tree object poses (qpos slots).
  Thresholds and smoothing share TACO's owner (`dexengine/data/contact/common.py` CONTACT_THRESHOLD_M · SMOOTH_*).
Schema = the same keys as the TACO sidecar (consumers gate/cross_repair read only `contact` and take any over axis=2).
Differences are explicit: regions=["fingertip"] (R=1; no whole_finger without MANO vertices) · penetration cannot be
  computed (0, penetration_available=False) · embodiment="mano_keypoints" · palm_dist relative to the wrist keypoint.
Articulated objects (ARCTIC): bottom/top part surfaces are sampled separately and top is rotated by the hinge angle
(qpos column) before measuring; no information is dropped.
Check field: mano_obj_mismatch_mm = object pose in the MANO file vs base tree object pose (median), an oracle that the
two worlds agree (shared objects use the original side's key).
Input fingerprint: each sidecar records an input fingerprint (object slots · pose witnesses · MANO source hash · code
version; owner `dexengine/data/contact/fingerprint.py`). Before writing it is compared with **the fingerprint computed
from this hand's base**:
  fresh (equal) → neither computed nor written (idempotent) · missing / unknown (old version without fingerprint,
  reported loudly) / stale → rebuilt.
  When another hand's **live base** expects other inputs, the generation rule
  (dexengine.data.contact.fingerprint.disagreement_kind) decides: if this hand is the newer generation it overwrites and
  records [SIDECAR_SUPERSEDE]; otherwise (mismatch within the same generation, etc.) it does not write, rc 4.
Done = counts match (fresh + written + same content == expected) · 0 conflicts. Ledger
data/<ds>/INTAKE/contact.sidecar[_all]_<hand>.json.
Usage: $(paths.py --what retarget-py) dexengine/data/contact/sidecar.py --dataset arctic --hand <hand> [--all-objects]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import capacity  # noqa: E402
from dexengine.core import paths as P  # noqa: E402
from dexengine.core.errors import InputMissingError
from dexengine.data.contact import (
    fingerprint as SF,  # noqa: E402  owner of input fingerprints, code version and palm coefficients
)
from dexengine.data.contact.sidecar_compute import VERSION, compute_side, stack_objects

dl = P.data_layout
ROOT = P.main_root()
STEP = "dexengine.data.contact.sidecar"  # capacity.NEEDS key (this module)
# SF is the only accessor for the value (callers use this name; see the data.rules D-MANO description)
palm_alpha = SF.palm_alpha


def _dest(layer, ds, **ax):
    return ROOT / dl.render(layer, ds, **ax)


def _surf_key(o: dict) -> str:
    """Surface dictionary key: obj_id plus whether articulated. Rigid uses surface_points.npz · articulated uses
    surface_parts.npz."""
    return f"{o['obj_id']}#artic" if o.get("articulation") else str(o["obj_id"])


def _axis_rot(axis, theta):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return Rot.from_rotvec(a[None, :] * np.asarray(theta, float)[:, None]).as_matrix()  # (T,3,3)


def _same_npz(out: Path, payload: dict, ignore: tuple = ()) -> bool:
    """Whether the existing sidecar has **the same content**: same key set + equal arrays (float arrays within 1e-9
    noise, NaN=NaN). Sidecars computed from two hands' bases have equal contact and distance arrays and differ only by
    float noise (≈3e-15 mm) in the check field mano_obj_mismatch_mm; that noise must not make the hand-independent
    layer be rewritten per hand (equal content is not rewritten). Fields in `ignore` are left out of the comparison, to
    tell 'only the fingerprint was added' from 'the contact content changed' in the regeneration report."""
    try:
        with np.load(out, allow_pickle=False) as old:
            if set(old.files) - set(ignore) != set(payload) - set(ignore):
                return False
            for k, v in payload.items():
                if k in ignore:
                    continue
                a, b = np.asarray(old[k]), np.asarray(v)
                if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
                    return False
                if a.dtype.kind == "f":
                    if not np.allclose(a, b, rtol=0.0, atol=1e-9, equal_nan=True):
                        return False
                elif not np.array_equal(a, b):
                    return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def _tmp_path(out: Path, pid: int) -> Path:
    """Intermediate write file: same directory (os.replace is atomic) and a different name per process. The four hands'
    stage 1 write the hand-independent layer together; with a fixed name one process's replace would take another's
    intermediate file (FileNotFoundError · BadZipFile)."""
    # the repo's intermediate file form (<name>.<pid>.tmp.npz, known to stale_tmp)
    return out.with_suffix(f".{pid}.tmp.npz")


def _save_npz_if_changed(out: Path, payload: dict) -> bool:
    """The **only write** of the hand-independent layer (contact_mano_*): equal content is not written (mtime
    untouched). Returns whether it wrote."""
    if out.is_file() and _same_npz(out, payload):
        return False
    tmp = _tmp_path(out, os.getpid())
    try:
        np.savez(tmp, **payload)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return True


#: **Record and version** fields left out of the 'content changed' report; they are not contact content. The real effect
#: of a version difference shows in dist · palm_dist.
#:   obj_role_at_build  role at build time (see stack_objects); a later D23 change alters only this
#:   version·source·palm_source·palm_alpha  old schema versions (old sidecars exist that differ only in these fields)
_RECORD_ONLY = ("obj_role_at_build", "version", "source", "palm_source", "palm_alpha")


def _regen(job: dict, tin, side: str, layer: str, mano: dict) -> dict:
    """Compute the sidecar content of (clip, side) from this hand's base (without the fingerprint fields)."""
    ti, kin = tin.ti, tin.kin
    objs = SF.objects_for(ti, side, layer)
    miss = [str(o["name"]) for o in objs if _surf_key(o) not in job["surf"]]
    if miss:  # surface assets missing; silently dropping objects would split fingerprint (objects) and content
        raise FileNotFoundError(f"{side}: surface samples missing {miss}")
    surf = [Path(job["surf"][_surf_key(o)]) for o in objs]
    if layer == SF.LAYER_ALL:
        pls = [compute_side(ti, kin, mano, side, o, f, job.get("alpha")) for o, f in zip(objs, surf, strict=True)]
        return stack_objects(pls, objs)
    return compute_side(ti, kin, mano, side, objs[0], surf[0])


def _new_result(tp: Path) -> dict:
    """Empty per-tree result: one list per outcome bucket, mismatches, not-live bases and the error."""
    res = {
        "clip": tp.name,
        "emb": tp.parent.name,
        "out": [],
        "same": [],
        "fresh": [],
        "new": [],
        "regen": [],
        "unknown": [],
        "unknown_changed": [],
        "conflict": [],
        "superseded": [],
        "zero": [],
        "mismatch_mm": {},
        "notlive": [],
        "err": None,
    }
    return res


def _mano_keypoints(tin, ds: str, tp: Path) -> dict:
    """This tree's raw MANO keypoints, in the base tree's frame."""
    mano = tin.mano_raw()
    if ds == "h2o":
        # D10-e'': base is yaw after the desk plane rotation; MANO keypoints must get the same rotation first
        # (single owner h2o_frame.tilt_transform) to share the object's frame
        from dexengine.data.readers import h2o_frame as HF

        mano = HF.apply_rigid_keypoints(mano, HF.tilt_transform(tp.name)["M"])
    return mano


def _write_side(job: dict, tin, side: str, layer: str, mano: dict, exp: dict, out: Path, v: str, why: str, res: dict):
    """Compute (clip, side), write it when the content changed, and file the side under its outcome buckets."""
    pl = _regen(job, tin, side, layer, mano)
    if layer == SF.LAYER_BASE and np.isfinite(pl["mano_obj_mismatch_mm"]):
        res["mismatch_mm"][side] = float(pl["mano_obj_mismatch_mm"])
    pl.update(SF.payload(exp))
    changed = out.is_file() and not _same_npz(out, pl, ignore=SF.FP_KEYS + _RECORD_ONLY)
    (res["out"] if _save_npz_if_changed(out, pl) else res["same"]).append(side)
    if v == "missing":
        res["new"].append(side)
    elif v == "unknown":  # old version without fingerprint; reported loudly (main prints counts and lists)
        res["unknown"].append(side)
        if changed:
            res["unknown_changed"].append(side)
    else:
        res["regen"].append(f"{side}: {why}")
    if not pl["contact"].any():
        res["zero"].append(side)


def do_tree(job: dict) -> dict:
    """Per (clip, side): compare fingerprints → done if fresh (no compute, no write) → compare inputs with other hands'
    live bases → compute and write.

    dexengine.data.contact.fingerprint alone owns the fingerprint and comparison rules; only the order lives here.
    """
    ds, tp = job["ds"], Path(job["tree"])
    layer = SF.layer_of(job.get("all_objects"))
    res = _new_result(tp)
    try:
        tin = SF.TreeInputs.load(ROOT, ds, tp, job["hand"])
        mano = None
        for side in SF.sides_for(tin.ti, layer):
            out = SF.sidecar_path(ROOT, ds, layer, tp.name, side, job["hand"])  # location = data_layout.sidecar_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            exp = tin.expected(layer, side)
            v, why = SF.compare(SF.read(out), exp) if out.is_file() else ("missing", "")
            if v == "fresh":  # same inputs; kept even if another hand's base produced the same content (0 writes)
                res["fresh"].append(side)
                continue
            # Shared layer of two hands: when another hand's live base expects **other inputs** for this file, it is not
            # silently overwritten. Per-hand locations (registered clips) are not shared, so they skip this comparison.
            dis = (
                []
                if SF.sidecar_is_own(ROOT, ds, layer, tp.name, side, job["hand"])
                else SF.live_disagreement(
                    ROOT, ds, tp.parent.name, tp.name, layer, side, exp, job["hand"], res["notlive"]
                )
            )
            if dis:
                rec = {
                    "side": side,
                    "verdict": v,
                    "why": why,
                    "mine": f"{job['hand']}@{exp['prov']['base_time']}",
                    "others": dis,
                }
                if not all(d["kind"] == SF.GEN_CHANGE for d in dis):
                    res["conflict"].append(rec)
                    continue
                res["superseded"].append(rec)
            if mano is None:
                mano = _mano_keypoints(tin, ds, tp)
            _write_side(job, tin, side, layer, mano, exp, out, v, why, res)
    except Exception as e:  # noqa: BLE001  per-tree pool worker: the error is the tree's result, the pool goes on
        res["err"] = f"{type(e).__name__}: {e}"
    return res


def _plan(a, layer: str, alpha) -> tuple[list, dict, dict, int]:
    """Count the base layer into jobs, surfaces and the expected count.

    Expected = one file per (clip, side) (SF.sides_for, the same function as the consumers).
    """
    want = None
    if a.clips_file:  # only clips D23 changed; also accepts the 3-column TSV (object_roles --out-changed)
        want = set()
        with open(a.clips_file, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.rstrip("\n")
                if ln.strip() and not ln.startswith("dataset\t"):
                    want.add(ln.split("\t")[-1].strip())
        if not want:
            raise InputMissingError(
                f"[contact] --clips-file {a.clips_file}: 0 clips — an empty file is not taken for nothing to do"
            )
    jobs, objs, pre_fail, expected = [], {}, {}, 0
    for emb in dl.EMBODIMENTS:
        d = _dest("base", a.dataset, hand=a.hand, emb=emb)
        if not d.is_dir():
            continue
        for tp in sorted(p for p in d.iterdir() if (p / "task_info.json").exists()):
            if want is not None and tp.name not in want:
                continue
            ti = json.loads((tp / "task_info.json").read_text())
            try:
                sides = SF.sides_for(ti, layer)
            except ValueError as e:
                pre_fail[tp.name] = str(e)
                continue
            for side in sides:
                # --all-objects: roles are **not looked at** (which hand used which object is the answer sought).
                # The same obj_id can be rigid or articulated, so surface keys are split.
                for o in SF.objects_for(ti, side, layer):
                    objs[_surf_key(o)] = o
            expected += len(sides)
            jobs.append(
                {"ds": a.dataset, "tree": str(tp), "hand": a.hand, "all_objects": a.all_objects, "alpha": alpha}
            )
    return jobs, objs, pre_fail, expected


_BUCKETS = ("out", "same", "fresh", "new", "regen", "unknown", "unknown_changed", "conflict", "superseded", "notlive")


def _report(ds: str, layer: str, hand: str, n: dict, conflicts: list, superseded: list) -> None:
    # marker: proof this run used the fingerprint rule; without it old code ran.
    print(
        f"[SIDECAR_FP] ds={ds} layer={layer} hand={hand} fresh={n['fresh']} new={n['new']} regen={n['regen']} "
        f"unknown={n['unknown']}(content changed {n['unknown_changed']}) superseded={n['superseded']} "
        f"conflict={len(conflicts)} notlive={n['notlive']} · written {n['out']} · same content {n['same']}",
        flush=True,
    )
    if n["unknown"]:
        print(
            f"[SIDECAR_FP] ★ {n['unknown']} old sidecars without fingerprint rebuilt from this hand's base — "
            f"contact content changed in {n['unknown_changed']} (ledger unknown_content_changed). "
            "Cases, roles and bakes of those trees were based on the old sidecars.",
            flush=True,
        )
    if (
        superseded
    ):  # generation replacement record (marker): count per hand and generation pair, names for the first few
        g = collections.Counter((d["other"], d["other_gen"], d["mine_gen"]) for c in superseded for d in c["others"])
        print(
            "[SIDECAR_SUPERSEDE] generation replaced "
            + " · ".join(f"{h}@{og} → {hand}@{mg} {k}" for (h, og, mg), k in sorted(g.items())),
            flush=True,
        )
    for c in superseded[:5]:
        oth = "; ".join(f"{d['other']}@{d['other_base_time']}({d['why'][:80]})" for d in c["others"])
        print(f"[SIDECAR_SUPERSEDE] {c['clip']}_{c['side']} mine={c['mine']} ← {oth}", flush=True)
    if not conflicts:
        return
    by = collections.Counter((d["other"], d["kind"], d["newer"]) for c in conflicts for d in c["others"])
    lab = {
        **SF.GEN_LABEL,
        "mine": "this hand is newer",
        "other": "that hand is newer",
        "same": "same version",
        "unknown": "version order unknown",
    }
    print(
        f"[SIDECAR_FP] ✗ conflicts {len(conflicts)} — another hand's live base expects **other inputs** (object list, "
        "poses, …) for this (clip, side). Not silently overwritten. Kind · which side is newer (version order): "
        + " · ".join(f"{h}: {lab.get(kd, kd)} · {lab.get(w, w)} {k}" for (h, kd, w), k in sorted(by.items())),
        file=sys.stderr,
        flush=True,
    )
    for c in conflicts[:4]:  # hand@time/generation
        oth = "; ".join(
            f"{d['other']}@{d['other_base_time']}/{d['other_gen'] or '?'} {d['newer']} ({d['why'][:90]})"
            for d in c["others"]
        )
        print(f"    {c['clip']}_{c['side']} {c['mine']}/{c['others'][0]['mine_gen'] or '?'} · {oth}", file=sys.stderr)
    print(
        "    remedy: same generation = defect (find the cause) · if that hand is a newer generation, rebuild "
        "this hand's base",
        file=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    """Command line: dataset, hand, workers, --all-objects, --limit, --clips-file."""
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--hand",
        required=True,
        help="hand id (required: whose base trees to read; no default). Spec directory from paths.hand_spec_dir",
    )
    ap.add_argument("--workers", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument(
        "--all-objects",
        action="store_true",
        help="before roles: measure per hand for **every object** in the clip → contact_all layer",
    )
    # Skipping happens only when fingerprints are equal (fresh); skipping because a file exists would be silent reuse
    # regardless of the input generation (hence no `--skip-existing`).
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--clips-file",
        default=None,
        help="only these clips (one per line; also the 3 columns of object_roles --out-changed), "
        "the follow-up for clips D23 changed",
    )
    return ap.parse_args()


def _run_jobs(a, jobs: list) -> list[dict]:
    """do_tree over the jobs in a capacity-sized pool (progress every 500 trees) -> results in (clip, emb) order."""
    results, t0 = [], time.time()
    with capacity.sized(STEP, len(jobs), a.workers) as cap, Pool(cap.workers, capacity.worker_init, cap.initargs) as p:
        for i, r in enumerate(p.imap_unordered(do_tree, jobs, chunksize=4), 1):
            results.append(r)
            if i % 500 == 0 or i == len(jobs):
                s = {k: sum(len(x[k]) for x in results) for k in ("out", "fresh", "conflict")}
                print(
                    f"  … {i}/{len(jobs)} · written {s['out']} · fingerprint equal {s['fresh']} · conflicts "
                    f"{s['conflict']} · "
                    f"failed {sum(1 for x in results if x['err'])} · {time.time() - t0:.0f}s",
                    flush=True,
                )
    return sorted(results, key=lambda r: (r["clip"], r["emb"]))


def _run_record(a, ds, layer, jobs, results, n, expected, produced, fails, zero, mism, conflicts, superseded):
    """The run record written to INTAKE: counts, conflicts, failures, zero-contact hands and completeness."""
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "dexengine/data/contact/sidecar.py",
        "version": VERSION,
        "dataset": ds,
        "hand": a.hand,
        "layer": layer,
        "fp_schema": SF.FP_SCHEMA,
        "fp_fresh": n["fresh"],
        "written": n["out"],
        "same_not_rewritten": n["same"],
        "new": n["new"],
        "regen_stale": n["regen"],
        "regen_unknown": n["unknown"],
        "regen_unknown_content_changed": n["unknown_changed"],
        "unknown_content_changed": sorted(f"{r['clip']}_{s}" for r in results for s in r["unknown_changed"])[:5000],
        "regen_stale_why": sorted(f"{r['clip']}_{w}" for r in results for w in r["regen"])[:2000],
        "n_conflict": len(conflicts),
        "conflicts": conflicts[:500],
        "n_superseded": len(superseded),
        "superseded": superseded[:2000],
        "n_trees": len(jobs),
        "expected": expected,
        "produced": produced,
        # field compatibility (pipeline_status reads produced+skipped_existing); always 0 without --skip-existing
        "skipped_existing": 0,
        "n_fail": len(fails),
        "fails": fails,
        "zero_contact_hands": sum(len(v) for v in zero.values()),
        "zero_contact": zero,
        "mano_obj_mismatch_gt20mm": len(mism),
        "mano_obj_mismatch": mism,
        "out_dir": os.path.relpath(_dest(layer, ds), ROOT),
        "all_objects": bool(a.all_objects),
        "complete": produced == expected and not fails and not conflicts,
    }
    return rec


def _finish(ds, n, produced, expected, fails, conflicts, rec, mism) -> int:
    """Summary line and return code: 4 on conflicts, 0 when complete, else 3."""
    print(
        f"[contact] {ds}: produced {produced} (fingerprint equal {n['fresh']} + written {n['out']} + same "
        f"content {n['same']}) / expected "
        f"{expected} · failed {len(fails)} · conflicts {len(conflicts)} · hands with zero contact "
        f"{rec['zero_contact_hands']} · "
        f"MANO↔base object mismatch >20mm {len(mism)} · {'COMPLETE' if rec['complete'] else 'INCOMPLETE'}"
    )
    for c, e in list(fails.items())[:4]:
        print(f"    {c}: {e[:200]}")
    if conflicts:
        return 4
    return 0 if rec["complete"] else 3


def main() -> int:
    a = _parse_args()
    ds = a.dataset
    a.hand = P.hand_spec_dir(a.hand)
    layer = SF.layer_of(a.all_objects)
    led = _dest("intake", ds) / f"base_{a.hand}.json"
    if not led.exists() or not json.loads(led.read_text()).get("complete"):
        print(f"[contact] {ds}: base ledger missing or incomplete ({led}) — run base_adapter first", file=sys.stderr)
        return 2
    jobs, objs, pre_fail, expected = _plan(a, layer, palm_alpha() if a.all_objects else None)
    if a.limit:
        jobs = jobs[: a.limit]
    from dexengine.data.contact import geom as CG

    t0 = time.time()
    surf = {oid: str(CG.ensure_surface(ds, o)) for oid, o in sorted(objs.items())}
    print(f"[contact] {ds}: trees {len(jobs)} · object surfaces {len(surf)} ({time.time() - t0:.0f}s)", flush=True)
    for j in jobs:
        j["surf"] = surf
    results = _run_jobs(a, jobs)
    n = {k: sum(len(r[k]) for r in results) for k in _BUCKETS}
    if a.limit:  # scoped run: expected counts only what this run's jobs expected
        expected = sum(n[k] for k in ("out", "same", "fresh", "conflict")) + sum(1 for r in results if r["err"])
    fails = {r["clip"]: r["err"] for r in results if r["err"]}
    fails.update(pre_fail)
    zero = {r["clip"]: r["zero"] for r in results if r["zero"]}
    mism = {r["clip"]: r["mismatch_mm"] for r in results if any(v > 20 for v in r["mismatch_mm"].values())}
    # expected count check: files left untouched because fingerprints matched are outputs too
    produced = n["out"] + n["same"] + n["fresh"]
    conflicts = [{"clip": r["clip"], "emb": r["emb"], **c} for r in results for c in r["conflict"]]
    superseded = [{"clip": r["clip"], "emb": r["emb"], **c} for r in results for c in r["superseded"]]
    rec = _run_record(a, ds, layer, jobs, results, n, expected, produced, fails, zero, mism, conflicts, superseded)
    # The run record is **per hand** (it holds hand and time), like base_<spec>.json · reference_check_<spec>.json in
    # the same INTAKE layer; a name without the hand would be overwritten by another hand's run.
    name = f"contact_sidecar_all_{a.hand}.json" if a.all_objects else f"contact_sidecar_{a.hand}.json"
    # scoped runs (--clips-file · --limit) do not overwrite the whole-dataset record (paths.clips_record)
    rec_f = P.clips_record(led.parent / name, bool(a.clips_file or a.limit))
    rec_f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    _report(ds, layer, a.hand, n, conflicts, superseded)
    return _finish(ds, n, produced, expected, fails, conflicts, rec, mism)


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.contact.sidecar")
    sys.exit(main())
