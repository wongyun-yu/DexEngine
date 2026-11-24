#!/usr/bin/env python3
"""Base-layer adapter: delivered retargeting clips (retarget layer) -> stage-1 trees (base layer), for every dataset.

    data/<ds>/retarget/<hand>/<emb>/<clip>/          ->   data/<ds>/base/<hand>/<emb>/<clip>[_segNN]/
      task_info.json · scene.xml · scene_eq.xml            task_info.json (retarget keys + the keys below) · scene files
      0/trajectory_kinematic.npz · trajectory_ikrollout    0/trajectory_kinematic.npz · trajectory_ikrollout.npz ·
      0/retarget_diagnostics.{json,npz}                    retarget_diagnostics.npz (segment slice) ·
          0/body_positions.npz

What it does:
  1 Duplicate objects: the same mesh in two bodies at the same pose -> the second body is dropped from the scene and
    qpos and the object becomes shared (side "both"); nq drops by 7, the FK of the remaining body is unchanged.
  2 Frame alignment (yaw): every dataset is rotated to the fixed reference hand direction (align.REF_DEG):
    psi = ref - dir(ds), dir = circular mean of the MANO wrist-to-fingertip xy angle (INTAKE/hand_dir.json; per clip for
    PER_CLIP_YAW datasets). The virtual wrist (slide xyz + hinge XYZ), the object free joints and the extra objects get
    Rz(psi); qvel is recomputed with mujoco; the FK palm pose must equal Rz * (old palm pose) (oracle).
  3 Roles pass through unchanged; the hand-object assignment (cases.hand_object_assign) decides them per tree.
  4 Articulated part pairs (same object number, oakink2 o02_<n>_<part>) form an articulation_group whose relative
    motion is fitted to a screw axis (rigid / revolute / prismatic / screw / free).
  5 Marker quality.finger_jump_frames (frames with |dq| > JUMP_RAD).
  7 Object convex parts are re-derived from the object's current decomposition (objects.scene_parts, marker
    [SCENE_PARTS], record provenance.scene_parts); a tree whose scene differs from it is stale (tree_stale_reason).
Done = counts match: sum of segments == produced + existing (skipped). Ledger data/<ds>/INTAKE/base_<hand>.json; a
failed clip stays in the ledger and gives rc 3.

    $(paths.py --what retarget-py) dexengine/data/base_adapter.py --dataset arctic [--hand <hand>] [--workers 8] \
        [--skip-existing] [--clips a,b] [--limit N]

Shared names, geometry, supplements and the staleness checks live in ``dexengine.data.base_layer``; this module is the
CLI and the per-clip adapter and re-exports their names.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

from dexengine.core import capacity
from dexengine.core import paths as P
from dexengine.core.errors import input_missing
from dexengine.data.base_layer.clip import adapt_clip
from dexengine.data.base_layer.hand_dir import hand_dir_stats, ref_dir_oracle, yaw_for

STEP = "dexengine.data.base_adapter"  # capacity.NEEDS key (this module)
HELP = (
    'base adapter -- delivered retargeting clips (retarget layer) -> stage-1 trees (base layer), every dataset (v"c").'
)

from dexengine.data.base_layer.common import (
    ADAPTER_VERSION,
    _dest,
    data_root,
    dl,
)
from dexengine.data.base_layer.geom import (
    segments_for,
)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=HELP.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--hand",
        required=True,
        help="hand id (HAND_REGISTRY key, no default); paths.hand_spec_dir gives the spec directory",
    )
    ap.add_argument("--workers", type=int, default=None, help="pool size (default: capacity.plan — CPUs and memory)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--clips", default="", help="comma list (for tests)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recompute-dir", action="store_true")
    return ap


def main() -> int:
    a = _parser().parse_args()
    ds, hand = a.dataset, P.hand_spec_dir(a.hand)
    idir = _dest("intake", ds)
    sup = (
        json.loads((idir / "support_audit.json").read_text())["clips"] if (idir / "support_audit.json").exists() else {}
    )
    segj = (
        json.loads((idir / "segment_audit.json").read_text())["clips"] if (idir / "segment_audit.json").exists() else {}
    )
    if not sup or not segj:
        from dexengine.data.read_raw import derive  # noqa: PLC0415  (only on this refusal)

        print(
            f"[base] {ds}: INTAKE/support_audit.json and segment_audit.json are required ({idir}) -- make them with "
            f"`{derive.command(ds)}`",
            file=sys.stderr,
        )
        return 2
    if a.recompute_dir:
        hand_dir_stats(ds, force=True)
    yaw = yaw_for(ds)
    ref_dir_oracle()
    print(
        f"[base] {ds}: yaw {yaw[0]:+.1f}° (hand direction {yaw[1]['mean_deg']:.1f}°±{yaw[1]['spread_deg']:.0f} "
        f"→ taco {yaw[2]['mean_deg']:.1f}°)",
        flush=True,
    )
    jobs, excluded = _clip_jobs(ds, hand, sup, segj, a.skip_existing, yaw)
    missing = [j["clip"] for j in jobs if j["sup"] is None or j["seg"] is None]
    if missing:
        print(f"[base] {ds}: {len(missing)} clips missing from the audit ledgers (e.g. {missing[:3]})", file=sys.stderr)
        return 2
    if a.clips:
        keep = set(P.read_clip_list(a.clips.split(","), dataset=ds, label="clips", quiet=True))
        jobs = [j for j in jobs if j["clip"] in keep]
    if a.limit:
        jobs = jobs[: a.limit]
    expected = sum(len(segments_for(j["seg"], int(j["seg"]["T"]))) for j in jobs)
    print(
        f"[base] {ds}/{hand}: input clips {len(jobs)} → expected outputs {expected} (pool: the {capacity.TAG} line)",
        flush=True,
    )
    results = _adapt_all(jobs, a.workers)
    rec = _base_record(ds, hand, yaw, jobs, excluded, expected, results)
    # a scoped run (--clips) does not overwrite the dataset ledger (the live-base verdict; the hand pages read it)
    # (paths.clips_record)
    idir.mkdir(parents=True, exist_ok=True)
    P.clips_record(idir / f"base_{hand}.json", bool(a.clips)).write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    _print_summary(ds, hand, rec, jobs, excluded)
    return 0 if rec["complete"] else 3


def _clip_jobs(ds: str, hand: str, sup: dict, segj: dict, skip_existing: bool, yaw) -> tuple[list[dict], dict]:
    """(one adapt_clip job per delivered clip, {clip: reason} for clips without a kinematic trajectory)."""
    jobs = []
    excluded = {}
    for emb in dl.EMBODIMENTS:
        d = _dest("retarget", ds, hand=hand, emb=emb)
        if not d.is_dir():
            continue
        for c in sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "task_info.json").exists()):
            if not (d / c / "0/trajectory_kinematic.npz").exists():
                excluded[c] = "0/trajectory_kinematic.npz missing (clip excluded from retargeting)"
                continue
            jobs.append(
                {
                    "ds": ds,
                    "hand": hand,
                    "emb": emb,
                    "clip": c,
                    "sup": sup.get(c),
                    "seg": segj.get(c),
                    "skip_existing": skip_existing,
                    "yaw": yaw,
                }
            )
    if excluded:
        print(
            f"[base] {ds}: {len(excluded)} clips without retargeting output excluded (recorded in the "
            f"ledger): {list(excluded)[:3]}"
        )
    return jobs, excluded


def _adapt_all(jobs: list[dict], workers: int | None) -> list[dict]:
    """Run adapt_clip over the jobs in a process pool sized by capacity (`workers` = an explicit option), printing
    progress every 100 clips."""
    t0 = time.time()
    results = []
    with capacity.sized(STEP, len(jobs), workers) as cap, Pool(cap.workers, capacity.worker_init, cap.initargs) as pool:
        for i, r in enumerate(pool.imap_unordered(adapt_clip, jobs, chunksize=1), 1):
            results.append(r)
            if i % 100 == 0 or i == len(jobs):
                print(
                    f"  … {i}/{len(jobs)} · built {sum(len(x['out']) for x in results)} · skip "
                    f"{sum(len(x['skipped']) for x in results)} · failed {sum(1 for x in results if x['err'])} "
                    f"· {time.time() - t0:.0f}s",
                    flush=True,
                )
    return results


def _base_record(ds: str, hand: str, yaw, jobs: list, excluded: dict, expected: int, results: list) -> dict:
    """The dataset's base ledger (INTAKE/base_<hand>.json): counts, failures and the completeness verdict."""
    produced = sum(len(r["out"]) for r in results)
    skipped = sum(len(r["skipped"]) for r in results)
    fails = {
        r["clip"]: r["err"] for r in sorted(results, key=lambda r: r["clip"]) if r["err"]
    }  # clip order, not finish
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": "dexengine/data/base_adapter.py",
        "adapter_version": ADAPTER_VERSION,
        "dataset": ds,
        "hand": hand,
        "yaw_deg": yaw[0],
        "hand_dir": yaw[1],
        "hand_dir_taco": yaw[2],
        "n_clips_in": len(jobs),
        "excluded_no_kinematic": excluded,
        "expected_out": expected,
        "produced": produced,
        "skipped_existing": skipped,
        "n_fail": len(fails),
        "fails": fails,
        "merged_duplicate_trees": sum(r["merged"] for r in results),
        "articulation_groups": sum(r["art_groups"] for r in results),
        "supplemented_trees": sum(r["supplemented"] for r in results),
        "by_emb": dict(Counter(j["emb"] for j in jobs)),
        "desk_mode": dict(Counter((j["sup"] or {}).get("desk_mode") for j in jobs)),
        "base_dir": os.path.relpath(_dest("base", ds, hand=hand, emb="bimanual").parent, data_root()),
        "complete": (produced + skipped == expected) and not fails and expected > 0,
    }


def _print_summary(ds: str, hand: str, rec: dict, jobs: list, excluded: dict) -> None:
    produced, skipped, expected, fails = rec["produced"], rec["skipped_existing"], rec["expected_out"], rec["fails"]
    print(
        f"[base] {ds}/{hand}: built {produced} + existing {skipped} = {produced + skipped} / expected "
        f"{expected} · failed {len(fails)} · merged {rec['merged_duplicate_trees']} · supplemented "
        f"{rec['supplemented_trees']} · articulation groups {rec['articulation_groups']} · "
        f"{'COMPLETE' if rec['complete'] else 'INCOMPLETE'}"
    )
    for c, e in list(fails.items())[:5]:
        print(f"    {c}: {e[:220]}")
    if not expected:  # zero inputs (layer or list not found · all excluded) is not '0 = 0 complete';
        # this ledger is the gate for downstream (sidecars)
        print(
            input_missing(
                f"[base] ✗ {ds}/{hand}: expected outputs 0 (input clips {len(jobs)} · excluded without "
                f"kinematic {len(excluded)}) — not counted as complete"
            ),
            file=sys.stderr,
        )


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.base_adapter")
    sys.exit(main())
