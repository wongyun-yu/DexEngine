#!/usr/bin/env python3
"""**Single owner** of the data case taxonomy (CASE TAXONOMY) — the case names and the flag thresholds (TH_*) live
here. The hand–object assignment (contact · grasp ·
carry · active hands · used objects · share kind · timeline) and its thresholds belong to hand_object_assign.py —
this file reads the arrays (MANO sidecar · object trajectory) and hands them over.

Computed per tree (reads the base layer only): hands present → per hand×object frame layers → assignment
(hand_object_assign.assign) → case (mutually exclusive) + flags. What cannot be classified is left loudly as
UNCLASSIFIED. Outputs: task_info["case"] · task_info["rules"]["ASSIGN"] (with --write) · the per-hand audit
`case_census.json` (data.rules.audit_out, under the hand's audit/ folder; dashboard input) · partial scopes go to
the hand's run log (run_log cases.taxonomy).

Usage (retarget venv interpreter — dexengine.data.contact.geom uses trimesh/scipy):
  $RETARGET_PY dexengine/data/cases/taxonomy.py --dataset hocap --emb bimanual --clips a,b [--write]
  $RETARGET_PY dexengine/data/cases/taxonomy.py --all --jobs 16 --write
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
from dexengine.core import capacity  # noqa: E402
from dexengine.core import paths as P  # noqa: E402
from dexengine.data import rules as RR  # noqa: E402 — owner of the hand axis

STEP = "dexengine.data.cases.taxonomy"  # capacity.NEEDS key (this module)
#: argparse description, kept separate from the module docstring.
HELP = (
    "Single owner of the data case taxonomy (CASE TAXONOMY): case names and flag thresholds (TH_*) live here, the "
    "hand–object assignment in hand_object_assign.py."
)


#: Window field of this stage: cases are measured on the tree window. D36 pieces re-measure on their own window
#: (segment.pieces.WINDOW_DECL).
WINDOW_TOP = ("case",)
from dexengine.data.cases.assign_version import (
    ASSIGN_KEY,  # noqa: E402 — assignment record key (written here, read by the roles stage)
)

DATASETS = ("taco", "hocap", "h2o", "grab", "oakink2", "arctic", "dexcanvas", "hrdexdb", "hot3d")
#: Print which hand, source and thresholds this run uses. The values come from data.rules, which reads them from
#: their owner files.
RR.hand_thresholds()


#: Contact is measured on MANO: human fingertip keypoints vs the object surface (contact_all layer), independent of the
#: robot hand. A case describes the source motion, not the robot. Measured with each robot hand, labels would differ per
#: hand, and per-hand comparisons on the same clip would measure label drift instead of hand differences.
#: The MANO keypoint frame rule lives in `mano_frame`, a dependency-free module, so that the case layer and the GRAB
#: desk oracle (D14′) call the same function.
from dexengine.data.cases import (
    assign_version as AR,  # noqa: E402 — assignment versions (one shared ACTIVE version, hand registry HANDS)
)
from dexengine.data.cases import classify as CL  # owner of classify_tree and the TH_* thresholds
from dexengine.data.cases import (
    hand_object_assign as H,  # noqa: E402 — hand–object assignment (A3–A7, timeline) and its thresholds
)
from dexengine.data.cases.classify import HAND, VERSION, _assign_mode, classify_tree
from dexengine.data.contact import (
    fingerprint as SF,  # noqa: E402 — sidecar input fingerprint, checked against this base before reading
)
from dexengine.data.contact.source import CONTACT_SOURCE  # noqa: E402
from dexengine.data.tree_io import (
    write_json_if_changed,  # noqa: E402 — writes base task_info only when the content changed (keeps the f0 cache fingerprint)
)


def _job(args):
    ds, emb, clip, write = args
    try:
        rec = classify_tree(ds, emb, clip)
    except SF.SidecarFingerprintError as e:
        # The sidecar comes from a different input than this base: do not write, and do not overwrite the old case with
        # UNCLASSIFIED (roles stop with StaleCase when the case's contact_input is missing or differs). main exits rc≠0.
        return (
            {
                "clip": clip,
                "dataset": ds,
                "emb": emb,
                "case": "UNCLASSIFIED",
                "why": str(e)[:400],
                "flags": [],
                "version": VERSION,
                "contact_source": CONTACT_SOURCE,
                "sidecar_fp_fail": True,
            },
            False,
            None,
        )
    except Exception as e:  # noqa: BLE001 — unmeasurable (missing input, I/O, compute error) is not 'unclassifiable'
        # Only an UNCLASSIFIED returned by classify_tree (a rule decision) becomes a case. Exceptions are not written:
        # the old case stays, the error goes to the census errors, and main exits rc≠0.
        return (
            {
                "clip": clip,
                "dataset": ds,
                "emb": emb,
                "case": "ERROR",
                "why": f"{type(e).__name__}: {e}"[:300],
                "tb": traceback.format_exc()[-1500:],
                "flags": [],
                "version": VERSION,
                "contact_source": CONTACT_SOURCE,
                "error": True,
            },
            False,
            None,
        )
    assign = rec.pop(ASSIGN_KEY, None)  # the assignment lives only in rules.ASSIGN, not in the case record
    # Validity gate, counted by the [ASSIGN] marker (returned beside rec, which goes into task_info). ASSIGN-1 = no gate
    # '-'.
    gv = AR.gate_of(assign)
    if write:
        p = P.layer_path("base", ds, hand=HAND, emb=emb, clip=clip) / "task_info.json"
        ti = json.loads(p.read_text())
        ti["case"] = rec
        rules = ti.setdefault("rules", {})
        if assign is None:
            rules.pop(ASSIGN_KEY, None)  # no objects: drop the old assignment (the roles stage then sees no case)
        else:
            rules[ASSIGN_KEY] = assign
        # Write only when the content changed (tree_io; keeps the f0 cache fingerprint). Whether it wrote is returned
        # beside rec, because rec itself goes into task_info.
        return rec, write_json_if_changed(p, ti), gv
    return rec, False, gv


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--emb", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--clips", default="", help="comma list (within the given dataset/emb)")
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--write", action="store_true", help="write task_info['case'] (otherwise census only)")
    # The canonical census path is only for --all runs. The campaign runs one dataset at a time, and a partial run that
    # wrote the canonical file would replace `unclassified`, version and thresholds with that dataset's (only `census`
    # rows merge). Rule: --all → canonical, otherwise a scoped path; an explicit --census-out wins.
    ap.add_argument(
        "--census-out",
        default="",
        help="census output path (default: with --all, case_census.json in this hand's audit folder; "
        "for a partial scope, under this hand's log folder case_taxonomy/)",
    )
    return ap


def _census_path(a) -> tuple[Path, str]:
    """Where the census goes (canonical only for --all; a partial scope gets its own path) and why."""
    # The census differs per hand (coverage, robot wrist geometry), so it is written under the hand root.
    HID = RR.active_hand()
    CANON_CENSUS = RR.audit_path("case_census.json", HID)
    if a.census_out:
        census_out, census_why = Path(a.census_out), "set by caller"
    elif a.all:
        census_out, census_why = RR.audit_out("case_census.json", HID), "canonical (all datasets, this hand)"
    else:
        scope = a.dataset or "none"
        census_out = P.run_path("run_log", HID, tool="case_taxonomy") / f"case_census__{scope}.json"
        census_why = f"partial scope ({scope}); canonical {CANON_CENSUS.name} untouched"
    return census_out, census_why


def _targets(a) -> list:
    """(dataset, emb, clip, write) of every base tree in scope; dies when there is none."""
    targets = []
    dss = DATASETS if a.all else [a.dataset]
    for ds in dss:
        for emb in ["right", "left", "bimanual"] if not a.emb else [a.emb]:
            root = P.layer_path("base", ds, hand=HAND, emb=emb)
            if not root.is_dir():
                continue
            # names from outside go through the single entry point (P.read_clip_list)
            clips = (
                P.read_clip_list([c for c in a.clips.split(",") if c], dataset=ds, quiet=True)
                if a.clips
                else sorted(os.listdir(root))
            )
            targets += [(ds, emb, c, a.write) for c in clips if (root / c / "task_info.json").is_file()]
    if not targets:
        raise SystemExit("[case] no target trees")
    return targets


def _run(targets: list, jobs: int | None, t0: float) -> tuple[list, int, collections.Counter]:
    """classify every target in a process pool sized by capacity (`jobs` = an explicit option) → (records in target
    order — not finish order, so nothing depends on the pool size —, trees written, validity-gate counts)."""
    recs: list = [None] * len(targets)
    n_written = 0
    gates: collections.Counter = collections.Counter()  # validity gate counts (unmeasurable trees excluded)
    with (
        capacity.sized(STEP, len(targets), jobs) as cap,
        ProcessPoolExecutor(cap.workers, initializer=capacity.worker_init, initargs=cap.initargs) as ex,
    ):
        futs = {ex.submit(_job, t): k for k, t in enumerate(targets)}
        for i, f in enumerate(as_completed(futs), 1):
            rec, wrote, gv = f.result()
            recs[futs[f]] = rec
            if gv is not None:
                gates[gv] += 1
            n_written += int(wrote)
            if i % 500 == 0:
                print(f"[case] {i}/{len(targets)} {time.time() - t0:.0f}s", flush=True)
    return recs, n_written, gates


def _aggregate(recs: list) -> tuple[dict, list]:
    """Per dataset/emb census counters and the unmeasurable trees."""
    # Aggregate
    census = {}
    # unmeasurable trees: counted under errors, not in the case distribution
    errs = [r for r in recs if r.get("error")]
    for r in recs:
        key = f"{r['dataset']}/{r['emb']}"
        c = census.setdefault(
            key,
            {
                "n": 0,
                "errors": 0,
                "cases": collections.Counter(),
                "flags": collections.Counter(),
                "n_objects": collections.Counter(),
                "n_used": collections.Counter(),
                "obj_kind": collections.Counter(),
                "articulation": collections.Counter(),
                "obj_ids": set(),
                "desk_mode": collections.Counter(),
                "fps": collections.Counter(),
                "T_s": [],
            },
        )
        c["n"] += 1
        if r.get("error"):
            c["errors"] += 1
            continue
        c["cases"][r["case"]] += 1
        for f in r.get("flags", []):
            c["flags"][f] += 1
        if "n_objects" in r:
            c["n_objects"][r["n_objects"]] += 1
            c["n_used"][r["n_used"]] += 1
            for k in r["obj_kind"].values():
                c["obj_kind"][k] += 1
            for k in r.get("articulation") or []:
                c["articulation"][k] += 1
            c["obj_ids"].update(r["obj_ids"])
            c["desk_mode"][str(r.get("desk_mode"))] += 1
            c["fps"][round(r["fps"])] += 1
            c["T_s"].append(r["T"] / r["fps"])
    return census, errs


def _census_doc(census: dict, recs: list, errs: list, amode: str) -> dict:
    # The `hand` field lets the mixed-hand guard fire, so the census records which hand's base layer it counted.
    out = {
        "version": VERSION,
        "contact_source": CONTACT_SOURCE,
        "hand": HAND,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "thresholds": {**H.thresholds(), **CL.thresholds()},
        "assign_version": amode,
        "census": {
            k: {
                "n": v["n"],
                "errors": v["errors"],
                "cases": dict(v["cases"]),
                "flags": dict(v["flags"]),
                "n_objects": {str(a): b for a, b in sorted(v["n_objects"].items())},
                "n_used": {str(a): b for a, b in sorted(v["n_used"].items())},
                "obj_kind": dict(v["obj_kind"]),
                "articulation": dict(v["articulation"]),
                "unique_obj_ids": len(v["obj_ids"]),
                "desk_mode": dict(v["desk_mode"]),
                "fps": {str(a): b for a, b in v["fps"].items()},
                "T_s_median": round(float(np.median(v["T_s"])), 1) if v["T_s"] else None,
            }
            for k, v in sorted(census.items())
        },
        "unclassified": [
            {"clip": r["clip"], "dataset": r["dataset"], "why": r.get("why")}
            for r in recs
            if r["case"].startswith("UNCLASSIFIED")
        ][:200],
        "errors": [{"clip": r["clip"], "dataset": r["dataset"], "emb": r["emb"], "why": r.get("why")} for r in errs][
            :200
        ],
    }
    return out


def _write_census(a, out: dict, census_out: Path, census_why: str) -> None:
    if a.all or (a.dataset and not a.clips):
        census_out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(census_out.read_text()) if census_out.is_file() else {"census": {}}
        prev.update({k: v for k, v in out.items() if k != "census"})
        prev.setdefault("census", {}).update(out["census"])
        prev["unclassified"], prev["errors"] = out["unclassified"], out["errors"]
        census_out.write_text(json.dumps(prev, ensure_ascii=False, indent=1))
        # marker line: where the census went
        print(f"[case] census → {census_out} ({census_why}) hand={HAND}", flush=True)
        # Not written to the INTAKE layer (data/<ds>/INTAKE/): that layer is hand-independent, and per-hand censuses
        # would overwrite each other there.


def _print_summary(a, census: dict, recs: list, n_written: int, t0: float) -> None:
    for k, v in sorted(census.items()):
        print(
            f"[case] {k}: n={v['n']} errors={v['errors']} "
            + " ".join(f"{c}={n}" for c, n in v["cases"].most_common())
            + " | flags "
            + " ".join(f"{f}={n}" for f, n in v["flags"].most_common(6)),
            flush=True,
        )
    if a.clips:
        for r in recs:
            print(
                json.dumps(
                    {
                        k: r.get(k)
                        for k in (
                            "clip",
                            "case",
                            "flags",
                            "active",
                            "used",
                            "share",
                            "hand_total_s",
                            "wrist_desk_m",
                            "main_by_adapter",
                        )
                    },
                    ensure_ascii=False,
                )[:600]
            )
    print(f"[case] {len(recs)} trees in {time.time() - t0:.0f}s · write={a.write}", flush=True)
    n_mv = sum("move_runs" in r for r in recs)
    print(
        f"[CUT_RULE] case {VERSION} · motion layer case.move_runs {n_mv}/{len(recs)} trees · "
        f"carry ∧ body (≥ {H.TH_OBJ_STILL_MPS} m/s ∨ > {H.TH_ROT_DPS}°/s) ∨ joint drive · "
        "read by dexengine.data.segment.split (D36 v6)",
        flush=True,
    )
    if a.write:  # marker; trees with unchanged content were not written (tree_io.write_json_if_changed)
        print(f"[case] --write rewrote {n_written} · unchanged skipped {len(recs) - n_written}", flush=True)


def _final_rc(recs: list, errs: list, gates, amode: str) -> int:
    """Assignment marker line, then rc: 4 exceptions · 3 sidecar fingerprint mismatch · 5 marker/gate mismatch."""
    mline, mbad = AR.marker(HAND, amode, gates)
    print(mline, flush=True)  # applied assignment version and validity gate counts (confirmation line)
    if errs:  # unmeasurable trees were not written (old cases kept); a partial success is a failure
        ex = collections.Counter(r["why"].split(":", 1)[0] for r in errs)
        print(
            f"[case] ✗ exceptions {len(errs)}/{len(recs)} trees — cases not written (old cases kept) · kinds "
            f"{dict(ex)} · e.g. "
            f"{errs[0]['dataset']}/{errs[0]['emb']}/{errs[0]['clip']}: {errs[0]['why']}\n{errs[0].get('tb', '')}",
            file=sys.stderr,
            flush=True,
        )
        return 4
    fp_fail = [r for r in recs if r.get("sidecar_fp_fail")]
    if fp_fail:  # sidecar input differs from base; a partial success is a failure
        print(
            f"[case] ✗ sidecar fingerprint mismatch {len(fp_fail)} trees — cases not written. e.g. "
            f"{fp_fail[0]['why'][:300]}",
            file=sys.stderr,
            flush=True,
        )
        return 3
    if mbad:  # applied-version marker and gate counts disagree (reported after exceptions and fingerprint failures)
        print(
            f"[case] ✗ assignment version {amode} disagrees with the validity gates: {mbad}",
            file=sys.stderr,
            flush=True,
        )
        return 5
    return 0


def main() -> int:
    a = _parser().parse_args()
    amode = _assign_mode("")  # assignment version of this run; unregistered hands fail here, before any work
    census_out, census_why = _census_path(a)
    targets = _targets(a)
    t0 = time.time()
    recs, n_written, gates = _run(targets, a.jobs, t0)
    census, errs = _aggregate(recs)
    out = _census_doc(census, recs, errs, amode)
    _write_census(a, out, census_out, census_why)
    _print_summary(a, census, recs, n_written, t0)
    return _final_rc(recs, errs, gates, amode)


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.cases.taxonomy")
    raise SystemExit(main())
