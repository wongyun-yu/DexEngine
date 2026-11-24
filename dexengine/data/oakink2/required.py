#!/usr/bin/env python3
"""OakInk2 intake rule: chooses the objects that must be in a clip's scene from source object poses and source MANO
contact, and exports them as supplements.

Why: the meta list the delivery followed (obj_list) misses objects people held and moved (insert_lightbulb s007, the
bulb). This replaces the old contact supplement branch (only zero-contact hand trees, by fingertip keypoints).

Rule (the requirement statement owns thresholds, categories and production slack: scene_requirement; values are not
restated here):
  target  every tree window of the axis-hand base (the root + each of its pieces _tN) × every source object in the
          sequence's obj_transf (obj_list is not read). Each window gets a category (first match) and is merged into
          the root; pieces inherit objects from the root.
  choose  objects with a category that are not in the tree (tree_part_ids). Files already exported for the same root
          (exported_rids) are 'pending'.
  write   --export: <root>__<category>__<rid>.npz (the root's whole timeline · base frame ·
      dexengine.data.oakink2.raw_objects.export_npz).
  role    not decided here: base_adapter loads them as extra, and the case (hand_object_assign) and roles
          (object_roles · coords.support_of) decide.
  undecidable = an error for that root (rc 1) · objects without a mesh are reported as 'nomesh' (no asset to load).

Measurement (not shared with the audit oracle's code):
  frame   base (yaw), the same transform as the exported poses (tree_transform · transform oracle oracle_residual) ·
          time on the tree's 50 Hz axis.
  hand    source raw_mano → smplx MANO (flat_hand_mean · wxyz · joint 0 = tsl · left hand = mano_conventions) surface
          vertices. Self-oracle: a median distance to the delivered MANO keypoints (wrist, five fingertips) above
          HAND_ORACLE_MAX_M is an error (measured: wrist 0.1 · fingertips 1.1 mm median · max 3.1 mm).
  object  source mesh surface samples + vertices · support = distance between the surfaces of the first and last
          samples of a 'moved' object.
Independence (independent-oracle clause): the statement lives in scene_requirement alone, and this module (production)
  and the audit (dexengine.data.readers.raw.common · dexengine.data.readers.raw.oakink2) each **read** it. Measurement
  is not shared: the audit measures in the source world · 30 Hz · its own MANO, this module in base · 50 Hz · its own
  MANO (checked against the delivered keypoints). Neither imports the other. Production is looser by the production
  slack (objects on a threshold boundary never fall between the two measurements; the same contract as judgement vs bake
  filter).

Output: [REQUIRED_OBJ] · --report · with --export INTAKE/raw_objects/ (old ownerless tags are retired, retire_unowned).
        Done = rows == roots · written == new pairs · errors 0.
Usage: $(paths.py --what retarget-py) dexengine/data/oakink2/required.py [--trees root,…] [--jobs 16] [--report <json>]
[--export]
       $(paths.py --what retarget-py) dexengine/data/oakink2/required.py --selftest   # negative test (read-only)
Regeneration order: **this tool --export** → dexengine/data/oakink2/raw_objects.py --supports --export →
dexengine/data/oakink2/raw_objects.py --parts
       --export (siblings of supplements not yet in base) → base_adapter ….
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dexengine.core import capacity
from dexengine.core.errors import InputMissingError
from dexengine.data.oakink2 import parts as OP  # noqa: E402 — owner of supplement tags (SUPP_TAGS · supp_tag)
from dexengine.data.oakink2 import raw_base as RO  # noqa: E402 — owner of the transform, export, roots and part ids
from dexengine.data.oakink2.required_measure import TREE_DT, measure_root

TAG = "[REQUIRED_OBJ]"
STEP = "dexengine.data.oakink2.required"  # capacity.NEEDS key (this module)


def seq_job(job) -> list:
    """Roots of one sequence (errors go into their row)."""
    pkl, items, base = job
    with open(pkl, "rb") as fh:
        d = pickle.load(fh)
    keys = list(json.loads((RO.RAW / "program/program_info" / f"{Path(pkl).stem}.json").read_text()))
    return [_root_row(root, ti, cuts, d, keys, base) for root, ti, cuts in items]


def _root_row(root, ti, cuts, d, keys, base) -> dict:
    """measure_root of one root, or its error row."""
    try:
        return measure_root(root, ti, cuts, d, keys, Path(base))
    except Exception as e:  # noqa: BLE001 — a single root's failure goes into its row, is counted, and gives rc 1
        return {"root": root, "error": f"{type(e).__name__}: {e}"[:300]}


def universe(base: Path, only: set | None) -> tuple[dict, list]:
    """Axis-hand base → {hash12: [(root, ti, [(piece, ti)…])]} · error rows."""
    roots, cuts, errs = {}, collections.defaultdict(list), []
    for p in sorted(x for x in base.iterdir() if (x / "task_info.json").is_file()):
        ti = json.loads((p / "task_info.json").read_text())
        try:
            r = RO.root_tree(base, p, ti)
        except RO.ClipError as e:
            errs.append({"root": p.name, "error": f"ClipError: {e}"})
            continue
        if r == p.name:
            roots[p.name] = ti
        else:
            cuts[r].append((p.name, ti))
    by_seq = collections.defaultdict(list)
    for r, ti in roots.items():
        if only is not None and r not in only:
            continue
        try:
            h = RO.SEQ_KEY.fullmatch(RO.intake_clip(ti)).group(3)
        except RO.ClipError as e:
            errs.append({"root": r, "error": f"ClipError: {e}"})
            continue
        by_seq[h].append((r, ti, cuts.get(r, [])))
    orphan = sorted(set(cuts) - set(roots))
    errs += [{"root": r, "error": "the piece's root is not in base"} for r in orphan]
    return by_seq, errs


def run(only: set | None, jobs: int | None, export: bool, report: str) -> int:
    base = RO.ROOT / RO.dl.render("base", "oakink2", hand=RO.HAND, emb="bimanual")
    by_seq, errs = universe(base, only)
    pk = {p.name.split("seq__")[1][:12]: p for p in (RO.RAW / "anno_preview").glob("*.pkl")}
    rows = list(errs)
    work = []
    for h, items in sorted(by_seq.items()):
        if h not in pk:
            rows += [{"root": r, "error": "source pkl missing"} for r, _ti, _c in items]
        else:
            work.append((str(pk[h]), items, str(base)))
    n_roots = sum(len(v) for v in by_seq.values())
    print(f"{TAG} axis hand {RO.HAND} · root trees {n_roots} · sequences {len(work)}", flush=True)
    t0 = time.time()
    with capacity.sized(STEP, len(work), jobs) as cap, Pool(cap.workers, capacity.worker_init, cap.initargs) as pool:
        for i, out in enumerate(pool.imap_unordered(seq_job, work), 1):
            rows += out
            if i % 50 == 0:
                print(f"{TAG} … sequence {i}/{len(work)} · {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: str(r["root"]))  # root order, not finish order (nothing depends on the pool size)
    return finish(rows, n_roots + len(errs), export, report, full=only is None)


def finish(rows: list, n_expect: int, export: bool, report: str, full: bool = False) -> int:
    """Chosen pairs → retire old ownerless supplements → (optional) write · marker · report (completion rule in the
    module docstring)."""
    req = {(r["root"], rid): e["cat"] for r in rows if "error" not in r for rid, e in r["required"].items()}
    S, again = RO.retire_unowned(req, {r["root"] for r in rows if "error" not in r}, export, full)
    todo = []
    for r in rows:
        if "error" in r:
            S["error"] += 1
            continue
        have, pend = set(r["in_tree"]), RO.exported_rids(r["root"])
        for rid, e in r["required"].items():
            e["state"] = (
                "reexport"
                if (r["root"], rid) in again
                else "in_tree"
                if rid in have
                else "pending"
                if rid in pend
                else "new"
            )
            S[f"{e['state']}:{e['cat']}"] += 1
            if e["state"] in ("new", "reexport"):
                todo.append((r["root"], rid, e["cat"]))
    roots_new = len({x[0] for x in todo})
    S["written"] = write(todo, rows) if export else 0
    lines = " · ".join(f"{k}={v}" for k, v in sorted(S.items()))
    print(
        f"{TAG} roots {len(rows)}/{n_expect} · newly chosen (root, rid) {len(todo)} · roots {roots_new} · {lines}"
        f" · write {'on' if export else 'off (count only)'}",
        flush=True,
    )
    if report:
        Path(report).write_text(
            json.dumps(
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "hand": RO.HAND,
                    "summary": dict(S),
                    "todo": todo,
                    "rows": rows,
                },
                ensure_ascii=False,
            )
        )
    bad = S["error"] or len(rows) != n_expect or (export and S["written"] != len(todo))
    if bad:
        print(
            f"{TAG} ✗ errors {S['error']} · rows {len(rows)}/{n_expect} · written {S['written']}/"
            f"{len(todo) if export else '—'}"
            f" — e.g. {[r for r in rows if 'error' in r][:3]}",
            flush=True,
        )
    return 1 if bad else 0


def write(todo: list, rows: list) -> int:
    """Write new pairs as pose sequences over the root's whole timeline."""
    base = RO.ROOT / RO.dl.render("base", "oakink2", hand=RO.HAND, emb="bimanual")
    by_root = collections.defaultdict(list)
    for root, rid, cat in todo:
        by_root[root].append((rid, cat))
    info = {r["root"]: r for r in rows if "error" not in r}
    by_seq = collections.defaultdict(list)
    for root in by_root:
        ti = json.loads((base / root / "task_info.json").read_text())
        by_seq[RO.SEQ_KEY.fullmatch(RO.intake_clip(ti)).group(3)].append((root, ti))
    pk = {p.name.split("seq__")[1][:12]: p for p in (RO.RAW / "anno_preview").glob("*.pkl")}
    n = 0
    for h, lst in by_seq.items():
        with open(pk[h], "rb") as fh:
            ot = pickle.load(fh)["obj_transf"]
        for root, ti in lst:
            r = info[root]
            a0, b0, T = r["a0"], r["b0"], r["T"]
            t50 = (np.arange(T) + int(ti["segment"]["start"])) * TREE_DT
            R, t = RO.tree_transform(ti)
            for rid, cat in by_root[root]:
                fr2 = [k for k in range(a0, b0) if k in ot[rid]]
                pos, Rb = RO.pose_base(ot[rid], fr2, t50, a0, R, t)
                RO.export_npz(root, cat, rid, pos, Rb, -1.0, r["oracle"][0] / 1e3, r["oracle"][1], yaw=RO.scene_yaw(ti))
                n += 1
    return n


#: Positive case of the self-test: the bulb the delivery missed
SELFTEST_SEQ = "oakink2_insert_lightbulb__scene_02_486fccbc8bd7_s007"
SELFTEST_ROOTS = ("_seg02", "_seg03", "_seg04")
SELFTEST_RID = "O02@0028@00001"


def selftest() -> int:
    """Negative test (read-only): A as is → 'moved' · B removed from obj_list = A · C removed from obj_transf = not
    chosen."""
    base = RO.ROOT / RO.dl.render("base", "oakink2", hand=RO.HAND, emb="bimanual")
    by_seq, errs = universe(base, {SELFTEST_SEQ + r for r in SELFTEST_ROOTS})
    if errs or len(by_seq) != 1:
        raise InputMissingError(f"{TAG} ✗ self-test inputs missing: {errs[:2]} · sequences {len(by_seq)}")
    ((h, items),) = by_seq.items()
    pkl = next((RO.RAW / "anno_preview").glob(f"*seq__{h}*.pkl"))
    with open(pkl, "rb") as fh:
        d = pickle.load(fh)
    keys = list(json.loads((RO.RAW / "program/program_info" / f"{pkl.stem}.json").read_text()))

    def pick(dd) -> dict:
        return {
            r: measure_root(r, ti, cuts, dd, keys, base)["required"].get(SELFTEST_RID, {}).get("cat")
            for r, ti, cuts in items
        }

    a = pick(d)
    b = pick({**d, "obj_list": [x for x in d["obj_list"] if x != SELFTEST_RID]})
    c = pick({**d, "obj_transf": {k: v for k, v in d["obj_transf"].items() if k != SELFTEST_RID}})
    try:
        OP.supp_tag("right", "selftest")  # D: an old ownerless tag is not loaded and fails
        dok = False
    except ValueError:
        dok = OP.supp_tag("moved", "selftest") == "moved"
    ok = len(a) == len(SELFTEST_ROOTS) and set(a.values()) == {"moved"} and b == a and set(c.values()) == {None} and dok
    print(
        f"{TAG} self-test {'PASS' if ok else 'FAIL'} — A as is {a} · B removed from meta list {b} · C removed "
        f"from poses {c}"
        f" · D old tag fails {dok}",
        flush=True,
    )
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", default="", help="root tree names (comma) — tests, partial re-measurement")
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--report", default="")
    ap.add_argument(
        "--export",
        action="store_true",
        help="write the newly chosen pairs to INTAKE/raw_objects (otherwise count only)",
    )
    ap.add_argument("--selftest", action="store_true", help="negative tests A · B · C (selftest docstring), read-only")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    only = {c.strip() for c in a.trees.split(",") if c.strip()} or None
    return run(only, a.jobs, a.export, a.report)


if __name__ == "__main__":
    raise SystemExit(main())
