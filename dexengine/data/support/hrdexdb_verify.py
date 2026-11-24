#!/usr/bin/env python3
"""Base verification of the HRDexDB desk stamp (read-only): owner of measurement, verdict, marker and thresholds.

The only entry is the stamp writer (its main hands over to verify_main; no second entry):
  dexengine/data/support/hrdexdb.py --verify-base --hand <hand> --dataset <ds> --root <data root> --out <dir>
Kept apart from the writer (version, writes, self-test) because of the read window.
Targets: base lineage trees (paths.parent_clip) of the stamp targets (base_layer.stale.stamp_targets).
  g0   start gap |g0| ≤ G0_MAX_M of the main objects in the g0 trees (g0_trees): whether h1p (desk_z = frame-0 bottom)
       carried through to base. g0 trees = first-frame trees (segment.start 0) with main objects; when every head tree
       has no main object (D23), the tree with the clip's earliest start that has main objects. Clips whose lineage
           trees
       are all D23 are outside the g0 domain and only counted in the marker field g0_na (no candidate tree uses the
       stamped desk_z; the same decision as D24's d24_na).
  D24  every lineage tree = candidate_rules.d24_state (the D24 input state the first-round candidates read; the depth
       against the desk is the judge's, placement/scene/object_desk). D23 trees without main objects are outside the
       D24 domain and only counted in d24_na (not unevaluated).
Verdict (judge_verify): every g0 target (stamp targets − stage-1 exclusions − outside the g0 domain) passes · measured
≥ 1 · D24 unevaluated 0 · unmatched 0 → rc 0.
Marker (marker): one VERIFY_TAG line, read by the gate (stamp verification column of T0_GATE H-1). Unmeasurable runs
fail with VERIFY_DEAD.
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

from dexengine.core import paths as P  # noqa: E402
from dexengine.core.errors import InputMissingError
from dexengine.data.support import hrdexdb as HR  # noqa: E402 — the stamp writer (name of the version owner · entry)

#: --verify-base threshold: start gap |g0| of base trees starting at the take's first frame. When h1p (desk_z = frame-0
#: bottom) carried through to base, g0 ≈ 0. This constant is the one statement.
G0_MAX_M = 0.003
#: Verification marker head; the gate (T0_GATE H-1) reads this line (field names and order are defined by marker()
#: alone).
VERIFY_TAG = "[C693_VERIFY]"
#: Head of the line for unmeasurable failures; a string distinct from the marker head (the reader counts marker lines
#: only; one name = one meaning).
VERIFY_DEAD = "[C693_VERIFY_DEAD]"


def _start_g0(tree: Path, ti: dict, scene: str) -> list:
    """Start gap g0 per main object (m; None in aerial scenes), owned by coords.start_gaps (the same estimator as D24′ ·
    D26′)."""
    from dexengine.placement.coords import core as C  # noqa: PLC0415

    return [g["g0_m"] for g in C.start_gaps(tree, ti, aerial=scene == "aerial")]


def g0_trees(rows: list) -> list:
    """Lineage trees of one clip whose g0 is measured; called by both measurement (verify_base) and verdict
    (judge_verify).
    Head trees (segment.start 0) with main objects; when every head has no main object (D23), the tree with the
    earliest start among those with main objects. The stamp statement (desk_z = take frame-0 bottom) holds as long as
    the object stayed placed until then; if it was moved, g0 exceeds the threshold and shows up as a failure (no silent
    pass). No head · all D23 = [] (the former never measured = failure · the latter g0_na)."""
    from dexengine.data import candidate_rules as TC  # noqa: PLC0415

    if not any(r["start"] == 0 for r in rows):
        return []
    live = [r for r in rows if r["d24"] != TC.D24_NA_D23]
    first = min((r["start"] for r in live), default=None)
    return [r for r in live if r["start"] == (0 if any(r["start"] == 0 for r in live) else first)]


def _run_meta() -> dict:
    """Code tree, commit, branch, dirty flag, data root and time of this verification run (stored with its output)."""
    import datetime  # noqa: PLC0415

    from dexengine.core.run_identity import tree_identity  # noqa: PLC0415

    return {
        **tree_identity(),
        "main_root": str(P.main_root()),
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def verify_base(hand: str, dataset: str, g0_of=_start_g0) -> dict:
    """Measurement of the verification (read-only): per lineage tree of the stamp target clips {clip, emb, tree, scene,
    start, d24, g0_m} (g0_m only for g0 trees, else None). The caller sets the data root and hand through env. A
    different version owner · missing or incomplete stage-1 ledger · missing base layer · missing lineage fields
    fail."""
    from dexengine.data import candidate_rules as TC  # noqa: PLC0415 — owner of the D24 verdict and the scene
    from dexengine.data import rules as RR  # noqa: PLC0415
    from dexengine.data.base_layer import stale as BS  # noqa: PLC0415 — owner of the desk-rule stamp target set

    me = HR.__name__  # version owner = the stamp writer (base_layer.stale.DESK_RULE_OWNERS)
    owner = BS.desk_rule_owner(dataset)
    if owner is None or owner[0] != me:
        raise SystemExit(
            f"{VERIFY_DEAD} ✗ {dataset}: desk rule version owner {owner[0] if owner else None} ≠ {me} — "
            "this verification's threshold (G0_MAX_M) is not that dataset's rule (base_layer.stale.DESK_RULE_OWNERS)"
        )
    stamped = BS.stamp_targets(owner, BS.intake_clips(dataset))
    spec = P.hand_spec_dir(hand)
    lp = P.layer_path("intake", dataset) / f"base_{spec}.json"
    if not lp.is_file():
        raise InputMissingError(f"{VERIFY_DEAD} ✗ stage-1 ledger missing {lp} — build base first (stage 1)")
    led = json.loads(lp.read_text())
    if led.get("hand") != spec or led.get("complete") is not True:
        raise SystemExit(
            f"{VERIFY_DEAD} ✗ {lp}: hand {led.get('hand')!r}(≠ {spec}) · complete {led.get('complete')!r} — "
            "measured only on a finished stage-1 ledger"
        )
    excl = set(led.get("excluded_no_kinematic") or {}) | set(led.get("fails") or {})
    rows, layers, src = [], 0, {}
    for emb in RR.DATASET_PROFILE[dataset].embs:
        root = P.layer_path("base", dataset, hand=spec, emb=emb)
        if not root.is_dir():
            continue
        layers += 1
        for name in sorted(n for n in os.listdir(root) if (root / n).is_dir()):
            ti = json.loads((root / name / "task_info.json").read_text())
            if not ti.get("parent_clip") or (ti.get("segment") or {}).get("start") is None:
                raise InputMissingError(
                    f"{VERIFY_DEAD} ✗ {emb}/{name}: parent_clip · segment.start missing — "
                    "lineage and start frame cannot be determined"
                )
            clip = P.parent_clip(ti["parent_clip"])
            if clip not in stamped:
                continue
            scene = TC.scene_of(dataset, ti)
            rows.append(
                {
                    "clip": clip,
                    "emb": emb,
                    "tree": name,
                    "scene": scene,
                    "start": int(ti["segment"]["start"]),
                    "d24": TC.d24_state(root / name, ti, TC.rule_on("D24", scene)),
                    "g0_m": None,
                }
            )
            src[(emb, name)] = (root / name, ti)
    if not layers:
        raise InputMissingError(f"{VERIFY_DEAD} ✗ {dataset}/{spec}: no base layer — stage 1 has not run")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["clip"]].append(r)
    for rs in by.values():
        for r in g0_trees(rs):
            r["g0_m"] = g0_of(*src[(r["emb"], r["tree"])], r["scene"])
    return {
        "dataset": dataset,
        "hand": hand,
        "spec": spec,
        "owner": me,
        "stamped": sorted(stamped),
        "stage1_excluded": sorted(excl),
        "ledger": str(lp),
        "rows": rows,
    }


def judge_verify(m: dict) -> dict:
    """Measurement → verdict. Exactly one per clip: measured target (has lineage trees) · stage-1 exclusion (ledger) ·
    unmatched (both or neither). g0 = all main objects of the g0 trees (g0_trees); no g0 tree · empty list · None =
    never measured · all lineage trees D23 = outside the g0 domain (g0_na, counted only). D24 = all lineage trees."""
    from dexengine.data import candidate_rules as TC  # noqa: PLC0415

    by = collections.defaultdict(list)
    for r in m["rows"]:
        by[r["clip"]].append(r)
    excl = set(m["stage1_excluded"])
    need, stage1, unmatched = [], [], {}
    for c in m["stamped"]:
        if by.get(c) and c in excl:
            unmatched[c] = "stage-1 exclusion, yet base has lineage trees"
        elif not by.get(c) and c not in excl:
            unmatched[c] = "neither in base nor in the stage-1 ledger (excluded_no_kinematic · fails)"
        elif c in excl:
            stage1.append(c)
        else:
            need.append(c)
    measured, g0_ok, g0_na = [], [], []
    for c in need:
        if all(r["d24"] == TC.D24_NA_D23 for r in by[c]):
            g0_na.append(c)
            continue
        gs = [r["g0_m"] for r in g0_trees(by[c])]
        vals = [g for lst in gs for g in (lst or [None])]
        if gs and all(g is not None for g in vals):
            measured.append(c)
            if max(abs(g) for g in vals) <= G0_MAX_M:
                g0_ok.append(c)
    d24 = collections.Counter(r["d24"] for r in m["rows"])
    na = sum(d24[k] for k in TC.D24_NA)
    not_eval = sum(n for k, n in d24.items() if k not in TC.D24_MEASURED + TC.D24_NA)
    need_g0 = [c for c in need if c not in g0_na]
    rc = int(not (len(g0_ok) == len(need_g0) and measured and not not_eval and not unmatched))
    return {
        "n_measured": len(measured),
        "g0_ok": len(g0_ok),
        "g0_need": len(need_g0),
        "g0_na": len(g0_na),
        "d24_not_evaluated": not_eval,
        "d24_na": na,
        "unmatched": unmatched,
        "stage1": stage1,
        "d24_states": dict(d24),
        "g0_fail": sorted(set(need_g0) - set(g0_ok)),
        "g0_na_clips": sorted(g0_na),
        "rc": rc,
    }


def marker(hand: str, v: dict) -> str:
    """Verification marker line: the only definition of field names and order (read by the gate and the self-test)."""
    return (
        f"{VERIFY_TAG} hand={hand} n_measured={v['n_measured']} g0_ok={v['g0_ok']}/{v['g0_need']} "
        f"g0_na={v['g0_na']} d24_not_evaluated={v['d24_not_evaluated']} "
        f"d24_na={v['d24_na']} "
        f"unmatched={len(v['unmatched'])} rc={v['rc']}"
    )


def verify_main(a, ap) -> int:
    """--verify-base entry: set env (data root · hand), measure, write to --out and print the marker (rc = verdict)."""
    if a.mode or a.write_intake or not (a.hand and a.dataset and a.root and a.out):
        ap.error(
            "--verify-base: --hand · --dataset · --root · --out are required (separate from --mode · --write-intake)"
        )
    root = Path(a.root).resolve()
    if not (root / "data").is_dir() or os.environ.get("DEXENGINE_HAND", a.hand) != a.hand:
        raise InputMissingError(f"{VERIFY_DEAD} ✗ --root {root} has no data/ or DEXENGINE_HAND ≠ --hand {a.hand}")
    os.environ.update(DEXENGINE_MAIN_ROOT=str(root), DEXENGINE_HAND=a.hand)
    from dexengine.data import candidate_rules as TC  # noqa: PLC0415

    meta = _run_meta()
    print(f"[HRDEXDB_DESK] verify code {meta['tree']} @{meta['commit'][:9]} · data {meta['main_root']}")
    m = verify_base(a.hand, a.dataset)
    v = judge_verify(m)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    th = {"G0_MAX_M": G0_MAX_M, "D24_INPUT_VERSION": TC.D24_INPUT_VERSION}
    (out / "c693_verify.json").write_text(
        json.dumps(
            dict(meta=meta, thresholds=th, verdict=v, marker=marker(a.hand, v), **m), ensure_ascii=False, indent=1
        )
    )
    print(
        f"[HRDEXDB_DESK] verify {a.dataset}: stamp targets {len(m['stamped'])} · stage-1 exclusions "
        f"{len(v['stage1'])} · trees "
        f"{len(m['rows'])} · D24 {v['d24_states']} · g0 over {v['g0_fail']} · outside g0 domain {v['g0_na_clips']} · "
        f"unmatched {v['unmatched']} → {out}"
    )
    print(marker(a.hand, v), flush=True)
    return v["rc"]
