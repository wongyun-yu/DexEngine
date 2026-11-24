#!/usr/bin/env python3
"""First-pass candidate clips for judging and training: the one place the first-pass rules are applied.

Rules: D4 drops trees without a known support surface (``support.desk_mode`` not in {table, table_recovered}; a
recovered desk counts). ``quality.train_candidate`` False drops the tree. The other first-pass rules are listed in
``FIRST_PASS_RULES``; the scenes each applies to are declared by the rule registry. Stage 2 and the training-set
selection read this same list.

    python3 dexengine/data/candidates.py --dataset h2o --emb bimanual --out <list.txt>
    python3 dexengine/data/candidates.py --dataset hocap --emb bimanual            # stdout
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.data import rules as RR
from dexengine.data.artic.convert_version import EXCLUDE_SRC as _ARTIC_SRC  # generator name in the exclusion ledger
from dexengine.data.candidate_rules import (
    AERIAL_DESK,
    CASE_DROP_PREFIX,
    D24_BELOW,
    D24_STALE,
    D24_UNMEASURED,
    DESK_OK,
    LEDGER_EXCLUDED,
    MIN_DURATION_S,
    NO_MAIN_OBJECT,
    OBJ_JUMP,
    OBJ_SPEED_MAX_MPS,
    PART_MOVING_FITS,
    QUALITY_FLAGS_DROP,
    ROUTE_PREFIX,
    SPLIT_PARENT,
    SUPPORT_ARTIC,
    WRIST_BELOW_M,
    WRIST_SLOW_MPS,
    BaseLayerMissing,
    _artic_excluded_mark,
    _ledger_excluded_mark,
    _support_artic_mark,
    d24_state,
    desk_mode_of,
    desk_reason,
    desk_route,
    excluded_reason,
    is_route,
    no_main_object,
    quality_reason,
    rule_on,
    support_articulated,
)

#: Version of the D24 input; a tree whose rules.D24.version differs (old or missing) is fatal at the end.
from dexengine.data.desk_penetration import D24_VERSION as D24_INPUT_VERSION  # noqa: E402
from dexengine.data.support import gap as SG  # aerial support gaps (head, middle, whole)
from dexengine.data.support.grab import ORACLE_DROP_M  # D10-b threshold


def _d31_fast_object_slow_wrist(clip_dir, d: dict):
    """D31 verdict -> None (nothing to drop, or not measurable) or {obj, t, v_mps, run_s, wrist_mps} (the evidence).

    Main-object trajectory = base qpos (object columns are the source's, hand-independent); wrist = the MANO layer
    (`provenance.mano_dir`). One fast-object, slow-wrist run is enough. An unreadable object trajectory gives None; a
    fast run whose MANO wrist cannot be read is fatal."""
    import numpy as _np

    c = d.get("case") or {}
    fps = float(c.get("fps") or 30.0)
    mains = [o for o in (d.get("objects") or []) if o.get("role") == "main" and o.get("qpos_start") is not None]
    f = clip_dir / "0/trajectory_kinematic.npz"
    if not mains or not f.is_file():
        return None
    try:
        with _np.load(f) as z:
            q = z["qpos"]
    except Exception:  # noqa: BLE001
        return None
    if len(q) < 3:
        return None
    wrist = None
    for o in mains:
        p = q[:, int(o["qpos_start"]) : int(o["qpos_start"]) + 3]
        v = _np.linalg.norm(_np.diff(p, axis=0), axis=-1) * fps
        fast = v > OBJ_SPEED_MAX_MPS
        t = 0
        while t < len(v):
            if not fast[t]:
                t += 1
                continue
            e = t
            while e + 1 < len(v) and fast[e + 1]:
                e += 1
            if wrist is None:
                wrist = _mano_wrist_speed(d, len(p), fps)
            sides = ["right", "left"] if o.get("side") in (None, "both") else [o["side"]]
            vw = max(float(wrist[sd][max(0, t - 2) : min(len(v), e + 3)].max()) for sd in sides if sd in wrist)
            if vw < WRIST_SLOW_MPS:
                return {
                    "obj": o.get("obj_id") or o.get("name"),
                    "t": t,
                    "v_mps": round(float(v[t : e + 1].max()), 3),
                    "run_s": round((e - t + 1) / fps, 3),
                    "wrist_mps": round(vw, 3),
                }
            t = e + 1
    return None


def _mano_wrist_speed(d: dict, T: int, fps: float) -> dict:
    """{side: (T-1,) MANO wrist speed} over this tree's window (T frames from segment.start + frame_map.offset)."""
    import numpy as _np

    seg = d.get("segment") or {}
    md = (d.get("provenance") or {}).get("mano_dir")
    if not md:
        raise InputMissingError(
            f"[train_candidates] D31: {d.get('clip_id')} has no provenance.mano_dir -- cannot "
            "judge a jump without the wrist"
        )
    a = int(seg.get("start") or 0) + int((seg.get("frame_map") or {}).get("offset") or 0)
    f = P.main_root() / md / "0/trajectory_keypoints.npz"
    if not f.is_file():
        raise InputMissingError(
            f"[train_candidates] D31: no MANO wrist file {f} -- cannot judge a jump without the wrist"
        )
    with _np.load(f) as z:
        out = {}
        for sd in ("right", "left"):
            k = f"qpos_wrist_{sd}"
            if k in z.files:
                w = z[k][a : a + T, :3]
                if len(w) < T:
                    raise SystemExit(
                        f"[train_candidates] D31: {f} wrist {sd} length {len(w)} < tree {T} (start {a}) -- "
                        "windows disagree"
                    )
                out[sd] = _np.linalg.norm(_np.diff(w, axis=0), axis=-1) * fps
    if not out:
        raise InputMissingError(f"[train_candidates] D31: {f} has no MANO wrist")
    return out


def axis_sets(dataset: str, hand: str, emb: str) -> dict[str, set[str]]:
    """Axis key -> the trees that belong to that axis (what is judged and accepted there). The axes come from the
    profile:
    - aerial-scene datasets (arctic): candidates and routed trees share the one profile axis
    - desk-scene datasets: candidates on the canonical axis, `route:` trees on the `_aerial` variant axis"""
    reasons: dict = {}
    keep, _ = candidates(dataset, hand, emb, reasons=reasons)
    route = {t for t, why in reasons.items() if is_route(why)}
    pr = RR.DATASET_PROFILE[dataset]
    if pr.scene == "aerial":
        return {RR.ledger_key(dataset, emb): set(keep) | route}
    out = {RR.ledger_key(dataset, emb): set(keep)}
    if route:
        out[RR.ledger_key(dataset, emb, "_aerial")] = route
    return out


#: Markers whose absence on a tree is fatal at the end of a pass: quality (step 4b), D24 input (step 4),
#: D14' wrist (case.wrist_fp, step 3), support-gap marker (step 5b).
_MISSING_MARKERS = ("quality", "d24", "d14", "c640")


def candidates(dataset: str, hand: str, emb: str, reasons: dict | None = None) -> tuple[list[str], collections.Counter]:
    """(keep: candidate trees, drop: count per reason). With `reasons`, fills {tree: reason} for every tree not kept."""
    root = P.layer_path("base", dataset, hand=hand, emb=emb)
    if not root.is_dir():
        raise BaseLayerMissing(f"[train_candidates] no base layer: {root}")
    keep, drop = [], collections.Counter()
    missing: dict[str, list[str]] = {k: [] for k in _MISSING_MARKERS}  # trees without a required marker
    # trees the articulated converter excluded ({tree: word}, the generator's ledger lines): not judge inputs
    artic_excl, artic_hit = RR.retarget_excluded_by(dataset, _ARTIC_SRC), []
    ledger, ledger_hit = RR.retarget_excluded(dataset, P.hand_id(hand)), []  # whole ledger (hand-scoped: this hand)
    sup_artic: list[str] = []  # trees with an articulated support (SUPPORT_ARTIC)

    def _rej(t, key):
        drop[key] += 1
        if reasons is not None:
            reasons[t] = key

    names = sorted(os.listdir(root))
    split_parents = {n.rsplit("_t", 1)[0] for n in names if re.fullmatch(r".+_t\d+", n)}  # D36: a piece exists
    for t in names:
        f = root / t / "task_info.json"
        if not f.is_file():
            _rej(t, "no_task_info")
            continue
        d = json.loads(f.read_text())
        if t in artic_excl:
            artic_hit.append((t, artic_excl[t], _ARTIC_SRC in (d.get("rules") or {})))
            _rej(t, excluded_reason(artic_excl[t]))
            continue
        if support_articulated(d.get("objects") or []):
            sup_artic.append(t)
            _rej(t, SUPPORT_ARTIC)
            continue
        if t in ledger:  # the other ledger lines; the more specific reasons above come first
            ledger_hit.append(t)
            _rej(t, LEDGER_EXCLUDED)
            continue
        kind, key, infos = _rule_verdict(dataset, t, f.parent, d, t in split_parents)
        for info in infos:  # report-only counters, in the order the rules met them
            drop[info] += 1
        if kind == "keep":
            keep.append(t)
        elif kind == "drop":
            _rej(t, key)
        else:
            missing[kind].append(t)
    _artic_excluded_mark(dataset, emb, artic_excl, artic_hit, len(names), len(keep))
    _ledger_excluded_mark(dataset, emb, ledger, ledger_hit)
    _support_artic_mark(dataset, emb, sup_artic)
    _raise_missing_markers(f"{dataset}/{hand}/{emb}", dataset, len(names), missing)
    return keep, drop


def _rule_verdict(dataset: str, t: str, tdir: Path, d: dict, split_parent: bool) -> tuple[str, str | None, list[str]]:
    """One tree after the ledger checks -> (kind, drop reason, info counters). kind = "keep", "drop" or the name of
    a missing marker (_MISSING_MARKERS)."""
    sup = d.get("support") or {}  # the tree's own value (not the parent's)
    mode = desk_mode_of(sup, f"{dataset}/{t}")
    q = d.get("quality") or {}
    c = d.get("case") or {}
    # D23 before the route (D20/D18): a clip without a main object is no judge input on any axis.
    if no_main_object(tdir, d):
        return "drop", NO_MAIN_OBJECT, []
    # Scene choice: the desk verdict (D4/D20) picks a route (scene), not an exclusion. Only `route` and `scene` are
    # set here, so the later rules also apply to aerial-route clips. Rule scenes come from the registry.
    route, scene = None, "table"
    if mode not in DESK_OK:
        if not desk_route(dataset, d):  # desk dataset with desk_none / unknown = D4 drop
            return "drop", desk_reason(mode), []
        route, scene = AERIAL_DESK, "aerial"
    verdict = _marker_verdict(tdir, d, sup, q, c, scene, split_parent)
    if verdict is not None:
        return (*verdict, [])
    verdict, infos = _desk_verdict(dataset, tdir, d, sup, c, scene)
    if verdict is not None:
        return (*verdict, infos)
    if route:  # an aerial-route clip that passed every scene-independent rule
        return "drop", route, infos
    return "keep", None, infos + _partgroup_info(c, d)


def _marker_verdict(
    tdir: Path, d: dict, sup: dict, q: dict, c: dict, scene: str, split_parent: bool
) -> tuple[str, str | None] | None:
    """Quality, split, desk-oracle, case and duration rules -> None (passes) or (kind, drop reason)."""
    # no marker = step 4b (quality) did not run on this hand's base: not passed (every scene)
    if "train_candidate" not in q:
        return "quality", None
    if rule_on("D12-c", scene) and q.get("train_candidate") is False:
        return "drop", "not_train_candidate"
    if rule_on("D36", scene) and split_parent:
        return "drop", SPLIT_PARENT  # two tasks in one clip: the pieces _tN are the candidates
    od = sup.get("oracle_diff_m")  # D10-b GRAB raw desk oracle
    if rule_on("D10-b", scene) and od is not None and abs(float(od)) > ORACLE_DROP_M:
        return "drop", "desk_oracle>2cm"
    if not c:  # no case (step 3 output) -- every scene
        return "drop", "no_case"
    if rule_on("D12-a", scene) and any(str(c.get("case", "")).startswith(p) for p in CASE_DROP_PREFIX):
        return "drop", "case:BI_NONE"
    # Quality flags come from quality.flags, not the copy in case.flags (cases run before 4b on a new hand).
    bad = [fl for fl in (q.get("flags") or []) if fl in QUALITY_FLAGS_DROP]
    if rule_on("D12-c", scene) and bad:
        return "drop", quality_reason(bad[0])
    if rule_on("D12-d", scene) and float(c.get("T", 0)) / max(float(c.get("fps") or 30), 1.0) < MIN_DURATION_S:
        return "drop", "short<1s"
    if rule_on("D31", scene) and _d31_fast_object_slow_wrist(tdir, d) is not None:
        return "drop", OBJ_JUMP  # fast object, slow wrist
    return None


def _desk_verdict(
    dataset: str, tdir: Path, d: dict, sup: dict, c: dict, scene: str
) -> tuple[tuple[str, str | None] | None, list[str]]:
    """Desk-contact and support-gap rules (D24, D24', D14', D14-fb, gap) -> (None or (kind, reason), info counters)."""
    from dexengine.placement.coords import (
        core as _C,  # noqa: PLC0415  (start-embed geometry; lazy: placement sits above data)
    )

    infos: list[str] = []
    # D24 (collision hull): what the simulation presses against the desk top. Input = support.obj_zmin_hull_traj
    # written by object_roles (every frame, every vertex).
    _d24 = d24_state(tdir, d, rule_on("D24", scene))  # D23 trees left above, so no NA here
    if _d24 == D24_STALE:  # old input -- fatal at the end
        return ("d24", None), infos
    if _d24 == D24_UNMEASURED:  # basis=no_usd: no USD / collision mesh to measure
        return ("drop", "d24_unmeasured(no_usd)"), infos
    if _d24 == D24_BELOW:
        return ("drop", "obj_below_desk(D24)"), infos  # kinematic reference through the desk collider
    # D24': a main object not grasped at the start whose mesh bottom is more than START_EMBED_M below the desk is
    # embedded from the start. coords.start_embedded uses the same estimator as the D26' start discs
    # (coords.start_gaps, visual.obj initial pose); the sign makes it exclusive with D26'.
    if rule_on("D24′", scene) and sup.get("desk_z") is not None and _C.start_embedded(tdir, d):
        return ("drop", "start_embedded(D24′)"), infos
    wd = [v for v in (c.get("wrist_desk_m") or {}).values() if v is not None]
    # D14': drop only when the wrist is below the desk inside the real table footprint (+ margin). Applies when the
    # source has a table outline (support.desk_footprint_raw); value = this window's wrist (case.wrist_fp;
    # wrist_below_fp_m None = never inside the outline -> kept).
    if rule_on("D14'", scene) and sup.get("desk_footprint_raw"):
        if c.get("wrist_fp") is None:
            return ("d14", None), infos
        fp = c["wrist_fp"].get("wrist_below_fp_m")
        if fp is not None and float(fp) < WRIST_BELOW_M:
            return ("drop", "wrist_below_desk_in_footprint(D14')"), infos
        if wd and min(wd) < WRIST_BELOW_M:
            infos.append("info:wrist_below_desk_outside_footprint(kept)")
    # D14-fb: fallback when the table outline cannot be computed; its datasets are the registry scope
    # (D14_FB_SCOPE). Other datasets are only counted.
    elif rule_on("D14-fb", scene) and wd and min(wd) < WRIST_BELOW_M:
        if dataset in RR.rule_datasets("D14-fb"):
            return ("drop", "wrist_below_desk_no_footprint(D14-fb)"), infos
        infos.append("info:wrist_below_desk(kept)")
    if rule_on("C640", scene) and (
        _gk := SG.marker_kind(d, scene)
    ):  # unsupported head/middle/whole gap (D41 cuts tails)
        return (("c640", None) if _gk == SG.MISSING else ("drop", SG.reason(_gk))), infos
    return None, infos


def _partgroup_info(c: dict, d: dict) -> list[str]:
    """D1-f: a used unit that is a part group with a moving part (fit != rigid) loses that part when only main
    objects spawn -> counted only (one counter for the first used part group)."""
    used = set(c.get("used") or [])
    pg = c.get("part_groups") or {}
    objs = {o["name"]: o for o in d.get("objects") or []}
    for u in used & set(pg):
        fits = [((objs.get(m, {}).get("articulation_group") or {}).get("fit") or {}).get("type") for m in pg[u]]
        return [
            "info:partgroup_moving(kept)" if any(f in PART_MOVING_FITS for f in fits) else "info:partgroup_static(kept)"
        ]
    return []


def _raise_missing_markers(axis: str, dataset: str, n_trees: int, missing: dict[str, list[str]]) -> None:
    """A required marker missing on any tree is fatal: name the step that writes it."""
    no_d24, no_d14, no_c640, no_quality = missing["d24"], missing["d14"], missing["c640"], missing["quality"]
    if no_d24:
        raise SystemExit(
            f"[train_candidates] {axis}: {len(no_d24)}/{n_trees} desk trees lack the D24 "
            f"input marker (rules.D24 version {D24_INPUT_VERSION}), e.g. {no_d24[0]} -- rerun step 4 "
            f"(stages --emit-run 4 = object_roles.py --datasets {dataset} --write)"
        )
    if no_d14:
        raise SystemExit(
            f"[train_candidates] {axis}: {len(no_d14)}/{n_trees} table-outline trees lack "
            f"the D14' wrist marker (case.wrist_fp), e.g. {no_d14[0]} -- rerun step 3 (stages --emit-run 3)"
        )
    if no_c640:
        raise SystemExit(
            f"[train_candidates] {axis}: {len(no_c640)}/{n_trees} aerial trees lack a current "
            f"support-gap marker (rules.C640 version {SG.VERSION}), e.g. {no_c640[0]} -- stages --emit-run 5b"
        )
    if no_quality:
        raise SystemExit(
            f"[train_candidates] {axis}: {len(no_quality)}/{n_trees} trees lack the quality "
            f"marker (quality.train_candidate), e.g. {no_quality[0]} -- run step 4b first "
            f"(stages --emit-run 4b = filtering.py --hand <hand> --dataset {dataset})"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hand", default=RR.hand_dir(), help="retargeting hand spec dir (base layer axis)")
    ap.add_argument("--emb", required=True, choices=("right", "left", "bimanual"), help="no default")
    ap.add_argument("--out", default="", help="list file (default: stdout)")
    a = ap.parse_args()
    RR.hand_thresholds()  # logs the thresholds this run uses
    keep, drop = candidates(a.dataset, a.hand, a.emb)
    total = len(keep) + sum(v for k, v in drop.items() if not k.startswith("info:"))  # info: counters are reports only
    route = sum(v for k, v in drop.items() if k.startswith(ROUTE_PREFIX))  # routed, not dropped (D20)
    print(
        f"[CANDIDATES] dataset={a.dataset} emb={a.emb} hand={a.hand} total={total} keep={len(keep)} route={route} "
        f"drop={dict(drop)} rule=D4(desk∈{DESK_OK})+train_candidate+D10b(oracle>{ORACLE_DROP_M}m)+D12(case∉"
        f"{CASE_DROP_PREFIX},quality∉{len(QUALITY_FLAGS_DROP)}flags,dur≥{MIN_DURATION_S}s)+D14'/D14-fb("
        f"{RR.rule_datasets('D14-fb')} "
        f"wrist≥{WRIST_BELOW_M}; fb=footprint unknown)+D1f(main-only spawn)",
        file=sys.stderr,
        flush=True,
    )
    print(SG.marker(drop), file=sys.stderr, flush=True)
    text = "\n".join(keep) + ("\n" if keep else "")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
