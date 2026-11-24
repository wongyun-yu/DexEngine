"""First-pass candidate rules: reason words, ledger marks, the D24 desk verdict, scene/route helpers and the thresholds
(owner of the values; candidates.py applies them clip by clip)."""

from __future__ import annotations

import collections
import json
import sys

from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.data.artic.convert_version import EXCLUDE_SRC as _ARTIC_SRC
from dexengine.data.desk_penetration import D24_VERSION as D24_INPUT_VERSION
from dexengine.data.oakink2 import part_kinds as _PK
from dexengine.data.support import plane as SA

DESK_OK = (SA.DESK_TABLE, SA.DESK_RECOVERED)  # D4


# D20 route: a clip whose desk is not the work surface is not dropped but routed. It leaves the canonical (desk) axis
# and is judged and trained on the aerial variant axis `<ds>-aerial`. Its reason starts with this prefix.
ROUTE_PREFIX = "route:"


def is_route(reason: str | None) -> bool:
    """Is this reason a route (another axis) rather than an exclusion?"""
    return bool(reason) and str(reason).startswith(ROUTE_PREFIX)


#: D23 reason: no hand has a main object. Also the out-of-domain value of `d24_state`.
NO_MAIN_OBJECT = "no_main_object(D23)"


#: D36 reason: a parent clip with two tasks (its pieces `_tN` are the candidates instead).
SPLIT_PARENT = "split_parent(D36)"


def support_articulated(objs: list) -> list[str]:
    """Names of the support objects (role support) that are articulated; empty when none. Candidate selection drops
    such trees (SUPPORT_ARTIC) and the support-asset bake does not count them."""
    return [str(o.get("name")) for o in objs or [] if o.get("role") == "support" and o.get("articulation")]


def desk_reason(mode: str) -> str:
    """D4 reason for a desk dataset's tree without a known support surface (desk_mode not in DESK_OK)."""
    return f"desk:{mode}"


def excluded_reason(word: str) -> str:
    """Reason for a tree the articulated converter excluded (its word in the exclusion ledger). A word outside
    part_kinds.EXCLUDE_WORDS is fatal."""
    _PK.exclude_label(word)
    return f"artic:{word}"


#: Reason for a tree listed in the exclusion ledger (precompute/<ds>/diag/excluded_clips.txt) by a line other than the
#: articulated converter's (those use excluded_reason).
LEDGER_EXCLUDED = "ledger:excluded"


#: Reason for a tree whose support object is articulated: support baking and the judge's support measure handle rigid
#: bodies only.
SUPPORT_ARTIC = excluded_reason("support_articulated")


def _artic_excluded_mark(dataset: str, emb: str, listed: dict, hit: list, n_trees: int, n_keep: int) -> None:
    """Finish applying the converter's exclusion lines. hit = [(tree, word, has a conversion record)]. A listed tree
    that also has a conversion record (rules.<generator>) means ledger and tree disagree: fatal. With a ledger, print
    one marker line (stderr: stdout carries the list)."""
    clash = [t for t, _w, done in hit if done]
    if clash:
        raise SystemExit(
            f"[train_candidates] {dataset}/{emb}: {len(clash)}/{n_trees} trees are in the exclusion ledger "
            f"(generator {_ARTIC_SRC}) and also have a conversion record (rules.{_ARTIC_SRC}), e.g. "
            f"{clash[0]} -- ledger and trees disagree; rerun step 2b (oakink2/artic_convert.py --write) "
            "for this hand"
        )
    if listed:
        by = collections.Counter(w for _t, w, _d in hit)
        print(
            f"[ARTIC_EXCLUDE] skip n={len(hit)} by_reason={json.dumps(dict(sorted(by.items())), ensure_ascii=False)}"
            f" · ledger {len(listed)} lines · {dataset}/{emb} · candidates {n_keep}",
            file=sys.stderr,
            flush=True,
        )


def _ledger_excluded_mark(dataset: str, emb: str, listed: dict, hit: list) -> None:
    """Finish applying the whole exclusion ledger: one marker line when a ledger exists (stderr)."""
    if listed:
        print(
            f"[LEDGER_EXCLUDE] {dataset}/{emb} dropped {len(hit)} · ledger {len(listed)} lines"
            f"({RR.retarget_excluded_path(dataset)}) · e.g. {hit[:3]}",
            file=sys.stderr,
            flush=True,
        )


def _support_artic_mark(dataset: str, emb: str, hit: list) -> None:
    """Finish the articulated-support rule: one marker line when trees were dropped (stderr)."""
    if hit:
        print(
            f"[SUPPORT_ARTIC] {dataset}/{emb} dropped {len(hit)} -- articulated support object · e.g. {hit[:3]}",
            file=sys.stderr,
            flush=True,
        )


def quality_reason(flag: str) -> str:
    """D12-c reason: the first quality flag of the tree that is in QUALITY_FLAGS_DROP."""
    return f"quality:{flag}"


def no_main_object(tree_dir, ti: dict) -> bool:
    """D23: no hand manipulates a main object (``qpos_layout.Layout.manip_sides`` is empty, the same predicate f0, the
    judge and collision refuse on). `ti` is the tree's task_info, already read by the caller."""
    return not P.qpos_layout.layout_of(str(tree_dir), ti).manip_sides()


AERIAL_DESK = "route:aerial(D20 no desk)"


#: The first-pass rules applied here (the set of `on(<rule>)` calls in `candidates`). The scenes each applies to are the
#: registry's `scenes`; `rules --verify` checks that every id is a registry rule.
FIRST_PASS_RULES = ("D12-c", "D36", "D10-b", "D12-a", "D12-d", "D31", "D24", "D24′", "D14'", "D14-fb", "C640")


def _scenes(rid: str) -> tuple:
    """Rule -> the scenes it applies to (rules.rule_scenes). A rule outside FIRST_PASS_RULES is fatal."""
    if rid not in FIRST_PASS_RULES:
        raise KeyError(f"dexengine.data.candidates: rule {rid!r} is not in FIRST_PASS_RULES -- add it there")
    return RR.rule_scenes(rid)


def rule_on(rid: str, scene: str) -> bool:
    """Does first-pass rule `rid` apply to scene `scene`?"""
    return scene in _scenes(rid)


class BaseLayerMissing(SystemExit):
    """This (dataset, hand, emb) has no base layer: the only failure a caller may skip as "not applicable". A missing
    marker is a defect and stays a plain SystemExit."""


# D12: first-pass drops by case and quality (the case fields are task_info.case written by cases/taxonomy.py).
CASE_DROP_PREFIX = ("BI_NONE",)  # D12-a no manipulation (non-grasp touch cases are dropped with it)


QUALITY_FLAGS_DROP = (
    "retarget_error",
    "keypoint_glitch",
    "object_moves_without_contact",
    "no_object_contact",
    "object_gt_gap",
)  # D12-c · object_gt_gap: a long gap in the main object's source GT


MIN_DURATION_S = 1.0  # D12-d


WRIST_BELOW_M = -0.05  # D14: lowest wrist point this far below the support surface


# D31: main-object pose jumps (source tracking defects). A run of steps where the main object moves faster than
# OBJ_SPEED_MAX_MPS (any length) while the MANO wrist (hand-independent) stays below WRIST_SLOW_MPS within +-2 steps is
# a
# tracking defect: a held object cannot move several times faster than the wrist. A fast wrist is real motion
# (hammering, shaking a bottle) and stays. Source GT gaps are handled by the reader (interpolation) or the quality flag
# object_gt_gap. The verdict lives in `_d31_fast_object_slow_wrist`.
OBJ_SPEED_MAX_MPS = 3.0  # fast run: main-object speed above this (p99 of accepted human manipulation ~3.16)


WRIST_SLOW_MPS = 1.5  # wrist max speed in that run (+-2 steps) below this = tracking defect


#   measured: grab real motion 2.0-4.7 m/s wrist · oakink2 jumps 0.03-0.44 m/s
OBJ_JUMP = f"obj_jump>{OBJ_SPEED_MAX_MPS:.0f}m/s(D31)"  # D31 reason (built from the threshold)


OBJ_BELOW_M = 0.02  # D24: main-object collision-hull lowest z over the trajectory (support.obj_zmin_hull_traj,


#: D24 states of one tree (`d24_state`): measured OK / BELOW, out of domain (NA), or why it was not measured.
D24_OK, D24_BELOW = "ok", "below"  # hull min - desk_z >= -OBJ_BELOW_M / < -OBJ_BELOW_M


D24_UNMEASURED = "unmeasured"  # main object present, no hull value (basis no_usd)


D24_STALE = "stale"  # rules.D24 version != D24_INPUT_VERSION


D24_NO_DESK = "no_desk"  # no support.desk_z


D24_OFF_SCENE = "off_scene"  # D24 does not apply to this tree's scene


D24_NA_D23 = NO_MAIN_OBJECT  # no main object (D23): D24 is undefined


D24_MEASURED = (D24_OK, D24_BELOW)


D24_NA = (D24_NA_D23,)


def d24_state(tree_dir, ti: dict, applies: bool) -> str:
    """D24 (main-object collision hull through the desk top) state of one tree. `applies` = D24 applies to the
    caller's scene (`rule_on("D24", scene)`). Order (first wins): no main object (D23) -> off scene -> no desk_z ->
    stale input -> no hull value -> the measured comparison."""
    if no_main_object(tree_dir, ti):
        return D24_NA_D23
    if not applies:
        return D24_OFF_SCENE
    sup = ti.get("support") or {}
    if sup.get("desk_z") is None:
        return D24_NO_DESK
    if ((ti.get("rules") or {}).get("D24") or {}).get("version") != D24_INPUT_VERSION:
        return D24_STALE
    oz = sup.get("obj_zmin_hull_traj")
    if oz is None:
        return D24_UNMEASURED
    return D24_BELOW if float(oz) - float(sup["desk_z"]) < -OBJ_BELOW_M else D24_OK


# D1-f: the first RL scene spawns only the left/right main objects, not the other parts of a part group. Candidates with
# a moving part are only counted (info).
PART_MOVING_FITS = ("revolute", "prismatic", "screw", "free")


def desk_mode_of(sup: dict, where: str) -> str:
    """The tree's support.desk_mode; a value outside support.plane.DESK_MODES is fatal."""
    mode = sup.get("desk_mode")
    if mode not in SA.DESK_MODES:
        raise SystemExit(
            f"[train_candidates] {where}: support.desk_mode {mode!r} not in DESK_MODES {SA.DESK_MODES} "
            "-- rerun the support step (support audit, desk recovery)"
        )
    return mode


def desk_route(dataset: str, ti: dict) -> bool:
    """D20: no desk is this dataset's normal form (profile scene=aerial). In a desk dataset desk_none is a D4 drop. A
    dataset without a profile is fatal (KeyError)."""
    scene = RR.DATASET_PROFILE[dataset].scene
    return (ti.get("support") or {}).get("desk_mode") == SA.DESK_NONE and scene == "aerial"


def scene_of(dataset: str, ti: dict) -> str:
    """The tree's scene: "aerial" for a D20 tree, else "table" (a desk dataset is a table scene whatever the hand
    height)."""
    return "aerial" if desk_route(dataset, ti) else "table"
