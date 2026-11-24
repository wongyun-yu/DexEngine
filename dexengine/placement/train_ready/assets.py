"""Stateless helpers of the trainability survey: the object asset table and the asset check, the role fingerprint,
the fingerprinted ledger reader, the T5-judge row scope, the 1st-pass exclusion rule, and the code files whose bytes
key the T5 caches.

The helpers of the train-ready survey; their owners are today's modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from dexengine.core import artifact_paths as AP
from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.placement import final_ledger as FL
from dexengine.placement import fold_rules
from dexengine.placement import ledger_states as LS
from dexengine.placement.judge import input_record as JFP

REPO = Path(__file__).resolve().parents[3]

#: The assets one object must have before it can be used in training — (display name, layer, file name).
#: When a consumer is added, add it here.
OBJ_ASSETS = (
    ("usd", "usd", "object.usd"),
    ("surface", "meshes", "surface_points.npz"),
    ("keypoints", "meshes", "keypoints_fps2.npy"),
)

#: T3 — an articulated object cannot be spawned from a rigid USD. An object with `objects[].articulation` in the
#: tree needs its own articulated USD (base link + moving part link + 1 joint) before it comes up as an
#: ArticulationCfg. The bake is `dexengine/robots/objects/usd_bake.py --articulated`; the folder is the asset id
#: (artifact_paths.object_asset_id) and the file name's owner is artifact_paths.TacoPaths.ARTIC_USD_NAME.
ARTIC_ASSET = ("usd_artic", "usd", AP.TacoPaths.ARTIC_USD_NAME)

#: Per-dataset USD folder exceptions — the owner is the rule registry's dataset profile.
USD_DIR_OVERRIDE = {k: v.usd_dir for k, v in RR.DATASET_PROFILE.items() if v.usd_dir}


def object_asset(ds: str, o: dict) -> tuple[str, list[str]]:
    """(asset id, T3 misses) of one task_info objects[] entry. Asset id = artifact_paths.object_asset_id(obj_id,
    articulation) (the usd · meshes folder; the object id for a rigid object). An articulated entry whose joint is
    not the committed spec's canonical one misses `artic_not_canonical` (rule owner artic.ranges.not_canonical)."""
    art = o.get("articulation")
    aid = AP.object_asset_id(o["obj_id"], art)
    if not art:
        return aid, []
    from dexengine.data.artic import ranges as artic_ranges  # noqa: PLC0415 — numpy · scipy, only for a joint

    return aid, ["artic_not_canonical"] if artic_ranges.not_canonical(ds, o["obj_id"], art) else []


def missing_assets(ds: str, obj_id: str, articulated: bool = False, aerial: bool = False) -> list[str]:
    """Names of the assets this object (`obj_id` = its asset id, object_asset) is missing; empty = trainable.
    With `articulated`, the articulated USD (T3) replaces the rigid one (the env spawns an ArticulationCfg from
    `object_artic.usd`; a rigid `object.usd` is not consumed). With `aerial` (D20 tree), the env must have the
    deskless scene wired (E-1) — the owner of both statements is the registry."""
    out = []
    if articulated and not RR.ARTIC_ENV_WIRED:
        out.append("env_artic")  # the env cannot spawn an articulated object yet, asset or not
    if aerial and not RR.AERIAL_ENV_WIRED:
        out.append("env_aerial")
    assets = tuple(a for a in OBJ_ASSETS if a[0] != "usd") + (ARTIC_ASSET,) if articulated else OBJ_ASSETS
    root = str(P.main_root())
    for name, layer, fn in assets:
        if name == "usd" and ds in USD_DIR_OVERRIDE:
            ok = os.path.isfile(os.path.join(root, USD_DIR_OVERRIDE[ds], obj_id, fn))
        else:
            ok = os.path.isfile(os.path.join(root, "data", ds, layer, obj_id, fn))
        if not ok:
            out.append(name)
    return out


def roles_of(ti: dict) -> list:
    """Role fingerprint — (name, role, side). These three fix the reward target, contact derivation and spawn set."""
    return sorted((o.get("name"), o.get("role"), o.get("side")) for o in (ti.get("objects") or []))


def judged_active(base_dir, bti: dict) -> tuple:
    """The T4-b reference: the active sides the judge would pick on the current base. Same owner as the judge
    (collision.idle_arm.active_idle_for_tree) with the same inputs (case · the base layout's hands · manip_sides).
    Comparing with the raw case.active would leave a tree with an assisting hand (active, no main object — the judge
    idles it) stale forever. With no manipulating hand among the active ones the judge cannot stand on this base → ()
    (differs from every tree, so stale — the re-judge dies with that reason)."""
    from dexengine.placement.collision import idle_arm as COL  # noqa: PLC0415 — numpy only

    lay = P.qpos_layout.layout_of(str(base_dir), bti)
    try:
        return COL.active_idle_for_tree(bti.get("case"), tuple(lay.hands), tuple(lay.manip_sides()))[0]
    except ValueError:
        return ()


def ledger_rows(path: str) -> list[tuple[str, str, str]]:
    """Every row of a final ledger as (clip, status, judge-input fingerprint). Reading has one owner
    (final_ledger.read_final_rows); the fingerprint column is found by its header name (JFP.COLUMN), and an old
    ledger without it (4 columns) or a short row gives ""."""
    return [
        (r["clip"], r["status"], r.get(JFP.COLUMN) or "") for r in FL.read_final_rows(path) if r["status"] is not None
    ]


def adopted(path: str) -> list[str]:
    """ADOPTED clips of a final ledger, in file order (reader final_ledger.read_final_rows)."""
    return [r["clip"] for r in FL.read_final_rows(path) if r["status"] == "ADOPTED"]


def t5j_scope(status: str) -> bool:
    """Is this row compared by its judge-input fingerprint — the judge's non-adoptions (ledger_states.JUDGED_STATUSES)
    and every adoption, with or without a tree (once nobody read the fingerprint of an adoption with a tree, so
    adoptions judged before the object reached the judging machine (objcol=error) stayed)."""
    return status == "ADOPTED" or status in LS.JUDGED_STATUSES


def first_pass_out(ds: str, clip: str, variant: bool, excluded: set, route: set) -> bool:
    """The 1st-pass exclusion of one clip on its axis — one rule for the survey loop and the prefetch job lists.
    Canonical axis: a clip with a 1st-pass reason is out (a route clip is out on this axis too). Variant axis (D20
    aerial · 2c artic): out only with a real exclusion reason; a route clip is this axis' target and stays."""
    return (ds, clip) in excluded or (not variant and (ds, clip) in route)


def is_carry_note(note: str | None) -> bool:
    """A hand-carried row (its note is `hand_carry(<source hand>:<category>)`, owner fold_rules.CARRY_NOTE_HEAD)."""
    return bool(note) and note.startswith(fold_rules.CARRY_NOTE_HEAD)


def first_pass_reasons(datasets, hand: str) -> dict:
    """(dataset, clip) → 1st-pass reason (candidates' reasons) on this hand's base, over every emb of the dataset.
    Only a missing base layer (BaseLayerMissing) is skipped — a defect dies."""
    from dexengine.data import candidates as TC  # noqa: PLC0415 — the rule and route owner

    out: dict = {}
    for ds in sorted(datasets):
        for emb in RR.DATASET_PROFILE[ds].embs:
            reasons: dict = {}
            try:
                TC.candidates(ds, hand, emb, reasons=reasons)
            except TC.BaseLayerMissing:
                continue
            for clip, why in reasons.items():
                out[(ds, clip)] = why
    return out


def _py_files(*rel: str) -> list[Path]:
    """The .py files named by `rel` (a file, or every .py of a folder), under the repo."""
    out = []
    for r in rel:
        p = REPO / r
        out += sorted(p.glob("*.py")) if p.is_dir() else [p]
    return out


def rule_sources() -> list[Path]:
    """The code the T5 setup computation runs (part of the T5 cache key): the coords package, the rest rule, the
    stamp reader (qpos_layout), the desk owner, and the pedestal rules (extra slots · scene requirement · static
    fixed · static desk · place rule). A rule edit changes these bytes, so a cached verdict never outlives it."""
    return _py_files(
        "dexengine/placement/coords",
        "dexengine/data/support/rest.py",
        "dexengine/core/qpos_layout.py",
        "dexengine/core/desk.py",
        "dexengine/data/objects/extra_slots.py",
        "dexengine/data/objects/extra_slots_geom.py",
        "dexengine/data/objects/scene_requirement.py",
        "dexengine/placement/scene/static_fixed.py",
        "dexengine/placement/scene/static_desk.py",
        "dexengine/placement/scene/place_rule.py",
    )


def code_sources() -> list[Path]:
    """The code that computes a clip's judge-input parts (setup · case · contact · base) — the T5-judge cache key.
    Run parts (rs · rules · budget · collider) are not cached (measured fresh every run)."""
    extra = _py_files(
        "dexengine/placement/judge/input_record.py",
        "dexengine/placement/collision/core.py",
        "dexengine/data/contact/fingerprint.py",
        "dexengine/data/contact/fingerprint_core.py",
        "dexengine/data/readers/h2o_frame.py",
        "dexengine/core/paths.py",
        "dexengine/data/cases/hand_object_assign.py",
    )
    return sorted(set(rule_sources()) | set(extra), key=str)
