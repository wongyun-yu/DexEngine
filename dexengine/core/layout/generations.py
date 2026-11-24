"""Preserved and compare generations of the data: named rosters (frozen training lists), stage roots outside the
layer table, and compare-generation data roots that sit beside the canonical root."""

from __future__ import annotations

import json
import os

from dexengine.core.layout.registry import DATA_ROOT_ENV, DATASETS

#: **Named roster sources** — training rosters of frozen generations that have no per-hand folded-ledger folder
#: (`run_ledger`); read by dexengine/placement/train_ready_util.py named_roster. The training entry uses a source only
#: when the caller names it with `DEXENGINE_ROSTER=<name>` — never picked from the combo (no silent substitution).
#: Source = the roster files matching the `files` glob under the views layer (one suffix-free clip name per line);
#: a file count or unique-clip count that differs from the registration raises.
#: `origin` = where the generation's data roots were **before** they were moved (current place = roster_root).
NAMED_ROSTERS: dict[str, dict] = {
    "v1_o64": {
        "dataset": "taco",
        "emb": "bimanual",
        "combo": "rby1_sharpa_wave",
        "files": "o64_[0-9][0-9].txt",
        "n_files": 34,
        "n_clips": 2120,
        "per_clip": "overfit/o64_stats/per_clip.tsv",  # frozen per-clip results (V1 is never re-run)
        "runs": "O64_",  # RL run-name prefix of the generation (O64_<NN>_… folders), read by the delete guard
        "origin": {"base": "data/taco/trajectories", "trees": "data/taco/trajectories_gen_v2/rby1_sharpa_wave"},
        "why": "V1 frozen generation: union of the 34 bundle rosters = 2,120 V1 adopted clips",
    },
}
#: Roster name of V1 (official, frozen) — V1 consumers find roster_root through this one name.
V1_ROSTER = "v1_o64"
#: Name of the preservation folder — preserved generations live **outside** the layers, at
#: `<top>/<next>/_generations/<generation>/<rest>`.
GENERATIONS_DIR = "_generations"

#: **Stage roots** outside the layer table — every place where dataset derivatives live outside a layer address
#: (single owner; read by dexengine.core.paths.stage_root). The hand axis is the **hand name** ({handname}, not the
#: spec dir). The real entries of that folder must have one of these forms (stage_kind; others fail archiving).
STAGE_DIR = "data/_stage"
STAGE_ROOTS: dict[str, str] = {
    "retarget_stage": STAGE_DIR + "/reretarget_{handname}",  # per-hand retargeting stage (…/processed/<ds>)
    "reread_stage": STAGE_DIR + "/reread_{ds}",  # source re-read stage (reader, contact, first retarget)
    "hand_vendor": STAGE_DIR + "/{handname}",  # hand vendor assets (URDF, meshes), not a derivative
}


def stage_kind(name: str, hands, datasets) -> str | None:
    """Entry name in `data/_stage` → its STAGE_ROOTS kind (the preservation folder GENERATIONS_DIR maps to itself),
    None when it has none of the forms. Only registered hand and dataset names match (the caller passes the registries;
    no name globs).

    >>> stage_kind("reretarget_wuji", {"wuji"}, {"hot3d"}), stage_kind("reread_hot3d", {"wuji"}, {"hot3d"})
    ('retarget_stage', 'reread_stage')
    >>> stage_kind("wuji", {"wuji"}, {"hot3d"}), stage_kind("reretarget_nope", {"wuji"}, {"hot3d"})
    ('hand_vendor', None)
    """
    if name == GENERATIONS_DIR:
        return GENERATIONS_DIR
    for kind, tpl in STAGE_ROOTS.items():
        leaf = tpl[len(STAGE_DIR) + 1 :]
        if any(leaf.format(handname=h, ds="\0") == name for h in hands) or any(
            leaf.format(handname="\0", ds=d) == name for d in datasets
        ):
            return kind
    return None


def generation_rel(rel: str, label: str) -> str:
    """The one preservation naming rule. A root or generation name outside the
    rule raises ValueError.

    >>> generation_rel("data/taco/trajectories", "v1_o64")
    'data/taco/_generations/v1_o64/trajectories'
    """
    parts = str(rel).split("/")
    if (
        len(parts) < 3
        or any(x in ("", ".", "..") for x in parts)
        or GENERATIONS_DIR in parts
        or not label
        or "/" in label
        or label in (".", "..")
    ):
        raise ValueError(
            f"outside the preservation naming rule: root {rel!r} · generation {label!r} "
            f"(`<top>/<next>/<rest…>`, not inside a preserved copy, generation = one segment)"
        )
    return "/".join(parts[:2] + [GENERATIONS_DIR, label] + parts[2:])


def roster_root(name: str, kind: str) -> str:
    """**Current** place of the root of frozen generation `name` (`kind` = an origin layer, base or trees)
    = generation_rel(origin, name). Frozen-generation consumers find it only through this (never by joining the origin
    text). An unknown name or layer raises KeyError.

    >>> roster_root("v1_o64", "trees")
    'data/taco/_generations/v1_o64/trajectories_gen_v2/rby1_sharpa_wave'
    """
    if name not in NAMED_ROSTERS:
        raise KeyError(f"unknown roster source {name!r} — registered: {sorted(NAMED_ROSTERS)}")
    origin = NAMED_ROSTERS[name]["origin"]
    if kind not in origin:
        raise KeyError(f"roster source {name}: no root for layer {kind!r} — registered {sorted(origin)}")
    return generation_rel(origin[kind], name)


def taco_source_root() -> str:
    """Place of the TACO source tree (MANO and object trajectories — the original exists on no machine), read-only.
    The stage-1 lift input, reference material (D12c), the RL passive-hand check and the placement tools all find it
    through this (it is not the base layer of the layer table). It is the V1 base preserved root — to delete V1,
    point this line elsewhere and move the directory.

    >>> taco_source_root()
    'data/taco/_generations/v1_o64/trajectories'
    """
    return roster_root(V1_ROSTER, "base")


def frozen_datasets() -> frozenset[str]:
    """Datasets read by frozen generations (NAMED_ROSTERS) — their object assets (`convex/`, object USD) are never
    rebuilt.

    A rule, not a name list: the answer is the datasets of the frozen rosters. The object USD layer is hand-independent,
    so V2 reads the same files — a re-bake would silently change the frozen generation's evaluation conditions. The
    object bake and the decomposition entry (dexengine/data/objects/decompose.py) refuse through this function.
    >>> sorted(frozen_datasets())
    ['taco']
    """
    return frozenset(spec["dataset"] for spec in NAMED_ROSTERS.values())


# ── Compare-generation data roots — named roots that re-run the same chain without overwriting the canon ──
#: A compare generation re-runs the same chain (reader → stage 1 → judge/bake → RL bundles) without overwriting
#: the canonical data and compares its RL results with the canon (replacing the canon is a separate decision). One
#: generation = a **separate data root** in the folder `<canonical root name>_<generation>` **beside** the canonical
#: data root (compare_root), written at the same relative places as the layer table.
#: The chain runs with that root in DATA_ROOT_ENV, so every data-root reader sees the same value and an input missing
#: from that root fails as missing (never silently falls back to the canon).
#: The root's identity is the GENERATION_FILE inside it (written when the root is built, read only by
#: data_generation); the canonical root has no such file. Run layers (RUN_LAYERS, RUN_SHARED — ledgers, audits,
#: campaigns) also live in the generation root (place owner: dexengine.core.paths.run_root).
#:   kind     compare = for comparison: does not change the canonical clip ownership (owner ledger) — the queue and
#:            the owner ledger read this field
#:   dataset  the dataset this generation rebuilds (no other dataset layers exist in that root)
#:   why      why it is built (the reason for the registration — required)
#:   goal     goal line of the RL-status comparison column — when absent, why is that line
#:   subjects subjects rebuilt by this generation (comparison population, adoption scope) — when absent, the whole
#:            dataset
#: Generation readers (e.g. the RL-status comparison column) **derive** from this table — no second name table.
GENERATION_FILE = "DATA_GENERATION.json"
GENERATION_KINDS = ("compare",)
COMPARE_GENERATIONS: dict[str, dict] = {
    "hot3d_lfix": {
        "kind": "compare",
        "dataset": "hot3d",
        "why": "HOT3D left-hand MANO defect fix (official toolkit rule, reader hot3d.mano_models) — "
        "rebuild the left-hand-active clips of the 3 large-b0 subjects (P0009, P0012, P0015) "
        "and compare them with the canon in RL",
        "goal": "HOT3D left-hand MANO fix — compare old vs new success on the same clips, then replace per hand",
        "subjects": ("p0009", "p0012", "p0015"),
    },
    "dexcanvas_rf": {
        "kind": "compare",
        "dataset": "dexcanvas",
        "why": "DexCanvas object Euler angles = intrinsic XYZ (OBJECT_EULER) — re-run the whole reader, "
        "build the new generation beside the canon (read, intake, stage 1, judge) "
        "and swap it with the canon at the bake/archive gate",
    },
    "oakink2_artic": {
        "kind": "compare",
        "dataset": "oakink2",
        "why": "per-object joint specs + intake and extra-slot repairs, built beside the canon and "
        "compared with the approved videos",
    },
}

#: The from-raw release gate (dexengine.release.fromraw) rebuilds sample clips of one dataset from the raw downloads
#: and the shipped inputs in a scratch root and compares them with the released data: one compare generation per
#: dataset, by rule (a new dataset gets its own without a table edit).
FROMRAW_PREFIX = "fromraw_"
COMPARE_GENERATIONS.update(
    {
        FROMRAW_PREFIX + ds: {
            "kind": "compare",
            "dataset": ds,
            "why": "from-raw release gate: rebuild sample clips from the raw downloads and the shipped inputs in a "
            "scratch root, then compare them with the released data (dexengine.release.fromraw)",
        }
        for ds in DATASETS
    }
)


def generation_record(name: str) -> dict:
    """Identity record of generation `name` (the content of GENERATION_FILE) — an unknown name raises KeyError."""
    if name not in COMPARE_GENERATIONS:
        raise KeyError(f"unknown compare generation {name!r} — registered: {sorted(COMPARE_GENERATIONS)}")
    g = COMPARE_GENERATIONS[name]
    if g["kind"] not in GENERATION_KINDS:
        raise KeyError(f"compare generation {name}: kind {g['kind']!r} is not in {GENERATION_KINDS}")
    return {"name": name, "kind": g["kind"], "dataset": g["dataset"]}


def compare_root(canon_root: str, name: str) -> str:
    """Data root (absolute) of generation `name` — `<its name>_<generation>` beside the canonical data root
    `canon_root`. The canonical root must be the main checkout (`.git` is a folder); a worktree (code only, the data
    root comes from env) or an existing generation root raises, so a generation is never built in the wrong place."""
    generation_record(name)
    c = os.path.normpath(os.path.abspath(canon_root))
    if data_generation(c) is not None:
        raise SystemExit(
            f"[DATA_GEN] {c} is already a generation root ({GENERATION_FILE}) — pass the canonical data root"
        )
    if not os.path.isdir(os.path.join(c, ".git")):
        raise SystemExit(
            f"[DATA_GEN] {c} is not the main checkout (.git is not a folder) — from a worktree, pass "
            f"the canonical data root as {DATA_ROOT_ENV}=<main checkout>"
        )
    return os.path.join(os.path.dirname(c), f"{os.path.basename(c)}_{name}")


def data_generation(root: str) -> dict | None:
    """Generation of data root `root` — None without GENERATION_FILE (canonical). Otherwise it must equal the
    registration (generation_record): an unknown name, kind or dataset raises SystemExit (the chain never runs on a
    root of unknown identity)."""
    f = os.path.join(str(root), GENERATION_FILE)
    if not os.path.isfile(f):
        return None
    with open(f) as fh:
        rec = json.load(fh)
    try:
        want = generation_record(rec.get("name", "") if isinstance(rec, dict) else "")
    except KeyError as e:
        raise SystemExit(f"[DATA_GEN] {f}: {e}") from None
    if rec != want:
        raise SystemExit(f"[DATA_GEN] {f}: record {rec} ≠ registration {want}")
    return rec
