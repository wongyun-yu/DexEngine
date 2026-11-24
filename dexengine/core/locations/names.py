"""Clip and tree names: the suffixes of each output kind, clip ids built from raw dataset fields, derived
(variant) clips and IK-pool names."""

from __future__ import annotations

import os
import re
import unicodedata

from dexengine.core.locations.modules import data_layout
from dexengine.core.repo import REPO_ROOT

# ── Naming-rule constants (the literals live here only) ─────────────────────
TREE_SFX = "_pl_rt"  # placement+retime tree suffix. Every combo output uses it.
PLACEMENT_SFX = ".adopted.json"
IK_SIDECAR_SFX = ".pre_rt.npz"  # arm IK **before** retiming (left behind by retime.py)
WRIST_SIDECAR_SFX = ".pre_f0r.npz"  # wrist **before** the f0 repair (left by an f0 re-repair)
PL_SFX = "_pl"  # tree with placement applied, **before** retiming (retime's input)
TREE_F0_SFX = TREE_SFX + "_f0"  # tree with the frame 0 repair applied too (diagnostics and the
#   training list look at this)
IKC_ANCHOR_SEP = "_a"  # the anchor separator in a judging IK cache `<clip>_a<anchor>_<side>.npz`
IKC_SFX = ".npz"

# ── Name grammar ──────────────────────────────────────────────────────────────
# clip_id := task_id "__" take_id,  each of them field ("_" field)*,  field := [a-z0-9-]
# The separator hierarchy is the whole grammar: "-" inside a field / "_" between fields /
# "__" the task-take boundary.
FIELD_SEP = "_"
TAKE_SEP = "__"
_FIELD_RE = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
CLIP_ID_RE = re.compile(
    rf"^{_FIELD_RE}(?:{FIELD_SEP}{_FIELD_RE})*"
    rf"{TAKE_SEP}{_FIELD_RE}(?:{FIELD_SEP}{_FIELD_RE})*$"
)
# Every suffix that can follow a name — longest first. Taking a name apart needs only this list.
# ── Derived-clip suffixes ──────────────────────────────────────
# One take can be split **twice**. The producers of the two suffixes live in different files, and
# the reverse direction (finding the parent) needs one owner — so it goes here.
#   `_segNN`  a stretch cut out of a long take by time during base intake (base_adapter).
#   `_tN`     a stretch cut at an **object-set transition** in the case timeline
#             (segment/split.py:125).
# Why it matters: the stage 1 (retargeting) output is per **take**, while everything from stage 2
# on is per **segment**. Put the two side by side as counts in one table and you get a false
# conclusion — e.g. arctic: retarget 301 vs base 800 with an **intersection of 0** (all of them
# _segNN).
DERIVED_SFX_RE = re.compile(r"(_seg\d+|_t\d+)+$")


def parent_clip(name: str) -> str:
    """Derived clip → the **original take** that produced it. Not derived: returned unchanged.

    `X_seg03_t1` → `X`. The suffixes can stack, so all of them come off at once.
    The input must be a clip_id (if a tree suffix is attached, run it through clip_name first)."""
    return DERIVED_SFX_RE.sub("", name)


#: Placement-variant suffix pattern — `<clip>_pv<NN>` (NN = the variant's index · two digits or more, k ≥ 100 = three).
#: Writer = variant_name (dexengine.record.augment.place_gen) · readers = variant_source · the shadow packer's path
#: regexes · the per-hand sidecar registry rule (dexengine.core.layout.sidecar_rel). The pattern lives in
#: dexengine.core.layout (PV_SFX_PAT · variant_source); this module re-exports it below, after the layout is loaded — no
#: second statement here.


def variant_name(clip: str, k: int) -> str:
    """Placement variant k of a clip (clip_id in, clip_id out — add the tree suffix with tree_name)."""
    return f"{clip}_pv{k:02d}"


def variant_source(name: str) -> str:
    """Placement variant → the clip it was made from (`X_seg03_pv07` → `X_seg03`). Not a variant: unchanged.
    The input must be a clip_id (if a tree suffix is attached, run it through clip_name first). Tables keyed by the
    source clip (per-take frames · desk planes …) look a variant up through this — a variant has no row of its own."""
    return data_layout.variant_source(name)


def known_suffixes() -> tuple[str, ...]:
    """The suffix combinations that follow a clip_id. Keeps consumers from re-typing the list."""
    return tuple(
        sorted({TREE_SFX + PLACEMENT_SFX, TREE_F0_SFX, TREE_SFX, PL_SFX, PLACEMENT_SFX, ""}, key=len, reverse=True)
    )


COLLISION_MARK = "~"  # collision-resolution suffix `~2`. A character banned in fields, so
#   it can never clash with a normal name.

WT = REPO_ROOT  # repository root

# Identity inputs (hand, combo) have no silent default — there is no default-combo or default-hand constant.
# With a default combo, a caller that passes no combo would silently resolve another hand's layers.
# When a combo is needed but not given it is **derived from the hand** (combo_for); if that hand does not have
# exactly one registered combo, it raises instead of guessing.


def ik_suffix() -> str:
    """Arm IK cache directory suffix. This is a **live contract** — no literals.

    `_cont25` = the all-frames continuity guard 2.5° (maximum joint step per frame).
    Training, rendering and judging must see the **same suffix** to be the same task. A different
    suffix means a different experiment.
    """
    return os.environ.get("DEXENGINE_ARM_IK_DIR_SUFFIX", "_cont25")


# ── Name conversion (idempotent in both directions) ─────────────────────────
def clip_name(name: str) -> str:
    """Whatever comes in, out comes the **clip name** (no suffix) — the form the training set and
    the f0 shadow use."""
    return name[: -len(TREE_SFX)] if name.endswith(TREE_SFX) else name


def tree_name(name: str) -> str:
    """Whatever comes in, out comes the **tree name** (`_pl_rt`) — the form every combo output
    uses."""
    return name if name.endswith(TREE_SFX) else name + TREE_SFX


# ── Name contract: normalisation, derivation, collisions ──
def normalize_field(s: str) -> str:
    """One field into the contract's character set. The N1~N7 order is the contract — do not
    reorder it.

    N1 NFKC · N2 strip · N3 lowercase · N4 `[^a-z0-9]+`→`-` · N5 collapse repeated `-` ·
    N6 strip `-` at both ends · N7 reject an empty field (never dropped silently).

    Why the replacement character is `-` and not `_`: `_` is the **field separator**. Turning
    spaces into `_` makes `pour in some_bowl_bowl` → `pour_in_some_bowl_bowl`, and the name alone
    no longer tells you whether that is 3 fields or 5 fields. With `-`, `split("_")` is always
    exact.

    It is idempotent: normalize_field(normalize_field(s)) == normalize_field(s).
    """
    t = unicodedata.normalize("NFKC", s).strip().lower()  # N1·N2·N3
    t = re.sub(r"[^a-z0-9]+", "-", t)  # N4
    t = re.sub(r"-{2,}", "-", t).strip("-")  # N5·N6
    if not t:  # N7
        raise ValueError(f"normalisation gives an empty field: {s!r}")
    return t


def is_clip_id(name: str) -> bool:
    """Is this a clip_id that obeys the contract grammar (a suffix or tree name gives False)."""
    return bool(CLIP_ID_RE.fullmatch(name))


def make_clip_id(task_fields, take_fields) -> str:
    """Join normalised fields with the contract grammar. The only assembly point."""
    task = FIELD_SEP.join(normalize_field(f) for f in task_fields)
    take = FIELD_SEP.join(normalize_field(f) for f in take_fields)
    return task + TAKE_SEP + take


def clip_id_from_raw(raw_task: str, raw_take: str, dataset: str = "taco") -> str:
    """Build a clip_id **from MANO raw** — there is exactly one dataset adapter, and it is here.

    TACO: raw_task = the directory name under `Hand_Poses/`, `"(action, tool, target)"`,
          raw_take = the take directory name inside it, `"YYYYMMDD_NNN"`.
          The field order follows the dataset's own semantic order
          (action_tool_target__date_index).

    A new dataset adds **one branch here**. Do not create a new module (invariant d).
    """
    if dataset == "taco":
        t = raw_task.strip()
        if not (t.startswith("(") and t.endswith(")")):
            raise ValueError(f"not a TACO triplet: {raw_task!r}")
        fields = t[1:-1].split(", ")
        if len(fields) != 3:
            raise ValueError(f"a TACO triplet must have 3 fields: {raw_task!r} → {fields}")
    else:
        raise ValueError(
            f"no adapter for dataset: {dataset} (add a branch here; the naming rules are in docs/datasets.md)"
        )
    return make_clip_id(fields, raw_take.split(FIELD_SEP))


# ── Anchor suffix of the judging IK cache (`<clip>_a<anchor>_<side>.npz`) ───
def ikc_name(clip: str, anchor: str, side: str) -> str:
    """The IK cache filename judging leaves behind. Only here knows the anchor suffix."""
    return f"{clip_name(clip)}{IKC_ANCHOR_SEP}{anchor}{FIELD_SEP}{side}{IKC_SFX}"


def clip_from_ikc(filename: str) -> str:
    """Just the clip name out of `<clip>_a<anchor>_<side>.npz`. Do not `rsplit('_a')` by hand.

    Working from the end, `_<side>` and `_a<anchor>` are each stripped **exactly once**. Cutting
    from the front (`sed 's/_a.*//'`) cuts at an `_a` inside the clip name and silently shortens
    it.
    """
    stem = os.path.basename(filename)
    if stem.endswith(IKC_SFX):
        stem = stem[: -len(IKC_SFX)]
    head, sep, _side = stem.rpartition(FIELD_SEP)  # strip `_<side>`
    if not sep:
        return stem
    cut = head.rfind(IKC_ANCHOR_SEP)  # strip the last `_a<anchor>`
    return head[:cut] if cut > 0 else head
