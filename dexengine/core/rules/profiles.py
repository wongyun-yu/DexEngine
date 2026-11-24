"""Per-dataset pipeline shape (`DATASET_PROFILE`): embodiments, judge/bake scene, scene variant, source layer, USD
folder, and the scopes derived from it (f0 shadow axis, launch flags, campaign exclusions)."""

from __future__ import annotations

import pathlib
import re
import sys
from dataclasses import dataclass

from dexengine.core import paths as _paths_mod
from dexengine.core.rules.source import _paths

ALL = ("taco", "grab", "h2o", "hocap", "hot3d", "oakink2", "dexcanvas", "hrdexdb", "arctic")


def campaign_excluded() -> dict[str, str]:
    """Datasets outside the campaign → exclusion reason. The only statement is `Profile.campaign_exclude`; the
    re-judge chain, dashboards, hand pages and the manifest all read it through this function (no copies)."""
    return {d: DATASET_PROFILE[d].campaign_exclude for d in ALL if DATASET_PROFILE[d].campaign_exclude}


_SCOPE_SAID: set = set()


def datasets_scope(spec: str) -> tuple[str, ...]:
    """Comma list → dataset tuple (in the given order). Empty = the campaign scope (ALL − campaign_excluded). An
    unknown name raises KeyError (never filtered silently). An excluded dataset named explicitly is included (an
    explicit diagnostic choice). Both cases print the `[SCOPE]` marker to stderr (the shell reads stdout as the
    scope). The campaign entry and the stage-1 axis (`--datasets`) both set the scope through **this one function**
    (`--datasets-scope`)."""
    want = [d.strip() for d in (spec or "").split(",") if d.strip()]
    bad = [d for d in want if d not in ALL]
    if bad:
        raise KeyError(f"unknown datasets {bad} (known: {list(ALL)})")
    out = campaign_excluded()
    said = (
        (f"explicitly included {d} (campaign exclusion reason: {out[d]})" for d in want if d in out)
        if want
        else (f"excluded {d} ({w})" for d, w in out.items())
    )
    for m in said:
        if m not in _SCOPE_SAID:
            _SCOPE_SAID.add(m)
            print(f"[SCOPE] {m}", file=sys.stderr, flush=True)
    return tuple(dict.fromkeys(want)) if want else tuple(d for d in ALL if d not in out)


# ── 1) Per-dataset pipeline shape — this file is the only owner ────────────────────────────────
#: Scene vocabulary (the judge/bake scene, and the scene first-pass rules apply to). table = canonical desk,
#: aerial = deskless D20 disc. A clip's scene comes from the dataset profile (scene) and the path rule (D20 — a deskless
#: tree of a dataset where no desk is normal → aerial).
SCENES = ("table", "aerial")


@dataclass(frozen=True)
class Profile:
    embs: tuple  # embodiments that exist
    scene: str  # judge·bake scene: table (canonical desk) | aerial (D20 disc)
    variant: str = ""  # combo variant suffix (_artic/_aerial) — absent = canonical rby1_wuji
    src_layer: str = "base"  # stage-2 source layer — owner dexengine.core.paths.default_src_layer (all base)
    usd_dir: str = ""  # when it is not the default data/<ds>/usd
    obj_key: str = "base"  # **object-key source** for grouping training sets (object_shape.target_object):
    #   "base" = main object of the base-layer task_info (role main → mesh_dir basename) · "clip" = last task field
    #   of the clip id. Base because roles (D23) are declared in the base layer; a tree's roles are a bake-time copy
    #   that is not refreshed until re-judged. Only TACO (V1 layout) uses "clip" — there the last field is the real
    #   object; in the other datasets it is a subject, session, verb or hash. Declared per dataset, not a fallback.
    #: Manipulation space (data property) — tabletop = captured at a table (hands work on it) · room = room-scale
    #: capture (a standing person also works off the table; desk surface = the lowest surface holding an interaction
    #: object, higher ones are pedestals). Read by the rule scope `profile:workspace=room`. **The value lives in
    #: `dexengine.core.layout.DatasetSpec.workspace`** (the scene desk top reads it too) — `_with_workspace` fills it.
    workspace: str = ""
    #: Captures where a resting surface can be above the desk surface — objects rest on a stand, shelf or untracked
    #: item on the desk. Read by the pedestal-rule (D45) scope `profile:raised_rest=True`. Independent of how the desk
    #: surface is chosen (workspace): raised rests exist even when the desk is the capture's table.
    raised_rest: bool = False
    #: The source has a real table shape (mesh) — scope of wrist-inside-the-table-outline (D14′). The fallback for
    #: trees without an outline (D14-fb) has a decided scope no single property yields — see D14_FB_SCOPE.
    desk_outline: bool = False
    #: Campaign exclusion reason (empty = in the campaign). The only per-dataset exclusion statement — read it through
    #: `campaign_excluded()`.
    campaign_exclude: str = ""
    note: str = ""


#: The tree suffix (`_pl_rt`) is owned by dexengine.core.paths — never written out here.
#: It is read from the paths module directly.
def _tree_sfx() -> str:
    return _paths_mod.TREE_SFX


_TREE_SFX = _tree_sfx()
#: Owner file of the scene desk top (centre x, size) — the D22 card thresholds point at it.
_DESK_OWNER = "dexengine/core/desk.py"


def _with_workspace(table: dict) -> dict:
    """Profile.workspace ← `dexengine.core.layout.DatasetSpec.workspace` (single owner). Fails at import when the
    table writes that field itself (a second statement) or a dataset is missing from the layout."""
    import dataclasses as _dc

    specs = _paths_mod.data_layout.DATASETS
    if wrote := sorted(d for d, p in table.items() if p.workspace):
        raise ValueError(
            f"DATASET_PROFILE {wrote}: workspace is written here — its home is "
            "dexengine.core.layout.DatasetSpec.workspace"
        )
    if miss := sorted(set(table) - set(specs)):
        raise ValueError(f"DATASET_PROFILE {miss}: not in dexengine.core.layout.DATASETS — no workspace to read")
    return {d: _dc.replace(p, workspace=specs[d].workspace) for d, p in table.items()}


DATASET_PROFILE = _with_workspace(
    {
        "taco": Profile(
            ("bimanual",),
            "table",
            usd_dir="data/usd_cache/objects_taco",
            obj_key="clip",
            note="TACO layout: the tree place is owned by the layout (trees/<combo>/<emb>; V1 = the "
            "preserved root) · identity = objects[] (same rule as other datasets — the flat "
            "<side>_object_mesh_dir form exists only in the V1 preserved root). Source layer is "
            "base",
        ),
        "grab": Profile(("bimanual",), "table", desk_outline=True),
        "h2o": Profile(("bimanual",), "table"),
        "hocap": Profile(("bimanual", "right", "left"), "table"),
        "hot3d": Profile(
            ("bimanual", "right"),
            "table",
            raised_rest=True,
            note="one desk scene (independent of hand height) — desk surface = the lowest surface "
            "holding a handled object (D10-h desk recovery) · objects resting higher are pedestals "
            "(D45). No aerial variant axis (the D20 aerial scene is only for desk_mode none)",
        ),
        "oakink2": Profile(
            ("bimanual",),
            "table",
            raised_rest=True,
            note="desk surface = the session table (session_w) as is — raised rests on it (stands, "
            "cutting boards, lamps) are pedestals (D45)",
        ),
        "dexcanvas": Profile(("right",), "table"),
        "hrdexdb": Profile(("right",), "table"),
        "arctic": Profile(
            ("bimanual",),
            "aerial",
            variant="_artic",
            note="articulated (1 hinge) + no desk. On the rigid path (_aerial) the bake fails with OD-6",
        ),
    }
)

#: Scope of D14-fb (wrist below the desk in a tree without a table outline → first-pass exclusion), as decided:
#: clips in h2o and grab that clearly go below the desk are excluded, other datasets are only counted. No single
#: property yields this set, so each dataset carries a reason (the form scope_of accepts; a bare name tuple raises).
D14_FB_SCOPE = {
    "grab": "the source has a table mesh — fallback for trees without a table outline (oracle footprint)",
    "h2o": "no table shape in the source, so there is never an outline (D14′) — drop wrists that go below the desk by "
    "more than the static desk estimate's error (2–4 cm)",
}


#: **Superseded rules** are **not written** in this file:
#: a rule that is replaced is removed from it.


#: **Axes that are not canonical** but remain in the final ledger — the code holds the reason so the
#: table is not misread.
DEAD_AXES = {
    "arctic-aerial": "dropped: arctic on the rigid path (_aerial) — the bake fails with OD-6 ('qpos width is not a "
    "multiple of 7'). The canonical axis is arctic-artic",
    "hot3d-aerial": "folded (D45): hot3d has a single desk scene. The axis's clips live in the hot3d axis "
    "(the RL queue closes its slots — rl_queue_label.gone_axes)",
    "hot3d-right-aerial": "folded (D45): same as hot3d-aerial — the one-hand axis is hot3d-right",
}


#: T3 — the single declaration of **whether an articulated object can go into training**.
#: True only when both conditions hold:
#:   ① asset: `data/<ds>/usd/<asset id>/object_artic.usd` (base + moving part, 2 links + 1 joint; asset id =
#:      `dexengine.core.artifact_paths.object_asset_id`) — `dexengine/robots/objects/usd_bake.py --articulated`
#:   ② env: `ArticulationCfg` spawn + joint-value restore on reset + observation/reward (D1-c·D1-i)
#: Do not flip this value to True when only one of the two exists. The flipping condition is written here
#: to stop the "trainable" count from quietly growing once only the assets have been baked.
#: Consumer: the dashboard numbers.
#: Both conditions hold, so it is True. ② is wired in dexengine/rl/env_setup/build.py (build_object_entity_cfg,
#: add_obj_artic_terms), events/reset.py (joint restore), observations/v2.py (ObjectStateV2 87→89), rewards/task.py
#: (RewardObjArtic), agents/models.py (_object_extra_dims) and trajectory/manager.py (object_pose_window); the run logs
#: show the `[ARTIC][spawn]`, `[ARTIC][reward]`, `[ARTIC][surface]` and `[ARTIC][theta-judge]` markers.
#: Open before an articulated campaign: the per-link MassPropertiesCfg(mass=0.2) in apply_object_mass doubles the
#: articulated mass (0.398 kg total), and the eval-path articulated spawn has not been boot-tested.
ARTIC_ENV_WIRED = True

#: E-1 — the single declaration of **whether a D20 aerial (deskless) tree can go into training**.
#: True only when the env does all three, proven by a boot smoke whose log shows the `[AERIAL]` markers:
#:   ① desk prims sunk (no desk collision) ② per-clip support-disc slots spawned from `task_info.aerial_support.discs`
#:   ③ per-clip z offset = `aerial_support.z_off` applied by the trajectory loader (`[AERIAL] D20 … dz` line).
#: Owner of the wiring: dexengine/rl/env_setup/build.py build_aerial_scene and the trajectory loader.
#: Consumer: the dashboard numbers.
#: All three are wired, so it is True — the boot log shows the `[AERIAL]` and `[AERIAL][discs]` markers.
AERIAL_ENV_WIRED = True


#: Combo variant suffixes (canonical desk "", aerial scene, articulated) — the **single list** read when enumerating
#: tree directories `data/<ds>/trees/<combo(hand, variant)>` and ledger axes (`ledger_key(ds, emb, variant)`).
#: Trees are never found with an `rby1_<hand>*` glob — it would also catch experimental variants and hard-code hands.
COMBO_VARIANTS = ("", "_aerial", "_artic")


#: **Grammar** of a variant suffix (a rule, not a table) — stage2 also accepts validation variants (`_val50`, …)
#: outside the canonical three (COMBO_VARIANTS).
_VARIANT_RE = re.compile(r"(_[a-z0-9]+)?")


def f0_axis(ds: str, variant: str, hand: str = "") -> dict:
    """**Single owner** of the f0 shadow axis: for a (dataset, variant) axis, **which scene f0 is solved in and where
    it is stored**.

    Returns (repository-relative paths):
      scene   the scene whose desk plane f0 reads — `_aerial` means the D20 aerial axis by definition, otherwise the
              profile scene (arctic `_artic` = profile aerial). The f0 repair's desk plane depends on the scene
              (`coords.scene_z_offset`: table −0.13 m, aerial 0.00 m on arctic), so a different scene is a
              different f0.
      node    shadow root node = the `f0_stage` node of the f0 layer template (layout "f0_root") + the variant
              (isolated per variant)
      tail    tail below the node (set by the layer template — never written out here)
      shadow  node/tail = the value f0 prejudge takes as `--f0-root` = the place the stage-1 completion check counts
      hand    (only when `hand` = a hand spec directory is given) that hand's shadow place — written by f0 prejudge
              and read by stage2 (ledger LEDGER.tsv, shadow). Where the hand goes is set by where the per-hand layer
              template (layout "f0_stage") sits under "f0_root". The dataset-wide root (shadow) is the **union of
              every hand**, so it cannot tell whether this hand's f0 exists.

    Callers (none restates the value): stage-1 step 7 (dexengine/data/stages.py emit_run), the stage-1 completion
    check, the judge-prep defaults (stage2), the prep scene check and the tree checker's f0 layer (`--f0-hand-dir`).
    An unknown dataset or a variant outside the grammar raises. A per-hand template not under the f0 root raises
    instead of building a hand place (the two templates have diverged).
    """
    p = DATASET_PROFILE[ds]  # unknown dataset → KeyError
    if not _VARIANT_RE.fullmatch(variant or ""):
        raise ValueError(f"variant {variant!r} must be '' or '_<lowercase letters/digits>'")
    parts = pathlib.PurePosixPath(_paths().data_layout.render("f0_root", ds)).parts
    if "f0_stage" not in parts:
        raise ValueError(f"the f0 layer template has no f0_stage node: {'/'.join(parts)}")
    i = max(k for k, x in enumerate(parts) if x == "f0_stage")
    node = str(pathlib.PurePosixPath(*parts[: i + 1])) + (variant or "")
    tail = str(pathlib.PurePosixPath(*parts[i + 1 :])) if i + 1 < len(parts) else "."
    out = {
        "scene": "aerial" if variant == "_aerial" else p.scene,
        "node": node,
        "tail": tail,
        "shadow": node if tail == "." else f"{node}/{tail}",
        "variant": variant or "",
    }
    if hand:
        dl = _paths().data_layout
        try:  # hand tail = per-hand template relative to the f0 root — outside the root raises ValueError
            sub = pathlib.PurePosixPath(dl.render("f0_stage", ds, hand=hand)).relative_to(dl.render("f0_root", ds))
        except ValueError as e:
            raise ValueError(f"the per-hand f0 template (f0_stage) is not under the f0 root (f0_root): {e}") from None
        out["hand"] = str(pathlib.PurePosixPath(out["shadow"]) / sub)
    return out


def launch_flags(ds: str) -> list:
    """Judge·bake launch flags — so shell scripts never write dataset names out again (side branch removed)."""
    p = DATASET_PROFILE[ds]
    out = []
    if p.variant:
        out += ["--variant", p.variant]
    if p.scene != "table":
        out += ["--scene", p.scene]
    if p.src_layer != "base":
        out += ["--src-layer", p.src_layer]
    return out
