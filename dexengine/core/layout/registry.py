"""Dataset and hand-asset registries of the data layout: datasets, embodiments, retargeting hand assets."""

from __future__ import annotations

import os
from dataclasses import dataclass

EMBODIMENTS = ("right", "left", "bimanual")

# ── Data root — **which checkout** the (repo-relative) templates below are joined to ───────────────────────
# Data (the untracked data/·precompute/·logs/·archive/) lives in one checkout only. A code-only worktree points at
# that checkout with this env and creates **no symlinks at all**.
# This constant alone owns the name — the script entry `dexengine.core.paths.main_root()` (default = that code
# tree) and the package entry `data_abspath()` (default = the working directory, same as os.path.abspath) read it.
DATA_ROOT_ENV = "DEXENGINE_MAIN_ROOT"


# ── Dataset registry ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetSpec:
    """The **input properties** of one dataset. No paths here — `render()` builds the paths."""

    name: str
    source_fps: float  # source capture rate (inverse of the delivered task_info.source_ref_dt)
    object_id_kind: str  # "numeric" (TACO 162 mesh ids) | "name" (delivered datasets: cansoup·apple …)
    embodiments: tuple[str, ...]  # embodiments this dataset actually holds
    status: str  # "canon" | "intake" (intake/validation stage, not registered for training)
    notes: str = ""
    #: Manipulation space (a data property) — values in WORKSPACES. tabletop = captured at a table (hands work on the
    #: table) · room = room-scale capture (a standing person also works off the table). Read by rule scopes
    #: (`dexengine.core.datasets.Profile`) and the scene desk size (judge and env read the same function).
    workspace: str = "tabletop"


#: Every value `DatasetSpec.workspace` may take. Any other value dies at import (so a typo cannot mean something
#: else).
WORKSPACES = ("tabletop", "room")


DATASETS: dict[str, DatasetSpec] = {
    "taco": DatasetSpec(
        "taco",
        30.0,
        "numeric",
        ("bimanual",),
        "canon",
        notes="canonical campaign dataset (2,318 sequences · 162 meshes). Only the hand-independent "
        "input layers redefine STORAGE.",
    ),
    # The datasets below come from external retargeting deliveries (rby1+wuji target). Training entry points accept a
    # dataset only once its status is "canon" (intake/validation: dexengine/data/intake.py).
    "hrdexdb": DatasetSpec(
        "hrdexdb",
        30.0,
        "name",
        ("right",),
        "canon",
        notes="T2 promoted: 429 adopted clips (single hand, right), object assets 429/429 "
        "complete (measured by train_ready).",
    ),
    # Training entry points come in through --sequence_dir (tree view) — base is for traj_root.
    "hocap": DatasetSpec(
        "hocap",
        30.0,
        "name",
        ("right", "left", "bimanual"),
        "canon",
        notes="several objects per clip (task_info.extra_objects). canon: base·meshes·usd·trees "
        "(rby1_wuji_h91 79) checked.",
    ),
    "dexcanvas": DatasetSpec(
        "dexcanvas",
        30.0,
        "name",
        ("right",),
        "canon",
        notes="T2 promoted: 115 adopted clips (single hand, right), object assets 115/115 "
        "complete (measured by train_ready).",
    ),
    # The four below are one delivery (single tar.zst, zstd checksum verified, hand = wuji).
    # All bimanual.
    "grab": DatasetSpec(
        "grab",
        120.0,
        "name",
        ("bimanual",),
        "canon",
        notes="1,335 clips. In clips where both hands handle one object, the right/left object "
        "fields point to the same mesh (shared object). T2 promoted: object USDs of the "
        "298 adopted clips 298/298 complete (the 12 missing were baked by the "
        "object USD bake).",
    ),
    "h2o": DatasetSpec(
        "h2o",
        30.0,
        "name",
        ("bimanual",),
        "canon",
        notes="184 clips, egocentric bimanual. 56 clips have a shared object (right/left fields "
        "name the same mesh). T2 promoted: 32 adopted clips, object assets 32/32 "
        "complete (measured by train_ready).",
    ),
    "oakink2": DatasetSpec(
        "oakink2",
        120.0,
        "name",
        ("bimanual",),
        "canon",
        notes="2,828 clips (of the 2,840 source stages, 12 with missing objects excluded). Task "
        "names carry a stage suffix (_s000). T2 promoted: 1,982 adopted clips, object "
        "assets 1,982/1,982 complete (measured by train_ready).",
    ),
    # hot3d (Meta, 2024): taken in directly. Pose-only download (Aria 198 · Quest 226) → dexengine/data/readers/hot3d.py
    # cuts clips by contact segmentation (<seq>__s<a>_e<b>), 30 fps → 50 Hz. World z-up (headset start origin), desk =
    # static estimate. Hand annotation licence CC BY-NC-SA.
    "hot3d": DatasetSpec(
        "hot3d",
        30.0,
        "name",
        ("bimanual", "right", "left"),
        "canon",
        workspace="room",
        notes="T2 promoted: 402 adopted clips, object assets 402/402 complete (measured by "
        "train_ready). 424 recordings (Aria 198 + Quest 226) → contact-segmented "
        "clips. Several objects per recording (extra_objects). MANO PCA15 → smplx "
        "convention (additive shift).",
    ),
    # arctic: the articulated-object (T3) variant axis. Whether it can train is owned by
    #   dexengine.core.datasets.ARTIC_ENV_WIRED (T3) alone — this status only says T2 (intake complete).
    "arctic": DatasetSpec(
        "arctic",
        30.0,
        "name",
        ("bimanual",),
        "canon",
        notes="301 clips (722 adopted after cutting). **Articulated objects** "
        "(task_info.articulated_object, one hinge) — qpos width 60 = 59 + hinge. "
        "The loader reads the joint column as get_frame's 8th value and the env "
        "spawns it with ArticulationCfg.",
    ),
}

if _bad := {k: v.workspace for k, v in DATASETS.items() if v.workspace not in WORKSPACES}:
    raise ValueError(f"dexengine.core.layout.DATASETS: unknown workspace {_bad} — must be one of {WORKSPACES}")


def dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(
            f"unregistered dataset {name!r} (registered: {sorted(DATASETS)}). "
            "Add a DatasetSpec to dexengine.core.layout.DATASETS — do not make up paths."
        )
    return DATASETS[name]


# ── Retargeting hand asset registry — identity is fixed by a **hash**, not by a name ─────────────
# One hand name can cover **different URDFs** with equal joint names but different limits (effort, velocity), so
# the files alone cannot tell them apart. The identity is therefore the
# sha256 of the asset directory's `right.xml+left.xml+bimanual.xml` concatenated (`asset_hash()`). The intake tool
# **refuses** any hash not in this table.
@dataclass(frozen=True)
class RetargetHandAsset:
    spec_dir: str  # assets/robots/<spec_dir>
    sha256_xml3: str  # sha256(right.xml ‖ left.xml ‖ bimanual.xml)
    joints_right_xml: int  # number of <joint in right.xml (incl. 6 wrist dof)
    status: str  # "canon" | "legacy" | "intake-only" | "rejected"
    isaac_hand_id: str | None  # dexengine.core.hands.HAND_REGISTRY key (None before registration)
    provenance: str


RETARGET_HAND_ASSETS: dict[str, RetargetHandAsset] = {
    "sharpa_wave": RetargetHandAsset(
        "sharpa_wave",
        "32cdcc31d86437cac6b31c6c882bb76a24a2b53669112400d8abf8c0718f66a4",
        29,
        "canon",
        "sharpa_wave",
        "rig assets/robots/hands/sharpa_wave/mjcf (canonical campaign hand)",
    ),
    # A hand must be in this table as well as in HAND_REGISTRY·MOUNT_REGISTRY·HAND_SPEC_DIR, or intake rejects every
    # dataset ("unregistered hand asset"). The hash is measured — the staged asset and the repo copy
    # (assets/robots/hands/dg5f_s/mjcf) have byte-identical asset_hash.
    # Hash = rig identity (incl. the palm site registration and fingertip pad offsets). When the rig changes, all
    # stage-1 outputs of the old hash (retargeting … trees) must be regenerated; the training entry compares the tree's
    # hand_asset_sha256_xml3 with this value and dies on a mix (train_ready_util.tree_rig_check).
    "dg5f_s": RetargetHandAsset(
        "dg5f_s",
        "ea6063a9b9633931d46224fefbf3c42dbb814afad4b7fb41723bbb1f5d5c46c8",
        26,
        "canon",
        "dg5f_s",
        "Tesollo DG-5F-S 20-DoF. Repo copy assets/robots/hands/dg5f_s/mjcf/. "
        "Joints 26 = free 6 + fingers 20. Vendor source tesollodelto/dg5f_s_ros2 + hardware manual Ver 2.0.1.",
    ),
    # Canonical rig = the palm site registration (HandSpec.palm_site_offset) applied — same hash rule as dg5f_s above.
    "xhand_pro": RetargetHandAsset(
        "xhand_pro",
        "b85025c6d68c7e83019e3dcc7f44053cfcc9a4407579b05a110b1163bae2a34c",
        27,
        "canon",
        "xhand_pro",
        "ROBOTERA XHand Pro 21-DoF. "
        "Repo copy assets/robots/hands/xhand_pro/mjcf/. Joints 27 = free 6 + fingers 21 (incl. ltspalm). "
        "Vendor source ROBOTERA XHAND1 Pro URDF V1.0 (2026-05-13) + brochure XHAND_1_pro-EN2.",
    ),
    # Wuji Hand 2 beta2 — a different hand from the first-generation wuji (joint names · geometry · tactile
    # pads). Same hash rule as above.
    "wuji2": RetargetHandAsset(
        "wuji2",
        "4dca842a784b52e214a36db6df12c35169972aafe0e249bb0775753be40c7eda",
        27,
        "canon",
        "wuji2",
        "Wuji Hand 2 beta2 20-DoF (the real-robot hand). Repo copy assets/robots/hands/wuji2/mjcf/ "
        "(rig builder, hand wuji2 · build source data/_stage/wuji2/wujihand2_beta2_{side}.urdf = the hand URDF as "
        "mounted on RB-Y1, assets/robots/vendor/wuji2/). "
        "`<joint` 27 = default 1 + free 6 + fingers 20 (same count as the first-generation wuji). "
        "Palm site pose = the MANO wrist frame of the hardware setup (HandSpec.palm_site_mano_frame).",
    ),
}


#: ── Hand of the shared axis — **this constant alone is the owner** ────────────────────────────────────
#: Meaning: **which hand** owns the one set of shared outputs without a hand suffix (dashboard numbers · audit JSON ·
#: a dataset's default training path). The running hand (DEXENGINE_HAND) is still an explicit input, never derived
#: from this. **`RETARGET_HAND_ASSETS[…].status == "canon"` does not answer this** — it means "the asset (rig xml3
#: hash) is officially registered", and several hands are canon; picking "the one canon hand" breaks as hands are
#: added. Why it lives here: the datasets package and the rules modules **both** read it but cannot import each
#: other, so it lives in this dependency-free module instead of a copy.
#: The value is a **hand id**. The spec directory name is not written here — `dexengine.core.paths.hand_spec_dir()`
#: resolves it.
AXIS_HAND = "xhand_pro"


def axis_spec_dir() -> str:
    """Retargeting asset name of the axis hand (= the hand axis of the base layer) — the one RETARGET_HAND_ASSETS
    entry whose isaac_hand_id is AXIS_HAND and that is canon. Anything but exactly one raises (never picked by
    count)."""
    c = [k for k, a in RETARGET_HAND_ASSETS.items() if a.isaac_hand_id == AXIS_HAND and a.status == "canon"]
    if len(c) != 1:
        raise KeyError(
            f"axis hand {AXIS_HAND!r} does not have exactly one canon retargeting asset: {c} — see RETARGET_HAND_ASSETS"
        )
    return c[0]


def asset_hash(asset_dir: str) -> str:
    """Identity hash of an asset directory — sha256 of right.xml ‖ left.xml ‖ bimanual.xml. Any missing file is an
    error."""
    import hashlib

    h = hashlib.sha256()
    for name in ("right.xml", "left.xml", "bimanual.xml"):
        p = os.path.join(asset_dir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"cannot measure the hand asset identity — {p} is missing")
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def identify_asset(asset_dir: str) -> RetargetHandAsset:
    """Directory → registered asset. A hash not in the table raises KeyError (with how to register). Names are
    ignored."""
    digest = asset_hash(asset_dir)
    for a in RETARGET_HAND_ASSETS.values():
        if a.sha256_xml3 == digest:
            return a
    raise KeyError(
        f"unregistered hand asset (xml3 sha256 {digest[:16]}…) at {asset_dir} — "
        "intake only after registering it with provenance and status in "
        "dexengine.core.layout.RETARGET_HAND_ASSETS."
    )


# ── clip_id derivation for external datasets ───────────────────────────────────────────────────────
# TACO derives it from the raw directory name (dexengine.core.paths.clip_id_from_raw). For delivered datasets we
# have no raw data and the delivered `task` string is the only name. The rule applies the grammar of
# canonical_clip **forward**:
#   1) Split task/take at the first `__`. Without `__`, take is `data_id` (the retarget layer's 0/ directory).
#   2) Split each side into fields at `_` and normalize_field them (NFKC · lowercase · disallowed chars → `-`).
#      Extra `__` inside the take (s02__cube1__09__001) count as field boundaries → s02__cube1_09_001.
#   3) The original string is kept as `source_task` in task_info — the table (INTAKE ledger) does the reverse.
def clip_id_from_task(task: str, data_id: int = 0, normalize_field=None) -> str:
    if normalize_field is None:  # receives the canonical function (no duplicate table)
        raise ValueError("pass normalize_field (dexengine.core.paths.normalize_field)")
    head, sep, tail = task.partition("__")
    if not sep:
        tail = str(data_id)
    task_fields = [f for f in head.split("_") if f != ""]
    take_fields = [f for part in tail.split("__") for f in part.split("_") if f != ""]
    if not task_fields or not take_fields:
        raise ValueError(f"no fields could be taken from the task string: {task!r}")
    return "_".join(normalize_field(f) for f in task_fields) + "__" + "_".join(normalize_field(f) for f in take_fields)
