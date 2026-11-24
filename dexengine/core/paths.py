#!/usr/bin/env python3
"""Canonical path resolver — one clip name produces every kind of output path.

Why this file exists
--------------------
The same clip is called by a **different name** in different places. Some places carry the suffix
and some do not, and a consumer that concatenates the strings by hand sooner or later reads "the
file is missing" as "the clip failed". The places that know about the suffix are gathered **here,
into one**.

  training-set list  brush_brush_bowl__20230919_036            ← **no** suffix
  tree / outputs     brush_brush_bowl__20230919_036_pl_rt      ← **with** suffix
  f0 shadow          brush_brush_bowl__20230919_036            ← **no** suffix

`_pl_rt` is the tree name meaning "placement applied + retime finished". The clip name (the raw
TACO sequence) and the tree name (that clip's processed output) are **different namespaces**: the
training set and the f0 shadow live on the clip-name side, combo outputs on the tree-name side.

Usage (Python)
--------------
    from dexengine.core.paths import ClipPaths

    # identity inputs (combo · hand · dataset · emb) have no default — the caller names all four
    P = ClipPaths("brush_brush_bowl__20230919_036", COMBO, HAND, DATASET, EMB)   # name with or without the suffix
    P.placement          # …/robots/<combo>/placement/<tree>.adopted.json
    P.ik("right")        # …/robots/<combo>/arm_ik_poses_right_cont25/<tree>.npz
    P.wrist("left")      # …/robots/<combo>/wrist_poses_left/<tree>.npz
    P.f0_wrist("right")  # …/f0_stage/…/<hand>/wrist_poses_right/<clip>.npz  (no suffix)
    P.tree_qk            # …/trees/<combo>/<emb>/<tree>/0/trajectory_kinematic.npz
    P.contact("left")    # …/<tree>/0/contact_mano_left.npz

Usage (shell — so the suffix is never reassembled in the shell)
---------------------------------------------------------------
    python -m dexengine.core.paths --clip "$C" --what ik_right --combo "$COMBO" --hand "$HAND" \
        --dataset "$DS" --emb "$EMB"
    python -m dexengine.core.paths --what dirs --combo "$COMBO"                  # the directories to sync
    python -m dexengine.core.paths --what counts --combo "$COMBO" --dataset "$DS"  # canonical files only

Why `--what counts` matters: in these directories the sidecars (backups) live next to the
canonical files. `ls | wc -l` counts both together, so a count check spins uselessly (backups such
as `.pre_rt.npz` can outnumber the canonical files).

The only dependency is the standard library — the shell calls this once per clip, so a numpy
import is avoided.

The code lives in `dexengine.core.locations`; this module re-exports the public names and is the command line.

Run by file path: shell callers run `python3 dexengine/core/paths.py --what …` without an installed package, so the
three lines below put this checkout's root on `sys.path` when the module has no package (the one sys.path edit here).
"""

from __future__ import annotations

import sys
from pathlib import Path

if not __package__:  # run by file path (`python3 dexengine/core/<name>.py …`): import this checkout's package
    sys.path[0] = str(Path(__file__).resolve().parents[2])

from dexengine.core.locations.cli import HELP, build_parser, main  # noqa: F401
from dexengine.core.locations.hands import (  # noqa: F401
    ARM_IDS,
    HAND_IDS,
    HAND_SPEC_DIR,
    MODE_RE,
    RETIRED_HAND_IDS,
    RUNTIME_DIRS,
    clips_record,
    combo_dirname,
    combo_for,
    combo_id,
    combo_ids,
    combo_usd_relpath,
    hand_id,
    hand_rig_sha,
    hand_spec_dir,
    mode_record,
    split_combo,
)
from dexengine.core.locations.layers import (  # noqa: F401
    ENTRY_LAYER_ROOTS,
    SRC_LAYERS,
    bake_outputs,
    contact_derive_dir,
    contact_derive_ledger,
    default_src_layer,
    entry_layer_roots,
    gate_table,
    ik_dir,
    layer_path,
    layer_root_records,
    meta_dir,
    named_roster_of,
    placement_dir,
    ref_base_dir,
    robots_dir,
    roster_root,
    sidecar_file,
    taco_source_root,
    trees_dir,
    units_registry,
    wrist_base_dir,
    wrist_dir,
)
from dexengine.core.locations.modules import (  # noqa: F401
    PV_SFX_PAT,
    data_layout,
    hand_rig_dir,
    load_registry,
    qpos_layout,
)
from dexengine.core.locations.names import (  # noqa: F401
    FIELD_SEP,
    IK_SIDECAR_SFX,
    PLACEMENT_SFX,
    TAKE_SEP,
    TREE_SFX,
    WRIST_SIDECAR_SFX,
    WT,
    clip_from_ikc,
    clip_id_from_raw,
    clip_name,
    ik_suffix,
    ikc_name,
    is_clip_id,
    known_suffixes,
    make_clip_id,
    normalize_field,
    parent_clip,
    tree_name,
    variant_name,
    variant_source,
)
from dexengine.core.locations.roots import (  # noqa: F401
    cluster_tree,
    data_rel,
    data_root_of,
    f0_root,
    main_root,
    mano_model_dir,
    output_root,
    owner_root,
    pool_root,
    repo_rel_path,
    require_retarget_py,
    reread_stage_root,
    retarget_py,
    retarget_py_candidates,
    retarget_stage_root,
    run_out,
    run_rel,
    run_root,
    run_root_of,
    urdf_mesh,
    work_area,
)
from dexengine.core.locations.runs import (  # noqa: F401
    RL_CAMPAIGN_SEP,
    RL_ID_RE,
    canonical_clip,
    check_rl_campaign,
    hand_of_run_path,
    raw_from_clip_id,
    read_clip_list,
    rl_bundle_dir,
    rl_cell_of,
    run_glob,
    run_path,
    scope_takes,
    stage_banner,
)
from dexengine.core.locations.trees import (  # noqa: F401
    ClipPaths,
    canon_count,
    combo_from_seq_dir,
    seq_dir_identity,
    sync_dirs,
)
from dexengine.core.repo import CONFIGS_DIR, FULL_ROBOT_URDF_DIR, HAND_ASSETS_DIR, REPO_ROOT  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
