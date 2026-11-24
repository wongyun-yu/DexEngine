"""Canonical data layout — one table owns the paths of dataset × hand × embodiment × combo × stage (layer).

The three axes (dataset, hand, embodiment) are **explicit** in every path — leave one out and names collide from the
second dataset on, and a single-hand clip's qpos width (33) differs from bimanual (59), so a loader would silently go
wrong. Deliverables follow the retarget layout
`processed/<dataset>/<robot>/<embodiment>/<task>/<data_id>/` (`dexengine/retarget/mjutil.py processed_data_dir`).

Rules
-----
* Path strings are assembled **only here**. `dexengine/core/paths.py` (shell and preprocessing entry),
  `dexengine/core/datasets.py` and `dexengine/robots/selection.py combo_data_paths` (env entry) **read** this table.
  Another file must not join `data/<ds>/…` by hand.
* Only TACO's hand-independent input layers live in a separate physical place (`STORAGE`, one spot). Hand-axis outputs
  (base, trees) use the canonical templates for every hand; the V1 frozen generation (SharpA × TACO) lives in the
  **preserved root** of its named roster (`roster_root`) — the layer table does not know it.
* This module uses the standard library only (no Isaac, no numpy): paths.py, called per clip from the shell, imports it.

Axes
----
  ds     dataset id (lower case, registered in `DATASETS`)            taco · hrdexdb · hocap · dexcanvas · grab
  hand   retargeting hand **spec directory** (= assets/robots/hands/<hand>)    sharpa_wave · wuji · dg5f_s
         (may differ from the Isaac hand id. The asset identity is fixed by the
         `RETARGET_HAND_ASSETS` hashes)
  emb    embodiment (retarget term)                                  right · left · bimanual
  combo  `<arm>_<hand_id><variant>` (built by paths.combo_id)        rby1_sharpa_wave · rby1_sharpa_wave_hx2
  clip   clip_id (canonical_clip grammar) / tree = clip + `_pl_rt`

Stages (layers) — each derives from the ones above
-------------------------------------------------
  raw             copy of the original dataset (annotations, meshes)    input, immutable (the basis for processing that
                                                                        needs the original, e.g. desk-height recovery)
  mano            human hand keypoints (MANO)                           input, immutable
  retarget        hand kinematic retargeting output (IK)                input, immutable (scene.xml is the canonical
                                                                        qpos layout)
  assets_objects  object meshes (visual.obj + convex/)                  input, immutable
  assets_robot    hand MJCF assets (right/left/bimanual.xml + meshes)   input, immutable, identity fixed by hash
  visualization   renders (mp4)                                         input, reference
  intake          intake ledger (INTAKE/<handoff_id>.json)              written by us
  base            our first cleanup (session z repair, object
                  assignment, task_info rewrite)                        our output
  meshes          derived object meshes (surface_points, keypoints)     our output
  usd             baked object USD + cooked-hull dump (cooked_hull_file) our output
  trees           placed + retimed combo trees (IK grounding output)    our output
  views           training views (per-set symlinks)                     our output (TACO: lad/)
  precompute      per-combo caches (arm IK, wrist, placement judge,
                  ledgers)                                              our output
  f0_root/f0_stage f0 shadow staging root / per hand (placement-free)   our output
  contact_base    base contact sidecar (combo-free)                     our output
  contact_all     base contact sidecar — **all objects** (before role
                  assignment)                                           our output
  ikpool          IK pool                                               our output (combo-free, depends on the arm
                                                                        generation)
"""

from __future__ import annotations

from dexengine.core.layout.collect import (  # noqa: F401
    COLLECT_SESSIONS_PER_GPU,
    COLLECT_STAGE1_FILES,
    collect_final,
    collect_job_archive,
    collect_job_archive_root,
    collect_job_archive_roots,
    collect_round_roots,
    collect_sessions_per_gpu,
    collect_stage1_archive,
    collect_stage1_variant,
    traj_archive_root,
)
from dexengine.core.layout.generations import (  # noqa: F401
    COMPARE_GENERATIONS,
    GENERATION_FILE,
    NAMED_ROSTERS,
    STAGE_ROOTS,
    compare_root,
    data_generation,
    frozen_datasets,
    generation_record,
    generation_rel,
    roster_root,
    stage_kind,
    taco_source_root,
)
from dexengine.core.layout.layers import (  # noqa: F401
    COOKED_HULL_EXT,
    LAYERS,
    PV_SFX_PAT,
    STORAGE,
    cooked_hull_file,
    data_tops,
    object_part_ref,
    object_path,
    object_ref,
    render,
    variant_source,
)
from dexengine.core.layout.registry import (  # noqa: F401
    AXIS_HAND,
    DATA_ROOT_ENV,
    DATASETS,
    EMBODIMENTS,
    RETARGET_HAND_ASSETS,
    WORKSPACES,
    DatasetSpec,
    asset_hash,
    axis_spec_dir,
    clip_id_from_task,
    dataset,
    identify_asset,
)
from dexengine.core.layout.roots import (  # noqa: F401
    _LAYOUT_CFG,
    LAYOUT_CONFIG_ENV,
    SANDBOX_ENV,
    SANDBOX_LAYERS,
    LayoutConfig,
    configured_data_root,
    confine_layer,
    data_abspath,
    data_path,
    data_root,
    holds_data,
    layout_config,
    owner_root,
    repo_place,
    root_kind,
    sandbox_root,
    urdf_mesh,
)
from dexengine.core.layout.runs import (  # noqa: F401
    CANON_KEY,
    COLLECT_WORK_ROOT_ENV,
    HAND_OWNER_FILE,
    PROVISIONAL_FILE,
    RL_MIXED,
    RUN_CKPT_ROOT,
    RUN_HAND_ROOTS,
    RUN_LAYERS,
    RUN_RENDER_OUT,
    RUN_SHARED,
    RUN_WORK_AREAS,
    collect_work_root_env,
    hand_provisional,
    render_run,
)
from dexengine.core.layout.shadow import (  # noqa: F401
    _SHADOW_GUARD,
    SHADOW_FILE,
    data_shadow,
    shadow_guard,
    write_shadow_record,
)
from dexengine.core.layout.sidecars import (  # noqa: F401
    SIDECAR_OWN_LAYER,
    SIDECAR_OWN_REGISTRY,
    sidecar_own_clips,
    sidecar_own_registry,
    sidecar_rel,
    sidecar_shared_rel,
)
