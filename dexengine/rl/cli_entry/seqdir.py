"""PRE-KIT: resolve the training data of a trainer run and preflight every file the env will read.

Two data paths, exactly one runs (``resolve_data``):

* ``--sequence_dir <folder>``: the folder's clips are trained by roster (``train_ready_util.resolve_roster`` over
  the hand's folded ledger, not by counting the folder). For every roster clip the active-side object (and, for
  --bimanual, the left partner object) is read from its ``task_info.json`` layout, and the object USD / surface
  points and the wrist / arm-IK caches are required on disk. The clips must form one run (one run key: shared
  object, idle side, articulated, aerial scene). The result feeds the existing plumbing: data_dir = the folder,
  sequence = comma-list, object flags = comma-lists (--multi_object switched on when objects differ).
* otherwise (``--sequence <clip> --data_dir <root>`` or a whole root): the tree rig and the checkpoint/clip canon
  check run on those trees.

Both paths end with the contact_mano check. Standard library and dependency-free dexengine modules only; the
articulation canon check needs numpy, so it runs in a child process.
"""

from __future__ import annotations

import os

from dexengine.core import paths as _pp
from dexengine.rl.cli_entry.seqdir_clips import (
    apply_run_group,
    check_artic_and_missing,
    collect_per_clip,
    contact_mano_check,
)
from dexengine.robots import selection as _rs


def resolve_data(parser, args_cli) -> None:
    """Run the --sequence_dir resolution, or the tree checks of the single-clip / data-root path."""
    if getattr(args_cli, "sequence_dir", None):
        _resolve_sequence_dir(parser, args_cli)
    else:
        _check_trees(parser, args_cli)


def _resolve_sequence_dir(parser, args_cli) -> None:
    _arm_sel, _hand_sel, _sd, _sd_bi, _sd_ds, _sd_seqs, _sd_side = _check_dataset_and_side(parser, args_cli)
    (
        _ql_mod,
        _roster,
        _sd_artic_q,
        _sd_desk,
        _sd_desk_absent,
        _sd_discs,
        _sd_groups,
        _sd_ids,
        _sd_ids_left,
        _sd_ik_dir,
        _sd_ik_dir_L,
        _sd_missing,
        _sd_need_ik,
        _sd_paths,
        _sd_seqs,
        _sd_src,
        _sd_wrist_dir,
        _sd_wrist_dir_L,
        _tp_mod,
        _tru,
    ) = _artifact_paths_and_roster(parser, args_cli, _arm_sel, _hand_sel, _sd, _sd_bi, _sd_ds, _sd_seqs, _sd_side)
    collect_per_clip(
        _ql_mod,
        _sd,
        _sd_artic_q,
        _sd_bi,
        _sd_desk,
        _sd_desk_absent,
        _sd_discs,
        _sd_groups,
        _sd_ids,
        _sd_ids_left,
        _sd_ik_dir,
        _sd_ik_dir_L,
        _sd_missing,
        _sd_need_ik,
        _sd_paths,
        _sd_seqs,
        _sd_side,
        _sd_src,
        _sd_wrist_dir,
        _sd_wrist_dir_L,
        _tp_mod,
        _tru,
    )
    check_artic_and_missing(parser, _sd_artic_q, _sd_ds, _sd_missing, _sd_side)
    apply_run_group(
        parser,
        args_cli,
        _roster,
        _sd,
        _sd_bi,
        _sd_desk,
        _sd_desk_absent,
        _sd_discs,
        _sd_groups,
        _sd_ids,
        _sd_ids_left,
        _sd_paths,
        _sd_seqs,
        _sd_side,
        _sd_src,
        _tru,
    )


def _artifact_paths_and_roster(parser, args_cli, _arm_sel, _hand_sel, _sd, _sd_bi, _sd_ds, _sd_seqs, _sd_side):
    """Roster, rig and canon checks, checkpoint / artifact paths and the wrist / arm-IK cache dirs."""
    # Training runs on the roster, owned by the hand's folded ledger (ADOPTED): a combo tree folder also holds
    # trees that passed the judge but failed a gate. Rules (train_ready_util.resolve_roster): no ledger -> refuse;
    # DEXENGINE_SEQ_LIST=<file> -> every entry in ADOPTED and in the folder; a view folder -> the whole view in
    # ADOPTED; a tree root folder -> ADOPTED ∩ folder; DEXENGINE_SEQ_LIST=all -> explicit opt-out (marked);
    # a frozen generation without a ledger folder -> only with an explicit DEXENGINE_ROSTER=<named source>
    # (dexengine.core.layout.NAMED_ROSTERS). Prints the [ROSTER] marker and writes <exp>_roster.txt in the run tree's
    # work_log (source, ledger path, sha256, count, list).
    from dexengine.placement import train_ready_util as _tru

    _sd_emb = os.environ.get("DEXENGINE_EMBODIMENT", "").strip()
    if not _sd_emb:
        parser.error("--sequence_dir needs DEXENGINE_EMBODIMENT (roster = the folded ledger axis, no default)")
    if _sd_bi and _sd_emb != "bimanual":
        parser.error(
            f"--bimanual but DEXENGINE_EMBODIMENT={_sd_emb} — bimanual training uses only the bimanual-axis roster"
        )
    try:
        _sd_id_combo, _sd_is_view = _pp.seq_dir_identity(_sd)
    except ValueError as _e:
        parser.error(f"[ROSTER] {_e}")
    try:
        _r_arm, _r_hand, _r_var = _pp.split_combo(_sd_id_combo, _rs.ARM_REGISTRY.keys(), _rs.HAND_REGISTRY.keys())
    except ValueError as _e:
        parser.error(f"[ROSTER] sequence folder: {_e}")
    try:
        _roster = _tru.resolve_roster(
            combo=_sd_id_combo,
            hand=_r_hand,
            dataset=_sd_ds,
            emb=_sd_emb,
            variant=_r_var,
            trees=_sd_seqs,
            is_view=_sd_is_view,
            seq_dir=_sd,
            seq_list=os.environ.get("DEXENGINE_SEQ_LIST", "").strip(),
            source=os.environ.get("DEXENGINE_ROSTER", "").strip(),
            root=_pp.main_root(),
        )
    except ValueError as _e:
        parser.error(f"[ROSTER] {_e}")
    _sd_seqs = _roster["train"]
    print(_roster["marker"], flush=True)
    # The trees' rig must be the run hand's registered rig (train_ready_util.tree_rig_check).
    try:
        print(
            _tru.tree_rig_check(hand=_hand_sel, seq_dir=_sd, trees=_sd_seqs, roster_source=_roster["source"]),
            flush=True,
        )
    except ValueError as _e:
        parser.error(f"[TREE_RIG] {_e}")
    # [CANON_USE] (dexengine.rl.checkpoints): a campaign cell's checkpoint runs only when each clip's task_info
    # fingerprint matches the cell's training fingerprint; refused before the env is built.
    from dexengine.rl import checkpoints as _cu

    try:
        for _cu_m in _cu.check_args(args_cli, _sd, _sd_seqs, _r_hand):
            print(_cu_m, flush=True)
    except ValueError as _e:
        parser.error(f"[CANON_USE] {_e}")
    if getattr(args_cli, "exp_name", None):
        _ros = _pp.work_area("work_log") / f"{args_cli.exp_name}_roster.txt"  # the run tree, not the cwd
        _tru.write_roster_record(_roster, str(_ros))
    return _dataset_artifact_paths(parser, _arm_sel, _hand_sel, _sd_bi, _sd_ds, _sd_seqs, _sd_side, _roster, _tru)


def _dataset_artifact_paths(parser, _arm_sel, _hand_sel, _sd_bi, _sd_ds, _sd_seqs, _sd_side, _roster, _tru):
    """Per-dataset artifact paths (object USDs, meshes, precompute caches)."""
    # Per-dataset artifact paths (object USDs, meshes, precompute caches).
    from dexengine.core import artifact_paths as _tp_mod

    try:
        _sd_paths = _tp_mod.get(_sd_ds)
    except KeyError as _e:
        parser.error(f"--sequence_dir: dataset '{_sd_ds}' has no path resolver — {_e}")
    # Object identity comes from qpos_layout.Layout (objects[] role=main; legacy trees: the basename of
    # <side>_object_mesh_dir).
    from dexengine.core import qpos_layout as _ql_mod

    # Cache preflight on the combo caches the env loads (robots.selection.combo_data_paths).
    _sd_wrist_dir = _sd_ik_dir = None
    _cdp = _rs.combo_data_paths(_rs.ARM_REGISTRY.get(_arm_sel), _rs.HAND_REGISTRY.get(_hand_sel))
    _sd_wrist_dir = _cdp["wrist_poses_dir_right"].replace("_right", f"_{_sd_side}")
    # A floating hand (arm spec FLOATING) has no arm IK cache (wrist = root, reset = reference wrist).
    _sd_need_ik = not getattr(_rs.ARM_REGISTRY.get(_arm_sel), "is_floating", False)
    # The env's arm-IK cache loader appends DEXENGINE_ARM_IK_DIR_SUFFIX after the side swap and falls back to the
    # unsuffixed directory when the suffixed one is missing, so the preflight requires the same final path, with
    # the same suffix rule: default `_cont25`, and an explicit "" adopts paths.ik_suffix() when that directory
    # exists.
    _sd_ik_base = _cdp["arm_ik_poses_dir_right"].replace("_right", f"_{_sd_side}")
    _sd_ik_sfx = os.environ.get("DEXENGINE_ARM_IK_DIR_SUFFIX", "_cont25")
    if not _sd_ik_sfx and os.path.isdir(_sd_ik_base + _pp.ik_suffix()):
        _sd_ik_sfx = _pp.ik_suffix()
        print(
            f"[arm_ik] preflight: no suffix given → {_sd_ik_sfx} chosen "
            f"(same rule as events.py). To state it, set "
            f"DEXENGINE_ARM_IK_DIR_SUFFIX={_sd_ik_sfx}",
            flush=True,
        )
    _sd_ik_dir = _sd_ik_base + _sd_ik_sfx
    # The left cache directories are the active-side ones with the side suffix swapped.
    _sd_wrist_dir_L = _sd_wrist_dir.replace("_right", "_left") if (_sd_bi and _sd_wrist_dir) else None
    _sd_ik_dir_L = _sd_ik_dir.replace("_right", "_left") if (_sd_bi and _sd_ik_dir) else None
    _sd_ids, _sd_ids_left, _sd_missing = [], [], []
    # Run attributes (shared object, idle side, articulated, aerial scene) are collected per clip as one
    # run_group; run homogeneity is decided below by train_ready_util.one_run (the key the train-set selector
    # bundles by).
    _sd_groups, _sd_src = [], set()  # per-clip run_group · identity source (objects[]|legacy)
    _sd_discs = {}  # support discs of aerial-scene clips (tree aerial_support)
    # Desk support of desk-scene trees (task_info[qpos_layout.DESK_SUPPORT_KEY]); a tree without it is from an
    # older bake and reads as no support (the count is printed).
    _sd_desk, _sd_desk_absent = {}, []

    # The articulation canon check (artic.ranges.not_canonical) imports numpy, which must not load before Kit
    # ([PRE_APP]); the loop collects the queries and one child process answers them after the loop.
    _sd_artic_q = []  # (report prefix, object id, articulation statement)
    return (
        _ql_mod,
        _roster,
        _sd_artic_q,
        _sd_desk,
        _sd_desk_absent,
        _sd_discs,
        _sd_groups,
        _sd_ids,
        _sd_ids_left,
        _sd_ik_dir,
        _sd_ik_dir_L,
        _sd_missing,
        _sd_need_ik,
        _sd_paths,
        _sd_seqs,
        _sd_src,
        _sd_wrist_dir,
        _sd_wrist_dir_L,
        _tp_mod,
        _tru,
    )


def _check_dataset_and_side(parser, args_cli):
    """Sequence / side / dataset axis checks."""
    _arm_sel, _hand_sel = args_cli.arm, args_cli.hand
    if args_cli.sequence:
        parser.error("--sequence_dir is mutually exclusive with --sequence")
    # Uni runs resolve one side; bimanual runs also resolve the left partner object of every clip
    # (task_info.json). --dual_eval is not supported here.
    _sd_bi = bool(getattr(args_cli, "bimanual", 0))
    if getattr(args_cli, "dual_eval", 0):
        parser.error("--sequence_dir supports uni or --bimanual builds (no --dual_eval)")
    _sd_side = "left" if getattr(args_cli, "left_uni", 0) else "right"
    if _sd_bi and _sd_side != "right":
        parser.error("--sequence_dir + --bimanual: the active side must be right (exclusive with --left_uni)")
    for _s2 in [_sd_side] + (["left"] if _sd_bi else []):
        if getattr(args_cli, f"object_usd_{_s2}", None) or getattr(args_cli, f"object_mesh_dir_{_s2}", None):
            parser.error("--sequence_dir resolves objects itself — drop the --object_usd_*/--object_mesh_dir_* flags")
    _sd = os.path.abspath(args_cli.sequence_dir)
    if not os.path.isdir(_sd):
        parser.error(f"--sequence_dir does not exist: {_sd}")
    _sd_seqs = sorted(
        d for d in os.listdir(_sd) if os.path.isfile(os.path.join(_sd, d, "0", "trajectory_kinematic.npz"))
    )
    if not _sd_seqs:
        parser.error(f"--sequence_dir: no sequence dirs (<seq>/0/trajectory_kinematic.npz) under {_sd}")
    # Dataset axis: --dataset and DEXENGINE_DATASET must agree (combo_data_paths reads the env, the object
    # resolver reads --dataset; disagreeing values would mix the caches of one dataset with the objects of
    # another). There is no default. The roster is keyed by this axis, so it is resolved first.
    _sd_ds_env = os.environ.get("DEXENGINE_DATASET", "").strip()
    _sd_ds = args_cli.dataset or _sd_ds_env
    if not _sd_ds:
        parser.error("--sequence_dir needs --dataset or DEXENGINE_DATASET (no default)")
    if _sd_ds_env and args_cli.dataset and _sd_ds_env != args_cli.dataset:
        parser.error(
            f"--dataset {args_cli.dataset} and DEXENGINE_DATASET={_sd_ds_env} differ — "
            f"the dataset has one owner (the launcher)"
        )
    if not _sd_ds_env:
        os.environ["DEXENGINE_DATASET"] = _sd_ds  # combo_data_paths (env) reads the same value
    return _arm_sel, _hand_sel, _sd, _sd_bi, _sd_ds, _sd_seqs, _sd_side


def _check_trees(parser, args_cli) -> None:
    """Tree rig and checkpoint canon checks of the data path outside --sequence_dir (single clip, render, judge).

    Same rule functions as the --sequence_dir path (train_ready_util.tree_rig_check, checkpoints.check_args).
    Trees without a rig stamp (frozen generations from before the stamp) pass only with an explicit
    DEXENGINE_ROSTER=<registered name>; only the name's registration is checked here (no roster on this path).
    heldout_data_dir is not checked: nothing in the entry or the env reads it.
    """
    _hand_sel = args_cli.hand
    from dexengine.placement import train_ready_util as _tru

    _tr_root = os.path.abspath(args_cli.data_dir)
    if args_cli.sequence:
        _tr_trees = [t.strip() for t in args_cli.sequence.split(",") if t.strip()]
    elif os.path.isfile(os.path.join(_tr_root, "0", "trajectory_kinematic.npz")):
        _tr_root, _tr_trees = os.path.dirname(_tr_root), [os.path.basename(_tr_root)]  # --data_dir is one tree
    else:
        _tr_trees = (
            sorted(
                d
                for d in os.listdir(_tr_root)
                if os.path.isfile(os.path.join(_tr_root, d, "0", "trajectory_kinematic.npz"))
            )
            if os.path.isdir(_tr_root)
            else []
        )
    _tr_named = os.environ.get("DEXENGINE_ROSTER", "").strip()
    if _tr_named:
        try:
            _tru.named_roster(_tr_named)  # an unregistered name is refused here
        except (KeyError, ValueError) as _e:
            parser.error(f"[TREE_RIG] DEXENGINE_ROSTER={_tr_named}: {_e}")
    try:
        print(
            _tru.tree_rig_check(
                hand=_hand_sel,
                seq_dir=_tr_root,
                trees=_tr_trees,
                roster_source=f"named:{_tr_named}" if _tr_named else "one-clip branch (no roster)",
            ),
            flush=True,
        )
    except ValueError as _e:
        parser.error(f"[TREE_RIG] {_e}")
    from dexengine.rl import checkpoints as _cu  # [CANON_USE] of --checkpoint / --dual_ckpt_*

    try:
        for _cu_m in _cu.check_args(args_cli, _tr_root, _tr_trees, _hand_sel):
            print(_cu_m, flush=True)
    except ValueError as _e:
        parser.error(f"[CANON_USE] {_e}")
    contact_mano_check(parser, args_cli, _tr_root, _tr_trees)
