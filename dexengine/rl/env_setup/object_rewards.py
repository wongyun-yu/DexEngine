"""Object reward terms added to the env cfg: object keypoint tracking and articulated-object joint angles."""

from __future__ import annotations

from dexengine.rl.env_setup.object_rewards_left import add_obj_keypoint_terms_left
from dexengine.rl.env_setup.objects import abspath_multi


def add_obj_keypoint_terms(cfg, args_cli, *, optimize: bool, extend_ramp: bool):
    """Register object-keypoint reward/metric terms on *cfg*.

    Always (regardless of *optimize*):
      - Adds diag_object_keypoint_{i+1}_err (RewardObjKeypointDiag, weight=1e-9,
        component='err_cm') for logging only — no optimisation effect.
      - Adds diag_object_keypoint_mean_err when mode='both'.
      - Mirrors all diag terms for the left hand when --bimanual is set.

    Only when *optimize* is True:
      - ALSO adds the main object_keypoint_{i+1} term (RewardObjKeypoint,
        weight=--obj_keypoint_weight) that enters the training loss.

    Only when *optimize* and *extend_ramp* are both True:
      - Extends cfg.curriculum.obj_follow_schedule params['obj_terms'] with
        the main term names so they ramp identically to obj_pos.

    Returns the list of MAIN term names added (empty when optimize=False).
    """
    return _wire_obj_keypoint_terms(cfg, args_cli, optimize, extend_ramp)


def _wire_obj_keypoint_terms(cfg, args_cli, optimize, extend_ramp):
    """Load the keypoints of the right object and register its keypoint terms (and the left ones when bimanual)."""
    if not hasattr(cfg, "rewards"):
        return []

    import os as _os_kp

    import numpy as _np_kp

    from isaaclab.managers import RewardTermCfg as _RewKP

    from dexengine.rl.tasks.mdp.terms import RewardObjKeypoint as _ROK
    from dexengine.rl.tasks.mdp.terms import RewardObjKeypointDiag as _ROKD

    _ctx = "eval_cfg_overrides" if not extend_ramp else "train_cfg_overrides"
    _kp_mesh_src = getattr(args_cli, "object_mesh_dir", None)
    if not _kp_mesh_src:
        raise ValueError(
            "[obj-keypoint] the active-side --object_mesh_dir_* is required "
            "(the knife-120 fallback default was removed)"
        )
    _kp_mesh_dir = abspath_multi(_kp_mesh_src)
    _kp_multi = "," in _kp_mesh_dir
    _KP_W = float(getattr(args_cli, "obj_keypoint_weight", 1.25))
    _KP_MODE = str(getattr(args_cli, "obj_keypoint_mode", "both"))
    _KP_DIAG_W = 1e-9
    _kp_base = {"side": "right", "k": 80.0, "mesh_dir": _kp_mesh_dir}
    _kp_diag_base = {"side": "right", "mesh_dir": _kp_mesh_dir}
    _kp_far_flag = 0
    if _kp_multi:
        # --multi_object: per-clip keypoint tables + per-clip farthest selection live in
        # the terms (kp_farthest=1); registration keeps single W&B-stable term names.
        if _KP_MODE == "farthest":
            _kp_indices = [0]
            _kp_far_flag = 1
            print(f"[{_ctx}] OBJ-KEYPOINT mode=FARTHEST (multi-object: per-clip farthest in-term)")
        else:
            _kp_indices = [0, 1]
        _far_idx = 0  # tip-z diags use per-clip farthest via kp_farthest
    else:
        _kp_coords = _np_kp.load(_os_kp.path.join(_kp_mesh_dir, "keypoints_fps2.npy"))  # (2,3) object-local

        _far_idx, _kp_indices = _farthest_keypoint(_KP_MODE, _ctx, _kp_coords, _kp_mesh_dir, _np_kp, _os_kp)

    _kp_term_names = _register_keypoint_terms(
        cfg,
        optimize,
        extend_ramp,
        _KP_DIAG_W,
        _KP_W,
        _ROK,
        _ROKD,
        _RewKP,
        _ctx,
        _far_idx,
        _kp_base,
        _kp_diag_base,
        _kp_far_flag,
        _kp_indices,
        _kp_multi,
    )

    if optimize:
        _kp_coord_str = "per-clip (multi-object)" if _kp_multi else str(_kp_coords.tolist())
        print(
            f"[{_ctx}] OBJ-KEYPOINT REWARD: mode={_KP_MODE}, indices={_kp_indices}, "
            f"terms={_kp_term_names}; local coords={_kp_coord_str}; "
            f"weight={_KP_W} each; k=80; ramps with obj_pos"
        )
    else:
        print(
            f"[{_ctx}] OBJ-KEYPOINT DIAG (logging only): mode={_KP_MODE}, indices={_kp_indices}; "
            f"mesh_dir={_kp_mesh_dir}; diag weight={_KP_DIAG_W} (no loss contribution)"
        )

    add_obj_keypoint_terms_left(
        cfg, args_cli, optimize, extend_ramp, _KP_DIAG_W, _KP_MODE, _KP_W, _ROK, _ROKD, _RewKP, _ctx, _np_kp, _os_kp
    )

    return _kp_term_names


def _farthest_keypoint(_KP_MODE, _ctx, _kp_coords, _kp_mesh_dir, _np_kp, _os_kp):
    """Farthest keypoint (the knife tip), computed in every mode."""
    # Farthest keypoint (= knife TIP) — computed regardless of mode: the tip-z diags below
    # always target it, even when the reward optimizes 'both' keypoints.
    _sp = _np_kp.load(_os_kp.path.join(_kp_mesh_dir, "surface_points.npz"))["points"].astype(_np_kp.float64)
    _centroid = _sp.mean(0)
    _dists = _np_kp.linalg.norm(_kp_coords.astype(_np_kp.float64) - _centroid, axis=1)
    _far_idx = int(_np_kp.argmax(_dists))

    # Select which keypoint index/indices to register
    if _KP_MODE == "farthest":
        _kp_indices = [_far_idx]
        print(
            f"[{_ctx}] OBJ-KEYPOINT mode=FARTHEST: centroid={_np_kp.round(_centroid, 4).tolist()}; "
            f"|kp-centroid|={[round(float(d) * 100, 2) for d in _dists]}cm -> farthest=kp_index {_far_idx} "
            f"({_dists[_far_idx] * 100:.2f}cm)"
        )
    else:
        _kp_indices = [0, 1]
    return _far_idx, _kp_indices


def _register_keypoint_terms(
    cfg,
    optimize,
    extend_ramp,
    _KP_DIAG_W,
    _KP_W,
    _ROK,
    _ROKD,
    _RewKP,
    _ctx,
    _far_idx,
    _kp_base,
    _kp_diag_base,
    _kp_far_flag,
    _kp_indices,
    _kp_multi,
):
    """One diag term per selected keypoint, plus the main term when the keypoints are optimized."""
    # One diag term (always) + optional main term (optimize only) per selected index
    _kp_term_names = []
    for _i in _kp_indices:
        _nm = f"object_keypoint_{_i + 1}"
        if optimize:
            _kp_term_names.append(_nm)
            setattr(
                cfg.rewards,
                _nm,
                _RewKP(func=_ROK, weight=_KP_W, params=dict(_kp_base, kp_index=_i, kp_farthest=_kp_far_flag)),
            )
        setattr(
            cfg.rewards,
            f"diag_{_nm}_err",
            _RewKP(
                func=_ROKD,
                weight=_KP_DIAG_W,
                params=dict(_kp_diag_base, component="err_cm", kp_index=_i, kp_farthest=_kp_far_flag),
            ),
        )

    if len(_kp_indices) == 2:  # mean-error diag only meaningful with both keypoints
        cfg.rewards.diag_object_keypoint_mean_err = _RewKP(
            func=_ROKD, weight=_KP_DIAG_W, params=dict(_kp_diag_base, component="mean", kp_index=0)
        )

    # TIPZ diags (knife-specific): tip (= farthest keypoint) height above the
    # DESK + drag indicator. ALWAYS registered (both train and eval paths call this helper), so
    # every future run logs Diag/object_tip_z_cm + Diag/object_tip_drag to W&B/TB — the metrics
    # that actually SEE tip drag (completion/center-lift are blind to it; ref lifts the tip by
    # rotating the knife). tip_drag episode-mean = fraction of ref-airborne time spent dragging.
    cfg.rewards.diag_object_tip_z_cm = _RewKP(
        func=_ROKD,
        weight=_KP_DIAG_W,
        params=dict(_kp_diag_base, component="tip_z_cm", kp_index=_far_idx, kp_farthest=1 if _kp_multi else 0),
    )
    cfg.rewards.diag_object_tip_drag = _RewKP(
        func=_ROKD,
        weight=_KP_DIAG_W,
        params=dict(_kp_diag_base, component="tip_drag", kp_index=_far_idx, kp_farthest=1 if _kp_multi else 0),
    )
    print(
        f"[{_ctx}] TIPZ DIAGS registered: diag_object_tip_z_cm + diag_object_tip_drag "
        f"(tip=kp{_far_idx}, desk_z=0.76, airborne>+5cm, drag<+2cm)"
    )

    # Extend ObjFollowSchedule to ramp the main keypoint term(s) identically to obj_pos
    if optimize and extend_ramp:
        _ramp_keypoint_terms(cfg, _ctx, _kp_term_names)
    return _kp_term_names


def _ramp_keypoint_terms(cfg, _ctx, _kp_term_names):
    """Add the keypoint terms to the object-follow schedule (or warn that they will not ramp)."""
    _sched = getattr(getattr(cfg, "curriculum", None), "obj_follow_schedule", None)
    if _sched is not None:
        _sched.params["obj_terms"].extend(_kp_term_names)
        print(f"[{_ctx}] OBJ-FOLLOW SCHEDULE extended with keypoint terms: {', '.join(_kp_term_names)}")
    else:
        print(
            f"[{_ctx}] WARNING: OBJ-KEYPOINT REWARD set but ObjFollowSchedule "
            f"not active (--obj_follow_schedule_target not >1). Keypoints will NOT ramp."
        )


#: Joint-angle tracking reward (rule D1-c): the same size as obj_rot (cfg/task.py: obj_rot weight 2.0, k 3.0). k is per
#: unit of the joint kind (hinge rad · slide travel; unit owner artic_judge.err_units, the same function as the θ
# judge).
#: These two constants are the only owner (right, left, train and eval read them here).
ARTIC_REWARD_WEIGHT = 2.0
ARTIC_REWARD_K = 3.0


def _add_obj_artic_done(cfg, ctx: str, sides) -> None:
    """Two termination terms per articulated side: sanity (cut before a reset penetration kick spins the hinge until
    the state goes non-finite) and the θ judge (artic_judge: training termination = fresh-process judge termination,
    term names owned by judge_terms). `sides` = (side, reward name, object_key). A cfg without a termination manager
    has no place for the judge terms, so it fails."""
    if not hasattr(cfg, "terminations"):
        raise ValueError(
            f"[ARTIC] {ctx}: articulated object but no cfg.terminations — nowhere to put the θ judge terms"
        )
    from isaaclab.managers import TerminationTermCfg as _DoneA

    from dexengine.rl.judge import artic as artic_judge
    from dexengine.rl.judge import terms as judge_terms
    from dexengine.rl.tasks.mdp.rewards.task_artic import ObjArtiThetaJudge as _OATJ
    from dexengine.rl.tasks.mdp.terminations.hand import ArticulatedObjectSanity as _AOS

    _san, _theta = [], []
    for _sd, _nm, _ok in sides:
        _sfx = "_left" if _nm.endswith("_left") else ""
        for _tn, _func, _params, _out in (
            ("obj_arti_sanity" + _sfx, _AOS, {"object_key": _ok}, _san),
            (judge_terms.ARTIC_THETA_TERM + _sfx, _OATJ, {"side": _sd, "object_key": _ok}, _theta),
        ):
            if getattr(cfg.terminations, _tn, None) is not None:
                raise ValueError(f"[ARTIC] {ctx}: cfg.terminations.{_tn} already exists — two owners")
            setattr(cfg.terminations, _tn, _DoneA(func=_func, time_out=False, params=_params))
            _out.append(_tn)
    print(
        f"[ARTIC][sanity] {ctx}: articulated-object sanity terminations {_san} — |v_root|>10 m/s · |ω|>55 rad/s · "
        "non-finite state (the speed rule skips steps that start inside the SPAWN_HOLD window; window owner "
        "object_pin)",
        flush=True,
    )
    print(judge_terms.artic_marker(_theta, artic_judge.describe()), flush=True)


def add_obj_artic_terms(cfg, args_cli, ctx: str, *, extend_ramp: bool):
    """Rule D1-c: register `obj_arti[_left]` = RewardObjArtic **at runtime**, on articulated clips only (single owner).

    The one gate is `args_cli.object_articulated_<side>` (a run property the sequence_dir entry sets from the tree's
    `objects[].articulation`). Rigid clips get no term, so the reward manager table stays byte-identical (default off).
    - Sides: right always; left with `--bimanual` and `object_articulated_left` (with a shared object the left term
      reads the same prim, like obj_pos_left/obj_rot_left; `_apply_shared_object` re-points object_key).
    - Ramp: with `--obj_follow_schedule_target` on, the term joins obj_terms with the keypoints and ramps like obj_pos.
    - Must run **before** `_apply_shared_object` (a term registered after it cannot find object_left).
    Marker `[ARTIC][reward] <ctx>: …`. Returns the registered names.
    """
    if not hasattr(cfg, "rewards"):
        return []
    from isaaclab.managers import RewardTermCfg as _RewA

    from dexengine.rl.tasks.mdp.terms import RewardObjArtic as _ROA

    _active_side = str(getattr(args_cli, "side", "right") or "right")
    _bim = bool(getattr(args_cli, "bimanual", 0))
    _cands = []  # (side, term name, object_key)
    if _bim or _active_side == "right":
        _cands.append(("right", "obj_arti", "object_right"))
    if _bim:
        _cands.append(("left", "obj_arti_left", "object_left"))
    elif _active_side == "left":
        _cands.append(("left", "obj_arti", "object_left"))
    # An idle side (rule D16) has no object, hence no reward and no termination. Its articulation flag mirrors the
    # active side's for the spawn placeholder (train_ready_util.run_flags), so going by the flag alone would register
    # an idle θ term, print it, and have apply_idle_side delete it: the marker and the termination manager would
    # disagree and the sidecar would void the judge row. Leave it out of the registration.
    _idle = str(getattr(args_cli, "idle_side", "") or "")
    added = []
    for _sd, _nm, _ok in _cands:
        if _sd == _idle or not int(getattr(args_cli, f"object_articulated_{_sd}", 0) or 0):
            continue
        if getattr(cfg.rewards, _nm, None) is not None:
            raise ValueError(f"[ARTIC][reward] {ctx}: cfg.rewards.{_nm} already exists — two owners")
        setattr(
            cfg.rewards,
            _nm,
            _RewA(func=_ROA, weight=ARTIC_REWARD_WEIGHT, params={"k": ARTIC_REWARD_K, "side": _sd, "object_key": _ok}),
        )
        added.append(_nm)
    if not added:
        return []
    _add_obj_artic_done(cfg, ctx, [c for c in _cands if c[1] in added])
    _sched = getattr(getattr(cfg, "curriculum", None), "obj_follow_schedule", None) if extend_ramp else None
    if _sched is not None:
        _sched.params["obj_terms"].extend(added)
    _ramp = "obj_follow_schedule" if _sched is not None else "none"
    print(
        f"[ARTIC][reward] {ctx}: joint-angle tracking reward registered {added} = RewardObjArtic("
        f"weight={ARTIC_REWARD_WEIGHT}, k={ARTIC_REWARD_K}, exp(-k·err), err = |ref_θ−θ| in the joint kind's unit "
        f"(hinge rad · slide travel), reference {{side}}_obj_arti) · ramp={_ramp} · gate "
        f"object_articulated_right={int(getattr(args_cli, 'object_articulated_right', 0) or 0)} "
        f"left={int(getattr(args_cli, 'object_articulated_left', 0) or 0)}",
        flush=True,
    )
    return added
