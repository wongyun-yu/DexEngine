"""Left-side object keypoint terms (bimanual): the mirror of object_rewards.add_obj_keypoint_terms on object_left
(keypoint paths, per-keypoint reward / diag terms, the tip-z and tip-drag diagnostics)."""

from __future__ import annotations

from dexengine.rl.env_setup.objects import abspath_multi


def add_obj_keypoint_terms_left(
    cfg, args_cli, optimize, extend_ramp, _KP_DIAG_W, _KP_MODE, _KP_W, _ROK, _ROKD, _RewKP, _ctx, _np_kp, _os_kp
):
    """Left-hand mirror of the keypoint terms (bimanual)."""
    _add_left_keypoint_terms(
        cfg, args_cli, optimize, extend_ramp, _KP_DIAG_W, _KP_MODE, _KP_W, _ROK, _ROKD, _RewKP, _ctx, _np_kp, _os_kp
    )


def _add_left_keypoint_terms(
    cfg, args_cli, optimize, extend_ramp, _KP_DIAG_W, _KP_MODE, _KP_W, _ROK, _ROKD, _RewKP, _ctx, _np_kp, _os_kp
):
    """Bimanual: the symmetric left keypoint reward / metric (object_left)."""
    # BIMANUAL: symmetric LEFT keypoint reward/metric (object_key='object_left',
    # mesh_dir=object_mesh_dir_left). No left object → skip; a left object without
    # keypoints_fps2.npy → error (keypoints are a declared training asset). Mirrors the
    # right block exactly (same mode/weight/k, _left term names, diag, schedule ramp).
    if bool(getattr(args_cli, "bimanual", 0)):
        _kp_mesh_dir_l = getattr(args_cli, "object_mesh_dir_left", None)
        _kp_l_path, _kp_multi_l, _kp_ok_l, _kp_mesh_dir_l = _left_keypoint_paths(_os_kp, _kp_mesh_dir_l)
        if _kp_ok_l:
            if not _kp_multi_l:
                _kp_mesh_dir_l = _os_kp.path.abspath(_kp_mesh_dir_l)
            _kp_coords_l, _kp_diag_base_l, _kp_indices_l, _kp_term_names_l = _register_left_keypoint_terms(
                cfg,
                optimize,
                _KP_DIAG_W,
                _KP_MODE,
                _KP_W,
                _ROK,
                _ROKD,
                _RewKP,
                _np_kp,
                _os_kp,
                _kp_l_path,
                _kp_mesh_dir_l,
                _kp_multi_l,
            )
            if len(_kp_indices_l) == 2:
                cfg.rewards.diag_object_keypoint_mean_err_left = _RewKP(
                    func=_ROKD, weight=_KP_DIAG_W, params=dict(_kp_diag_base_l, component="mean", kp_index=0)
                )
            _add_left_tip_z_terms(
                cfg,
                optimize,
                extend_ramp,
                _KP_DIAG_W,
                _KP_MODE,
                _KP_W,
                _ROKD,
                _RewKP,
                _ctx,
                _np_kp,
                _os_kp,
                _kp_coords_l,
                _kp_diag_base_l,
                _kp_indices_l,
                _kp_mesh_dir_l,
                _kp_multi_l,
                _kp_term_names_l,
            )
        elif not _kp_mesh_dir_l:
            print(f"[{_ctx}] (bimanual) OBJ-KEYPOINT LEFT SKIPPED: no left object (--object_mesh_dir_left unset)")
        else:
            # A left object without keypoints is an asset gap, not a design case — never a silent no-op.
            _miss_l = [
                _d
                for _d in _kp_mesh_dir_l.split(",")
                if _d and not _os_kp.path.exists(_os_kp.path.join(_os_kp.path.abspath(_d), "keypoints_fps2.npy"))
            ]
            raise FileNotFoundError(
                f"[{_ctx}] (bimanual) OBJ-KEYPOINT LEFT: keypoints_fps2.npy missing for {len(_miss_l)} left mesh dirs "
                f"({', '.join(_miss_l[:3])}{' …' if len(_miss_l) > 3 else ''}) — restore them "
                "(check: python3 -m dexengine.placement.train_ready) or launch with KEYPOINT=0."
            )


def _register_left_keypoint_terms(
    cfg,
    optimize,
    _KP_DIAG_W,
    _KP_MODE,
    _KP_W,
    _ROK,
    _ROKD,
    _RewKP,
    _np_kp,
    _os_kp,
    _kp_l_path,
    _kp_mesh_dir_l,
    _kp_multi_l,
):
    """Left keypoint parameter bases and one term per selected left keypoint."""
    _kp_far_flag_l = 0  # 1 only for a multi-object left side in farthest mode (the term picks the point per clip)
    _kp_base_l = {"side": "left", "k": 80.0, "mesh_dir": _kp_mesh_dir_l, "object_key": "object_left"}
    _kp_diag_base_l = {"side": "left", "mesh_dir": _kp_mesh_dir_l, "object_key": "object_left"}
    if _kp_multi_l:
        _kp_coords_l = None  # the term holds the per-clip table
        if _KP_MODE == "farthest":
            _kp_indices_l, _kp_far_flag_l = [0], 1
        else:
            _kp_indices_l = [0, 1]
    else:
        _kp_coords_l = _np_kp.load(_kp_l_path)
        if _KP_MODE == "farthest":
            _sp_l = _np_kp.load(_os_kp.path.join(_kp_mesh_dir_l, "surface_points.npz"))["points"].astype(_np_kp.float64)
            _centroid_l = _sp_l.mean(0)
            _dists_l = _np_kp.linalg.norm(_kp_coords_l.astype(_np_kp.float64) - _centroid_l, axis=1)
            _kp_indices_l = [int(_np_kp.argmax(_dists_l))]
        else:
            _kp_indices_l = [0, 1]
    _kp_term_names_l = []
    for _i in _kp_indices_l:
        _nm_l = f"object_keypoint_{_i + 1}_left"
        if optimize:
            _kp_term_names_l.append(_nm_l)
            setattr(
                cfg.rewards,
                _nm_l,
                _RewKP(func=_ROK, weight=_KP_W, params=dict(_kp_base_l, kp_index=_i, kp_farthest=_kp_far_flag_l)),
            )
        setattr(
            cfg.rewards,
            f"diag_{_nm_l}_err",
            _RewKP(
                func=_ROKD,
                weight=_KP_DIAG_W,
                params=dict(_kp_diag_base_l, component="err_cm", kp_index=_i, kp_farthest=_kp_far_flag_l),
            ),
        )
    return _kp_coords_l, _kp_diag_base_l, _kp_indices_l, _kp_term_names_l


def _left_keypoint_paths(_os_kp, _kp_mesh_dir_l):
    """(keypoint file, multi-object?, keypoint file(s) exist?, mesh dir) of the left object. A multi-object (comma)
    dir comes back made absolute, entry by entry (abspath_multi), as the terms read it."""
    # The left side handles multi-object (comma) dirs too, detected by the same rule as the right _kp_multi;
    # the per-clip keypoint table and the farthest-point choice live inside the term (kp_farthest=1).
    _kp_multi_l = bool(_kp_mesh_dir_l) and ("," in _kp_mesh_dir_l)
    if _kp_multi_l:
        _kp_mesh_dir_l = abspath_multi(_kp_mesh_dir_l)
        _kp_l_path = None
        _kp_ok_l = all(
            _os_kp.path.exists(_os_kp.path.join(_d, "keypoints_fps2.npy")) for _d in _kp_mesh_dir_l.split(",")
        )
    else:
        _kp_l_path = (
            _os_kp.path.join(_os_kp.path.abspath(_kp_mesh_dir_l), "keypoints_fps2.npy") if _kp_mesh_dir_l else None
        )
        _kp_ok_l = bool(_kp_mesh_dir_l and _kp_l_path and _os_kp.path.exists(_kp_l_path))
    return _kp_l_path, _kp_multi_l, _kp_ok_l, _kp_mesh_dir_l


def _add_left_tip_z_terms(
    cfg,
    optimize,
    extend_ramp,
    _KP_DIAG_W,
    _KP_MODE,
    _KP_W,
    _ROKD,
    _RewKP,
    _ctx,
    _np_kp,
    _os_kp,
    _kp_coords_l,
    _kp_diag_base_l,
    _kp_indices_l,
    _kp_mesh_dir_l,
    _kp_multi_l,
    _kp_term_names_l,
):
    """Left mirror of the tip-z and tip-drag diagnostics (object_left)."""
    # TIPZ LEFT mirror: left-object tip-z + drag diag (object_left). Farthest
    # left keypoint computed regardless of _KP_MODE (matches the right tip-z block above).
    # Auto-exempt from diag-stripping via the 'diag_object_tip_' prefix rule (line ~2069).
    if _kp_multi_l:
        _far_idx_l = 0  # per-clip farthest point inside the term (kp_farthest=1), as on the right
    else:
        _far_idx_l = int(
            _np_kp.argmax(
                _np_kp.linalg.norm(
                    _kp_coords_l.astype(_np_kp.float64)
                    - _np_kp.load(_os_kp.path.join(_kp_mesh_dir_l, "surface_points.npz"))["points"]
                    .astype(_np_kp.float64)
                    .mean(0),
                    axis=1,
                )
            )
        )
    cfg.rewards.diag_object_tip_z_cm_left = _RewKP(
        func=_ROKD,
        weight=_KP_DIAG_W,
        params=dict(_kp_diag_base_l, component="tip_z_cm", kp_index=_far_idx_l, kp_farthest=1 if _kp_multi_l else 0),
    )
    cfg.rewards.diag_object_tip_drag_left = _RewKP(
        func=_ROKD,
        weight=_KP_DIAG_W,
        params=dict(_kp_diag_base_l, component="tip_drag", kp_index=_far_idx_l, kp_farthest=1 if _kp_multi_l else 0),
    )
    print(
        f"[{_ctx}] (bimanual) TIPZ DIAGS LEFT registered: diag_object_tip_z_cm_left + "
        f"diag_object_tip_drag_left (tip=kp{_far_idx_l}, object_left)"
    )
    if optimize and extend_ramp:
        _sched_l = getattr(getattr(cfg, "curriculum", None), "obj_follow_schedule", None)
        if _sched_l is not None:
            _sched_l.params["obj_terms"].extend(_kp_term_names_l)
    print(
        f"[{_ctx}] (bimanual) OBJ-KEYPOINT {'REWARD' if optimize else 'DIAG'} LEFT: mode={_KP_MODE}, "
        f"indices={_kp_indices_l}, terms={_kp_term_names_l if optimize else '(diag only)'}; "
        f"mesh_dir={_kp_mesh_dir_l}; weight={_KP_W if optimize else _KP_DIAG_W} each"
    )
