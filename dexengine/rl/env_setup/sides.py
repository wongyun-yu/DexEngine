"""Side wiring of the env cfg: left-hand-only runs, the shared object of bimanual clips, idle sides and the per-hand
observation layout."""

from __future__ import annotations

from dexengine.rl.env_setup.objects import abspath_multi

# Body tracking indices (depth-first, 13 bodies per hand)
FINGERTIP_BODY_INDICES = {
    "thumb": 4,
    "index": 6,
    "middle": 8,
    "ring": 10,
    "pinky": 12,
}
PALM_BODY_INDEX = 0


def apply_left_uni(cfg, args_cli, tag):
    """LEFT-UNI side sweep: flip every rewards/terminations term with
    params['side']=='right' to 'left'.

    The left-uni factory build keeps the cfg-level reward/termination params at
    side='right' (single-sweep policy), and the runtime-registered terms
    (contact-pilot contact_match/closure/..., contact_hand termination, keypoint
    terms, residual_l2_wrist, hand-tracking fingertip/velocity_sanity) are ALSO
    side='right' — so ONE sweep here converts the whole term set to the LEFT
    hand. Term NAMES are untouched (object_tracking stays unsuffixed → the
    ObjTermAdaptive curriculum + threshold flags bind natively; runtime names
    like 'contact_match_right' keep their name for curriculum-patch lookups —
    only the params flip).

    MUST be called at the VERY END of BOTH train_cfg_overrides and
    _eval_cfg_overrides (after every runtime term registration, incl. obs_v2 /
    keypoint blocks) — a term registered AFTER the sweep would silently stay a
    dead right-side reward (risk #3 of the L-uni verify checklist; train/eval
    parity per the --arm_kp lesson).

    Also fail-fasts on any residual object_key=='object_left' (the left-uni
    scene has no such key — a leftover bimanual mirror would KeyError later or,
    worse, silently mis-bind).
    """
    if not getattr(args_cli, "left_uni", 0):
        return
    _swept, _left_over = [], []
    for _grp_name in ("rewards", "terminations"):
        _grp = getattr(cfg, _grp_name, None)
        if _grp is None:
            continue
        for _nm in list(vars(_grp).keys()):
            _term = getattr(_grp, _nm, None)
            _p = getattr(_term, "params", None)
            if not isinstance(_p, dict):
                continue
            if _p.get("side") == "right":
                _p["side"] = "left"
                _swept.append(f"{_grp_name}.{_nm}")
                # Terms relying on the SIGNATURE default object_key="object_right"
                # would silently bind the hidden right placeholder — inject the left
                # key when the term function accepts it (class terms: __call__).
                import inspect as _ins

                _fn = getattr(_term, "func", None)
                _sig_fn = _fn.__call__ if isinstance(_fn, type) else _fn
                try:
                    _has_ok = "object_key" in _ins.signature(_sig_fn).parameters
                except (TypeError, ValueError):
                    _has_ok = False
                if _has_ok and "object_key" not in _p:
                    _p["object_key"] = "object_left"
            if _p.get("object_key") == "object_right":
                _p["object_key"] = "object_left"
            if _p.get("object_name") == "object_right":
                _p["object_name"] = "object_left"
            if _p.get("object_key") == "object_right" or _p.get("object_name") == "object_right":
                _left_over.append(f"{_grp_name}.{_nm}")
    if _left_over:
        raise ValueError(f"[{tag}] LEFT-UNI: terms still bound to the RIGHT object after the sweep: {_left_over}.")
    print(f"[{tag}] LEFT-UNI side sweep: {len(_swept)} reward/termination terms side right->left: {_swept}")


# ── D16 idle side (bimanual): the side the clips declare inactive has no object ───────────────────────────────────────
#: The parameter names through which a manager term names the scene object it reads.
_OBJ_PARAM_KEYS = ("object_key", "object_name", "object_cfg")


def _term_object_binding(term):
    """The scene-object key a manager term reads — an explicit object param (a key string or a SceneEntityCfg), else the
    term class's ``OBJECT_KEY_DEFAULT`` (the default its __init__ reads), else the default of such a parameter in the
    term function's (or term class __call__'s) signature. None = the term names no object."""
    import inspect as _ins

    p = getattr(term, "params", None)
    p = p if isinstance(p, dict) else {}
    for k in _OBJ_PARAM_KEYS:
        if k in p:
            v = p[k]
            return v if isinstance(v, str) else getattr(v, "name", None)
    fn = getattr(term, "func", None)
    if isinstance(fn, type) and getattr(fn, "OBJECT_KEY_DEFAULT", None):
        return fn.OBJECT_KEY_DEFAULT
    try:
        sig = _ins.signature(fn.__call__ if isinstance(fn, type) else fn).parameters
    except (TypeError, ValueError):
        return None
    for k in _OBJ_PARAM_KEYS:
        if k in sig and sig[k].default is not _ins.Parameter.empty:
            d = sig[k].default
            return d if isinstance(d, str) else getattr(d, "name", None)
    return None


def manager_groups(cfg):
    """(group name, group cfg) for every manager group whose terms can bind an object — observations per obs group."""
    for _g in ("rewards", "terminations", "curriculum", "events"):
        _grp = getattr(cfg, _g, None)
        if _grp is not None:
            yield _g, _grp
    _obs = getattr(cfg, "observations", None)
    if _obs is not None:
        for _og in list(vars(_obs).keys()):
            _grp = getattr(_obs, _og, None)
            if _grp is not None and not isinstance(_grp, (str, int, float, bool)):
                yield f"observations.{_og}", _grp


def _idle_drop_scene(cfg, key, prim):
    """Remove the idle object prim and every scene sensor filtered on it (the five `<side>_*_obj_s`)."""
    dropped = []
    if getattr(cfg.scene, key, None) is not None:
        setattr(cfg.scene, key, None)
        dropped.append(key)
    for _nm in list(vars(cfg.scene).keys()):
        _ent = getattr(cfg.scene, _nm, None)
        if _ent is None or not hasattr(_ent, "prim_path"):
            continue
        _fpp = getattr(_ent, "filter_prim_paths_expr", None) or []
        if _ent.prim_path == prim or any(x == prim for x in _fpp):
            setattr(cfg.scene, _nm, None)
            dropped.append(_nm)
    return dropped


def _idle_sweep_terms(cfg, key, zero_width):
    """Drop every reward / termination / curriculum / event term bound to the idle object, and replace its object
    observation (ObjectStateV2) by zeros of the same width under the same name. An observation term of another kind
    bound to it has no known width — that raises instead of guessing."""
    from isaaclab.managers import ObservationTermCfg as _ObsTerm

    from dexengine.rl.tasks.mdp.observations.object_state_v2 import ObjectStateV2, idle_object_obs

    report = {"dropped": [], "zeroed": []}
    for _gname, _grp in manager_groups(cfg):
        for _nm in list(vars(_grp).keys()):
            _term = getattr(_grp, _nm, None)
            if _term is None or not hasattr(_term, "func") or _term_object_binding(_term) != key:
                continue
            if _gname.startswith("observations."):
                if getattr(_term, "func", None) is not ObjectStateV2:
                    raise ValueError(
                        f"[IDLE_SIDE] {_gname}.{_nm} reads idle object {key} but is not ObjectStateV2 — "
                        "the width to zero is unknown (no guessing)"
                    )
                setattr(_grp, _nm, _ObsTerm(func=idle_object_obs, params={"width": int(zero_width)}))
                report["zeroed"].append(f"{_gname}.{_nm}")
            else:
                setattr(_grp, _nm, None)
                report["dropped"].append(f"{_gname}.{_nm}")
    # a curriculum term that names a dropped termination by name would look it up at runtime — drop it too
    _gone = {d.split(".", 1)[1] for d in report["dropped"] if d.startswith("terminations.")}
    _cur = getattr(cfg, "curriculum", None)
    for _nm in list(vars(_cur).keys()) if _cur is not None else []:
        _t = getattr(_cur, _nm, None)
        if _t is not None and any(
            v in _gone for v in (getattr(_t, "params", None) or {}).values() if isinstance(v, str)
        ):
            setattr(_cur, _nm, None)
            report["dropped"].append(f"curriculum.{_nm}")
    return report


def _idle_oracle(cfg, key, prim, tag):
    """Nothing may still name the idle object — scene fields, term params, or a term's default binding."""
    root = prim.split("/")[1]  # "ObjectLeft" | "ObjectRight" — any prim path under it counts

    def _mentions(_x, _depth=0):
        if _depth > 6:
            return False
        if isinstance(_x, str):
            return _x == key or root in _x
        if isinstance(_x, dict):
            return any(_mentions(v, _depth + 1) for v in _x.values())
        if isinstance(_x, (list, tuple)):
            return any(_mentions(v, _depth + 1) for v in _x)
        return getattr(_x, "name", None) == key

    left_over = [
        f"scene.{_nm}"
        for _nm in list(vars(cfg.scene).keys())
        if getattr(cfg.scene, _nm, None) is not None
        and (
            _mentions(getattr(getattr(cfg.scene, _nm), "prim_path", None))
            or _mentions(getattr(getattr(cfg.scene, _nm), "filter_prim_paths_expr", None))
        )
    ]
    for _gname, _grp in manager_groups(cfg):
        for _nm in list(vars(_grp).keys()):
            _term = getattr(_grp, _nm, None)
            if (
                _term is not None
                and hasattr(_term, "func")
                and (_mentions(getattr(_term, "params", None)) or _term_object_binding(_term) == key)
            ):
                left_over.append(f"{_gname}.{_nm}")
    if left_over:
        raise ValueError(
            f"[{tag}] IDLE SIDE: entries still bound to the idle object {key} after the sweep: {left_over}"
        )


def apply_idle_side(cfg, args_cli, tag):
    """D16 IDLE SIDE (bimanual): the side the run's clips declare inactive has NO object — no prim, no object reward,
    no object termination, a zero object observation. Default (``args_cli.idle_side`` empty) = return before touching
    anything = the two-object bimanual build, byte-identical.

    Set by rl/cli.py --sequence_dir from each tree (qpos_layout.Layout.idle_sides — a run-level property like the
    shared object; mixed patterns die at the entry). The idle hand itself stays: its reference is the rest pose the
    tree and the idle-rest IK cache carry (fingers at the hand's rest pose hand_surface.rest_finger_q, arm at the rest
    IK), and its hand-tracking rewards and terminations are kept — that is what holds it at rest. "Judged on the
    active arm only" (D16) follows: nothing of the idle side can end an episode except a hand leaving its rest pose
    (velocity / fingertip gates).

    The entry gave the idle side the ACTIVE side's object as a placeholder so every registration path above ran
    unchanged; this removes that prim, the five object-filtered fingertip sensors, every term bound to it
    (``_term_object_binding``), a curriculum term naming a dropped termination, and zeroes its ObjectStateV2 under the
    same name (width 87, +3 under the A-5 fixed layout). Then ``cfg.idle_side`` is the statement runtime code reads
    (dexengine.rl.tasks.scene.collision_bodies.scene_object_key) and ``cfg.trajectory.idle_side`` the one the loader
    checks every clip against. MUST run at the very end of train_cfg_overrides / _eval_cfg_overrides (same contract as
    _apply_shared_object — a term registered after it would bind the removed prim)."""
    idle = str(getattr(args_cli, "idle_side", "") or "")
    if not idle:
        return
    if idle not in ("right", "left"):
        raise ValueError(f"[{tag}] idle_side={idle!r} — right|left only")
    if not getattr(args_cli, "bimanual", 0):
        raise ValueError(f"[{tag}] idle_side={idle} requires --bimanual 1 (a uni build has one object already)")
    if int(getattr(args_cli, "shared_object", 0) or 0) or getattr(cfg, "shared_object", False):
        raise ValueError(f"[{tag}] idle_side={idle} with a shared object — the entry forbids it (contradiction)")
    key, prim = f"object_{idle}", f"{{ENV_REGEX_NS}}/Object{idle.capitalize()}/base_link"
    from dexengine.rl.tasks.mdp.observations.object_state_v2 import FIXED_TAIL_DIMS, OBJECT_STATE_DIMS

    width = OBJECT_STATE_DIMS + (FIXED_TAIL_DIMS if int(getattr(args_cli, "obj_obs_fixed", 0) or 0) else 0)
    scene_dropped = _idle_drop_scene(cfg, key, prim)
    report = _idle_sweep_terms(cfg, key, width)
    cfg.idle_side = idle
    if getattr(cfg, "trajectory", None) is not None:
        cfg.trajectory.idle_side = idle
    _idle_oracle(cfg, key, prim, tag)
    print(
        f"[{tag}] [IDLE_SIDE] applied: side={idle} · scene removed {scene_dropped} · "
        f"terms dropped {len(report['dropped'])} {report['dropped']} · obs zeroed(width {width}) {report['zeroed']}",
        flush=True,
    )


def apply_obs_v2(cfg, args_cli, tag):
    """Wire the launch-time mesh_dir params into the V2-native obs cfg.

    The obs terms themselves are declared V2-natively in the cfg factory
    (cfg/task.py RightHandTaskObservationsCfg; the V1 449 layout and the
    runtime term swap are gone). This helper only injects the
    CLI-owned mesh paths.

    Per-hand 207 dims, in DECLARATION order (= concat order, uni/left example):
        <side>_hand        96   [  0: 96)
        last_base_action   12   [ 96:108)   reference base action
        actions            12   [108:120)   previous policy action
        object_state       87   [120:207)   kin 15 | ref 18 | surface 35 | priv 19
    (An earlier version of this docstring listed "96 + 12 + 87 + 12", i.e. object_state
    before actions. That order is WRONG and cost an hour of misattributing a diagnostic
    dim. The authority is observation_manager.active_terms['policy'] /
    group_obs_term_dim['policy']; the EVAL FINGERPRINT prints the resolved spans.)

    MUST be called from BOTH train_cfg_overrides and _eval_cfg_overrides — a
    train-only obs override silently breaks standalone eval (the --arm_kp lesson).
    """
    if not getattr(args_cli, "reference_base", 0):
        raise ValueError("the V2 obs layout requires --reference_base 1 (single-stage recipe)")
    pol = cfg.observations.policy

    mesh_dir = getattr(args_cli, "object_mesh_dir", None)  # active side (comma-safe multi)
    if not mesh_dir:
        raise ValueError(
            "[obs] the active-side --object_mesh_dir_* is required (surface_points.npz feeds object_state)"
        )
    mesh_dir = abspath_multi(mesh_dir)
    pol.object_state.params["mesh_dir"] = mesh_dir
    # A-5: the fixed 87+3 object layout is a launch-time switch — the canonical launcher passes the table value
    # (OBJ_OBS_FIXED — its value lives there); argparse default 0 = the 87/89 layout, used only by a bare call.
    _fixed = int(getattr(args_cli, "obj_obs_fixed", 0) or 0)
    if _fixed:
        pol.object_state.params["fixed_layout"] = 1

    if bool(getattr(args_cli, "bimanual", 0)):
        mesh_dir_l = getattr(args_cli, "object_mesh_dir_left", None)
        if not mesh_dir_l:
            raise ValueError("--bimanual 1 requires --object_mesh_dir_left (left surface_points.npz)")
        # The left side is comma-safe too (abspath_multi): per-clip left meshes in multi-object bimanual runs.
        mesh_dir_l = abspath_multi(mesh_dir_l)
        pol.object_state_left.params["mesh_dir"] = mesh_dir_l
        if _fixed:
            pol.object_state_left.params["fixed_layout"] = 1
    print(
        f"[{tag}] OBS_V2 (native): object_state mesh_dir={mesh_dir}"
        + (f", left={mesh_dir_l}" if getattr(args_cli, "bimanual", 0) else "")
        + (" · [OBJ_OBS_FIXED] fixed object observation 87+3 (θ · ref−θ · valid bit, rigid 0)" if _fixed else "")
    )


def perhand_obs_slices(raw_env, per_hand_act_dim, verbose=True):
    """Partition the bimanual policy-obs indices into (right_idx, left_idx) for the two towers.

    Rules by obs term name (group 'policy'), in declaration order:
      - name startswith 'left_' or endswith '_left'  -> LEFT tower
      - name == 'actions' (both-hand prev action)    -> SPLIT first per_hand_act_dim -> right, rest -> left
      - everything else (right hand/obj/last_base_action, object_surface_geometry) -> RIGHT tower
    Covers [0, obs_dim) exactly once. Fails loud on any gap/overlap/unexpected layout.
    """
    om = raw_env.unwrapped.observation_manager
    names = om.active_terms["policy"]
    dims = om.group_obs_term_dim["policy"]
    right_idx, left_idx, assign = [], [], []
    cursor = 0
    for name, shp in zip(names, dims, strict=True):
        n = int(shp[0]) if isinstance(shp, (tuple, list)) else int(shp)
        rng = list(range(cursor, cursor + n))
        if name.startswith("left_") or name.endswith("_left"):
            left_idx += rng
            assign.append((name, n, "L"))
        elif name == "actions":
            assert n == 2 * per_hand_act_dim, f"'actions' dim {n} != 2*{per_hand_act_dim}"
            right_idx += rng[:per_hand_act_dim]
            left_idx += rng[per_hand_act_dim:]
            assign.append((name, n, "R+L split"))
        else:
            right_idx += rng
            assign.append((name, n, "R"))
        cursor += n
    total = om.group_obs_dim["policy"]
    total = int(total[0]) if isinstance(total, (tuple, list)) else int(total)
    assert cursor == total, f"obs term dims sum {cursor} != group_obs_dim {total}"
    assert len(right_idx) + len(left_idx) == total, "partition does not cover all dims"
    assert len(set(right_idx) & set(left_idx)) == 0, "right/left index overlap"
    if verbose:
        print(f"  [PERHAND] obs partition: right={len(right_idx)} left={len(left_idx)} total={total}")
        for nm, n, side in assign:
            print(f"  [PERHAND]   {nm:26s} dim={n:3d} -> {side}")
    return right_idx, left_idx
