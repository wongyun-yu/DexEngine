"""Boot smoke of the training env (``DEXENGINE_REFBASE_SMOKE=1``): reference base, wrist, sensors, then hard exit."""

import os
import sys


def _refbase_smoke(raw_env, args_cli, *, _residual_act_name, _active_side):
    """Verify-before-train boot check (env-var-gated DEXENGINE_REFBASE_SMOKE=1).

    Confirms, before any long training: (1) residual=0 finger PD target == post-reset
    reference finger joints PER JOINT (resolving get_reference_state vs
    get_reference_state_for_envs as the q_ref source), (2) wrist == kinematic reference,
    (3) kinematic trajectory source, (4) object-filtered sensors present + populate,
    (5) no PhysX crash. Hard-exits with code 0 (PASS) / 1 (FAIL).
    """
    import numpy as _np
    import torch

    scene = raw_env.scene
    # LEFT-UNI: the smoke generalizes by side — term name, reference qpos key,
    # hand-id list, dataset-order source, wrist query and object-filtered
    # sensor names all resolve through _side (default 'right').
    _side = "left" if getattr(args_cli, "left_uni", 0) else "right"
    _act_nm = _residual_act_name()
    # Idle right side (bimanual): the right hand has no object — the smoke verifies the ACTIVE (left) side instead.
    if str(getattr(args_cli, "idle_side", "") or "") == "right":
        _side, _act_nm = "left", "left_hand_joint_pos"
    term = raw_env.action_manager._terms.get(_act_nm)
    print("\n" + "=" * 78)
    print(f"REFBASE SMOKE (DEXENGINE_REFBASE_SMOKE=1) — verify-before-train (side={_side})")
    if term is None or not hasattr(term, "_inner"):
        print(f"  FAIL: residual action term '{_act_nm}' missing / no _inner")
        sys.stdout.flush()
        os._exit(1)
    inner = term._inner
    fids = inner._finger_joint_ids
    lower = inner._finger_lower.detach().cpu().numpy()
    rng = inner._finger_range.detach().cpu().numpy()
    fnames = list(inner._finger_joint_names)
    print(
        f"  reference_base={getattr(term, '_reference_base', None)}  "
        f"use_traj_ref={getattr(term, '_use_traj_ref', None)}"
    )
    print(f"  finger joints (articulation/cfg order) = {fnames}")
    hand = scene[f"{_side}_hand"] if f"{_side}_hand" in scene.keys() else scene["robot"]  # noqa: SIM118  — InteractiveScene has no __contains__
    tm = raw_env.trajectory_manager
    dt = float(raw_env.physics_dt)
    OBJ_SENSOR_NAMES, _ra, _rh, _ro_s, _smk_pin, eids, fih, nenv, ok = _check_floating_reset(
        raw_env, _side, fids, scene, tm, torch
    )

    _smoke_frame_parity(
        raw_env,
        args_cli,
        _active_side,
        OBJ_SENSOR_NAMES,
        _np,
        _ra,
        _rh,
        _ro_s,
        _side,
        _smk_pin,
        dt,
        eids,
        fids,
        fih,
        hand,
        lower,
        nenv,
        ok,
        rng,
        scene,
        term,
        tm,
        torch,
    )


def _check_floating_reset(raw_env, _side, fids, scene, tm, torch):
    """Floating hands have no arm IK reset (reset_arm_to_trajectory would fail on its joint_expr default). The env cfg's
    reset_arm term decides which reset plays the arm's role (floating hands: reset_floating_hands_to_trajectory), so
    the smoke calls that term's function.
    """
    from dexengine.rl.tasks import hooks as _hooks
    from dexengine.rl.tasks.mdp.terms import reset_arm_to_trajectory as _ra
    from dexengine.rl.tasks.mdp.terms import reset_hands_to_trajectory as _rh
    from dexengine.rl.tasks.mdp.terms import reset_object_to_trajectory as _ro

    if _hooks.floating_active():
        _ra = raw_env.cfg.events.reset_arm.func
        # the floating-hand job logs select this line by its [FLOATING_HAND] marker and "REFBASE SMOKE"
        print(
            f"{_hooks.floating_hand().MARK} REFBASE SMOKE: arm-slot reset = env cfg reset_arm term {_ra.__name__} "
            "(not the arm IK reset)",
            flush=True,
        )
    from dexengine.rl.tasks.mdp.rewards.contact_base import OBJ_SENSOR_NAMES

    eids = torch.tensor([0], device=raw_env.device)
    raw_env.reset()
    nenv = raw_env.num_envs
    ok = True
    rh_ids = tm._right_hand_ids if _side == "right" else tm._left_hand_ids
    rh_ids_list = rh_ids.tolist() if torch.is_tensor(rh_ids) else list(rh_ids)
    fih = [rh_ids_list.index(int(x)) for x in fids]  # finger positions within {side}_hand_qpos (PhysX)
    print(f"  finger positions within {_side}_hand_qpos (PhysX) = {fih}")
    # Side-aware object probe: the L-uni build has object_left ONLY — the
    # default object_cfg ("object_right") KeyErrors there. Right-side builds keep the
    # default call path.
    from isaaclab.managers import SceneEntityCfg as _SEC_SMK

    _smk_obj = f"object_{_side}"
    if _smk_obj not in scene.keys():  # noqa: SIM118  — InteractiveScene has no __contains__
        _smk_obj = "object_right"
    # Shared-object evidence: which object entities the
    # scene actually built, and how many `Object*` prims stand under env_0 on the stage (shared -> exactly 1).
    try:
        _smk_objs = sorted(k for k in scene.rigid_objects)
        _smk_env0 = scene.env_prim_paths[0] if getattr(scene, "env_prim_paths", None) else "/World/envs/env_0"
        _smk_root = scene.stage.GetPrimAtPath(_smk_env0)
        _smk_prims = (
            sorted(c.GetName() for c in _smk_root.GetChildren() if c.GetName().startswith("Object"))
            if _smk_root and _smk_root.IsValid()
            else ["<env_0 prim invalid>"]
        )
        print(
            f"[OBJECTS][scene] rigid_objects={_smk_objs} env_0 Object* prims={_smk_prims} (n={len(_smk_prims)})",
            flush=True,
        )
    except Exception as _smk_e:  # noqa: BLE001 — evidence line only; never hide the probe behind it
        print(f"[OBJECTS][scene] probe failed: {type(_smk_e).__name__}: {_smk_e}", flush=True)

    def _ro_s(_e, _ids):
        _ro(_e, _ids, object_cfg=_SEC_SMK(_smk_obj), side=("left" if _smk_obj == "object_left" else "right"))

    def _smk_pin(_f):
        """Pin env 0 to frame f the way the real reset does.

        With ``per_side_clock`` (bimanual, reset_scheme=decoupled) ``get_reference_state`` reads ``_side_start``, not
        ``start_frames``, so both clocks are set, as the trajectory manager's reset does.
        """
        tm.start_frames[eids] = _f
        if getattr(tm, "per_side_clock", False):
            tm._side_start[eids] = _f
            tm._side_birth[eids] = 0

    return OBJ_SENSOR_NAMES, _ra, _rh, _ro_s, _smk_pin, eids, fih, nenv, ok


def _smoke_frame_parity(
    raw_env,
    args_cli,
    _active_side,
    OBJ_SENSOR_NAMES,
    _np,
    _ra,
    _rh,
    _ro_s,
    _side,
    _smk_pin,
    dt,
    eids,
    fids,
    fih,
    hand,
    lower,
    nenv,
    ok,
    rng,
    scene,
    term,
    tm,
    torch,
):
    """Frame parity of the reference finger targets at a few frames (the IMPLEMENTED mapping, actionable range)."""
    for f in [40, 53, 120]:
        _smk_pin(f)
        _rh(raw_env, eids)
        _ra(raw_env, eids)
        _ro_s(raw_env, eids)
        q_gt = hand.data.joint_pos[0, fids].detach().cpu().numpy()  # post-reset reference (ground truth)
        ok = _check_reference_mapping(raw_env, _np, _side, fih, lower, nenv, rng, tm, torch, f, q_gt, ok)
    # wrist reference + sensor populate (step once at a contact frame, re-pinned)
    _smk_pin(53)
    _rh(raw_env, eids)
    _ra(raw_env, eids)
    _ro_s(raw_env, eids)
    fr0 = torch.zeros(nenv, dtype=torch.long, device=raw_env.device)
    wpos = (tm.get_wrist_pos_w(fr0, side=_side)[0]).detach().cpu().numpy()
    print(f"\n  wrist reference (frame 53, use_traj_ref path, side={_side}) pos_w = {_np.round(wpos, 3).tolist()}")
    sn = OBJ_SENSOR_NAMES[_side]
    missing = [n for n in sn if n not in scene.keys()]  # noqa: SIM118  — InteractiveScene has no __contains__
    present = [n for n in sn if n in scene.keys()]  # noqa: SIM118  — InteractiveScene has no __contains__
    fm_ok = False
    if present:
        for _ in range(4):
            _rh(raw_env, eids)
            _ra(raw_env, eids)
            _ro_s(raw_env, eids)
            raw_env.sim.step(render=False)
            scene.update(dt)
        fm = scene[present[0]].data.force_matrix_w
        fm_ok = fm is not None
    print(
        f"  object-filtered fingertip sensors present: {len(present)}/{len(sn)} missing={missing}  "
        f"force_matrix_w populated={fm_ok}"
    )
    if missing or not fm_ok:
        ok = False
    _side_advantage_smoke(raw_env, args_cli, _active_side, ok, term)


def _check_reference_mapping(raw_env, _np, _side, fih, lower, nenv, rng, tm, torch, f, q_gt, ok):
    """IMPLEMENTED mapping: get_reference_state(episode_length_buf=0) + start_frames=f -> frame f; read right_hand_qpos
    (PhysX order) and index at the finger positions. Clear the per-step cache (keyed by frame_ids.sum()) so it re-
    evaluates frame f rather than returning a stale frame_ids=0 entry from a previous loop iteration.
    """
    tm._cache_state = None
    tm._cache_state_p1 = None
    frame_ids = torch.zeros(nenv, dtype=torch.long, device=raw_env.device)
    ref = tm.get_reference_state(frame_ids)
    q_ref = ref[f"{_side}_hand_qpos"][0, fih].detach().cpu().numpy()
    q_old = (
        (tm._last_rh_finger_dataset if _side == "right" else tm._last_lh_finger_dataset)[0].detach().cpu().numpy()
    )  # OLD dataset/MuJoCo-order source
    base_n = _np.clip(2.0 * (q_ref - lower) / rng - 1.0, -1.0, 1.0)
    fwd = 0.5 * (base_n + 1.0) * rng + lower  # implemented finger PD target (residual=0)
    # Parity is judged within the ACTIONABLE range: the PD target is limit-clamped by
    # the normalize round-trip, while the state write can carry a slightly out-of-limit
    # reference (e.g. LEFT thumb_1 ref -0.003 rad < lower 0.0, retarget noise). Comparing
    # against the clamped ground truth keeps the gate strict for permutation bugs
    # (~0.5 rad) without failing on millirad limit clips.
    q_gt_cl = _np.clip(q_gt, lower, lower + rng)
    e = float(_np.abs(fwd - q_gt_cl).max())
    e_old = float(_np.abs(q_old - q_gt).max())
    _clip_note = "" if _np.allclose(q_gt_cl, q_gt) else "  (q_gt limit-clipped for parity)"
    print(f"\n  frame {f}: q_gt(post-reset) = {_np.round(q_gt, 3).tolist()}{_clip_note}")
    print(
        f"    implemented q_ref (PhysX) = {_np.round(q_ref, 3).tolist()}  PD target = "
        f"{_np.round(fwd, 3).tolist()}  "
        f"|PD-q_gt|max={e:.5f}  {'PASS' if e < 1e-3 else 'FAIL'}"
    )
    print(f"    (old dataset-order source |q_old-q_gt|max={e_old:.4f} — confirms the permutation bug)")
    if e >= 1e-3:
        ok = False
    return ok


def _side_advantage_smoke(raw_env, args_cli, _active_side, ok, term):
    """PER-SIDE ADVANTAGE smoke: compute one reward step and verify the split reproduces the total (reward_right +
    reward_left == RewardManager total).
    """
    if int(getattr(args_cli, "per_side_advantage", 0)) == 1:
        try:
            _psa_total = raw_env.reward_manager.compute(dt=raw_env.step_dt)
            _psa_r, _psa_l = raw_env.compute_per_side_rewards()
            _psa_err = float((_psa_r + _psa_l - _psa_total).abs().max().item())
            _psa_mag = float(_psa_total.abs().max().item())
            _psa_ok = _psa_err <= 1e-4 + 1e-4 * _psa_mag
            print(
                f"\n  PER-SIDE ADVANTAGE split: mean r_R={_psa_r.mean().item():.4f} "
                f"r_L={_psa_l.mean().item():.4f} total={_psa_total.mean().item():.4f} | "
                f"max|r_R+r_L-total|={_psa_err:.3e} (|total|max={_psa_mag:.3f})  "
                f"{'PASS' if _psa_ok else 'FAIL'}"
            )
            if not _psa_ok:
                ok = False
        except Exception as _psa_exc:  # noqa: BLE001 — loud: split must be checkable pre-train
            print(f"\n  PER-SIDE ADVANTAGE split: FAILED TO RUN ({_psa_exc})")
            ok = False
    if not getattr(term, "_reference_base", False):
        print("  WARN: term._reference_base is False (CLI flag not applied)")
        ok = False
    if not getattr(term, "_use_traj_ref", False):
        print("  WARN: term._use_traj_ref is False")
        ok = False
    # --multi_object: verify env_i -> asset alignment (the invariant everything rests
    # on): the composed USD reference of /World/envs/env_i/Object must be usds[i % n]
    # (the same loader-ordered list the trajectory manager cycles for clip i % n).
    if getattr(args_cli, "multi_object", 0):
        try:
            from pxr import Usd

            _env0 = getattr(raw_env, "unwrapped", raw_env)
            stage = _env0.scene.stage
            _usds = [u for u in args_cli.object_usd.split(",") if u]
            _n = len(_usds)
            for i in range(min(_env0.num_envs, 2 * _n)):
                prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Object{_active_side.capitalize()}")
                q = Usd.PrimCompositionQuery.GetDirectReferences(prim)
                refs = [a.GetTargetLayer().realPath for a in q.GetCompositionArcs() if a.GetTargetLayer() is not None]
                want = os.path.abspath(_usds[i % _n])
                aligned = any(os.path.abspath(r) == want for r in refs)
                print(
                    f"  multi-object env_{i}: want obj "
                    f"{os.path.basename(os.path.dirname(want))} refs="
                    f"{[os.path.basename(os.path.dirname(r)) for r in refs]} "
                    f"{'OK' if aligned else 'MISMATCH'}"
                )
                if not aligned:
                    ok = False
        except Exception as e:  # noqa: BLE001 — check must not mask the base smoke verdict
            print(f"  WARN: multi-object alignment check failed to run: {e}")
    print(f"\n  ==== REFBASE SMOKE {'PASS' if ok else 'FAIL'} ====")
    print("=" * 78)
    sys.stdout.flush()
    os._exit(0 if ok else 1)
