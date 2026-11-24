"""Reference-base audit: how well the PD-tracked reference (zero residual) follows the clip."""

from __future__ import annotations

import torch

from dexengine.rl.env_setup.refbase_report import report_tracking_bands
from dexengine.rl.env_setup.robot_gains import arm_joint_expr
from dexengine.rl.env_setup.sides import FINGERTIP_BODY_INDICES


def refbase_pd_audit(raw_env, args_cli):
    import os

    import numpy as _np

    import isaaclab.utils.math as _mu

    from dexengine.rl.tasks.mdp.rewards.contact_base import OBJ_SENSOR_NAMES

    # SIDE: resolved from the live action terms. It used to be hardcoded "right",
    # so this audit ABORTED on every LEFT-uni run ("residual action term missing").
    _terms = raw_env.action_manager._terms
    side = "left" if ("left_hand_joint_pos" in _terms and "right_hand_joint_pos" not in _terms) else "right"
    scene = raw_env.scene
    device = raw_env.device
    hand = scene[f"{side}_hand"] if f"{side}_hand" in scene.keys() else scene["robot"]  # noqa: SIM118  — InteractiveScene has no __contains__
    tm = raw_env.trajectory_manager

    term = _terms.get(f"{side}_hand_joint_pos")
    return _audit_reference_base_pd(
        raw_env, args_cli, OBJ_SENSOR_NAMES, _mu, _np, _terms, device, hand, os, scene, side, term, tm
    )


def _audit_reference_base_pd(
    raw_env, args_cli, OBJ_SENSOR_NAMES, _mu, _np, _terms, device, hand, os, scene, side, term, tm
):
    """Read the reference-base PD gains of the arm term, then roll out the tracking audit."""
    if term is None or not hasattr(term, "_inner"):
        print("[refbase_pd_audit] ABORT — residual action term missing / no _inner")
        return
    inner = term._inner
    if not getattr(term, "_reference_base", False):
        print("[refbase_pd_audit] ABORT — term._reference_base is False (pass --reference_base 1)")
        return
    fids = inner._finger_joint_ids
    fnames = list(inner._finger_joint_names)
    rh_ids = getattr(tm, f"_{side}_hand_ids")
    rh_ids_list = rh_ids.tolist() if torch.is_tensor(rh_ids) else list(rh_ids)
    fih = [rh_ids_list.index(int(x)) for x in fids]  # finger positions within right_hand_qpos (PhysX)

    body_ids_t = torch.tensor(getattr(tm, f"{side}_body_ids_isaac"), device=device, dtype=torch.long)
    tip_names = ["thumb", "index", "middle", "ring", "pinky"]
    tip_body_idx = [FINGERTIP_BODY_INDICES[n] for n in tip_names]  # indices into the 13-body ref array

    sn = OBJ_SENSOR_NAMES[side]
    obj_sensors = [scene[n] for n in sn if n in scene.keys()]  # noqa: SIM118  — InteractiveScene has no __contains__

    def objforce():  # (N,5) per-fingertip object-filtered force magnitude
        cols = []
        for s in obj_sensors:
            fm = s.data.force_matrix_w
            cols.append(torch.zeros(raw_env.num_envs, device=device) if fm is None else fm.sum(dim=(1, 2)).norm(dim=-1))
        return torch.stack(cols, dim=1) if cols else torch.zeros((raw_env.num_envs, 5), device=device)

    _bimanual, arm_kd_used, arm_kp_used, hand_l, inner_l = _audit_left_wrist_tracking(
        raw_env, _terms, device, fih, fnames, obj_sensors, scene, side, sn, term, tm
    )
    try:
        _aid, _ = inner._asset.find_joints([arm_joint_expr()(side)])
        _aidt = torch.tensor(_aid, device=device, dtype=torch.long)
        arm_kp_used = float(inner._asset.data.joint_stiffness[0, _aidt].mean().item())
        arm_kd_used = float(inner._asset.data.joint_damping[0, _aidt].mean().item())
        print(f"  ARM PD gains (effective): Kp={arm_kp_used:.0f} Kd={arm_kd_used:.1f}")
    except (AttributeError, IndexError, RuntimeError, ValueError) as _exc:
        print(f"  [REFBASE_PD_AUDIT] arm PD gains not read: {_exc!r}")

    T = int(tm.clip_lengths[0].item())
    slowdown = float(getattr(tm, "_slowdown_factor", getattr(args_cli, "slowdown_factor", 1.0) or 1.0))
    start0 = int(tm.start_frames[0])
    act_dim = raw_env.action_manager.total_action_dim
    zero_act = torch.zeros((raw_env.num_envs, act_dim), device=device)
    _rollout_wrist_tracking(
        raw_env,
        T,
        _bimanual,
        _mu,
        _np,
        act_dim,
        arm_kd_used,
        arm_kp_used,
        body_ids_t,
        fids,
        fih,
        fnames,
        hand,
        hand_l,
        inner,
        inner_l,
        obj_sensors,
        objforce,
        os,
        side,
        slowdown,
        start0,
        tip_body_idx,
        tip_names,
        tm,
        zero_act,
    )


def _audit_left_wrist_tracking(raw_env, _terms, device, fih, fnames, obj_sensors, scene, side, sn, term, tm):
    """Bimanual: audit the left hand's wrist tracking (target vs current)."""
    # Bimanual: also audit the LEFT hand's wrist tracking (target-vs-current pos/rot).
    # Uses the LEFT action term's inner (its _r_offset_fwd_wxyz is now side-aware Rz(+90)
    # after the mount-rotation fix) + get_wrist_pos/quat_w(side='left').
    # the extra LEFT block is only meaningful when the PRIMARY audited side is right
    _bimanual = ("left_hand_joint_pos" in _terms) and (side == "right")
    inner_l = hand_l = None
    if _bimanual:
        hand_l = scene["left_hand"] if "left_hand" in scene.keys() else scene["robot"]  # noqa: SIM118  — InteractiveScene has no __contains__
        term_l = raw_env.action_manager._terms.get("left_hand_joint_pos")
        inner_l = getattr(term_l, "_inner", None)
        print(
            f"  BIMANUAL: left wrist audit ENABLED (term=left_hand_joint_pos, "
            f"left ee_body_idx={getattr(inner_l, '_ee_body_idx', None)}, "
            f"left r_offset_fwd={getattr(inner_l, '_r_offset_fwd_wxyz', None)})"
        )

    # Neutralise ALL termination so a single episode plays the whole clip (no auto-reset
    # confound). Override compute() to always return no-termination, leaving term funcs +
    # their .reset() intact (class-based terms need .reset on env reset). The fingertip-
    # tracking divergence is computed/reported offline instead.
    import types as _types

    tmgr = raw_env.termination_manager
    n_terms = len(tmgr._term_cfgs)

    def _compute_none(self):
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        self._term_dones[:] = False
        return self._truncated_buf | self._terminated_buf

    tmgr.compute = _types.MethodType(_compute_none, tmgr)

    print("\n" + "=" * 78)
    print("REFBASE PD AUDIT (DEXENGINE_REFBASE_PD_AUDIT=1) — PD reference base, residual=0")
    print(
        f"  reference_base={getattr(term, '_reference_base', None)}  "
        f"use_traj_ref={getattr(term, '_use_traj_ref', None)}  zero_residual={getattr(term, '_zero_residual', None)}"
    )
    print(f"  finger joints (articulation order) = {fnames}")
    print(f"  finger positions within {side}_hand_qpos (PhysX) = {fih}")
    print(f"  object-filtered fingertip sensors present: {len(obj_sensors)}/{len(sn)}")
    print(f"  neutralised {n_terms} termination terms for full-clip replay")

    raw_env.reset()
    # Force a clean frame-0 start (no RSI) so the replay deterministically covers the full clip:
    # pin start_frames=0, re-seat hand/arm/object at frame 0, and zero episode_length_buf.
    from dexengine.rl.tasks.mdp.terms import reset_arm_to_trajectory as _r_arm
    from dexengine.rl.tasks.mdp.terms import reset_hands_to_trajectory as _r_hands
    from dexengine.rl.tasks.mdp.terms import reset_object_to_trajectory as _r_obj

    eids = torch.arange(raw_env.num_envs, device=device)
    tm.start_frames[:] = 0
    # The object reset defaults to SceneEntityCfg("object_right"); a LEFT-uni scene only has
    # `object_left`, so pass this side's object explicitly (KeyError otherwise).
    from isaaclab.managers import SceneEntityCfg as _SEC_aud

    _r_hands(raw_env, eids)
    _r_arm(raw_env, eids)
    _r_obj(raw_env, eids, object_cfg=_SEC_aud(f"object_{side}" if side == "left" else "object_right"), side=side)
    raw_env.episode_length_buf[:] = 0

    # Report the EFFECTIVE arm PD gains. The DEXENGINE_ARM_KP/KD override (if any) is applied at the
    # run_eval level via apply_arm_gain_override so it ALSO covers the render path; read-only here.
    arm_kp_used = arm_kd_used = None
    return _bimanual, arm_kd_used, arm_kp_used, hand_l, inner_l


def _rollout_wrist_tracking(
    raw_env,
    T,
    _bimanual,
    _mu,
    _np,
    act_dim,
    arm_kd_used,
    arm_kp_used,
    body_ids_t,
    fids,
    fih,
    fnames,
    hand,
    hand_l,
    inner,
    inner_l,
    obj_sensors,
    objforce,
    os,
    side,
    slowdown,
    start0,
    tip_body_idx,
    tip_names,
    tm,
    zero_act,
):
    """Roll the reference-base policy out and collect wrist / finger tracking per step (both sides)."""
    F, fte, q_ref_arr, q_robot_arr, qe, we, we_l, wre, wre_l = _rollout_tracking_errors(
        raw_env,
        T,
        _bimanual,
        _mu,
        _np,
        act_dim,
        body_ids_t,
        fids,
        fih,
        hand,
        hand_l,
        inner,
        inner_l,
        obj_sensors,
        objforce,
        side,
        slowdown,
        start0,
        tip_body_idx,
        tm,
        zero_act,
    )

    report_tracking_bands(
        raw_env,
        T,
        _bimanual,
        _np,
        arm_kd_used,
        arm_kp_used,
        fnames,
        os,
        slowdown,
        tip_names,
        F,
        fte,
        q_ref_arr,
        q_robot_arr,
        qe,
        we,
        we_l,
        wre,
        wre_l,
    )


def _rollout_tracking_errors(
    raw_env,
    T,
    _bimanual,
    _mu,
    _np,
    act_dim,
    body_ids_t,
    fids,
    fih,
    hand,
    hand_l,
    inner,
    inner_l,
    obj_sensors,
    objforce,
    side,
    slowdown,
    start0,
    tip_body_idx,
    tm,
    zero_act,
):
    """Allocate the per-frame error arrays and step the reference-base rollout over them."""
    N_steps = int(_np.ceil(T * slowdown)) + 4
    print(f"  T={T} frames  slowdown={slowdown}  start_frame(env0)={start0}  act_dim={act_dim}  N_steps={N_steps}")

    we = _np.full(T, _np.nan)  # wrist pos err (m)
    wre = _np.full(T, _np.nan)  # wrist rot err (deg)
    qe = _np.full((T, 6), _np.nan)  # per-finger qpos err (rad)
    fte = _np.full((T, 5), _np.nan)  # per-fingertip pos err (m)
    F, q_ref_arr, q_robot_arr, we_l, wre_l = _step_rollout(
        raw_env,
        T,
        _bimanual,
        _mu,
        _np,
        body_ids_t,
        fids,
        fih,
        hand,
        hand_l,
        inner,
        inner_l,
        obj_sensors,
        objforce,
        side,
        slowdown,
        start0,
        tip_body_idx,
        tm,
        zero_act,
        N_steps,
        fte,
        qe,
        we,
        wre,
    )
    return F, fte, q_ref_arr, q_robot_arr, qe, we, we_l, wre, wre_l


def _step_rollout(
    raw_env,
    T,
    _bimanual,
    _mu,
    _np,
    body_ids_t,
    fids,
    fih,
    hand,
    hand_l,
    inner,
    inner_l,
    obj_sensors,
    objforce,
    side,
    slowdown,
    start0,
    tip_body_idx,
    tm,
    zero_act,
    N_steps,
    fte,
    qe,
    we,
    wre,
):
    """Step the env with zero residual and record each frame's wrist, finger and force errors."""
    F = _np.full((T, 5), _np.nan)  # per-fingertip object force (N)
    q_robot_arr = _np.full((T, 6), _np.nan)
    q_ref_arr = _np.full((T, 6), _np.nan)
    we_l = _np.full(T, _np.nan)  # LEFT wrist pos err (m)
    wre_l = _np.full(T, _np.nan)  # LEFT wrist rot err (deg)

    for _step in range(N_steps):
        elen_t = raw_env.episode_length_buf.clone()  # frame whose ref drives the PD target this step
        raw_env.step(zero_act)  # residual = zero action -> PD reference base
        f_eff = int((elen_t[0].float() / slowdown).floor().item()) + start0
        if f_eff < 0 or f_eff >= T:
            continue
        _record_tracking_step(
            raw_env,
            _bimanual,
            _mu,
            body_ids_t,
            fids,
            fih,
            hand,
            hand_l,
            inner,
            inner_l,
            obj_sensors,
            objforce,
            side,
            tip_body_idx,
            tm,
            F,
            elen_t,
            f_eff,
            fte,
            q_ref_arr,
            q_robot_arr,
            qe,
            we,
            we_l,
            wre,
            wre_l,
        )
    return F, q_ref_arr, q_robot_arr, we_l, wre_l


def _record_tracking_step(
    raw_env,
    _bimanual,
    _mu,
    body_ids_t,
    fids,
    fih,
    hand,
    hand_l,
    inner,
    inner_l,
    obj_sensors,
    objforce,
    side,
    tip_body_idx,
    tm,
    F,
    elen_t,
    f_eff,
    fte,
    q_ref_arr,
    q_robot_arr,
    qe,
    we,
    we_l,
    wre,
    wre_l,
):
    """Record one step's wrist (link7 to virtual wrist, env-local) and fingertip tracking errors."""
    # --- wrist (robot link7 -> virtual wrist, env-local) vs reference wrist ---
    link7_pos = hand.data.body_pos_w[:, inner._ee_body_idx]
    link7_quat = hand.data.body_quat_w[:, inner._ee_body_idx]
    r_off = inner._r_offset_fwd_wxyz.unsqueeze(0).expand(link7_pos.shape[0], -1)
    wq = _mu.quat_mul(link7_quat, r_off)
    l7r = _mu.matrix_from_quat(wq)
    t_col = inner._link7_to_wrist_trans.view(1, 3, 1).expand(link7_pos.shape[0], -1, -1)
    wp = link7_pos + torch.bmm(l7r, t_col).squeeze(-1)
    wp_local = (wp - raw_env.scene.env_origins)[0]
    ref_wp = tm.get_wrist_pos_w(elen_t, side=side)[0]
    ref_wq = tm.get_wrist_quat_w(elen_t, side=side)
    we[f_eff] = float((wp_local - ref_wp).norm().item())
    wre[f_eff] = float(_mu.quat_error_magnitude(wq[:1], ref_wq[:1])[0].item() * 180.0 / 3.14159265)
    # --- LEFT wrist (same construction, side-aware reference + Rz(+90) mount) ---
    if _bimanual and inner_l is not None:
        l7p_l = hand_l.data.body_pos_w[:, inner_l._ee_body_idx]
        l7q_l = hand_l.data.body_quat_w[:, inner_l._ee_body_idx]
        r_off_l = inner_l._r_offset_fwd_wxyz.unsqueeze(0).expand(l7p_l.shape[0], -1)
        wq_l = _mu.quat_mul(l7q_l, r_off_l)
        t_col_l = inner_l._link7_to_wrist_trans.view(1, 3, 1).expand(l7p_l.shape[0], -1, -1)
        wp_l = l7p_l + torch.bmm(_mu.matrix_from_quat(wq_l), t_col_l).squeeze(-1)
        wp_local_l = (wp_l - raw_env.scene.env_origins)[0]
        ref_wp_l = tm.get_wrist_pos_w(elen_t, side="left")[0]
        ref_wq_l = tm.get_wrist_quat_w(elen_t, side="left")
        we_l[f_eff] = float((wp_local_l - ref_wp_l).norm().item())
        wre_l[f_eff] = float(_mu.quat_error_magnitude(wq_l[:1], ref_wq_l[:1])[0].item() * 180.0 / 3.14159265)
    # --- finger qpos (robot vs reference) ---
    ref = tm.get_reference_state(elen_t)
    q_robot = hand.data.joint_pos[0, fids]
    q_ref = ref[f"{side}_hand_qpos"][0, fih]
    qe[f_eff] = (q_robot - q_ref).abs().detach().cpu().numpy()
    q_robot_arr[f_eff] = q_robot.detach().cpu().numpy()
    q_ref_arr[f_eff] = q_ref.detach().cpu().numpy()
    # --- fingertip world->env-local vs reference body pos ---
    ref_body = tm.get_reference_body_pos(elen_t)[f"{side}_body_pos"][0]  # (13,3) env-local
    cur_body = hand.data.body_link_pos_w[0, body_ids_t] - raw_env.scene.env_origins[0]
    fte[f_eff] = (cur_body[tip_body_idx] - ref_body[tip_body_idx]).norm(dim=-1).detach().cpu().numpy()
    # --- object-filtered force ---
    if obj_sensors:
        F[f_eff] = objforce()[0].detach().cpu().numpy()
