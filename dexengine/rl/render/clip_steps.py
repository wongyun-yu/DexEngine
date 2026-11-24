"""One policy episode, step by step: action, env step, cumulative reward sums, the fingertip overlay and the frame
capture.
"""

from __future__ import annotations

import numpy as _np
import torch

# Hand-only envs have no obj_pos / obj_rot reward terms; their rank metric falls back to these five fingertip-tracking
# terms (fingertip_quality = their sum).
_FT_TERMS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")


def _grab_frame(cam):
    """RGB frame of env 0 as HxWx3 uint8; None when the view is disabled (cam None) or has no rgb output."""
    if cam is None:
        return None
    rgb = cam.data.output.get("rgb")
    if rgb is None:
        return None
    fr = rgb[0].detach().cpu().numpy()
    if fr.dtype != _np.uint8:
        fr = (fr * 255).clip(0, 255).astype(_np.uint8)
    if fr.shape[-1] == 4:
        fr = fr[..., :3]
    return fr


def _rollout_steps(
    wrapped,
    raw_env,
    traj_mgr,
    policy_fn,
    max_episode_steps,
    _front_only,
    _AUDIT,
    _ar_log,
    _extract_logged_cum,
    _ft_local,
    _ft_marker,
    _ft_root_pos,
    _ft_root_q,
    _ft_slowdown,
    _read_cum,
    ego_cam,
    ego_fs,
    final_ft_cum,
    final_pos_cum,
    final_rot_cum,
    front_cam,
    front_fs,
    have_fingertips,
    have_obj_pos,
    have_obj_rot,
    info,
    obs,
    pre_step_ft,
    pre_step_pos,
    pre_step_rot,
    side_cam,
    side_fs,
    sidef_cam,
    sidef_fs,
    term_b,
    trunc_b,
):
    """Step the pinned episode: record frames and overlays, the cumulative tracking rewards and the contact audit."""
    return _rollout_episode(
        wrapped,
        raw_env,
        traj_mgr,
        policy_fn,
        max_episode_steps,
        _front_only,
        _AUDIT,
        _ar_log,
        _extract_logged_cum,
        _ft_local,
        _ft_marker,
        _ft_root_pos,
        _ft_root_q,
        _ft_slowdown,
        _read_cum,
        ego_cam,
        ego_fs,
        final_ft_cum,
        final_pos_cum,
        final_rot_cum,
        front_cam,
        front_fs,
        have_fingertips,
        have_obj_pos,
        have_obj_rot,
        info,
        obs,
        pre_step_ft,
        pre_step_pos,
        pre_step_rot,
        side_cam,
        side_fs,
        sidef_cam,
        sidef_fs,
        term_b,
        trunc_b,
    )


def _rollout_episode(
    wrapped,
    raw_env,
    traj_mgr,
    policy_fn,
    max_episode_steps,
    _front_only,
    _AUDIT,
    _ar_log,
    _extract_logged_cum,
    _ft_local,
    _ft_marker,
    _ft_root_pos,
    _ft_root_q,
    _ft_slowdown,
    _read_cum,
    ego_cam,
    ego_fs,
    final_ft_cum,
    final_pos_cum,
    final_rot_cum,
    front_cam,
    front_fs,
    have_fingertips,
    have_obj_pos,
    have_obj_rot,
    info,
    obs,
    pre_step_ft,
    pre_step_pos,
    pre_step_rot,
    side_cam,
    side_fs,
    sidef_cam,
    sidef_fs,
    term_b,
    trunc_b,
):
    """Step the policy through one episode, collecting frames and cumulative tracking terms."""
    for _ in range(max_episode_steps):
        _place_fingertip_marker(raw_env, traj_mgr, _ft_local, _ft_marker, _ft_root_pos, _ft_root_q, _ft_slowdown)
        final_pos_cum, final_rot_cum, info, obs, term_b, trunc_b = _rollout_one_step(
            wrapped,
            policy_fn,
            _front_only,
            _AUDIT,
            _ar_log,
            _read_cum,
            ego_cam,
            ego_fs,
            final_ft_cum,
            final_pos_cum,
            final_rot_cum,
            front_cam,
            front_fs,
            have_fingertips,
            have_obj_pos,
            have_obj_rot,
            obs,
            pre_step_ft,
            pre_step_pos,
            pre_step_rot,
            side_cam,
            side_fs,
            sidef_cam,
            sidef_fs,
            info,
            term_b,
            trunc_b,
        )
        if term_b or trunc_b:
            final_pos_cum, final_rot_cum = _read_logged_sums(
                _extract_logged_cum, final_ft_cum, final_pos_cum, final_rot_cum, have_fingertips, info
            )
            break
    else:
        final_pos_cum, final_rot_cum = _read_final_sums(
            _read_cum, final_ft_cum, final_pos_cum, final_rot_cum, have_fingertips, have_obj_pos, have_obj_rot
        )
    return final_pos_cum, final_rot_cum, term_b, trunc_b


def _read_final_sums(
    _read_cum, final_ft_cum, final_pos_cum, final_rot_cum, have_fingertips, have_obj_pos, have_obj_rot
):
    """No termination within max_episode_steps: the sums were not reset, read them after the last step."""
    # max_episode_steps reached without termination: the sums were not reset, read them after the step.
    if have_obj_pos:
        v = _read_cum("obj_pos")
        if v is not None and v >= final_pos_cum:
            final_pos_cum = v
    if have_obj_rot:
        v = _read_cum("obj_rot")
        if v is not None and v >= final_rot_cum:
            final_rot_cum = v
    if have_fingertips:
        for t in _FT_TERMS:
            v = _read_cum(t)
            if v is not None and v >= final_ft_cum[t]:
                final_ft_cum[t] = v
    return final_pos_cum, final_rot_cum


def _place_fingertip_marker(raw_env, traj_mgr, _ft_local, _ft_marker, _ft_root_pos, _ft_root_q, _ft_slowdown):
    """Place the fingertip overlay before the step so the marker is in this step's frame."""
    # The fingertip overlay is placed before the step so the marker is in this step's frame.
    if _ft_marker is not None:
        import isaaclab.utils.math as _mu_ft

        _ef = int((raw_env.episode_length_buf[0].float() / _ft_slowdown).floor() + traj_mgr.start_frames[0])
        _ef = max(0, min(_ef, _ft_local.shape[0] - 1))
        _wpos = _ft_root_pos[0] + _mu_ft.quat_apply(
            _ft_root_q.unsqueeze(0).expand(_ft_local.shape[1], -1), _ft_local[_ef]
        )  # (5*n,3)
        _ft_marker.visualize(translations=_wpos)


def _read_logged_sums(_extract_logged_cum, final_ft_cum, final_pos_cum, final_rot_cum, have_fingertips, info):
    """On termination the env reset the sums: take the logged episode values when they are larger."""
    logged = _extract_logged_cum(info, "obj_pos")
    if logged is not None and logged >= final_pos_cum:
        final_pos_cum = logged
    logged_rot = _extract_logged_cum(info, "obj_rot")
    if logged_rot is not None and logged_rot >= final_rot_cum:
        final_rot_cum = logged_rot
    if have_fingertips:
        for t in _FT_TERMS:
            logged_ft = _extract_logged_cum(info, t)
            if logged_ft is not None and logged_ft >= final_ft_cum[t]:
                final_ft_cum[t] = logged_ft
    return final_pos_cum, final_rot_cum


def _name_termination_terms_fired(wrapped, ego_fs, term_b, trunc_b):
    """Name the termination terms that fired, so a render whose outcome disagrees with the judge shows whether contact
    termination or a tracking gate ended the episode.
    """
    if term_b and not trunc_b:
        try:
            _tm_mgr = wrapped.unwrapped.termination_manager
            _fired = [n for n in _tm_mgr.active_terms if bool(_tm_mgr.get_term(n)[0].item())]
            print(f"  [TERM-WHY] step {len(ego_fs)}: {', '.join(_fired) or '(unknown)'}")
        except Exception as _e_tw:  # noqa: BLE001 — diagnostic lookup only; the rollout goes on
            print(f"  [TERM-WHY] lookup failed: {type(_e_tw).__name__}")


def _rollout_one_step(
    wrapped,
    policy_fn,
    _front_only,
    _AUDIT,
    _ar_log,
    _read_cum,
    ego_cam,
    ego_fs,
    final_ft_cum,
    final_pos_cum,
    final_rot_cum,
    front_cam,
    front_fs,
    have_fingertips,
    have_obj_pos,
    have_obj_rot,
    obs,
    pre_step_ft,
    pre_step_pos,
    pre_step_rot,
    side_cam,
    side_fs,
    sidef_cam,
    sidef_fs,
    info,
    term_b,
    trunc_b,
):
    """One policy step: action, env step, reward bookkeeping and frame capture."""
    final_pos_cum, final_rot_cum, pre_step_pos, pre_step_rot = _pre_step_rewards(
        _read_cum,
        final_ft_cum,
        have_fingertips,
        have_obj_pos,
        have_obj_rot,
        pre_step_ft,
        pre_step_pos,
        pre_step_rot,
        final_pos_cum,
        final_rot_cum,
    )
    with torch.no_grad():
        obs_t = obs["states"] if isinstance(obs, dict) else obs
        actions = policy_fn(obs_t)
    obs, _, terminated, truncated, info = wrapped.step(actions)
    ego_f = _grab_frame(ego_cam)
    if ego_f is not None:
        ego_fs.append(ego_f)
    if front_cam is not None:
        fr_f = _grab_frame(front_cam)
        if fr_f is not None:
            front_fs.append(fr_f)
    if side_cam is not None:
        sd_f = _grab_frame(side_cam)
        if sd_f is not None:
            side_fs.append(sd_f)
    if sidef_cam is not None:
        sdf_f = _grab_frame(sidef_cam)
        if sdf_f is not None:
            sidef_fs.append(sdf_f)
    if _AUDIT:
        # Frame index of the view actually captured (the ego list stays empty under RENDER_FRONT_ONLY).
        _ar_log(len(front_fs if _front_only else ego_fs) - 1)
    term_b = bool(terminated.squeeze().item())
    trunc_b = bool(truncated.squeeze().item())
    _name_termination_terms_fired(wrapped, ego_fs, term_b, trunc_b)
    return final_pos_cum, final_rot_cum, info, obs, term_b, trunc_b


def _pre_step_rewards(
    _read_cum,
    final_ft_cum,
    have_fingertips,
    have_obj_pos,
    have_obj_rot,
    pre_step_ft,
    pre_step_pos,
    pre_step_rot,
    final_pos_cum,
    final_rot_cum,
):
    """Pre-step cumulative tracking rewards (object position / rotation, fingertips) of one step."""
    if have_obj_pos:
        v = _read_cum("obj_pos")
        if v is not None and v >= pre_step_pos:
            pre_step_pos = v
            final_pos_cum = v
    if have_obj_rot:
        v = _read_cum("obj_rot")
        if v is not None and v >= pre_step_rot:
            pre_step_rot = v
            final_rot_cum = v
    if have_fingertips:
        for t in _FT_TERMS:
            v = _read_cum(t)
            if v is not None and v >= pre_step_ft[t]:
                pre_step_ft[t] = v
                final_ft_cum[t] = v
    return final_pos_cum, final_rot_cum, pre_step_pos, pre_step_rot
