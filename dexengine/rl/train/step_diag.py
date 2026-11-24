"""Per-step episode and termination counters, accumulated on the GPU without a per-step sync."""

import torch


def install_step_diag(*, num_envs, raw_env, runner):
    """Per-step episode/termination counters accumulated on the GPU (patched ``record_transition``)."""
    # snapshot_metrics reads body positions of this side (fixed to "right", also for left-uni runs).
    side = "right"
    raw_env.scene["right_hand"]  # unused lookup, kept: it raises KeyError on a scene without a right_hand entity
    traj_mgr = raw_env.trajectory_manager

    diag = {
        "total_episodes": 0,
        "early_terminations": 0,
        "timeouts": 0,
        "ep_lengths_sum": 0.0,
        "chunk_episodes": 0,
        "chunk_early": 0,
        "chunk_timeouts": 0,
        "chunk_ep_len_sum": 0.0,
        "chunk_progress_sum": 0.0,  # sum of (start+ep_len)/clip_length per terminated episode
        "chunk_f0_episodes": 0,
        "chunk_f0_early": 0,
        "chunk_rsi_episodes": 0,
        "chunk_rsi_early": 0,
    }
    history = []
    ep_len_tracker = torch.zeros(num_envs, device=raw_env.device, dtype=torch.long)
    ep_start_frame = torch.zeros(num_envs, device=raw_env.device, dtype=torch.long)

    original_record = runner.agent.record_transition

    return _install_step_counters(
        raw_env, runner, diag, ep_len_tracker, ep_start_frame, history, original_record, side, traj_mgr
    )


def _install_step_counters(
    raw_env, runner, diag, ep_len_tracker, ep_start_frame, history, original_record, side, traj_mgr
):
    """Step diagnostics accumulate in GPU tensors without a sync; a consumer (the periodic eval block, the end-of-
    training summary) collects them right before reading (one sync). Boolean-mask indexing (x[done]) would also sync,
    so the counts use mask arithmetic.
    """
    _tm_fd = raw_env.termination_manager
    _fd_nt_idx = [i for i, c in enumerate(_tm_fd._term_cfgs) if not c.time_out]
    _fd_nt_names = [_tm_fd._term_names[i] for i in _fd_nt_idx]
    for _n in _fd_nt_names:
        diag.setdefault(f"chunk_term_{_n}", 0)
    diag.setdefault("chunk_term_object_tracking_pos", 0)
    diag.setdefault("chunk_term_object_tracking_rot", 0)
    _fd_idx_t = torch.tensor(_fd_nt_idx, device=raw_env.device, dtype=torch.long)
    _fd_acc = {
        # [done, early, timeout, ep_len_sum, progress_sum, f0_eps, f0_early, rsi_eps, rsi_early]
        "scal": torch.zeros(9, device=raw_env.device, dtype=torch.float64),
        "term": torch.zeros(len(_fd_nt_idx), device=raw_env.device, dtype=torch.float64),
        "objtrack": torch.zeros(2, device=raw_env.device, dtype=torch.float64),
    }

    def _fast_diag_flush():
        """Move the GPU accumulators into ``diag`` (one sync); call right before reading ``diag``."""
        scal = _fd_acc["scal"].cpu()
        termv = _fd_acc["term"].cpu()
        objv = _fd_acc["objtrack"].cpu()
        _fd_acc["scal"].zero_()
        _fd_acc["term"].zero_()
        _fd_acc["objtrack"].zero_()
        nd, ne, nt = int(scal[0]), int(scal[1]), int(scal[2])
        diag["total_episodes"] += nd
        diag["early_terminations"] += ne
        diag["timeouts"] += nt
        diag["chunk_episodes"] += nd
        diag["chunk_early"] += ne
        diag["chunk_timeouts"] += nt
        diag["ep_lengths_sum"] += float(scal[3])
        diag["chunk_ep_len_sum"] += float(scal[3])
        diag["chunk_progress_sum"] += float(scal[4])
        diag["chunk_f0_episodes"] += int(scal[5])
        diag["chunk_f0_early"] += int(scal[6])
        diag["chunk_rsi_episodes"] += int(scal[7])
        diag["chunk_rsi_early"] += int(scal[8])
        for _n, _v in zip(_fd_nt_names, termv.tolist(), strict=True):
            if _v:
                diag[f"chunk_term_{_n}"] += int(_v)
        diag["chunk_term_object_tracking_pos"] += int(objv[0])
        diag["chunk_term_object_tracking_rot"] += int(objv[1])

    return _install_record_hook(
        raw_env,
        runner,
        _fast_diag_flush,
        _fd_acc,
        _fd_idx_t,
        _tm_fd,
        diag,
        ep_len_tracker,
        ep_start_frame,
        history,
        original_record,
        side,
        traj_mgr,
    )


def _install_record_hook(
    raw_env,
    runner,
    _fast_diag_flush,
    _fd_acc,
    _fd_idx_t,
    _tm_fd,
    diag,
    ep_len_tracker,
    ep_start_frame,
    history,
    original_record,
    side,
    traj_mgr,
):
    """Wrap record_transition with the per-step episode/termination counters."""

    def patched_record(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        result = original_record(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )
        ep_len_tracker.add_(1)
        term = terminated.squeeze(-1).bool()
        trunc = truncated.squeeze(-1).bool()
        done = term | trunc
        early_mask = term & ~trunc  # early ⊂ done
        donef = done.to(torch.float64)
        ep_lens = ep_len_tracker.to(torch.float64)
        clip_len = traj_mgr.clip_lengths.to(torch.float64).clamp(min=1)
        progress = ((ep_start_frame.to(torch.float64) + ep_lens) / clip_len).clamp(max=1.0)
        f0_mask = ep_start_frame == 0
        _fd_acc["scal"] += torch.stack(
            [
                donef.sum(),
                early_mask.sum().to(torch.float64),
                trunc.sum().to(torch.float64),
                (ep_lens * donef).sum(),
                (progress * donef).sum(),
                (f0_mask & done).sum().to(torch.float64),
                (f0_mask & early_mask).sum().to(torch.float64),
                (~f0_mask & done).sum().to(torch.float64),
                (~f0_mask & early_mask).sum().to(torch.float64),
            ]
        )
        _fd_acc["term"] += (_tm_fd._term_dones[:, _fd_idx_t] & early_mask.unsqueeze(-1)).to(torch.float64).sum(0)
        _pf = getattr(raw_env, "_objtrack_pos_fired", None)
        _rf = getattr(raw_env, "_objtrack_rot_fired", None)
        if _pf is not None and _rf is not None:
            _em = early_mask.to(torch.float64)
            _fd_acc["objtrack"] += torch.stack(
                [
                    (_pf.to(torch.float64) * _em).sum(),
                    (_rf.to(torch.float64) * _em).sum(),
                ]
            )
        ep_len_tracker.masked_fill_(done, 0)
        ep_start_frame.copy_(torch.where(done, traj_mgr.start_frames.to(ep_start_frame.dtype), ep_start_frame))
        return result

    runner.agent.record_transition = patched_record
    return _fast_diag_flush, diag, history, side, traj_mgr
