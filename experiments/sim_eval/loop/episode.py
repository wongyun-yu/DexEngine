"""One closed-loop episode, step by step (called by closed_loop._closed, which owns the env patch and its undo):
episode_start_and_reset → mode_options → pin_reference_clock → step_rows → strict_pad_hold → build_and_save_meta.
Also policy_cfg (which policy JSON a mode runs). Markers and the episode.json fields are documented in
experiments/sim_eval/closed_loop.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.sim_eval.loop.common import (
    MARK,
    RECORDER,
    _mkdir,
    _say,
    box_down,
)
from experiments.sim_eval.loop.control import (
    _markers_visible,
    arm_ctrl_label,
    arm_lead_setup,
    arm_lead_update,
)
from experiments.sim_eval.loop.record import Recorder
from experiments.sim_eval.loop.replay import (
    SubRec,
    _hold,
    fixed_start_check,
)
from experiments.sim_eval.loop.reset import _episode_start_state, _park, _phys_marker, _reset, _start_override
from experiments.sim_eval.session_cfg import flag01 as _flag01
from experiments.sim_eval.session_cfg import value as _cfg


def policy_cfg(job: dict, base: str) -> dict:
    """Mode base `policy` → job.policy · `policy-<name>` → job.policies[<name>] (several policies in one session —
    one
    scene setup). Missing dies (no default)."""
    if base == "policy":
        if "policy" not in job:
            raise SystemExit(f"{MARK} mode policy — no job.policy")
        return job["policy"]
    name = base[len("policy-") :]
    if not base.startswith("policy-") or name not in (job.get("policies") or {}):
        raise SystemExit(
            f"{MARK} mode {base} — {name!r} not in job.policies (present: {sorted(job.get('policies') or {})})"
        )
    return job["policies"][name]


def episode_start_and_reset(S, raw_env, tg, job: dict, mode: str, data_jpos, fs, names: list) -> float:
    """Park + reset + start override, the reset-state checks and the target init; returns the reset ↔ data row 0
    joint max difference."""
    import numpy as np

    n_ep = getattr(S, "_e5_episodes", 0)
    if n_ep > 0:
        _say(f"[E5_PARK] episode {n_ep + 1}: {_park(S)} rigid bodies moved far + 1 physics step → reset")
    S._e5_episodes = n_ep + 1
    _episode_start_state(S, n_ep)
    _reset(S)
    _start_override(S, tg, job, raw_env)  # E5_START_OVERRIDE (SPIDER rows · default off = no change)
    from dexengine.rl.eval import traj_record as TR

    s0 = TR.state_vector(S.env.scene, TR.scene_entities(S.env.scene))[0].detach().cpu().numpy()
    if n_ep == 0:
        S._e5_s0 = s0
    else:
        d = float(np.abs(s0 - S._e5_s0).max())
        _say(f"[E5_PARK] reset state ↔ first episode's reset state max difference {d:.3g}")
        if d > 0:
            raise SystemExit(
                f"{MARK} reset state differs from the first episode's ({d:.3g}) — an entity the reset does not cover"
            )
    _say("[E5_PHYS] " + _phys_marker(S.env))
    tg.t = {k: a.data.joint_pos_target[0:1].clone() for k, a in tg.arts.items()}
    for n, v in tg.head.items():
        tg.t["robot"][0, tg.head_ids[n]] = float(v)
    tg.last = {k: v.clone() for k, v in tg.t.items()}
    tg.prev = {k: v.clone() for k, v in tg.t.items()}
    tg.sub = dict.fromkeys(tg.arts, 10**9)
    j0 = tg.joint_pos_flat()[0].cpu().numpy()
    d0 = float(np.abs(j0 - data_jpos[0]).max())
    _say(f"{mode}: reset state ↔ data row 0 joint max difference {d0:.4f} rad (every column but the head)")
    if fs is not None:
        fs["reset_maxdiff"] = fixed_start_check(fs, names, j0)
    return d0


def mode_options(
    raw_env,
    tg,
    job: dict,
    mode: str,
    base: str,
    act_mode: bool,
    cap_rows: int,
    fr,
    fs,
    n_data: int,
    row_to_frame: dict,
    sub,
):
    """The mode's policy or chunk replay, the policy_until_t / img_codec / arm_lead options, the recorder and the
    substep recorder."""
    pol = None
    if base.startswith("policy"):
        from experiments.sim_eval import policy_adapter as EP

        pol = EP.make(dict(job, policy=policy_cfg(job, base)), tg, row_to_frame, fr)
    elif base.partition("~")[0] in (
        "replay_chunk",
        "replay_chunk_ns",
    ):  # recorded targets (_ns = next frame's recorded state) · ~smoothing
        from experiments.sim_eval import policy_adapter as EP

        rbase, _, sm = base.partition("~")
        prm = dict(job.get("replay_chunk") or EP.SCHED_DEFAULT)
        prm["source"] = "next_state" if rbase == "replay_chunk_ns" else prm.get("source", "target")
        prm["smooth"] = sm
        pol = EP.ReplayChunk(job, tg, row_to_frame, fr, params=prm)
        _say(f"[E5_REPLAY] {mode}: source {prm['source']} · smoothing {sm or 'none'}")
    # policy_until_t=1 (default off): the policy mode ends like the action replay — policy targets only for rows <
    # T
    # (data length), then the last command target held until the cap (cap_rows), then the same hold window.
    # The replay holds the last recorded action after the data end, so the two modes differ only in who chose the
    # actions
    # (off-data actions in a 1.5×T tail were a failure pattern).
    on = _flag01("policy_until_t")
    until_t = n_data if (on and pol is not None and base.startswith("policy")) else None
    if fs is not None:
        if until_t is None:
            raise SystemExit(f"{MARK} job.fixed_start needs the policy_until_t=1 policy mode — {mode}")
        until_t += fs["extra_rows"]
        _say(
            f"[E5_FIXED_START] {mode}: extra_rows={fs['extra_rows']} · data_rows={n_data} · until_t={until_t} · "
            f"cap_rows={cap_rows} · reset ↔ statement max difference {fs['reset_maxdiff']} · source {fs['source']}"
        )
    if on:
        msg = (
            f"policy targets = rows < T {until_t} · rows {until_t}–{cap_rows - 1} hold the last "
            "command · then the hold window"
            if until_t is not None
            else "not applicable (not a policy mode)"
        )
        _say(f"[E5_POLICY_UNTIL_T] {mode}: {msg}")
    # policy_img_codec=dataset (default off): only the images the policy receives pass the training data's codec
    # chain
    # (record mp4 → converter area + x264); recorded images, videos and judging are unchanged (img_codec.py).
    from experiments.sim_eval import img_codec as IC

    codec = IC.from_env(policy_cfg(job, base)) if (pol is not None and base.startswith("policy")) else None
    if _cfg("policy_img_codec") and codec is None:
        _say(f"[E5_IMG_CODEC] {mode}: not applicable (not a policy mode)")
    codec0 = {}
    lead_setup = arm_lead_setup(raw_env, tg)
    if lead_setup is not None and (pol is None or not hasattr(pol, "s")):
        raise SystemExit(f"{MARK} arm_lead=self needs a mode with a plan (Schedule) — {mode}")
    _say(f"[E5_ARMCTRL] {mode}: arm controller {arm_ctrl_label()}")
    rec, frames = Recorder(raw_env), {"egoL": [], "egoR": [], "row": []}
    if act_mode and _flag01("rec_sub"):
        sub = SubRec(raw_env, tg)
    return pol, on, until_t, codec, codec0, lead_setup, rec, frames, sub


def pin_reference_clock(raw_env):
    """The one env's clip last frame and its episode clock (the step loop keeps the clock at ≤ L−1)."""
    # Reference clock pinned: the env clock (episode_length_buf) is not moved past the clip's last frame L−1. The
    # canonical env
    # ends at the clip end, so some code reads beyond it — the slowdown 1.0 branch of
    # trajectory_manager.get_wrist_pos_w /
    # get_wrist_quat_w reads poses[clip, frame] without clamping (one-clip session: table length = L), out of CUDA
    # range at frame L.
    # Other reads (_abs_frames · _interp_pos · extra slots) clamp to L−1 — with the clock at L−1 every read
    # follows
    # that rule.
    tmg = raw_env.trajectory_manager
    # clip_lengths is per env (trajectory_manager: [num_envs], the length of the clip on that env)
    # — do not index it by clip number (a scene with the source + 2 variant clips gave env 0 = clip 1 →
    # IndexError).
    # An evaluation session has one env.
    if raw_env.num_envs != 1:
        raise SystemExit(f"{MARK} the closed loop takes one env — num_envs {raw_env.num_envs}")
    clip_last = int(tmg.clip_lengths[0]) - 1
    elb = raw_env.episode_length_buf
    return clip_last, elb


def _capture_frame(row_to_frame, n, S, frames, codec, pol, until_t, codec0, tg, t_pol, n_q):
    """Capture both eyes at a video-frame row, encode them for the policy when it is due and feed or query the
    policy; returns (codec0, t_pol, n_q)."""
    import numpy as np

    from dexengine.record.render import episode as CR

    j = row_to_frame[n]
    cap = S._e5_swap.capture(CR.FIRST_WARMUP if j == 0 else CR.FRAME_RENDERS)
    L, R = box_down(cap["egoL"]), box_down(cap["egoR"])
    frames["egoL"].append(L)
    frames["egoR"].append(R)
    frames["row"].append(n)
    Lp, Rp = L, R
    need = codec is not None and (
        getattr(pol, "EVERY_FRAME", False) or ((until_t is None or n < until_t) and pol.due(j))
    )
    if need:
        Lp, Rp = codec(cap["egoL"]), codec(cap["egoR"])
        if not codec0:
            codec0 = {"egoL": Lp, "egoR": Rp, "frame": np.int64(j), "raw_egoL": L, "raw_egoR": R}
    if pol is not None and getattr(pol, "EVERY_FRAME", False):  # dp observation history (no other kind)
        pol.observe(j, n, Lp, Rp, tg.joint_pos_flat()[0].cpu().numpy())
    if pol is not None and (until_t is None or n < until_t) and pol.due(j):
        tq = time.perf_counter()
        pol.query(j, n, Lp, Rp, tg.joint_pos_flat()[0].cpu().numpy())
        t_pol += time.perf_counter() - tq
        n_q += 1
    return codec0, t_pol, n_q


def _set_next_target(base, n, n_data, tg, acts, jpos_data_cols, arm_cols, pol, until_t, lead_setup, step_dt):
    """Target of row n+1 by mode (recorded target, recorded state, arm-state mix or the policy chunk)."""
    if base == "replay":
        if n + 1 < n_data:
            tg.set_flat(acts[n + 1])  # target of row n+1 = the target that made state n+1
    elif base == "replay_ns":  # test variant: target of row n+1 = recorded state n+1
        if n + 1 < n_data:
            tg.set_flat(jpos_data_cols[n + 1])
    elif base == "replay_nsarm":  # test variant: arm = state n+1 · hand and head = stored targets
        if n + 1 < n_data:
            v = acts[n + 1].copy()
            v[arm_cols] = jpos_data_cols[n + 1][arm_cols]
            tg.set_flat(v)
    elif pol is not None and (until_t is None or n + 1 < until_t):
        v = pol.target_for_row(n + 1)
        if v is not None:  # zero-order hold rows have equal values — interp's previous = this, so nothing changes
            tg.set_flat(v, scene_order=bool(getattr(pol, "SCENE_ORDER", False)))
            if lead_setup is not None:
                arm_lead_update(tg, lead_setup, pol.s, n + 1, step_dt, bool(getattr(pol, "SCENE_ORDER", False)))


def step_rows(
    S,
    raw_env,
    tg,
    mode: str,
    base: str,
    rec,
    frames: dict,
    pol,
    codec,
    codec0: dict,
    until_t,
    lead_setup,
    acts,
    act_rows,
    act_mode: bool,
    arm_cols: list,
    jpos_data_cols,
    n_data: int,
    cap_rows: int,
    row_to_frame: dict,
    step_dt: float,
    elb,
    clip_last: int,
    zero_act,
    pdr,
):
    """The episode rows: capture and policy query per video frame, the next row's target, the pinned clock and
    env.step; returns (pinned clock steps, policy time [s], policy queries, codec frame 0)."""
    import torch

    n_pinned, t_pol, n_q = 0, 0.0, 0
    for n in range(cap_rows):
        rec.add(tg)
        if n in row_to_frame:
            codec0, t_pol, n_q = _capture_frame(row_to_frame, n, S, frames, codec, pol, until_t, codec0, tg, t_pol, n_q)
        if n == cap_rows - 1:
            break
        _set_next_target(base, n, n_data, tg, acts, jpos_data_cols, arm_cols, pol, until_t, lead_setup, step_dt)
        if int(elb[0]) >= clip_last:  # stays ≤ L−1 after the in-step += 1
            elb.clamp_(max=clip_last - 1)
            n_pinned += 1
        if act_mode:  # the recorded action that made row n+1 (after the data end = last row held)
            a = torch.as_tensor(act_rows[min(n + 1, n_data - 1)], device=raw_env.device)
            step_act = a.unsqueeze(0).expand(raw_env.num_envs, -1).contiguous()
        else:
            step_act = zero_act
        try:
            raw_env.step(step_act)
            if act_mode or pdr is not None:  # recorder targets column = the target actually written (last substep)
                tg.t = {k: a_.data.joint_pos_target[0:1].clone() for k, a_ in tg.arts.items()}
        except Exception:
            print(f"{MARK} {mode}: env.step failed — row {n} (data {n_data} · cap {cap_rows})", flush=True)
            raise
    return n_pinned, t_pol, n_q, codec0


def strict_pad_hold(
    raw_env, tg, mode: str, rec, act_mode: bool, act_rows, n_data: int, zero_act, n_hold: int, elb, clip_last: int, pdr
) -> int:
    """Strict pad: the hold window simulated after the last action; returns the first hold row."""
    import torch

    # strict pad: simulate the hold window after the last action (action replay = the last recorded action ·
    # absolute targets = tg as is)
    last_act = (
        torch.as_tensor(act_rows[n_data - 1], device=raw_env.device)
        .unsqueeze(0)
        .expand(raw_env.num_envs, -1)
        .contiguous()
        if act_mode
        else zero_act
    )
    hold_from = len(rec.rows)
    _hold(raw_env, tg, rec, last_act, n_hold, elb, clip_last, act_mode or pdr is not None)
    from dexengine.rl.tasks.scene.all_obj_collision import check_all_obj_pose

    check_all_obj_pose(raw_env, "rollout end")  # off = nothing
    _say(
        f"[E5_HOLD] {mode}: hold window of {n_hold} rows simulated after the last action (rows "
        f"{hold_from}–{len(rec.rows) - 1})"
    )
    return hold_from


def _check_markers_hidden(raw_env, mode):
    """Debug markers visible at the episode end make the episode invalid (dies); returns the visible list."""
    vis = _markers_visible(raw_env)
    if vis:
        raise SystemExit(
            f"{MARK} {mode}: debug markers {vis} visible at the episode end — the reference pose was "
            "drawn in the policy's view (invalid)"
        )
    return vis


def _add_optional_meta(fs, meta, until_t, on, codec):
    """Fields written only when their switch is on: fixed start, policy_until_t, image codec."""
    if fs is not None:  # only with a fixed start
        meta["fixed_start"] = {
            "extra_rows": fs["extra_rows"],
            "until_t": until_t,
            "source": fs["source"],
            "reset_maxdiff": fs["reset_maxdiff"],
        }
    if on:  # the field is written only when on
        meta["policy_until_t"] = until_t
    if codec is not None:  # only when on
        meta["img_codec"] = {
            "value": _cfg("policy_img_codec"),
            "frames": codec.n,
            "stage1": codec.s1,
            "stage2": codec.s2,
        }


def _save_episode(out, mode, rec, frames, meta, sub, n_steps, pol, codec0):
    """Write <out>/<mode>/: episode.npz, substep targets, policy io, codec frame 0 and episode.json."""
    import numpy as np

    _mkdir(out / mode)
    rec.save(out / mode / "episode.npz", frames, meta)
    if sub is not None:
        meta["substep_spec"] = sub.save(out / mode / "substep_targets.npz", n_steps)
    if pol is not None and hasattr(pol, "save_io"):
        pol.save_io(out / mode / "policy_io.npz")
    if codec0:
        np.savez_compressed(out / mode / "codec_frame0.npz", **codec0)
    (out / mode / "episode.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=float))


def _say_episode_end(mode, clip_last, n_pinned, rec, cap_rows, frames, wall, t_pol, n_q, out):
    """Log the reference-clock and episode summary lines."""
    _say(f"[E5_CLOCK] {mode}: reference clock last frame {clip_last} · pinned steps after it {n_pinned}")
    _say(
        f"{mode}: rows {len(rec.rows)}/{cap_rows} · frames {len(frames['row'])} · {wall:.1f} s "
        f"(policy {t_pol:.1f} s · queries {n_q}) → {out / mode}"
    )


def build_and_save_meta(
    S,
    raw_env,
    job: dict,
    out: Path,
    mode: str,
    base: str,
    tg,
    rec,
    frames: dict,
    pol,
    codec,
    codec0: dict,
    sub,
    pdr,
    act_info,
    fs,
    on,
    until_t,
    names: list,
    n_data: int,
    cap_rows: int,
    step_dt: float,
    d0: float,
    t0: float,
    clip_last: int,
    n_pinned: int,
    n_hold: int,
    hold_from: int,
    hold_info,
    t_pol: float,
    n_q: int,
) -> dict:
    """Episode meta (episode.json) and the saved arrays under <out>/<mode>/; returns the meta."""

    wall = time.perf_counter() - t0
    n_steps = len(rec.rows) - 1
    vis = _check_markers_hidden(raw_env, mode)
    meta = {
        "mode": mode,
        "base": base,
        "ctrl": tg.ctrl,
        "arm_ctrl": arm_ctrl_label(),
        "profile": _cfg("profile"),
        "world": S._e5_swap.why,
        "world_swaps": S._e5_swap.n,
        "markers_visible": vis,
        "decimation": tg.D,
        "rows": len(rec.rows),
        "cap_rows": cap_rows,
        "data_rows": n_data,
        "cap_mult": job["cap_mult"],
        "frames": len(frames["row"]),
        "step_dt": step_dt,
        "reset_joint_maxdiff_rad": d0,
        "wall_s": round(wall, 2),
        "clock_last_frame": clip_last,
        "clock_pinned_steps": n_pinned,
        "hold_rows": n_hold,
        "hold_from": hold_from,
        "hold": "simulated",
        "data_hold": hold_info,
        "policy_s": round(t_pol, 2),
        "policy_queries": n_q,
        "joint_names": names,
        "recorder": RECORDER,
        "held": rec.held_entities,
        "held_sides": [s for s, _o in rec.held],
        "link_names": rec.hoc.link_names(),
        "data_cols": tg.cols,
        "policy": (pol.describe() if pol is not None else None),
        "act_replay": act_info,
        "pd_replay": (
            {"src": str(pdr.src_dir), "spec": pdr.spec, "hold": pdr.hold or None} if pdr is not None else None
        ),
    }
    _add_optional_meta(fs, meta, until_t, on, codec)
    _save_episode(out, mode, rec, frames, meta, sub, n_steps, pol, codec0)
    _say_episode_end(mode, clip_last, n_pinned, rec, cap_rows, frames, wall, t_pol, n_q, out)
    return meta
