"""Body of a closed-loop evaluation session: replaces the canonical frozen render point
(record render `_run_per_clip_video_render`, swapped in by session.py).

Why this point: a `RENDER_ONLY=<checkpoint>` dump has the argv and env of the stage-1 record dump (only the render
fields and the env count differ — prep compares both and stores the result in job.json) and reaches here after the
frozen-eval neutralisation (curriculum, DR, gravity ramp, RSI 0) → physics = the data's stage-1 physics. The render
layer
is the record stage-2 code as is (record Session, load_traj — camera calibration, body visual layer, supports, extra
objects, look D settle), dressed with the data episode's stage-1 trajectory (seed) → the same scene as the training
images. The camera pose is set from that whole trajectory (the data's value; the camera is the only user of the future
trajectory).

Modes (job.json "modes", run in order in one session, env reset per mode):
  gate      pixel gate — writes the seed trajectory's recorded states frame by frame (no physics, the data's render
  count)
            and captures both eyes → frames_gate.npz (320×180, video-frame order) + a few full-resolution frames.
  replay    open-loop oracle — the data episode's action.joint_target (row k = the target that produced state k) is
            the physics-step target of row k.
  zero      zero action — the post-reset target is held to the end.
  policy    policy server (job.policy) — at video frame j it observes (image j, state row i(j)); chunk target a_k is
  held
            from row i(j+k)+1 until the next chunk element (training contract: action = target[i(j)+1]).
Physics step: raw_env.step(0) unchanged (canonical step — spawn hold, extra slots, reset events, 50 Hz, decimation 2);
what changes is (1) the action term's apply_actions → absolute joint targets (every joint of robot and robot_left, the
head at its pinned angle), (2) terminations off (the length cap is cap_mult × the data length, counted here), (3) no
render inside env.step (the data renders once per frame).
Record: per row the scene state (traj_record.state_vector — the render binding's columns), hand–object link forces
(hand_object_contact — the record force columns' measurement), targets, and both eyes at 320×180 per frame →
<out>/<mode>/episode.npz + episode.json. Judging is offline (judges/).
Markers [E5_EVAL] …, last line [E5_EVAL] DONE modes=<k>/<n>.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from experiments.sim_eval.loop.common import (
    ACT_REPLAY,
    CTRLS,
    DATA_HOLD,
    HOLD_WINDOW_S,
    MARK,
    PD_REPLAY,
    ROW_HZ,
    _mkdir,
    _say,
    frame_rows,
    hold_rows,
)
from experiments.sim_eval.loop.control import (
    Targets,
    _patch_env,
    _unpatch,
)
from experiments.sim_eval.loop.episode import (
    build_and_save_meta,
    episode_start_and_reset,
    mode_options,
    pin_reference_clock,
    policy_cfg,
    step_rows,
    strict_pad_hold,
)
from experiments.sim_eval.loop.record import _gate, _reference, _render_states
from experiments.sim_eval.loop.replay import (
    PdReplay,
    _act_rows,
    _data_hold_start,
    _load_actions,
    fixed_start_cfg,
)
from experiments.sim_eval.loop.scene import RenderShift, _paint_objects


def mode_ok(job: dict, mode: str) -> bool:
    """Mode syntax `<base>@<controller>` · base ∈ replay | replay_chunk | zero | policy | policy-<name> (job.policies)
    | act_replay | act_replay0 (action replay, no controller suffix) · controller ∈ CTRLS."""
    base, _, ctrl = mode.partition("@")
    if ctrl not in ("",) + CTRLS:
        return False
    rbase, _, sm = base.partition(
        "~"
    )  # smoothing suffix (replay_chunk · replay_chunk_ns only): ~c<w> centred · ~k<w> causal
    if sm:
        from experiments.sim_eval import policy_adapter as EP

        EP.parse_smooth(sm)
        return rbase in ("replay_chunk", "replay_chunk_ns")
    if base in ("replay", "zero", "replay_chunk", "replay_ns", "replay_nsarm", "replay_chunk_ns"):
        return True
    if base in ACT_REPLAY or base in (PD_REPLAY, DATA_HOLD):
        return not ctrl
    if base == "policy" or base.startswith("policy-"):
        policy_cfg(job, base)
        return True
    return False


def _bind_targets(raw_env, S, ctrl, job, base, mode):
    """Absolute-target writer of the mode and the data columns it binds; returns targets, joint names, data actions
    and joint states."""
    tg = Targets(raw_env, S.rig.pinned, ctrl or "hold")
    names = [n for _k, n in tg.order]
    acts, act_names, data_jpos = _load_actions(job)
    jpos_data_cols = data_jpos  # joint states in data-column order (target source of replay_ns)
    arm_cols = [i for i, nm in enumerate(act_names) if "_arm_" in nm]  # columns replay_nsarm switches to the next state
    if base == "replay_nsarm":
        if len(arm_cols) != 14:
            raise SystemExit(f"{MARK} replay_nsarm: arm columns {len(arm_cols)} ≠ 14 — names {act_names}")
        n_rest = len(act_names) - len(arm_cols)
        _say(f"[E5_NSARM] arm {len(arm_cols)} columns = recorded next state · other {n_rest} columns = stored targets")
    bound = tg.bind(act_names)
    data_jpos = data_jpos[:, tg.cols]
    _say(f"[E5_CTRL] {mode}: controller {tg.ctrl} · substeps {tg.D} · {bound}")
    return tg, names, act_names, acts, data_jpos, jpos_data_cols, arm_cols


def _row_plan(acts, job, base, raw_env):
    """Data rows, row cap (cap_mult × data, + fixed-start rows), step period and the video-frame rows."""
    n_data = int(acts.shape[0])
    cap_rows = int(math.ceil(float(job["cap_mult"]) * n_data))
    fs = fixed_start_cfg(job, base)  # fixed start (off unless job.fixed_start)
    if fs is not None:
        cap_rows += fs["extra_rows"]
    step_dt = float(raw_env.step_dt)
    if abs(step_dt - 1 / ROW_HZ) > 1e-9:
        raise SystemExit(f"{MARK} step_dt {step_dt} ≠ 1/{ROW_HZ} — differs from the data row period")
    fr = frame_rows(cap_rows, step_dt)
    row_to_frame = {r: j for j, r in enumerate(fr)}
    return n_data, cap_rows, fs, step_dt, fr, row_to_frame


def _actuation(base, ctrl, mode, job, S, raw_env, acts, act_names, tg, out):
    """How the mode drives the robot: recorded-action replay rows or the PD replayer, and the env patch that applies
    the targets; returns (action mode, action info, action rows, PD replayer, saved env state for _unpatch)."""
    act_mode = base in ACT_REPLAY
    if (act_mode or base == PD_REPLAY) and ctrl:
        raise SystemExit(
            f"{MARK} {mode}: action replay takes no controller suffix (@{ctrl}) — the canonical action "
            "term sets the substeps"
        )
    pinned = {
        k
        for k, v in ((job.get("episode") or {}).get("head_joints") or {}).items()
        if isinstance(v, dict) and v.get("source") == "render_pinned"
    }
    act_rows, act_info = (
        _act_rows(S, raw_env, acts, act_names, base == "act_replay0", pinned) if act_mode else (None, None)
    )
    pdr = PdReplay(raw_env, tg, job, out) if base == PD_REPLAY else None
    if pdr is not None:
        act_info = pdr.src_meta.get(
            "act_replay"
        )  # same scene · same recorded trajectory (reference rows of the object error)
    saved = _patch_env(raw_env, tg, absolute=not act_mode, apply_fn=(pdr.apply if pdr is not None else None))
    return act_mode, act_info, act_rows, pdr, saved


def _rollout_setup(raw_env, base, saved, S, T, tg, acts, cap_rows, row_to_frame):
    """Pinned reference clock, zero action, counters, object-readiness check and the data-hold start of the mode."""
    import torch

    clip_last, elb = pin_reference_clock(raw_env)
    zero_act = torch.zeros(raw_env.num_envs, raw_env.action_manager.total_action_dim, device=raw_env.device)
    t0 = time.perf_counter()
    from dexengine.rl.tasks.scene.all_obj_collision import assert_all_obj_ready

    assert_all_obj_ready(raw_env)  # an option was given but its consumer was not reached → dies (off = nothing)
    hold_info = None
    if (
        base == DATA_HOLD
    ):  # data baseline: no rollout, only the hold window from the stage-1 record's last row (the policy's _hold)
        # Extra slots (kinematic, teleported to the reference frame) follow the data reference as in stage 1 — the
        # reference clock is pinned at L−1, so they keep the last frame's pose through the hold window. With extra_f0
        # (policy-eval rule: frame 0) the written last row would teleport to frame 0 in one step and the judge would
        # read a blow-up (seen as extra_obj_0 at 8.48 m/s).
        if saved.get("extra_f0") is not None:
            del saved["extra_f0"].pre_physics
            saved["extra_f0"] = None
            _say(
                f"[E5_HOLD] {DATA_HOLD}: extra slots = data reference (last frame) — "
                "extra_f0 frame-0 pin left out in this mode"
            )
        hold_info = _data_hold_start(S, T, raw_env, tg, acts, elb, clip_last)
        cap_rows, row_to_frame = 1, {}  # no policy — no images captured
    return clip_last, elb, zero_act, t0, hold_info, cap_rows, row_to_frame


def _hold_window():
    """Hold-window length in rows (logged once per mode)."""
    n_hold = hold_rows()
    _say(
        f"[E5_HOLD_WINDOW] window {HOLD_WINDOW_S} s = {n_hold} rows · simulated with the last action kept "
        f"(owner loop.common.HOLD_WINDOW_S · hold length = judge rule put.hold_s)"
    )
    return n_hold


def _undo_patch(sub, raw_env, saved):
    """Undo the substep recorder and the env patch of the mode."""
    if sub is not None:
        sub.undo()
    _unpatch(raw_env, saved)


def _closed(S, T, raw_env, out: Path, job: dict, mode: str) -> dict:
    base, _, ctrl = mode.partition("@")
    n_hold = _hold_window()
    tg, names, act_names, acts, data_jpos, jpos_data_cols, arm_cols = _bind_targets(raw_env, S, ctrl, job, base, mode)
    n_data, cap_rows, fs, step_dt, fr, row_to_frame = _row_plan(acts, job, base, raw_env)
    act_mode, act_info, act_rows, pdr, saved = _actuation(base, ctrl, mode, job, S, raw_env, acts, act_names, tg, out)
    sub = None  # the substep recorder is attached right before the episode loop (reset and setup steps are not counted)
    try:
        d0 = episode_start_and_reset(S, raw_env, tg, job, mode, data_jpos, fs, names)
        pol, on, until_t, codec, codec0, lead_setup, rec, frames, sub = mode_options(
            raw_env, tg, job, mode, base, act_mode, cap_rows, fr, fs, n_data, row_to_frame, sub
        )
        clip_last, elb, zero_act, t0, hold_info, cap_rows, row_to_frame = _rollout_setup(
            raw_env, base, saved, S, T, tg, acts, cap_rows, row_to_frame
        )
        n_pinned, t_pol, n_q, codec0 = step_rows(
            S,
            raw_env,
            tg,
            mode,
            base,
            rec,
            frames,
            pol,
            codec,
            codec0,
            until_t,
            lead_setup,
            acts,
            act_rows,
            act_mode,
            arm_cols,
            jpos_data_cols,
            n_data,
            cap_rows,
            row_to_frame,
            step_dt,
            elb,
            clip_last,
            zero_act,
            pdr,
        )
        hold_from = strict_pad_hold(
            raw_env, tg, mode, rec, act_mode, act_rows, n_data, zero_act, n_hold, elb, clip_last, pdr
        )
        return build_and_save_meta(
            S,
            raw_env,
            job,
            out,
            mode,
            base,
            tg,
            rec,
            frames,
            pol,
            codec,
            codec0,
            sub,
            pdr,
            act_info,
            fs,
            on,
            until_t,
            names,
            n_data,
            cap_rows,
            step_dt,
            d0,
            t0,
            clip_last,
            n_pinned,
            n_hold,
            hold_from,
            hold_info,
            t_pol,
            n_q,
        )
    finally:
        _undo_patch(sub, raw_env, saved)


def _plan_episode(plan, names, loaded):
    """Plan = one trajectory (that trajectory, look D) or the canonical render session form [source (calibrate only),
    episode trajectory] (per_look=fixed · aug — the source sets head angle and camera, the episode trajectory is
    checked with check_view) · one episode. Returns (episode, two trajectories)."""
    eps = plan.get("episodes") or [{"traj": names[0], "look": "D"}]
    ep = eps[0]
    two = len(names) == 2
    if (
        len(eps) != 1
        or ep.get("traj") not in names
        or sorted(loaded) != sorted(names)
        or len(names) > 2
        or (two and (plan["trajs"][0].get("role") != "orig" or ep["traj"] == names[0]))
    ):
        raise SystemExit(
            f"{MARK} session clips {loaded} · plan trajectories {names} · episodes {[e.get('traj') for e in eps]} — "
            "one trajectory or [source (role orig), episode trajectory] · one episode only"
        )
    return ep, two


def _source_camera(S, plan, loaded, raw_env, names, ep):
    """A two-trajectory plan: the source trajectory sets the camera and the pinned head angle, then is cleared."""
    from dexengine.record.render import clip_phases as PH

    head = {"prim_oracle": []}
    PH.orig_camera(S, plan, loaded, head, raw_env, measure=False)
    _say(
        f"[E5_PLAN] source {names[0]} sets camera and head angle (pinned {S.rig.pinned}) → cleared "
        f"({len(head['prim_oracle'])}) → "
        f"episode trajectory {ep['traj']}"
    )


def _settle_look(look, plan, S, T, ep, job):
    """Dress the scene in the episode look (random look via the look run, or look D) and settle it; returns
    (settle info, exposure ev)."""
    from dexengine.record.looks import look_d as LOOK
    from dexengine.record.looks import rev as LREV

    if (
        look != "D"
    ):  # same path as the canonical render session: allow (head session file) → spec overlay → canonical exposure 0
        from dexengine.record.looks import look_run as LRN

        lo = Path(plan["out"]) / "_look"
        LR = LRN.look_run_cls()(S, lo)
        LRN.adopt(LR, plan)
        spec, info = LR.spec_for(T, ep, plan)
        ls = LOOK.settle(S.rig)
        ev = 0.0
        _say(
            f"[E5_LOOK] random look {look} · key {plan.get('look_key')} · deck {plan.get('deck')} · "
            f"overlays {sorted(info)[:8]}"
        )
    else:
        ls = LOOK.settle(S.rig)
        ev = float(((job["episode"].get("look") or {}).get("exposure") or {}).get("ev", LREV.LOOK_REV["d_ev"]))
    return ls, ev


def _run_modes(traj_mgr, out, job, S, T, raw_env):
    """Run every job mode in order; returns (reference poses, per-mode results)."""
    ref = _reference(traj_mgr, out)
    res = []
    for mode in job["modes"]:
        if mode == "gate":
            res.append(_gate(S, T, out, job))
        elif mode.startswith("render-"):
            res.append(_render_states(S, T, out, job, mode))
        elif mode_ok(job, mode):
            res.append(_closed(S, T, raw_env, out, job, mode))
        else:
            raise SystemExit(f"{MARK} unknown mode {mode}")
    return ref, res


def _write_session(ep, res, ref, ls, body, ev, S, t_all, out):
    """Write <out>/session.json; returns the summary."""
    summ = {
        "clip": ep["traj"],
        "modes": res,
        "reference": ref,
        "settle": ls,
        "body": body.get("method"),
        "exposure_ev": ev,
        "camera": S.cal,
        "total_s": round(time.perf_counter() - t_all, 1),
        "tree": os.environ.get("COLLECT_TREE", ""),
    }
    (out / "session.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=float))
    return summ


def run(
    wrapped,
    policy_fn,
    traj_mgr,
    num_clips,
    clip_names,
    raw_env,
    checkpoint_path=None,
    out_dir=None,
    *_a,
    args_cli=None,
    **_kw,
):
    from dexengine.core import paths as P
    from dexengine.record.looks import rev as LREV
    from dexengine.record.render import episode as CR
    from dexengine.record.session import runner as SS

    t_all = time.perf_counter()
    job = json.loads(Path(os.environ["E5_JOB"]).read_text())
    bad = [m for m in job["modes"] if m != "gate" and not m.startswith("render-") and not mode_ok(job, m)]
    if bad:  # dies before the scene setup (≈ 4 min)
        raise SystemExit(f"{MARK} unknown modes {bad}")
    plan = json.loads(Path(os.environ["COLLECT_PLAN"]).read_text())
    out = _mkdir(job["out"])
    names = [t["name"] for t in plan["trajs"]]
    loaded = [P.tree_name(c) for c in clip_names]
    ep, two = _plan_episode(plan, names, loaded)
    traj_of = {t["name"]: t for t in plan["trajs"]}
    S = CR.Session(wrapped, traj_mgr, raw_env, args_cli, checkpoint_path)
    tm = dict.fromkeys(CR.TM_KEYS, 0.0)
    if two:
        _source_camera(S, plan, loaded, raw_env, names, ep)
    T = CR.load_traj(S, loaded.index(ep["traj"]), ep["traj"], traj_of[ep["traj"]]["traj"], tm)
    S._e5_traj_src = (traj_of[ep["traj"]]["traj"], ep["traj"])  # action-replay source (load_traj's file and clip)
    body = T["body"]
    if body["method"] == "desk_stretch" and any(m != "gate" for m in job["modes"]):
        raise SystemExit(
            f"{MARK} body method desk_stretch — moves the desk-top collider into the render frame (differs "
            "from the stage-1 physics frame) · closed loop not implemented"
        )
    t = time.perf_counter()
    look = str(ep.get("look") or "D")
    _paint_objects(
        S, T, look, ep["traj"]
    )  # obj_color (default off) — before settling (the data look is painted too → settle)
    ls, ev = _settle_look(look, plan, S, T, ep, job)
    LREV.set_exposure(ev)
    S._e5_swap = RenderShift(S, T, body)
    _say(
        f"scene dressed {ep['traj']} · body {body['method']} · settle {time.perf_counter() - t:.1f} s "
        f"({ls.get('renders')} renders) · "
        f"exposure ev {ev} · head pinned {S.rig.pinned} · {S._e5_swap.describe()}"
    )
    ref, res = _run_modes(traj_mgr, out, job, S, T, raw_env)
    summ = _write_session(ep, res, ref, ls, body, ev, S, t_all, out)
    _say(f"DONE modes={len(res)}/{len(job['modes'])} · {summ['total_s']} s · {out}")
    SS.finish(len(res) == len(job["modes"]), wrapped)
