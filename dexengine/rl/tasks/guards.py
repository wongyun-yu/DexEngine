"""Single owner of the absurd-state (ABSURD) rules.

Keeps a runaway articulated object from making training diverge.

Why. In a bundle of sliding-door and drawer articulations (0.2 kg) one env's state jumped to 1.18e14 in a single step
  and training diverged ([NONFINITE_UPDATE], 226 non-finite policy tensors). The articulated-object sanity termination
  (mdp.terminations.ArticulatedObjectSanity) skips the speed rule for steps that start inside the post-reset pin window
  (SPAWN_HOLD) and always counts non-finite states only; 1.18e14 is finite, so it passed both. That step's reward is
  computed before the termination and became NaN (total, right and left rewards), entered the rollout and corrupted
  the PPO update.

Two rules (one place):
  (A) absurd(obj, env_origins): True when one of the articulated object's root linear speed, angular speed or joint
      speeds exceeds ABSURD_VEL, or its root is farther than ABSURD_DIST_M from the env origin. The sanity termination
      adds it to the 'always counted' set next to non-finite (independent of the pin window and the judge park).
  (B) sanitize_step(env, ret): only for the envs cut this step by a non-finite or ABSURD state, replace the non-finite
      elements of the rewards (total and per-side extras) and observations with 0 (nan_to_num) and print the marker
      [ABSURD_STATE] (step · env · clip · cause). Without such envs it writes no tensor and draws no RNG, so normal
      steps are byte-identical (cost = one .any() sync per step).

Thresholds (the two constants below own the values):
  * speed 1e3: reference object speed < ~2 m/s, reference hinge speed < ~45 rad/s (joint speed limit 60), the sanity
    speed thresholds are 10 m/s and 55 rad/s. No reference motion reaches 1e3, so it never cuts a normal env
    (measured runaway 1.18e14).
  * distance 10 m: env spacing 2.5 m, workspace < 1.5 m. 10 m from the origin is four neighbouring envs away.
"""

from __future__ import annotations

import torch

#: (A) speed threshold, shared by linear (m/s), angular (rad/s) and joint speeds (rad/s or m/s)
ABSURD_VEL = 1.0e3
#: (A) distance threshold (m) between the object root and the env origin
ABSURD_DIST_M = 10.0
#: (B) marker
MARK = "[ABSURD_STATE]"
#: (B) marker line cap: the first this many every time, then every 100th (the count goes on)
MARK_FIRST = 50


def absurd(obj, env_origins: torch.Tensor) -> torch.Tensor:
    """(num_envs,) bool: absurd state of the articulated object obj.

    NaN compares False (the caller counts non-finite states separately)."""
    d = obj.data
    pos = d.root_state_w[:, :3] - env_origins
    return (
        (d.root_lin_vel_w.norm(dim=-1) > ABSURD_VEL)
        | (d.root_ang_vel_w.norm(dim=-1) > ABSURD_VEL)
        | (d.joint_vel.abs() > ABSURD_VEL).any(dim=-1)
        | (pos.norm(dim=-1) > ABSURD_DIST_M)
    )


def note(env, key: str, bad: torch.Tensor, absurd_m: torch.Tensor) -> None:
    """The sanity termination leaves this step's (non-finite, ABSURD) masks on the env.

    Read by sanitize_step. No sync."""
    step = int(env.common_step_counter)
    if getattr(env, "_absurd_step", None) != step:
        env._absurd_step, env._absurd_terms = step, {}
    env._absurd_terms[key] = (bad, absurd_m)


def _fix_rows(t, ids: torch.Tensor) -> int:
    """Zero the non-finite elements of t[ids]; returns how many changed. 0 for a non-float tensor."""
    if not torch.is_tensor(t) or not t.is_floating_point():
        return 0
    rows = t[ids]
    nf = int((~torch.isfinite(rows)).sum().item())
    if nf:
        t[ids] = torch.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    return nf


def _clip_names(env, ids: torch.Tensor) -> str:
    tm = getattr(env, "trajectory_manager", None)
    if tm is None or not hasattr(tm, "clip_ids"):
        return "-"
    names = getattr(getattr(tm, "_loader", None), "sequence_names", None) or []
    cid = tm.clip_ids[ids].tolist()
    return ",".join(names[c] if c < len(names) else str(c) for c in cid)


def _fh_nan_mark(env, ret) -> None:
    """DEXENGINE_FH_NAN_RESET=1: envs whose reward this step (total or per side) is non-finite get a finite (0) reward;
    those not ending this step are flagged for termination next step (mdp.terminations.fh_nan_reset); marker
    [FH_NAN_RESET]. Without such envs it writes nothing."""
    import torch

    obs, rew, term, trunc, extras = ret
    bad = ~torch.isfinite(rew.reshape(rew.shape[0], -1)).all(dim=-1)
    for k in ("reward_right", "reward_left"):
        v = extras.get(k) if isinstance(extras, dict) else None
        if torch.is_tensor(v):
            bad = bad | ~torch.isfinite(v.reshape(v.shape[0], -1)).all(dim=-1)
    if not bool(bad.any()):
        return
    ids = bad.nonzero(as_tuple=False).squeeze(-1)
    n_r = _fix_rows(rew, ids)
    n_s = sum(_fix_rows(extras.get(k), ids) for k in ("reward_right", "reward_left")) if isinstance(extras, dict) else 0
    pend = getattr(env, "_fh_nan_pending", None)
    if pend is None:
        pend = env._fh_nan_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=rew.device)
    pend |= bad & ~(term.bool() | trunc.bool())
    env._fh_nan_count = int(getattr(env, "_fh_nan_count", 0)) + 1
    env._fh_nan_envs = int(getattr(env, "_fh_nan_envs", 0)) + int(ids.numel())
    k = env._fh_nan_count
    if k <= MARK_FIRST or k % 100 == 0:
        print(
            f"[FH_NAN_RESET] event {k} · step {int(env.common_step_counter)} · {int(ids.numel())} envs (total "
            f"{env._fh_nan_envs})"
            f" {ids.tolist()[:8]} · clips {_clip_names(env, ids[:8])} · elements made finite: reward {n_r} · per side "
            f"{n_s}"
            " · terminated next step",
            flush=True,
        )


def sanitize_step(env, ret):
    """(B) Make the rewards and observations of the envs cut this step by a non-finite or ABSURD state finite.

    ret = (obs, rew, terminated, truncated, extras); returns the same tuple (modified in place)."""
    on = getattr(env, "_fh_nan_on", None)
    if on is None:
        from dexengine.rl.tasks import hooks

        on = env._fh_nan_on = hooks.floating_nan_reset()
        if on:  # the flag on without the termination term would only pile up flags and never cut: fail loudly
            act = list(getattr(getattr(env, "termination_manager", None), "active_terms", []) or [])
            if "fh_nan_reset" not in act:
                raise RuntimeError(f"[FH_NAN_RESET] on, but the termination term fh_nan_reset is not active ({act})")
            print(f"[FH_NAN_RESET] termination term active — {act}", flush=True)
    if on:
        _fh_nan_mark(env, ret)
    terms = getattr(env, "_absurd_terms", None)
    if not terms or getattr(env, "_absurd_step", None) != int(env.common_step_counter):
        return ret
    m = None
    for bad, ab in terms.values():
        x = bad | ab
        m = x if m is None else (m | x)
    if m is None or not bool(m.any()):
        return ret
    ids = m.nonzero(as_tuple=False).squeeze(-1)
    obs, rew, _term, _trunc, extras = ret
    n_r = _fix_rows(rew, ids)
    n_s = 0
    if isinstance(extras, dict):
        n_s = sum(_fix_rows(extras.get(k), ids) for k in ("reward_right", "reward_left"))
    if isinstance(obs, dict):
        n_o = sum(_fix_rows(v, ids) for v in obs.values())
    else:
        n_o = _fix_rows(obs, ids)
    env._absurd_count = int(getattr(env, "_absurd_count", 0)) + 1
    env._absurd_envs = int(getattr(env, "_absurd_envs", 0)) + int(ids.numel())
    k = env._absurd_count
    if k <= MARK_FIRST or k % 100 == 0:
        why = []
        for key, (bad, ab) in terms.items():
            b = (bad & m).nonzero(as_tuple=False).flatten().tolist()
            a = (ab & m).nonzero(as_tuple=False).flatten().tolist()
            if b:
                why.append(f"{key}:nonfinite{b[:8]}")
            if a:
                why.append(f"{key}:absurd{a[:8]}")
        more = "…" if ids.numel() > 8 else ""
        print(
            f"{MARK} event {k} · step {int(env.common_step_counter)} · {int(ids.numel())} envs (total "
            f"{env._absurd_envs})"
            f" {ids.tolist()[:8]}{more}"
            f" · clips {_clip_names(env, ids[:8])} · cause {' '.join(why) or '-'}"
            f" · elements made finite: reward {n_r} · per side {n_s} · obs {n_o}",
            flush=True,
        )
    return ret
