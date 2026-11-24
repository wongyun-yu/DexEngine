"""Non-finite guard: a run whose weights went NaN stops at the update that made them, and a NaN checkpoint is never
loaded.

One PPO update can turn every policy/value weight and both running-stat preprocessors NaN (a non-finite or
overflowing rollout input poisons the RunningStandardScaler / value targets; the joint grad-norm clip then spreads NaN
from the critic to the actor). Without a check the next env step feeds NaN joint targets to PhysX, the scene slows by
an order of magnitude, the trainer finishes with rc 0 and saves NaN checkpoints, and the judge that loads one fails at
its first step with an error that points at the judge, not at the checkpoint.

Two gates:
  * ``assert_checkpoint_finite(ck, where, path)`` — called by ``assert_checkpoint_obs_dim`` (dexengine.rl.agents) on
    every loaded checkpoint (train resume, the fresh-process judge, eval) and by the dual-policy render path. Marker
    ``[CKPT_FINITE]`` (✓ on pass; RuntimeError on fail).
  * ``check_after_update(...)`` — called by the trainer's post-interaction hook
    after every post_interaction; it checks on the call that closed a rollout (ran the update). Healthy run: one GPU
    reduction per tensor + one stack + one sync per iteration (≈4 ms for ~230 tensors); nothing is
    written and no RNG is drawn. Non-finite: names the first bad tensors, reports which raw rollout tensors (states,
    rewards, per-side rewards) were already non-finite and their largest finite magnitude (input vs update numerics),
    moves a checkpoint skrl wrote in the same call aside (``.nonfinite``) so it is neither judged nor resumed, and
    raises ``[NONFINITE_UPDATE]``. The launcher's auto-resume reacts to memory signatures only, so the run ends with a
    non-zero rc (a training failure).
"""

from __future__ import annotations

import os

import torch

#: checkpoint sections a load restores (the skrl save layout)
CKPT_SECTIONS = ("policy", "value", "state_preprocessor", "value_preprocessor", "optimizer")
#: raw rollout tensors (record_transition writes them unnormalised) — the update overwrites values/returns in place
ROLLOUT_INPUTS = ("states", "rewards", "rewards_right", "rewards_left", "next_states", "actions")


def _walk(obj, prefix: str):
    if torch.is_tensor(obj):
        yield prefix.rstrip("."), obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{prefix}{k}.")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{prefix}{i}.")


def checkpoint_nonfinite(ck: dict) -> tuple[int, list[tuple[str, int, int]]]:
    """(float tensors checked, [(key, non-finite count, numel)]) over CKPT_SECTIONS of a loaded skrl checkpoint."""
    n, bad = 0, []
    for sec in CKPT_SECTIONS:
        for k, v in _walk(ck.get(sec), f"{sec}.") if isinstance(ck, dict) else ():
            if v.is_floating_point():
                n += 1
                nb = int((~torch.isfinite(v)).sum())
                if nb:
                    bad.append((k, nb, v.numel()))
    return n, bad


def assert_checkpoint_finite(ck: dict, where: str, path: str) -> None:
    """Raise unless every float tensor of the checkpoint sections is finite; prints ``[CKPT_FINITE] ... ✓`` on pass."""
    n, bad = checkpoint_nonfinite(ck)
    if n == 0:
        raise RuntimeError(f"[CKPT_FINITE] {where}: no float tensor to check — unknown checkpoint layout: {path}")
    if bad:
        secs = sorted({k.split(".")[0] for k, _, _ in bad})
        raise RuntimeError(
            f"[CKPT_FINITE] {where}: non-finite (NaN/Inf) tensors {len(bad)}/{n} — sections {secs} · first {bad[:3]} — "
            f"a diverged checkpoint. Not loaded for judge, resume or render — use the run's last finite checkpoint: "
            f"{path}"
        )
    print(f"  [CKPT_FINITE] {where}: {n} float tensors finite ✓", flush=True)


def _agent_tensors(agent):
    """(name, tensor) of the policy/value parameters and the preprocessors' running statistics."""
    seen = set()
    for role in ("policy", "value"):
        mod = getattr(agent, role, None)
        if mod is None or id(mod) in seen:
            continue
        seen.add(id(mod))
        for k, p in mod.named_parameters():
            yield f"{role}.{k}", p
    for role in ("_state_preprocessor", "_value_preprocessor"):
        pre = getattr(agent, role, None)
        for attr in ("running_mean", "running_variance"):
            t = getattr(pre, attr, None)
            if torch.is_tensor(t):
                yield f"{role.strip('_')}.{attr}", t


def agent_nonfinite(agent) -> list[tuple[str, int, int]]:
    """[] when every policy/value parameter and preprocessor running stat is finite (one reduction, one sync)."""
    ts = list(_agent_tensors(agent))
    if not ts:
        raise RuntimeError("[NONFINITE_GUARD] the agent has no policy/value parameters — nothing to check")
    dev = ts[0][1].device  # one stack needs one device — .to() is a no-op when the tensors already share it
    if bool(torch.stack([torch.isfinite(t.detach()).all().to(dev) for _, t in ts]).all()):
        return []
    return [
        (k, int((~torch.isfinite(t.detach())).sum()), t.numel())
        for k, t in ts
        if not bool(torch.isfinite(t.detach()).all())
    ]


def rollout_report(agent) -> list[str]:
    """Per raw rollout tensor: non-finite count, the first non-finite (t, env, dim) and max |finite| — tells an input
    that was already non-finite (physics/obs/reward) from an update that made NaN out of finite inputs."""
    mem = getattr(agent, "memory", None)
    names = set(mem.get_tensor_names()) if mem is not None and hasattr(mem, "get_tensor_names") else set()
    out = []  # the texts below are printed inside [NONFINITE_UPDATE] lines and kept as they are
    for name in ROLLOUT_INPUTS:
        if name not in names:
            continue
        x = mem.get_tensor_by_name(name).detach()
        if not x.is_floating_point():
            continue
        fin = torch.isfinite(x)
        nb = int((~fin).sum())
        mx = float(x[fin].abs().max()) if bool(fin.any()) else float("nan")
        first = ""
        if nb:
            first = f" first (t, env, dim)={tuple((~fin).nonzero()[0].tolist())}"
        out.append(f"{name}{tuple(x.shape)} non-finite {nb}{first} · max|finite| {mx:.3g}")
    return out


def check_after_update(agent, timestep: int, rollouts: int, ckpt_path_at) -> None:
    """Call after every post_interaction (see module doc). Checks only on the call that closed a rollout — the one
    that ran the update; ``ckpt_path_at(timestep + 1)`` (the trainer's rule for the checkpoint skrl wrote in that
    call, or None) is asked only when something is non-finite."""
    if (timestep + 1) % rollouts:
        return
    iteration = (timestep + 1) // rollouts
    if not getattr(check_after_update, "_armed", False):
        check_after_update._armed = True
        print(
            f"[NONFINITE_GUARD] on — policy, value and preprocessors checked for finiteness after every update · "
            f"first check iter {iteration}",
            flush=True,
        )
    bad = agent_nonfinite(agent)
    if not bad:
        return
    print(
        f"\n[NONFINITE_UPDATE] iter={iteration}: non-finite tensors after the update {len(bad)} — first {bad[:4]}",
        flush=True,
    )
    for line in rollout_report(agent):
        print(f"[NONFINITE_UPDATE]   rollout input {line}", flush=True)
    saved_ckpt = ckpt_path_at(timestep + 1)
    if saved_ckpt and os.path.exists(saved_ckpt):
        aside = saved_ckpt + ".nonfinite"
        os.replace(saved_ckpt, aside)
        print(
            f"[NONFINITE_UPDATE]   moved the checkpoint this call wrote {os.path.basename(saved_ckpt)} → "
            f"{os.path.basename(aside)} (no [CKPT_SAVED] · not judged or resumed)",
            flush=True,
        )
    raise RuntimeError(
        f"[NONFINITE_UPDATE] iter={iteration}: policy/value/preprocessor became non-finite — "
        "training diverged. Training and checkpoints after this are invalid, so it stops here"
    )
