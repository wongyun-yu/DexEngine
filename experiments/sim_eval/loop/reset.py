"""Episode start: state independence between episodes of one session, physics marker, SPIDER start override and the
canonical pinned reset."""

from __future__ import annotations

import json
from pathlib import Path

from dexengine.rl.tasks import switches
from experiments.sim_eval.loop.common import MARK, _say
from experiments.sim_eval.loop.control import Targets, arm_ctrl_label
from experiments.sim_eval.session_cfg import value as _cfg


def reset_velocities(scene) -> dict:
    """Per-entity velocities right after reset (env 0) — {name: (max |linear| m/s, max |angular| rad/s, max |joint|
    rad/s)}.
    Source of the [E5_RESET_VEL] marker that shows whether state leaks between episodes of one session."""
    from dexengine.rl.eval import traj_record as TR

    out = {}
    for e in TR.scene_entities(scene):
        a = (scene.articulations if e["kind"] == "artic" else scene.rigid_objects)[e["name"]]
        d = a.data
        lin = float(d.root_lin_vel_w[0].norm()) if hasattr(d, "root_lin_vel_w") else 0.0
        ang = float(d.root_ang_vel_w[0].norm()) if hasattr(d, "root_ang_vel_w") else 0.0
        jv = float(d.joint_vel[0].abs().max()) if e["kind"] == "artic" and d.joint_vel.shape[-1] else 0.0
        out[e["name"]] = (round(lin, 6), round(ang, 6), round(jv, 6))
    return out


def _park(S) -> int:
    """From the session's second episode on, before the reset: move the rigid bodies to distant separate places (zero
    velocity) and step physics once, so the contact pairs of the previous episode's end (PhysX friction anchors, warm
    start) do not carry into the next episode's first step (without it, episodes 2 and 3 of a multi-episode session
    diverged from row 1 while standalone runs were byte-identical). The first episode is untouched (byte-identical to a
    standalone run). Returns the number of moved bodies."""
    import torch

    from dexengine.rl.eval import traj_record as TR

    sc, env = S.env.scene, S.env
    seen, k = set(), 0
    for name in sorted(sc.rigid_objects):
        a = sc.rigid_objects[name]
        if id(a) in seen or TR.all_obj_body(name):  # static bodies are not moved
            continue
        seen.add(id(a))
        pose = torch.zeros(1, 7, device=env.device)
        pose[:, :3] = sc.env_origins[S.eids] + torch.tensor([30.0 + 3.0 * k, 30.0, 10.0], device=env.device)
        pose[:, 3] = 1.0
        a.write_root_pose_to_sim(pose, env_ids=S.eids)
        a.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=S.eids)
        k += 1
    env.sim.step(render=False)
    sc.update(env.physics_dt)
    return k


#: Env attribute prefixes made independent of the session history (step-ramp state — friction decay done table, base
#: values, gravity ramp)
EPSTATE_ATTR_PREFIX = ("_friction", "_fric_", "_grav")


#: Physics DR reset terms (name prefix) · off = identity ranges (scale 1, centre of mass 0) — nominal properties at
#: every reset
DR_TERM_PREFIX = "physics_dr_"


DR_IDENTITY = {
    "mass_scale_range": (1.0, 1.0),
    "static_friction_scale_range": (1.0, 1.0),
    "dynamic_friction_scale_range": (1.0, 1.0),
    "restitution_scale_range": (1.0, 1.0),
    "offset_range_m": (0.0, 0.0),
    "com_offset_range_m": (0.0, 0.0),
}


def _dr_terms(env) -> list:
    em = env.event_manager
    out = []
    for name, cfg in zip(em._mode_term_names.get("reset", []), em._mode_term_cfgs.get("reset", []), strict=True):
        if name.startswith(DR_TERM_PREFIX):
            out.append((name, cfg))
    return out


def _snap_physics(env, names: list[str]) -> dict:
    """Copy of the object properties DR touches (friction, restitution, mass, centre of mass, inertia) — owners
    physx_props (materials, mass) and the views (com, inertia).
    """
    from dexengine.rl.tasks.mdp import physx_props as PP

    snap = {}
    for n in names:
        v = env.scene[n].root_physx_view
        snap[n] = {
            "material": PP.read(env, n, "material").clone(),
            "mass": PP.read(env, n, "mass").clone(),
            "com": v.get_coms().clone(),
            "inertia": v.get_inertias().clone(),
        }
    return snap


def _restore_physics(env, snap: dict) -> None:
    import torch

    from dexengine.rl.tasks.mdp import physx_props as PP

    for n, d in snap.items():
        v = env.scene[n].root_physx_view
        ids = torch.arange(env.num_envs, device="cpu")
        PP.write(env, n, "material", d["material"].clone(), ids)
        PP.write(env, n, "mass", d["mass"].clone(), ids)
        v.set_coms(d["com"].clone(), ids)
        v.set_inertias(d["inertia"].clone(), ids)


def _episode_start_state(S, n_ep: int) -> None:
    """Env-level state before the reset, made independent of the session history (all off by default). Without it
    physics DR is redrawn at every reset, friction DR multiplies the current value after its ramp, and only the first
    slot ran an incomplete step ramp.
    physics_dr=on (default, untouched) | off (DR reset ranges = identity — nominal properties at every reset).
    warm_counter=N: env.common_step_counter = N before the first episode's reset (after the ramp, as the judge's
    warm-up). episode_seed=s: torch, numpy and random seeds = s before the first episode's reset.
    epstate=restore: snapshot the state before the first reset (rngs, clock, ramp attributes, DR object properties) and
    restore it before every later reset → every slot starts like the first; off (default) re-applies clock and seed per
    episode. Marker [E5_EPSTATE]."""
    import copy
    import random

    import numpy as np
    import torch

    env = S.env
    w, sd = _cfg("warm_counter", ""), _cfg("episode_seed", "")
    dr, rs = _cfg("physics_dr", "on"), _cfg("epstate", "off")
    for k, v, ok in (
        ("E5_WARM_COUNTER", w, w == "" or w.isdigit()),
        ("E5_EPISODE_SEED", sd, sd == "" or sd.isdigit()),
        ("E5_PHYSICS_DR", dr, dr in ("on", "off")),
        ("E5_EPSTATE", rs, rs in ("off", "restore")),
    ):
        if not ok:
            raise SystemExit(f"{MARK} {k}={v!r} — value not accepted")
    if n_ep == 0 and dr == "off":
        terms = _dr_terms(env)
        if not terms:
            raise SystemExit(f"{MARK} physics_dr=off but no {DR_TERM_PREFIX}* reset term")
        for name, cfg in terms:
            hit = [k for k in DR_IDENTITY if k in cfg.params and cfg.params[k] is not None]
            if not hit:
                raise SystemExit(f"{MARK} unknown range arguments of DR term {name} {sorted(cfg.params)}")
            for k in hit:
                cfg.params[k] = DR_IDENTITY[k]
        _say(f"[E5_EPSTATE] physics DR off: {[n for n, _ in terms]} ranges = identity")
    if rs == "off" or n_ep == 0:
        if w:
            env.common_step_counter = int(w)
        if sd:
            torch.manual_seed(int(sd))
            np.random.seed(int(sd))
            random.seed(int(sd))
    if rs == "restore":
        names = sorted({c.params["object_name"] for _n, c in _dr_terms(env) if c.params.get("object_name")})
        if n_ep == 0:
            S._e5_epsnap = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "np": np.random.get_state(),
                "py": random.getstate(),
                "counter": int(env.common_step_counter),
                "attrs": {k: copy.deepcopy(v) for k, v in vars(env).items() if k.startswith(EPSTATE_ATTR_PREFIX)},
                "phys": _snap_physics(env, names),
            }
        else:
            sn = S._e5_epsnap
            for k in [k for k in vars(env) if k.startswith(EPSTATE_ATTR_PREFIX) and k not in sn["attrs"]]:
                delattr(env, k)
            for k, v in sn["attrs"].items():
                setattr(env, k, copy.deepcopy(v))
            env.common_step_counter = sn["counter"]
            _restore_physics(env, sn["phys"])
            torch.set_rng_state(sn["torch"])
            torch.cuda.set_rng_state_all(sn["cuda"])
            np.random.set_state(sn["np"])
            random.setstate(sn["py"])
    _say(
        f"[E5_EPSTATE] episode {n_ep + 1}: restore {rs} · clock {env.common_step_counter} · seed "
        f"{sd or '-'} · physics DR {dr}"
        + (
            f" · objects {sorted(S._e5_epsnap['phys'])} · attributes {sorted(S._e5_epsnap['attrs'])}"
            if rs == "restore" and n_ep == 0
            else ""
        )
    )


def _phys_marker(env) -> str:
    """Post-reset marker: mass, friction (static/dynamic), restitution and centre of mass of the DR objects, max robot
    effort target, clock, arm feed-forward (DEXENGINE_ARM_FF).
    """
    parts = [
        f"clock {int(env.common_step_counter)}",
        f"ARM_FF={switches.text('DEXENGINE_ARM_FF', '1(default)')}",
        f"VEL_SCALE={switches.text('DEXENGINE_ARM_FF_VEL_SCALE', '1.0(default)')}",
        f"E5_ARM_LEAD={_cfg('arm_lead', '-')}",
        f"arm controller {arm_ctrl_label()}",
        f"SPAWN_F0={_cfg('spawn_f0', '0')}",
        f"EXTRA_F0={_cfg('extra_f0', '0')}",
    ]
    for n in sorted({c.params["object_name"] for _n, c in _dr_terms(env) if c.params.get("object_name")}):
        v = env.scene[n].root_physx_view
        m = float(v.get_masses().reshape(env.num_envs, -1)[0].sum())
        mp = v.get_material_properties().reshape(env.num_envs, -1, 3)[0, 0]
        com = v.get_coms().reshape(env.num_envs, -1)[0, :3]
        parts.append(
            f"{n}: mass {m:.5f} kg · friction {float(mp[0]):.4f}/{float(mp[1]):.4f} · restitution {float(mp[2]):.3f} · "
            f"com {[round(float(x), 4) for x in com]}"
        )
    for n, a in env.scene.articulations.items():
        if n.startswith("robot"):
            parts.append(f"{n} max effort target {float(a.data.joint_effort_target[0].abs().max()):.3f}")
    return " · ".join(parts)


START_OVERRIDE_KEY = "start_override"


def _start_override(S, tg: Targets, job: dict, raw_env) -> None:
    """SPIDER rows: SPIDER policies start from SPIDER's own start pose. Off unless the session option start_override
    names a JSON file {scenes: {<variant>.<look>: {source, sides, joints: {joint name: rad}}}} (the SPIDER start files).
    Right after the canonical reset (fixed clip, frame 0, head pinned) the named joints (active sides' arm + fingers
    only; idle side, head and torso untouched) get position = target = the given value, velocity 0. Objects are not
    touched. A scene missing from the file, or a joint the scene's articulations lack, dies.
    Marker per episode: [E5_START] source=...."""
    p = _cfg(START_OVERRIDE_KEY, "")
    if not p:
        return
    import torch

    v = job.get("variant") or {}
    if not v.get("name") or not v.get("look"):
        raise SystemExit(f"{MARK} {START_OVERRIDE_KEY}: job has no variant.name/look — scene key unknown")
    key = f"{v['name']}.{v['look']}"
    d = json.loads(Path(p).read_text())
    rec = (d.get("scenes") or {}).get(key)
    if rec is None:
        raise SystemExit(f"{MARK} {START_OVERRIDE_KEY}={p}: scene {key} missing")
    want = {str(n): float(x) for n, x in rec["joints"].items()}
    if any(n.startswith(("head_", "torso_")) for n in want):
        raise SystemExit(f"{MARK} {START_OVERRIDE_KEY}: head/torso joints are not overridden ({key})")
    hit, dmax = set(), 0.0
    for k, a in tg.arts.items():
        names = tg.names[k]
        ids = [i for i, n in enumerate(names) if n in want]
        if not ids:
            continue
        q = a.data.joint_pos[0:1].clone()
        for i in ids:
            dmax = max(dmax, abs(float(q[0, i]) - want[names[i]]))
            q[0, i] = want[names[i]]
            hit.add(names[i])
        a.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=S.eids)
        a.set_joint_position_target(q, env_ids=S.eids)
    miss = sorted(set(want) - hit)
    if miss:
        raise SystemExit(f"{MARK} {START_OVERRIDE_KEY}: joints not in the scene articulations {miss[:6]} ({key})")
    raw_env.sim.physics_sim_view.update_articulations_kinematic()
    sides = ",".join(rec.get("sides") or [])
    _say(
        f"[E5_START] source={rec['source']} · scene {key} · sides {sides} · joints {len(hit)} · "
        f"max |override − canonical reset| {dmax:.4f} rad · file {p}"
    )


def _reset(S, tg: Targets | None = None) -> None:
    """Canonical reset (fixed clip S.k, start frame 0 — the record Session's pinned reset) → head pinned."""
    S.wrapped._reset_once = True
    S.wrapped.reset()
    S.rig.set_heads(S.rig.pinned)
    v = reset_velocities(S.env.scene)
    top = sorted(v.items(), key=lambda kv: -max(kv[1]))[:4]
    _say(f"[E5_RESET_VEL] post-reset velocities (linear · angular · joint max) {', '.join(f'{n}={x}' for n, x in top)}")
