"""Contact diagnostic of ``--task eval``: a one-env policy rollout that prints a contact-force table, then exits.

``contact_audit`` (DEXENGINE_CONTACT_AUDIT set to 1) logs, per fingertip and
step, the object-filtered force, the net force and the nearest object-surface distance.

It ends the process with ``os._exit`` (``simulation_app.close()`` deadlocks on a started env). Isaac-side: called
from ``eval.run`` after the env is built.
"""

import os
import sys

import torch


def contact_audit(raw_env, wrapped, policy_fn, args_cli, checkpoint_path, _ACTIVE_OBJ_KEY):
    """Per-finger contact audit on env 0 (up to 200 steps or the first episode end).

    With --render_per_clip_videos the caller skips this quick force-only path in favour of the render-path audit,
    which logs the same rollout it saves to video (``_run_per_clip_video_render``).
    """
    import numpy as _np

    side = "right"
    scene = raw_env.scene
    hand = scene["right_hand"] if "right_hand" in scene.keys() else scene["robot"]  # noqa: SIM118  — InteractiveScene has no __contains__
    obj = scene[_ACTIVE_OBJ_KEY]
    from dexengine.rl.tasks.mdp.rewards.contact_base import (
        OBJ_SENSOR_NAMES,
        load_object_surface_points,
        resolve_tip_body_ids,
        tip_to_surface_dist,
    )

    F_TH, cnt_dist, cnt_gate, cnt_netF, cnt_objF, maxobjF, n, names, steps = _count_contacts(
        raw_env,
        wrapped,
        policy_fn,
        args_cli,
        checkpoint_path,
        OBJ_SENSOR_NAMES,
        _np,
        hand,
        load_object_surface_points,
        obj,
        resolve_tip_body_ids,
        scene,
        side,
        tip_to_surface_dist,
    )
    print(f"\n  SUMMARY over {steps} steps (frames where condition holds):")
    for i, nm in enumerate(names):
        print(
            f"    {nm:7}: objF>{F_TH}N: {int(cnt_objF[i]):3d}  netF>{F_TH}N: {int(cnt_netF[i]):3d}  "
            f"nearest<1cm: {int(cnt_dist[i]):3d}  GATE(contact): {int(cnt_gate[i]):3d}  maxObjF={maxobjF[i]:.2f}N"
        )
    print(f"  robot_contact_frac (mean over 5 fingers x {steps} steps) = {cnt_gate.sum() / (n * steps):.4f}")
    print("=" * 78 + "\n")
    sys.stdout.flush()
    os._exit(0)  # hard exit: simulation_app.close() deadlocks on a started env


def _count_contacts(
    raw_env,
    wrapped,
    policy_fn,
    args_cli,
    checkpoint_path,
    OBJ_SENSOR_NAMES,
    _np,
    hand,
    load_object_surface_points,
    obj,
    resolve_tip_body_ids,
    scene,
    side,
    tip_to_surface_dist,
):
    """Roll the policy out and count fingertip contacts per gate (distance, object force, net force)."""
    D_TH, F_TH, _nearest, _netforce, _objforce, n, names, obs = _contact_audit_setup(
        raw_env,
        wrapped,
        args_cli,
        checkpoint_path,
        OBJ_SENSOR_NAMES,
        hand,
        load_object_surface_points,
        obj,
        resolve_tip_body_ids,
        scene,
        side,
        tip_to_surface_dist,
    )
    cnt_objF = _np.zeros(n)
    cnt_netF = _np.zeros(n)
    cnt_dist = _np.zeros(n)
    cnt_gate = _np.zeros(n)
    maxobjF = _np.zeros(n)
    steps = 0
    print(f"  {'step':>4} " + " ".join(f"{nm[:3]}:objF/netF/d_cm/gate" for nm in names))
    for t in range(200):
        with torch.no_grad():
            act = policy_fn(obs)
        step_out = wrapped.step(act)
        obs = step_out[0]
        term = step_out[2]
        trunc = step_out[3]
        ofo = _objforce()[0].cpu().numpy()
        nfo = _netforce()[0].cpu().numpy()
        dst = _nearest()[0].cpu().numpy()
        gate = ((_np.maximum(ofo, nfo) > F_TH) & (dst < D_TH)).astype(float)
        cnt_objF += ofo > F_TH
        cnt_netF += nfo > F_TH
        cnt_dist += dst < D_TH
        cnt_gate += gate
        maxobjF = _np.maximum(maxobjF, ofo)
        steps += 1
        if t % 8 == 0 or gate.any():
            cells = " ".join(f"{ofo[i]:5.1f}/{nfo[i]:5.1f}/{dst[i] * 100:5.1f}/{int(gate[i])}" for i in range(n))
            print(f"  {t:>4} {cells}")
        if bool(term[0]) or bool(trunc[0]):
            print(f"  -> env0 episode end at step {t}")
            break
    return F_TH, cnt_dist, cnt_gate, cnt_netF, cnt_objF, maxobjF, n, names, steps


def _contact_audit_setup(
    raw_env,
    wrapped,
    args_cli,
    checkpoint_path,
    OBJ_SENSOR_NAMES,
    hand,
    load_object_surface_points,
    obj,
    resolve_tip_body_ids,
    scene,
    side,
    tip_to_surface_dist,
):
    """Tip and sensor ids, the force / distance probes and the first observation of the contact audit."""
    tip_ids = resolve_tip_body_ids(hand, side)
    surf = load_object_surface_points(args_cli.object_mesh_dir, raw_env.device)
    hbn = hand.data.body_names
    cs = scene["right_hand_contact"]
    sbn = cs.body_names
    sensor_tip_ids = [sbn.index(hbn[i]) if hbn[i] in sbn else -1 for i in tip_ids]
    obj_sensors = [scene[n] for n in OBJ_SENSOR_NAMES[side]]
    names = ["thumb", "index", "middle", "ring", "pinky"]
    F_TH, D_TH = 0.5, 0.01

    def _objforce():
        cols = []
        for s in obj_sensors:
            fm = s.data.force_matrix_w
            cols.append(
                torch.zeros(raw_env.num_envs, device=raw_env.device) if fm is None else fm.sum(dim=(1, 2)).norm(dim=-1)
            )
        return torch.stack(cols, dim=1)  # (N,5)

    def _netforce():
        net = cs.data.net_forces_w
        return torch.stack(
            [
                torch.norm(net[:, sid, :], dim=-1) if sid >= 0 else torch.zeros(raw_env.num_envs, device=raw_env.device)
                for sid in sensor_tip_ids
            ],
            dim=1,
        )

    def _nearest():
        tips = hand.data.body_link_pos_w[:, tip_ids, :]
        return tip_to_surface_dist(tips, obj, surf)

    print("\n" + "=" * 78)
    print("CONTACT AUDIT (DEXENGINE_CONTACT_AUDIT=1) — per-finger, env0, policy rollout")
    print(f"  ckpt={checkpoint_path}")
    print(f"  gate: objORnet force >{F_TH}N AND nearest-surface <{D_TH * 100:.0f}cm")
    reset_out = wrapped.reset()
    obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
    n = len(names)
    return D_TH, F_TH, _nearest, _netforce, _objforce, n, names, obs
