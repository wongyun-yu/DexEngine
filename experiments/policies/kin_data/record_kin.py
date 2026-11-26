"""E5 kinematic baseline — a **kinematic record** (no physics, no policy) in place of the stage-1 physics record.

Entry = the record session stage record_kin (kin_data/plugin.py) with --dump <canonical kinematic replay render dump
(RENDER_ONLY=replay)>. In the same dump and scene (the scene of the render stage), per clip and frame, the canonical
reset functions (reset_hands/arm/object_to_trajectory, extra slots reset_extra_objects — the functions of RSI and the
canonical kinematic replay) write the robot and objects onto the reference trajectory (tree + arm IK cache of every
frame); the joint targets are set to those joint values (target = state), and the canonical recorder (TrajRecorder
under the record chain's recorder class) captures one row at a time. File layout, layout version and count identities
belong to the canonical recorder. What the kinematic record changes (KinRecorder, this module only):
  action channel = 0 (no policy) · force channels (force_total, force_object, force_partner, force_object_partner) = 0
  · contact_sep = NaN (no contact points measured) · fingertip sensor oracle and partner split oracle not measured (0
  physics steps) · meta.env.collect.kinematic = {rule, what was set to 0 / NaN}.
Row convention: row k state = q_kin[k] and action.joint_target.robot[k] = q_kin[k] — the convention of our data, 'the
  target of row k produced state k' (the converter pairs observation j with target[i(j)+1], so the next frame = motion).
Time: s_0 = right after writing frame 0 · s_k = frame k · length = tree length (human time, no retiming).
Output: env COLLECT_KIN_OUT (.npz — every session clip, one chosen trajectory per clip).
Markers [COLLECT_KIN] …, the canonical recorder's [TRAJ_REC] DONE … (count identity).
"""

from __future__ import annotations

import os
import time

MARK = "[COLLECT_KIN]"
OUT_ENV = "COLLECT_KIN_OUT"  # record file (.npz) — set by the record_kin session stage's --out
# optional (default empty = target = state) · the only value 'spider_cmd' (tree 0/spider_cmd.npz —
CMD_ENV = "KIN_CMD_TARGETS"
# owner experiments/policies/spider_arm/data.py); other values die. On: joint_target of rows k ≥ 1 =
# the command (fingers = SPIDER ctrl, arm = IK of the commanded wrist), row 0 = state · marker [KIN_CMD]
CMD_VALUES = ("spider_cmd",)
RULE = "e5_kinematic_replay_v1"


FINAL_MARK = f"{MARK} no oracle (0 physics steps)"  # printed by KinRecorder.finalize — read by the session check


def log_problems(text: str) -> list[str]:
    """Marker check of a record_kin session log — did the kinematic recorder's finalize run (render-stage markers do
    not exist here)?
    """
    return [] if FINAL_MARK in text else [f"kinematic record marker '{FINAL_MARK}' missing"]


def kin_recorder_class(base):
    """Canonical recorder base (TrajRecorder) → record chain recorder → kinematic recorder (physics channels 0, no
    oracle).
    """
    from dexengine.record.rollout import recorder as REC

    Coll = REC.collect_recorder_class(base)

    class KinRecorder(Coll):
        def _row(self):
            # channels set to 0 / NaN (force, contact, action) are not read — they are built from the declared width.
            # Reading them looks up caches the physics-step hook fills (a KeyError on partner.cache — the kinematic
            # record has 0 steps).
            import torch

            b = base._row(self)
            vals = []
            for c in self._chs:
                k = REC.contact_kind(c)
                if k is not None or c["kind"] == "action":
                    fill = float("nan") if k == REC.KIND_SEP else 0.0
                    vals.append(torch.full((b.shape[0], int(c["width"])), fill, dtype=torch.float32, device=b.device))
                else:
                    vals.append(c["read"]())
            return torch.cat([b, *vals], dim=-1)

        def finalize(self, aggregate):
            col = self.meta_in.setdefault("collect", {})
            col["kinematic"] = {
                "rule": RULE,
                "owner": "experiments/policies/kin_data/record_kin.py",
                "zero": ["action", REC.KIND_TOTAL, REC.KIND_OBJECT, REC.KIND_PARTNER, REC.KIND_OBJ_PARTNER],
                "nan": [REC.KIND_SEP],
                "joint_target": "= joint_pos (target = state)",
                "oracles": "none (no physics step)",
            }
            print(f"{FINAL_MARK} · force 0 · contact points NaN · action 0 · joint target = state", flush=True)
            return super(Coll, self).finalize(
                aggregate
            )  # skip the record recorder's oracle gate and its rank-key layer → canonical recorder

    return KinRecorder


def _recorder(raw_env, clip_names, out: str, args_cli):
    """Kinematic recorder of the session clips (meta = run axis, no checkpoint) and the command-target mode."""
    from dexengine.rl.eval import traj_meta as TRMETA
    from dexengine.rl.eval import traj_record as TRM

    Rec = kin_recorder_class(TRM.TrajRecorder)
    if args_cli is None:
        raise SystemExit(f"{MARK} no args_cli — cannot set the run axis (hand · arm) of the record meta")
    meta = dict(TRMETA.run_meta(args_cli, evals_per_clip=1), ckpt="", kinematic=RULE)  # no policy, no checkpoint
    cmd_mode = os.environ.get(CMD_ENV, "")
    if cmd_mode and cmd_mode not in CMD_VALUES:
        raise SystemExit(f"{MARK} {CMD_ENV}={cmd_mode!r} unknown — {CMD_VALUES}")
    if cmd_mode:
        meta["kin_cmd_targets"] = cmd_mode
    rec = Rec(raw_env, list(clip_names), out, k_keep=1, meta=meta)
    rec.warmup_batches = 0
    return rec, cmd_mode


def _put_frame(raw_env, traj_mgr, robot, eids, objs: list[tuple[str, str]], xslots: bool, f: int) -> None:
    """Write reference frame f onto the robot and objects (canonical reset functions); joint target = state."""
    from isaaclab.managers import SceneEntityCfg

    from dexengine.rl.tasks.mdp.events.reset import reset_extra_objects as RX
    from dexengine.rl.tasks.mdp.terms import (
        reset_arm_to_trajectory as RA,
    )
    from dexengine.rl.tasks.mdp.terms import (
        reset_hands_to_trajectory as RH,
    )
    from dexengine.rl.tasks.mdp.terms import (
        reset_object_to_trajectory as RO,
    )

    traj_mgr.start_frames[eids] = f
    raw_env.episode_length_buf[eids] = 0
    RH(raw_env, eids)
    RA(raw_env, eids)
    for n, s in objs:
        RO(raw_env, eids, object_cfg=SceneEntityCfg(n), side=s)
    if xslots:
        RX(raw_env, eids)
    robot.set_joint_position_target(robot.data.joint_pos[eids].clone(), env_ids=eids)
    raw_env.sim.physics_sim_view.update_articulations_kinematic()


def _cmd_targets(traj_mgr, scene, robot, dev, cmd_mode: str, k: int, clip: str, T: int):
    """KIN_CMD_TARGETS=spider_cmd: the clip tree's command file → (hand joint ids, (T, n) fingers in PhysX order)
    for both hands + arm ids, (T, 7).
    """
    import json
    from pathlib import Path

    import numpy as np
    import torch

    from dexengine.core import qpos_layout as QL

    seq = traj_mgr._loader.sequence_names[k]
    tdir = Path(traj_mgr._loader.data_dir) / seq
    f = tdir / "0" / f"{cmd_mode}.npz"
    if seq != clip:
        raise SystemExit(f"{MARK} {CMD_ENV}: session clip {clip} ≠ loader {seq}")
    if "e5_spider" not in json.loads((tdir / "task_info.json").read_text()):
        # the source of a KIN_VAR_ORIG_D session (variant source, not stored) is a kin tree — target = state (as
        # before) · marker
        print(f"{MARK} [KIN_CMD] {clip} kin tree (no e5_spider) — target = state", flush=True)
        return None
    if not f.is_file():
        raise SystemExit(f"{MARK} {CMD_ENV}: SPIDER tree {clip}: command file {f} missing")
    z = np.load(f)
    hc = np.asarray(z["hand_cmd"], np.float64)
    if len(hc) != T:
        raise SystemExit(f"{MARK} {CMD_ENV}: {clip} command {len(hc)} rows ≠ clip length {T}")
    if not getattr(traj_mgr, "_fingers_only", False):
        raise SystemExit(f"{MARK} {CMD_ENV}: full-robot (fingers_only) scenes only")
    lay = QL.read(str(tdir), nq=hc.shape[1], use_cache=False)
    H = int(lay.hand_qpos_dim)
    parts = []
    sides = [str(x) for x in z["sides"]]  # SPIDER-driven = the tree's active sides; others keep target = state
    if not sides or any(x not in ("right", "left") for x in sides):
        raise SystemExit(f"{MARK} {CMD_ENV}: {clip} command file hands {sides}")
    # the side's articulation = the one the trajectory manager built that side's hand ids on: right = scene "robot",
    # left = scene alias "left_hand" (the evaluation env: robot_left when DEXENGINE_BI_SPLIT_ARTIC=1, else robot)
    if "left_hand" not in scene.articulations:
        raise SystemExit(f"{MARK} {CMD_ENV}: no left_hand alias in the scene — cannot pick the left-hand articulation")
    art_of = {"right": ("robot", robot), "left": ("left_hand", scene.articulations["left_hand"])}
    for side, perm, ids in (
        ("right", traj_mgr._right_perm, traj_mgr._right_hand_ids),
        ("left", traj_mgr._left_perm, traj_mgr._left_hand_ids),
    ):
        if side not in sides:
            continue
        if ids is None or perm is None or len(ids) == 0:
            raise SystemExit(f"{MARK} {CMD_ENV}: {clip} SPIDER hand {side} is not on the scene robot")
        an_, art = art_of[side]
        o = lay.hands[side]
        fing = hc[:, o + 6 : o + H][:, np.asarray(perm.tolist() if hasattr(perm, "tolist") else perm)]
        parts.append((art, list(ids.tolist() if hasattr(ids, "tolist") else ids), fing))
        names = [str(x) for x in z[f"arm_joint_names_{side}"]]
        aid, jn = art.find_joints(names, preserve_order=True)
        if list(jn) != names:
            raise SystemExit(f"{MARK} {CMD_ENV}: arm joints {names} ↔ {an_} {jn}")
        parts.append((art, list(aid), np.asarray(z[f"arm_cmd_{side}"], np.float64)))
    print(
        f"{MARK} [KIN_CMD] {cmd_mode} {clip} · rows {T} · hands {sides} · articulations "
        f"{sorted({art_of[s][0] for s in sides})} · joints {sum(len(i) for _, i, _ in parts)} · other "
        "hands target = state",
        flush=True,
    )  # applied marker
    return [(a_, i, torch.as_tensor(v, dtype=torch.float32, device=dev)) for a_, i, v in parts]


def _put_cmd(parts, f: int, eids) -> None:
    """Overwrite the joint targets of frame f with the command values (one write per articulation)."""
    tg = {}
    for art, ids, v in parts:
        k = id(art)
        if k not in tg:
            tg[k] = (art, art.data.joint_pos_target[eids].clone())
        tg[k][1][:, ids] = v[f][None, :]
    for art, tgt in tg.values():
        art.set_joint_position_target(tgt, env_ids=eids)


def record_kin(
    wrapped, policy_fn, traj_mgr, num_clips, clip_names, raw_env, checkpoint_path, out_dir, *_a, args_cli=None, **_kw
):
    """Write and record the reference state per session clip and frame — in place of the render function
    (the trainer's render seam, dexengine.rl.session_render).
    """
    import torch

    from dexengine.record.session.runner import finish

    out = os.environ.get(OUT_ENV, "")
    if not out:
        raise SystemExit(f"{MARK} {OUT_ENV} is empty — no record file location")
    if int(raw_env.num_envs) != 1:
        raise SystemExit(f"{MARK} env {raw_env.num_envs} — the kinematic record runs only with env 1 (render dump)")
    scene, dev = raw_env.scene, raw_env.device
    eids = torch.tensor([0], device=dev)
    keys = set(scene.keys())
    robot = scene["robot"]
    objs = [(n, s) for n, s in (("object_right", "right"), ("object_left", "left")) if n in keys]
    xslots = bool(getattr(raw_env, "_extra_slots", None)) or any(k.startswith("extra_obj") for k in keys)
    rec, cmd_mode = _recorder(raw_env, clip_names, out, args_cli)
    state = {"k": 0}

    def _pinned(env_ids):
        traj_mgr.clip_ids[env_ids] = state["k"]
        traj_mgr.clip_lengths[env_ids] = traj_mgr._loader.get_clip_lengths(traj_mgr.clip_ids[env_ids])
        traj_mgr.start_frames[env_ids] = 0

    traj_mgr.reset = _pinned

    t0 = time.perf_counter()
    nfr = 0
    for k, clip in enumerate(clip_names):
        state["k"] = k
        wrapped._reset_once = True
        wrapped.reset()
        T = int(traj_mgr.clip_lengths[0].item())
        parts = _cmd_targets(traj_mgr, scene, robot, dev, cmd_mode, k, clip, T) if cmd_mode else None
        _put_frame(raw_env, traj_mgr, robot, eids, objs, xslots, 0)
        rec.begin_batch(k, torch.tensor([k], device=dev), 1)
        done = torch.zeros(1, dtype=torch.bool, device=dev)
        no = torch.zeros(1, dtype=torch.bool, device=dev)
        for f in range(1, T):
            _put_frame(raw_env, traj_mgr, robot, eids, objs, xslots, f)
            if parts is not None:
                _put_cmd(parts, f, eids)
            rec.after_step(f, no, no, done)
        rec.after_step(
            T, no, torch.ones(1, dtype=torch.bool, device=dev), done
        )  # length = T (the last capture is dropped)
        rec.end_batch()
        nfr += T
        print(
            f"{MARK} {k + 1}/{len(clip_names)} {clip} · frames {T} · objects {[n for n, _ in objs]} · extra slots "
            f"{'on' if xslots else 'off'}",
            flush=True,
        )
    K = len(clip_names)
    agg = {"num_episodes": K, "per_clip": [{"completed": 1} for _ in range(K)], "completion_rate": 1.0}
    rec.finalize(agg)
    print(f"{MARK} DONE clips {K} · frames {nfr} · {time.perf_counter() - t0:.1f} s → {out}", flush=True)
    finish(True, wrapped)
