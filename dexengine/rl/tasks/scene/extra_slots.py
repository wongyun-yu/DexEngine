"""The env side of extra object slots: single owner of the reset writes, the per-policy-step pose writes and the
hand-extra contact reference line.

Slots are built by `env_setup.extra_slots.build_extra_slots` (default on · off = `DEXENGINE_EXTRA_SLOTS=0`); the rule
and slot count are owned by `dexengine/data/objects/extra_slots.py`. Statements (cfg): `extra_object_slots` (K) ·
`extra_slot_valid` ((clip, K) bool) · `extra_slot_clips` (clip names in loader order) · `extra_slot_hidden_z` (the
hidden z of empty slots). Without slots (K 0, no statement) this module does nothing; the default path stays
byte-identical.

Physics: slots = kinematic bodies (gravity off) · collision with hands and arms on · with main objects off (authoring
`collision_authoring.author_extra_collision`). Motion = a teleport to the **next** reference frame pose every policy
step (before physics) (`write_root_link_pose_to_sim`, velocity 0). Kinematic targets (`set_kinematic_targets`) are not
in this build's GPU pipeline (omni.physics.tensors: "GPU Rigid Body View kinematic target setting not implemented",
measured in a canary), so contacts only resolve penetration (relative velocity 0: no friction drags the fingers; the
same class as the main-object SPAWN_HOLD and PARK writes). A reset teleports to the start frame pose.
Observations, rewards, terminations and the judge's completion condition do not read the slots (the policy input
width is unchanged).

Reference line `[EXTRA_SLOTS] contact …`: among slot-steps where the reference moves (valid slot · inter-frame
translation > `MOVING_EPS_M` or rotation > `MOVING_EPS_DEG`), the share with a contact force > `CONTACT_N` on the
slot body (slots collide only with the robot, so that force = robot contact).
Per clip and slot `[EXTRA_TRACE] …`: among the frames where a person held that extra (statement
`cfg.extra_slot_held`, owner `extra_slots.held_intervals`), the robot contact share, and the length distribution of
runs where the extra moved without contact (step = a 50 Hz frame). Only steps where the reference frame really
advanced are counted (judge park PARK and the clip-end hold excluded); not a judge condition.
On the same line `empty_contact_slot_steps` · `empty_max_force` = contact on empty-place bodies (slots absent from
that clip, the stand-in bodies); stand-ins are hidden with no collider, so this must always be 0 (evidence that the
physics is the same whatever the stand-in is).

Empty-place stand-in bodies: slot bodies stand per env without physics replication, so an empty place
also stands some object USD, hidden; that mesh costs memory per env. Owner of the choice = this module's `stand_in`:
the smallest file among this run's slot object USDs (ties by path). Stand-ins are hidden, without a collider,
kinematic. Marker = one `[EXTRA_SLOTS] empty place stand-in body …` line per slot from
`env_setup.extra_slots._extra_slot_assets`.
"""

from __future__ import annotations

import os

import torch

PREFIX = "extra_obj_"  # scene entity name prefix (slot k = f"{PREFIX}{k}")
SENSOR_PREFIX = (
    "extra_contact_"  # contact sensor of slot k = f"{SENSOR_PREFIX}{k}" (a name that does not overlap PREFIX)
)
PRIM_LEAF = "ExtraObj"  # prim = {ENV_REGEX_NS}/ExtraObj{k} · body = …/base_link (object USD bake contract)
MARK = "[EXTRA_SLOTS]"
CONTACT_N = 0.05  # force counted as contact (N), above the contact sensor noise floor
MOVING_EPS_M = 5e-4  # a step where the reference 'moves' (inter-frame translation, m)
MOVING_EPS_DEG = 0.5  # or inter-frame rotation (°)
ATTR = "_extra_slots"  # place of the slot state in env.__dict__
REPORT_EVERY = 250  # reference line interval (policy steps)
STANDIN_DEFAULT = "lightest"  # empty-slot stand-in body: the smallest slot USD of the run


def standin_rule() -> str:
    """The empty-slot stand-in rule (always ``lightest``)."""
    return STANDIN_DEFAULT


def stand_in(usds: list, k: int, rule: str) -> str:
    """Stand-in body USD for empty slot k: the smallest file among every slot USD of the run (ties by path).
    `usds` = per clip, the USD path of each of its slots (None = empty in that clip). Raises when no clip has slot k."""
    if not any(len(r) > k and r[k] is not None for r in usds):
        raise RuntimeError(f"{MARK} no clip has slot {k}: no stand-in body to choose")
    if rule != STANDIN_DEFAULT:
        raise RuntimeError(f"{MARK} unknown stand-in rule {rule!r}")
    return min({str(u) for r in usds for u in r if u is not None}, key=lambda u: (os.path.getsize(u), u))


def render_fill(ids: list, k: int) -> tuple | None:
    """The object that fills slot k's empty places in a one-env render (collection stage 2: the original and its
    placement variants placed on env 0 one after another).
    `ids` = per clip, the identity list of its slots (slot order; identity = (training USD, kind statement), the same
    form as `ExtraSlots.sig`). None without empty places (nothing to fill). When the clips with slot k share one
    identity it is that one: whichever trajectory is placed, the body on env 0 is the same object as that trajectory's
    valid slot. A split fails (one env cannot stand different objects per trajectory).
    On empty-place trajectories that body is at the hidden pose (reset `_pose`, collection replay
    `dexengine.record.render.slots.pad_empty_slots`).
    A None cell = that clip's empty place (axis alignment: the middle empty cells of
    env_setup.extra_slots._extra_slot_axis)."""
    have = [r[k] for r in ids if len(r) > k and r[k] is not None]
    if len(have) == len(ids):
        return None
    got = sorted({tuple(x) for x in have})
    if len(got) != 1:
        raise RuntimeError(
            f"{MARK} slot {k} of a one-env render — the clips with the slot disagree on the object {got} · "
            f"the object to fill empty places cannot be decided"
        )
    return got[0]


def empty_slot_names(cfg, k_clip: int) -> list[str]:
    """Scene entity names of the slots absent from clip k_clip (loader order) per the scene statement
    (`extra_slot_valid`): the bodies at the hidden pose on that trajectory. [] without slots. A clip number outside the
    statement fails."""
    K = int(getattr(cfg, "extra_object_slots", 0) or 0)
    if not K:
        return []
    valid = cfg.extra_slot_valid
    if not 0 <= int(k_clip) < len(valid):
        raise RuntimeError(f"{MARK} clip number {k_clip} — outside the {len(valid)} clips of the slot statement")
    return [f"{PREFIX}{k}" for k, v in enumerate(valid[int(k_clip)]) if not v]


def hidden_root(cfg) -> list[float]:
    """env-local root pose of an empty slot body [x y z | qw qx qy qz]: the same value `_pose` writes for empty places
    (hidden z = the statement `extra_slot_hidden_z` · unit quaternion)."""
    return [0.0, 0.0, float(cfg.extra_slot_hidden_z), 1.0, 0.0, 0.0, 0.0]


def slots_of(env) -> int:
    return int(getattr(env.cfg, "extra_object_slots", 0) or 0)


class ExtraSlots:
    """Slot state of one env with slots on, built by the first `attach(env)` call (statement checks; a mismatch
    fails)."""

    def __init__(self, env):
        K = slots_of(env)
        tm = env.trajectory_manager
        names = [f"{PREFIX}{k}" for k in range(K)]
        scene = sorted(str(k) for k in env.scene.keys() if str(k).startswith(PREFIX))  # noqa: SIM118  — InteractiveScene has no __contains__
        if scene != sorted(names):
            raise RuntimeError(f"{MARK} scene slots {scene} ≠ statement K={K} {names}")
        if int(tm._loader.n_extra_slots) != K:
            raise RuntimeError(
                f"{MARK} loader slots {tm._loader.n_extra_slots} ≠ statement K={K} — the trajectory cfg did "
                "not receive the slot columns"
            )
        clips = list(getattr(env.cfg, "extra_slot_clips", None) or [])
        if clips != list(tm._loader.sequence_names):
            raise RuntimeError(
                f"{MARK} slot table clip order {clips[:3]}… ≠ loader {list(tm._loader.sequence_names)[:3]}… "
                "— the spawned assets (env i -> clip i % C) and the trajectories are different clips"
            )
        dev = env.device
        valid = torch.tensor(env.cfg.extra_slot_valid, dtype=torch.bool, device=dev).reshape(len(clips), K)
        self.K, self.names, self.valid, self.clips = K, names, valid, clips
        held_cfg = self._check_slot_asset_identity(env, K, clips, dev, names, tm)
        print(
            f"{MARK} attached: K={K} · clips {len(clips)} · valid slots {int(valid.sum())} · teleport (next frame "
            f"every "
            f"policy step) · contact threshold {CONTACT_N} N · reference line every {REPORT_EVERY} steps · held span "
            f"statement "
            f"{'present' if held_cfg is not None else 'missing (unknown)'}",
            flush=True,
        )

    def _check_slot_asset_identity(self, env, K, clips, dev, names, tm):
        """Slot asset identity (clip × slot: (USD, collision kind) · empty place None): the asset standing on env i is
        clip i % C's. When another clip is placed on that env (collection render: placement variants of the same
        object), the identities of the valid slots must agree.
        """
        usd, kinds = getattr(env.cfg, "extra_slot_usd", None), getattr(env.cfg, "extra_slot_kinds", None)
        if usd is None or kinds is None or len(usd) != len(clips) or len(kinds) != len(clips):
            raise RuntimeError(
                f"{MARK} slot asset statement (extra_slot_usd · extra_slot_kinds) missing or clip count ≠ "
                f"{len(clips)} — the identity of the assets standing in the env cannot be checked"
            )
        self.sig = [
            tuple((u[k], kd[k]) if v else None for k, v in enumerate(row))
            for u, kd, row in zip(usd, kinds, env.cfg.extra_slot_valid, strict=True)
        ]
        # Identity of the bodies standing in the env (clip × slot) = that clip's identity · empty places None (hidden
        # stand-ins); but with the empty place fill statement of a one-env render (`extra_slot_fill`, writer
        # env_setup.extra_slots._extra_slot_fills, render context only; none means no fill) the body in an empty place =
        # that object.
        fill = getattr(env.cfg, "extra_slot_fill", None)
        if fill is not None and len(fill) != K:
            raise RuntimeError(f"{MARK} empty place fill statement {fill!r} — must have {K} slots")
        self.spawned = [
            tuple(s if s is not None or fill is None or fill[k] is None else tuple(fill[k]) for k, s in enumerate(row))
            for row in self.sig
        ]
        self.hidden_z = float(env.cfg.extra_slot_hidden_z)
        self.objs = [env.scene[n] for n in names]
        self.sensors = [env.scene[f"{SENSOR_PREFIX}{k}"] for k in range(K)]
        self.all_ids = torch.arange(env.num_envs, device=dev)
        self.n_steps = 0
        # held span table (clip, slot, frame); unknown without a statement (old cfg)
        T = int(tm.clip_lengths.max())
        self.held = torch.zeros(len(clips), K, T, dtype=torch.bool, device=dev)
        self.held_side = [["-"] * K for _ in clips]
        held_cfg = getattr(env.cfg, "extra_slot_held", None)
        self.objs_id = getattr(env.cfg, "extra_slot_objs", None) or [["?"] * K for _ in clips]
        for c, row in enumerate(held_cfg or []):
            for k, h in enumerate(row):
                if h is None:
                    # held_side values ("unknown", "none") are written into the collected episode meta
                    # (record.store.meta robot_contact.held_side).
                    self.held_side[c][k] = "unknown"
                    continue
                self.held_side[c][k] = "+".join(sorted(h)) or "none"
                for iv in h.values():
                    for a, b in iv:
                        self.held[c, k, int(a) : int(b) + 1] = True
        # Statistics: global [moving · of which in contact · contact · max force · empty-place (stand-in) contact ·
        # empty-place
        # max force] · clip×slot [held · held in contact · moving · moving without contact] · length distribution of
        # contactless moving runs
        self.stat = torch.zeros(6, device=dev, dtype=torch.float64)
        self.cs = torch.zeros(len(clips) * K, 4, device=dev, dtype=torch.long)
        self.hist = torch.zeros(len(clips) * K, T + 1, device=dev, dtype=torch.long)
        self.run = torch.zeros(env.num_envs, K, device=dev, dtype=torch.long)
        self.prev = self.prev_q = self.prev_f = None
        return held_cfg

    def _check_clip_binding(self, env, ids: torch.Tensor) -> torch.Tensor:
        """Every valid slot of clip ci placed on an env must match the identity of the asset standing in that env (clip
        i % C). Training uses static clip assignment, so always ci = i % C (no pairs to check); the collection render
        places variants of the same object on env 0."""
        ci = env.trajectory_manager.clip_ids[ids]
        si = ids % self.valid.shape[0]
        off = ci != si
        if not bool(off.any()):
            return ci
        bad = [
            (int(e), c, s, k)
            for e, c, s in zip(ids[off].tolist(), ci[off].tolist(), si[off].tolist(), strict=True)
            for k in range(self.K)
            if self.sig[c][k] is not None and self.sig[c][k] != self.spawned[s][k]
        ]
        if bad:
            e, c, s, k = bad[0]
            raise RuntimeError(
                f"{MARK} env {e} -> slot {k} {self.sig[c][k]} of clip {self.clips[c]} ≠ the asset standing "
                f"in that env (clip {self.clips[s]} · spawn assignment i % {self.valid.shape[0]}) "
                f"{self.spawned[s][k]} — the slot asset is another object (mismatches {len(bad)})"
            )
        return ci

    def _pose(self, env, ids: torch.Tensor, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(n, K, 3) world positions · (n, K, 4) wxyz; empty slots at the hidden pose."""
        ci = self._check_clip_binding(env, ids)
        pos, quat = env.trajectory_manager.extra_slot_poses(ids, frames)
        ok = self.valid[ci]  # (n, K)
        pos = torch.where(ok[..., None], pos, torch.tensor([0.0, 0.0, self.hidden_z], device=pos.device))
        quat = torch.where(ok[..., None], quat, torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device))
        return pos + env.scene.env_origins[ids][:, None, :], quat

    def _flush(self, ids: torch.Tensor, ci: torch.Tensor) -> None:
        """Add the ongoing contactless moving runs of ids to the length distribution and zero them."""
        r = self.run[ids]
        end = r > 0
        if bool(end.any()):
            idx = (ci[:, None] * self.K + torch.arange(self.K, device=r.device)[None, :])[end]
            lens = r[end].clamp(max=self.hist.shape[1] - 1)
            self.hist.index_put_((idx, lens), torch.ones_like(lens), accumulate=True)
        self.run[ids] = 0

    def reset(self, env, ids: torch.Tensor) -> None:
        """Reset: teleport to the start frame pose (velocity 0). Start frame = trajectory_manager.start_frames (at reset
        event time episode_length_buf still holds the old value; the same rule as the main object reset
        `reset_object_to_trajectory`)."""
        f0 = env.trajectory_manager.start_frames[ids]
        pos, quat = self._pose(env, ids, f0)
        for k, obj in enumerate(self.objs):
            rs = torch.zeros(len(ids), 13, device=env.device)
            rs[:, 0:3], rs[:, 3:7] = pos[:, k], quat[:, k]
            obj.write_root_state_to_sim(rs, ids)
        self._flush(ids, env.trajectory_manager.clip_ids[ids])
        if (
            self.prev is not None
        ):  # the 'previous' of a reset env = its start pose and frame (other envs' statistics stay)
            self.prev[ids], self.prev_q[ids], self.prev_f[ids] = pos, quat, f0

    def pre_physics(self, env) -> None:
        """Before the policy step's physics: teleport to the next reference frame pose (= the frame rewards and
        terminations see after this step)."""
        ids = self.all_ids
        tm = env.trajectory_manager
        nxt = env.reference_frames(ids) + 1
        nxt = torch.minimum(nxt, tm.clip_lengths[tm.clip_ids[ids].clamp(max=int(tm.clip_lengths.numel()) - 1)] - 1)
        pos, quat = self._pose(env, ids, nxt)
        self._tick(env, pos, quat, nxt)
        for k, obj in enumerate(self.objs):
            obj.write_root_link_pose_to_sim(torch.cat([pos[:, k], quat[:, k]], dim=-1))
        self.prev, self.prev_q, self.prev_f = pos, quat, nxt

    def _tick(self, env, pos: torch.Tensor, quat: torch.Tensor, frames: torch.Tensor) -> None:
        """Accumulate the reference line statistics from the contact sensors of the previous step's physics (previous
        target pose, frame prev_f)."""
        if self.prev is not None:
            f = torch.stack(
                [s.data.net_forces_w.reshape(env.num_envs, -1, 3).norm(dim=-1).amax(dim=-1) for s in self.sensors],
                dim=1,
            )  # (n, K)
            ci = env.trajectory_manager.clip_ids[self.all_ids]
            ok = self.valid[ci]
            adv = ok & (frames != self.prev_f)[:, None]  # only steps where the reference advanced
            dq = (quat * self.prev_q).sum(-1).abs().clamp(max=1.0)
            moving = adv & (
                ((pos - self.prev).norm(dim=-1) > MOVING_EPS_M) | (torch.rad2deg(2 * torch.acos(dq)) > MOVING_EPS_DEG)
            )
            touch = ok & (f > CONTACT_N)
            K = self.K
            kk = torch.arange(K, device=f.device)
            held = adv & self.held[ci[:, None], kk[None, :], self.prev_f.clamp(max=self.held.shape[2] - 1)[:, None]]
            idx = ci[:, None] * K + kk[None, :]
            for j, m in enumerate((held, held & touch, moving, moving & ~touch)):
                self.cs[:, j].index_add_(0, idx[m], torch.ones_like(idx[m]))
            self.stat[0] += moving.sum()
            self.stat[1] += (moving & touch).sum()
            self.stat[2] += touch.sum()
            self.stat[3] = torch.maximum(self.stat[3], torch.where(ok, f, torch.zeros_like(f)).max().double())
            self.stat[4] += (~ok & (f > CONTACT_N)).sum()  # empty-place stand-in bodies: must always be 0
            self.stat[5] = torch.maximum(self.stat[5], torch.where(ok, torch.zeros_like(f), f).max().double())
            run_on = moving & ~touch
            ended = (self.run > 0) & ~run_on & adv
            if bool(ended.any()):
                lens = self.run[ended].clamp(max=self.hist.shape[1] - 1)
                self.hist.index_put_((idx[ended], lens), torch.ones_like(lens), accumulate=True)
            self.run = torch.where(run_on, self.run + 1, torch.where(adv, torch.zeros_like(self.run), self.run))
        self.n_steps += 1
        if self.n_steps % REPORT_EVERY == 0:
            self.report()

    def summary(self) -> dict:
        """{clip: [per slot {slot, obj, held_side, held_steps, held_contact, moving_steps, moving_nocontact}]}: this
        session's accumulated reference values (input of the trajectory record meta extra_slots and the collection meta
        extra_ref_motion)."""
        cs = self.cs.tolist()
        return {
            clip: [
                dict(
                    slot=k,
                    obj=str(self.objs_id[c][k]),
                    held_side=self.held_side[c][k],
                    **dict(
                        zip(
                            ("held_steps", "held_contact", "moving_steps", "moving_nocontact"),
                            (int(x) for x in cs[c * self.K + k]),
                            strict=True,
                        )
                    ),
                )
                for k in range(self.K)
                if bool(self.valid[c, k])
            ]
            for c, clip in enumerate(self.clips)
        }

    def report(self) -> None:
        m, mt, t, fmax, et, efmax = (float(v) for v in self.stat.tolist())
        print(
            f"{MARK} contact step={self.n_steps} moving_slot_steps={int(m)} with_contact={int(mt)} "
            f"frac={(mt / m if m else 0.0):.3f} contact_slot_steps={int(t)} max_force={fmax:.1f}N "
            f"empty_contact_slot_steps={int(et)} empty_max_force={efmax:.1f}N "
            f"(reference line · not a judge condition)",
            flush=True,
        )
        cs, hist = self.cs.tolist(), self.hist.tolist()
        for c, clip in enumerate(self.clips):
            for k in range(self.K):
                if not bool(self.valid[c, k]):
                    continue
                h, ht, mv, mn = cs[c * self.K + k]
                hl = hist[c * self.K + k]
                runs = ",".join(f"{n}:{v}" for n, v in enumerate(hl) if v)
                print(
                    f"[EXTRA_TRACE] step={self.n_steps} clip={clip} slot={k} obj={self.objs_id[c][k]} "
                    f"held_side={self.held_side[c][k]} held_steps={h} held_contact={ht} "
                    f"held_frac={(ht / h if h else float('nan')):.3f} moving_steps={mv} moving_nocontact={mn} "
                    f"nocontact_runs={runs or '-'}",
                    flush=True,
                )


def attach(env) -> ExtraSlots | None:
    """The env's slot state (None without slots, the default path). Built on the first call and kept in env.__dict__."""
    if not slots_of(env):
        return None
    xs = env.__dict__.get(ATTR)
    if xs is None:
        xs = env.__dict__[ATTR] = ExtraSlots(env)
    return xs


def summary_of(env) -> dict | None:
    """Accumulated slot session values (ExtraSlots.summary); None for envs that built no slots (and builds none)."""
    xs = getattr(env, "__dict__", {}).get(ATTR)
    return None if xs is None else xs.summary()


def final_report(env) -> None:
    """One reference line at the end of an eval (end of run_eval_balanced), so judges shorter than the step interval
    (REPORT_EVERY) also leave final values. Does nothing for envs that built no slots (and builds none)."""
    xs = getattr(env, "__dict__", {}).get(ATTR)
    if xs is not None:
        xs.report()


def pre_physics(env) -> None:
    """Pre-physics window of env.step; does nothing without slots (the default path)."""
    xs = attach(env)
    if xs is not None:
        xs.pre_physics(env)


def reset_extra_slots(env, env_ids: torch.Tensor) -> None:
    """Body of the reset event (`events.reset_extra_objects`)."""
    xs = attach(env)
    if xs is None:
        raise RuntimeError(f"{MARK} a reset event is registered but there is no slot statement")
    xs.reset(env, env_ids)
