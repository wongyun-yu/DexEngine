"""Joint targets and the evaluation arm controller: absolute targets in place of the action term, own-target lead, env
patches (terminations, render, debug markers, frame-0 pins, resting extras)."""

from __future__ import annotations

import math
import os

from dexengine.rl.tasks import switches
from experiments.sim_eval.loop.common import CTRLS, MARK, _say
from experiments.sim_eval.session_cfg import flag01 as _flag01
from experiments.sim_eval.session_cfg import value as _cfg


class Targets:
    """Home of the absolute joint targets — articulation name → (asset, [J] tensor); replaces the action term's
    apply_actions. Scene articulations = robot (+ robot_left when present). Data columns are bound by **name** — every
    scene joint must be in the data (else dies); data columns the scene lacks (the idle arm of a one-hand scene) are
    unused (counted in the marker). Controller ctrl (CTRLS): with interp, apply moves from the previous to this target
    by
    s/decimation when it is called per substep (s = 1 … D)."""

    def __init__(self, raw_env, head_pin: dict, ctrl: str = "hold"):
        import torch

        if ctrl not in CTRLS:
            raise SystemExit(f"{MARK} controller {ctrl!r} — one of {CTRLS}")
        sc = raw_env.scene
        self.arts = {k: sc.articulations[k] for k in ("robot", "robot_left") if k in sc.articulations}
        if "robot" not in self.arts:
            raise SystemExit(f"{MARK} the scene has no articulation robot")
        self.names = {k: list(a.joint_names) for k, a in self.arts.items()}
        self.t = {k: a.data.joint_pos_target[0:1].clone() for k, a in self.arts.items()}
        self.last = {k: v.clone() for k, v in self.t.items()}
        self.prev = {k: v.clone() for k, v in self.t.items()}
        self.sub = dict.fromkeys(self.arts, 10**9)
        self.head = dict(head_pin.items())
        self.head_ids = {n: self.names["robot"].index(n) for n in self.head}
        self.dev = raw_env.device
        self.order = [(k, n) for k in self.arts for n in self.names[k]]
        self.ctrl, self.D = ctrl, int(raw_env.cfg.decimation)
        self.cols = None
        self.torch = torch
        self.lead = {}  # articulation → (joint indices, dq [1, n]) — only with arm_lead=self (empty dict otherwise)

    def width(self) -> int:
        return len(self.order)

    def bind(self, data_names: list[str]) -> str:
        """Data column names → scene-order indices. A scene joint missing from the data dies."""
        miss = [n for _k, n in self.order if n not in data_names]
        if miss:
            raise SystemExit(f"{MARK} {len(miss)} scene joints are not in the data columns {miss[:4]}")
        self.cols = [data_names.index(n) for _k, n in self.order]
        unused = len(data_names) - len(self.cols)
        return (
            f"data columns {len(data_names)}, scene uses {len(self.cols)} (unused {unused}) · articulations "
            f"{list(self.arts)}"
        )

    def _scene_vec(self, v, scene_order: bool):
        torch = self.torch
        v = torch.as_tensor(v, dtype=torch.float32, device=self.dev).reshape(1, -1)
        if scene_order:
            if v.shape[1] != self.width():
                raise SystemExit(f"{MARK} scene-order target width {v.shape[1]} ≠ {self.width()}")
        else:
            if self.cols is None:
                raise SystemExit(f"{MARK} set_flat before Targets.bind")
            v = v[:, self.cols]
        return v

    def art_vec(self, v, scene_order: bool = False) -> dict:
        """Target vector → {articulation: [1, J]} (not written)."""
        v, out, lo = self._scene_vec(v, scene_order), {}, 0
        for k in self.arts:
            n = len(self.names[k])
            out[k] = v[:, lo : lo + n]
            lo += n
        return out

    def set_flat(self, v, scene_order: bool = False) -> None:
        """One target → per articulation. Default = data-column order (the bound names); scene_order=True = scene order
        (self.order — vectors the policy adapter builds by scene names). The head is pinned. The interp start = the
        last
        written target."""
        v = self._scene_vec(v, scene_order)
        lo = 0
        for k in self.arts:
            n = len(self.names[k])
            self.prev[k] = self.last[k].clone()
            self.t[k] = v[:, lo : lo + n].clone()
            self.sub[k] = 0
            lo += n
        for n, val in self.head.items():
            self.t["robot"][0, self.head_ids[n]] = float(val)

    def flat(self):
        return self.torch.cat([self.t[k] for k in self.arts], dim=1)

    def joint_pos_flat(self):
        return self.torch.cat([self.arts[k].data.joint_pos[0:1] for k in self.arts], 1)

    def apply(self, key: str) -> None:
        if self.ctrl == "interp" and self.sub[key] < self.D:
            self.sub[key] += 1
            w = self.sub[key] / self.D
            cur = self.prev[key] + (self.t[key] - self.prev[key]) * w
        else:
            cur = self.t[key]
        self.last[key] = cur
        if key in self.lead:  # the lead goes to the write only — the interp start (last) is the target before the lead
            ids, dq = self.lead[key]
            cur = cur.clone()
            cur[:, ids] += dq
        self.arts[key].set_joint_position_target(cur)


#: Arm controller variants of the evaluation (default = the canonical action term). This table is the one owner.
#: arm_lead=self — on top of the canonical feed-forward with velocity scale 0 (DEXENGINE_ARM_FF_VEL_SCALE=0, gravity
#: compensation only), add the centred-difference velocity of the policy's own target plan × Kd/Kp on the same joints
#: (non-privileged).
ARM_LEAD_W = 13  # centred-difference half width (rows · 50 Hz) — the default width of DEXENGINE_ARM_FF_SMOOTH


def arm_ctrl_label() -> str:
    """Arm controller variant name (marker, table column); an unknown combination dies.
    The evaluation profile must be official: a session without it would run the env default controller (reference
    lead on top of the stored labels, i.e. a double lead).

    Called by the reset marker (_phys_marker) of every session."""
    if _cfg("profile") != "official":
        raise SystemExit(f"{MARK} profile={_cfg('profile')!r} — official only (set by run_scene)")
    ff, vs = switches.text("DEXENGINE_ARM_FF", "1"), switches.text("DEXENGINE_ARM_FF_VEL_SCALE", "1.0")
    ld = _cfg("arm_lead", "")
    mode = switches.text("DEXENGINE_ARM_FF_MODE", "lead")
    if mode != "lead":
        raise SystemExit(f"{MARK} DEXENGINE_ARM_FF_MODE={mode!r} — the evaluation knows lead only")
    if ld not in ("", "self"):
        raise SystemExit(f"{MARK} arm_lead={ld!r} — '' or self")
    if ff == "0":
        if ld:
            raise SystemExit(f"{MARK} arm_lead=self needs DEXENGINE_ARM_FF=1 (gravity compensation)")
        return "ga_off"
    if float(vs) == 0.0:
        return "da_grav_selflead" if ld else "na_grav"
    if ld:
        raise SystemExit(
            f"{MARK} arm_lead=self with DEXENGINE_ARM_FF_VEL_SCALE={vs} — overlaps the reference lead (must be 0)"
        )
    return "hyeon_reflead"


def arm_lead_setup(raw_env, tg: Targets) -> dict | None:
    """With arm_lead=self → (articulation, joint indices, Kd/Kp) where the RL controller that generated the paper's data
    applied the lead: the left term of the split articulation took none. The RL controller now leads both arms; this
    keeps the rule the paper's evaluation ran with.
    """
    if _cfg("arm_lead", "") != "self":
        return None
    import torch

    from dexengine.rl.tasks.scene import split_artic as SA

    am, out = raw_env.action_manager, {}
    for name in am.active_terms:
        term = am.get_term(name)
        if not getattr(term, "_arm_ff", False):
            continue
        side = getattr(term, "_side_rb", "right")
        if SA.enabled() and side == "left":  # the paper evaluation's rule (see the docstring)
            continue
        # same index source as the canonical code (before the first feed-forward call: the env's
        # _arm_ik_joint_ids_<side>, else the old right-side name)
        ids = list(
            getattr(term, "_ff_vel_ids", None)
            or getattr(raw_env, f"_arm_ik_joint_ids_{side}", None)
            or getattr(raw_env, "_arm_ik_joint_ids_right", None)
            or []
        )
        if not ids:
            continue
        asset = term._inner._asset
        key = next((k for k, a in tg.arts.items() if a is asset), None)
        if key is None:
            raise SystemExit(f"{MARK} the asset of lead term {name} is not a scene articulation")
        st = asset.root_physx_view.get_dof_stiffnesses()[0, ids].to(raw_env.device)
        dp = asset.root_physx_view.get_dof_dampings()[0, ids].to(raw_env.device)
        out[key] = (
            torch.as_tensor(ids, dtype=torch.long, device=raw_env.device),
            (dp / st.clamp(min=1e-6)).reshape(1, -1),
        )
    if not out:
        raise SystemExit(f"{MARK} arm_lead=self but no action term to apply the lead to")
    _say(
        "[E5_ARM_LEAD] own-target lead: "
        + " · ".join(f"{k} joints {len(v[0])} Kd/Kp mean {float(v[1].mean()):.4f} s" for k, v in out.items())
        + f" · half width {ARM_LEAD_W} rows"
    )
    return out


def arm_lead_update(tg: Targets, setup: dict, sched, row: int, step_dt: float, scene_order: bool) -> None:
    """Lead at row `row` = Kd/Kp × (planned target(row+W) − target(row−W)) / (actual row span × step_dt); outside
    the plan, the plan's end value (zero order).
    """
    if not sched.rows:
        return
    r0 = sched.rows[0][0]
    lo, hi = max(r0, row - ARM_LEAD_W), row + ARM_LEAD_W
    a, b = sched.target_for_row(lo), sched.target_for_row(hi)
    if a is None or b is None or hi <= lo:
        return
    va, vb = tg.art_vec(a, scene_order), tg.art_vec(b, scene_order)
    for k, (ids, kdkp) in setup.items():
        tg.lead[k] = (ids, kdkp * (vb[k][:, ids] - va[k][:, ids]) / ((hi - lo) * step_dt))


def _patch_env(raw_env, tg: Targets, absolute: bool = True, apply_fn=None) -> dict:
    """Action term apply_actions → absolute targets (absolute; off for action replay = the canonical action term) ·
    terminations off · no render inside step. Returns the values to restore."""
    import torch

    am = raw_env.action_manager
    saved = {}
    for name in am.active_terms if absolute else []:
        term = am.get_term(name)
        inner = getattr(term, "_inner", None)
        tgt = inner if inner is not None else term
        asset = getattr(tgt, "_asset", None)
        key = next((k for k, a in tg.arts.items() if a is asset), None)
        if key is None:
            raise SystemExit(
                f"{MARK} the asset of action term {name} is not robot/robot_left — cannot set absolute targets"
            )
        saved[f"apply:{name}"] = (tgt, tgt.apply_actions)
        tgt.apply_actions = (lambda k=key: tg.apply(k)) if apply_fn is None else (lambda k=key: apply_fn(k))
    tm = raw_env.termination_manager
    saved["term"] = tm.compute
    zero = torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)

    def _no_term():
        tm._truncated_buf = zero.clone() if hasattr(tm, "_truncated_buf") else None
        tm._terminated_buf = zero.clone() if hasattr(tm, "_terminated_buf") else None
        return zero.clone()

    tm.compute = _no_term
    saved["render_interval"] = raw_env.cfg.sim.render_interval
    raw_env.cfg.sim.render_interval = 10**9
    _patch_nonpriv(raw_env, saved)
    _say("[E5_MARKERS] " + _no_markers(raw_env))
    what = (
        f"absolute joint targets {tg.width()}"
        if absolute
        else "canonical (action replay — IK, lead and gravity per substep)"
    )
    _say(
        f"patch: action terms {list(am.active_terms)} → {what} · terminations off · no render in step "
        f"(render_interval {saved['render_interval']} → 1e9)"
    )
    return saved


MARKER_ATTRS = (
    "_marker_wrist_current_frame",
    "_marker_wrist_target_frame",
    "_marker_obj_current_frame",
    "_marker_obj_target_frame",
    "_marker_wrist_current_frame_left",
    "_marker_wrist_target_frame_left",
    "_marker_obj_current_frame_left",
    "_marker_obj_target_frame_left",
)


def _no_markers(raw_env) -> str:
    """Remove the env debug markers (wrist, object reference (GT) pose and current pose axes — created late by the eval
    env on the first step and updated every step) from both eye images, as the canonical render does (markers are never
    drawn). A harness that steps physics would otherwise create them and draw the reference pose (the answer) in the
    policy's view. If not created yet, mark them attempted-not-ready so they are never built; if they exist, stop their
    update (ready False) and hide them."""
    if not getattr(raw_env, "_debug_markers_attempted", False):
        raw_env._debug_markers_attempted, raw_env._debug_markers_ready = True, False
        return "debug markers not created (before the first step)"
    raw_env._debug_markers_ready = False
    n = 0
    for mn in MARKER_ATTRS:
        m = getattr(raw_env, mn, None)
        if m is not None:
            m.set_visibility(False)
            n += 1
    return f"{n} debug markers hidden · updates stopped"


def _markers_visible(raw_env) -> list[str]:
    """Names of visible debug markers (any → invalid): the update came back or hiding failed."""
    out = [mn for mn in MARKER_ATTRS if getattr(raw_env, mn, None) is not None and getattr(raw_env, mn).is_visible()]
    if getattr(raw_env, "_debug_markers_ready", False):
        out.append("_debug_markers_ready")
    return out


def _extra_attach(raw_env, xs) -> list:
    """Extra slots resting on a main object → [(slot k, main object asset, relative position (1,3), relative quaternion
    (1,4) wxyz)]; rule owner extra_attach.support_pairs (tree frame-0 mesh support). Slot ↔ extra name =
    extra_slots.slot_rows (qpos column order = slot order). The relative pose uses the tree-frame values (the scene
    transform moves main object and extra by the same rigid transform); it is re-measured from the env's frame-0 slot
    and
    main-object reference poses and dies above 1 mm / 0.5° (wrong slot pairing). Marker [EXTRA_ATTACH]."""

    import numpy as np
    import torch

    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    from dexengine.data.objects import extra_slots as ES
    from experiments.sim_eval import extra_attach as XA

    tm = raw_env.trajectory_manager
    names = list(tm._loader.sequence_names)
    if raw_env.num_envs != 1 or len(names) != 1:
        raise SystemExit(
            f"{MARK} {XA.MARK} an evaluation session has one env and one clip — env {raw_env.num_envs} · "
            f"clips {len(names)}"
        )
    td = os.path.join(str(tm._loader.data_dir), names[0])
    pairs = XA.support_pairs(td)
    if not pairs:
        _say(f"{XA.MARK} none n_extra={len(XA.extra_names(td))} · {names[0]}")
        return []
    rows, _routed = ES.slot_rows(td)
    slot_of = {r["name"]: k for k, r in enumerate(rows)}
    ids = xs.all_ids
    f0 = tm.start_frames[ids].clamp(min=0)
    pos, quat = xs._pose(raw_env, ids, f0)
    out, dev = [], raw_env.device
    for h in pairs:
        if h["extra"] not in slot_of:
            raise SystemExit(
                f"{MARK} {XA.MARK} resting extra {h['extra']} is not in the slot table {list(slot_of)} — "
                "no slot to attach"
            )
        k = slot_of[h["extra"]]
        side = {"right_object": "right", "left_object": "left"}.get(h["main"])
        if side is None:
            raise SystemExit(f"{MARK} {XA.MARK} main object name {h['main']} — right_object or left_object only")
        mo = raw_env.scene.rigid_objects[f"object_{side}"]
        mp, mq = tm.object_pose_window(side, f0, 0)
        mp, mq = mp[:, 0] + raw_env.scene.env_origins[ids], mq[:, 0]
        p_env = quat_apply(quat_inv(mq), pos[:, k] - mp)
        q_env = quat_mul(quat_inv(mq), quat[:, k])
        p_rel = torch.tensor(h["p_rel"], dtype=torch.float32, device=dev)[None]
        R = np.asarray(h["R_rel"], dtype=np.float64)
        w = 0.5 * math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2]))
        if w < 1e-3:
            raise SystemExit(f"{MARK} {XA.MARK} relative rotation near 180° — this formula cannot build the quaternion")
        q_rel = torch.tensor(
            [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)],
            dtype=torch.float32,
            device=dev,
        )[None]
        dp = float((p_env - p_rel).norm(dim=-1).max())
        dq = float(torch.rad2deg(2 * torch.arccos((q_env * q_rel).sum(-1).abs().clamp(max=1.0))).max())
        if dp > 1e-3 or dq > 0.5:
            raise SystemExit(
                f"{MARK} {XA.MARK} {h['extra']}→{h['main']}: env frame-0 relative pose ≠ tree value "
                f"(position {dp * 1000:.2f} mm · "
                f"rotation {dq:.2f}°) — wrong slot pairing or transform"
            )
        out.append((k, mo, p_rel, q_rel))
        _say(
            f"{XA.MARK} {h['extra']}→{h['main']} slot {k} · frac={h['frac']} gap_mm={h['gap_mm']} · check "
            f"{dp * 1000:.2f} mm {dq:.2f}° · {names[0]}"
        )
    return out


def _patch_nonpriv(raw_env, saved: dict) -> None:
    """Pin the two reference-following mechanisms to frame 0 (evaluation only; the canonical code is unchanged).
    spawn_f0=1 — the object pose held during the SPAWN_HOLD window (N steps after reset) = reference frame 0
    (canonical: follows frames 0…N−1). extra_f0=1 — the pre-physics teleport pose of the (kinematic) extra slots =
    reference frame 0
    (canonical: the next reference frame — it moves regardless of the policy)."""
    import torch

    tm = raw_env.trajectory_manager
    if _flag01("spawn_f0"):
        orig_pin = raw_env.pin_objects_to_reference

        def pin_f0(ids, *, joints):
            had = "reference_frames" in raw_env.__dict__
            raw_env.reference_frames = lambda i: tm.start_frames[i].clamp(min=0)
            try:
                return orig_pin(ids, joints=joints)
            finally:
                if not had:
                    del raw_env.reference_frames

        raw_env.pin_objects_to_reference = pin_f0
        saved["spawn_f0"] = True
        _say("[E5_SPAWN_F0] objects held during SPAWN_HOLD at the reference frame-0 pose (not following frames 0…N−1)")
    if _flag01("extra_f0"):
        from dexengine.rl.tasks.scene import extra_slots as XS

        xs = XS.attach(raw_env)
        if xs is None:
            _say("[E5_EXTRA_F0] no extra slots — nothing to do")
            _say(
                "[EXTRA_ATTACH] none n_extra=0 (no extra slots)"
            )  # always one line (present/absent) — the run checks look for it
        else:
            att = _extra_attach(raw_env, xs)

            def pre_f0(env, xs=xs, att=att):
                ids = xs.all_ids
                f0 = tm.start_frames[ids].clamp(min=0)
                pos, quat = xs._pose(env, ids, f0)
                if att:  # resting extra = the main object's current pose ∘ the frame-0 relative pose (extra_attach)
                    from isaaclab.utils.math import quat_apply, quat_mul

                    for k, mo, p_rel, q_rel in att:
                        pm, qm = mo.data.root_pos_w[ids], mo.data.root_quat_w[ids]
                        pos[:, k] = pm + quat_apply(qm, p_rel.expand(len(ids), 3))
                        quat[:, k] = quat_mul(qm, q_rel.expand(len(ids), 4))
                xs._tick(env, pos, quat, f0)
                for k, obj in enumerate(xs.objs):
                    obj.write_root_link_pose_to_sim(torch.cat([pos[:, k], quat[:, k]], dim=-1))
                xs.prev, xs.prev_q, xs.prev_f = pos, quat, f0

            xs.pre_physics = pre_f0
            saved["extra_f0"] = xs
            _say(
                f"[E5_EXTRA_F0] pre-physics pose of {len(xs.objs)} extra slots = reference frame 0 "
                f"(reference motion off) · {len(att)} resting ones attached to their main object"
            )


def _unpatch(raw_env, saved: dict) -> None:
    if saved.get("spawn_f0"):
        del raw_env.pin_objects_to_reference
    if saved.get("extra_f0") is not None:
        del saved["extra_f0"].pre_physics
    for k, v in saved.items():
        if k.startswith("apply:"):
            v[0].apply_actions = v[1]
    raw_env.termination_manager.compute = saved["term"]
    raw_env.cfg.sim.render_interval = saved["render_interval"]
