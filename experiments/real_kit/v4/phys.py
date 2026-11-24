"""v4 approach physics (session hook; only with AUG_V4_APPROACH — without it the augment session does not even read
this module).

Right after the judge batch reset (RSI: arm = IK cache[0], fingers = reference frame 0, joint speeds 0, objects =
reference pose at speed 0) and before the recorder's begin_batch: (1) keep the RSI state (= the join target); (2)
teleport the robot joints to the schedule's first row (active arm = fixed pose rules.BASE_POSE, relaxed fingers, idle
arm = park, head unchanged) at speed 0; (3) step the physics directly, writing PD targets along each variant's approach
schedule (plan.py — active arm route, idle arm fixed at park) — bypassing the policy, action manager, terminations and
episode_length_buf (env.step is not called). Controller = the two shares of the canonical arm feed-forward (gravity
compensation effort, position-channel lead Kd/Kp·q̇_target — dexengine.rl.tasks.mdp.actions.residual_ik's lead); the
fingers get the same lead (PD target + lead). Approach lengths differ per variant, so the ends are aligned
(right-aligned): an env that starts later rewrites the fixed pose after every physics step until its start (so each
episode's t=0 row is exactly the fixed pose — zero spread). Objects are held at the reference pose for the first
SPAWN_HOLD steps only (owner spawn_hold.spawn_hold_steps), free afterwards (as a canonical episode start); the gravity
wrench is applied right before physics (gravity_wrench.apply, as in env.step).
Modes: warmup = the approach only in the warm-up batch (0, dropped by the judge), real batches from the v2 RSI (stitch
       mode) · every = the policy takes over from the post-approach state in every batch (no teleport — only the first
       observation is the one computed at reset) · every_pin = every + objects pinned to the RSI pose (speed 0) during
       the whole approach (rewritten after every physics step), so the policy starts from the v2 object state ·
       every_rsi = diagnostic — after the approach the robot and objects are written back to RSI (tells the residual
       from other approach effects).
Record: <AUG_V4_APPROACH_OUT>/approach_b<batch>.npz — rows (N+1, bs, D) (the recorder's _row columns; rows before K_e
        are discarded), K, start, clip, residual to RSI (arm and finger position rad, speed rad/s, objects mm, °).
        Marker [AUG_V4_APPROACH].
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

MARK = "[AUG_V4_APPROACH]"
MODES = ("warmup", "every", "every_pin", "every_rsi")
_ARM = re.compile(r"^(right|left)_arm_(\d+)$")
_HEAD = re.compile(r"^head_\d+$")


def idle_of(p: dict) -> list[str]:
    return [str(x) for x in p["idle"]] if "idle" in p else []


def from_env():
    p = os.environ.get("AUG_V4_APPROACH", "")
    if not p:
        return None
    mode, out = os.environ.get("AUG_V4_APPROACH_MODE", ""), os.environ.get("AUG_V4_APPROACH_OUT", "")
    if mode not in MODES or not out:
        raise SystemExit(
            f"{MARK} AUG_V4_APPROACH is set but MODE={mode!r}({'|'.join(MODES)}) · OUT={out!r} — both are required"
        )
    return Approach(Path(p), mode, Path(out))


class Approach:
    def __init__(self, plan_json: Path, mode: str, out: Path):
        from experiments.real_kit.v4 import rules as V4

        self.V4, self.mode, self.out = V4, mode, out
        out.mkdir(parents=True, exist_ok=True)
        plan = json.loads(plan_json.read_text())
        self.plan = {}
        for name, rec in plan.items():
            if rec.get("fail"):
                continue
            z = np.load(rec["npz"], allow_pickle=False)
            self.plan[name] = {k: z[k] for k in z.files}
        bases = {str(v["base_name"]) for v in self.plan.values()}
        if len(bases) != 1:
            raise SystemExit(f"{MARK} the schedule has more than one fixed pose {bases}")
        self.base = bases.pop()
        print(
            f"{MARK} on · branch {mode} · variants {len(self.plan)} · fixed pose {self.base} · record {out}", flush=True
        )

    def wants(self, batch_idx: int) -> bool:
        return self.mode != "warmup" or int(batch_idx) == 0

    def _key(self, clip: str) -> str:
        for k in (clip, clip + "_pl_rt"):
            if k in self.plan:
                return k
        raise SystemExit(f"{MARK} clip not in the schedule {clip!r}")

    def run(self, rec, batch_idx: int, assign, bs: int) -> None:
        import torch

        from dexengine.rl.tasks.mdp import gravity_wrench as GW

        env, scene = rec.env, rec.env.scene
        dev = env.device
        t0 = time.time()
        V4 = self.V4
        bs = int(bs)
        ids = torch.arange(bs, device=dev)
        clips = [rec.clip_names[int(c)] for c in assign.detach().cpu().numpy()[:bs]]
        keys = [self._key(c) for c in clips]
        K = np.array([int(self.plan[k]["K"]) for k in keys])
        N = int(K.max())
        start = N - K
        arts = [(e["name"], scene.articulations[e["name"]]) for e in rec.ents if e["kind"] == "artic"]
        rigs = [(e["name"], scene.rigid_objects[e["name"]]) for e in rec.ents if e["kind"] == "rigid"]
        S, info, park_seen = {}, {}, {}
        for (
            an,
            art,
        ) in (
            arts
        ):  # per articulation the schedule end = RSI (else the schedule read another bake — dies per articulation)
            S[an], info[an], d_end = self._art_schedule(an, art, keys, K, start, N, bs, dev, park_seen, batch_idx)
            if d_end > 1e-3:
                raise SystemExit(
                    f"{MARK} {an} schedule end differs from RSI by {d_end:.3g} rad — bake and schedule disagree"
                )
        for (an, side), ps in park_seen.items():
            print(
                f"{V4.PARK_TAG} b{batch_idx} {an} side={side} q={ps['q']} · RSI target max diff {ps['d_tgt']:.2e} · "
                f"RSI state max diff {ps['d_state']:.2e} rad · statement {ps['src']}",
                flush=True,
            )
        obj_rsi = {rn: (r.data.root_link_pose_w[:bs].clone(), r.data.root_state_w[:bs].clone()) for rn, r in rigs}
        from dexengine.rl.tasks.scene import spawn_hold as SH

        hold = int(SH.spawn_hold_steps())
        rows = self._drive(rec, GW, arts, rigs, S, info, obj_rsi, ids, start, N, hold, bs)
        # diagnostic mode: after the approach, write the robot and objects back to the RSI state (speed 0) — if policy
        # success returns, the cause is the join state (residual), otherwise something else the approach left
        # (contacts, sensor history)
        if self.mode == "every_rsi":
            for an, art in arts:
                q = info[an]["q_rsi"]
                art.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
                art.set_joint_position_target(q, env_ids=ids)
            for rn, r in rigs:
                rs = obj_rsi[rn][1].clone()
                rs[:, 7:] = 0.0
                r.write_root_state_to_sim(rs, env_ids=ids)
        res = self._residuals(arts, rigs, info, obj_rsi, bs)
        R = torch.stack(rows).numpy()
        np.savez(
            self.out / f"approach_b{int(batch_idx)}.npz",
            rows=R,
            K=K,
            start=start,
            clips=np.array(clips),
            keys=np.array(keys),
            hold=np.int64(hold),
            base=np.array(self.base),
            mode=np.array(self.mode),
            **{f"res_{k}": v for k, v in res.items()},
        )
        agg = " · ".join(f"{k} max {float(np.max(v)):.4g}" for k, v in res.items())
        print(
            f"{MARK} b{batch_idx} approach N={N} steps (K {K.min()}–{K.max()}) · env {bs} · "
            f"{time.time() - t0:.1f}s · residual "
            f"{agg}",
            flush=True,
        )

    def _art_schedule(self, an, art, keys, K, start, N, bs, dev, park_seen, batch_idx) -> tuple:
        """One articulation's approach schedule (N+1, bs, joints), control info and max schedule end ↔ RSI difference
        (rad).
        Idle-arm park checks accumulate in park_seen (printed as a marker by run)."""
        import torch

        V4 = self.V4
        jn = list(art.joint_names)
        q_rsi = art.data.joint_pos[:bs].clone()
        tgt_rsi = art.data.joint_pos_target[:bs].clone()  # the target the reset wrote (idle-arm park oracle)
        arm = {i: _ARM.match(n) for i, n in enumerate(jn) if _ARM.match(n)}
        head = [i for i, n in enumerate(jn) if _HEAD.match(n)]
        fing = [i for i in range(len(jn)) if i not in arm and i not in head]
        relax = torch.tensor([V4.relaxed_finger(jn[i]) for i in fing], dtype=q_rsi.dtype, device=dev)
        sched = torch.empty((N + 1, bs, len(jn)), dtype=q_rsi.dtype, device=dev)
        sched[:] = q_rsi[None]
        for e in range(bs):
            p = self.plan[keys[e]]
            k_e = int(K[e])
            for i, m in arm.items():
                side = m.group(1)
                if f"q_{side}" not in p:  # active arm = approach route · idle arm = fixed park (plan.py)
                    raise SystemExit(f"{MARK} schedule {keys[e]} has no {side} arm (no straight-line fallback)")
                names = [str(x) for x in p[f"joints_{side}"]]
                col = p[f"q_{side}"][:, names.index(jn[i])].astype(np.float64)
                if (
                    side in idle_of(p)
                    # idle arm = park (one value for the window; the plan statement is checked against the RSI reset
                    # target)
                ):
                    if float(np.ptp(col)) > 0.0:
                        raise SystemExit(f"{MARK} {keys[e]} idle arm {jn[i]} schedule is not fixed at park")
                    d_t = V4.park_check(col[:1], [float(tgt_rsi[e, i])], f"{keys[e]} {an} {jn[i]}")
                    ps = park_seen.setdefault(
                        (an, side), {"q": {}, "d_tgt": 0.0, "d_state": 0.0, "src": str(p[f"park_src_{side}"])}
                    )
                    ps["q"][jn[i]] = round(float(col[0]), 4)
                    ps["d_tgt"] = max(ps["d_tgt"], d_t)
                    ps["d_state"] = max(ps["d_state"], abs(float(q_rsi[e, i]) - float(col[0])))
                v = torch.tensor(col, dtype=q_rsi.dtype, device=dev)
                sched[start[e] :, e, i] = v
                sched[: start[e], e, i] = v[0]
            # fingers: active hand = relax → RSI measured (rules.finger_schedule — the plan's function, whole window)
            # · idle hand = park as is (one RSI-measured value, whole window)
            for sd in ("right", "left"):
                cols = [c for c, i in enumerate(fing) if jn[i].startswith(sd + "_")]
                if not cols:
                    continue
                ii = [fing[c] for c in cols]
                f_end = q_rsi[e, ii].detach().cpu().numpy().astype(np.float64)
                rl = relax[cols].detach().cpu().numpy().astype(np.float64)
                fv = np.repeat(f_end[None], k_e + 1, 0) if sd in idle_of(p) else V4.finger_schedule(k_e, rl, f_end)
                fv = torch.tensor(fv, dtype=q_rsi.dtype, device=dev)
                sched[start[e] :, e, ii] = fv
                sched[: start[e], e, ii] = fv[0]
            bad = [jn[i] for i in fing if not jn[i].startswith(("right_", "left_"))]
            if bad:
                raise SystemExit(f"{MARK} finger joints of unknown side {bad}")
        arm_ids = sorted(arm)
        # lead (Kd/Kp·q̇_target) on arm and fingers — the finger Kd/Kp is 0.25 s, without it they lagged 0.1–0.24
        # rad at the end
        lead = arm_ids + fing
        st = art.root_physx_view.get_dof_stiffnesses().to(dev)[:bs][:, lead]
        dp = art.root_physx_view.get_dof_dampings().to(dev)[:bs][:, lead]
        info_an = {
            "q_rsi": q_rsi,
            "arm_ids": arm_ids,
            "fing": fing,
            "lead": lead,
            "kdkp": dp / st.clamp(min=1e-6),
            "jn": jn,
        }
        # schedule end = RSI (the active arm uses the same IK cache frame 0 — otherwise the schedule read another bake)
        d_end = (sched[-1] - q_rsi).abs().max().item()
        print(
            f"{MARK} b{batch_idx} {an}: schedule end ↔ RSI max {d_end:.2e} rad · arm {len(arm_ids)} · fingers "
            f"{len(fing)} · head {len(head)}",
            flush=True,
        )
        return sched, info_an, d_end

    def _drive(self, rec, GW, arts, rigs, S, info, obj_rsi, ids, start, N, hold, bs) -> list:
        """Teleport to the fixed pose, then step the physics N times with the schedule as PD targets (+ lead) →
        recorded rows (N+1).
        """
        import torch

        env, scene = rec.env, rec.env.scene
        dev = env.device

        def write_base(env_ids):
            for an, art in arts:
                q = S[an][0][env_ids]
                art.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
                art.set_joint_position_target(q, env_ids=env_ids)

        write_base(ids)
        rows = [rec._row()[:bs].detach().float().cpu()]
        dt = float(env.step_dt)
        st_t = torch.tensor(start, device=dev)
        for g in range(N):
            if g < hold:
                for rn, r in rigs:
                    rs = obj_rsi[rn][1].clone()
                    rs[:, 7:] = 0.0
                    r.write_root_state_to_sim(rs, env_ids=ids)
            for an, art in arts:
                inf = info[an]
                tgt = S[an][g + 1].clone()
                vref = (S[an][g + 1][:, inf["lead"]] - S[an][g][:, inf["lead"]]) / dt
                tgt[:, inf["lead"]] += inf["kdkp"] * vref
                art.set_joint_position_target(tgt, env_ids=ids)
            GW.apply(env)
            for _ in range(int(env.cfg.decimation)):
                for an, art in arts:
                    inf = info[an]
                    gc = art.root_physx_view.get_gravity_compensation_forces().to(dev)[:bs][:, inf["arm_ids"]]
                    art.set_joint_effort_target(gc, joint_ids=inf["arm_ids"], env_ids=ids)
                scene.write_data_to_sim()
                env.sim.step(render=False)
                scene.update(dt=env.physics_dt)
            if (
                self.mode == "every_pin"
                # objects pinned to the RSI pose (speed 0) during the approach — written after physics and before
                # recording, so
            ):
                for (
                    rn,
                    r,
                ) in (
                    rigs
                ):  # the recorded rows and the join state are exactly RSI (the policy starts from the v2 object state)
                    rs = obj_rsi[rn][1].clone()
                    rs[:, 7:] = 0.0
                    r.write_root_state_to_sim(rs, env_ids=ids)
            pend = ids[
                st_t >= g + 1
            ]  # before its start in the next row too → the fixed pose as is (speed 0), so that row `start`
            if len(pend):  # is exactly the fixed pose (removes the sag of one physics step; zero t=0 spread)
                write_base(pend)
            rows.append(rec._row()[:bs].detach().float().cpu())
        return rows

    @staticmethod
    def _residuals(arts, rigs, info, obj_rsi, bs) -> dict:
        """Approach end ↔ RSI residual (arm and finger position rad, speed rad/s, objects mm, °) per env."""
        import torch

        res = {}
        for an, art in arts:
            inf = info[an]
            dq = (art.data.joint_pos[:bs] - inf["q_rsi"]).abs()
            qd = art.data.joint_vel[:bs].abs()
            res[f"{an}_arm_pos"] = dq[:, inf["arm_ids"]].max(-1).values.cpu().numpy()
            res[f"{an}_arm_vel"] = qd[:, inf["arm_ids"]].max(-1).values.cpu().numpy()
            res[f"{an}_fing_pos"] = dq[:, inf["fing"]].max(-1).values.cpu().numpy()
            res[f"{an}_fing_vel"] = qd[:, inf["fing"]].max(-1).values.cpu().numpy()
        for rn, r in rigs:
            p0, _ = obj_rsi[rn]
            p1 = r.data.root_link_pose_w[:bs]
            res[f"{rn}_pos_mm"] = (1000 * (p1[:, :3] - p0[:, :3]).norm(dim=-1)).cpu().numpy()
            dotq = (p1[:, 3:7] * p0[:, 3:7]).sum(-1).abs().clamp(max=1.0)
            res[f"{rn}_rot_deg"] = torch.rad2deg(2 * torch.acos(dotq)).cpu().numpy()
        return res
