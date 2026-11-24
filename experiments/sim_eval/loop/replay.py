"""Replay modes and the hold window: fixed start, data actions, substep target record, plain PD replay, action replay,
the strict hold and the data-hold start."""

from __future__ import annotations

import json
from pathlib import Path

from dexengine.rl.tasks import switches
from experiments.sim_eval.loop.common import DATA_HOLD, MARK, PD_REPLAY, _say
from experiments.sim_eval.loop.control import Targets, arm_ctrl_label
from experiments.sim_eval.session_cfg import value as _cfg

#: Fixed start (off unless the scene's job.json has a fixed_start field):
#: extra_rows = approach length K (50 Hz rows of the approach) — widens the policy target window (rows < T + K) and the
#: cap (⌈cap_mult × T⌉ + K). joints = the stated start pose (joint name → rad, from the set's frame 0, head
#: excluded) — checked against the reset state; above FIXED_START_TOL it dies. The start pose itself comes from the
#: scene shadow (arm IK cache, tree hand qpos frame 0) — here only the widening and the check.
FIXED_START_TOL = 1e-3


def fixed_start_cfg(job: dict, base: str) -> dict | None:
    fs = job.get("fixed_start")
    if fs is None:
        return None
    k, jn = fs.get("extra_rows"), fs.get("joints")
    if not (isinstance(k, int) and 0 <= k <= 1000 and isinstance(jn, dict) and jn):
        raise SystemExit(
            f"{MARK} job.fixed_start format — extra_rows (0–1000 integer) and joints (name → rad) required"
        )
    if not base.startswith("policy"):
        raise SystemExit(f"{MARK} job.fixed_start applies to policy modes only — {base}")
    return {"extra_rows": k, "joints": {str(n): float(v) for n, v in jn.items()}, "source": str(fs.get("source", ""))}


def fixed_start_check(fs: dict, names: list[str], j0) -> dict:
    """Reset state (scene order j0) ↔ stated joints — max difference per joint group (arm · hand · side). A stated
    joint missing from the scene or a difference > TOL dies.
    """
    miss = [n for n in fs["joints"] if n not in names]
    if miss:
        raise SystemExit(f"{MARK} [E5_FIXED_START] {len(miss)} stated joints are not in the scene {miss[:4]}")
    g = {}
    for n, v in fs["joints"].items():
        k = f"{n.split('_', 1)[0]}_{'arm' if '_arm_' in n else 'hand'}"
        g[k] = max(g.get(k, 0.0), abs(float(j0[names.index(n)]) - v))
    g = {k: round(v, 6) for k, v in sorted(g.items())}
    if max(g.values()) > FIXED_START_TOL:
        raise SystemExit(
            f"{MARK} [E5_FIXED_START] reset state ↔ statement max difference {g} > {FIXED_START_TOL} rad — "
            "the scene shadow did not apply"
        )
    return g


def _load_actions(job: dict):
    """The data episode's action.joint_target.robot (56, joint-name order checked) — the npz episode_arrays extracted
    from the parquet.
    """
    import numpy as np

    z = np.load(job["episode_arrays"], allow_pickle=False)
    return z["action_joint_target"], json.loads(str(z["names"])), z["joint_pos"]


class SubRec:
    """Record per physics step (substep, right after apply_action = the value that physics step uses) the articulations'
    position, velocity and effort targets (rec_sub=1, action replay only). Index = step n × D + substep s (step n makes
    state row n+1). The replay controller spec (gains, period, gravity compensation, decimation, lead) goes into the
    meta — the contract for a data user replaying from these targets alone."""

    def __init__(self, raw_env, tg: Targets):
        self.env, self.tg = raw_env, tg
        self.am = raw_env.action_manager
        self.orig = self.am.apply_action
        self.buf = {k: {"pos": [], "vel": [], "eff": []} for k in tg.arts}

        def wrapped():
            self.orig()
            for k, a in tg.arts.items():
                d = a.data
                self.buf[k]["pos"].append(d.joint_pos_target[0].detach().cpu().numpy().copy())
                self.buf[k]["vel"].append(d.joint_vel_target[0].detach().cpu().numpy().copy())
                self.buf[k]["eff"].append(d.joint_effort_target[0].detach().cpu().numpy().copy())

        self.am.apply_action = wrapped

    def undo(self) -> None:
        self.am.apply_action = self.orig

    def save(self, path: Path, n_steps: int) -> dict:
        import numpy as np

        D = int(self.env.cfg.decimation)
        arrs, spec = (
            {},
            {
                "decimation": D,
                "physics_dt": float(self.env.physics_dt),
                "step_dt": float(self.env.step_dt),
                "index": "step n × decimation + substep s · step n makes state row n+1",
                "arm_ctrl": arm_ctrl_label(),
                "arm_ff": switches.text("DEXENGINE_ARM_FF", "1"),
                "arm_ff_vel_scale": switches.text("DEXENGINE_ARM_FF_VEL_SCALE", "1.0"),
                "arm_ff_mode": switches.text("DEXENGINE_ARM_FF_MODE", "lead"),
                "arts": {},
            },
        )
        for k, a in self.tg.arts.items():
            b = self.buf[k]
            if len(b["pos"]) != n_steps * D:
                raise SystemExit(f"{MARK} substep record {k}: {len(b['pos'])} ≠ steps {n_steps} × {D}")
            for f in ("pos", "vel", "eff"):
                arrs[f"{k}.{f}"] = np.asarray(b[f], np.float32)
            st = a.root_physx_view.get_dof_stiffnesses()[0].detach().cpu().numpy()
            dp = a.root_physx_view.get_dof_dampings()[0].detach().cpu().numpy()
            arrs[f"{k}.stiffness"], arrs[f"{k}.damping"] = st.astype(np.float32), dp.astype(np.float32)
            spec["arts"][k] = {
                "joint_names": list(a.joint_names),
                "vel_target_nonzero": bool(np.abs(arrs[f"{k}.vel"]).max() > 0),
                "effort_target_absmax": float(np.abs(arrs[f"{k}.eff"]).max()),
            }
        np.savez_compressed(path, spec=json.dumps(spec, ensure_ascii=False), **arrs)
        _say(
            f"[E5_REC_SUB] physics-step targets {n_steps} × {D} → {path} · "
            + " · ".join(
                f"{k} joints {len(v['joint_names'])} velocity target "
                f"{'set' if v['vel_target_nonzero'] else '0'} max effort target "
                f"{v['effort_target_absmax']:.3g}"
                for k, v in spec["arts"].items()
            )
        )
        return spec


class PdReplay:
    """Plain PD replay: the physics-step targets (position, velocity) an action replay recorded are written per substep
    as is — no IK, reference or lead. Gravity compensation is the env's canonical feed-forward (DEXENGINE_ARM_FF=1 +
    VEL_SCALE=0 = gravity only · DEXENGINE_ARM_FF=0 = none), logged in the marker.
    Source = <runs>/<pd_src_batch>_<scene folder>/act_replay/substep_targets.npz (no default)."""

    def __init__(self, raw_env, tg: Targets, job: dict, out: Path):
        import numpy as np
        import torch

        b = _cfg("pd_src_batch", "")
        if not b:
            raise SystemExit(
                f"{MARK} {PD_REPLAY}: no pd_src_batch (batch name of the action-replay record) — no default"
            )
        if arm_ctrl_label() == "hyeon_reflead":
            raise SystemExit(
                f"{MARK} {PD_REPLAY}: reference lead is on (DEXENGINE_ARM_FF_VEL_SCALE≠0) — the recorded "
                "targets already contain it, it would be added twice"
            )
        scene = Path(job["plan"]).parent.name
        # source root = this run's runs folder (default) or pd_src_root
        self.src_dir = Path(_cfg("pd_src_root") or out.parent) / f"{b}_{scene}" / "act_replay"
        z = np.load(self.src_dir / "substep_targets.npz", allow_pickle=False)
        self.spec = json.loads(str(z["spec"]))
        self.D = int(self.spec["decimation"])
        if self.D != tg.D or abs(float(self.spec["physics_dt"]) - float(raw_env.physics_dt)) > 1e-12:
            raise SystemExit(
                f"{MARK} {PD_REPLAY}: record period (D {self.D} · dt {self.spec['physics_dt']}) ≠ env (D "
                f"{tg.D} · dt {raw_env.physics_dt})"
            )
        # pd_hold=last (default off): of the physics-step targets only each frame's last substep value is used and held
        # for the frame — the information of a replay that has only the data's action labels (50 Hz frame = last
        # substep target).
        self.hold = _cfg("pd_hold", "")
        if self.hold not in ("", "last"):
            raise SystemExit(f"{MARK} {PD_REPLAY}: pd_hold={self.hold!r} — '' or last")
        self.tg, self.k, self.pos, self.vel = tg, dict.fromkeys(tg.arts, 0), {}, {}
        for key, a in tg.arts.items():
            if list(self.spec["arts"][key]["joint_names"]) != list(a.joint_names):
                raise SystemExit(f"{MARK} {PD_REPLAY}: {key} joint-name order differs from the record")
            self.pos[key] = torch.as_tensor(z[f"{key}.pos"], device=raw_env.device)
            self.vel[key] = torch.as_tensor(z[f"{key}.vel"], device=raw_env.device)
            st = a.root_physx_view.get_dof_stiffnesses()[0].detach().cpu().numpy()
            if float(np.abs(st - z[f"{key}.stiffness"]).max()) > 1e-6:
                raise SystemExit(f"{MARK} {PD_REPLAY}: {key} PD stiffness differs from the record")
        self.src_meta = json.loads((self.src_dir / "episode.json").read_text())
        _say(
            f"[E5_PDREPLAY] source {self.src_dir} · physics steps {len(next(iter(self.pos.values())))} · D {self.D} · "
            f"arm controller {arm_ctrl_label()} · at record {self.spec['arm_ctrl']} · substeps "
            f"{'last value of the frame held (pd_hold=last)' if self.hold else 'all'} · velocity targets "
            + " · ".join(f"{k} {'used' if v['vel_target_nonzero'] else '0'}" for k, v in self.spec["arts"].items())
        )

    def apply(self, key: str) -> None:
        i = min(self.k[key], len(self.pos[key]) - 1)
        if self.hold == "last":
            i = min((i // self.D) * self.D + self.D - 1, len(self.pos[key]) - 1)
        self.k[key] += 1
        a = self.tg.arts[key]
        a.set_joint_position_target(self.pos[key][i : i + 1])
        a.set_joint_velocity_target(self.vel[key][i : i + 1])


def _joint_target_columns(cj):
    """Record column of every joint in the recorded joint-target channels; a joint name twice dies."""
    col = {}
    for c in cj:
        for i, nm in enumerate(c["names"]):
            if nm in col:
                raise SystemExit(
                    f"{MARK} action replay: joint name {nm} is in two recorded joint-target channels "
                    f"({[x['name'] for x in cj]})"
                )
            col[nm] = int(c["lo"]) + i
    return col


def _match_row_offset(act_names, pinned, col, cj, st, acts):
    """The record-row offset where the recorded joint targets equal the data targets (render-pinned joints left out)."""
    import numpy as np

    cmp = [i for i, n in enumerate(act_names) if n not in pinned]
    miss = [n for n in act_names if n not in col]
    if miss:
        raise SystemExit(
            f"{MARK} action replay: data joints {miss[:4]} are not in the recorded joint targets "
            f"({[x['name'] for x in cj]})"
        )
    JT = np.asarray(st[:, [col[n] for n in act_names]], np.float32)
    n = int(acts.shape[0])
    if len(st) < n:
        raise SystemExit(f"{MARK} action replay: recorded trajectory {len(st)} rows < data {n} rows")
    diffs = [float(np.abs(JT[o : o + n, cmp] - acts[:, cmp]).max()) for o in range(len(st) - n + 1)]
    off = int(np.argmin(diffs))
    if diffs[off] > 1e-5:
        raise SystemExit(
            f"{MARK} action replay: no record-row offset matches the data joint targets (smallest max "
            f"difference {diffs[off]:.3g} @ {off})"
        )
    return n, diffs, off


def _act_rows(S, raw_env, acts, act_names: list[str], zero: bool, pinned: set[str]):
    """Action-replay input: the recorded 'action' column (raw policy output — wrist residual + hand) of the trajectory
    the scene loaded (load_traj's file and selected trajectory); column range = meta.env.collect.channels (lo · hi).
    Data rows ↔ record rows are aligned by one rule: the offset at which every recorded joint_target/<articulation>
    column (robot · robot_left; a one-hand scene has the left arm separately; names checked, a name twice dies) equals
    the data action (joint targets) — max difference > 1e-5 dies. The only joints left out of the check are those the
    episode's head_joints marks as source 'render_pinned' (columns the data rewrote with the render-pinned head angle,
    outside the action term)."""
    import numpy as np

    from dexengine.rl.eval import traj_file as TF

    path, clip = S._e5_traj_src
    tf = TF.load(path)
    st = tf.selected(clip)
    if st is None:
        raise SystemExit(f"{MARK} action replay: no selected trajectory for {clip} — {path}")
    chs = (tf.meta.get("env", {}).get("collect") or {}).get("channels") or []
    ca = [c for c in chs if c.get("kind") == "action"]
    cj = [c for c in chs if c.get("kind") == "joint_target"]
    if len(ca) != 1 or not cj:
        raise SystemExit(
            f"{MARK} action replay: recorded channels action {len(ca)} · joint_target {len(cj)} — one "
            f"action and ≥ 1 joint target needed — {path}"
        )
    ca = ca[0]
    am = raw_env.action_manager
    A = np.asarray(st[:, ca["lo"] : ca["hi"]], np.float32)
    if A.shape[1] != int(am.total_action_dim) or list(ca["names"]) != list(am.active_terms):
        raise SystemExit(
            f"{MARK} action replay: recorded action {list(ca['names'])} width {A.shape[1]} ≠ env action "
            f"terms {list(am.active_terms)} "
            f"width {am.total_action_dim}"
        )
    col = _joint_target_columns(cj)
    n, diffs, off = _match_row_offset(act_names, pinned, col, cj, st, acts)
    A = A[off : off + n]
    if zero:
        A = np.zeros_like(A)
    info = {
        "src": path,
        "clip": clip,
        "action_cols": [int(ca["lo"]), int(ca["hi"])],
        "action_terms": list(ca["names"]),
        "oracle_skip_pinned": sorted(pinned),
        "term_modes": ca.get("term_modes"),
        "row_offset": off,
        "oracle_joint_target_maxdiff": diffs[off],
        "rows": n,
        "zero": zero,
    }
    _say(
        f"[E5_ACTREPLAY] source {path} · clip {clip} · action columns {ca['lo']}:{ca['hi']} {list(ca['names'])} "
        f"({ca.get('term_modes')}) · row offset {off} · oracle joint_target ↔ data max difference {diffs[off]:.2g}"
        f"(render-pinned head {sorted(pinned)} excluded) · "
        f"rows {n} · {'negative test: action 0' if zero else 'recorded as is'}"
    )
    return A, info


def _hold(raw_env, tg, rec, step_act, n_hold: int, elb, clip_last: int, read_targets: bool) -> int:
    """Strict pad: simulate n_hold rows after the last action with the last command kept (absolute-target modes = tg
    as is · action replay = the last action as is), recording state, targets and forces per step. Policy rollouts, the
    data baseline (data_hold) and replay modes share this function. The reference clock is pinned at L−1."""
    for _ in range(n_hold):
        if int(elb[0]) >= clip_last:
            elb.clamp_(max=clip_last - 1)
        raw_env.step(step_act)
        if read_targets:
            tg.t = {k: a_.data.joint_pos_target[0:1].clone() for k, a_ in tg.arts.items()}
        rec.add(tg)
    return n_hold


def _data_hold_start(S, T, raw_env, tg, acts, elb, clip_last: int) -> dict:
    """Start of data_hold: write the last row of the stage-1 record (the physics rollout that made the data episode —
    the scene trajectory file's selected rows), give velocities from the last two rows (joints, root linear and
    angular), then
    absolute target = the data's last joint target (action.joint_target)."""
    import numpy as np
    import torch

    from isaaclab.utils import math as M

    from dexengine.rl.eval import traj_record as TR

    tp, clip = S._e5_traj_src
    TF = _tf_mod().load(str(tp))
    rows = TF.selected(clip)
    if rows is None or len(rows) < 2:
        raise SystemExit(f"{MARK} {DATA_HOLD}: no selected trajectory for {clip} or fewer than 2 rows ({tp})")
    n_rows = len(rows)
    if n_rows != len(acts):
        raise SystemExit(f"{MARK} {DATA_HOLD}: stage-1 record {n_rows} rows ≠ data targets {len(acts)} rows ({clip})")
    dt = float(raw_env.step_dt)
    dev = raw_env.device
    st = torch.as_tensor(np.ascontiguousarray(rows[n_rows - 2 : n_rows]), dtype=torch.float32, device=dev)
    TR.write_states(S.env.scene, T["binds"], st[1:2], S.eids)
    vmax = 0.0
    from dexengine.rl.tasks.scene import extra_slots as XS

    for b in T["binds"]:
        r0, r1 = st[0:1, b.root_cols], st[1:2, b.root_cols]
        v = torch.cat([(r1[:, :3] - r0[:, :3]) / dt, M.quat_box_minus(r1[:, 3:7], r0[:, 3:7]) / dt], dim=-1)
        if b.name.startswith(XS.PREFIX):  # extra slots = kinematic teleport (velocity 0 — as in stage 1)
            v = torch.zeros_like(v)
        b.asset.write_root_link_velocity_to_sim(v, env_ids=S.eids)
        vmax = max(vmax, float(v.abs().max()))
        if b.kind == "artic":
            q0, q1 = st[0:1, b.joint_cols], st[1:2, b.joint_cols]
            b.asset.write_joint_state_to_sim(q1, (q1 - q0) / dt, env_ids=S.eids)
    tg.set_flat(acts[n_rows - 1])
    elb[0] = clip_last - 1
    _say(
        f"[E5_HOLD] {DATA_HOLD}: wrote stage-1 record last row {n_rows - 1} · max finite-difference "
        f"velocity {vmax:.4f} · target = data last joint target · "
        f"reference clock {clip_last}"
    )
    return {"src": str(tp), "clip": clip, "data_rows": n_rows, "vel_max": vmax}


def _tf_mod():
    from dexengine.rl.eval import traj_file as TF

    return TF
