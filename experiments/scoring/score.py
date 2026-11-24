"""The shared object-tracking scorer of Tab. 2: every method's rollouts, on the method's own time axis, scored with the
released FlashCHORD metrics (ADD-AUC, SPIDER SR = SP-SR, ManipTrans SR = MP-SR) against the reference it was given.

Paper metrics (``score_all`` → ``chord_metrics``; owner ``flash_chord.evaluation.metrics``, imported, not copied):
  steps with original-motion time τ in [0, τ_end] are scored, τ_end = the end of the reference or the method's own
  horizon if shorter; a rollout that stops before τ_end gets ADD = ∞ for its missing steps and fails both SRs;
  the table's ADD-AUC of a clip = the best of its rollouts (``add_auc_best``); ADD vertices = the mesh the method
  simulates, sampled by CHORD's rule (``sample_vertices``).
Appendix completion rule (``score_rollouts``): our judge's object-tracking gate applied to every rollout over the whole
horizon — values of the RL judge (``dexengine/rl/args.py`` defaults, ``dexengine/rl/tasks/mdp/terminations/task.py``
ObjectTrackingExceeded, lag-window branch):
  POS_M 0.05 m · ROT_DEG 90° · LAG_BACK 25 frames (0.5 s at 50 Hz) · LAG_AHEAD 5 frames · GRACE 60 policy steps
  (1.2 s): at reference frame F the rollout lives while ONE reference frame in [F − 25, F + 5] is within 5 cm and 90°
  at the same frame; frames before the grace are not checked. Fingertip and contact terminations are off in the judge,
  so not used. Completed = never broken to the last reference frame; bimanual = both objects. Clip success: policy
  methods (N ≥ 2 rollouts) = completion rate ≥ PASS_FRAC; optimisation methods (1 rollout) = that trajectory
  completes.
Quaternions are wxyz (MuJoCo qpos and Isaac Lab). Time alignment: each reference frame takes the nearest rollout state.

Common rollout form (the adapters build it): t (T_r,) s · obj_pos_<body> (N,T_r,3) · obj_quat_<body> (N,T_r,4 wxyz)
[· step_dt · tau_end]. Reference: a SPIDER-form clip folder, 0/trajectory_kinematic.npz qpos (T,70) — objects =
qpos[:, -14:-7] (right) · qpos[:, -7:] (left), ref_dt in task_info.json — or the original CHORD motion parquet.
usage: python -m experiments.scoring.score spider|egoengine|npz --clip-dir <clip> [--rollout F] [--sim-dt 0.01] --out J
       python -m experiments.scoring.score chord --traj <parquet> --motion-parquet <motion_v1.parquet> --time-scale S
       --out J
       python -m experiments.scoring.score motion --rollout <npz> --motion-parquet <motion_v1.parquet> --time-scale S
       --out J
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__:
    from experiments.scoring.score_chord import chord_metrics, ref_at, vertices_mjcf, vertices_urdf
else:  # run by file path (`<FlashCHORD venv python> -I score.py …`): the repository is not importable there
    import importlib.util

    _spec = importlib.util.spec_from_file_location("score_chord", Path(__file__).with_name("score_chord.py"))
    _chord = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_chord)
    chord_metrics, ref_at, vertices_mjcf, vertices_urdf = (
        _chord.chord_metrics,
        _chord.ref_at,
        _chord.vertices_mjcf,
        _chord.vertices_urdf,
    )

POS_M, ROT_DEG, PASS_FRAC = 0.05, 90.0, 0.05
POLICY_DT = 0.02  # policy period (dexengine/rl/tasks/cfg/arm_ik_env.py: sim.dt = 0.02 / decimation)
LAG_BACK_S, LAG_AHEAD_S, GRACE_S = (
    25 * POLICY_DT,
    5 * POLICY_DT,
    60 * POLICY_DT,
)  # 0.5 s · 0.1 s · 1.2 s (reference time)
LAG_BACK, LAG_AHEAD, GRACE = 25, 5, 60  # in 50 Hz reference frames (= the above at ref_dt 0.02)


def _quat_angle(q1, q2):
    d = np.abs(np.sum(q1 * q2, axis=-1) / (np.linalg.norm(q1, axis=-1) * np.linalg.norm(q2, axis=-1) + 1e-12))
    return 2.0 * np.arccos(np.clip(d, 0.0, 1.0))


def load_reference(clip_dir: str) -> dict:
    q = np.load(os.path.join(clip_dir, "0", "trajectory_kinematic.npz"))["qpos"]
    ti = json.loads(Path(os.path.join(clip_dir, "task_info.json")).read_text())
    ref = {"dt": float(ti["ref_dt"]), "objs": {"right": (q[:, -14:-11], q[:, -11:-7])}}
    bim = (ti.get("embodiment_type") == "bimanual") if "embodiment_type" in ti else (nq_obj_count(clip_dir) == 2)
    if not bim:
        ref["objs"]["right"] = (q[:, -7:-4], q[:, -4:])
    if bim:
        ref["objs"]["left"] = (q[:, -7:-4], q[:, -4:])
    return ref


def completed_one(t_r, pos_r, quat_r, ref_pos, ref_quat, ref_dt):
    """One rollout, one object → (completed, first failing reference frame or -1)."""
    T = ref_pos.shape[0]
    t_ref = np.arange(T) * ref_dt
    lag_b, lag_a, grace = (int(round(x / ref_dt)) for x in (LAG_BACK_S, LAG_AHEAD_S, GRACE_S))
    if len(t_r) == 0:  # no frame at all (e.g. EgoEngine committed 0 chunks) = not completed
        return False, grace
    if t_r[-1] + 0.5 * ref_dt < t_ref[-1]:
        last_ok = int(np.searchsorted(t_ref, t_r[-1] + 0.5 * ref_dt))
        return False, max(last_ok, grace)  # the rollout ends before the reference does = not completed
    idx = np.clip(np.searchsorted(t_r, t_ref), 0, len(t_r) - 1)
    prev = np.clip(idx - 1, 0, len(t_r) - 1)
    idx = np.where(np.abs(t_r[prev] - t_ref) < np.abs(t_r[idx] - t_ref), prev, idx)
    p, q = pos_r[idx], quat_r[idx]
    rot = np.deg2rad(ROT_DEG)
    for F in range(grace, T):
        lo, hi = max(0, F - lag_b), min(T - 1, F + lag_a)
        dp = np.linalg.norm(ref_pos[lo : hi + 1] - p[F], axis=-1)
        da = _quat_angle(ref_quat[lo : hi + 1], q[F][None])
        if not np.any((dp <= POS_M) & (da <= rot)):
            return False, F
    return True, -1


def score_rollouts(roll: dict, ref: dict) -> dict:
    sides = [s for s in ref["objs"] if f"obj_pos_{s}" in roll]
    if sorted(sides) != sorted(ref["objs"]):
        raise ValueError(
            f"rollout objects {sides} ≠ reference objects {sorted(ref['objs'])} — every object must be scored"
        )
    N = roll[f"obj_pos_{sides[0]}"].shape[0]
    done = np.ones(N, bool)
    first_fail = np.full(N, -1)
    for s in sides:
        for n in range(N):
            ok, f = completed_one(
                roll["t"], roll[f"obj_pos_{s}"][n], roll[f"obj_quat_{s}"][n], *ref["objs"][s], ref["dt"]
            )
            if not ok:
                done[n] = False
                first_fail[n] = f if first_fail[n] < 0 else min(first_fail[n], f)
    rate = float(done.mean())
    success = (rate >= PASS_FRAC) if N >= 2 else bool(done[0])
    return {
        "n_rollouts": int(N),
        "n_complete": int(done.sum()),
        "complete_rate": rate,
        "success": bool(success),
        "sides": sides,
        "first_fail_frame": first_fail.tolist()[:8],
        "ref_frames": int(ref["objs"][sides[0]][0].shape[0]),
        # rollout length in frames — 0 = the method committed nothing (first_fail_frame then shows grace, not a frame
        # that failed)
        "rollout_frames": int(len(roll["t"])),
    }


def score_all(roll: dict, ref: dict, verts: dict) -> dict:
    """The paper's numbers (CHORD's released metrics) + our old completion rule as an appendix field. Both stop at the
    method's own horizon (roll['tau_end'] when the method has one)."""
    tau_end = roll.get("tau_end")
    sd = roll.get("step_dt")  # npz rollouts saved before step_dt existed: the median spacing of their own τ axis
    sd = float(sd) if sd is not None else float(np.median(np.diff(np.asarray(roll["t"], float))))
    res = chord_metrics(roll, ref, verts, sd, tau_end)
    rref = ref
    if tau_end is not None:  # the old rule judges the same span: original frames up to the method's horizon
        k = int(np.floor(float(tau_end) / ref["dt"] + 1e-6)) + 1
        rref = {"dt": ref["dt"], "objs": {b: (p[:k], q[:k]) for b, (p, q) in ref["objs"].items()}}
    res["appendix_completion_rule"] = score_rollouts(
        {k: v for k, v in roll.items() if k.startswith(("t", "obj_")) and k not in ("tau_end",)}, rref
    )
    return res


def save_scored(out_json: str, roll: dict) -> None:
    """Keep the scored common-form rollout (t · object poses) next to the result json as <name>.scored.npz, so
    per-frame
    errors can be measured again without the method's raw output."""
    base = out_json[:-5] if out_json.endswith(".json") else out_json
    np.savez_compressed(base + ".scored.npz", **{k: np.asarray(v) for k, v in roll.items()})


# ---------- adapters: each method's output → the common rollout form ----------
def from_qpos(qpos: np.ndarray, dt: float, bimanual: bool) -> dict:
    """MuJoCo-qpos rollout (objects = the last 14 / 7 qpos entries) → common form. qpos (T,nq) or (N,T,nq)."""
    if qpos.ndim == 2:
        qpos = qpos[None]
    out = {"t": np.arange(qpos.shape[1]) * dt}
    if bimanual:
        out.update(
            obj_pos_right=qpos[..., -14:-11],
            obj_quat_right=qpos[..., -11:-7],
            obj_pos_left=qpos[..., -7:-4],
            obj_quat_left=qpos[..., -4:],
        )
    else:
        out.update(obj_pos_right=qpos[..., -7:-4], obj_quat_right=qpos[..., -4:])
    return out


def adapter_spider(clip_dir: str, sim_dt: float) -> dict:
    """SPIDER trajectory_mjwp.npz → common form. Time follows SPIDER's own convention (the rule's owner is SPIDER's
    code; only the formula is restated here):
      · record time: run_mjwp.py advances mj_data.time += sim_dt after each physics step and stacks qpos → npz 'time'
        (first value = sim_dt);
      · that step's target = interp(reference, n)[k] (k = time/sim_dt · n = ref_dt/sim_dt) — spider/interp.py is
        F.interpolate(size=H·n, align_corners=True), so the k-th target = original frame k·(H−1)/(H·n−1).
      → reference time τ = k·(H−1)/(H·n−1)·ref_dt. A 'time' that breaks the convention is not scored
      (SystemExit)."""
    d = np.load(os.path.join(clip_dir, "0", "trajectory_mjwp.npz"))
    q = d["qpos"].reshape(-1, d["qpos"].shape[-1])  # (ticks, steps, nq) → (T, nq): one row per physics step
    tm = d["time"].reshape(-1).astype(float)
    if len(tm) != len(q) or abs(tm[0] - sim_dt) > 1e-6 or not np.allclose(np.diff(tm), sim_dt, atol=1e-6):
        raise SystemExit(
            f"[SCORE] spider: npz time breaks SPIDER's convention (first value sim_dt={sim_dt}, step sim_dt) — "
            f"len {len(tm)}/{len(q)} first {tm[:2]} — check that --sim-dt is that run's value"
        )
    ti = json.loads(Path(os.path.join(clip_dir, "task_info.json")).read_text())
    H = np.load(os.path.join(clip_dir, "0", "trajectory_kinematic.npz"))["qpos"].shape[0]
    ref_dt = float(ti["ref_dt"])
    n = int(np.round(ref_dt / sim_dt))
    if not (ref_dt > sim_dt and abs(n * sim_dt - ref_dt) < 1e-9):
        raise SystemExit(
            f"[SCORE] spider: ref_dt {ref_dt} / sim_dt {sim_dt} is not SPIDER's interpolation branch "
            "(integer multiple · ref_dt > sim_dt)"
        )
    out = from_qpos(q, sim_dt, ti.get("embodiment_type") == "bimanual")
    out["t"] = np.rint(tm / sim_dt) * (H - 1) / (H * n - 1) * ref_dt
    out["step_dt"] = (H - 1) / (H * n - 1) * ref_dt
    return out


def load_chord_motion_reference(motion_parquet: str) -> dict:
    """Object poses of the original CHORD motion (HF motion_v1.parquet): its own time axis (fps, TACO 30) and world
    frame
    (the frame of CHORD's simulation)."""
    import pyarrow.parquet as pq

    r = pq.read_table(
        motion_parquet, columns=["fps", "object_body_names", "object_body_position", "object_body_wxyz"]
    ).to_pylist()[0]
    pos = np.asarray(r["object_body_position"], float)
    quat = np.asarray(r["object_body_wxyz"], float)
    return {
        "dt": 1.0 / float(r["fps"]),
        "objs": {n: (pos[:, k], quat[:, k]) for k, n in enumerate(r["object_body_names"])},
    }


def adapter_chord(path: str, motion_parquet: str, time_scale: float):
    """FlashCHORD object_trajectories_output (parquet · schema flash_chord_object_trajectories_v1 · xyzw · rows
    ordered step, world, body). Reference = the original motion parquet (independent, original time axis). A rollout
    step's original time τ = FlashCHORD's playback schedule at its reference_frame:
        τ = flash_chord.data.resampling.playback_times(source frames, source fps, control_fps,
        time_scale)[reference_frame]
    (the rule's owner is FlashCHORD; with a speed factor ≠ 1 the schedule is linspace(0, length,
    int(length·fps/factor)), so one step is not control_dt·factor).
    Independent check: FlashCHORD's own reference positions in the same parquet must equal the original motion at τ
    (≤ 1 mm) — otherwise time, world or object pairing is off and the clip is not scored.
    time_scale = that run's task.motion_speed (no default). Reads with pyarrow + numpy only (no pandas in that venv)."""
    if time_scale is None or not (time_scale > 0):
        raise SystemExit("[SCORE] chord: --time-scale is required (that run's task.motion_speed · no default)")
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    assert md.get("flash_chord.schema") == "flash_chord_object_trajectories_v1", md.get("flash_chord.schema")
    assert md.get("flash_chord.quaternion_order") == "xyzw"
    assert md.get("flash_chord.row_order") == "step,environment_id,object_body_id"
    S, W, K = (int(md[f"flash_chord.{k}"]) for k in ("step_count", "world_count", "object_body_count"))
    ctrl_dt = 1.0 / float(md["flash_chord.control_fps"])

    def col(n):
        return t.column(n).to_numpy().reshape(S, W, K)

    names = t.column("object_body_name").to_pylist()[:K]
    ach_p = np.stack([col(f"achieved_position_{a}") for a in "xyz"], -1)
    ach_q = np.stack([col(f"achieved_quaternion_{a}") for a in "wxyz"], -1)
    frames = col("reference_frame")[:, 0, 0].astype(np.int64)
    ref = load_chord_motion_reference(motion_parquet)
    if sorted(names) != sorted(ref["objs"]):
        raise ValueError(f"trajectory objects {names} ≠ original-motion objects {sorted(ref['objs'])}")
    from flash_chord.data.resampling import playback_times  # owner of the step → original-time rule (FlashCHORD venv)

    n_src = ref["objs"][names[0]][0].shape[0]
    src_t, play_t = playback_times(n_src, 1.0 / ref["dt"], 1.0 / ctrl_dt, float(time_scale))
    if frames.min() < 0 or frames.max() >= len(play_t):
        raise SystemExit(
            f"[SCORE] chord: reference_frame {frames.min()}..{frames.max()} outside the playback schedule "
            f"({len(play_t)} entries) — speed factor or fps differ from that run"
        )
    tau = play_t[frames]
    ref_p = np.stack([col(f"reference_position_{a}") for a in "xyz"], -1)[:, 0]  # (S,K,3) FlashCHORD's own reference
    for k, name in enumerate(names):
        rp = ref["objs"][name][0]
        e = float(np.abs(np.stack([np.interp(tau, src_t, rp[:, i]) for i in range(3)], -1) - ref_p[:, k]).max())
        if e > 1e-3:
            raise SystemExit(
                f"[SCORE] chord: FlashCHORD's own reference ≠ original motion(τ) — {name} max {e * 1000:.1f} mm "
                "> 1 mm (time scale, world frame or object pairing is off — not scored)"
            )
    roll = {
        "t": tau,
        "step_dt": float(np.median(np.diff(tau))) if len(tau) > 1 else 1.0 / float(md["flash_chord.control_fps"]),
    }
    for k, name in enumerate(names):
        roll[f"obj_pos_{name}"] = np.transpose(ach_p[:, :, k], (1, 0, 2))
        roll[f"obj_quat_{name}"] = np.transpose(ach_q[:, :, k], (1, 0, 2))
    return roll, ref


def adapter_motion(rollout_npz: str, motion_parquet: str, time_scale: float):
    """Generic policy-rollout record (ManipTrans, FlashDexRetarget): npz keys t (T,) s of the method's sim time ·
    obj_pos_<body> (N,T,3) · obj_quat_<body> (N,T,4 wxyz) in the CHORD world (the recorder maps back to it); body names
    = the parquet object_body_names. Reference = the original motion parquet; original time τ = t × time_scale
    (required). The method's own reference and horizon (ref_t · ref_obj_* · horizon_t) must equal the original motion
    (≤ 1 mm ·
    ≤ 0.01 rad) or the clip is not scored."""
    if time_scale is None or not (time_scale > 0):
        raise SystemExit("[SCORE] motion: --time-scale is required (no default)")
    d = dict(np.load(rollout_npz))
    ref = load_chord_motion_reference(motion_parquet)
    names = sorted(k[len("obj_pos_") :] for k in d if k.startswith("obj_pos_"))
    if names != sorted(ref["objs"]):
        raise ValueError(f"rollout objects {names} ≠ original-motion objects {sorted(ref['objs'])}")
    d["t"] = np.asarray(d["t"], float) * float(time_scale)
    # the method's own reference and horizon (recorder: baselines/maniptrans/record_lib.method_reference) — required
    need = ["ref_t", "horizon_t"] + [f"ref_obj_pos_{n}" for n in names] + [f"ref_obj_quat_{n}" for n in names]
    if [k for k in need if k not in d]:
        raise SystemExit(
            f"[SCORE] motion: rollout lacks the method reference {[k for k in need if k not in d]} — "
            "re-record it (baselines/maniptrans/eval_record.py)"
        )
    rt = np.asarray(d["ref_t"], float) * float(time_scale)
    for (
        n
    ) in names:  # independent oracle: the reference the method was given = the original motion (≤ 1 mm · ≤ 0.01 rad)
        op, oq = ref_at(*ref["objs"][n], ref["dt"], rt)
        ep = float(np.linalg.norm(op - d[f"ref_obj_pos_{n}"], axis=-1).max())
        er = float(_quat_angle(oq, d[f"ref_obj_quat_{n}"]).max())
        if ep > 1e-3 or er > 1e-2:
            raise SystemExit(
                f"[SCORE] motion: the method's reference ≠ original motion — {n} {ep * 1000:.2f} mm · {er:.4f} rad "
                "(time · world · body pairing — not scored)"
            )
    d["tau_end"] = float(d["horizon_t"]) * float(time_scale)
    d["step_dt"] = float(np.median(np.diff(d["t"]))) if len(d["t"]) > 1 else 1.0 / 60.0
    return d, ref


def chord_object_ids(motion_parquet: str) -> dict:
    """{body name: TACO object id} from the original motion's object_mesh_paths ('…/taco/171_cm.obj' → '171')."""
    import pyarrow.parquet as pq

    r = pq.read_table(motion_parquet, columns=["object_body_names", "object_mesh_paths"]).to_pylist()[0]
    return {
        n: os.path.basename(m).split("_")[0]
        for n, m in zip(r["object_body_names"], r["object_mesh_paths"], strict=False)
    }


#: The mesh each method simulates, per TACO object id, under the workspace data/ folder — CHORD: its rigid URDF ·
#: ManipTrans (the 'motion' adapter): the port's object URDF.
OBJECT_URDF = {
    "chord": "chord/assets/urdfs/taco/{id}_rigid.urdf",
    "motion": "maniptrans/objects/{id}/{id}_maniptrans.urdf",
}


def object_urdf(method: str, obj_id: str) -> str:
    """Path of the object URDF ``method`` simulates for TACO object ``obj_id`` (workspace data/)."""
    from experiments import workspace

    return str(workspace.data(OBJECT_URDF[method].format(id=obj_id)))


def adapter_egoengine(clip_dir: str, sim_dt: float, npz: str | None = None) -> dict:
    """EgoEngine run_mjwp_mixed.py output (trajectory_mjwp_mixed.npz): qpos per physics step of the committed chunks
    only
    (shorter on failure = not completed)."""
    d = np.load(
        npz or os.path.join(clip_dir, "0", "trajectory_mjwp_mixed.npz"), allow_pickle=True
    )  # --rollout: the CLI's --output dir
    # zero committed chunks (the run failed at its first window) → qpos is saved as (0, 0): a zero-length rollout =
    # not completed
    q = np.asarray(d["qpos"])
    nq = int(np.asarray(d["initial_qpos"]).shape[-1])
    q = q.reshape(-1, nq) if q.size else np.zeros((0, nq))
    ti = json.loads(Path(os.path.join(clip_dir, "task_info.json")).read_text())
    bim = (ti.get("embodiment_type") == "bimanual") if "embodiment_type" in ti else (nq_obj_count(clip_dir) == 2)
    out = from_qpos(q, sim_dt, bim)
    # time: EgoEngine runs SPIDER's load_data/interp (its spider/interp.py is byte-identical) → the same rule as
    # adapter_spider, committed physics step k (1-based) ↔ τ = k·(H−1)/(H·n−1)·ref_dt
    H = np.load(os.path.join(clip_dir, "0", "trajectory_kinematic.npz"))["qpos"].shape[0]
    ref_dt = float(ti["ref_dt"])
    n = int(np.round(ref_dt / sim_dt))
    out["t"] = (np.arange(q.shape[0]) + 1) * (H - 1) / (H * n - 1) * ref_dt
    out["step_dt"] = (H - 1) / (H * n - 1) * ref_dt
    return out


def nq_obj_count(clip_dir: str) -> int:
    """Number of free (object) joints of the clip's MuJoCo scene — for task_info files without embodiment_type."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(os.path.join(clip_dir, "scene.xml"))
    return int(sum(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(m.njnt)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["spider", "egoengine", "chord", "motion", "npz"])
    ap.add_argument("--clip-dir", help="SPIDER-form clip folder (the reference — spider / egoengine / npz)")
    ap.add_argument("--traj", help="chord: FlashCHORD evaluate_policy's object_trajectories_output parquet")
    ap.add_argument(
        "--motion-parquet", help="chord / motion: the sequence's original motion_v1.parquet (independent reference)"
    )
    ap.add_argument(
        "--time-scale", type=float, default=None, help="chord / motion: that run's task.motion_speed (required)"
    )
    ap.add_argument(
        "--rollout", help="npz: common-form rollout file · egoengine: the method's npz · motion: the recorder's npz"
    )
    ap.add_argument("--sim-dt", type=float, default=0.01)
    ap.add_argument("--out", help="result json (the scored rollout is kept beside it as <name>.scored.npz)")
    a = ap.parse_args(argv)
    if a.method in ("chord", "motion"):
        roll, ref = (
            adapter_chord(a.traj, a.motion_parquet, a.time_scale)
            if a.method == "chord"
            else adapter_motion(a.rollout, a.motion_parquet, a.time_scale)
        )
        ids = chord_object_ids(a.motion_parquet)
        verts = {n: vertices_urdf(object_urdf(a.method, ids[n])) for n in ref["objs"]}
        res = score_all(roll, ref, verts)
        print("[SCORE] " + json.dumps(res))
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(res, fh, indent=1)
            save_scored(a.out, roll)
        return 0
    ref = load_reference(a.clip_dir)
    empty_rc = 0
    roll = (
        adapter_spider(a.clip_dir, a.sim_dt)
        if a.method == "spider"
        else adapter_egoengine(a.clip_dir, a.sim_dt, a.rollout)
        if a.method == "egoengine"
        else dict(np.load(a.rollout))
    )
    vl = vertices_mjcf(os.path.join(a.clip_dir, "scene.xml"), len(ref["objs"]))
    res = score_all(roll, ref, dict(zip(["right", "left"][: len(vl)], vl, strict=False)))
    if a.method == "egoengine":
        # the method's own window/attempt record (replay · MPC · RL steps per window) lives next to its npz; keep a
        # copy beside our result
        wj = os.path.splitext(a.rollout or os.path.join(a.clip_dir, "0", "trajectory_mjwp_mixed.npz"))[0] + ".json"
        res["method_windows_json"] = wj if os.path.isfile(wj) else None
        if a.out and res["method_windows_json"]:
            import shutil

            shutil.copyfile(wj, os.path.splitext(a.out)[0] + ".method_windows.json")
        # an empty trajectory (0 committed steps) never ends as a silent rc 0: non-finite attempts, or no attempt
        # record to check → not a result (no [SCORE] line, rc 3). Every attempt finite → the method's own failure on
        # window 0 (its 5 cm / 1 rad rule) → scored as not completed, with a marker.
        res["committed_steps"] = int(len(roll["t"]))
        if res["committed_steps"] == 0:
            att = json.loads(Path(wj).read_text()).get("attempts", []) if res["method_windows_json"] else []
            nf = sum(1 for x in att if x.get("mean_error") is None or x.get("final_error") is None)
            res["empty_rollout"] = {"attempts": len(att), "nonfinite_attempts": nf}
            empty_rc = 3 if (not att or nf) else 0
            print(
                f"[EGO_EMPTY] committed_steps=0 · attempts {len(att)} · non-finite {nf} — "
                + (
                    "not a result; the row fails (rc 3)"
                    if empty_rc
                    else "all attempts finite: the method committed no window — scored as not completed"
                ),
                file=sys.stderr,
            )
    # a non-finite / unverifiable empty trajectory is not a score: no '[SCORE]' line (the row worker reads FAIL
    # 'no [SCORE] line'; it does not read the rc); the json is kept as the record
    print(("[SCORE_EMPTY] " if empty_rc else "[SCORE] ") + json.dumps(res))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
        save_scored(a.out, roll)
    return empty_rc


if __name__ == "__main__":
    sys.exit(main())
