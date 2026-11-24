"""Kinematic pair trees: constants, the naive arm IK of a pair tree, its finishing step (reference arm, idle side) and
the input checks shared by the prep commands (prep.py)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from experiments.policies.kin_data.prep_ik import dls_step, fk_err, seed_step

MARK = "[KIN_PREP]"
HAND, HAND_DIR, ARM = "wuji2", "wuji2", "rby1"
SHADOW_PURPOSE = "e5_kin"


SIDES = ("right", "left")


def inside(root: Path, p: Path) -> bool:
    """Is p a real location inside the shadow (False when an ancestor symlink leads to the base)?"""
    rr, rp = os.path.realpath(root), os.path.realpath(p)
    return rp == rr or rp.startswith(rr + os.sep)


def _arm_model(side: str):
    """Pinocchio model of one arm (resolve_combo): data, end-effector frame, arm joint indices and velocity limits."""
    import numpy as np  # noqa: PLC0415

    from dexengine.placement import arm_ik as AIK  # noqa: PLC0415

    urdf, ee_frame, joints, base_xyz, _ = AIK.resolve_combo(ARM, HAND, side)
    from dexengine.robots import urdf_load as UL  # noqa: PLC0415

    model = UL.build_model(str(urdf))
    data = model.createData()
    ee_id = model.getFrameId(ee_frame)
    qidx = tuple(model.joints[model.getJointId(n)].idx_q for n in joints)
    vlim = np.array([float(model.velocityLimit[model.joints[model.getJointId(n)].idx_v]) for n in joints])
    return model, data, ee_id, qidx, joints, base_xyz, vlim


def _seed_arrays(seed: dict, joints, T: int):
    """Our arm seed in this arm's joint order, whether its frames map 1:1 to the tree, and its FK error buffers.

    Our arm seed (--arm-seed, default none = unchanged): frame 0 = our episode's arm frame 0; when the length equals
    the tree's (frames correspond) every frame also starts from our arm; candidates = DLS (our arm), our arm as is,
    DLS (previous), previous — among those that reach, the one closest to our arm (else to the previous); none
    reaches → smallest reach measure. Our arm's FK error (RL wrist ↔ this target) is recorded too.
    """
    import numpy as np  # noqa: PLC0415

    S_q = np.asarray(seed["q"], float)[:, [list(seed["names"]).index(n) for n in joints]]
    S_al = len(S_q) == T
    s_pe, s_re = np.full(T, np.nan, np.float32), np.full(T, np.nan, np.float32)
    return S_q, S_al, s_pe, s_re


def _frame_buffers(T: int, n_joints: int):
    """Per-frame buffers: arm joints, position error, solver residual, rotation error, fallback flag, solution source.

    Solution source (ik_step_src): 0 DLS from previous · 1 restart · 2 divergence guard (previous DLS) · 3 divergence
    guard (previous) · 4 frame 0.
    """
    import numpy as np  # noqa: PLC0415

    qa = np.zeros((T, n_joints), np.float32)
    pe = np.zeros(T, np.float32)
    pv = np.zeros(T, np.float32)
    re_ = np.zeros(T, np.float32)
    fb = np.zeros(T, bool)
    src = np.zeros(T, np.int8)
    return qa, pe, pv, re_, fb, src


def _frame_target(AIK, P, Q, t: int, palm_off, rz, zoff: float, base):
    """Frame t's IK target in the robot base frame: palm position and rotation matrix."""
    import numpy as np  # noqa: PLC0415

    r = AIK.Rotation.from_quat([Q[t][1], Q[t][2], Q[t][3], Q[t][0]])
    p = P[t] + r.apply(palm_off)
    tp = rz.apply(p) + np.array([0.0, 0.0, zoff]) - base
    tr = (rz * r).as_matrix()
    return tp, tr


def _record_frame(model, data, ee_id, q, tp, tr, t: int, pe, re_, pv):
    """Store frame t's FK errors of the chosen solution (Euclidean, geodesic) and the solver residual (pv).

    The recorded position error is the Euclidean distance of the palm origin |p_FK − p_target|; the solver residual
    ‖v of log(SE3)‖ mixes in rotation and grows far beyond the distance when the orientation error is large, so it is
    kept separately as wrist_pos_se3v_all.
    """
    import numpy as np  # noqa: PLC0415
    import pinocchio as pin  # noqa: PLC0415

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    _s, pe[t], re_[t] = fk_err(model, data, ee_id, q, tp, tr)
    pv[t] = float(np.linalg.norm(pin.log(pin.SE3(tr, tp).actInv(data.oMf[ee_id])).vector[:3]))
    e_p, e_r = pv[t], re_[t]
    return e_p, e_r


def _save_cache(
    out_npz: Path, extra: dict, base_xyz, joints, qa, pe, pv, re_, base_z_off, T, zoff, fb, vlim, freq, src
) -> Path:
    """Write the arm IK cache to a temporary file next to out_npz (renamed by the caller); returns its path."""
    import numpy as np  # noqa: PLC0415

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(
        tmp,
        base_xyz=np.asarray(base_xyz, np.float64),
        arm_joint_names=np.array(joints),
        **extra,
        arm_joint_positions=qa[0],
        arm_joint_positions_all=qa,
        wrist_pos_error=np.float32(pe[0]),
        wrist_pos_error_all=pe,
        wrist_pos_se3v_all=pv,
        wrist_pos_error_def=np.array("euclid_palm_origin_m", dtype="<U24"),
        wrist_rot_error_deg=np.float32(re_[0]),
        wrist_rot_error_deg_all=re_,
        base_z_off=np.float32(base_z_off),
        num_frames=np.int32(T),
        scene_z_offset=np.float32(zoff),
        solver=np.array("e5_kin_dls_framewise_fb", dtype="<U24"),
        ik_fallback_frames=fb,
        arm_vel_limit=vlim,
        frequency=np.float64(freq),
        ik_step_src=src,
    )
    return tmp


def _print_seed(tree: Path, side: str, S_al: bool, s_pe, s_re, src) -> None:
    import numpy as np  # noqa: PLC0415

    print(
        f"{MARK} [KIN_SEED] {tree.name} {side} our arm seed · frame map {S_al} · frame0 seed FK "
        f"{s_pe[0] * 1e3:.1f} mm "
        f"{s_re[0]:.1f}° · chosen solution source {dict(zip(*np.unique(src, return_counts=True), strict=True))}",
        flush=True,
    )  # applied marker


def _ik_summary(side: str, T: int, qa, pe, re_, cap, zoff: float, n_re: int, n_fb: int) -> dict:
    """Per-side IK statistics of the cache (errors in mm and degrees, largest step, jumps over the velocity cap)."""
    import numpy as np  # noqa: PLC0415

    from experiments.policies.kin_data import reach as KR  # noqa: PLC0415 — owner of the reach measure (tolerances)

    dq = np.abs(np.diff(qa, axis=0)).max() if T > 1 else 0.0
    n_jump, jr = KR.jumps(qa, cap)
    return {
        "side": side,
        "T": T,
        "pos_mm_mean": float(pe.mean() * 1e3),
        "pos_mm_p95": float(np.percentile(pe, 95) * 1e3),
        "pos_mm_max": float(pe.max() * 1e3),
        "rot_deg_mean": float(re_.mean()),
        "rot_deg_p95": float(np.percentile(re_, 95)),
        "rot_deg_max": float(re_.max()),
        "max_step_rad": float(dq),
        "zoff": zoff,
        "restarted": n_re,
        "fallback_frames": n_fb,
        "n_jump": n_jump,
        "max_dq_ratio": jr,
    }


def kin_ik(
    tree: Path,
    side: str,
    wrist_npz: Path,
    out_npz: Path,
    base_z_off: float,
    a_tol_m: float = 0.002,
    a_tol_deg: float = 1.0,
    restarts: int = 12,
    retry_every: int = 10,
    seed: dict | None = None,
) -> dict:
    """Per-frame 6-DoF DLS IK (arm_ik's solve_ik_pose, solve_ik_multistart, resolve_combo and palm correction as they
    are). Target = wrist cache (tree-frame palm) → Rz(-90°) + scene z offset (coords.scene_z_offset — the env loader's
    rule) − robot base. Frame 0 = multi-start (20), later frames start from the previous frame's solution. No rejection
    or repair — errors are recorded as they are. Joint continuity is not repaired (the kinematic baseline keeps jumping
    frames): jumps between frames are only measured (cache arm_vel_limit, frequency, ik_step_src → reach.jumps,
    [KIN_JUMP], episode meta). Marker [KIN_IK] naive."""
    import numpy as np  # noqa: PLC0415

    from dexengine.placement import arm_ik as AIK  # noqa: PLC0415
    from dexengine.placement.coords import core as CO  # noqa: PLC0415

    model, data, ee_id, qidx, joints, base_xyz, vlim = _arm_model(side)
    freq = float(np.load(tree / "0/trajectory_kinematic.npz")["frequency"])
    cap = vlim / freq  # per-frame joint cap (measured only — the formula of reach.dq_cap; stored in the cache)

    palm_off = AIK.palm_frame_offset(HAND)
    zoff = float(CO.scene_z_offset(tree))
    base = np.asarray(base_xyz, float) + np.array([0.0, 0.0, base_z_off])
    wp = np.load(wrist_npz)
    P, Q = wp["positions"], wp["quaternions"]
    T = len(P)
    rz = AIK.Rotation.from_euler("z", -90, degrees=True)
    S_al = None
    if seed is not None:
        S_q, S_al, s_pe, s_re = _seed_arrays(seed, joints, T)
    q = None
    n_re, last_re, prev_ok, n_fb = 0, -(10**9), True, 0
    qa, pe, pv, re_, fb, src = _frame_buffers(T, len(joints))
    for t in range(T):
        tp, tr = _frame_target(AIK, P, Q, t, palm_off, rz, zoff, base)
        if S_al is not None:
            q, src[t] = seed_step(model, data, ee_id, qidx, t, tp, tr, q, S_q, S_al, s_pe, s_re)
        elif q is None:
            q, e_p, e_r, _ = AIK.solve_ik_multistart(model, data, tp, tr, ee_id, qidx, num_starts=20, max_iters=500)
            src[t] = 4
        else:
            q, src_t, last_re, restarted, guarded = dls_step(
                model, data, ee_id, qidx, t, tp, tr, q, a_tol_m, a_tol_deg, restarts, retry_every, prev_ok, last_re
            )
            if restarted:
                n_re += 1
            if guarded:
                n_fb += 1
                fb[t] = True
            if src_t is not None:
                src[t] = src_t
        qa[t] = [q[i] for i in qidx]
        e_p, e_r = _record_frame(model, data, ee_id, q, tp, tr, t, pe, re_, pv)
        prev_ok = not (e_p > a_tol_m or e_r > a_tol_deg)
    extra = (
        {}
        if S_al is None
        else {
            "seed_pos_error_all": s_pe,
            "seed_rot_error_deg_all": s_re,
            "seed_aligned": np.bool_(S_al),
            "seed_scene": np.array(str(seed.get("scene", ""))),
        }
    )
    tmp = _save_cache(out_npz, extra, base_xyz, joints, qa, pe, pv, re_, base_z_off, T, zoff, fb, vlim, freq, src)
    if S_al is not None:
        _print_seed(tree, side, S_al, s_pe, s_re, src)
    os.replace(tmp, out_npz)
    return _ik_summary(side, T, qa, pe, re_, cap, zoff, n_re, n_fb)


def _finish_specs(troot: Path, tree: str, rdir: Path, idle, robot_z: float, combo: str, ds: str) -> list[str]:
    """Canonical stage specs of the finish: body FK, wrist caches, D16 idle-arm rest pose, then body/wrist again for
    the idle hands (their hand block changed)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    specs = [f"body:{troot / tree},-q"]
    specs += [
        f"wrist:--data_dir,{troot},--output_dir,{rdir / f'wrist_poses_{s}'},--side,{s},--sequence,{tree}" for s in SIDES
    ]
    for (
        s
    ) in idle:  # D16 idle arm: canonical rest pose (stage_dp._idle_rest — no IK, no judge) → that hand block updated
        specs.append(
            f"ik:--sequence,{tree},--tree,{troot / tree},--side,{s},--arm,{ARM},--hand,{HAND},--base_z_off={robot_z},"
            f"--wrist_dir,{rdir / f'wrist_poses_{s}'},--output_dir,{P.ik_dir(combo, s, ds)},--no-idle-guard"
        )
    if idle:
        specs.append(f"body:{troot / tree},-q")
        specs += [
            f"wrist:--data_dir,{troot},--output_dir,{rdir / f'wrist_poses_{s}'},--side,{s},--sequence,{tree}"
            for s in idle
        ]
    return specs


def finish_tree(
    combo: str,
    ds: str,
    troot: Path,
    tree: str,
    rdir: Path,
    robot_z: float,
    act,
    idle,
    seed: dict | None = None,
    ref_main: Path | None = None,
) -> tuple[dict, dict]:
    """The common finish once a tree stands (source = bake, variant = make_variant): body FK, wrist cache, idle arm
    (D16 rest pose), active arm IK per frame, reach (reach.py) → (IK summary, reach). Sources and variants use this one
    function."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.placement.bake import chain as SR  # noqa: PLC0415
    from experiments.policies.kin_data import reach as KR  # noqa: PLC0415 — owner of the reach rule

    if (
        ref_main is not None
    ):  # --ref-arm: our tree and cache are complete — the tree-rewriting steps (body, wrist, idle arm) are not run
        return ref_arm(combo, ds, tree, act, idle, ref_main)
    specs = _finish_specs(troot, tree, rdir, idle, robot_z, combo, ds)
    for sp in specs:
        rc, _ex = SR.run_one(sp)
        if rc:
            raise RuntimeError(f"step failed rc={rc}: {sp.split(':', 1)[0]}")
    ik = {
        s: kin_ik(
            troot / tree,
            s,
            rdir / f"wrist_poses_{s}" / f"{tree}.npz",
            P.ik_dir(combo, s, ds) / f"{tree}.npz",
            robot_z,
            **({} if seed is None else {"seed": seed}),
        )
        for s in act
    }
    reach = KR.reach_of({s: P.ik_dir(combo, s, ds) / f"{tree}.npz" for s in act})
    nj = sum(reach["n_jump"].values())
    print(
        f"{MARK} [KIN_JUMP] {tree} n_jump {reach['n_jump']} · max_dq/limit {reach['max_dq_ratio']} · "
        f"{'no jump' if nj == 0 else 'jump'} (recorded only · not a filter) · reach {reach['reach']} → "
        f"pass {reach['passes']}",
        flush=True,
    )
    return ik, reach


def _copy_ref_wrists(main: Path, combo: str, ds: str, tree: str, act) -> None:
    """Copy our wrist caches (body FK and wrist steps are not run) — present sides only; an active side must exist."""
    from dexengine.core import paths as P  # noqa: PLC0415

    for s in SIDES:  # the wrist cache is ours too (body FK and wrist steps are not run) — present sides only
        ws, wd = (
            main / f"precompute/{ds}/robots/{combo}/wrist_poses_{s}/{tree}.npz",
            P.robots_dir(combo, ds) / f"wrist_poses_{s}/{tree}.npz",
        )
        if ws.is_file():
            wd.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ws, wd)
        elif s in act:
            raise SystemExit(f"{MARK} --ref-arm: reference wrist cache missing {ws}")


def _ref_arm_stats(s: str, src: Path, dst: Path, sha: dict, per: dict, ik: dict) -> None:
    """Record the copied active-arm cache's wrist error (in-tolerance share in per[s]) and its IK summary in ik[s]."""
    import numpy as np  # noqa: PLC0415

    z = np.load(dst)
    pe = np.asarray(z["wrist_pos_error_all"], float) if "wrist_pos_error_all" in z.files else None
    re_ = np.asarray(z["wrist_rot_error_deg_all"], float) if "wrist_rot_error_deg_all" in z.files else None
    per[s] = None if pe is None or re_ is None else round(float(((pe <= 0.01) & (re_ <= 10.0)).mean()), 4)
    ik[s] = {
        "side": s,
        "T": int(len(z["arm_joint_positions_all"])),
        "ref_arm": str(src),
        "sha256": sha[s],
        "pos_mm_mean": float(pe.mean() * 1e3) if pe is not None else -1.0,
        "pos_mm_max": float(pe.max() * 1e3) if pe is not None else -1.0,
        "rot_deg_mean": float(re_.mean()) if re_ is not None else -1.0,
        "rot_deg_max": float(re_.max()) if re_ is not None else -1.0,
        "fallback_frames": 0,
    }


def ref_arm(combo: str, ds: str, tree: str, act, idle, main: Path) -> tuple[dict, dict]:
    """--ref-arm: the kinematic arm = the reference arm our RL followed through RSI, as is — no IK; the base's arm
    cache
    (precompute/<ds>/robots/<combo>/arm_ik_poses_<side>_cont25/<tree>.npz, arm_joint_positions_all = the file the
    render's
    reset_arm_to_trajectory reads every frame) is copied byte for byte for both arms. There is nothing to reach (it is
    the
    reference); the cache's wrist error is only recorded. A missing cache dies."""
    import hashlib  # noqa: PLC0415

    from dexengine.core import paths as P  # noqa: PLC0415

    def rel(s):
        return Path(f"precompute/{ds}/robots/{combo}") / P.ik_dir(combo, s, ds).name / f"{tree}.npz"

    ik, per, sha = {}, {}, {}
    _copy_ref_wrists(main, combo, ds, tree, act)
    for s in list(act) + list(idle):
        src, dst = main / rel(s), P.ik_dir(combo, s, ds) / f"{tree}.npz"
        if not src.is_file():
            raise SystemExit(f"{MARK} --ref-arm: reference arm cache missing {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(f".tmp{os.getpid()}.npz")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        sha[s] = hashlib.sha256(dst.read_bytes()).hexdigest()[:16]
        if s in act:
            _ref_arm_stats(s, src, dst, sha, per, ik)
    print(
        f"{MARK} [KIN_REFARM] {tree} arm = copy of our reference cache {sha} (no IK · no reach to measure)", flush=True
    )  # applied marker
    reach = {
        "rule": "ref-arm: our RSI reference arm played back (no IK) — nothing to reach",
        "per_side": per,
        "reach": 1.0,
        "passes": True,
        "frames": {s: v["T"] for s, v in ik.items()},
        "reach_min": None,
        "solver": "ref_arm_copy",
        "n_jump": dict.fromkeys(act),
        "max_dq_ratio": dict.fromkeys(act),
        "ref_wrist_frac_in_tol": per,
    }
    return ik, reach


def match_inputs(a, ds: str, tree: str, act, idle):
    """Pair inputs (default off): --idle-from = our idle hands (per tree) → overrides (active, idle); --arm-seed = our
    arm
    seed npz. A flag on without that tree's value dies."""
    seed = None
    if getattr(a, "idle_from", ""):
        tab = a._idle_tab
        if (
            (ds, tree) not in tab
        ):  # a tree missing from the pair table (--idle-from) is only a variant source (not rendered) · marker
            print(
                f"{MARK} [KIN_MATCH] {tree} not a pair scene (variant source) — active-hand rule · no seed", flush=True
            )
            return list(act), list(idle), None
        idle = [s for s in SIDES if s in tab[(ds, tree)]]
        act = [s for s in SIDES if s not in idle]
        print(f"{MARK} [KIN_IDLE] {tree} our idle hand {idle or 'none'} → active {act}", flush=True)  # applied marker
    if getattr(a, "arm_seed", ""):
        import numpy as np  # noqa: PLC0415

        f = Path(a.arm_seed) / ds / f"{tree}.npz"
        if not f.is_file():
            raise SystemExit(f"{MARK} --arm-seed lacks {f}")
        z = np.load(f)
        seed = {"names": [str(x) for x in z["names"]], "q": z["q"], "scene": str(z["scene"])}
    return list(act), list(idle), seed


def drop_tree(root: Path, combo: str, ds: str, troot: Path, tree: str) -> list[str]:
    """Delete the outputs (tree, placement, wrist, IK cache) of variant draws that failed reach — real paths inside the
    root
    only (links or outside = death). The variant's base alias (bake.alias_base's relative link to the source sibling)
    stays:
    the shadow guard (data_layout.shadow_guard) reads link removal through realpath and would kill it as a base write.
    An
    alias without a tree is read by nobody, and the layer statement is built from the remaining trees only."""
    from dexengine.core import paths as P  # noqa: PLC0415

    gone = []
    td = troot / tree
    cands = [P.placement_dir(combo, ds) / f"{tree}{P.PLACEMENT_SFX}"]
    cands += [P.robots_dir(combo, ds) / f"wrist_poses_{s}" / f"{tree}.npz" for s in SIDES]
    cands += [P.ik_dir(combo, s, ds) / f"{tree}.npz" for s in SIDES]
    for f in [td, *cands]:
        if not f.exists():
            continue
        if f.is_symlink() or not inside(root, f):
            raise SystemExit(f"{MARK} location to delete {f} is a link or outside the root {root} — not deleted")
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()
        gone.append(str(f.relative_to(root)))
    return gone
