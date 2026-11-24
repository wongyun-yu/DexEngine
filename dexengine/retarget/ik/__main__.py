"""Hand tracker CLI (`python -m dexengine.retarget.ik`): human keypoints → robot-hand joint trajectory for one rig.

Callers pass: --dataset-dir --dataset-name --robot-type --embodiment-type --task --data-id 0 --ref-dt 0.02
--no-show-viewer [--save-video] [--seed 0]; stdin may carry 'y\\n' (ignored). Unknown flags fail.

Pipeline: validate the keypoint archive → assemble the IK model from the scene → solve frame by frame
(`solve.Tracker`) → smooth and post-process (`post`) → write the kinematic, rollout and diagnostics files
(`outputs`, which states every output key).
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

H_STEP = 0.01


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m dexengine.retarget.ik",
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--robot-type", required=True)
    ap.add_argument("--embodiment-type", required=True, choices=("bimanual", "right", "left"))
    ap.add_argument("--task", required=True)
    ap.add_argument("--data-id", type=int, default=0)
    ap.add_argument("--ref-dt", type=float, default=0.02)
    ap.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def run(args: argparse.Namespace) -> str:
    """Solve one clip; returns the output folder."""
    import numpy as np  # noqa: PLC0415

    from dexengine.core.repo import HAND_ASSETS_DIR  # noqa: PLC0415
    from dexengine.retarget import mjutil  # noqa: PLC0415
    from dexengine.retarget.ik import outputs as O  # noqa: PLC0415
    from dexengine.retarget.ik import post as PP  # noqa: PLC0415
    from dexengine.retarget.ik import prior as P  # noqa: PLC0415
    from dexengine.retarget.ik.inputs import load_keypoints  # noqa: PLC0415

    np.random.seed(args.seed)
    ref_dt, h = float(args.ref_dt), H_STEP
    sides = P.emb_sides(args.embodiment_type)
    mano_dir = mjutil.processed_data_dir(
        args.dataset_dir, args.dataset_name, "mano", args.embodiment_type, args.task, args.data_id
    )
    out_dir = mjutil.processed_data_dir(
        args.dataset_dir, args.dataset_name, args.robot_type, args.embodiment_type, args.task, args.data_id
    )
    scene_xml = os.path.join(os.path.dirname(out_dir), "scene.xml")
    kp = load_keypoints(os.path.join(mano_dir, "trajectory_keypoints.npz"), PP.WINDOW)
    prior_path = str(HAND_ASSETS_DIR / args.robot_type / "mjcf" / "retarget_prior.yaml")
    conf, meta = P.build_prior_conf(prior_path, args.robot_type, sides)
    print(P.prior_marker(meta))

    scene, ik, limits, prior, clip_names, raw = _track(scene_xml, kp, conf, sides, h)
    branch = _branch(scene, ik, raw, kp, conf)

    sm = PP.smooth(ik, raw.qpos, limits, ref_dt)
    PP.object_passthrough(ik, sm, kp, ref_dt)
    n = sm.qpos.shape[0]
    rows = slice(PP.WINDOW // 2 + 1, PP.WINDOW // 2 + 1 + n)
    contact = kp.contact_ref[:, :10][rows]
    contact_pos = raw.contact_pos[rows]
    roll, peak, diverged = PP.rollout(scene, sm.qpos, sm.qvel, ref_dt)
    if diverged:
        print(f"[IK] debug rollout diverged (max |qpos| {peak:.3g}); written anyway")
    post_palm, post_tips = PP.post_filter_errors(scene, sm.qpos, kp)

    shared = O.extra_keys(kp, sm.qpos, _extra_object_columns(scene, kp.n_extra))
    kin = {
        "qpos": sm.qpos,
        "qpos_rollout": roll,
        "qvel": sm.qvel,
        "contact": contact,
        "contact_pos": contact_pos,
        "frequency": np.float64(1.0 / ref_dt),
        **shared,
    }
    rollout = {"qpos": roll, **shared}
    diag_npz, diag_json = _diagnostics(
        raw, sm, post_palm, post_tips, kp, h, ref_dt, conf, clip_names, peak, diverged, prior, prior_path
    )
    if branch:
        diag_json["branch_resolve"] = [r.as_json() for r in branch]
    O.write_all(out_dir, kin, rollout, diag_npz, diag_json)
    print(
        f"[IK] {args.dataset_name}/{args.task} {args.robot_type} {args.embodiment_type}: {kp.T} frames -> {n} rows, "
        f"palm {diag_json['palm_error_right_mean_m'] * 1e3:.2f}/{diag_json['palm_error_left_mean_m'] * 1e3:.2f} mm, "
        f"tips {diag_json['fingertip_error_mean_m'] * 1e3:.2f} mm, valid {diag_json['num_valid_frames']}/{kp.T} "
        f"-> {out_dir}"
    )
    return out_dir


def _track(scene_xml: str, kp, conf: dict | None, sides: tuple[str, ...], h: float) -> tuple:
    """Assemble the IK model from the scene, apply the prior's hyperextension clip and joint damping, and track every
    frame. Returns (scene model, IK model, limits, posture prior or None, clipped joint names, raw tracking result)."""
    import mujoco  # noqa: PLC0415

    from dexengine.retarget.ik.model import (  # noqa: PLC0415
        LimitSets,
        apply_hyperext_clip,
        build_ik_model,
        hinge_table,
        target_mocap_ids,
        tracked_entities,
    )
    from dexengine.retarget.ik.posture import PosturePrior  # noqa: PLC0415
    from dexengine.retarget.ik.solve import Tracker  # noqa: PLC0415

    scene = mujoco.MjModel.from_xml_path(scene_xml)
    scene.opt.timestep = h
    tracked = tracked_entities(scene, sides, kp.n_extra)
    ik = build_ik_model(scene_xml, tracked, h)
    clip_names, clipped = apply_hyperext_clip(ik, conf)
    if conf is not None and conf["limit_hyperextension"]:
        print(f"[PRIOR] clipped={clipped} considered={len(clip_names)}")
    prior = PosturePrior(conf, ik, kp.contact_raw) if conf is not None else None
    if prior is not None:
        prior.apply_damping(ik)
        print(prior.band_report())
    limits = LimitSets.from_model(ik)
    hinges = hinge_table(ik, kp.arti)
    tracker = Tracker(ik, scene, tracked, target_mocap_ids(ik, tracked), kp, limits, prior, hinges, sides)
    return scene, ik, limits, prior, clip_names, tracker.run()


def _branch(scene, ik, raw, kp, conf) -> list:
    """Branch re-solve of the windowed fingers (ik/branch.py) on the raw solve, then refresh that finger's raw records
    (tip error and track-site position) so the diagnostics describe the written rows. No window = no-op."""
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from dexengine.retarget.ik import branch as BR  # noqa: PLC0415
    from dexengine.retarget.ik.inputs import FINGERS, tip_col  # noqa: PLC0415

    reports = BR.resolve_branches(ik, raw.qpos, conf)
    if not reports:
        return reports
    d, ds = mujoco.MjData(ik), mujoco.MjData(scene)
    for r in reports:
        print(r.marker())
        side, role = r.finger.split("_", 1)
        k = FINGERS.index(role)
        col = (0 if side == "right" else 5) + k
        site = mujoco.mj_name2id(ik, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{role}_tip")
        track = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_SITE, f"track_hand_{side}_{role}_tip")
        for t in range(raw.qpos.shape[0]):
            d.qpos[:] = raw.qpos[t]
            mujoco.mj_kinematics(ik, d)
            raw.tip_err[t, col] = float(np.linalg.norm(d.site_xpos[site] - kp.ref[t, tip_col(side, k), :3]))
            if track >= 0:
                ds.qpos[:] = raw.qpos[t]
                mujoco.mj_kinematics(scene, ds)
                raw.contact_pos[t, col] = ds.site_xpos[track]
    return reports


def _extra_object_columns(scene, n_extra: int) -> dict[int, int]:
    """Extra object index -> qpos address of its free joint in the scene (objects without a joint are absent)."""
    import mujoco  # noqa: PLC0415

    extra_cols = {}
    for i in range(n_extra):
        j = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_JOINT, f"extra_object_{i}_joint")
        if j >= 0:
            extra_cols[i] = int(scene.jnt_qposadr[j])
    return extra_cols


def _diagnostics(raw, sm, post_palm, post_tips, kp, h, ref_dt, conf, clip_names, peak, diverged, prior, prior_path):
    """The diagnostics arrays and JSON of one clip, with the posture-prior report when a prior is active."""
    from dexengine.retarget.ik import outputs as O  # noqa: PLC0415
    from dexengine.retarget.ik.solve import INIT_SETTLE_STEPS, SETTLE_STEPS  # noqa: PLC0415

    diag_npz, diag_json = O.diagnostics(
        raw,
        sm.qpos,
        sm.qvel,
        post_palm,
        post_tips,
        T=kp.T,
        h=h,
        ref_dt=ref_dt,
        settle=SETTLE_STEPS,
        init_settle=INIT_SETTLE_STEPS,
        multistart=4,
        limit_hx=bool(conf and conf["limit_hyperextension"]),
        clipped=clip_names,
        rollout_peak=peak,
        rollout_diverged=diverged,
    )
    if prior is not None:
        from dexengine.core.repo import REPO_ROOT  # noqa: PLC0415

        # the record names the prior by its place in the code tree, not by the checkout's absolute path
        arrays, block = O.prior_report(prior, os.path.relpath(prior_path, REPO_ROOT), sm.qpos, ref_dt)
        diag_npz.update(arrays)
        diag_json["posture_prior"] = block
    return diag_npz, diag_json


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.show_viewer:
        raise NotImplementedError("the interactive viewer is not supported; run with --no-show-viewer")
    if args.save_video:
        warnings.warn(
            "[IK] --save-video: video output is not produced by this tracker (optional feature)", stacklevel=1
        )
    run(args)
    return 0


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.retarget.ik")
    sys.exit(main())
