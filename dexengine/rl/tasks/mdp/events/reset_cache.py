"""Arm IK caches used by the resets (per clip, frame-0 and all-frame), and the per-env robot height offset."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from dexengine.rl.tasks import env_flags

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _dz_bound_m() -> float:
    """Max |base_z_off| of the dz layer policy (dexengine/placement/search_policy.py, the dependency-free single
    owner)."""
    from dexengine.placement.search_policy import dz_bound_m

    return dz_bound_m()


def _load_reset_cache(env, side, data_side, ik_dir, seq_names, per_clip_all, cache_all, lengths) -> None:
    """RL ablation A2 (default OFF): DEXENGINE_ARM_IK_RESET_SUFFIX=<sfx> makes the RESET (RSI arm teleport
    and the idle-side snap) read <canonical dir><sfx> — an arm-joint cache with the same wrist poses solved per frame
    independently (experiments/ablations/rl_arm/arm_ik_cold_frames.py). The arm feedforward and ENVZ keep the canonical
    cache.
    A file that differs from the canonical one in anything but the arm joints and their IK residuals is refused."""
    rs = env_flags.get("DEXENGINE_ARM_IK_RESET_SUFFIX")
    if not rs:
        return
    rdir = ik_dir.rstrip("/") + rs
    rc = cache_all.clone()
    for i, seq_name in enumerate(seq_names):
        rp, cp = os.path.join(rdir, f"{seq_name}.npz"), os.path.join(ik_dir, f"{seq_name}.npz")
        if not os.path.isfile(rp):
            raise RuntimeError(f"[ARM_IK_RESET] {rp} missing — the reset cache must cover every clip")
        rz, cz = np.load(rp), np.load(cp)
        ra = np.asarray(rz["arm_joint_positions_all"], dtype=np.float32)
        same = ra.shape == per_clip_all[i].shape and all(
            (k in rz.files) == (k in cz.files) and (k not in cz.files or np.array_equal(rz[k], cz[k]))
            for k in ("base_xyz", "base_z_off", "arm_joint_names", "num_frames")
        )
        if not same:
            raise RuntimeError(f"[ARM_IK_RESET] {rp} differs from {cp} in more than the arm joints")
        ti = ra.shape[0]
        rc[i, :ti] = torch.tensor(ra, dtype=torch.float32, device=rc.device)
        rc[i, ti:] = rc[i, ti - 1]
    for s in {side, data_side}:
        setattr(env, f"_arm_ik_reset_all_{s}", rc)
        setattr(env, f"_arm_ik_reset_all_len_{s}", lengths)
    valid = torch.arange(rc.shape[1], device=rc.device).unsqueeze(0) < lengths.unsqueeze(1)
    diff = torch.rad2deg((rc - cache_all).abs().amax(-1))[valid]
    print(
        f"[ARM_IK_RESET] side={side} dir={rdir} clips={len(seq_names)} frames={int(valid.sum())} — the RSI reset and "
        f"the idle snap read this cache; feedforward + ENVZ keep {ik_dir} · frames > 5 deg from canonical "
        f"{float((diff > 5).float().mean()):.1%}",
        flush=True,
    )


def reset_cache_all(env, side: str):
    """(cache_all, lengths) the RESET reads for ``side`` — the A2 cache when loaded, else the canonical one."""
    r = getattr(env, f"_arm_ik_reset_all_{side}", None)
    if r is not None:
        return r, getattr(env, f"_arm_ik_reset_all_len_{side}")
    return getattr(env, f"_arm_ik_cache_all_{side}", None), getattr(env, f"_arm_ik_cache_all_len_{side}", None)


def _load_arm_ik_cache(env: ManagerBasedEnv, side: str = "right") -> torch.Tensor:
    """Load pre-computed arm IK poses from disk.

    Looks for per-clip .npz files in the arm_ik_poses_dir_right (from TrajectoryManagerCfg)
    or a default location derived from wrist_poses_dir_right. Each file contains
    ``arm_joint_positions`` (7,) from the arm IK precompute.

    Returns:
        (num_clips, 7) tensor of arm joint positions on the env device. There is no None: a missing or legacy
        directory, a directory with no file for any clip, and partial coverage all die (the old None sent the left arm
        to its default pose and the right arm to the ~5 cm single-IK step, both silently).
    """
    traj_mgr = env.trajectory_manager

    # Determine arm IK poses directory
    ik_dir = getattr(traj_mgr.cfg, "arm_ik_poses_dir_right", None)
    if ik_dir is None:
        # Try default location: sibling of wrist_poses
        wrist_dir = getattr(traj_mgr.cfg, "wrist_poses_dir_right", None)
        if wrist_dir:
            ik_dir = wrist_dir.replace("wrist_poses", "arm_ik_poses")

    _envz_on, arm_jpos, ik_dir, num_clips, seq_names = _arm_ik_dir_for_side(env, side, ik_dir, traj_mgr)
    _dz_arr = torch.zeros(num_clips, dtype=torch.float32)

    loaded, missing_seqs, n_all, per_clip_all = _load_clip_ik_caches(
        env, _dz_arr, _envz_on, arm_jpos, ik_dir, num_clips, seq_names
    )

    if loaded == 0:
        raise RuntimeError(
            f"[arm_ik] side='{side}': cache directory {ik_dir} holds no file for any of this run's {num_clips} clips "
            f"(e.g. {missing_seqs[:3]}) — no silent fallback to the left arm's default pose or the right arm's single "
            f"IK step. Check the clip names, DEXENGINE_ARM_IK_DIR_SUFFIX and the data root."
        )

    # STRICT guard: PARTIAL coverage would SILENTLY zero-fill the
    # uncovered clips' arm joints — teleporting the arm to joint=0 (far off-trajectory) with NO
    # single-IK fallback (this returns non-None, so reset_arm_to_trajectory takes the cache path).
    # That silent left-cache gap corrupts left tracking init + disables left RSI. Refuse loudly
    # instead of silently zero-filling.
    if loaded < num_clips:
        raise ValueError(
            f"[reset_arm_to_trajectory] PARTIAL arm-IK cache coverage for side='{side}': "
            f"{loaded}/{num_clips} clips have a cache in {ik_dir}. The {num_clips - loaded} "
            f"uncovered clip(s) would be SILENTLY zero-filled (arm -> joint=0, off-trajectory). "
            f"Missing: {missing_seqs}. Generate them with "
            f"`dexengine/placement/arm_ik.py --dataset <dataset> --side {side} --all` "
            f"(or restrict the run to the covered clips). Refusing to silently zero-fill."
        )

    _install_all_frame_cache(
        env, side, _dz_arr, _envz_on, arm_jpos, ik_dir, loaded, n_all, num_clips, per_clip_all, seq_names
    )

    print(
        f"[reset_arm_to_trajectory] Loaded pre-computed arm IK poses "
        f"for {loaded}/{num_clips} clips from {ik_dir}" + (f" (all-frame: {n_all}/{loaded})" if n_all else "")
    )
    return arm_jpos


def _arm_ik_dir_for_side(env, side, ik_dir, traj_mgr):
    """Per-side arm-IK dir: cfg carries the RIGHT dir (arm_ik_poses_right/); the left mirror swaps the side suffix
    (generated by the arm IK stage, left side).
    """
    if ik_dir is not None and side != "right":
        _base = ik_dir.rstrip("/")
        ik_dir = (_base[: -len("_right")] if _base.endswith("_right") else _base) + f"_{side}"
    # A/B override: DEXENGINE_ARM_IK_DIR_SUFFIX="_cont" selects the
    # continuity-guarded cache (arm_ik_poses_<side>_cont, generated by
    # the arm IK stage with a 10° repair step) without touching the canonical
    # dir, so the pre-guard baselines stay reproducible.
    # The default suffix is "_cont25" (canonical): combo paths always read precompute/<ds>/robots/<combo>/. The old
    # unsuffixed behaviour needs DEXENGINE_ARM_IK_DIR_SUFFIX="" set **explicitly** (only then does the automatic
    # fallback below run).
    # ── Mixed-cache guard ─────────────────────────────────────────
    # No training on combo-tree data with a legacy global arm-IK cache: the objects would follow the re-placed
    # trajectory while reset/ENVZ read the old placement IK, and completion drops structurally to 0. All training
    # data is a combo product now, so the arm-IK cache must be a robots/<combo>/ product too; a missing cache or
    # suffix is a boot-time failure, not a silent fallback. Generations are not told apart by path names (the data
    # place is owned by the layer owner dexengine.core.layout; an old naming rule would switch off silently when a
    # generation moves).
    _data_dir = str(getattr(traj_mgr.cfg, "data_dir", "") or "")
    _ik_sfx = env_flags.get("DEXENGINE_ARM_IK_DIR_SUFFIX", "_cont25")  # same default as core names.ik_suffix
    # Unset-suffix hole: the guard above only checks when a suffix is given. Without one the newest pipeline writes
    # `_cont25` while the unsuffixed directory still holds the **previous generation** (objects current, arm on the old
    # placement: 283 vs 332 frames measured). So an unset suffix adopts `_cont25` automatically and **always logs it**.
    # With "_cont25" as the default this fallback is reached only when DEXENGINE_ARM_IK_DIR_SUFFIX="" is given.
    if not _ik_sfx and ik_dir is not None:
        _auto = ik_dir.rstrip("/") + "_cont25"
        if os.path.isdir(_auto):
            _ik_sfx = "_cont25"
            print(
                f"[arm_ik] suffix unset → _cont25 adopted automatically ({side}). "
                f"To be explicit: DEXENGINE_ARM_IK_DIR_SUFFIX=_cont25"
            )
    if _ik_sfx and ik_dir is not None:
        _cand = ik_dir.rstrip("/") + _ik_sfx
        if os.path.isdir(_cand):
            print(f"[arm_ik] DIR OVERRIDE ({side}): {ik_dir} -> {_cand}")
            ik_dir = _cand
        else:
            raise RuntimeError(
                f"[arm_ik][MIXED_CACHE] combo data but no suffixed cache ({_cand}) — no silent fallback to an "
                f"unsuffixed/unguarded cache. Build {_cand} in preprocessing or fix the suffix."
            )
    if ik_dir is not None and "/robots/" not in ik_dir.replace("\\", "/"):
        raise RuntimeError(
            f"[arm_ik][MIXED_CACHE] combo data (data_dir={_data_dir}) wired to a legacy arm-IK cache ({ik_dir}) — no "
            f"mixed training. The arm-IK cache must be a robots/<combo>/ product too "
            f"(robots.selection.combo_data_paths contract)."
        )
    if ik_dir is None or not os.path.isdir(ik_dir):
        raise RuntimeError(
            f"[arm_ik][MIXED_CACHE] combo data but no arm-IK cache directory ({ik_dir}) — "
            f"no single-IK-step fallback (~5 cm error)."
        )

    loader = traj_mgr._loader
    num_clips = loader.num_clips
    seq_names = loader.sequence_names
    arm_jpos = torch.zeros(num_clips, 7, dtype=torch.float32, device=env.device)  # frame-0 per clip

    # ── ENVZ: per-env 'desk-relative height dz' ──────────────────────────
    # With DEXENGINE_ENVZ=1 the dz of each clip is collected **as recorded in its cache's base_z_off** (single source,
    # no new file) and applied at reset to the robot root z of the envs assigned to that clip. Mutually exclusive with
    # the global DEXENGINE_ROBOT_Z_OFF (both would shift twice), refused below. Default 1 (on): 99.81% of the canonical
    # caches have dz≠0, so without ENVZ the first clip dies on MOUNT MISMATCH. On a legacy global cache without
    # base_z_off, ENVZ=1 takes the `if _envz_on:` branch with _dz_i=0.0 and behaves like off. DEXENGINE_ENVZ=0 gives
    # the old behaviour.
    _envz_on = env_flags.get("DEXENGINE_ENVZ") == "1"
    if _envz_on and abs(float(env_flags.get("DEXENGINE_ROBOT_Z_OFF") or 0)) > 1e-9:
        raise RuntimeError("[arm_ik][ENVZ] DEXENGINE_ENVZ=1 and a global DEXENGINE_ROBOT_Z_OFF cannot be combined")
    return _envz_on, arm_jpos, ik_dir, num_clips, seq_names


def _load_clip_ik_caches(env, _dz_arr, _envz_on, arm_jpos, ik_dir, num_clips, seq_names):
    """Per clip: the frame-0 arm joints, the all-frame cache, mount-consistency and base-xy guards."""
    loaded, missing_seqs, n_all, per_clip_all = _frame_cache_rsi_mid(num_clips)
    for i, seq_name in enumerate(seq_names):
        path = os.path.join(ik_dir, f"{seq_name}.npz")
        if os.path.isfile(path):
            data = np.load(path)
            _mount_consistency_guard(_dz_arr, _envz_on, data, i, path)
            # ── BASE-XY GUARD ─────────────────────────────────
            # The guard above looks at z only. But the joint angles are **relative to the base**, so an x·y split
            # misses the hand by as much, and nothing dies. Measured: DEXENGINE_RBY1_BASE_POS="-0.573,0.0,0.0" given to
            # the render only moved rby1 15 cm off its real base y (-0.15); the joint angles matched the cache exactly
            # and the fingertips missed the object by 0.141 m. Legacy caches have no base_xyz record and are skipped.
            if "base_xyz" in data.files:
                try:
                    from dexengine.robots.selection import resolve_robot_selection as _rrs_b

                    _spec_b = _rrs_b()[0]
                    _rt_b = np.asarray(getattr(_spec_b, "base_pos", None), dtype=float)
                except Exception:  # noqa: BLE001 — an unreadable robot spec leaves the base offset unknown (None)
                    _rt_b = None
                if _rt_b is not None and _rt_b.size == 3:
                    _cache_b = np.asarray(data["base_xyz"], dtype=float).reshape(-1)[:3]
                    _dxy = float(np.max(np.abs(_cache_b[:2] - _rt_b[:2])))
                    if _dxy > 1e-4:
                        raise RuntimeError(
                            f"[arm_ik] BASE-XY MISMATCH {path}: cache base_xy="
                            f"({_cache_b[0]:+.4f},{_cache_b[1]:+.4f}) != runtime "
                            f"({_rt_b[0]:+.4f},{_rt_b[1]:+.4f}) — the joint angles are base-relative, so "
                            f"every reset misses the hand by {_dxy * 100:.1f} cm. "
                            f"Never give DEXENGINE_*_BASE_POS to one side only."
                        )
            arm_jpos[i] = torch.tensor(data["arm_joint_positions"], dtype=torch.float32, device=env.device)
            loaded += 1
            if "arm_joint_positions_all" in data.files:
                per_clip_all[i] = np.asarray(data["arm_joint_positions_all"], dtype=np.float32)
                n_all += 1
        else:
            missing_seqs.append(seq_name)
    return loaded, missing_seqs, n_all, per_clip_all


def _mount_consistency_guard(_dz_arr, _envz_on, data, i, path):
    """MOUNT-CONSISTENCY GUARD The cache stores arm JOINT angles, which are base- relative: executing them on a base
    moved by DEXENGINE_ROBOT_Z_OFF displaces the hand by exactly that offset at every reset. A cache built without
    the offset therefore spawns the hand z_off away from the reference — measured SOK 298: −48.5 mm into the desk;
    BEP 069: +73 mm in the air — and NOTHING crashes, so the damage is silent (it read as "clip difficulty" for two
    weeks). Refuse to run rather than train on a wrong spawn. Legacy caches carry no record; they are accepted only
    when the runtime offset is 0 (all legacy runs were).
    """
    _exp_z = float(env_flags.get("DEXENGINE_ROBOT_Z_OFF") or 0)
    # DEXENGINE_MOUNT_TOL (bundle_generalize): allowed |cache - runtime|
    # mount gap in meters. Bundles group clips whose adopted dz differ by O(mm)
    # under ONE spawn height, so they raise this to the z-bucket half-width
    # (e.g. 0.0025). Keep it far below the cm-scale silent bugs this guard was
    # built to catch (SOK 298 −48.5 mm / BEP 069 +73 mm).
    # Default 0.0025 (2.5 mm, the bundle z-bucket half-width). With ENVZ=1 (the default) this guard is skipped
    # in the `if _envz_on:` branch above, so it matters only on explicit ENVZ=0 runs. DEXENGINE_MOUNT_TOL=1e-6
    # gives the original exact match.
    # +1e-9: with the tolerance at the z-bucket half-width, **values exactly on the boundary** are common (the
    # bundle's common z is the midrange, so the worst member is off by exactly the half-width). If float error
    # splits that boundary, training passes and only the render dies at the same z and TOL (a clip at
    # |−0.0003 − 0.0022| = exactly 2.5 mm died three times that way, each time holding the GPU for 40 minutes
    # in a stuck Isaac shutdown).
    _tol_raw = float(env_flags.get("DEXENGINE_MOUNT_TOL") or 0.0025)
    _tol = _tol_raw + 1e-9
    # Compare the value **before** the +1e-9: comparing `_tol != 1e-6` after it was always true and printed
    # "MOUNT TOL override" on runs that overrode nothing.
    if _tol_raw != 1e-6 and i == 0:
        print(f"[arm_ik] MOUNT TOL override: {_tol * 1000:.2f} mm (bundle z-bucket mode)")
    _rec_z = float(data["base_z_off"]) if "base_z_off" in data.files else None
    if _envz_on:
        # ENVZ: this clip's dz is applied per env as recorded in the cache, so the "global z match" guard does
        # not apply. Only insane values are filtered; bound = the largest |dz| the layer policy can stamp
        # (owner search_policy via _dz_bound_m; it widens with the layer list).
        _dz_i = 0.0 if _rec_z is None else _rec_z
        if abs(_dz_i) > _dz_bound_m() + 1e-6:
            raise RuntimeError(
                f"[arm_ik][ENVZ] insane base_z_off {_dz_i:+.4f} m (layer policy bound ±{_dz_bound_m():.4f}) in {path}"
            )
        _dz_arr[i] = _dz_i
    elif _rec_z is not None:
        if abs(_rec_z - _exp_z) > _tol:
            # The order of the remedies matters. These caches are baked with a **different dz per clip** (43
            # canonical values, 99.81% non-zero) and ENVZ is how they are read. A message that only said
            # "regenerate the cache" once nearly sent someone the opposite way (re-baking 2,139 healthy caches);
            # the real remedy is to turn ENVZ on.
            raise RuntimeError(
                f"[arm_ik] MOUNT MISMATCH {path}: cache base_z_off={_rec_z:+.4f} m "
                f"!= runtime DEXENGINE_ROBOT_Z_OFF={_exp_z:+.4f} m — every reset would "
                f"spawn the hand {(_exp_z - _rec_z) * 100:+.1f} cm off the reference.\n"
                f"  remedies (check in this order):\n"
                f"  1) when the cache carries per-clip dz, **DEXENGINE_ENVZ=1** is the answer "
                f"— no single global z can satisfy it (43 canonical cache dz values). "
                f"The RL switch envz (dexengine/rl/tasks/switches.py) is on by default.\n"
                f"  2) for a bundle z-bucket run, DEXENGINE_MOUNT_TOL = the bucket half-width.\n"
                f"  3) only when neither applies, regenerate the cache"
                f" with the arm IK stage (it resolves the placement)."
            )
    elif abs(_exp_z) > _tol:
        raise RuntimeError(
            f"[arm_ik] MOUNT UNVERIFIED {path}: cache predates base_z_off recording "
            f"but DEXENGINE_ROBOT_Z_OFF={_exp_z:+.4f} m — it was built for a standard-"
            f"height base and would spawn the hand {_exp_z * 100:+.1f} cm off the "
            f"reference. Regenerate it with the arm IK stage (scripts/stage2_ground.sh)."
        )


def _frame_cache_rsi_mid(num_clips):
    """All-frame cache (for RSI / mid-trajectory resets). Built lazily: a (num_clips, T_max, 7) padded tensor + per-clip
    frame counts, populated only from files that carry `arm_joint_positions_all`. env._arm_ik_cache_all_right stays
    None unless ALL loaded clips have the all-frame key (mixed availability -> disable RSI-arm-reset to avoid wrong-
    frame resets; frame-0 path is unaffected). Stashed on env as a side effect.
    """
    per_clip_all: list = [None] * num_clips
    n_all = 0
    loaded = 0
    missing_seqs: list[str] = []
    return loaded, missing_seqs, n_all, per_clip_all


def _install_all_frame_cache(
    env, side, _dz_arr, _envz_on, arm_jpos, ik_dir, loaded, n_all, num_clips, per_clip_all, seq_names
):
    """Install the per-side all-frame IK cache (and the env-z table) on env when every clip has one."""
    sfx = f"_{side}"  # SYMMETRIC attr naming: _arm_ik_cache_all_right / _left
    setattr(env, f"_arm_ik_cache_all{sfx}", None)
    setattr(env, f"_arm_ik_cache_all_len{sfx}", None)
    if _envz_on:
        setattr(env, f"_envz_clip_dz{sfx}", _dz_arr.to(env.device))
        print(
            f"[ENVZ] desk-relative height dz per clip ({side}): "
            + ", ".join(f"{v * 100:+.2f}cm" for v in _dz_arr.tolist())
        )
    if n_all == loaded and n_all > 0:
        t_max = max(a.shape[0] for a in per_clip_all if a is not None)
        cache_all = torch.zeros(num_clips, t_max, 7, dtype=torch.float32, device=env.device)
        lengths = torch.ones(num_clips, dtype=torch.long, device=env.device)
        for i, a in enumerate(per_clip_all):
            if a is None:
                cache_all[i, 0] = arm_jpos[i]  # frame-0 fallback (pad rest with frame-0 too)
                cache_all[i, 1:] = arm_jpos[i]
                continue
            ti = a.shape[0]
            cache_all[i, :ti] = torch.tensor(a, dtype=torch.float32, device=env.device)
            if ti < t_max:
                cache_all[i, ti:] = cache_all[i, ti - 1]  # pad with last frame
            lengths[i] = ti
        setattr(env, f"_arm_ik_cache_all{sfx}", cache_all)
        setattr(env, f"_arm_ik_cache_all_len{sfx}", lengths)
        # The `side` PARAM is the caller's slot name ("right" = the active slot, even on a LEFT-uni
        # build whose cfg dir points at arm_ik_poses_left). Publish an alias under the side the DATA
        # actually belongs to, so side-aware consumers (ARM_FF velocity feedforward) find it without
        # relying on the legacy slot name. Additive: the slot-named attrs above are untouched.
        _data_side = "left" if str(ik_dir).rstrip("/").endswith("_left") else "right"
        if _data_side != side:
            setattr(env, f"_arm_ik_cache_all_{_data_side}", cache_all)
            setattr(env, f"_arm_ik_cache_all_len_{_data_side}", lengths)
        print(
            f"[reset_arm_to_trajectory] Loaded ALL-FRAME arm IK poses "
            f"(num_clips={num_clips}, T_max={t_max}) — RSI mid-trajectory arm reset ENABLED; "
            f"published as _arm_ik_cache_all{sfx}"
            + (f" + alias _arm_ik_cache_all_{_data_side}" if _data_side != side else "")
        )
        _load_reset_cache(env, side, _data_side, ik_dir, seq_names, per_clip_all, cache_all, lengths)
    elif env_flags.get("DEXENGINE_ARM_IK_RESET_SUFFIX"):
        raise RuntimeError(
            "[ARM_IK_RESET] DEXENGINE_ARM_IK_RESET_SUFFIX needs the all-frame canonical cache "
            f"({n_all}/{loaded} clips carry arm_joint_positions_all in {ik_dir})"
        )


def _envz_apply_root_z(env, robot, traj_mgr, env_ids) -> None:
    """ENVZ: per-env robot root z = the cache base_z_off of that env's clip.

    The floating root, the reference and the objects are unchanged: actions and observations are (target − current)
    differences, so the base translation cancels, along the same math path as a global-z run.

    Call-site contract: the dz tensors (_envz_clip_dz_*) are made by loading the arm-IK cache, so this must run **after
    loading**. Inlined before the load point of reset_arm_to_trajectory, the first reset found no tensor and skipped
    silently: training hid it through repeated resets (only the first batch at z=0), but pose_render, with a single
    reset, never had it and its frame-0 overlay was off by dz.
    """
    # Default 1 (on), the same default as the _envz_on decision of _load_arm_ik_cache above (if the two disagreed,
    # the cache would collect dz while this skips applying it: a half state).
    if env_flags.get("DEXENGINE_ENVZ") != "1":
        return
    _dzt = getattr(env, "_envz_clip_dz_right", None)
    if _dzt is None:
        _dzt = getattr(env, "_envz_clip_dz_left", None)
    if _dzt is None:
        # No loaded cache here means the call-site contract was broken: report it loudly instead of skipping
        # silently.
        print("[ENVZ] WARNING: no dz tensor — called before the arm-IK cache was loaded (check the call site)")
        return
    _cid_z = traj_mgr.clip_ids[env_ids]
    _rz7 = robot.data.default_root_state[env_ids, :7].clone()
    _rz7[:, :3] += env.scene.env_origins[env_ids]
    _rz7[:, 2] += _dzt[_cid_z]
    robot.write_root_pose_to_sim(_rz7, env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids)
    # Split articulation: the left has **its own fixed base**; writing one side only would stand the two sides at
    # different heights. Without the split it is a no-op (byte-identical).
    from dexengine.rl.tasks.scene import split_artic as _split_artic_z

    _split_artic_z.mirror_root_pose(env, _rz7, env_ids)
    if not hasattr(env, "_envz_applied_logged"):
        env._envz_applied_logged = True
        print(
            f"[ENVZ] per-env root z applied (e.g. first reset batch dz "
            f"min={_dzt[_cid_z].min().item() * 100:+.2f}cm max={_dzt[_cid_z].max().item() * 100:+.2f}cm)"
        )
