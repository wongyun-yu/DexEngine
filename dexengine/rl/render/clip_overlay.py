"""MANO reference fingertip overlay of the per-clip render (``--viz_mano_fingertips``).

The retargeting reference fingertips (``body_positions.npz`` ``<side>_fingertip_pos``, trajectory frame) are drawn as
green spheres, mapped to world with the same transform as the wrist markers: env origin + desk height, then rotated by
``traj_mgr.root_quat_rotated``.
"""

from __future__ import annotations

import os

import torch


def _mano_fingertip_overlay(raw_env, args_cli, traj_mgr):
    """Build the overlay marker for the active hand sides.

    Returns (marker, local tips [T, 5 * n_sides, 3], root quat, root pos [N, 3], slowdown factor); marker and tips are
    None when the flag is off or the clip has no ``body_positions.npz``.
    """
    _ft_marker = None
    _ft_local = None
    _ft_root_q = None
    _ft_root_pos = None
    _ft_slowdown = 1.0
    if getattr(args_cli, "viz_mano_fingertips", False):
        import numpy as _npft

        _seq = args_cli.sequence
        _ft_path = next(
            (p for p in (os.path.join(args_cli.data_dir, _seq, "0", "body_positions.npz"),) if os.path.isfile(p)), None
        )
        if _ft_path is None:
            print(f"  [Render] WARNING: --viz_mano_fingertips: no body_positions.npz for '{_seq}'; overlay skipped")
        else:
            # Tips are drawn for every active side (active_viz_sides, the same source as the arrow markers): a left-hand
            # render gets its tips and a parked hand draws none. Concatenated as (T, 5 * n_sides, 3).
            from dexengine.rl.tasks.env_base import active_viz_sides

            _ft_sides = active_viz_sides(raw_env.scene)
            _ft_npz = _npft.load(_ft_path, allow_pickle=True)
            _ft_arr = _npft.concatenate(
                [_ft_npz[f"{s}_fingertip_pos"][:, :5, :] for s in _ft_sides], axis=1
            )  # (T, 5*n, 3)
            # Frame-0 repair pads the episode with n_blend_effective frames during which MANO holds frame 0, and the
            # overlay
            # is indexed with the padded episode counter. The pad is decided by length: a legacy tree's body_positions
            # has
            # T - pad rows and gets pad copies of row 0 prepended (without them the tips lead the reference by pad
            # frames);
            # a v2 tree computed body_positions on the padded tree and already has T rows (prepending would make them
            # lag).
            _f0_meta = os.path.join(args_cli.data_dir, _seq, "F0_REPAIR.json")
            _t_kin = None
            _kin_p = os.path.join(args_cli.data_dir, _seq, "0", "trajectory_kinematic.npz")
            if os.path.isfile(_kin_p):
                _t_kin = int(_npft.load(_kin_p)["qpos"].shape[0])
            if os.path.isfile(_f0_meta):
                import json as _jsft

                with open(_f0_meta) as _fh:
                    _pad = int(_jsft.load(_fh).get("n_blend_effective", 0) or 0)
                if _pad > 0 and (_t_kin is None or len(_ft_arr) + _pad == _t_kin):
                    _ft_arr = _npft.concatenate([_npft.repeat(_ft_arr[0:1], _pad, axis=0), _ft_arr], axis=0)
                    print(f"  [Render] fingertip overlay: +{_pad} pad frames (legacy unpadded FK sync)")
                elif _pad > 0 and _t_kin is not None and len(_ft_arr) == _t_kin:
                    print(f"  [Render] fingertip overlay: padded FK (T={_t_kin}) — prepend skipped (v2)")
                elif _pad > 0:
                    print(
                        f"  [Render] WARNING: overlay T={len(_ft_arr)} pad={_pad} kin={_t_kin} "
                        f"— unknown length convention, prepend skipped"
                    )
            _ft_local = torch.tensor(_ft_arr, dtype=torch.float32, device=raw_env.device)
            print(f"  [Render] MANO fingertip overlay sides={_ft_sides} ({_ft_local.shape[1]} tips/frame)")
            import isaaclab.sim as _simft
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            _ft_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/mano_fingertips",
                markers={
                    "tip": _simft.SphereCfg(
                        radius=0.009, visual_material=_simft.PreviewSurfaceCfg(diffuse_color=(0.05, 0.95, 0.15))
                    )
                },
            )
            _ft_marker = VisualizationMarkers(_ft_cfg)
            _ft_root_pos = raw_env.scene.env_origins.clone()
            _ft_root_pos[:, 2] += traj_mgr._desk_height
            _ft_root_q = traj_mgr.root_quat_rotated
            _ft_slowdown = float(getattr(traj_mgr, "_slowdown_factor", 1.0))
            print(f"  [Render] MANO fingertip overlay ON ({_ft_local.shape[0]} frames from {_ft_path})")
    return _ft_marker, _ft_local, _ft_root_q, _ft_root_pos, _ft_slowdown
