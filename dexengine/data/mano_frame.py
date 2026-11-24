#!/usr/bin/env python3
"""**Single owner of turning MANO (human hand) keypoints into one tree window in the base (yaw) frame** (D39).

Why a separate module: two consumers use the same measurement, the case layer (`cases.taxonomy`, D35 wrist and
fingertips) and the GRAB desk oracle (`dexengine.data.support.grab`, D14′ `wrist_below_fp_m`). Measured separately, the
same
−0.05 m threshold would be read in two frames. Importing `cases.taxonomy` would make the oracle pull in trimesh,
dexengine.data.contact.geom and the hand specs, so the one function lives in **a dependency-free module**; it is not a
copy.

Dependencies: numpy · scipy (rotation) · `h2o_frame` in the readers package (h2o desk plane rotation) · the path
accessor `paths` (data root) only. The hand is not an input.

**Which root the MANO folder is joined to** is also decided here alone (`mano_dir` · `mano_keypoints_path`).
`provenance.mano_dir` is text relative to the data root; joined to the code tree it points nowhere in a separate
worktree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Keypoint file inside the MANO folder (take 0); owner of this text.
MANO_KP_REL = "0/trajectory_keypoints.npz"


def mano_dir(ti: dict, root=None) -> Path:
    """MANO folder of task_info (`provenance.mano_dir`), joined to the **data root** (`paths.main_root()` · env
    DEXENGINE_MAIN_ROOT). `root` is used only when the caller already holds the data root. KeyError without the record
    (no substitute path)."""
    from dexengine.core import (
        paths as P,  # noqa: PLC0415 — accessor of the data root (rule owner dexengine.core.layout.data_root)
    )

    return Path(P.main_root() if root is None else root) / ti["provenance"]["mano_dir"]


def mano_keypoints_path(ti: dict, root=None) -> Path:
    """Path of the MANO keypoint npz: MANO_KP_REL under `mano_dir`."""
    return mano_dir(ti, root) / MANO_KP_REL


def window_start(ti: dict) -> int:
    """First MANO layer row (50 Hz) of this tree window = segment.start + frame_map.offset; the index rule of
    `mano_keypoints_base`."""
    sg = ti.get("segment") or {}
    return int(sg.get("start", 0) or 0) + int(((sg.get("frame_map") or {}).get("offset", 0)) or 0)


def mano_keypoints_base(ti: dict, ds: str, clip: str, T: int, hands) -> dict:
    """MANO fingertips (T,5,3) and wrist (T,3) in **this tree window of the base (yaw) frame**; the same rule as the
    sidecar `sidecar_compute.compute_side` (idx = arange(T) + segment.start + frame_map.offset · M = Rz(frame.yaw_deg) ·
    h2o applies the desk plane rotation first). Another rule would put contact and speed in different frames. Fails
    when keypoints are missing or the window runs past them (no robot substitute)."""
    from scipy.spatial.transform import Rotation as _R

    kp = np.load(mano_keypoints_path(ti))
    kp = {k: kp[k] for k in kp.files}
    if ds == "h2o":
        from dexengine.data.readers import h2o_frame as HF  # noqa: PLC0415

        kp = HF.apply_rigid_keypoints(kp, HF.tilt_transform(clip)["M"])
    a = window_start(ti)
    psi = float((ti.get("frame") or {}).get("yaw_deg", 0.0))
    M = _R.from_euler("z", psi, degrees=True).as_matrix()
    out = {"tips": {}, "wrist": {}}
    for s in hands:
        kf, kw = f"qpos_finger_{s}", f"qpos_wrist_{s}"
        if kf not in kp or kw not in kp:
            raise KeyError(
                f"{clip}: MANO keypoints lack {kf}/{kw} — the D35 rule (MANO speed, wrist) cannot be measured"
            )
        Tm = len(kp[kf])
        if a + T > Tm + 2:
            raise ValueError(f"{clip}/{s}: MANO keypoint length {Tm} < window {a}+{T} — different generation")
        idx = np.clip(np.arange(T) + a, 0, Tm - 1)
        out["tips"][s] = np.asarray(kp[kf], float)[idx, :, :3] @ M.T
        out["wrist"][s] = np.asarray(kp[kw], float)[idx, :3] @ M.T
    return out
