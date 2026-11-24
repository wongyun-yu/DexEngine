"""Shared half of the raw-dataset -> MANO-layer readers (GRAB, ARCTIC, H2O, OakInk2).

The delivered MANO layer is the TACO keypoint format. Its rules were
recovered from the delivery and match the MANO keypoint library and TACO reader of this repository
(``dexengine/data/readers/mano_lib.py`` · ``taco.py``: forward kinematics, palm frame,
resampling, the object-only grounding):

* hand keypoints: the 16 MANO joints by forward kinematics (no pose blend shapes) and the five fingertips
  ``TIP_VERTS`` carried rigidly by their parent joint ``TIP_PARENTS``;
* wrist = joint 0; wrist orientation from the palm-rigid MCP joints (z: wrist -> middle MCP, auxiliary y: ring MCP
  -> index MCP, mirrored for the left hand);
* one clip-level z shift: the lowest vertex of the active objects' meshes at the first frame goes to z = 0, applied
  to both hands and both objects;
* the wrist-frame fingertips are computed at the source rate; every pose key is then resampled to 50 Hz (linear
  positions, quaternion Slerp with hemisphere continuity); the last fractional source interval is dropped.

A dataset module turns its raw files into world-frame (z-up, before grounding) joints, tips, object poses and object
meshes; ``build_clip`` does the rest and ``write_clip`` writes ``<out>/<clip>/{task_info.json,
0/trajectory_keypoints.npz}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

TARGET_DT = 0.02
#: fingertip vertices (thumb, index, middle, ring, pinky) and the MANO joint that carries each one rigidly
TIP_VERTS = (745, 317, 445, 556, 673)
TIP_PARENTS = (15, 3, 6, 12, 9)
#: MANO 16-joint indices used by the wrist frame
J_WRIST, J_INDEX_MCP, J_MIDDLE_MCP, J_RING_MCP = 0, 1, 4, 10
UNIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
#: Phase-1 key order of the delivered npz (contact_* keys are appended later by the contact detector)
POSE_KEYS = (
    "qpos_wrist_right",
    "qpos_finger_right",
    "qpos_obj_right",
    "qpos_wrist_left",
    "qpos_finger_left",
    "qpos_obj_left",
    "qpos_wrist_right_wrist_frame",
    "qpos_finger_right_wrist_frame",
    "qpos_wrist_left_wrist_frame",
    "qpos_finger_left_wrist_frame",
)
SIDES = ("right", "left")


# ------------------------------------------------------------------------------------------------- MANO model + FK
@dataclass(frozen=True)
class ManoModel:
    """The arrays of one MANO hand model that the keypoint FK needs (float64)."""

    side: str
    v_template: np.ndarray  # (778, 3)
    shapedirs: np.ndarray  # (778, 3, 10)
    j_regressor: np.ndarray  # (16, 778)
    parents: np.ndarray  # (16,), root = -1
    hands_mean: np.ndarray  # (45,)
    hands_components: np.ndarray  # (45, 45) PCA basis (rows = components)


_MODELS: dict[tuple[str, str], ManoModel] = {}


def load_mano(mano_dir: str | Path, side: str) -> ManoModel:
    """Load MANO_{RIGHT,LEFT}.pkl from `mano_dir` through smplx (cached). The published left model is used as is:
    the four datasets were fitted with it (dexengine/data/readers/mano_conventions.py; GRAB has no betas)."""
    key = (str(mano_dir), side)
    if key not in _MODELS:
        import smplx  # local import: only the readers that pose MANO need it

        m = smplx.MANO(str(mano_dir), is_rhand=(side == "right"), use_pca=True, num_pca_comps=45, flat_hand_mean=False)

        def f(t):
            return t.detach().cpu().numpy().astype(np.float64)

        parents = m.parents.detach().cpu().numpy().astype(np.int64).copy()
        parents[0] = -1
        _MODELS[key] = ManoModel(
            side=side,
            v_template=f(m.v_template),
            shapedirs=f(m.shapedirs),
            j_regressor=f(m.J_regressor),
            parents=parents,
            hands_mean=f(m.hand_mean).reshape(45),
            hands_components=np.asarray(m.np_hand_components, dtype=np.float64).reshape(45, 45),
        )
    return _MODELS[key]


@dataclass
class HandParams:
    """One hand's MANO parameters at the source rate, in the source frame.

    full_pose: (T, 48) absolute axis-angle (root + 15 joints; any hand mean already added).
    transl: (T, 3). root_at_transl=False: smplx/manopth additive translation (root at J0 + transl);
    True: the root joint is placed at transl (manotorch center_idx=0, as OakInk2 poses its hands).
    betas: (10,) shape, or v_template: (778, 3) subject template (GRAB) - exactly one of them.
    """

    full_pose: np.ndarray
    transl: np.ndarray
    betas: np.ndarray | None = None
    v_template: np.ndarray | None = None
    root_at_transl: bool = False


def pose_from_pca(model: ManoModel, pca: np.ndarray, flat_hand_mean: bool) -> np.ndarray:
    """(T, k) PCA coefficients -> (T, 45) absolute axis-angle (smplx convention)."""
    pca = np.asarray(pca, dtype=np.float64)
    pose = pca @ model.hands_components[: pca.shape[1]]
    return pose if flat_hand_mean else pose + model.hands_mean


def _rodrigues(aa: np.ndarray) -> np.ndarray:
    return R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(aa.shape[:-1] + (3, 3))


def hand_fk(model: ManoModel, hp: HandParams) -> tuple[np.ndarray, np.ndarray]:
    """(joints (T, 16, 3), tips (T, 5, 3)) in the source frame — MANO FK without pose blend shapes, tips rigid on
    their parent joint (mano_lib.mano_forward)."""
    if (hp.betas is None) == (hp.v_template is None):
        raise ValueError("HandParams needs exactly one of betas / v_template")
    if hp.v_template is not None:
        v_shaped = np.asarray(hp.v_template, dtype=np.float64)
        if v_shaped.shape != (778, 3):
            raise ValueError(f"v_template shape {v_shaped.shape} != (778, 3)")
    else:
        v_shaped = model.v_template + np.einsum("vcb,b->vc", model.shapedirs, np.asarray(hp.betas, float)[:10])
    J = model.j_regressor @ v_shaped  # (16, 3)
    pose = np.asarray(hp.full_pose, dtype=np.float64)
    T = pose.shape[0]
    if pose.shape != (T, 48):
        raise ValueError(f"full_pose shape {pose.shape} != (T, 48)")
    loc = _rodrigues(pose.reshape(T, 16, 3))  # (T, 16, 3, 3)
    grot = np.empty_like(loc)
    gpos = np.empty((T, 16, 3))
    for j in range(16):
        p = model.parents[j]
        if p < 0:
            grot[:, j], gpos[:, j] = loc[:, j], J[j]
        else:
            grot[:, j] = grot[:, p] @ loc[:, j]
            gpos[:, j] = gpos[:, p] + np.einsum("tab,b->ta", grot[:, p], J[j] - J[p])
    off = np.asarray(hp.transl, dtype=np.float64) - (gpos[:, 0] if hp.root_at_transl else 0.0)
    joints = gpos + off[:, None]
    tips = (
        np.stack(
            [
                np.einsum("tab,b->ta", grot[:, pj], v_shaped[v] - J[pj]) + gpos[:, pj]
                for v, pj in zip(TIP_VERTS, TIP_PARENTS, strict=True)
            ],
            1,
        )
        + off[:, None]
    )
    return joints, tips


def wrist_quat_wxyz(joints: np.ndarray, side: str) -> np.ndarray:
    """(T, 4) wxyz wrist orientation from palm-rigid MCP joints (taco.extract_hand_data, per frame)."""
    j = np.asarray(joints, dtype=np.float64)

    def nz(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    z = nz(j[:, J_MIDDLE_MCP] - j[:, J_WRIST])
    y_aux = nz(j[:, J_INDEX_MCP] - j[:, J_RING_MCP])
    x = nz(np.cross(y_aux, z))
    y = nz(np.cross(z, x))
    if side == "left":
        x, y = -x, -y
    # the writer stores the wrist as a rotation vector and re-reads it (taco.main: as_rotvec -> from_rotvec), so
    # every source quaternion has w >= 0 — the sign the resampler's hemisphere continuity starts from
    rotvec = R.from_matrix(np.stack([x, y, z], axis=-1)).as_rotvec()
    xyzw = R.from_rotvec(rotvec).as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def rigid_apply(Rm: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply x -> R x + t to (T, ..., 3) points; R (3, 3) or per frame (T, 3, 3), t (3,) or (T, 3)."""
    Rm, t, pts = np.asarray(Rm, float), np.asarray(t, float), np.asarray(pts, float)
    if Rm.ndim == 2:
        return pts @ Rm.T + t
    sh = pts.shape
    flat = pts.reshape(sh[0], -1, 3)
    out = np.einsum("tij,tkj->tki", Rm, flat) + t.reshape(sh[0], 1, 3)
    return out.reshape(sh)


def pose7_from_mats(T44: np.ndarray) -> np.ndarray:
    """(T, 4, 4) -> (T, 7) xyz + wxyz."""
    out = np.zeros((len(T44), 7))
    out[:, :3] = T44[:, :3, 3]
    out[:, 3:] = R.from_matrix(T44[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    return out


# ------------------------------------------------------------------------------------------------- resampling
def resample_times(num_frames: int, source_dt: float, target_dt: float = TARGET_DT) -> tuple[np.ndarray, np.ndarray]:
    """Source and fixed-rate target times; the last fractional source interval is dropped (mano_lib.resample_times)."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    src = np.arange(num_frames, dtype=np.float64) * source_dt
    if num_frames == 1:
        return src, np.zeros(1)
    dur = src[-1]
    n = int(np.floor(dur / target_dt + 1e-9)) + 1
    tgt = np.arange(n, dtype=np.float64) * target_dt
    if np.isclose(tgt[-1], dur, rtol=0.0, atol=1e-10):
        tgt[-1] = dur
    return src, tgt


def lerp(values: np.ndarray, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Linear interpolation along the first axis (taco._interp_along_time)."""
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    out = np.stack([np.interp(tgt, src, flat[:, c]) for c in range(flat.shape[1])], 1)
    return out.reshape((len(tgt),) + values.shape[1:])


def resample_pose(values: np.ndarray, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """(T, ..., 7) xyz + wxyz poses: linear positions, quaternion Slerp, hemisphere continuity before and after
    (taco._resample_pose_sequence)."""
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1, 7)
    out = np.zeros((len(tgt), flat.shape[1], 7))
    for i in range(flat.shape[1]):
        out[:, i, :3] = lerp(flat[:, i, :3], src, tgt)
        q = flat[:, i, 3:] / np.linalg.norm(flat[:, i, 3:], axis=-1, keepdims=True)
        for k in range(1, len(q)):
            if np.dot(q[k - 1], q[k]) < 0.0:
                q[k] *= -1.0
        res = Slerp(src, R.from_quat(q[:, [1, 2, 3, 0]]))(tgt).as_quat()[:, [3, 0, 1, 2]]
        for k in range(1, len(res)):
            if np.dot(res[k - 1], res[k]) < 0.0:
                res[k] *= -1.0
        out[:, i, 3:] = res
    return out.reshape((len(tgt),) + values.shape[1:])


# ------------------------------------------------------------------------------------------------- one clip
@dataclass
class ClipSource:
    """What a dataset module hands to build_clip — world frame (z-up), before grounding, at the source rate."""

    task: str
    dataset: str
    source_dt: float
    joints: dict  # side -> (T, 16, 3)
    tips: dict  # side -> (T, 5, 3)
    objects: dict  # side -> (T, 4, 4) or None (no object on that side)
    meshes: dict  # side -> (V, 3) object-local vertices in metres, or None
    object_names: dict  # side -> asset name under processed/<ds>/assets/objects, or None
    extra_series: dict = field(default_factory=dict)  # key -> ((T, k) values, dtype) lerped, after wrist-frame keys
    extra_task_info: dict = field(default_factory=dict)


def _pose7(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """(..., 3) positions + (..., 4) or (4,) wxyz -> (..., 7)."""
    out = np.zeros(pos.shape[:-1] + (7,))
    out[..., :3] = pos
    out[..., 3:] = quat
    return out


def ground_offset(objects: dict, meshes: dict, frame: int = 0) -> float:
    """z shift putting the lowest vertex of the active objects (at `frame`) on z = 0 (taco grounding)."""
    lows = []
    for s in SIDES:
        if objects.get(s) is None:
            continue
        if meshes.get(s) is None:
            raise ValueError(f"{s} object has a pose but no mesh — the ground cannot be set")
        T = objects[s][frame]
        lows.append(float((meshes[s] @ T[:3, :3][2] + T[2, 3]).min()))
    if not lows:
        raise ValueError("no active object — the ground cannot be set")
    return -min(lows)


def build_clip(cs: ClipSource) -> tuple[dict, dict]:
    """-> (npz arrays in the delivered key order, task_info dict)."""
    T = cs.joints["right"].shape[0]
    dz = ground_offset(cs.objects, cs.meshes)
    shift = np.array([0.0, 0.0, dz])
    src_arr = {}
    for s in SIDES:
        j = cs.joints[s] + shift
        tp = cs.tips[s] + shift
        wq = wrist_quat_wxyz(j, s)
        rot = R.from_quat(wq[:, [1, 2, 3, 0]])
        loc = np.stack([rot.inv().apply(tp[:, k] - j[:, 0]) for k in range(5)], 1)
        w, fi, fw, ww = (
            _pose7(j[:, 0], wq),
            _pose7(tp, UNIT_QUAT),
            _pose7(loc, UNIT_QUAT),
            _pose7(np.zeros((T, 3)), UNIT_QUAT),
        )
        ob = np.tile(np.eye(4), (T, 1, 1))
        if cs.objects.get(s) is not None:
            ob = np.array(cs.objects[s], dtype=np.float64, copy=True)
            ob[:, 2, 3] += dz
        src_arr[s] = {"wrist": w, "finger": fi, "obj": pose7_from_mats(ob), "wrist_wf": ww, "finger_wf": fw}
    src_t, tgt_t = resample_times(T, cs.source_dt)
    out = {}
    for s in SIDES:
        out[f"qpos_wrist_{s}"] = src_arr[s]["wrist"]
        out[f"qpos_finger_{s}"] = src_arr[s]["finger"]
        out[f"qpos_obj_{s}"] = src_arr[s]["obj"]
    for s in SIDES:
        out[f"qpos_wrist_{s}_wrist_frame"] = src_arr[s]["wrist_wf"]
        out[f"qpos_finger_{s}_wrist_frame"] = src_arr[s]["finger_wf"]
    out = {k: resample_pose(out[k], src_t, tgt_t) for k in POSE_KEYS}
    for k, (vals, dt) in cs.extra_series.items():
        out[k] = lerp(np.asarray(vals, dtype=np.float64), src_t, tgt_t).astype(dt)
    out["timestamps"] = tgt_t
    return out, task_info(cs)


def task_info(cs: ClipSource) -> dict:
    """task_info.json of the delivered layer (key order as delivered)."""

    def d(s):
        n = cs.object_names.get(s)
        return None if n is None else f"processed/{cs.dataset}/assets/objects/{n}"

    ti = {
        "task": cs.task,
        "dataset_name": cs.dataset,
        "robot_type": "mano",
        "embodiment_type": "bimanual",
        "data_id": 0,
        "right_object_mesh_dir": d("right"),
        "left_object_mesh_dir": d("left"),
        "right_object_convex_dir": d("right"),
        "left_object_convex_dir": d("left"),
        "ref_dt": TARGET_DT,
        "source_ref_dt": cs.source_dt,
    }
    ti.update(cs.extra_task_info)
    return ti


def write_clip(out_root: str | Path, clip: str, arrays: dict, ti: dict) -> Path:
    """Write <out_root>/<clip>/task_info.json and <out_root>/<clip>/0/trajectory_keypoints.npz."""
    d = Path(out_root) / clip
    (d / "0").mkdir(parents=True, exist_ok=True)
    (d / "task_info.json").write_text(json.dumps(ti, indent=2))
    np.savez(d / "0" / "trajectory_keypoints.npz", **arrays)
    return d


def run_cli(name: str, enumerate_fn, convert_fn, argv: list[str] | None = None, extra_args=None) -> int:
    """Common CLI: list clips or convert the named ones (or --all) into --out.

    enumerate_fn(args) -> {clip: handle}; convert_fn(args, clip, handle) -> (arrays, task_info)."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=f"{name}: raw dataset -> MANO layer (trajectory_keypoints.npz)")
    ap.add_argument("--raw", required=True, help="raw dataset root (the data/<ds>/raw layer)")
    ap.add_argument("--mano-dir", required=True, help="directory with MANO_RIGHT.pkl / MANO_LEFT.pkl")
    ap.add_argument("--out", help="output root = the mano/bimanual layer")
    ap.add_argument("--clips", nargs="*", default=[], help="clip names to convert")
    ap.add_argument("--all", action="store_true", help="convert every clip the raw data yields")
    ap.add_argument("--list", action="store_true", help="print the clip names and exit")
    if extra_args:
        extra_args(ap)
    a = ap.parse_args(argv)
    clips = enumerate_fn(a)
    if a.list:
        print("\n".join(sorted(clips)))
        return 0
    todo = sorted(clips) if a.all else a.clips
    if not todo or not a.out:
        ap.error("give --out and --clips ... or --all (or --list)")
    unknown = [c for c in todo if c not in clips]
    if unknown:
        raise SystemExit(f"{name}: unknown clips {unknown[:5]} (see --list)")
    for c in todo:
        arrays, ti = convert_fn(a, c, clips[c])
        write_clip(a.out, c, arrays, ti)
        print(f"[{name}] {c}: {len(arrays['timestamps'])} frames", file=sys.stderr)
    print(f"[{name}] wrote {len(todo)} clips to {a.out}", file=sys.stderr)
    print(
        f"[{name}] object assets: python -m dexengine.data.objects.build {name} --raw {a.raw} --mano {a.out} "
        f"--out <data/{name}/assets/objects>",
        file=sys.stderr,
    )
    return 0
