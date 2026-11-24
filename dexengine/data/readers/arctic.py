"""ARCTIC raw release -> MANO layer (`<clip>/{task_info.json, 0/trajectory_keypoints.npz}`).

Raw (``<raw>/data/arctic_data/data``):
  raw_seqs/<sNN>/<seq>.mano.npy  dict {right, left}: rot (T,3) · pose (T,45) · trans (T,3) m · shape (10,)
  raw_seqs/<sNN>/<seq>.object.npy (T, 7): articulation angle (rad) · rotation axis-angle · translation (mm)
  meta/object_vtemplates/<obj>/mesh.obj  (mm; the whole object at articulation 0)

Conventions (measured against the delivery, every frame of the verified clips):
  * MANO: no PCA, flat_hand_mean=False (pose + hands_mean), betas = shape, additive translation; ARCTIC's world is
    z-up, so the only change of frame is the grounding shift; 30 fps.
  * one articulated object per sequence on the right side; object pose = (R(rotvec), t / 1000); the grounding mesh
    is mesh.obj at the frame-0 global pose (articulation ignored).
  * qpos_obj_arti_right = the articulation angle, linearly resampled, stored float32; task_info.articulated_object
    joint_range = [min(0, min angle), max angle], widened to lo + 1e-6 when the angle never moves.
  * clip = ``<sNN>-<seq>__0``, task = ``<sNN>-<seq>``, object asset name = ``<seq>.split("_")[0]`` (object_name).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from dexengine.data.readers import mano_raw as C

DATASET = "arctic"
FPS = 30.0
ARTI_EPS = 1e-6
#: object templates `meta/object_vtemplates/<obj>/`: the whole mesh and the two part meshes (task_info key -> file; the
#: asset part files keep the raw names), in mm — read here and by the object asset builder (data/objects/build.py)
WHOLE_MESH = "mesh.obj"
PART_MESHES = {"bottom_mesh": "bottom.obj", "top_mesh": "top.obj"}
MESH_UNITS_PER_M = 1000.0


def data_root(raw: str | Path) -> Path:
    d = Path(raw) / "data" / "arctic_data" / "data"
    if not (d / "raw_seqs").is_dir():
        raise FileNotFoundError(f"ARCTIC raw_seqs missing under {d}")
    return d


def template_dir(raw: str | Path) -> Path:
    """The object template folders (one per object, named as the asset)."""
    return data_root(raw) / "meta" / "object_vtemplates"


def object_name(seq: str) -> str:
    """Asset name of a raw sequence `<obj>_<action>_<take>`."""
    return seq.split("_")[0]


def enumerate_clips(raw: str | Path) -> dict[str, Path]:
    """{clip: path of <seq>.mano.npy} for every raw sequence."""
    d = data_root(raw)
    return {
        f"{f.parent.name}-{f.name[: -len('.mano.npy')]}__0": f for f in sorted((d / "raw_seqs").glob("s*/*.mano.npy"))
    }


_MESH: dict[str, np.ndarray] = {}


def object_vertices(raw: str | Path, obj: str) -> np.ndarray:
    """mesh.obj of `obj` in metres (cached)."""
    if obj not in _MESH:
        p = template_dir(raw) / obj / WHOLE_MESH
        _MESH[obj] = (
            np.asarray(trimesh.load(str(p), force="mesh", process=False).vertices, dtype=np.float64) / MESH_UNITS_PER_M
        )
    return _MESH[obj]


def load_clip(raw: str | Path, mano_dir: str | Path, clip: str, mano_npy: Path) -> C.ClipSource:
    """Raw ARCTIC sequence -> ClipSource (world frame, before grounding)."""
    task = clip[: -len("__0")]
    seq = task.split("-", 1)[1]
    obj = object_name(seq)
    mano = np.load(mano_npy, allow_pickle=True).item()
    ob_raw = np.load(str(mano_npy).replace(".mano.npy", ".object.npy"), allow_pickle=True)
    T = min(len(ob_raw), len(mano["right"]["trans"]), len(mano["left"]["trans"]))
    joints, tips = {}, {}
    for side in C.SIDES:
        p = mano[side]
        m = C.load_mano(mano_dir, side)
        pose = np.concatenate([np.asarray(p["rot"], float)[:T], np.asarray(p["pose"], float)[:T] + m.hands_mean], 1)
        hp = C.HandParams(
            full_pose=pose, transl=np.asarray(p["trans"], float)[:T], betas=np.asarray(p["shape"], float).reshape(10)
        )
        joints[side], tips[side] = C.hand_fk(m, hp)
    ob = np.asarray(ob_raw, dtype=np.float64)[:T]
    T44 = np.tile(np.eye(4), (T, 1, 1))
    T44[:, :3, :3] = R.from_rotvec(ob[:, 1:4]).as_matrix()
    T44[:, :3, 3] = ob[:, 4:7] / 1000.0
    angle = np.asarray(ob_raw)[:T, 0]
    lo = min(0.0, float(angle.min()))
    hi = float(angle.max())
    if hi <= lo:
        hi = lo + ARTI_EPS
    arti = {"side": "right", "joint_axis": [0, 0, -1], "joint_range": [lo, hi], "joint_pos": [0, 0, 0], **PART_MESHES}
    return C.ClipSource(
        task=task,
        dataset=DATASET,
        source_dt=1.0 / FPS,
        joints=joints,
        tips=tips,
        objects={"right": T44, "left": None},
        meshes={"right": object_vertices(raw, obj), "left": None},
        object_names={"right": obj, "left": None},
        extra_series={"qpos_obj_arti_right": (angle[:, None], np.float32)},
        extra_task_info={"articulated_object": arti},
    )


def convert(raw: str | Path, mano_dir: str | Path, clip: str, handle: Path) -> tuple[dict, dict]:
    """One clip -> (npz arrays, task_info)."""
    return C.build_clip(load_clip(raw, mano_dir, clip, handle))


def main(argv: list[str] | None = None) -> int:
    return C.run_cli(DATASET, lambda a: enumerate_clips(a.raw), lambda a, c, h: convert(a.raw, a.mano_dir, c, h), argv)


if __name__ == "__main__":
    sys.exit(main())
