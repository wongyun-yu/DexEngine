"""OakInk2 raw release -> MANO layer (`<clip>/{task_info.json, 0/trajectory_keypoints.npz}`).

Raw (``<raw>/hf``):
  anno_preview/<scene>__<id>++seq__<hash>__<date>.pkl  raw_mano[frame]: {rh,lh}__pose_coeffs (16 quaternions wxyz,
      absolute — flat hand mean) · {rh,lh}__tsl (root joint position) · {rh,lh}__betas; obj_transf[rid][frame] (4x4,
      object -> world, metres); 120 Hz mocap frames
  program/program_info/<same stem>.json  ordered stages: key "(lh_interval, rh_interval)" -> obj_list_{rh,lh} …
  obj/object_repair/align_ds/<rid>/model.obj  object mesh (metres)

Conventions (measured against the delivery, every frame of the verified clips):
  * one clip per program stage: ``oakink2_<primitive>__<scene_NN>_<hash12>_s<stage:03d>``; the window is
    [min of the non-null hand interval starts, max of their ends) in mocap frames (end exclusive).
  * MANO: smplx flat_hand_mean, no PCA, the root joint placed at tsl (manotorch center_idx=0); both hands over the
    whole window. The OakInk2 world is y-up: everything is turned by Rx(+90 deg) to z-up.
  * right object = obj_list_rh[0], left object = obj_list_lh[0] (asset name ``<side>_<rid lower, '@' -> '_'>``).
  * a stage whose chosen objects are empty, unposed (rid not in obj_transf) or without a mesh yields no clip
    (the delivery's 2,840 - 12 = 2,828).
"""

from __future__ import annotations

import ast
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from dexengine.data.readers import mano_raw as C

DATASET = "oakink2"
MOCAP_HZ = 120.0
#: y-up OakInk2 world -> z-up
G = R.from_euler("x", 90, degrees=True).as_matrix()
MESH_SUB = Path("obj") / "object_repair" / "align_ds"
MESH_FILE = "model.obj"


def hf_root(raw: str | Path) -> Path:
    h = Path(raw) / "hf"
    if not (h / "anno_preview").is_dir():
        raise FileNotFoundError(f"OakInk2 anno_preview missing under {h}")
    return h


def asset_name(side: str, rid: str) -> str:
    return f"{side}_{rid.lower().replace('@', '_')}"


def _stages(h: Path, stem: str) -> list[tuple[tuple, dict]]:
    d = json.loads((h / "program" / "program_info" / f"{stem}.json").read_text())
    return [(ast.literal_eval(k), v) for k, v in d.items()]


def _chosen(v: dict) -> dict:
    return {"right": (v.get("obj_list_rh") or [None])[0], "left": (v.get("obj_list_lh") or [None])[0]}


_PKL: dict[str, dict] = {}


def _load_pkl(path: Path) -> dict:
    """anno_preview pickle (one cached; a stage list is converted in pickle order)."""
    if str(path) not in _PKL:
        _PKL.clear()
        with open(path, "rb") as fh:
            _PKL[str(path)] = pickle.load(fh)
    return _PKL[str(path)]


def stage_names(raw: str | Path) -> dict[str, tuple[Path, int]]:
    """Every program stage (no exclusion): {clip: (pkl path, stage index)}."""
    h = hf_root(raw)
    out = {}
    for f in sorted((h / "anno_preview").glob("*.pkl")):
        stem = f.name[:-4]
        hash12 = re.search(r"seq__([0-9a-f]+)__", stem).group(1)[:12]
        scene = stem.split("__")[0]
        for s, (_k, v) in enumerate(_stages(h, stem)):
            out[f"oakink2_{v['primitive']}__{scene}_{hash12}_s{s:03d}"] = (f, s)
    return out


def exclusion(raw: str | Path, pkl: Path, stage: int) -> str | None:
    """Why a stage yields no clip (None = it does)."""
    h = hf_root(raw)
    _iv, v = _stages(h, pkl.name[:-4])[stage]
    ch = {s: r for s, r in _chosen(v).items() if r is not None}
    if not ch:
        return "no object in obj_list_rh / obj_list_lh"
    ot = _load_pkl(pkl)["obj_transf"]
    for s, rid in ch.items():
        if rid not in ot:
            return f"{s} object {rid} has no pose (obj_transf)"
        if not (h / MESH_SUB / rid / MESH_FILE).is_file():
            return f"{s} object {rid} has no mesh ({MESH_SUB}/{rid}/{MESH_FILE})"
    return None


def enumerate_clips(raw: str | Path) -> dict[str, tuple[Path, int]]:
    """{clip: (pkl, stage)} for every stage that yields a clip (loads each pickle once)."""
    return {c: hd for c, hd in stage_names(raw).items() if exclusion(raw, *hd) is None}


#: cheap lookup for converting named clips (exclusion is checked at conversion time)
candidate_clips = stage_names


def load_clip(raw: str | Path, mano_dir: str | Path, clip: str, handle: tuple[Path, int]) -> C.ClipSource:
    """One program stage -> ClipSource (z-up world, before grounding)."""
    pkl, stage = handle
    why = exclusion(raw, pkl, stage)
    if why:
        raise ValueError(f"{clip}: excluded — {why}")
    h = hf_root(raw)
    (lh_iv, rh_iv), v = _stages(h, pkl.name[:-4])[stage]
    ivs = [x for x in (lh_iv, rh_iv) if x]
    a0, b0 = min(x[0] for x in ivs), max(x[1] for x in ivs)
    frames = np.arange(a0, b0)
    d = _load_pkl(pkl)
    rm = d["raw_mano"]
    missing = [f for f in frames if f not in rm]
    if missing:
        raise ValueError(f"{clip}: raw_mano lacks {len(missing)} window frames (first {missing[0]})")
    joints, tips = {}, {}
    for side, sk in (("right", "rh"), ("left", "lh")):
        q = np.stack([np.asarray(rm[f][f"{sk}__pose_coeffs"]).reshape(16, 4) for f in frames]).astype(np.float64)
        aa = R.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(len(frames), 48)
        tsl = np.stack([np.asarray(rm[f][f"{sk}__tsl"]).reshape(3) for f in frames]).astype(np.float64)
        be = np.asarray(rm[frames[0]][f"{sk}__betas"]).reshape(10).astype(np.float64)
        hp = C.HandParams(full_pose=aa, transl=tsl, betas=be, root_at_transl=True)
        j, t = C.hand_fk(C.load_mano(mano_dir, side), hp)
        joints[side], tips[side] = j @ G.T, t @ G.T
    objects, meshes, names = {}, {}, {}
    G4 = np.eye(4)
    G4[:3, :3] = G
    for side, rid in _chosen(v).items():
        if rid is None:
            objects[side] = meshes[side] = names[side] = None
            continue
        ot = d["obj_transf"][rid]
        Ts = np.stack([np.asarray(ot[f], dtype=np.float64) for f in frames])
        objects[side] = G4 @ Ts
        mp = h / MESH_SUB / rid / MESH_FILE
        meshes[side] = np.asarray(trimesh.load(str(mp), force="mesh", process=False).vertices, dtype=np.float64)
        names[side] = asset_name(side, rid)
    return C.ClipSource(
        task=clip,
        dataset=DATASET,
        source_dt=1.0 / MOCAP_HZ,
        joints=joints,
        tips=tips,
        objects=objects,
        meshes=meshes,
        object_names=names,
    )


def convert(raw: str | Path, mano_dir: str | Path, clip: str, handle: tuple[Path, int]) -> tuple[dict, dict]:
    """One clip -> (npz arrays, task_info)."""
    return C.build_clip(load_clip(raw, mano_dir, clip, handle))


def main(argv: list[str] | None = None) -> int:
    return C.run_cli(
        DATASET,
        lambda a: enumerate_clips(a.raw) if (a.list or a.all) else stage_names(a.raw),
        lambda a, c, hd: convert(a.raw, a.mano_dir, c, hd),
        argv,
    )


if __name__ == "__main__":
    sys.exit(main())
