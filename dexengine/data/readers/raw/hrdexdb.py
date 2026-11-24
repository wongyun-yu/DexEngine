"""Scene object oracle — HRDexDB raw loader (readers/raw/common owns the thresholds, the reduction and the classes).

Raw (<hrdexdb raw layer>/release, the official release layout): human/<object>/<seq>/hand/mano_params/NNNNN.json
(global_orient (1,1,3,3) · hand_pose (1,15,3,3) rotation matrices · betas · transl · joints (21,3) m) ·
object_6d_pose_v2/human/<object>_<seq>.npz (frame_<i> -> 4x4; v1 when no v2) · assets/mesh_v2/<object>/<object>.obj.
One right hand, one object per episode, 30 fps. Clip `<object>__<seq>[_segNN][_tN]` = the episode; a segment / cut =
the tree's segment.start/end (50 Hz from the episode start) -> round(x * 30 / 50).
MANO convention = CONV, chosen by `validate`: two oracles — the release's own 21 joints (order-free nearest-joint error)
and readers/raw/common.carried_distances.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from dexengine.data.readers.raw import common as C

FPS, TREE_HZ = 30.0, 50.0
CLIP_RE = re.compile(r"(?P<task>(?P<obj>.+?)__(?P<seq>\d+))(?P<seg>_seg\d+)?(?P<cut>_t\d+)?")
#: name -> (flat_hand_mean, transl = wrist position (re-centre on joint 0) instead of additive)
CONVS = {"A": (True, False), "B": (False, False), "C": (True, True), "D": (False, True)}
CONV = "A"  # validated: tables/hrdexdb_mano_validate.json (joint oracle median 0.0 mm vs 18.7 / 62 mm)
VAL_V = 0.10  # HRDexDB lifts are slow (max object speed ~0.25-0.28 m/s) — the carried oracle runs at 0.10 m/s
_MANO: dict = {}
obj_key = str


class Raw(C.NoParts):
    def __init__(self, raw_root: Path):
        self.r = Path(raw_root) / "release"
        if not (self.r / "human").is_dir():
            raise FileNotFoundError(f"HRDexDB release missing: {self.r}")
        self.task = {
            f"{o.name}__{s.name}": (o.name, s.name)
            for o in (self.r / "human").iterdir()
            for s in o.iterdir()
            if (s / "hand" / "mano_params").is_dir()
        }


def seq_key(clip: str) -> str:
    return CLIP_RE.fullmatch(clip).group("task")


def _rotvec(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_matrix(m.reshape(-1, 3, 3)).as_rotvec()


def load_seq(raw: Raw, key: str) -> dict:
    obj, sid = raw.task[key]
    files = sorted((raw.r / "human" / obj / sid / "hand" / "mano_params").glob("*.json"))
    go, hp, be, tr, jt = [], [], [], [], []
    for f in files:
        d = json.loads(f.read_text())
        go.append(_rotvec(np.asarray(d["global_orient"], float))[0])
        hp.append(_rotvec(np.asarray(d["hand_pose"], float)).reshape(45))
        be.append(np.asarray(d["betas"], float).reshape(10))
        tr.append(np.asarray(d["transl"], float).reshape(3))
        jt.append(np.asarray(d["joints"], float))
    T = len(files)
    pose = None
    for v in ("object_6d_pose_v2", "object_6d_pose_v1"):
        p = raw.r / v / "human" / f"{obj}_{sid}.npz"
        if p.is_file():
            with np.load(p) as z:
                pose = np.stack([z[f"frame_{i}"] if f"frame_{i}" in z else np.full((4, 4), np.nan) for i in range(T)])
            break
    if pose is None:
        raise FileNotFoundError(f"{key}: no object_6d_pose")
    mesh = raw.r / "assets" / "mesh_v2" / obj / f"{obj}.obj"
    return {
        "go": np.asarray(go),
        "hp": np.asarray(hp),
        "be": np.asarray(be),
        "tr": np.asarray(tr),
        "joints": np.asarray(jt),
        "obj": obj,
        "pose": pose.astype(float),
        "mesh": mesh if mesh.is_file() else None,
        "T": T,
    }


def window(raw: Raw, seq: dict, clip: str, ti: dict | None):
    m = CLIP_RE.fullmatch(clip)
    w0, w1, how = 0, seq["T"], "episode"
    if m.group("seg") or m.group("cut"):
        seg = (ti or {}).get("segment") or {}
        if seg.get("start") is None or seg.get("end") is None:
            raise ValueError("segment/cut clip without tree segment.start/end")
        w0, w1, how = int(round(seg["start"] * FPS / TREE_HZ)), int(round(seg["end"] * FPS / TREE_HZ)), "segment"
    return (0, seq["T"]), w0, w1, {"lh": False, "rh": True}, set(), how


def fk(seq: dict, idx, conv: str):
    import torch

    flat, wrist = CONVS[conv]
    m = C.mano_layer(_MANO["dir"], "rh", "hrdexdb", use_pca=False, flat_hand_mean=flat)
    with torch.no_grad():
        o = m(
            global_orient=torch.tensor(seq["go"][idx], dtype=torch.float32),
            hand_pose=torch.tensor(seq["hp"][idx], dtype=torch.float32),
            betas=torch.tensor(seq["be"][idx], dtype=torch.float32),
            transl=torch.tensor(seq["tr"][idx] * (0 if wrist else 1), dtype=torch.float32),
        )
    v, j = o.vertices.numpy().astype(float), o.joints.numpy().astype(float)
    if wrist:
        sh = seq["tr"][idx][:, None] - j[:, :1]
        v, j = v + sh, j + sh
    return v, j


def measure(raw: Raw, seq: dict, key: tuple, conv: str | None = None) -> dict:
    idx = list(range(key[0], key[1]))
    v, _ = fk(seq, idx, conv or CONV)
    hv = {"rh": v, "lh": np.full_like(v, np.nan)}
    T = seq["pose"][idx]
    poses = {seq["obj"]: (seq["mesh"], T[:, :3, :3], T[:, :3, 3], np.isfinite(T[:, 0, 0]))}
    return C.measure_objects(idx, hv, poses, 1.0 / FPS)


def validate(raw: Raw, keys: list[str]) -> dict:
    res = {c: {"carried": [], "joint": []} for c in CONVS}
    for k in keys:
        seq = load_seq(raw, k)
        idx = list(range(seq["T"]))
        for c in CONVS:
            st = measure(raw, seq, (0, seq["T"]), conv=c)
            res[c]["carried"] += list(C.carried_distances(st, np.arange(seq["T"]) / FPS, v_min=VAL_V))
            _, j = fk(seq, idx, c)
            g = seq["joints"]  # (T,21,3) release joints
            res[c]["joint"] += list(np.min(np.linalg.norm(g[:, :, None] - j[:, None, :16], axis=-1), axis=-1).ravel())
    return {
        c: {
            "carried": C.summarize_distances(np.asarray(r["carried"])),
            "joint_nearest": C.summarize_distances(np.asarray(r["joint"])),
        }
        for c, r in res.items()
    }
