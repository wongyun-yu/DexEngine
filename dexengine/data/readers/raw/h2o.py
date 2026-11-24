"""Scene object oracle — H2O raw loader (readers/raw/common owns the thresholds, the reduction and the classes).

Raw (<h2o raw layer>/dl, the official pose release): pose<k>/subject<k>/<scene>/<take>/cam4/
  hand_pose/NNNNNN.txt   [left valid, 21 joints x 3, right valid, 21 joints x 3]  (camera frame, metres)
  obj_pose_rt/NNNNNN.txt [object class, 4x4 row-major object-to-camera]
and objraw/object/<name>/<name>.obj. H2O publishes hand JOINTS, not a MANO surface: the hand points here are the 21
joints — the reference contact threshold is TOUCH_M_JOINTS (from the carried-frame distances, owned here), not
scene_requirement.TOUCH_M, and the six touch points are the wrist joint and TIP_JOINTS (no MANO surface to regress).
One object per take; both hands active; 30 fps.
Clip `subject<k>_<scene>__take<n>_cam4[_segNN][_tN]` = the take; a segment / cut = the tree's segment.start/end (50 Hz
from the take start) -> round(x * 30 / 50). Object class -> name = CLASS_NAMES (H2O's object list), checked against the
trees by `validate` (every tree's main obj_id must equal the class name of its take).
Convention = CONV, chosen by `validate` (readers/raw/common.carried_distances on the joints; controls: transposed
rotation · hands shifted 15 frames in time).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from dexengine.data.readers.raw import common as C
from dexengine.data.readers.raw import scene_classify as CL

FPS, TREE_HZ = 30.0, 50.0
CLIP_RE = re.compile(
    r"(?P<task>(?P<sub>subject\d+)_(?P<scene>[a-z]\d+)__take(?P<take>\d+)_(?P<cam>cam\d))"
    r"(?P<seg>_seg\d+)?(?P<cut>_t\d+)?"
)
CLASS_NAMES = {1: "book", 2: "espresso", 3: "lotion", 4: "spray", 5: "milk", 6: "cocoa", 7: "chips", 8: "cappuccino"}
CONVS = {"A": ("rowmajor", 0), "B": ("transposed", 0), "C": ("rowmajor", 15)}
CONV = "A"  # validated: tables/h2o_mano_validate.json (keypoints 42.9 % vs 5.2 % <= 10 mm · carried 2.2 vs 3.1 mm)
VAL_V = 0.10  # H2O hand-overs are slow — the carried oracle runs at 0.10 m/s
TOUCH_M_JOINTS = 0.010  # joint <= this from the surface = contact (validated: carried p90 5.0 mm · 98.6 % <= 10 mm)
TIP_JOINTS = [4, 8, 12, 16, 20]  # fingertips of H2O's 21 joints (wrist 0 · thumb 1–4 · index · middle · ring · pinky)
_MANO: dict = {}
obj_key = str


class Raw(C.NoParts):
    def __init__(self, raw_root: Path):
        self.r = Path(raw_root) / "dl"
        self.task = {}
        for d in self.r.glob("pose*/subject*/*/*/cam*"):
            sub, scene, take, cam = d.parts[-4:]
            self.task[f"{sub}_{scene}__take{take}_{cam}"] = d
        if not self.task:
            raise FileNotFoundError(f"H2O poses missing under {self.r}")
        self.mesh = self.r / "objraw" / "object"


def seq_key(clip: str) -> str:
    return CLIP_RE.fullmatch(clip).group("task")


def load_seq(raw: Raw, key: str) -> dict:
    d = raw.task[key]
    fs = sorted((d / "hand_pose").glob("*.txt"))
    hp = np.stack([np.loadtxt(f) for f in fs])
    ob = np.stack([np.loadtxt(d / "obj_pose_rt" / f.name) for f in fs])
    kp = np.stack([np.loadtxt(d / "obj_pose" / f.name)[1:].reshape(21, 3) for f in fs])  # release's object keypoints
    cls = {int(round(x)) for x in ob[:, 0] if x > 0}
    if len(cls) != 1:
        raise ValueError(f"{key}: object classes {cls}")
    name = CLASS_NAMES[cls.pop()]
    mesh = raw.mesh / name / f"{name}.obj"
    return {
        "hp": hp,
        "M": ob[:, 1:].reshape(-1, 4, 4),
        "kp": kp,
        "obj": name,
        "mesh": mesh if mesh.is_file() else None,
        "T": len(fs),
    }


def window(raw: Raw, seq: dict, clip: str, ti: dict | None):
    return C.segment_window(CLIP_RE, seq, clip, ti, FPS, TREE_HZ, "take")


def measure(raw: Raw, seq: dict, key: tuple, conv: str | None = None) -> dict:
    rot, shift = CONVS[conv or CONV]
    idx = list(range(key[0], key[1]))
    src = np.clip(np.asarray(idx) + shift, 0, seq["T"] - 1)
    hv = {}
    for s, o in (("lh", 0), ("rh", 64)):
        j = seq["hp"][src, o + 1 : o + 64].reshape(-1, 21, 3)
        j[seq["hp"][src, o] < 0.5] = np.nan
        hv[s] = j
    M = seq["M"][idx]
    R = M[:, :3, :3] if rot == "rowmajor" else np.transpose(M[:, :3, :3], (0, 2, 1))
    have = np.abs(M[:, 3, 3] - 1) < 1e-6
    six = {s: (j[:, TIP_JOINTS], j[:, 0]) for s, j in hv.items()}  # the six points from the joints (no surface)
    st = C.measure_objects(idx, hv, {seq["obj"]: (seq["mesh"], R, M[:, :3, 3], have)}, 1.0 / FPS, six=six)
    st["touch_m"] = TOUCH_M_JOINTS
    return st


def validate(raw: Raw, keys: list[str]) -> dict:
    """Two oracles: carried_distances (joints, >= VAL_V m/s) and the release's own object keypoints (obj_pose: 21
    points — each nearest to the posed mesh's bounding-box corners/centre under the convention's rotation)."""
    res = {c: {"carried": [], "keypoints": []} for c in CONVS}
    for k in keys:
        seq = load_seq(raw, k)
        if seq["mesh"] is None:
            continue
        lo, hi = CL.surface(str(seq["mesh"]))[3:5]
        box = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])] + [(lo + hi) / 2]
        )
        for c in CONVS:
            st = measure(raw, seq, (0, seq["T"]), conv=c)
            res[c]["carried"] += list(C.carried_distances(st, np.arange(seq["T"]) / FPS, v_min=VAL_V))
            o = st["objs"][seq["obj"]]
            pw = np.einsum("tij,kj->tki", o["R"], box) + o["t"][:, None]
            d = np.linalg.norm(seq["kp"][:, :, None] - pw[:, None], axis=-1).min(-1)[o["have"]]
            res[c]["keypoints"] += list(d.ravel())
    return {
        c: {
            "carried": C.summarize_distances(np.asarray(r["carried"])),
            "keypoints": C.summarize_distances(np.asarray(r["keypoints"])),
        }
        for c, r in res.items()
    }
