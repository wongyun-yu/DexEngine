"""Scene object oracle — TACO raw loader (readers/raw/common owns the thresholds, the reduction and the classes).

Raw: the lift's raw layout (side_objects.taco_stage_raw — the one place TACO object poses are read from):
hands = Hand_Poses/<(verb, tool, target)>/<seq>/{left,right}_hand.pkl (per frame hand_pose 48 axis-angle · hand_trans
3, shifted into the source frame) + {left,right}_hand_shape.pkl (hand_shape 10); objects = Object_Poses/<triplet>/<seq>/
{tool,target}_<id>.npy ((T,4,4), metres, the TACO source's tracks). Hands and objects share one frame. Missing dies.
Meshes = data/taco/meshes/<id>/visual.obj (metres). 30 fps; both hands active (TACO has no per-hand activity).
Clip -> window: `<verb>_<tool>_<target>__<seq>[_segNN][_tN]`; the triplet directory is found by the converter's own
name rule (taco_convert.taco_task_name: drop parentheses, ', ' -> '_') with a space inside a word -> '-'; the
clip is the whole sequence, a segment /
cut is the tree's segment.start/end (50 Hz from the sequence start) -> round(x * 30 / 50).
MANO convention = CONV, chosen by `validate` (readers/raw/common.carried_distances — see the CONVS table).
"""

from __future__ import annotations

import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from dexengine.data import side_objects
from dexengine.data.readers.raw import common as C

FPS, TREE_HZ = 30.0, 50.0
CLIP_RE = re.compile(r"(?P<task>.+?__\d{8}_\d{3})(?P<seg>_seg\d+)?(?P<cut>_t\d+)?")
#: name -> (smplx MANO kwargs, re-centre on joint 0 before + hand_trans) — the left model = mano_conventions (read
#: by readers/raw/common.mano_layer)
CONVS = {
    "A": ({"use_pca": False, "flat_hand_mean": False}, False),
    "B": ({"use_pca": False, "flat_hand_mean": True}, False),
    "C": ({"use_pca": False, "flat_hand_mean": False}, True),
    "F": ({"use_pca": False, "flat_hand_mean": True}, True),
}
CONV = "C"  # validated: tables/taco_mano_validate.json
_MANO: dict = {}
obj_key = str


class Raw(C.NoParts):
    def __init__(self, raw_root: Path):
        stage = side_objects.taco_stage_raw()  # not raw_root: the stage-1 lift layout (module doc)
        self.hp = stage / "Hand_Poses"
        self.op = stage / "Object_Poses"
        if not self.hp.is_dir() or not self.op.is_dir():
            raise FileNotFoundError(f"TACO raw missing: {self.hp} · {self.op}")
        self.mesh = Path(raw_root).parent / "meshes"
        self.task = {}
        for d in self.hp.iterdir():
            if not d.is_dir():  # a triplet is a folder; the layout may hold marker files (e.g. a restore stamp)
                continue
            for s in d.iterdir():  # tree names write a space inside a word as '-' (glue gun)
                k = d.name.replace("(", "").replace(")", "").replace(", ", "_").replace(" ", "-") + "__" + s.name
                if k in self.task:
                    raise ValueError(f"TACO name collision {k}: {self.task[k]} · {(d.name, s.name)}")
                self.task[k] = (d.name, s.name)


def seq_key(clip: str) -> str:
    return CLIP_RE.fullmatch(clip).group("task")


def load_seq(raw: Raw, key: str) -> dict:
    trip, sid = raw.task[key]
    hands = {}
    for side, name in (("lh", "left"), ("rh", "right")):
        with open(raw.hp / trip / sid / f"{name}_hand.pkl", "rb") as fh:
            d = pickle.load(fh)
        with open(raw.hp / trip / sid / f"{name}_hand_shape.pkl", "rb") as fh:
            sh = pickle.load(fh)
        ks = sorted(d)
        hands[side] = (
            np.stack([np.asarray(d[k]["hand_pose"], float) for k in ks]),
            np.stack([np.asarray(d[k]["hand_trans"], float) for k in ks]),
            np.asarray(sh["hand_shape"], float).reshape(10),
        )
    objs = {}
    for f in sorted((raw.op / trip / sid).glob("*.npy")):
        m = re.fullmatch(r"(tool|target)_(\w+)\.npy", f.name)
        if m:
            objs[m.group(2)] = np.load(f).astype(float)
    T = min(len(hands["lh"][0]), len(hands["rh"][0]), *(len(v) for v in objs.values()))
    return {"hands": hands, "objs": objs, "T": T}


def window(raw: Raw, seq: dict, clip: str, ti: dict | None):
    return C.segment_window(CLIP_RE, seq, clip, ti, FPS, TREE_HZ, "sequence")


def hand_verts(seq: dict, idx, side: str, conv: str) -> np.ndarray:
    import torch

    kw, recentre = CONVS[conv]
    pose, trans, beta = seq["hands"][side]
    m = C.mano_layer(_MANO["dir"], side, "taco", **kw)
    with torch.no_grad():
        o = m(
            global_orient=torch.tensor(pose[idx, :3], dtype=torch.float32),
            hand_pose=torch.tensor(pose[idx, 3:48], dtype=torch.float32),
            betas=torch.tensor(np.repeat(beta[None], len(idx), 0), dtype=torch.float32),
            transl=torch.tensor(trans[idx] * (0 if recentre else 1), dtype=torch.float32),
        )
    v = o.vertices.numpy().astype(float)
    if recentre:
        v = v - o.joints[:, :1].numpy() + trans[idx][:, None]
    return v


@lru_cache(maxsize=64)
def _mesh(raw_mesh: str, oid: str):
    p = Path(raw_mesh) / oid / "visual.obj"
    return p if p.is_file() else None


def measure(raw: Raw, seq: dict, key: tuple, conv: str | None = None) -> dict:
    idx = list(range(key[0], key[1]))
    conv = conv or CONV
    hv = {s: hand_verts(seq, idx, s, conv) for s in ("lh", "rh")}
    poses = {
        oid: (_mesh(str(raw.mesh), oid), T[idx, :3, :3], T[idx, :3, 3], np.isfinite(T[idx, 0, 0]))
        for oid, T in seq["objs"].items()
    }
    return C.measure_objects(idx, hv, poses, 1.0 / FPS)


def validate(raw: Raw, keys: list[str]) -> dict:
    """carried_distances per convention over `keys` (whole sequences)."""
    res = {c: [] for c in CONVS}
    for k in keys:
        seq = load_seq(raw, k)
        for c in CONVS:
            st = measure(raw, seq, (0, seq["T"]), conv=c)
            res[c] += list(C.carried_distances(st, np.arange(seq["T"]) / FPS))
    return {c: C.summarize_distances(np.asarray(v)) for c, v in res.items()}
