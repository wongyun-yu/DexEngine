"""Scene object oracle — HOT3D raw loader (readers/raw/common owns the thresholds, the reduction and the classes).

Raw (<hot3d raw layer>/dataset): <SEQ>/mano_hand_pose_trajectory.jsonl (per record: timestamp_ns · hand_poses
{"0" left | "1" right: wrist_xform {q_wxyz, t_xyz} · pose (15 PCA) · betas}) · <SEQ>/dynamic_objects.csv (object_uid,
t_ns, tx ty tz, qw qx qy qz, …) · assets/instance.json (uid -> instance_name, the trees' obj_id) ·
assets/<uid>.glb (metres).
Clip -> window: `<seq>__s<a>_e<b>[_segNN][_tN]` = hand-frame indices at 30 fps after de-duplicating the jsonl by the
ms-rounded timestamp (two records per frame); a segment / cut = the tree's segment.start/end (50 Hz from the clip
start; verified T_mano == (b-a)*5/3) -> a + round(x * 30 / 50). The window is measured on its own (sk = (w0, w1)) —
HOT3D sequences are long and clips short. An object pose is the nearest dynamic_objects.csv row within GT_TOL_NS of the
hand frame, else that object is absent at that frame.
No stage program and no part tree: both hands count as active (T3 'touch' therefore includes a hand the tree calls
idle — the clip row reports tree_idle beside it); no part siblings (C.NoParts).
MANO convention = CONV, chosen by `validate` (readers/raw/common.carried_distances — 12 sequences, 4,159 carried
frames: A median 0.3 mm · p90 0.6 mm · 97.8 % <= 10 mm; controls B 12.4 mm · C/D 13–17 mm).
"""

from __future__ import annotations

import collections
import csv
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from dexengine.data.readers.raw import common as C

FPS, TREE_HZ = 30.0, 50.0
GT_TOL_NS = 20e6
CLIP_RE = re.compile(r"(?P<seq>[pP]\d{4}_[0-9a-f]{8})__s(?P<a>\d+)_e(?P<b>\d+)(?P<seg>_seg\d+)?(?P<cut>_t\d+)?")
#: name -> (smplx MANO kwargs, re-centre on joint 0 before + wrist t) — the left model = mano_conventions (read by
#: readers/raw/common.mano_layer)
CONVS = {
    "A": ({"use_pca": True, "num_pca_comps": 15, "flat_hand_mean": False}, False),
    "B": ({"use_pca": True, "num_pca_comps": 15, "flat_hand_mean": False}, True),
    "C": ({"use_pca": True, "num_pca_comps": 15, "flat_hand_mean": True}, False),
}
CONV = "A"
_MANO: dict = {}
obj_key = str


class Raw(C.NoParts):
    def __init__(self, raw_root: Path):
        self.r = Path(raw_root) / "dataset"
        if not (self.r / "assets" / "instance.json").is_file():
            raise FileNotFoundError(f"HOT3D raw missing: {self.r}/assets/instance.json")
        self.inst = json.loads((self.r / "assets" / "instance.json").read_text())


def seq_key(clip: str) -> str:
    s = CLIP_RE.fullmatch(clip).group("seq")
    return "P" + s[1:]


def load_seq(raw: Raw, key: str) -> dict:
    sd = raw.r / key
    ts, hp, seen = [], [], set()
    with open(sd / "mano_hand_pose_trajectory.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            k = int(round(r["timestamp_ns"] / 1e6))
            if k in seen:
                continue
            seen.add(k)
            ts.append(int(r["timestamp_ns"]))
            hp.append(r["hand_poses"])
    o = np.argsort(ts)
    objs = collections.defaultdict(list)
    with open(sd / "dynamic_objects.csv") as fh:
        for r in csv.reader(fh):
            if r[0] != "object_uid":
                objs[r[0]].append([float(x) for x in r[1:]])
    out = {}
    for uid, rows in objs.items():
        a = np.asarray(rows)
        out[uid] = {"name": raw.inst[uid]["instance_name"], "a": a[np.argsort(a[:, 0])]}
    return {"ts": np.asarray(ts)[o], "hp": [hp[i] for i in o], "objs": out}


def window(raw: Raw, seq: dict, clip: str, ti: dict | None):
    m = CLIP_RE.fullmatch(clip)
    a, b = int(m.group("a")), int(m.group("b"))
    w0, w1, how = a, b, "clip"
    if m.group("seg") or m.group("cut"):
        seg = (ti or {}).get("segment") or {}
        if seg.get("start") is None or seg.get("end") is None:
            raise ValueError("segment/cut clip without tree segment.start/end")
        w0, w1, how = a + int(round(seg["start"] * FPS / TREE_HZ)), a + int(round(seg["end"] * FPS / TREE_HZ)), "cut"
    if not 0 <= w0 < w1 <= len(seq["ts"]):
        raise ValueError(f"window {w0}-{w1} outside the sequence's {len(seq['ts'])} hand frames")
    return (w0, w1), w0, w1, {"lh": True, "rh": True}, set(), how


def hand_verts(seq: dict, idx, side: str, conv: str) -> np.ndarray:
    import torch
    from scipy.spatial.transform import Rotation as R

    kw, recentre = CONVS[conv]
    key = "0" if side == "lh" else "1"
    V = np.full((len(idx), 778, 3), np.nan)
    ok = [i for i, k in enumerate(idx) if seq["hp"][k].get(key)]
    if not ok:
        return V
    ds = [seq["hp"][idx[i]][key] for i in ok]
    rv = np.stack([R.from_quat(np.asarray(d["wrist_xform"]["q_wxyz"])[[1, 2, 3, 0]]).as_rotvec() for d in ds])
    t = np.stack([np.asarray(d["wrist_xform"]["t_xyz"], float) for d in ds])
    pca = np.stack([np.asarray(d["pose"], float)[:15] for d in ds])
    be = np.stack([np.asarray(d["betas"], float)[:10] for d in ds])
    m = C.mano_layer(_MANO["dir"], side, "hot3d", **kw)
    with torch.no_grad():
        o = m(
            global_orient=torch.tensor(rv, dtype=torch.float32),
            hand_pose=torch.tensor(pca, dtype=torch.float32),
            betas=torch.tensor(be, dtype=torch.float32),
            transl=torch.tensor(t * (0 if recentre else 1), dtype=torch.float32),
        )
    v = o.vertices.numpy().astype(float)
    if recentre:
        v = v - o.joints[:, :1].numpy() + t[:, None]
    V[ok] = v
    return V


@lru_cache(maxsize=256)
def _mesh(root: str, uid: str):
    p = Path(root) / "assets" / f"{uid}.glb"
    return p if p.is_file() else None


def measure(raw: Raw, seq: dict, key: tuple, conv: str | None = None) -> dict:
    from scipy.spatial.transform import Rotation as R

    conv = conv or CONV
    ts = seq["ts"]
    idx = list(range(max(0, key[0]), min(len(ts), key[1])))
    hv = {s: hand_verts(seq, idx, s, conv) for s in ("lh", "rh")}
    poses = {}
    for uid, o in seq["objs"].items():
        arr = o["a"]
        j = np.clip(np.searchsorted(arr[:, 0], ts[idx]), 0, len(arr) - 1)
        j0 = np.clip(j - 1, 0, len(arr) - 1)
        jj = np.where(np.abs(arr[j, 0] - ts[idx]) <= np.abs(arr[j0, 0] - ts[idx]), j, j0)
        have = np.abs(arr[jj, 0] - ts[idx]) <= GT_TOL_NS
        poses[o["name"]] = (
            _mesh(str(raw.r), uid),
            R.from_quat(arr[jj][:, [5, 6, 7, 4]]).as_matrix(),
            arr[jj][:, 1:4],
            have,
        )
    return C.measure_objects(idx, hv, poses, 1.0 / FPS)


def validate(raw: Raw, keys: list[str]) -> dict:
    """carried_distances per convention over `keys` (whole sequences)."""
    res = {c: [] for c in CONVS}
    for k in keys:
        seq = load_seq(raw, k)
        t_s = (seq["ts"] - seq["ts"][0]) / 1e9
        for c in CONVS:
            st = measure(raw, seq, (0, len(seq["ts"])), conv=c)
            res[c] += list(C.carried_distances(st, t_s[st["frames"]]))
    return {c: C.summarize_distances(np.asarray(v)) for c, v in res.items()}
