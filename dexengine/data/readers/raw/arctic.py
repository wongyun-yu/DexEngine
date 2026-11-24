"""Scene object oracle — ARCTIC raw loader (readers/raw/common owns the thresholds, the reduction and the classes).

Raw (<arctic raw layer>/data/arctic_data/data): raw_seqs/<s>/<seq>.mano.npy ({left,right}: rot · pose (45) · trans (m)
· shape) and <seq>.object.npy ((T,7): articulation angle (rad) · rotation axis-angle · translation (mm)) ·
meta/object_vtemplates/<obj>/{bottom,top}.obj (mm; unpacked once, scaled to metres, into a cache folder).
One articulated object per sequence: the object is measured as ONE scene object (its tree obj_id) whose surface is the
bottom part at the global pose plus the top part turned by the articulation angle about the object z axis — the hand
distance is the nearer of the two; motion = the global pose. 30 fps; both hands active.
Clip `<s>-<seq>__<k>[_segNN][_tN]` = the sequence; a segment / cut = the tree's segment.start/end (50 Hz from the
sequence start) -> round(x * 30 / 50).
Conventions = HAND_CONV × OBJ_CONV, chosen by `validate` (readers/raw/common.carried_distances).
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import uuid
from pathlib import Path

import numpy as np

from dexengine.data.readers.raw import common as C

FPS, TREE_HZ = 30.0, 50.0
CLIP_RE = re.compile(
    r"(?P<task>(?P<sub>s\d+)-(?P<seq>(?P<obj>[a-z]+)_[a-z0-9_]+?))__(?P<k>\d+)"
    r"(?P<seg>_seg\d+)?(?P<cut>_t\d+)?"
)
#: hand: (flat_hand_mean, re-centre on joint 0) — the left model = mano_conventions (read by
#: readers/raw/common.mano_layer) · object: articulation axis sign (0 = ignore)
HAND_CONVS = {"A": (False, False), "B": (True, False), "C": (False, True)}
OBJ_CONVS = {"E": -1.0, "F": 1.0, "G": 0.0}
HAND_CONV, OBJ_CONV = "A", "E"  # validated: tables/arctic_mano_validate.json (hand A 0.2 mm · 100 % vs C 24.8 mm —
#  the carried oracle does not separate the left model (both readings 0.2 mm): that is `--hand-check`'s question;
#  the axis sign E/F/G is not separated by the carried oracle — hands hold the base while carrying; it can only move a
#  lid contact, never whether the one articulated object is in the scene)
_MANO: dict = {}
obj_key = str


class Raw(C.NoParts):
    def __init__(self, raw_root: Path):
        self.r = Path(raw_root) / "data" / "arctic_data" / "data"
        if not (self.r / "raw_seqs").is_dir():
            raise FileNotFoundError(f"ARCTIC raw_seqs missing under {self.r}")
        self.task = {
            f"{d.name}-{f.name[: -len('.mano.npy')]}": f
            for d in (self.r / "raw_seqs").iterdir()
            for f in d.glob("*.mano.npy")
        }
        self.mesh_dir = Path(tempfile.gettempdir()) / "scene_oracle_arctic_meshes"

    def part_mesh(self, obj: str, part: str) -> Path | None:
        out = self.mesh_dir / obj / f"{part}.obj"
        if not out.is_file():
            import trimesh

            src = self.r / "meta" / "object_vtemplates" / obj / f"{part}.obj"
            if not src.is_file():
                return None
            m = trimesh.load(str(src), force="mesh", process=False)
            m.apply_scale(0.001)
            out.parent.mkdir(parents=True, exist_ok=True)
            # a unique temp name per writer (pid · uuid); global-RNG names collided between workers, which moved each
            # other's temp files and raised FileNotFoundError. The .obj suffix lets trimesh pick the format.
            tmp = out.with_name(f".{out.name}.{os.getpid()}.{uuid.uuid4().hex}.obj")
            m.export(str(tmp))
            tmp.replace(out)
        return out


def seq_key(clip: str) -> str:
    return CLIP_RE.fullmatch(clip).group("task")


def load_seq(raw: Raw, key: str) -> dict:
    f = raw.task[key]
    mano = np.load(f, allow_pickle=True).item()
    ob = np.load(str(f).replace(".mano.npy", ".object.npy"), allow_pickle=True).astype(float)
    obj = CLIP_RE.fullmatch(key + "__0").group("obj")
    T = min(len(ob), len(mano["right"]["trans"]), len(mano["left"]["trans"]))
    return {
        "mano": mano,
        "ob": ob[:T],
        "obj": obj,
        "T": T,
        "bottom": raw.part_mesh(obj, "bottom"),
        "top": raw.part_mesh(obj, "top"),
    }


def window(raw: Raw, seq: dict, clip: str, ti: dict | None):
    return C.segment_window(CLIP_RE, seq, clip, ti, FPS, TREE_HZ, "sequence")


def hand_verts(seq: dict, idx, side: str, conv: str) -> np.ndarray:
    import torch

    flat, recentre = HAND_CONVS[conv]
    p = seq["mano"]["left" if side == "lh" else "right"]
    m = C.mano_layer(_MANO["dir"], side, "arctic", use_pca=False, flat_hand_mean=flat)
    tr = np.asarray(p["trans"][idx], float)
    with torch.no_grad():
        o = m(
            global_orient=torch.tensor(p["rot"][idx], dtype=torch.float32),
            hand_pose=torch.tensor(p["pose"][idx], dtype=torch.float32),
            betas=torch.tensor(
                np.repeat(np.asarray(p["shape"], float).reshape(1, 10), len(idx), 0), dtype=torch.float32
            ),
            transl=torch.tensor(tr * (0 if recentre else 1), dtype=torch.float32),
        )
    v = o.vertices.numpy().astype(float)
    if recentre:
        v = v - o.joints[:, :1].numpy() + tr[:, None]
    return v


def measure(raw: Raw, seq: dict, key: tuple, hand_conv: str | None = None, obj_conv: str | None = None) -> dict:
    from scipy.spatial.transform import Rotation as R

    idx = list(range(key[0], key[1]))
    hv = {s: hand_verts(seq, idx, s, hand_conv or HAND_CONV) for s in ("lh", "rh")}
    ob = seq["ob"][idx]
    Rg = R.from_rotvec(ob[:, 1:4]).as_matrix()
    t = ob[:, 4:7] / 1000.0
    sign = OBJ_CONVS[obj_conv or OBJ_CONV]
    Ra = R.from_rotvec(np.outer(ob[:, 0] * sign, [0, 0, 1])).as_matrix()
    have = np.ones(len(idx), bool)
    poses = {seq["obj"]: (seq["bottom"], Rg, t, have), "#top": (seq["top"], Rg @ Ra, t, have)}
    st = C.measure_objects(idx, hv, poses, 1.0 / FPS)
    top = st["objs"].pop("#top")
    o = st["objs"][seq["obj"]]
    if o["mesh"] and top["mesh"]:
        o["dist"] = {s: np.minimum(o["dist"][s], top["dist"][s]) for s in ("lh", "rh")}
    return st


def validate(raw: Raw, keys: list[str]) -> dict:
    """hand conventions with the object at OBJ_CONV, then object conventions with the best hand convention."""
    out = {}
    seqs = [load_seq(raw, k) for k in keys]
    for hc in HAND_CONVS:
        d = [C.carried_distances(measure(raw, s, (0, s["T"]), hand_conv=hc), np.arange(s["T"]) / FPS) for s in seqs]
        out[f"hand {hc}"] = C.summarize_distances(np.concatenate(d))
    best = min(HAND_CONVS, key=lambda c: out[f"hand {c}"].get("med_mm", 1e9))
    for oc in OBJ_CONVS:
        d = [
            C.carried_distances(measure(raw, s, (0, s["T"]), hand_conv=best, obj_conv=oc), np.arange(s["T"]) / FPS)
            for s in seqs
        ]
        out[f"obj {oc} (hand {best})"] = C.summarize_distances(np.concatenate(d))
    return out


def hand_check(n_seq: int, invert: bool = False) -> dict:
    """The left-model statement held against the delivered keypoints of whole-sequence clips (`<task>__0`) — the
    oracle's own MANO (HAND_CONV) under both left readings; readers/raw/common.left_convention_check decides."""
    from dexengine.core import paths as P

    _MANO.setdefault("dir", str(P.mano_model_dir()))
    raw = Raw(P.layer_path("raw", "arctic"))
    mano_root = P.layer_path("mano", "arctic", emb="bimanual")
    flat, recentre = HAND_CONVS[HAND_CONV]
    keys = [k for k in sorted(raw.task) if (mano_root / f"{k}__0" / "0" / "trajectory_keypoints.npz").is_file()]
    clips = []
    for key in keys[:: max(1, len(keys) // n_seq)][:n_seq]:
        seq = load_seq(raw, key)
        z = np.load(mano_root / f"{key}__0" / "0" / "trajectory_keypoints.npz")
        fr = np.round(z["timestamps"] * FPS).astype(int)
        sel = np.flatnonzero(fr < seq["T"])[::5]
        idx = fr[sel]
        pts = {}
        for side, inv in (("rh", False), ("lh", False), ("lh", True)):
            p = seq["mano"]["left" if side == "lh" else "right"]
            m = C.mano_layer(_MANO["dir"], side, "arctic", invert_left=inv, use_pca=False, flat_hand_mean=flat)
            pts[(side, inv)] = C.hand_points(
                m,
                p["rot"][idx],
                p["pose"][idx],
                np.repeat(np.asarray(p["shape"], float).reshape(1, 10), len(idx), 0),
                p["trans"][idx],
                recentre,
            )
        ref = C.delivered_points(z, sel)
        clips.append((ref["rh"], ref["lh"], pts[("rh", False)], pts[("lh", False)], pts[("lh", True)]))
    return C.left_convention_check("arctic", clips, invert=invert)


if __name__ == "__main__":
    # sink (entry_points-style); the command line (including --help) is parsed by readers/raw/common.hand_check_cli
    argparse.ArgumentParser(add_help=False, description=__doc__).parse_known_args()
    import sys

    sys.exit(C.hand_check_cli("arctic", hand_check))
