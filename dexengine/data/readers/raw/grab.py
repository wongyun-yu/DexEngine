"""Scene object oracle — GRAB raw loader (readers/raw/common owns the thresholds, the reduction and the classes).

Raw (<grab raw layer>, the official zips read in place): grab__s<N>.zip -> s<N>/<seq>.npz (rhand · lhand params
global_orient · hand_pose (PCA n_comps) · fullpose (45) · transl; object · table params transl · global_orient;
framerate 120) · tools__object_meshes__contact_meshes.zip -> contact_meshes/<obj>.ply (metres; unpacked once to a
cache folder — mesh_cache: extracted once per zip, renamed into place whole, so parallel workers only read a
complete copy). The table is not a scene object here — it is the desk (desk_support), so it is left out.
The subject hand templates (tools/subject_meshes vtemp) are not in the raw layer: the MANO mean template is used and
the validation below measures what that costs.
Clip `s<N>__<seq>[_segNN][_tN]` (lower case; the raw name is matched case-insensitively) = the sequence; a segment /
cut = the tree's segment.start/end (50 Hz from the sequence start) -> round(x * 120 / 50). Both hands active.
MANO convention = CONV, chosen by `validate` (readers/raw/common.carried_distances).
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from dexengine.data.readers.raw import common as C

HZ, TREE_HZ, STEP = 120.0, 50.0, 4
CLIP_RE = re.compile(r"(?P<task>s\d+__.+?)(?P<seg>_seg\d+)?(?P<cut>_t\d+)?")
#: name -> (pose source "pca" | "full", flat_hand_mean, re-centre on joint 0 before + transl)
CONVS = {"A": ("pca", True, False), "B": ("full", True, False), "C": ("full", False, False), "D": ("pca", True, True)}
CONV = "A"  # validated: tables/grab_mano_validate.json (0.3 mm · 91.7 % <= 10 mm vs recentred 41.6 mm)
_MANO: dict = {}
obj_key = str


class Raw(C.NoParts):
    def __init__(self, raw_root: Path):
        self.r = Path(raw_root)
        self.task = {}
        for zp in sorted(self.r.glob("grab__s*.zip")):
            for n in zipfile.ZipFile(zp).namelist():
                if n.endswith(".npz"):
                    s, f = n.split("/")[-2:]
                    self.task[f"{s}__{f[:-4]}".lower()] = (str(zp), n)
        if not self.task:
            raise FileNotFoundError(f"GRAB zips missing under {self.r}")
        self.mesh_dir = mesh_cache(self.r / "tools__object_meshes__contact_meshes.zip")


def mesh_cache(zp: Path, root: Path | None = None) -> Path:
    """The contact-mesh zip extracted under the temp root (`root` = a test's) — one complete copy per zip (size · mtime
    in the name). Each caller extracts into its own fresh folder and renames it into place: a reader sees no copy or a
    whole one. Workers extracting into one shared folder raced (os.mkdir FileExistsError — the audit died with 0 rows),
    and a worker that found the folder mid-extract read a missing mesh as 'no mesh' (a different verdict). A rename
    that loses to another caller's finished copy keeps that copy and drops its own."""
    st, root = zp.stat(), Path(root or tempfile.gettempdir())
    final = root / f"scene_oracle_grab_meshes_{st.st_size}_{int(st.st_mtime)}"
    if final.is_dir():
        return final
    root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=root))
    try:
        zipfile.ZipFile(zp).extractall(tmp)
        try:
            os.rename(tmp, final)
        except OSError:
            if not final.is_dir():
                raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return final


def seq_key(clip: str) -> str:
    return CLIP_RE.fullmatch(clip).group("task").lower()


def load_seq(raw: Raw, key: str) -> dict:
    from scipy.spatial.transform import Rotation as R

    zp, n = raw.task[key]
    d = np.load(io.BytesIO(zipfile.ZipFile(zp).read(n)), allow_pickle=True)
    hands = {s: d[k].item()["params"] for s, k in (("lh", "lhand"), ("rh", "rhand"))}
    o = d["object"].item()
    name = str(d["obj_name"])
    mesh = raw.mesh_dir / "contact_meshes" / f"{name}.ply"
    T = int(d["n_frames"])
    Ro = R.from_rotvec(o["params"]["global_orient"]).as_matrix()
    return {
        "hands": hands,
        "ncomps": int(d["n_comps"]),
        "obj": name,
        "R": Ro,
        "t": np.asarray(o["params"]["transl"], float),
        "mesh": mesh if mesh.is_file() else None,
        "T": T,
    }


def window(raw: Raw, seq: dict, clip: str, ti: dict | None):
    return C.segment_window(CLIP_RE, seq, clip, ti, HZ, TREE_HZ, "sequence")


def hand_verts(seq: dict, idx, side: str, conv: str) -> np.ndarray:
    import torch

    src, flat, recentre = CONVS[conv]
    p = seq["hands"][side]
    kw = {"use_pca": True, "num_pca_comps": seq["ncomps"]} if src == "pca" else {"use_pca": False}
    m = C.mano_layer(_MANO["dir"], side, "grab", flat_hand_mean=flat, **kw)
    hp = p["hand_pose"][idx] if src == "pca" else p["fullpose"][idx]
    tr = np.asarray(p["transl"][idx], float)
    with torch.no_grad():
        o = m(
            global_orient=torch.tensor(p["global_orient"][idx], dtype=torch.float32),
            hand_pose=torch.tensor(hp, dtype=torch.float32),
            betas=torch.zeros(len(idx), 10),
            transl=torch.tensor(tr * (0 if recentre else 1), dtype=torch.float32),
        )
    v = o.vertices.numpy().astype(float)
    if recentre:
        v = v - o.joints[:, :1].numpy() + tr[:, None]
    return v


def measure(raw: Raw, seq: dict, key: tuple, conv: str | None = None) -> dict:
    idx = list(range(key[0], key[1], STEP))
    conv = conv or CONV
    hv = {s: hand_verts(seq, idx, s, conv) for s in ("lh", "rh")}
    poses = {seq["obj"]: (seq["mesh"], seq["R"][idx], seq["t"][idx], np.ones(len(idx), bool))}
    return C.measure_objects(idx, hv, poses, STEP / HZ)


def validate(raw: Raw, keys: list[str]) -> dict:
    res = {c: [] for c in CONVS}
    for k in keys:
        seq = load_seq(raw, k)
        for c in CONVS:
            st = measure(raw, seq, (0, seq["T"]), conv=c)
            res[c] += list(C.carried_distances(st, st["frames"] / HZ))
    return {c: C.summarize_distances(np.asarray(v)) for c, v in res.items()}
