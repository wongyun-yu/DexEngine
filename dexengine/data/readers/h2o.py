"""H2O raw pose release -> MANO layer (`<clip>/{task_info.json, 0/trajectory_keypoints.npz}`).

Raw (``<raw>/dl``, the official archives read in place — nothing is extracted):
  subject<k>_pose_v1_1.tar.gz -> subject<k>/<scene>/<take>/cam4/
      hand_pose_mano/NNNNNN.txt  [left: valid, transl(3), pose(48), shape(10) | right: the same]  (cam4 frame)
      cam_pose/NNNNNN.txt        4x4 row-major camera -> world
      obj_pose_rt/NNNNNN.txt     [object class, 4x4 row-major object -> camera]
  object.zip -> object/<name>/<file>.obj  (metres)

Conventions (measured against the delivery, every frame of the verified clips):
  * MANO: smplx flat_hand_mean, no PCA, additive translation — these parameters reproduce the release's own
    21-joint hand_pose to 1e-4 mm, which also fixes the left-model reading (published MANO_LEFT.pkl).
  * world: X' = G · cam_pose_k · X_cam with G = Rx(-135 deg) (the delivery's z-up frame), then the grounding shift.
  * one object per take (class -> name below), on the side(s) chosen by ``--sides`` — there is no default:
      ``table``   the delivered per-take H2O side assignment (``configs/datasets/h2o/sides_delivered.json``) —
          reproduces the
                  delivered layer exactly;
      ``contact`` this reader's own rule (the delivery's rule is not recoverable): per hand, the frames
                  whose MANO surface (with pose blend shapes) comes within CONTACT_M of the object mesh; a hand gets
                  the object when its count >= CONTACT_REL x the larger hand's count (agrees with 152 / 184 takes,
                      thresholds chosen on those same labels).
  * 30 fps; clip = task = ``subject<k>_<scene>__take<n>_cam4``.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from dexengine.core.repo import CONFIGS_DIR
from dexengine.data.readers import mano_raw as C

DATASET = "h2o"
FPS = 30.0
G = R.from_euler("x", -135, degrees=True).as_matrix()
CLASS_NAMES = {1: "book", 2: "espresso", 3: "lotion", 4: "spray", 5: "milk", 6: "cocoa", 7: "chips", 8: "cappuccino"}
#: the object meshes (metres): `<raw>/dl/MESH_ZIP` member `mesh_member(<name>)` — read here and by the object asset
#: builder (dexengine/data/objects/build.py)
MESH_ZIP = "object.zip"
MESH_FILES = {"spray": "object/spray/lotion_spray.obj"}  # every other class: object/<name>/<name>.obj
KINDS = ("hand_pose_mano", "cam_pose", "obj_pose_rt")
SIDE_TABLE = CONFIGS_DIR / "datasets/h2o/sides_delivered.json"
CONTACT_M = 0.02
CONTACT_REL = 0.675
TAKE_RE = re.compile(r"(subject\d+)/([a-z]\d+)/(\d+)/(cam4)/([a-z_]+)/(\d+)\.txt$")


def dl_root(raw: str | Path) -> Path:
    d = Path(raw) / "dl"
    if not list(d.glob("subject*_pose_v1_1.tar.gz")):
        raise FileNotFoundError(f"H2O pose archives subject*_pose_v1_1.tar.gz missing under {d}")
    return d


def clip_name(sub: str, scene: str, take: str, cam: str) -> str:
    return f"{sub}_{scene}__take{take}_{cam}"


_ARCH: dict[str, dict] = {}


def read_archive(path: Path) -> dict[str, dict[str, dict[int, str]]]:
    """One streaming pass over a pose archive: {clip: {kind: {frame: text}}} (cached; ~80 MB of text each)."""
    if str(path) not in _ARCH:
        out: dict = {}
        with tarfile.open(path, "r:gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                g = TAKE_RE.search(m.name)
                if not g or g.group(5) not in KINDS:
                    continue
                c = clip_name(*g.group(1, 2, 3, 4))
                out.setdefault(c, {k: {} for k in KINDS})[g.group(5)][int(g.group(6))] = (
                    tf.extractfile(m).read().decode()
                )
        _ARCH[str(path)] = out
    return _ARCH[str(path)]


def enumerate_clips(raw: str | Path) -> dict[str, Path]:
    """{clip: its archive} for every cam4 take in the pose archives."""
    out = {}
    for arch in sorted(dl_root(raw).glob("subject*_pose_v1_1.tar.gz")):
        for c in read_archive(arch):
            out[c] = arch
    return out


_MESH: dict[str, np.ndarray] = {}


def mesh_member(name: str) -> str:
    return MESH_FILES.get(name, f"object/{name}/{name}.obj")


def object_vertices(raw: str | Path, name: str) -> np.ndarray:
    if name not in _MESH:
        z = zipfile.ZipFile(dl_root(raw) / MESH_ZIP)
        m = trimesh.load(io.BytesIO(z.read(mesh_member(name))), file_type="obj", force="mesh", process=False)
        _MESH[name] = np.asarray(m.vertices, dtype=np.float64)
    return _MESH[name]


def _table(texts: dict[int, str]) -> np.ndarray:
    fr = sorted(texts)
    if fr != list(range(len(fr))):
        raise ValueError(f"frames not contiguous from 0 ({fr[:3]} … {fr[-3:]})")
    return np.stack([np.array(texts[f].split(), dtype=np.float64) for f in fr])


def sides_from_table(clip: str, table: str | Path = SIDE_TABLE) -> dict[str, bool]:
    """The delivered side assignment of `clip` (dies for a take the table does not hold)."""
    sides = json.loads(Path(table).read_text())["sides"]
    if clip not in sides:
        raise KeyError(f"{clip}: not in the side table {table}")
    return {s: s in sides[clip] for s in C.SIDES}


def sides_from_contact(mano_dir: str | Path, hm: np.ndarray, Tc: np.ndarray, verts: np.ndarray) -> dict[str, bool]:
    """This reader's own rule (module doc): surface-contact frame counts, relative to the larger hand."""
    import smplx
    import torch
    from scipy.spatial import cKDTree

    tree = cKDTree(verts)
    Rinv = np.transpose(Tc[:, :3, :3], (0, 2, 1))
    count = {}
    for side, o in (("left", 0), ("right", 62)):
        m = smplx.MANO(str(mano_dir), is_rhand=(side == "right"), use_pca=False, flat_hand_mean=True)

        def f(x):
            return torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            v = (
                m(
                    global_orient=f(hm[:, o + 4 : o + 7]),
                    hand_pose=f(hm[:, o + 7 : o + 52]),
                    betas=f(hm[:, o + 52 : o + 62]),
                    transl=f(hm[:, o + 1 : o + 4]),
                )
                .vertices.numpy()
                .astype(np.float64)
            )
        loc = np.einsum("tij,tkj->tki", Rinv, v - Tc[:, None, :3, 3])  # camera -> object frame
        d = tree.query(loc.reshape(-1, 3))[0].reshape(len(hm), -1).min(1)
        count[side] = int((d < CONTACT_M).sum())
    top = max(count.values())
    if top == 0:
        raise ValueError("no hand comes near the object")
    return {s: count[s] >= CONTACT_REL * top for s in C.SIDES}


def load_clip(
    raw: str | Path, mano_dir: str | Path, clip: str, arch: Path, sides: str, side_table: str | Path = SIDE_TABLE
) -> C.ClipSource:
    """One take -> ClipSource (z-up world, before grounding); sides = "table" | "contact" (module doc)."""
    take = read_archive(arch).get(clip)
    if take is None:
        raise KeyError(f"{clip} not in {arch.name}")
    hm = _table(take["hand_pose_mano"])
    cp = _table(take["cam_pose"]).reshape(-1, 4, 4)
    orc = _table(take["obj_pose_rt"])
    T = len(hm)
    if not (len(cp) == len(orc) == T) or hm.shape[1] != 124:
        raise ValueError(f"{clip}: frame counts {len(hm)}/{len(cp)}/{len(orc)} or width {hm.shape[1]}")
    Rw = np.einsum("ij,tjk->tik", G, cp[:, :3, :3])
    tw = cp[:, :3, 3] @ G.T
    joints, tips = {}, {}
    for side, o in (("left", 0), ("right", 62)):
        hp = C.HandParams(full_pose=hm[:, o + 4 : o + 52], transl=hm[:, o + 1 : o + 4], betas=hm[0, o + 52 : o + 62])
        j, t = C.hand_fk(C.load_mano(mano_dir, side), hp)
        joints[side], tips[side] = C.rigid_apply(Rw, tw, j), C.rigid_apply(Rw, tw, t)
    cls = {int(round(c)) for c in orc[:, 0] if c > 0}
    if len(cls) != 1:
        raise ValueError(f"{clip}: object classes {cls}")
    name = CLASS_NAMES[cls.pop()]
    Tc = orc[:, 1:].reshape(T, 4, 4)
    W = np.tile(np.eye(4), (T, 1, 1))
    W[:, :3, :3], W[:, :3, 3] = Rw, tw
    Tw = W @ Tc
    verts = object_vertices(raw, name)
    if sides == "table":
        sides = sides_from_table(clip, side_table)
    elif sides == "contact":
        sides = sides_from_contact(mano_dir, hm, Tc, verts)
    else:
        raise ValueError(f"sides must be 'table' or 'contact', got {sides!r}")
    return C.ClipSource(
        task=clip,
        dataset=DATASET,
        source_dt=1.0 / FPS,
        joints=joints,
        tips=tips,
        objects={s: (Tw if sides[s] else None) for s in C.SIDES},
        meshes={s: (verts if sides[s] else None) for s in C.SIDES},
        object_names={s: (name if sides[s] else None) for s in C.SIDES},
    )


def convert(
    raw: str | Path, mano_dir: str | Path, clip: str, arch: Path, sides: str, side_table: str | Path = SIDE_TABLE
) -> tuple[dict, dict]:
    """One clip -> (npz arrays, task_info)."""
    return C.build_clip(load_clip(raw, mano_dir, clip, arch, sides, side_table))


def verify_kwargs(a) -> dict:
    """verify_readers hook: the side mode under test."""
    if not getattr(a, "sides", None):
        raise SystemExit("h2o: give --sides table|contact")
    return {"sides": a.sides}


def main(argv: list[str] | None = None) -> int:
    def extra(ap):
        ap.add_argument("--sides", required=True, choices=["table", "contact"], help="object side rule (module doc)")
        ap.add_argument("--side-table", default=str(SIDE_TABLE), help="side table for --sides table")

    return C.run_cli(
        DATASET,
        lambda a: enumerate_clips(a.raw),
        lambda a, c, h: convert(a.raw, a.mano_dir, c, h, a.sides, a.side_table),
        argv,
        extra,
    )


if __name__ == "__main__":
    sys.exit(main())
