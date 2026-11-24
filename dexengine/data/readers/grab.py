"""GRAB raw release -> MANO layer (`<clip>/{task_info.json, 0/trajectory_keypoints.npz}`).

Raw (``<raw>``, the official zips read in place):
  grab__s<N>.zip -> s<N>/<seq>.npz   rhand / lhand params {global_orient, hand_pose (PCA, n_comps), transl} and
      vtemp (``tools/subject_meshes/<gender>/s<N>_{r,l}hand.ply``); object params {global_orient, transl};
      contact['object'] (T, V) body-part label of the touching body part per object vertex; framerate 120
  tools__object_meshes__contact_meshes.zip -> contact_meshes/<obj>.ply  (metres)
  tools/subject_meshes (``--subject-meshes``): the per-subject MANO hand templates the vtemp entries name. They
      are part of the official GRAB download (tools) but not of this raw layer — required, no fallback: the mean
      template moves the wrist 1–2 mm and the tips ~1 cm.

Conventions (measured against the delivery, every frame of the verified clips):
  * MANO: hand_pose = PCA coefficients (n_comps), flat_hand_mean=True, v_template = the subject hand template,
    additive translation (GRAB's own loader); GRAB's world is z-up, so only the grounding shift is applied.
  * object pose: rotation = R(global_orient)^T (GRAB applies ``v @ R``), translation as given; grounding mesh =
    the contact mesh.
  * the object goes on every side whose hand touches it on >= 5 % of the frames (CONTACT_RATIO): a frame counts
    for the right hand when any object vertex carries a right-hand label (22, 41–55), for the left with 21, 26–40;
    both -> the same pose on both sides. Over all 1,335 sequences the touched fractions are >= 0.0504 for an
    assigned side and <= 0.0490 for an unassigned one, so the rule reproduces the delivery 1,335 / 1,335.
  * task = ``s<N>__<seq>`` (the zip's case), clip = task.lower().
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from dexengine.data.readers import mano_raw as C

DATASET = "grab"
#: GRAB contact_ids of the hand parts (grab/tools: L_Hand 21 · L_Index1 … L_Thumb3 26–40 · R_Hand 22 · R_* 41–55)
LEFT_HAND_LABELS = frozenset([21, *range(26, 41)])
RIGHT_HAND_LABELS = frozenset([22, *range(41, 56)])
#: a hand gets the object when it touches it on at least this fraction of the sequence's frames
CONTACT_RATIO = 0.05
#: the object meshes (metres): `<raw>/MESH_ZIP` member `mesh_member(<obj>)` — read here and by the object asset
#: builder (dexengine/data/objects/build.py)
MESH_ZIP = "tools__object_meshes__contact_meshes.zip"


def mesh_member(name: str) -> str:
    return f"contact_meshes/{name}.ply"


def enumerate_clips(raw: str | Path) -> dict[str, tuple[Path, str]]:
    """{clip: (zip, member)} for every sequence in the subject zips."""
    out = {}
    for zp in sorted(Path(raw).glob("grab__s*.zip")):
        for n in zipfile.ZipFile(zp).namelist():
            if n.endswith(".npz"):
                sbj, f = n.split("/")[-2:]
                task = f"{sbj}__{f[:-4]}"
                if task.lower() in out:
                    raise ValueError(f"two GRAB sequences fold to {task.lower()}")
                out[task.lower()] = (zp, n)
    if not out:
        raise FileNotFoundError(f"GRAB zips grab__s*.zip missing under {raw}")
    return out


_MESH: dict[str, np.ndarray] = {}


def object_vertices(raw: str | Path, name: str) -> np.ndarray:
    if name not in _MESH:
        z = zipfile.ZipFile(Path(raw) / MESH_ZIP)
        m = trimesh.load(io.BytesIO(z.read(mesh_member(name))), file_type="ply", force="mesh", process=False)
        _MESH[name] = np.asarray(m.vertices, dtype=np.float64)
    return _MESH[name]


def subject_template(subject_meshes: str | Path, vtemp: str) -> np.ndarray:
    """(778, 3) hand template named by a sequence's vtemp entry, under the tools/subject_meshes directory."""
    rel = str(vtemp).split("subject_meshes/", 1)
    if len(rel) != 2:
        raise ValueError(f"unexpected vtemp {vtemp!r}")
    p = Path(subject_meshes) / rel[1]
    if not p.is_file():
        raise FileNotFoundError(f"GRAB subject hand template {p} missing (tools/subject_meshes of the GRAB download)")
    v = np.asarray(trimesh.load(str(p), force="mesh", process=False).vertices, dtype=np.float64)
    if v.shape != (778, 3):
        raise ValueError(f"{p}: {v.shape} vertices, expected (778, 3)")
    return v


def read_sequence(handle: tuple[Path, str]) -> dict:
    zp, member = handle
    return dict(np.load(io.BytesIO(zipfile.ZipFile(zp).read(member)), allow_pickle=True))


def object_sides(d: dict) -> dict[str, bool]:
    """Which hands get the object: touched on >= CONTACT_RATIO of the frames (GRAB's own contact labels)."""
    lab = np.asarray(d["contact"].item()["object"])
    return {
        s: bool(np.isin(lab, list(L)).any(1).mean() >= CONTACT_RATIO)
        for s, L in (("right", RIGHT_HAND_LABELS), ("left", LEFT_HAND_LABELS))
    }


def load_clip(
    raw: str | Path,
    mano_dir: str | Path,
    clip: str,
    handle: tuple[Path, str],
    templates: dict[str, np.ndarray] | None = None,
    subject_meshes: str | Path | None = None,
) -> C.ClipSource:
    """One sequence -> ClipSource. The hand templates come from `subject_meshes` (or, for a test, `templates`:
    side -> (778, 3))."""
    d = read_sequence(handle)
    sbj, seq = handle[1].split("/")[-2:]
    task = f"{sbj}__{seq[:-4]}"
    T = int(d["n_frames"])
    ncomp = int(d["n_comps"])
    joints, tips = {}, {}
    for side, key in (("right", "rhand"), ("left", "lhand")):
        h = d[key].item()
        p = h["params"]
        m = C.load_mano(mano_dir, side)
        if templates is not None:
            vt = templates[side]
        elif subject_meshes is not None:
            vt = subject_template(subject_meshes, h["vtemp"])
        else:
            raise ValueError("GRAB needs --subject-meshes (tools/subject_meshes): no mean-template fallback")
        pose = np.concatenate(
            [
                np.asarray(p["global_orient"], float),
                C.pose_from_pca(m, np.asarray(p["hand_pose"], float)[:, :ncomp], flat_hand_mean=True),
            ],
            1,
        )
        hp = C.HandParams(full_pose=pose[:T], transl=np.asarray(p["transl"], float)[:T], v_template=vt)
        joints[side], tips[side] = C.hand_fk(m, hp)
    o = d["object"].item()["params"]
    T44 = np.tile(np.eye(4), (T, 1, 1))
    T44[:, :3, :3] = R.from_rotvec(np.asarray(o["global_orient"], float)[:T]).inv().as_matrix()
    T44[:, :3, 3] = np.asarray(o["transl"], float)[:T]
    name = str(d["obj_name"])
    sides = object_sides(d)
    if not any(sides.values()):
        raise ValueError(f"{clip}: no hand touches the object on >= {CONTACT_RATIO:.0%} of the frames")
    verts = object_vertices(raw, name)
    return C.ClipSource(
        task=task,
        dataset=DATASET,
        source_dt=1.0 / float(d["framerate"]),
        joints=joints,
        tips=tips,
        objects={s: (T44 if sides[s] else None) for s in C.SIDES},
        meshes={s: (verts if sides[s] else None) for s in C.SIDES},
        object_names={s: (name if sides[s] else None) for s in C.SIDES},
    )


def convert(raw: str | Path, mano_dir: str | Path, clip: str, handle: tuple[Path, str], **kw) -> tuple[dict, dict]:
    """One clip -> (npz arrays, task_info). kw: subject_meshes=<dir> or templates={side: (778, 3)}."""
    return C.build_clip(load_clip(raw, mano_dir, clip, handle, **kw))


def verify_kwargs(a) -> dict:
    """verify_mano_readers hook: the subject-mesh directory."""
    return {"subject_meshes": a.subject_meshes}


def main(argv: list[str] | None = None) -> int:
    def extra(ap):
        ap.add_argument(
            "--subject-meshes", required=False, help="GRAB tools/subject_meshes directory (required to convert)"
        )

    return C.run_cli(
        DATASET,
        lambda a: enumerate_clips(a.raw),
        lambda a, c, h: convert(a.raw, a.mano_dir, c, h, subject_meshes=a.subject_meshes),
        argv,
        extra,
    )


if __name__ == "__main__":
    sys.exit(main())
