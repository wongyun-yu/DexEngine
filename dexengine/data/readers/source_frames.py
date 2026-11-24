"""Source-world → +z-up frame rules of the readers whose raw world is not the +z-up world later layers assume — the one
owner. Dependency-free on purpose (numpy · scipy · csv only, never the retargeting runtime): the readers import it
(``dexengine.data.readers.hot3d`` / ``hrdexdb``) and so do the scripts that parse the same raw files themselves
(``dexengine/data/support/hot3d.py`` · ``dexengine/data/support/hrdexdb.py``) — a second
copy of a rotation is how a raw reader and its oracle drift apart.

``FRAME_VERSION`` is the generation of these rules: the readers write it into task_info ``source_frame`` and the
two-hand shared contact layer counts it in a base tree's generation (``dexengine/data/contact/fingerprint.py``).
Change a rule here → bump it.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

#: Generation of the frame rules below (date of the rule set) — task_info ``source_frame.version``.
FRAME_VERSION = "2026-10-07"

#: Fields of a task_info ``source_frame`` record that are bookkeeping, not the transform it states: the rule pointer and
#: the generation (a bump re-labels the rule set; it does not change a transform already applied to a clip).
FRAME_META = ("rule", "version")


def frame_key(record) -> tuple:
    """What a task_info ``source_frame`` record states about the transform applied to one clip — the key two layers
    compare. The MANO layer is rebuilt when a rule here changes; a robot retarget copies the MANO record at
    generate-xml, so it is current only while the two keys agree (dexengine/data/stages.py stage 1 cell).

    Rule: the record minus ``FRAME_META``. No record (the reader applied no rule) and a record whose transform is the
    identity rotation (``R_to_z_up`` = I — ``to_z_up`` leaves such a recording byte-identical) are one key. A record
    whose fields this function cannot read dies: a reader that writes a new kind of record adds it here."""
    if record is None:
        return ("identity",)
    if not isinstance(record, dict):
        raise ValueError(f"source_frame record {record!r} is not a mapping")
    body = {k: v for k, v in record.items() if k not in FRAME_META}
    if "R_to_z_up" in body:
        rot = np.asarray(body["R_to_z_up"], dtype=np.float64)
        if rot.shape != (3, 3):
            raise ValueError(f"source_frame R_to_z_up has shape {rot.shape}, not 3x3")
        return ("identity",) if np.array_equal(rot, np.eye(3)) else ("R_to_z_up", tuple(rot.ravel().tolist()))
    if "applied" in body:
        return ("applied",) + tuple(sorted((k, str(v)) for k, v in body.items()))
    raise ValueError(
        f"source_frame record with fields {sorted(record)} — frame_key reads R_to_z_up · applied "
        "(a new reader's record adds its transform here)"
    )


# ── HOT3D ────────────────────────────────────────────────────────────────────────────────────────
#: World up axis per capture device. Source: official HOT3D toolkit ``hot3d/Hot3DVisualizer.py:115-118``
#: (github.com/facebookresearch/hot3d @146b34a): ``Headset.Aria`` →
#: ``rr.ViewCoordinates.RIGHT_HAND_Z_UP``, any other device (Quest3) → ``RIGHT_HAND_Y_UP``. The viewer only declares the
#: axis and never rotates the data, so the stored world of a Quest3 recording is y-up. Keys are the exact
#: ``metadata.json["headset"]`` strings; a value not listed here dies in ``world_up`` (no default axis).
HEADSET_UP = {"Aria": "+z", "Quest3": "+y"}
#: Proper rotation taking each up axis to +z (right-handed frame kept): +y → +z is Rx(+90°) — the matrix
#: ``dexengine/data/support/oakink2_rule.py`` RX uses for the same y-up → z-up turn.
UP_TO_Z = {"+z": np.eye(3), "+y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])}


def world_up(headset) -> tuple[str, np.ndarray]:
    """(up axis, 3×3 rotation to the +z-up world) of a recording's ``metadata.json["headset"]`` — unknown values die."""
    if headset not in HEADSET_UP:
        raise ValueError(
            f"hot3d headset={headset!r} has no declared world up axis (HEADSET_UP knows {sorted(HEADSET_UP)})"
        )
    up = HEADSET_UP[headset]
    return up, UP_TO_Z[up]


#: Per-recording levelling of the device up axis: each listed recording stands its up axis on its desk-plane normal.
#: The stated device axis of a Quest3 recording is off gravity by ~1.5° (desk-plane tilt p50 1.53° over 141 recordings
#: vs Aria 0.51°; the object bottom-face oracle, measured apart from positions, tilts the same way on the desk 122/140
#: and on other surfaces 84/128 and drops 1.30 → 0.78° once levelled; Aria shows no such evidence and stays untouched).
#: The levelled up axis of one recording = the normal of its desk plane, fitted by
#: ``dexengine/data/support/hot3d_level.py`` (the table's only writer, rerun on the raw release reproduces it) and
#: stated once in ``LEVEL_TABLE``. A recording not in the table (no fit — fewer than 3 objects / 6 rests / 5 cm spread
#: on its desk) keeps the device axis.
LEVEL_TABLE = Path(__file__).resolve().parents[3] / "configs/datasets/hot3d_up_level.tsv"
#: Devices whose recordings are levelled — the evidence above; a table row of another device dies.
LEVEL_DEVICES = ("Quest3",)
#: Generation of the levelling rule, written into the ``source_frame.level`` record of a levelled clip only (the
#: frame rules of every other recording, and so ``FRAME_VERSION``, are unchanged — their clips stay byte-identical).
LEVEL_VERSION = "2026-11-22"
_LEVEL: dict | None = None


def level_rows(path: Path | None = None) -> dict[str, dict]:
    """{recording: {headset, a, b, …}} of the levelling table (read once). a, b = desk-plane slope dz/dx, dz/dy in the
    device +z-up world. A duplicate recording or a device outside LEVEL_DEVICES dies."""
    global _LEVEL
    if path is None and _LEVEL is not None:
        return _LEVEL
    rows: dict[str, dict] = {}
    with open(path or LEVEL_TABLE) as f:
        for r in csv.DictReader((ln for ln in f if not ln.startswith("#")), delimiter="\t"):
            if r["recording"] in rows:
                raise ValueError(f"{path or LEVEL_TABLE}: recording {r['recording']} listed twice")
            if r["headset"] not in LEVEL_DEVICES:
                raise ValueError(
                    f"{path or LEVEL_TABLE}: {r['recording']} headset={r['headset']!r} is not a levelled device "
                    f"({LEVEL_DEVICES})"
                )
            rows[r["recording"]] = {"headset": r["headset"], "a": float(r["a"]), "b": float(r["b"])}
    if path is None:
        _LEVEL = rows
    return rows


def level_rotation(a: float, b: float) -> np.ndarray:
    """Smallest rotation taking the desk normal of plane z = a·x + b·y + c, n ∝ (−a, −b, 1), onto +z (Rodrigues)."""
    n = np.array([-a, -b, 1.0]) / np.sqrt(a * a + b * b + 1.0)
    axis = np.cross(n, [0.0, 0.0, 1.0])
    s, c = float(np.linalg.norm(axis)), float(n[2])
    if s == 0.0:
        return np.eye(3)
    k = axis / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def plane_tilt_deg(a: float, b: float) -> float:
    """Tilt (degrees) of the plane z = a·x + b·y + c from horizontal. arctan2 · hypot are in the fixed transcendental
    set of the bake closure (arctan is not: its value differs across libm builds)."""
    return float(np.degrees(np.arctan2(np.hypot(a, b), 1.0)))


def recording_up(headset, recording: str) -> tuple[str, np.ndarray, dict | None]:
    """(up axis, rotation to the +z-up world, level record | None) of one recording: the device axis (``world_up``),
    then — when the recording is in ``LEVEL_TABLE`` — its desk-plane levelling. The level record goes into task_info
    ``source_frame.level``; None for every other recording (their rotation and record are the device rule's)."""
    up, rot = world_up(headset)
    row = level_rows().get(recording)
    if row is None:
        return up, rot, None
    if row["headset"] != headset:
        raise ValueError(f"{LEVEL_TABLE}: {recording} is listed as {row['headset']}, metadata says {headset!r}")
    lv = level_rotation(row["a"], row["b"])
    tilt = plane_tilt_deg(row["a"], row["b"])
    rec = {
        "table": "configs/datasets/hot3d_up_level.tsv",
        "version": LEVEL_VERSION,
        "a": row["a"],
        "b": row["b"],
        "tilt_deg": round(tilt, 4),
    }
    return up, lv @ rot, rec


def object_samples(seq_dir: Path) -> dict[str, np.ndarray]:
    """``<seq>/dynamic_objects.csv`` → {uid: (N,8) [t_ns x y z qw qx qy qz]} — raw GT samples, time-sorted."""
    raw = defaultdict(list)
    with open(Path(seq_dir) / "dynamic_objects.csv") as f:
        for r in csv.DictReader(f):
            raw[r["object_uid"]].append(
                [
                    int(r["timestamp[ns]"]),
                    float(r["t_wo_x[m]"]),
                    float(r["t_wo_y[m]"]),
                    float(r["t_wo_z[m]"]),
                    float(r["q_wo_w"]),
                    float(r["q_wo_x"]),
                    float(r["q_wo_y"]),
                    float(r["q_wo_z"]),
                ]
            )
    return {uid: np.asarray(sorted(rows), float) for uid, rows in raw.items()}


def rotate_samples(samples: dict[str, np.ndarray], rot: np.ndarray) -> dict[str, np.ndarray]:
    """GT samples ``{uid: (N,8) [t x y z qw qx qy qz]}`` in the rotated world: p' = R·p, q' = q_R ⊗ q.
    Left-multiplying every sample by the same unit quaternion keeps sample-to-sample dot products, so a gap test on
    bounding-sample rotation (hot3d ``_fill_gt_gaps``) and its hemisphere rule see exactly what they saw before."""
    q_r = Rot.from_matrix(rot)
    out = {}
    for uid, a in samples.items():
        b = a.copy()
        b[:, 1:4] = a[:, 1:4] @ rot.T
        b[:, 4:8] = (q_r * Rot.from_quat(a[:, [5, 6, 7, 4]])).as_quat()[:, [3, 0, 1, 2]]
        out[uid] = b
    return out


def to_z_up(samples: dict[str, np.ndarray], rot: np.ndarray) -> dict[str, np.ndarray]:
    """``rotate_samples``, except that the identity passes through untouched — a +z recording stays byte-identical."""
    return samples if np.array_equal(rot, np.eye(3)) else rotate_samples(samples, rot)


# ── HRDexDB ──────────────────────────────────────────────────────────────────────────────────────
#: Official HRDexDB code (github.com/snuvclab/HRDexDB @b8590393ed): ``hrdexdb/common.py:328-329``
#: ``robot_from_world = np.linalg.inv(load_c2r(episode_root))`` on the object poses, ``viser_viewer.py:308-309`` the
#: same inverse, ``:336`` the human MANO mesh drawn with ``root_transform=robot_from_world``, ``:332``
#: ``set_up_direction("+z")``. So ``C2R.npy`` is world_from_robot: hand joints and object poses (stored in the
#: camera-rig world) go to the robot base frame, +z up, through its **inverse** (``robot_from_world``).
C2R_FILE = "C2R.npy"
#: An episode without ``C2R.npy`` (106 of 441 human episodes): the official loader (``common.py:281-285``) returns
#: identity, but those episodes sit in the same camera-rig world as the ones that have the file (all 10 local ones:
#: frame-0 object t ≈ (-0.04, 0.28, 0.63) m like the 25 with C2R, lift direction rig-z -0.90…-0.99), so identity would
#: leave them upside down. No default either way: the reader dies unless its caller passes
#: ``allow_shared_c2r``; then ``release_c2r`` (the release's one matrix, checked equal in every file) is used.
#: The canonical reprocessing passes it — the command is in the reader's
#: docstring (``hrdexdb.py``); the per-episode marker says which C2R each episode used.
SHARED_C2R_ATOL = 1e-9


def load_c2r(path: Path) -> np.ndarray:
    mat = np.asarray(np.load(path), dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{path}: expected a 4x4 transform, got {mat.shape}")
    return mat


def release_c2r(human_root: Path) -> np.ndarray:
    """The release-wide C2R — every ``human/*/*/C2R.npy`` under ``human_root`` must be one matrix (else it dies)."""
    mats = [load_c2r(p) for p in sorted(Path(human_root).glob(f"*/*/{C2R_FILE}"))]
    if not mats:
        raise FileNotFoundError(f"No {C2R_FILE} anywhere under {human_root} -- no release-wide C2R to share.")
    if not all(np.allclose(m, mats[0], atol=SHARED_C2R_ATOL) for m in mats):
        raise ValueError(f"{C2R_FILE} differs between episodes under {human_root} -- not one release-wide matrix.")
    return mats[0]


def robot_from_world(c2r: np.ndarray) -> np.ndarray:
    """World (camera rig) → robot base, the official ``inv(C2R)``."""
    return np.linalg.inv(np.asarray(c2r, dtype=np.float64))


def hrdexdb_pose_npz(episode: Path) -> Path | None:
    """Per-frame object-pose npz for ``<root>/human/<object>/<seq>`` — v2 (the refit the dataset card recommends)
    before v1. The current release keeps them in ``object_6d_pose_v{2,1}/human/<object>_<seq>.npz``; the older
    release kept v2 in the episode folder (two current-release episodes ship v1 only). None = only the older
    ``object_6d/*.txt`` (v1) layout. Reader and desk oracle both call this (one owner)."""
    episode = Path(episode)
    cands = []
    if len(episode.parents) >= 3:
        root, stem = episode.parents[2], f"{episode.parent.name}_{episode.name}"
        cands.append(root / "object_6d_pose_v2" / "human" / f"{stem}.npz")
    cands.append(episode / "object_6d_pose_v2.npz")
    if len(episode.parents) >= 3:
        cands.append(root / "object_6d_pose_v1" / "human" / f"{stem}.npz")
    return next((f for f in cands if f.is_file()), None)


def hrdexdb_mesh(root: Path, obj: str) -> Path | None:
    """Object mesh under ``<root>/assets`` — ``mesh_v2`` first (the release's refit meshes), then ``mesh``."""
    return next(
        (
            m
            for m in (
                Path(root) / "assets/mesh_v2" / obj / f"{obj}.obj",
                Path(root) / "assets/mesh" / obj / f"{obj}.obj",
            )
            if m.is_file()
        ),
        None,
    )
