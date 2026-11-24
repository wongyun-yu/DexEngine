#!/usr/bin/env python3
"""Build a shim raw-TACO dir whose object data is LIFTED from prior pipeline output.

WHY
    The raw TACO ``Object_Poses/`` and ``object_models_released/`` directories
    are not available here (measured), yet every prior product
    derived from them does: per-sequence object trajectories live inside the
    legacy-hand ``trajectory_kinematic.npz`` (50 Hz output grid) and the prepared
    object meshes live in ``meshes/<id>/visual.obj`` (metres). This script
    reconstructs, for a REQUESTED SET of sequences only, the raw layout that
    ``taco_convert.py`` expects — so the stock handoff pipeline runs unpatched.

WHAT IT WRITES (under --shim-raw-dir)
    Hand_Poses/<triplet>/<sid>/{side}_hand.pkl        real files (frame shift applied, see below)
    Hand_Poses/<triplet>/<sid>/{side}_hand_shape.pkl  real COPY of the raw shape file (445 B) — not a symlink
                                         (a link farm of 4,634 entries otherwise; an older build's links are replaced)
    Object_Poses/<triplet>/<sid>/{tool_<id>.npy, target_<id>.npy}
                                         (T_src, 4, 4) float32, metres, 30 Hz
    object_models_released/<id>_cm.obj   visual.obj scaled x100 (m -> cm)

    Only the requested sequences appear under Hand_Poses, so batch enumeration
    is naturally limited to them — no filtering patches needed downstream.

HAND FRAME SHIFT (found by the wrist-object distance gate)
    The NEW pipeline works in the RAW capture frame end-to-end; the OLD pipeline
    lowered the world so the table sits near z=0. Feeding old-frame objects with
    raw-frame hands put wrists 49-66 cm from their objects — physically
    impossible, caught only by the wrist-to-object distance check (schema-level
    comparison passed). The normalisation was recovered by fitting MANO-wrist
    trajectories (raw hand_trans vs the dev-produced keypoints) on all 9
    sequences that have both: rotation <= 0.101 deg, translation
    [0, 0, -0.5476] with 2.3 mm cross-sequence scatter (4 of 9 fit at exactly
    0.00 mm residual). So Hand_Poses in the shim are REAL files with
    hand_trans += HAND_SHIFT (global constant; +-3 mm provisional error),
    not symlinks; hand_pose is untouched because the rotation is ~0.

FIDELITY NOTES (measured, documented in the run log)
    * Baseline right-object quaternions are UNNORMALISED (up to 10% off unit —
      an old-resampler artifact). They are sign-aligned, normalised, then
      n-lerped to source timestamps; the loader's proper-rotation check would
      reject anything less.
    * The old resampler dropped more tail than the new one (150 vs 154 frames
      on the probe sequence), so the last ~2 source frames may fall beyond the
      baseline grid; they are edge-held. Objects are typically settled by then.
    * Quaternion order in the baseline qpos block is MuJoCo wxyz. This is
      cross-validated on sequences where the true
      phase-1 output exists.

USAGE
    <retarget venv python> -m dexengine.data.readers.taco_lift \
        --tasks-file <bases.txt> \
        --trajectories-root "$(paths.py --what taco-source-root)" \
        --meshes-root .../data/taco/meshes \
        --real-raw .../data/taco/raw_unzipped \
        --shim-raw-dir .../data/taco_retarget/raw/taco
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import shutil

import numpy as np
import trimesh

from dexengine.core import shipped
from dexengine.data import side_objects
from dexengine.data.readers import taco_source
from dexengine.data.readers.taco_convert import taco_task_name  # the clip naming the converter uses
from dexengine.data.readers.taco_frames import HAND_SHIFT, hand_shift, is_session_exception

RIGHT_OBJ = slice(0, 7)  # the source object block (taco_source.OBJECT_COLS): right [pos3, quat4(wxyz)] · left
LEFT_OBJ = slice(7, 14)
SRC_DT = 1.0 / 30.0  # task_info.source_ref_dt, asserted per sequence


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def lift_block(qpos_blk: np.ndarray, freq: float, t_src: np.ndarray) -> np.ndarray:
    """(T50, 7) pos+quat(wxyz) on the output grid -> (T_src, 4, 4) at source times."""
    pos, quat = qpos_blk[:, :3].astype(np.float64), qpos_blk[:, 3:].astype(np.float64)
    # Sign-align consecutive quats (q and -q are the same rotation; unaligned
    # signs would make interpolation swing through zero), then normalise.
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    t_grid = np.arange(len(pos)) / freq
    t = np.clip(t_src, t_grid[0], t_grid[-1])  # edge-hold beyond coverage
    out = np.empty((len(t), 4, 4), dtype=np.float32)
    out[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
    for k, tk in enumerate(t):
        j = min(int(np.searchsorted(t_grid, tk, side="right")) - 1, len(t_grid) - 2)
        a = 0.0 if t_grid[j + 1] == t_grid[j] else (tk - t_grid[j]) / (t_grid[j + 1] - t_grid[j])
        p = (1 - a) * pos[j] + a * pos[j + 1]
        q = (1 - a) * quat[j] + a * quat[j + 1]  # n-lerp; 20 ms steps
        q /= np.linalg.norm(q)
        out[k, :3, :3] = quat_wxyz_to_mat(q)
        out[k, :3, 3] = p
    return out


def obj_id_of(mesh_dir: str) -> str:
    return pathlib.Path(mesh_dir).name


def _raw_index(real_raw: pathlib.Path) -> dict[str, tuple[str, str]]:
    """{clip id: (triplet, sequence id)} of every raw Hand_Poses sequence.

    Forward map task -> (triplet, sid) via the converter's OWN naming function,
    so no reverse-parsing of names that legally contain spaces/underscores.
    """
    hp = real_raw / "Hand_Poses"
    by_task = {}
    for trip in sorted(hp.iterdir()):
        for sid in sorted(trip.iterdir()):
            by_task[taco_task_name(trip.name, sid.name)] = (trip.name, sid.name)
    return by_task


def _write_side_assign(od: pathlib.Path, src: taco_source.Source, task: str, _shift: np.ndarray) -> None:
    """State the hand↔object assignment and the baked hand shift in the shim's sidecar.

    The converter (`taco_convert_io._read_side_assign`) reads it so it does not assume "the tool is in the right
    hand", and learns that the hand shift is already baked, which prevents applying it twice.
    """
    _sa = side_objects.resolve_ids(src.scene_ids, src.info, task)
    (od / "side_assign.json").write_text(
        json.dumps(
            {
                "right": _sa["right"][0],
                "left": _sa["left"][0],
                "right_slot": _sa["right"][1],
                "left_slot": _sa["left"][1],
                "method": side_objects.provenance_of(src.info),
                "hand_shift_z": float(-_shift[2]),
                "hand_shift_baked": True,
            },
            indent=1,
            ensure_ascii=False,
        )
    )


def _write_shifted_hands(src: pathlib.Path, hd: pathlib.Path, _shift: np.ndarray) -> None:
    """Real per-sequence Hand_Poses dirs with the frame shift applied (not symlinks to the raw dirs: see HAND FRAME
    SHIFT above for why); the shape pickles are copied as they are."""
    if hd.is_symlink():
        hd.unlink()  # a symlinked dir from an older shim
    hd.mkdir(parents=True, exist_ok=True)
    for side in ("right", "left"):
        with open(src / f"{side}_hand.pkl", "rb") as f:
            hp = pickle.load(f)
        for fr in hp.values():
            fr["hand_trans"] = (np.asarray(fr["hand_trans"], dtype=np.float64) + _shift).astype(np.float32)
        with open(hd / f"{side}_hand.pkl", "wb") as f:
            pickle.dump(hp, f)
        shp = hd / f"{side}_hand_shape.pkl"
        if shp.is_symlink():
            shp.unlink()  # older build linked the raw file — a copy now (no link farm)
        if not shp.exists():
            shutil.copy2(src / f"{side}_hand_shape.pkl", shp)


def slot_mesh_ids(src: taco_source.Source, task: str) -> tuple[str, str]:
    """(tool, target) mesh ids that name the slot files: qpos slot 0 (RIGHT_OBJ) carries scene.xml's right_visual mesh,
    slot 1 (LEFT_OBJ) its left_visual mesh (convention A, side_objects; taco_source.Source.slot_ids). task_info's
    right/left mesh dirs are the measured grasp, which hand holds which object (convention B); they name the other
    slot on left-hand-tool clips (52 of the 2,317 V1 trees), and naming the files from them paired each mesh with the
    other object's trajectory. side_assign.json keeps the grasp. A difference is printed, not an error:
    [TACO_MESH]."""
    rid, lid = src.slot_ids
    grasp = (obj_id_of(src.info["right_object_mesh_dir"]), obj_id_of(src.info["left_object_mesh_dir"]))
    if grasp != (rid, lid):
        print(
            f"[TACO_MESH] scene≠task_info clip={task} slots(tool,target)=({rid},{lid}) grasp(R,L)={grasp} "
            "→ the files follow the scene",
            flush=True,
        )
    return rid, lid


def _lift_sequence(a: argparse.Namespace, task: str, trip: str, sid: str) -> tuple[str, str]:
    """Write one sequence's shim (object poses at source times, grid-50 sidecars, side_assign.json, shifted hands);
    returns its (tool, target) object ids (slot 0, slot 1)."""
    src = taco_source.read(a.trajectories_root, a.source_kind, task)
    info = src.info
    assert abs(info["source_ref_dt"] - SRC_DT) < 1e-9, (task, info["source_ref_dt"])
    rid, lid = slot_mesh_ids(src, task)

    obj, freq = src.object_qpos, src.frequency  # (T50, 14): right object 7 · left object 7
    # the shift is **per session (capture date)**: 20231002 has a different world origin, and the global constant
    # would put the hand 0.72 m too low. Only the one resolver is called.
    _shift = hand_shift(sid)
    if is_session_exception(sid):
        print(f"[lift] {task}: session {sid[:8]} exception z shift {_shift[2]:+.4f} (default {HAND_SHIFT[2]:+.4f})")

    # Source frame count = hand pickle frame count (converter validates it).
    with open(a.real_raw / "Hand_Poses" / trip / sid / "right_hand.pkl", "rb") as f:
        rh = pickle.load(f)
    t_src = np.arange(len(rh)) * SRC_DT

    od = a.shim_raw_dir / "Object_Poses" / trip / sid
    od.mkdir(parents=True, exist_ok=True)
    np.save(od / f"tool_{rid}.npy", lift_block(obj[:, RIGHT_OBJ], freq, t_src))
    np.save(od / f"target_{lid}.npy", lift_block(obj[:, LEFT_OBJ], freq, t_src))
    # grid-50 sidecars: the baseline 7-dof block VERBATIM (float64, wxyz,
    # unnormalised quats included — the baseline trained on these values).
    # The kinematic phase passes them through, bypassing the 50->30->50
    # round trip that smeared fast tool motion by 5.6~11.0 mm median.
    np.save(od / f"tool_{rid}.grid50.npy", np.asarray(obj[:, RIGHT_OBJ], dtype=np.float64))
    np.save(od / f"target_{lid}.grid50.npy", np.asarray(obj[:, LEFT_OBJ], dtype=np.float64))
    _write_side_assign(od, src, task, _shift)

    src = a.real_raw / "Hand_Poses" / trip / sid
    hd = a.shim_raw_dir / "Hand_Poses" / trip / sid
    _write_shifted_hands(src, hd, _shift)
    return rid, lid


def _write_meshes(a: argparse.Namespace, mesh_ids: set[str]) -> int:
    """Each object mesh in the raw layout (centimetres) unless it is already there; returns how many were written."""
    n_mesh = 0
    rel = a.shim_raw_dir / "object_models_released"
    rel.mkdir(parents=True, exist_ok=True)
    for oid in sorted(mesh_ids):
        dst = rel / f"{oid}_cm.obj"
        if dst.exists():
            continue
        m = trimesh.load(a.meshes_root / oid / "visual.obj", force="mesh")
        m.apply_scale(100.0)  # metres -> the raw layout's centimetres
        m.export(dst)
        n_mesh += 1
    return n_mesh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-file", required=True, type=pathlib.Path)
    ap.add_argument("--trajectories-root", required=True, type=pathlib.Path)
    ap.add_argument("--meshes-root", required=True, type=pathlib.Path)
    ap.add_argument("--real-raw", required=True, type=pathlib.Path, help="dir containing the real Hand_Poses/")
    ap.add_argument("--shim-raw-dir", required=True, type=pathlib.Path)
    a = ap.parse_args()

    tasks = [ln.strip() for ln in a.tasks_file.read_text().splitlines() if ln.strip()]
    a.source_kind = taco_source.kind(a.trajectories_root)
    shipped.state("taco_meshes", a.meshes_root)  # a release root is verified; an unverified root passes as `site`
    by_task = _raw_index(a.real_raw)

    missing = [t for t in tasks if t not in by_task]
    if missing:
        print(f"[lift] {len(missing)} task(s) not found in raw Hand_Poses: {missing[:3]}")
        return 2

    n_seq = 0
    mesh_ids: set[str] = set()
    for task in tasks:
        trip, sid = by_task[task]
        rid, lid = _lift_sequence(a, task, trip, sid)
        mesh_ids |= {rid, lid}
        n_seq += 1
    n_mesh = _write_meshes(a, mesh_ids)

    print(
        f"[lift] sequences {n_seq}/{len(tasks)}  meshes written {n_mesh} "
        f"(ids total {len(mesh_ids)})  -> {a.shim_raw_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
