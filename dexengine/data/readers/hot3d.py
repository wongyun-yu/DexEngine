"""HOT3D (Meta, 2024) reader — pose-only download → contact-segmented clips → keypoint ``SequenceData``.

Raw layout (``<dataset_dir>/raw/hot3d/``, from the official downloader with data groups ``ground_truth`` +
``hand_data``; no VRS/video):
  <seq>/mano_hand_pose_trajectory.jsonl   records {timestamp_ns, hand_poses{'0': left, '1': right}} with per hand
                                          ``pose`` (15 MANO PCA coeffs), ``betas`` (10), ``wrist_xform`` {t_xyz, q_wxyz}
                                          — two records per frame (Δt 0.09 ms / 33.2 ms alternating) → deduped to 30 fps
  <seq>/dynamic_objects.csv               object_uid, timestamp[ns], t_wo_x/y/z, q_wo_w/x/y/z
                                          (world of the capture device — up axis per headset,
                                          ``source_frames.HEADSET_UP``)
  <seq>/metadata.json                     object_uids / object_names / participant_id / headset (``Aria`` | ``Quest3``)
  assets/<uid>.glb + assets/instance.json object meshes (metres) and names

World up axis: hand joints and object GT poses are turned into the gravity-up (+z) world every later layer assumes
(``_hoi_common.ground_offset`` grounds on z, base yaw turns about z) by ``source_frames.world_up`` — one rotation,
applied once to the joints (after ``_joints21``) and once to the raw object samples (before anything is derived).

MANO convention (HOT3D toolkit ``mano_layer.py``): ``smplx.create(..., use_pca=True, num_pca_comps=15)``,
``flat_hand_mean=False``, ``global_orient`` = wrist rotation as axis-angle, ``transl`` = wrist translation **as smplx
additive transl** (same as HO-Cap's ``trans_is_wrist=False``). We reuse HO-Cap's ``_joints21`` (params = rotvec 3 + 45
PCA (15 given, rest 0) + transl 3). The toolkit also corrects the published left model (``mano_layer.py`` "fix MANO
shapedirs of the left hand bug", smplx issue #48): when the left pkl's first shape direction ``shapedirs[:, 0, :]``
matches the right one's (sum of absolute differences < 1) it negates the left one. HOT3D's betas were fitted with that
correction (measured against the UmeTrack landmarks — ``mano_conventions``, the per-dataset statement every reader and
scene oracle reads), so ``mano_models`` applies it through ``taco.left_model`` and every clip records it in task_info
``mano_model`` (marker ``[HOT3D_MANO_LEFT]``).

Segmentation (D15, contact-based — a HOT3D recording is 1–2 min with several objects): per hand, per frame the nearest
object whose surface is within ``CONTACT_M`` of a fingertip; gaps ≤ ``BRIDGE`` frames bridged; maximal same-object runs
are contact windows; overlapping windows of the two hands merge into one clip; each hand's object in a clip = the object
it touches in ≥ ``MIN_FRAC`` of the clip's frames (else None → the hand is present but idle); pad ``PAD`` frames; drop
clips shorter than ``MIN_CLIP_S``. All other objects go to ``extra_objects``. Clip id: ``<seq>__s<start>_e<end>`` (frame
indices at 30 fps).

Usage (retarget venv):
  python -m dexengine.data.readers.hot3d --dataset-dir data/hot3d_stage --task P0001_10a27bf7 [--max-clips 3]
  python -m dexengine.data.readers.hot3d --dataset-dir data/hot3d_stage --all-sequences
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import loguru
import numpy as np
import trimesh
import tyro
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from dexengine.data.readers import mano_conventions as MC
from dexengine.data.readers.common import SequenceData, hand_track_from_joints21, write_sequence
from dexengine.data.readers.hocap import _joints21, _mano_dir
from dexengine.data.readers.hot3d_gt import has_ground_truth
from dexengine.data.readers.mano_lib import left_model, load_model
from dexengine.data.readers.source_frames import FRAME_VERSION, object_samples, to_z_up, world_up
from dexengine.retarget import mjutil

DATASET_NAME = "hot3d"
SOURCE_FPS = 30.0
HAND_KEY = {"left": "0", "right": "1"}  # HOT3D loader_hand_poses: LEFT_HAND_INDEX = 0, RIGHT_HAND_INDEX = 1
N_PCA = 15
CONTACT_M = 0.02
BRIDGE = 15
MIN_FRAC = 0.30
PAD = 10
MIN_CLIP_S = 1.0
MIN_HAND_VALID = 0.5  # a hand missing in more than half of the frames is dropped (clip becomes one-handed)
SURF_POINTS = 2000
#: Object GT gaps (see `_fill_gt_gaps`). A gap is the time between the GT samples that bound a run of hand frames with
#: no object sample. Up to this long it is interpolated (position linear · rotation slerp); longer gaps stay as they
#: are and are reported (task_info `object_gt_gap`) so the quality step flags the tree (`object_gt_gap`, D12-c). 0.3 s
#: (= 9 source frames); the gaps behind the 694 gap-type D31 drops were p10 0.33 s · median 0.73 s · p90 2.0 s — most
#: are occlusions the object moves through, which a straight line would invent. Declared in data.rules D12-c.
OBJ_GAP_INTERP_MAX_S = 0.3
#: A longer gap is still interpolated when the object did not move across it: the GT samples that bound the gap are
#: within this distance and rotation — the object sat still while its markers were hidden, so a straight line is the
#: truth, not an invention. Only a **moving** long gap is reported `long` (quality flag). Measured: 178 aerial · 184
#: desk candidate trees have only such still gaps. Declared in D12-c.
OBJ_GAP_STATIC_POS_M = 0.01
OBJ_GAP_STATIC_ROT_DEG = 5.0
#: a GT sample this close to a hand frame "measures" it (raw offsets: < 1 ms or ≥ 30 ms)
OBJ_SAMPLE_TOL_S = 0.25 / SOURCE_FPS
#: task_info ``source_frame.rule`` — where the up-axis rule lives (``HEADSET_UP`` · ``world_up``); its presence
#: says it ran.
SOURCE_FRAME_RULE = "dexengine/data/readers/source_frames.py:HEADSET_UP"
#: the left-model threshold, read from its owner (``mano_conventions``) for the [HOT3D_MANO_LEFT] marker
MANO_LEFT_FIX_SUM = MC.LEFT_SD0_DEFECT_SUM


def mano_models(mano_dir: str | Path) -> tuple[dict, dict, dict]:
    """(right, left, record) — the MANO models HOT3D's hand poses were fitted with. The left model is corrected by the
    toolkit's rule (module doc); the right model is the pkl as is. record = task_info ``mano_model`` (rule · the
    measured sum · whether the left model was negated) — a tree's generation is readable from it."""
    right = load_model(model_type="manor", model_path=str(mano_dir))
    left, record = left_model(DATASET_NAME, right, load_model(model_type="manol", model_path=str(mano_dir)))
    return right, left, record


class NoGroundTruth(ValueError):
    """Recording without published GT (test split); out of scope rather than a failure."""


def _root(dataset_dir: Path) -> Path:
    return dataset_dir / "raw" / "hot3d"


def list_sequences(dataset_dir: str | Path) -> list[str]:
    root = _root(Path(os.path.abspath(os.path.expanduser(str(dataset_dir)))))
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "mano_hand_pose_trajectory.jsonl").is_file())


def _frames(seq_dir: Path):
    """Dedupe the two-records-per-frame jsonl by ms-rounded timestamp → (ts_ns (T,), list of hand_poses dicts)."""
    ts, hp, seen = [], [], set()
    with open(seq_dir / "mano_hand_pose_trajectory.jsonl") as f:
        for line in f:
            r = json.loads(line)
            k = int(round(r["timestamp_ns"] / 1e6))
            if k in seen:
                continue
            seen.add(k)
            ts.append(int(r["timestamp_ns"]))
            hp.append(r["hand_poses"])
    order = np.argsort(ts)
    return np.asarray(ts)[order], [hp[i] for i in order]


def _hand_params(hp: list[dict], side: str):
    """(T,51) MANO params [rotvec 3 | 45 PCA (15 given) | transl 3], (T,) valid, betas (10) — NaN where the hand is
    absent."""
    T = len(hp)
    P = np.full((T, 51), np.nan)
    valid = np.zeros(T, bool)
    betas = []
    for t, h in enumerate(hp):
        d = h.get(HAND_KEY[side])
        if not d:
            continue
        q = d["wrist_xform"]["q_wxyz"]
        rv = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()
        pca = np.zeros(45)
        pca[:N_PCA] = np.asarray(d["pose"], float)[:N_PCA]
        P[t, :3] = rv
        P[t, 3:48] = pca
        P[t, 48:51] = d["wrist_xform"]["t_xyz"]
        valid[t] = True
        betas.append(d["betas"])
    if valid.mean() < MIN_HAND_VALID:
        return None, valid, None
    # gaps: linear interpolation on the parameter vector (rotvec linear is fine for short gaps; flagged in interp_mask)
    idx = np.arange(T)
    for j in range(51):
        P[:, j] = np.interp(idx, idx[valid], P[valid, j])
    return P, valid, np.median(np.asarray(betas, float), axis=0)


def _nearest(a: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Index of the GT sample nearest each hand-frame timestamp."""
    ots = a[:, 0]
    j = np.clip(np.searchsorted(ots, ts), 0, len(ots) - 1)
    j0 = np.clip(j - 1, 0, len(ots) - 1)
    return np.where(np.abs(ots[j] - ts) <= np.abs(ots[j0] - ts), j, j0)


def _objects(samples: dict[str, np.ndarray], ts: np.ndarray) -> dict[str, np.ndarray]:
    """{uid: (T,7) [x y z qw qx qy qz]} resampled (nearest) to the hand frame timestamps.

    This is the **segmentation input** (D15 contact windows) and stays nearest-pick on purpose: the clip names
    (`__s<a>_e<b>`) are derived from it, and changing it would rename clips. The poses that are written out go through
    `_fill_gt_gaps` (object GT gaps)."""
    return {uid: a[_nearest(a, ts), 1:8] for uid, a in samples.items()}


def _fill_gt_gaps(a: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, float, str]]]:
    """Output poses for one object and its GT gaps — **named** handling (D31 investigation).

    HOT3D `dynamic_objects.csv` has **no sample** while the object's markers are occluded. Nearest-pick freezes the
    object for the first half of such a gap and then delivers the whole displacement in one frame (D31 drops such
    trees; the true speed across those gaps is p99 2.0 m/s). A hand frame is *measured* when a GT sample lies within
    `OBJ_SAMPLE_TOL_S` of it (raw offsets are bimodal: < 1 ms or ≥ 30 ms). A maximal run of unmeasured frames is a gap;
    its length is the time between the GT samples that bound it:
      · ≤ `OBJ_GAP_INTERP_MAX_S` → position linear, rotation slerp between the bounding samples (`interp`)
      · longer, but the bounding samples are within `OBJ_GAP_STATIC_POS_M` · `OBJ_GAP_STATIC_ROT_DEG` (the object sat
        still while hidden) → interpolated the same way (`static`)
      · longer and the object moved (or open at a recording end) → left as nearest-pick and reported as `long` — the
        quality step (`dexengine.data.filtering`) flags a tree whose main object has one (`object_gt_gap`, D12-c).
    Measured frames are exactly the nearest samples.
    Returns (poses (T,7), [(i0, i1, gap_s, "interp"|"static"|"long")]) with frames i0..i1-1 unmeasured."""
    pick = _nearest(a, ts)
    out = a[pick, 1:8].copy()
    measured = np.abs(a[pick, 0] - ts) <= OBJ_SAMPLE_TOL_S * 1e9
    gaps = []
    i, T = 0, len(ts)
    while i < T:
        if measured[i]:
            i += 1
            continue
        j = i
        while j < T and not measured[j]:
            j += 1
        k_prev = int(np.searchsorted(a[:, 0], ts[i], side="right")) - 1  # last GT sample before the run
        k_next = int(np.searchsorted(a[:, 0], ts[j - 1], side="left"))  # first GT sample after the run
        if k_prev < 0 or k_next >= len(a):
            gaps.append((i, j, float("inf"), "long"))
            i = j
            continue
        t0, t1 = a[k_prev, 0], a[k_next, 0]
        gap_s = float((t1 - t0) / 1e9)
        q0, q1 = a[k_prev, 4:8], a[k_next, 4:8]
        kind = "interp"
        if gap_s > OBJ_GAP_INTERP_MAX_S:
            moved_m = float(np.linalg.norm(a[k_next, 1:4] - a[k_prev, 1:4]))
            turned_deg = float(
                np.degrees(
                    2 * np.arccos(min(1.0, abs(float(np.dot(q0, q1)) / (np.linalg.norm(q0) * np.linalg.norm(q1)))))
                )
            )
            if moved_m >= OBJ_GAP_STATIC_POS_M or turned_deg >= OBJ_GAP_STATIC_ROT_DEG:
                gaps.append((i, j, gap_s, "long"))
                i = j
                continue
            kind = "static"
        w = (ts[i:j] - t0) / (t1 - t0)
        out[i:j, :3] = (1 - w)[:, None] * a[k_prev, 1:4] + w[:, None] * a[k_next, 1:4]
        rot = Slerp([0.0, 1.0], Rot.from_quat(np.stack([q0, q1])[:, [1, 2, 3, 0]]))(w).as_quat()[:, [3, 0, 1, 2]]
        rot[np.sum(rot * q0, axis=1) < 0] *= -1.0  # keep the bounding sample's hemisphere
        out[i:j, 3:7] = rot
        gaps.append((i, j, gap_s, kind))
        i = j
    return out, gaps


def _clip_gaps(filled: dict, names: dict, a0: int, b0: int) -> dict:
    """task_info `object_gt_gap` for clip frames a0..b0-1 — **every** clip gets the key (its presence says the gaps were
    measured; the quality step dies on a hot3d tree without it). Intervals are in seconds from the clip start on the
    source clock and cover the unmeasured frames widened by one frame each side (nearest-pick moves the jump anywhere
    inside that span, and the 50 Hz resample smears it over the neighbours)."""
    objects = {}
    for uid, (_, gaps) in filled.items():
        rows = {"interp": [], "static": [], "long": []}
        for i0, i1, gap_s, kind in gaps:
            if i1 <= a0 or i0 >= b0:
                continue
            t0 = (max(i0 - 1, a0) - a0) / SOURCE_FPS
            t1 = (min(i1, b0 - 1) - a0) / SOURCE_FPS
            rows[kind].append([round(t0, 4), round(t1, 4), None if not np.isfinite(gap_s) else round(gap_s, 4)])
        if rows["interp"] or rows["static"] or rows["long"]:
            objects[names.get(uid, uid)] = rows
    return {
        "rule": "dexengine/data/readers/hot3d.py:_fill_gt_gaps",
        "interp_max_s": OBJ_GAP_INTERP_MAX_S,
        "static_pos_m": OBJ_GAP_STATIC_POS_M,
        "static_rot_deg": OBJ_GAP_STATIC_ROT_DEG,
        "clock": "source seconds from clip start",
        "objects": objects,
    }


def _surface_points(glb: Path, n: int = SURF_POINTS) -> np.ndarray:
    """Surface samples with a **fixed seed** (different samples shift contact boundaries by a frame and change clip
    names; measured on e00295/e00296)."""
    m = trimesh.load(str(glb), force="mesh", process=False)
    if not len(m.faces):
        return np.asarray(m.vertices, float)[:n]
    try:
        pts = trimesh.sample.sample_surface(m, n, seed=20260916)[0]
    except TypeError:  # old trimesh without a seed argument → fix the global RNG
        st = np.random.get_state()
        np.random.seed(20260916)
        pts = trimesh.sample.sample_surface(m, n)[0]
        np.random.set_state(st)
    return np.asarray(pts, float)


def _touch_labels(tips: np.ndarray, objs: dict[str, np.ndarray], surf: dict[str, np.ndarray]) -> list[str | None]:
    """Per frame: uid of the nearest object whose surface is within CONTACT_M of any fingertip, else None. Gaps ≤ BRIDGE
    bridged."""
    T = tips.shape[0]
    best = np.full(T, np.inf)
    lab = np.array([None] * T, dtype=object)
    for uid, pose in objs.items():
        pts = surf[uid]
        for t in range(T):
            if not np.isfinite(tips[t]).all():
                continue
            R = Rot.from_quat(pose[t, [4, 5, 6, 3]]).as_matrix()
            w = pts @ R.T + pose[t, :3]
            d = cKDTree(w).query(tips[t], k=1)[0].min()
            if d < CONTACT_M and d < best[t]:
                best[t] = d
                lab[t] = uid
    out = list(lab)
    # bridge: None runs ≤ BRIDGE between identical labels
    t = 0
    while t < T:
        if out[t] is None:
            u = t
            while u < T and out[u] is None:
                u += 1
            if t > 0 and u < T and out[t - 1] == out[u] and (u - t) <= BRIDGE:
                for k in range(t, u):
                    out[k] = out[u]
            t = u
        else:
            t += 1
    return out


def _windows(labels: list) -> list[tuple[int, int, str]]:
    out = []
    t = 0
    T = len(labels)
    while t < T:
        if labels[t] is None:
            t += 1
            continue
        u = t
        while u < T and labels[u] == labels[t]:
            u += 1
        out.append((t, u, labels[t]))
        t = u
    return out


def _merge(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ws = sorted(windows)
    out = []
    for a, b in ws:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _hand_joints(dataset_dir: Path, task: str, hp: list[dict], turn: bool, rot: np.ndarray) -> tuple[dict, dict, dict]:
    """(joints, valid, mano model record): each usable hand's (T, 21, 3) joints in the z-up world and its per-frame
    validity; a hand missing in too many frames is dropped, ValueError when none is left."""
    joints, valid = {}, {}
    m_right, m_left, mano_model = mano_models(_mano_dir(dataset_dir))
    print(
        f"[HOT3D_MANO_LEFT] task={task} left_sd0_sum={mano_model['left_sd0_sum']:.4f} "
        f"threshold={MANO_LEFT_FIX_SUM} left_flipped={int(mano_model['left_flipped'])}",
        flush=True,
    )
    for side in ("right", "left"):
        P, v, betas = _hand_params(hp, side)
        if P is None:
            loguru.logger.warning(f"{task}: {side} hand present in only {v.mean() * 100:.0f}% of frames — dropped")
            continue
        joints[side] = _joints21(betas, P, m_right if side == "right" else m_left)
        valid[side] = v
        if turn:
            joints[side] = joints[side] @ rot.T
    if not joints:
        raise ValueError(f"{task}: no usable hand")
    return joints, valid, mano_model


def _hands_and_objects(seq, joints, labels, filled, names, root, a0, b0) -> tuple[set, dict]:
    """Put each hand's track into the clip and, when it touches one object in at least MIN_FRAC of the clip, that
    object; returns (objects used, {side: object})."""
    used, used_side = set(), {}
    for side in joints:
        setattr(seq, side, hand_track_from_joints21(joints[side][a0:b0], side))
        cnt = defaultdict(int)
        for lb in labels[side][a0:b0]:
            if lb is not None:
                cnt[lb] += 1
        if cnt:
            uid, n = max(cnt.items(), key=lambda kv: kv[1])
            if n / (b0 - a0) >= MIN_FRAC:
                setattr(seq, f"obj_{side}", filled[uid][0][a0:b0].copy())
                setattr(seq, f"mesh_{side}", root / "assets" / f"{uid}.glb")
                setattr(seq, f"mesh_name_{side}", names.get(uid, uid))
                used.add(uid)
                used_side[side] = uid
    return used, used_side


def load_clips(dataset_dir: str | Path, task: str, max_clips: int = 0) -> list[SequenceData]:
    """The contact-segmented clips of one recording (module doc); NoGroundTruth for a recording without GT."""
    dataset_dir = Path(os.path.abspath(os.path.expanduser(str(dataset_dir))))
    root = _root(dataset_dir)
    seq_dir = root / task
    meta = json.loads((seq_dir / "metadata.json").read_text())
    inst = json.loads((root / "assets" / "instance.json").read_text())
    # only recordings with GT (owner hot3d_gt.has_ground_truth; a missing or inconsistent marker raises ValueError)
    if not has_ground_truth(seq_dir):
        raise NoGroundTruth(f"{task}: no hand/object GT (test split, headset={meta.get('headset')})")
    up, rot = world_up(meta.get("headset"))
    turn = up != "+z"  # +z-up recordings pass through untouched (byte-identical arrays)
    print(
        f"[HOT3D_UP] task={task} headset={meta['headset']} up={up} R={np.round(rot).astype(int).tolist()}", flush=True
    )
    ts, hp = _frames(seq_dir)
    T = len(ts)
    joints, valid, mano_model = _hand_joints(dataset_dir, task, hp, turn, rot)
    samples = to_z_up(object_samples(seq_dir), rot)  # objs · filled · ground_first_pose are all derived from these
    source_frame = {
        "rule": SOURCE_FRAME_RULE,
        "version": FRAME_VERSION,
        "headset": meta["headset"],
        "up": up,
        "R_to_z_up": rot.tolist(),
    }
    objs = _objects(samples, ts)
    names = {uid: inst[uid]["instance_name"] for uid in objs if uid in inst}
    # written poses + GT gaps (segmentation keeps `objs`)
    filled = {uid: _fill_gt_gaps(a, ts) for uid, a in samples.items()}
    surf = {uid: _surface_points(root / "assets" / f"{uid}.glb") for uid in objs}
    tips_idx = [4, 8, 12, 16, 20]
    labels = {s: _touch_labels(joints[s][:, tips_idx], objs, surf) for s in joints}
    wins = [(a, b) for s in labels for (a, b, _) in _windows(labels[s])]
    clips = []
    for a, b in _merge(wins):
        a0, b0 = max(0, a - PAD), min(T, b + PAD)
        if (b0 - a0) / SOURCE_FPS < MIN_CLIP_S:
            continue
        seq = SequenceData(task=f"{task}__s{a0:05d}_e{b0:05d}", source_fps=SOURCE_FPS)
        used, used_side = _hands_and_objects(seq, joints, labels, filled, names, root, a0, b0)
        if seq.obj_right is None and seq.obj_left is None:
            continue
        seq.extra_objects.extend(
            (names.get(uid, uid), root / "assets" / f"{uid}.glb", filled[uid][0][a0:b0].copy())
            for uid in objs
            if uid not in used
        )
        seq.object_gt_gap = _clip_gaps(filled, names, a0, b0)
        seq.source_frame = source_frame
        seq.mano_model = mano_model
        # the ground offset stays anchored on the nearest-pick first frame (the frame convention every INTAKE ledger
        # was measured in) — gap filling changes object poses, never where the floor is
        seq.ground_first_pose = {s: objs[u][a0 : a0 + 1].copy() for s, u in used_side.items()}
        m = np.zeros(b0 - a0, bool)
        for side in joints:
            m |= ~valid[side][a0:b0]
        seq.interp_mask = m
        clips.append(seq)
        if max_clips and len(clips) >= max_clips:
            break
    loguru.logger.info(f"{task}: frames {T} · hands {sorted(joints)} · objects {len(objs)} · clips {len(clips)}")
    return clips


def _write_recording(dataset_dir, tk, max_clips, embodiment_type, target_ref_dt, ground, fails, no_gt) -> int:
    """Write every clip of one recording; returns how many. No GT goes to ``no_gt``, any other failure to ``fails``."""
    n_clips = 0
    try:
        for seq in load_clips(dataset_dir, tk, max_clips=max_clips):
            emb = (
                embodiment_type
                if (seq.right is not None and seq.left is not None)
                else ("right" if seq.right is not None else "left")
            )
            out = write_sequence(seq, dataset_dir, DATASET_NAME, emb, target_ref_dt=target_ref_dt, ground=ground)
            n_clips += 1
            loguru.logger.info(f"  → {out}")
    except NoGroundTruth as e:
        no_gt.append(tk)
        loguru.logger.info(str(e))
    except Exception as e:  # noqa: BLE001  one bad recording is logged and counted; the batch continues
        fails.append((tk, f"{type(e).__name__}: {e}"))
        loguru.logger.error(f"{tk}: {e}")
    return n_clips


def main(
    dataset_dir: str = f"{mjutil.ROOT}/datasets",
    task: str = "P0001_10a27bf7",
    target_ref_dt: float = 0.02,
    ground: bool = True,
    all_sequences: bool = False,
    max_clips: int = 0,
    embodiment_type: str = "bimanual",
) -> None:
    tasks = list_sequences(dataset_dir) if all_sequences else [task]
    n_clips = 0
    fails = []
    no_gt = []
    for tk in tasks:
        n_clips += _write_recording(dataset_dir, tk, max_clips, embodiment_type, target_ref_dt, ground, fails, no_gt)
    print(
        f"[hot3d] sequences {len(tasks)} · clips written {n_clips} · no-GT(test split) {len(no_gt)} · failed "
        f"sequences {len(fails)}",
        flush=True,
    )
    for tk, why in fails[:10]:
        print(f"  FAIL {tk}: {why[:160]}")
    if fails:
        raise SystemExit(f"[hot3d] ✗ failed sequences {len(fails)}")


if __name__ == "__main__":
    tyro.cli(main)
