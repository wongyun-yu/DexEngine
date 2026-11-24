"""Raw scene object measurement for OakInk2: per clip window, what each raw object did (touch, motion, receiver),
read from the raw dataset beside a tree's objects[]. It imports no pipeline module and no path owner — the caller
(the record's render-only object list, record/scene/render_only_build.py) hands it the raw root, the tree and the
MANO dir.

Measurement (raw only):
  OakInk2 anno_preview pkl (obj_list · obj_transf · raw_mano) · program/program_info (stage intervals · obj_list per
  primitive) · object_affordance/object_part_tree.json (official part tree) ·
  obj/object_repair/align_ds/<rid>/model.obj.
  Per clip window and per raw object: human touch (scene_requirement's six hand points — five fingertips and the
  palm point near the object surface on any window frame, per hand; the hand-surface contact, MANO vertices <=
  TOUCH_M, is measured beside it — readers/raw/common),
  motion (centroid displacement >= MOVE_M or rotation >= ROT_DEG vs the window's first frame, median filtered),
  receiver (a human-moved object comes to rest on / inside it within the window — geometry only; the stage program's
  obj_list is carried as the `declared` column, not a requirement), official part siblings.
Statement (the tree a hand trains on): <clip>_pl_rt/task_info.json objects[] — role · side · obj_id · articulation ·
  idle_sides. Spawn rule (what the RL env builds — read from code, the RL env setup's
  apply_idle_side / _build_support_slots · qpos_layout.py main_articulation; quoted here, never imported — the
  extra half is read from its owner):
    spawned = (role main ∧ side ∉ idle_sides)  [+ both parts base_id/moving_id and the parts fixed to a link
              (articulation.parts.fixed[].id — one rigid body with that link) when that main carries `articulation`]
            ∨ (role support ∧ no articulation)  [kinematic, fixed at frame 0]
    An entry with `fixed_merge` (a merged-mesh asset: official parts of one instance joined into one rigid body, its
    obj_id = the asset id) stands for the raw parts it states (fixed_merge.parts) — each is spawned as the entry is;
    the raw scene is measured per part the asset covers (scene_classify.object_ids · tree_scene).
    role extra: spawned as its owner states it — extra_slots (switch requested · rule slot_rows → SLOTTED), never
    quoted here — `extra_slot` (kinematic) or `extra_static` (frame 0).
  Video (the collect render's extra choice) is approximated: an extra that is static, untouched and not
  a part of an instance with a part in the tree = video candidate (desk bounds · hand-near · rollout overlap not
  re-applied).

Clip -> raw window (rule): clip name `oakink2_<task>__scene_<NN>_<hash12>_s<stage>[_segNN][_tN]`.
  hash12 = the first 12 hex chars of the anno_preview pkl's seq id; stage = the index into program_info's key order;
  the stage's raw interval [a0, b0) = (min of the non-null hand interval starts, max of their ends) in 120 Hz mocap
  frames.
  A segment / cut (_segNN, _tN in the name) is the tree's `segment.start/end` — 50 Hz frames from a0 (base_adapter
  convention, verified: T_mano == (b0-a0)/2.4) -> window [a0 + round(start*2.4), a0 + round(end*2.4)); a name without
  them is the whole stage. Retiming changes the time warp, not the interval — only per-object scalars are compared.
  Active hand = its program interval overlaps the window (raw); the tree's idle_sides is reported beside it.

MANO (fitted on e8 open_gate s005: left hand to door median 1.1 mm): smplx MANO, flat_hand_mean, no PCA; pose_coeffs are
  16 quaternions wxyz (-> xyzw for scipy) · vertices re-centred on joint 0 then + tsl (manotorch center_idx=0) ·
  the left model = mano_conventions (held against the delivered keypoints by `--hand-check`).
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import pickle
import re
from pathlib import Path

import numpy as np

from dexengine.data.readers.raw import common as C

MOCAP_HZ, TREE_HZ = 120.0, 50.0
STEP = 4  # mocap frames per sample (30 Hz)
CLIP_RE = re.compile(
    r"oakink2_(?P<task>.+?)__scene_(?P<scene>\d+)_(?P<h>[0-9a-f]{12})_s(?P<stage>\d+)"
    r"(?P<seg>_seg\d+)?(?P<cut>_t\d+)?"
)


def rid_of(obj_id: str) -> str:
    """tree obj_id (`right_o02_0039_00002` · `raw_…` · `supp_…` · `left_…`) -> raw rid `O02@0039@00002`. Any other
    shape dies."""
    s = re.sub(r"^(right|left|raw|supp)_", "", str(obj_id))
    rid = s.upper().replace("_", "@")
    if not re.fullmatch(r"O\d\d@\d{4}@\d{5}|[A-Z]\d{4,6}", rid):  # anno_preview ids: O02@0039@00002 · O93001
        raise ValueError(f"obj_id {obj_id!r} -> {rid!r} is not an OakInk2 rid")
    return rid


# ---------------------------------------------------------------- raw side (measurement)
class OracleError(Exception):
    """raw input the oracle cannot read — the caller dies or records the clip as un-auditable."""


class Raw:
    def __init__(self, raw_root: Path):
        self.hf = Path(raw_root) / "hf"
        if not (self.hf / "anno_preview").is_dir():
            raise OracleError(f"raw OakInk2 not found: {self.hf}/anno_preview")
        self.pkl = collections.defaultdict(list)
        for f in (self.hf / "anno_preview").glob("*.pkl"):
            self.pkl[re.search(r"seq__([0-9a-f]+)__", f.name).group(1)[:12]].append(f)
        dup = [k for k, v in self.pkl.items() if len(v) > 1]
        if dup:
            raise OracleError(f"hash12 not unique in anno_preview: {dup[:3]}")
        tree = json.loads((self.hf / "object_affordance/object_part_tree.json").read_text())
        self.parent = {ch: p for p, chs in tree.items() for ch in chs}

    def root(self, rid: str) -> str | None:
        """official instance root of a part (None = not a part)."""
        if rid not in self.parent:
            return None
        r, n = rid, 0
        while r in self.parent and n < 8:
            r, n = self.parent[r], n + 1
        return r

    def program(self, h: str) -> list[tuple[tuple, dict]]:
        f = self.pkl[h][0]
        d = json.loads((self.hf / "program/program_info" / f"{f.stem}.json").read_text())
        return [(ast.literal_eval(k), v) for k, v in d.items()]


_MANO: dict = {}


def mano(side: str, invert_left: bool = False):
    """smplx MANO as OakInk2 posed it — the left model is the statement's (mano_conventions, read by
    readers/raw/common.mano_layer); invert_left = the other reading, for hand_check only."""
    return C.mano_layer(_MANO["dir"], side, "oakink2", invert_left=invert_left, use_pca=False, flat_hand_mean=True)


def hand_verts(raw_mano: dict, frames: list[int], side: str) -> np.ndarray:
    import torch
    from scipy.spatial.transform import Rotation as R

    q = np.stack([raw_mano[k][f"{side}__pose_coeffs"].reshape(16, 4).numpy() for k in frames])
    aa = R.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(len(frames), 16, 3).astype(np.float32)
    be = np.stack([raw_mano[k][f"{side}__betas"].reshape(10).numpy() for k in frames]).astype(np.float32)
    tsl = np.stack([raw_mano[k][f"{side}__tsl"].reshape(3).numpy() for k in frames]).astype(np.float64)
    with torch.no_grad():
        out = mano(side)(
            global_orient=torch.from_numpy(aa[:, 0]),
            hand_pose=torch.from_numpy(aa[:, 1:].reshape(-1, 45)),
            betas=torch.from_numpy(be),
            transl=torch.zeros(len(frames), 3),
        )
    v = out.vertices.numpy().astype(np.float64) - out.joints[:, :1].numpy().astype(np.float64)
    return v + tsl[:, None]


def measure_stage(raw: Raw, d: dict, a0: int, b0: int) -> dict:
    """Per raw object over [a0, b0) at STEP: centroid (world), rotation, per-hand min vertex distance."""
    frames = [k for k in range(a0, b0, STEP) if k in d["raw_mano"]]
    hv = {s: hand_verts(d["raw_mano"], frames, s) for s in ("lh", "rh")} if frames else {}
    poses = {}
    for rid in d["obj_list"]:
        mp = raw.hf / "obj/object_repair/align_ds" / rid / "model.obj"
        ot = d["obj_transf"].get(rid) or {}
        have = np.array([k in ot for k in frames], bool)
        Ts = (
            np.stack([np.asarray(ot[k], float) if k in ot else np.full((4, 4), np.nan) for k in frames])
            if frames
            else np.zeros((0, 4, 4))
        )
        poses[rid] = (mp if mp.is_file() else None, Ts[:, :3, :3], Ts[:, :3, 3], have)
    return C.measure_objects(frames, hv, poses, STEP / MOCAP_HZ)


# ---- the measurement interface (one sequence = one anno_preview pkl; one measurement = one program stage)
def seq_key(clip: str) -> str:
    return CLIP_RE.fullmatch(clip).group("h")


def load_seq(raw: Raw, key: str) -> tuple:
    prog = raw.program(key)
    with open(raw.pkl[key][0], "rb") as fh:
        return prog, pickle.load(fh)


def window(raw: Raw, seq: tuple, clip: str, ti: dict | None):
    """-> (measurement key, w0, w1, active hands, declared ids, how)."""
    stage, a0, b0, w0, w1, act, decl, how = window_of(seq[0], clip, ti)
    return (stage, a0, b0), w0, w1, act, decl, how


def measure(raw: Raw, seq: tuple, key: tuple) -> dict:
    return measure_stage(raw, seq[1], key[1], key[2])


obj_key = rid_of


def frame_key(obj_id: str) -> str:
    """Identity of an anchor = rigid frame rid (the anchor oracle of dexengine.record.scene.render_only.tree_rows:
    whether same-named objects in the record and the tree share a frame).
    A merged asset (oakink2.fixed_merge.is_asset_id) = the body rid of its FIXED.json (a merged asset's frame = the body
    frame) · otherwise rid_of.
    Even when each hand tree's rigid-joint result differs (body only · merged asset · relative pose variants), the frame
    is the one body. Fails when it cannot be resolved."""
    from dexengine.data.oakink2 import fixed_merge as FM  # noqa: PLC0415

    return FM.body_rid(obj_id) if FM.is_asset_id(obj_id) else rid_of(obj_id)


def frame_inputs(obj_id: str) -> list[str]:
    """Files frame_key(obj_id) reads: a merged asset = its FIXED.json · otherwise none (the collect lane statement
    carries it as a node)."""
    from dexengine.data.oakink2 import fixed_merge as FM  # noqa: PLC0415

    return [str(FM.fixed_json(obj_id))] if FM.is_asset_id(obj_id) else []


def window_of(prog: list, clip: str, ti: dict | None):
    m = CLIP_RE.fullmatch(clip)
    stage = int(m.group("stage"))
    if stage >= len(prog):
        raise ValueError(f"stage {stage} >= {len(prog)} program stages")
    (lh_iv, rh_iv), v = prog[stage]
    ivs = [x for x in (lh_iv, rh_iv) if x]
    a0, b0 = min(x[0] for x in ivs), max(x[1] for x in ivs)
    w0, w1, how = a0, b0, "stage"
    if m.group("seg") or m.group("cut"):
        seg = (ti or {}).get("segment") or {}
        if seg.get("start") is None or seg.get("end") is None:
            raise ValueError("segment/cut clip without tree segment.start/end")
        w0 = a0 + int(round(seg["start"] * MOCAP_HZ / TREE_HZ))
        w1 = a0 + int(round(seg["end"] * MOCAP_HZ / TREE_HZ))
        how = "segment"
    act = {"lh": bool(lh_iv and lh_iv[0] < w1 and lh_iv[1] > w0), "rh": bool(rh_iv and rh_iv[0] < w1 and rh_iv[1] > w0)}
    decl = set((v.get("obj_list") or []) + (v.get("obj_list_lh") or []) + (v.get("obj_list_rh") or []))
    return stage, a0, b0, w0, w1, act, decl, f"{how}:{v.get('primitive')}"


def hand_check(n_seq: int, invert: bool = False) -> dict:
    """The left-model statement held against the delivered keypoints of whole-stage clips — the oracle's own MANO
    (`mano`) under both left readings; readers/raw/common.left_convention_check decides."""
    from scipy.spatial.transform import Rotation as R

    from dexengine.core import paths as P

    _MANO.setdefault("dir", str(P.mano_model_dir()))
    raw = Raw(P.layer_path("raw", "oakink2"))
    mano_root = P.layer_path("mano", "oakink2", emb="bimanual")
    by: dict = {}
    for c in sorted(p.name for p in mano_root.iterdir()):
        m = CLIP_RE.fullmatch(c)
        if m and not (m.group("seg") or m.group("cut")) and m.group("h") in raw.pkl:
            by.setdefault(m.group("h"), c)
    keys = sorted(by)
    clips = []
    for h in keys[:: max(1, len(keys) // n_seq)][:n_seq]:
        prog, d = load_seq(raw, h)
        _stage, a0, _b0, *_ = window_of(prog, by[h], None)
        z = np.load(mano_root / by[h] / "0" / "trajectory_keypoints.npz")
        mf = a0 + np.round(np.arange(len(z["timestamps"])) * MOCAP_HZ / TREE_HZ).astype(int)
        sel = np.flatnonzero([int(k) in d["raw_mano"] for k in mf])[::3]
        fr = [int(k) for k in mf[sel]]
        pts = {}
        for side, inv in (("rh", False), ("lh", False), ("lh", True)):
            q = np.stack([d["raw_mano"][k][f"{side}__pose_coeffs"].reshape(16, 4).numpy() for k in fr])
            aa = R.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(len(fr), 16, 3)
            be = np.stack([d["raw_mano"][k][f"{side}__betas"].reshape(10).numpy() for k in fr])
            tsl = np.stack([d["raw_mano"][k][f"{side}__tsl"].reshape(3).numpy() for k in fr])
            pts[(side, inv)] = C.hand_points(mano(side, inv), aa[:, 0], aa[:, 1:].reshape(-1, 45), be, tsl, True)
        ref = C.delivered_points(z, sel)
        clips.append((ref["rh"], ref["lh"], pts[("rh", False)], pts[("lh", False)], pts[("lh", True)]))
    return C.left_convention_check("oakink2", clips, invert=invert)


if __name__ == "__main__":
    # sink (entry_points-style); the command line (including --help) is parsed by readers/raw/common.hand_check_cli
    argparse.ArgumentParser(add_help=False, description=__doc__).parse_known_args()
    import sys

    sys.exit(C.hand_check_cli("oakink2", hand_check))
