"""Scene object oracle — DexCanvas raw loader (readers/raw/common owns the thresholds, the reduction and the classes).

Raw (<dexcanvas raw layer>, provenance <raw layer>/SOURCE.json): hf/mocap_ver0.1.parquet — one row per take, the
source of every clip (149 rows = 149 parents): trajectory_meta_data.mocap_raw_data_source {operator, object, gesture,
sequence} · data_fps (100) · total_frames · sequence_info.hand_joint.{position (T,3) m · rotation (T,3) axis-angle} =
the MANO root transform · sequence_info.mano_model_output.mesh_vertices (T, 778*3) m in the hand's local frame ·
object_info.pose (T,6) = x y z m + Euler angles rad · toolkit/mano_assets/objects/<object>.stl (mm — read at 1e-3).
One right hand, one object per take; no part tree (C.NoParts).
Clip -> window: `<op>__<object>_<gesture>_<seq>[_segNN][_tN]` = take `<op>__<object>__<gesture>__<seq>` (the release's
operator__object__gesture__sequence; any other name dies); a segment / cut = the tree's segment.start/end (50 Hz from
the take start: segment.fps 50 · frame_map offset 0 · frame_map.T_mano == (T-1)*50/100 + 1, else it dies)
-> round(x * 100/50). The take is measured whole once (sk = (0, T)); active hand = right.
Convention: the hand = R(rotation)·v + position (the release visualizer's transform, which the reader applies to the
joints); the object Euler sequence is STATED by the reader's owner (dexcanvas_frames.OBJECT_EULER) and only checked
here: `validate` measures every candidate (convs()) over all 149 takes — contact (carried_distances ·
readers/raw/hocap.carried_depths) and the rigid-grasp drift (grasp_drift) — and `check` dies unless the statement A
touches what it carries and keeps the grasped object stillest. Measured: A = intrinsic 'XYZ' carried median 1.0 mm ·
drift 0.4 deg / 0.4 mm; B no root rotation (carried 8.4 mm) · C inverse rotation (2.3 deg / 2.3 mm) · extrinsic
'xyz' (the reader's old reading) 2.3 deg · extrinsic 'zyx' 2.3 deg all die.
`python readers/raw/dexcanvas.py --validate [--statement xyz] [--table <json>]` (negative test: --statement xyz dies).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from dexengine.core import paths as P
from dexengine.data.readers.raw import common as C
from dexengine.data.readers.raw import scene_classify as CL

FPS, TREE_HZ = 100.0, 50.0
PARQUET = "hf/mocap_ver0.1.parquet"
MESHES = "toolkit/mano_assets/objects"
CLIP_RE = re.compile(r"(?P<op>[a-z]\d+)__(?P<obj>[a-z]+\d+)_(?P<g>\d+)_(?P<seq>\d+)(?P<seg>_seg\d+)?(?P<cut>_t\d+)?")
COLS = (
    "sequence_info.hand_joint.position",
    "sequence_info.hand_joint.rotation",
    "sequence_info.mano_model_output.mesh_vertices",
    "sequence_info.object_info.pose",
)
#: the object Euler rule is STATED by the reader's owner (dexengine/data/readers/dexcanvas_frames.py
#: OBJECT_EULER) — this oracle never restates it: it measures every candidate reading and checks the statement wins.
from dexengine.data.readers import dexcanvas_frames as F  # noqa: E402

EULERS = ("XYZ", "xyz", "ZYX", "zyx")  # candidate readings of object_info.pose[:, 3:] (scipy: upper = intrinsic)
_STATED: dict = {}  # --statement: a stand-in statement (negative test), else F.OBJECT_EULER


def stated() -> str:
    s = _STATED.get("euler") or F.OBJECT_EULER
    if s not in EULERS:
        raise ValueError(f"stated object Euler {s!r} is not a candidate {EULERS}")
    return s


def convs() -> dict:
    """name -> (hand root rotation on the local vertices: rot | none | inv, object Euler sequence). A = the statement
    (the reader's transform for the hand · OBJECT_EULER); B · C = the other hand readings; E<seq> = the other Eulers."""
    s = stated()
    return {"A": ("rot", s), "B": ("none", s), "C": ("inv", s), **{f"E{e}": ("rot", e) for e in EULERS if e != s}}


CONV = "A"
TAGV = "[DEXCANVAS_CONV]"
_MANO: dict = {}
_RAW: dict = {}  # per process (validate)
obj_key = str


class Raw(C.NoParts):
    def __init__(self, raw_root: Path):
        import pyarrow.parquet as pq

        self.p, self.meshes = Path(raw_root) / PARQUET, Path(raw_root) / MESHES
        if not self.p.is_file() or not self.meshes.is_dir():
            raise FileNotFoundError(f"DexCanvas raw missing: {self.p} · {self.meshes}")
        self.meta = pq.read_table(self.p, columns=["trajectory_meta_data"]).column(0).to_pylist()
        self.row: dict = {}
        for i, m in enumerate(self.meta):
            s = m["mocap_raw_data_source"]
            k = "__".join([s["operator"], s["object"], s["gesture"], s["sequence"]])
            if k in self.row:
                raise ValueError(f"take {k} appears twice in {self.p}")
            self.row[k] = i
        self._tab = None

    def columns(self):
        """The trajectory columns of every row — read once per process (the file is one row group)."""
        if self._tab is None:
            import pyarrow.parquet as pq

            self._tab = pq.read_table(self.p, columns=list(COLS))
        return self._tab


def seq_key(clip: str) -> str:
    m = CLIP_RE.fullmatch(clip)
    return "__".join([m.group("op"), m.group("obj"), m.group("g"), m.group("seq")])


@lru_cache(maxsize=16)
def _mesh_m(stl: str) -> str:
    """The release mesh (mm) as a metre .obj (CL.surface reads a path): named by its content hash in one temp folder and
    written by rename, so parallel workers never read a half-written file."""
    import hashlib
    import os

    import trimesh

    m = trimesh.load(stl, force="mesh")
    m.apply_scale(1e-3)
    data = trimesh.exchange.obj.export_obj(m)
    out = (
        Path(tempfile.gettempdir())
        / "scene_oracle_dexcanvas"
        / f"{Path(stl).stem}_{hashlib.sha256(data.encode()).hexdigest()[:16]}.obj"
    )
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.{os.getpid()}")
        tmp.write_text(data)
        os.replace(tmp, out)
    return str(out)


def load_seq(raw: Raw, key: str) -> dict:
    i, m = raw.row[key], raw.meta[raw.row[key]]
    T, fps = int(m["total_frames"]), float(m["data_fps"])
    if fps != FPS:
        raise ValueError(f"{key}: data_fps {fps} ≠ {FPS}")
    tab = raw.columns()
    pos, rot, verts, obj = (np.asarray(tab.column(c)[i].as_py(), float) for c in range(len(COLS)))
    verts = verts.reshape(len(verts), 778, 3)
    if not (len(pos) == len(rot) == len(verts) == len(obj) == T):
        raise ValueError(f"{key}: lengths {len(pos)}/{len(rot)}/{len(verts)}/{len(obj)} ≠ total_frames {T}")
    name = m["mocap_raw_data_source"]["object"]
    stl = raw.meshes / f"{name}.stl"
    if not stl.is_file():
        raise FileNotFoundError(f"{key}: no mesh {stl}")
    return {"T": T, "pos": pos, "rot": rot, "verts": verts, "obj": obj, "name": name, "mesh": _mesh_m(str(stl))}


def window(raw: Raw, seq: dict, clip: str, ti: dict | None):
    m = CLIP_RE.fullmatch(clip)
    T = seq["T"]
    w0, w1, how = 0, T, "take"
    if m.group("seg") or m.group("cut"):
        seg = (ti or {}).get("segment") or {}
        fm = seg.get("frame_map") or {}
        if seg.get("start") is None or seg.get("end") is None:
            raise ValueError("segment/cut clip without tree segment.start/end")
        if (
            float(seg.get("fps") or 0) != TREE_HZ
            or fm.get("offset") != 0
            or fm.get("T_mano") is None
            or abs(fm["T_mano"] - ((T - 1) * TREE_HZ / FPS + 1)) > 2
        ):
            raise ValueError(
                f"segment timebase fps {seg.get('fps')} · frame_map {fm} ≠ 50 Hz from the take start of {T} raw frames"
            )
        w0, w1, how = int(round(seg["start"] * FPS / TREE_HZ)), int(round(seg["end"] * FPS / TREE_HZ)), "segment"
    if not 0 <= w0 < w1 <= T:
        raise ValueError(f"window {w0}-{w1} outside the take's {T} frames")
    return (0, T), w0, w1, {"lh": False, "rh": True}, set(), how


def hand_verts(seq: dict, idx, conv: str) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    how = convs()[conv][0]
    v = seq["verts"][idx]
    if how == "none":
        return v + seq["pos"][idx][:, None]
    Rm = R.from_rotvec(seq["rot"][idx]).as_matrix()
    return np.einsum("tij,tkj->tki", Rm if how == "rot" else Rm.transpose(0, 2, 1), v) + seq["pos"][idx][:, None]


def measure(raw: Raw, seq: dict, key: tuple, conv: str | None = None, hv: dict | None = None) -> dict:
    from scipy.spatial.transform import Rotation as R

    conv = conv or CONV
    idx = list(range(max(0, key[0]), min(seq["T"], key[1])))
    hv = hv or {"rh": hand_verts(seq, idx, conv), "lh": np.full((len(idx), 778, 3), np.nan)}
    o = seq["obj"][idx]
    have = np.all(np.isfinite(o), axis=1)
    Rm = np.tile(np.eye(3), (len(idx), 1, 1))
    if have.any():
        Rm[have] = R.from_euler(convs()[conv][1], o[have, 3:]).as_matrix()
    poses = {seq["name"]: (seq["mesh"], Rm, np.where(have[:, None], o[:, :3], 0.0), have)}
    return C.measure_objects(idx, hv, poses, 1.0 / FPS)


LAG = 10  # frames (0.1 s) between the two carried samples a grasp-drift pair compares
VAL_MAX_MM = 3.0  # carried median (mm) the convention must keep — the hand touches what it carries
TIE_DEG, TIE_MM = 1.0, 1.0


def grasp_drift(seq: dict, st: dict, conv: str, t_s) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-grasp oracle: while the object is carried (C.VAL_V <= speed < 3 m/s) the hand holds it, so the object's
    pose in the hand's root frame barely changes — (rotation deg, translation m) of that relative pose between carried
    samples LAG frames apart. A wrong hand-root rotation or Euler sequence turns object rotation into drift."""
    from scipy.spatial.transform import Rotation as R

    how = convs()[conv][0]
    o = next(iter(st["objs"].values()))
    Rh = R.from_rotvec(seq["rot"]).as_matrix()
    Rh = np.tile(np.eye(3), (len(Rh), 1, 1)) if how == "none" else Rh if how == "rot" else Rh.transpose(0, 2, 1)
    rel_R = np.einsum("tji,tjk->tik", Rh, o["R"])
    rel_t = np.einsum("tji,tj->ti", Rh, o["t"] - seq["pos"])
    v = np.r_[0, np.linalg.norm(np.diff(o["c"], axis=0), axis=1) / np.maximum(np.diff(t_s), 1e-3)]
    car = (v >= C.VAL_V) & (v < 3.0) & o["have"]
    i = np.flatnonzero(car[:-LAG] & car[LAG:])
    if not len(i):
        return np.zeros(0), np.zeros(0)
    ang = CL.rot_angle_deg(np.eye(3), np.einsum("tji,tjk->tik", rel_R[i], rel_R[i + LAG]))
    return ang, np.linalg.norm(rel_t[i + LAG] - rel_t[i], axis=1)


def _validate_one(job):
    from dexengine.data.readers.raw import hocap as HC  # the penetration oracle's owner

    raw_root, key, convs = job
    raw = _RAW.get(raw_root) or _RAW.setdefault(raw_root, Raw(Path(raw_root)))
    seq = load_seq(raw, key)
    idx = list(range(seq["T"]))
    t_s = np.arange(seq["T"]) / FPS
    out = {}
    for c in convs:
        hv = {"rh": hand_verts(seq, idx, c), "lh": np.full((len(idx), 778, 3), np.nan)}
        st = measure(raw, seq, (0, seq["T"]), conv=c, hv=hv)
        ang, dt = grasp_drift(seq, st, c, t_s)
        out[c] = {
            "carried": list(C.carried_distances(st, t_s)),
            "depth": list(HC.carried_depths(st, hv, t_s)),
            "drift_deg": list(ang / 1e3),
            "drift_m": list(dt),
        }  # deg stored /1e3 so summarize's "mm" reads deg
    return out


def validate(raw: Raw, keys: list[str], jobs: int | None = None) -> dict:
    """Per convention over `keys` (whole takes, one right hand): carried distance · depth · grasp drift summaries
    (drift_deg's *_mm fields are degrees)."""
    jl = [(str(raw.p.parent.parent), k, tuple(convs())) for k in keys]
    res: dict = {c: {} for c in convs()}
    from multiprocessing import Pool  # noqa: PLC0415

    from dexengine.core import capacity  # noqa: PLC0415

    step = "dexengine.data.readers.raw.dexcanvas"  # capacity.NEEDS key (this module)
    with capacity.sized(step, len(jl), jobs) as cap, Pool(cap.workers, capacity.worker_init, cap.initargs) as pool:
        items = list(pool.imap(_validate_one, jl))  # input order (not finish order): sums independent of the pool
    for d in items:
        for c, m in d.items():
            for k, v in m.items():
                res[c].setdefault(k, []).extend(v)
    return {
        "sequences": {"right": len(keys)},
        "convs": {c: {k: C.summarize_distances(np.asarray(v)) for k, v in r.items()} for c, r in res.items()},
    }


def check(res: dict, conv: str) -> list[str]:
    """Dies unless `conv` touches what it carries (carried median <= VAL_MAX_MM) and, among conventions that do, has
    the smallest grasp drift — rotation within TIE_DEG and translation within TIE_MM of the best."""
    rows = res["convs"]
    touch = [c for c, r in rows.items() if r["carried"].get("n") and r["carried"]["med_mm"] <= VAL_MAX_MM]
    bad = []
    if conv not in touch:
        bad.append(f"{conv} carried median {rows[conv]['carried'].get('med_mm')} mm > {VAL_MAX_MM}")
    else:
        for k, tie, u in (("drift_deg", TIE_DEG, "deg"), ("drift_m", TIE_MM, "mm")):
            best = min(touch, key=lambda c: rows[c][k]["med_mm"])
            if rows[conv][k]["med_mm"] > rows[best][k]["med_mm"] + tie:
                bad.append(
                    f"{conv} {k} median {rows[conv][k]['med_mm']} {u} > {best} {rows[best][k]['med_mm']} + {tie}"
                )
    if bad:
        raise SystemExit(f"{TAGV} ✗ convention {conv} fails — " + " ; ".join(bad))
    return [f"{TAGV} convention {conv} holds ({res['sequences']['right']} takes)"]


def main() -> int:
    ap = argparse.ArgumentParser(description="DexCanvas hand/object convention oracle (module doc)")
    ap.add_argument("--validate", action="store_true", required=True)
    ap.add_argument(
        "--statement",
        choices=EULERS,
        help="check this object Euler as the statement instead of "
        "dexcanvas_frames.OBJECT_EULER (negative test: xyz must die)",
    )
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--out", help="write the table (json)")
    ap.add_argument("--table", help="check a table --out wrote instead of measuring again")
    a = ap.parse_args()
    if a.statement:
        _STATED["euler"] = a.statement
    print(
        f"{TAGV} statement object Euler {stated()!r} "
        f"({'--statement' if a.statement else 'dexcanvas_frames.OBJECT_EULER'})",
        flush=True,
    )
    if a.table:
        for ln in check(json.loads(Path(a.table).read_text()), CONV):
            print(ln, flush=True)
        return 0
    raw = Raw(P.layer_path("raw", "dexcanvas"))
    res = validate(raw, sorted(raw.row), a.jobs)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    for ln in check(res, CONV):
        print(ln, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
