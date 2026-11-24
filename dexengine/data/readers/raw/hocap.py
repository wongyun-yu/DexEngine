"""Scene object oracle — HO-Cap raw loader (readers/raw/common owns the thresholds, the reduction and the classes).

Raw (<hocap raw layer>/release — the release's three label tars, provenance <raw layer>/SOURCE.json):
sequence_labels/<seq>.meta.yaml (mano_sides · object_ids · subject_id · num_frames) · <seq>.poses_m.npy (2,T,51) rows
[right, left], per frame 3 global orient (axis-angle) · 45 PCA coefficients · 3 additive transl, a frame of all -1 =
that hand absent · <seq>.poses_o.npy (K,T,7) qx qy qz qw x y z in object_ids order · calibration/mano.<subject>.yaml
betas (one per subject, both hands) · models/<obj>.cleaned_mesh_10000.obj (metres). Hands and objects share one world
(tag_1). 30 fps.
Hands: a row is absent on every frame exactly when meta.mano_sides omits its side — a listed side with an -1 frame, an
unlisted side with a pose, or a shape that disagrees with meta dies (NO_RAW). Active hands = meta.mano_sides
(no stage program — HOT3D's rule; the clip row reports tree_idle beside it); no part tree (C.NoParts).
Clip -> window: `<seq>__<n>[_segNN][_tN]` = the take; a segment / cut = the tree's segment.start/end (50 Hz from the
take start: segment.fps 50 · frame_map offset 0 · frame_map.T_mano == (T-1)*5/3 + 1, else it dies) -> round(x * 30/50).
The take is measured whole once (sk = (0, T)); each clip reduces its window of it.
MANO convention = CONV, held by `validate` + `check` over all 64 takes per hand group (readers/raw/common.
carried_distances + carried_depths here — the rule is in check's doc). Measured: 3,820 carried samples (right 35 · left
8 · both 21 takes): A carried median 0.4 / p90 0.6 mm · depth median 7.5 mm; controls B p90 23.5 · C p90 25.3 · D depth
12.1 mm (right p90 3.5) all die (depth = the triangle ruler's, collider_shape.Triangles). Contact cannot separate the
left-hand shape (E ties A); the release's own labels do: hand_joints_3d of subject_1's first take (left-only, 8 cameras
× 676 frames — its subject tar, not kept in the raw layer) under ONE rigid world->camera transform per camera fit A to
0.000 mm (16 joints, max) and E to 2.2 mm median / 4.9 mm max — HO-Cap fitted the left hand WITHOUT the shapedirs fix
(as the intake's FK, dexengine/data/readers/hocap.py, which A equals to 0.000 mm on both hands). B is a constant shift
there (invisible to the joint oracle) and dies on contact.
`python readers/raw/hocap.py --validate [--conv X] [--table <json>]` (negative test: --conv B / C / D die).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from dexengine.data.readers.raw import common as C

FPS, TREE_HZ = 30.0, 50.0
ROWS = (("rh", "right", 0), ("lh", "left", 1))  # poses_m row of each side
CLIP_RE = re.compile(r"(?P<seq>subject_\d+-\d{8}_\d{6})__\d+(?P<seg>_seg\d+)?(?P<cut>_t\d+)?")
#: name -> (pose[3 : 48] = PCA coefficients expanded through hands_components, hands_mean added (smplx flat_hand_mean
#: False), transl = wrist position (re-centre on joint 0)); the left model = mano_conventions. A = the release's
#: (module doc); B–D: its
#: controls, one axis each. smplx cannot expand 45 PCA coefficients itself — MANO(num_pca_comps=45) silently sets
#: use_pca=False and reads them as axis-angle (= control D) — so the expansion is done here (hand_verts).
CONVS = {
    "A": (True, True, False),
    "B": (True, True, True),  # transl = wrist position
    "C": (True, False, False),  # no hands_mean
    "D": (False, True, False),
}  # [3 : 48] read as axis-angle (smplx's own num_pca_comps=45 reading)
CONV = "A"
VAL_MAX_MM = 3.0  # carried p90 (mm) the convention must keep in every hand group
DEPTH_TIE_MM = 1.0  # carried-depth medians closer than this (mm) tie — the oracle claims no finer resolution
_MANO: dict = {}
obj_key = str


class Raw(C.NoParts):
    def __init__(self, raw_root: Path):
        self.r = Path(raw_root) / "release"
        for d in ("sequence_labels", "calibration", "models"):
            if not (self.r / d).is_dir():
                raise FileNotFoundError(f"HO-Cap release missing: {self.r / d}")
        self.seqs = sorted(p.name[: -len(".meta.yaml")] for p in (self.r / "sequence_labels").glob("*.meta.yaml"))


def seq_key(clip: str) -> str:
    return CLIP_RE.fullmatch(clip).group("seq")


def load_seq(raw: Raw, key: str) -> dict:
    import yaml

    sl = raw.r / "sequence_labels"
    meta = yaml.safe_load((sl / f"{key}.meta.yaml").read_text())
    pm = np.load(sl / f"{key}.poses_m.npy").astype(np.float64)
    po = np.load(sl / f"{key}.poses_o.npy").astype(np.float64)
    T, ids, sides = int(meta["num_frames"]), list(meta["object_ids"]), list(meta["mano_sides"])
    if pm.shape != (2, T, 51) or po.shape != (len(ids), T, 7) or len(set(ids)) != len(ids):
        raise ValueError(f"{key}: poses_m {pm.shape} · poses_o {po.shape} ≠ meta (T {T} · objects {ids})")
    if set(sides) - {"right", "left"} or not sides:
        raise ValueError(f"{key}: mano_sides {sides}")
    present = {}
    for s, side, row in ROWS:
        gone = np.all(pm[row] == -1, axis=1)
        part = np.any(pm[row] == -1, axis=1) & ~gone
        if part.any() or (side in sides) != (not gone.all()) or (side in sides and gone.any()):
            raise ValueError(
                f"{key}: {side} row (absent {int(gone.sum())}/{T} · partial -1 {int(part.sum())}) "
                f"disagrees with mano_sides {sides}"
            )
        present[s] = side in sides
    betas = np.asarray(
        yaml.safe_load((raw.r / "calibration" / f"mano.{meta['subject_id']}.yaml").read_text())["betas"], float
    ).reshape(10)
    meshes = {}
    for oid in ids:
        mp = raw.r / "models" / f"{oid}.cleaned_mesh_10000.obj"
        if not mp.is_file():
            raise FileNotFoundError(f"{key}: no mesh {mp}")
        meshes[oid] = mp
    return {"T": T, "pm": pm, "po": po, "ids": ids, "meshes": meshes, "present": present, "betas": betas}


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
    return (0, T), w0, w1, dict(seq["present"]), set(), how


def hand_verts(seq: dict, idx, s: str, conv: str) -> np.ndarray:
    import torch

    row = {a: r for a, _, r in ROWS}[s]
    V = np.full((len(idx), 778, 3), np.nan)
    if not seq["present"][s]:
        return V
    pca, mean, recentre = CONVS[conv]
    p = seq["pm"][row][idx]
    m = C.mano_layer(_MANO["dir"], s, "hocap", use_pca=False, num_pca_comps=45, flat_hand_mean=not mean)
    hp = p[:, 3:48] @ m.np_hand_components if pca else p[:, 3:48]
    f = {"dtype": torch.float32}
    with torch.no_grad():
        o = m(
            global_orient=torch.tensor(p[:, :3], **f),
            hand_pose=torch.tensor(hp, **f),
            betas=torch.tensor(np.tile(seq["betas"], (len(idx), 1)), **f),
            transl=torch.tensor(p[:, 48:51] * (0 if recentre else 1), **f),
        )
    V[:] = o.vertices.numpy().astype(float)
    if recentre:
        V = V - o.joints[:, :1].numpy() + p[:, None, 48:51]
    return V


def object_poses(seq: dict, idx) -> dict:
    """{obj id: (mesh, R (T,3,3), t (T,3), have (T,))} — readers/raw/common.measure_objects' input."""
    from scipy.spatial.transform import Rotation as R

    poses = {}
    for j, oid in enumerate(seq["ids"]):
        a = seq["po"][j, idx]
        have = np.all(np.isfinite(a), axis=1) & (np.abs(np.linalg.norm(a[:, :4], axis=1) - 1) < 1e-3)
        Rm = np.tile(np.eye(3), (len(idx), 1, 1))
        if have.any():
            Rm[have] = R.from_quat(a[have, :4]).as_matrix()  # scipy quat = x y z w, the release's order
        poses[oid] = (seq["meshes"][oid], Rm, np.where(have[:, None], a[:, 4:7], 0.0), have)
    return poses


def measure(raw: Raw, seq: dict, key: tuple, conv: str | None = None, hv: dict | None = None) -> dict:
    idx = list(range(max(0, key[0]), min(seq["T"], key[1])))
    hv = hv or {s: hand_verts(seq, idx, s, conv or CONV) for s, _, _ in ROWS}
    return C.measure_objects(idx, hv, object_poses(seq, idx), 1.0 / FPS)


@lru_cache(maxsize=128)
def _ruler(mesh_path: str):
    """The triangle ruler (collider_shape.Triangles, the one owner of inside and depth) of an object mesh; every
    release mesh is watertight (64/64), and one that is not dies here."""
    import trimesh

    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415 — owner of the triangle ruler

    m = trimesh.load(mesh_path, force="mesh")
    if not m.is_watertight:
        raise ValueError(f"{mesh_path}: not watertight — the release meshes are")
    return CS.Triangles(np.asarray(m.vertices, float), np.asarray(m.faces))


def carried_depths(st: dict, hv: dict, t_s, v_min: float = C.VAL_V) -> np.ndarray:
    """Penetration oracle beside carried_distances, on the same carried samples: the deepest vertex (m) of the nearer
    hand inside the object. carried_distances only asks that SOME vertex is near the surface — a hand shifted into the
    object passes it — so a convention must also not sink the hand into what it carries. No hand near = 0."""
    out = []
    for o in st["objs"].values():
        if not o["mesh"]:
            continue
        v = np.r_[0, np.linalg.norm(np.diff(o["c"], axis=0), axis=1) / np.maximum(np.diff(t_s), 1e-3)]
        tri = _ruler(o["mp"])
        for i in np.flatnonzero((v >= v_min) & (v < 3.0) & o["have"]):
            s = "lh" if o["dist"]["lh"][i] < o["dist"]["rh"][i] else "rh"
            if np.isinf(o["dist"][s][i]):
                out.append(0.0)
                continue
            d = tri.depths((hv[s][i] - o["t"][i]) @ o["R"][i])
            out.append(float(d.max()) if len(d) else 0.0)
    return np.asarray(out)


def group_of(seq: dict) -> str:
    p = seq["present"]
    return "both" if p["rh"] and p["lh"] else "right" if p["rh"] else "left"


def _validate_one(job):
    raw_root, mano_dir, key, convs = job
    _MANO.setdefault("dir", mano_dir)
    raw = Raw(Path(raw_root))
    seq = load_seq(raw, key)
    idx = list(range(seq["T"]))
    t_s = np.arange(seq["T"]) / FPS
    out = {}
    for c in convs:
        hv = {s: hand_verts(seq, idx, s, c) for s, _, _ in ROWS}
        st = measure(raw, seq, (0, seq["T"]), hv=hv)
        out[c] = {"carried": list(C.carried_distances(st, t_s)), "depth": list(carried_depths(st, hv, t_s))}
    return group_of(seq), out


def validate(raw: Raw, keys: list[str], jobs: int | None = None) -> dict:
    """Per convention × hand group (all · right · left · both) over `keys` (whole takes): carried_distances and
    carried_depths summaries."""
    jl = [(str(raw.r.parent), _MANO["dir"], k, tuple(CONVS)) for k in keys]
    res: dict = {c: {} for c in CONVS}
    nseq: dict = {}
    from multiprocessing import Pool  # noqa: PLC0415

    from dexengine.core import capacity  # noqa: PLC0415

    step = "dexengine.data.readers.raw.hocap"  # capacity.NEEDS key (this module)
    with capacity.sized(step, len(jl), jobs) as cap, Pool(cap.workers, capacity.worker_init, cap.initargs) as pool:
        items = list(pool.imap(_validate_one, jl))  # input order (not finish order): sums independent of the pool
    for g, d in items:
        nseq[g] = nseq.get(g, 0) + 1
        for c, m in d.items():
            for gg in ("all", g):
                for k, v in m.items():
                    res[c].setdefault(gg, {}).setdefault(k, []).extend(v)
    out = {
        c: {g: {k: C.summarize_distances(np.asarray(v)) for k, v in m.items()} for g, m in sorted(r.items())}
        for c, r in res.items()
    }
    return {"sequences": dict(sorted(nseq.items())), "convs": out}


def check(res: dict, conv: str) -> list[str]:
    """Dies (SystemExit) unless, in every hand group, `conv` (1) touches what it carries — carried p90 <= VAL_MAX_MM —
    and (2) sinks no deeper into it than any convention that also passes (1): carried-depth median within
    DEPTH_TIE_MM of the smallest. (2) is needed because (1) alone passes a hand shifted into the object (control B:
    median 0.6 mm), and (1) is needed because (2) alone prefers a hand that floats off it (control C: depth 0)."""
    bad = []
    for g in sorted({g for r in res["convs"].values() for g in r}):
        rows = {c: r[g] for c, r in res["convs"].items() if r.get(g, {}).get("carried", {}).get("n")}
        if conv not in rows:
            bad.append(f"{g}: {conv} has no carried sample")
            continue
        touch = {c: r["depth"]["med_mm"] for c, r in rows.items() if r["carried"]["p90_mm"] <= VAL_MAX_MM}
        if conv not in touch:
            bad.append(f"{g}: {conv} carried p90 {rows[conv]['carried']['p90_mm']} mm > {VAL_MAX_MM}")
            continue
        best = min(touch, key=touch.get)
        if touch[conv] > touch[best] + DEPTH_TIE_MM:
            bad.append(f"{g}: {conv} depth median {touch[conv]} mm > {best} {touch[best]} + {DEPTH_TIE_MM}")
    if bad:
        raise SystemExit(f"[HOCAP_MANO] ✗ convention {conv} fails — " + " ; ".join(bad))
    return [f"[HOCAP_MANO] convention {conv} holds in every group ({', '.join(sorted(res['sequences']))})"]


def main() -> int:
    from dexengine.core import paths as P

    ap = argparse.ArgumentParser(description="HO-Cap MANO convention oracle (module doc)")
    ap.add_argument("--validate", action="store_true", required=True)
    ap.add_argument("--conv", default=CONV, choices=sorted(CONVS), help="the convention to hold (default CONV)")
    ap.add_argument("--per-group", type=int, default=0, help="first N takes of each hand group (0 = every take)")
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--out", help="write the table (json)")
    ap.add_argument("--table", help="check a table --out wrote instead of measuring again")
    a = ap.parse_args()
    if a.table:
        for ln in check(json.loads(Path(a.table).read_text()), a.conv):
            print(ln, flush=True)
        return 0
    _MANO.setdefault("dir", str(P.mano_model_dir()))
    raw = Raw(P.layer_path("raw", "hocap"))
    keys = raw.seqs
    if a.per_group:
        by: dict = {}
        for k in keys:
            by.setdefault(group_of(load_seq(raw, k)), []).append(k)
        keys = [k for g in sorted(by) for k in by[g][: a.per_group]]
    res = validate(raw, keys, a.jobs)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    for ln in check(res, a.conv):
        print(ln, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
