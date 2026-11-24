"""Base layer: hand-direction statistics, the scene yaw, supplement pose series and the MANO/kinematic frame offset
(read by base_layer.clip.adapt_clip and the base_adapter entry)."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from dexengine.data import align as RD
from dexengine.data import raw_sample as RS
from dexengine.data.base_layer.common import ROOT, _dest, dl
from dexengine.data.base_layer.geom import _circ_mean_deg
from dexengine.data.base_layer.supplements import _supp_hand_rid, _supplement_files
from dexengine.data.handoff.checks import MAX_T_DIFF


def _mano_input(ds: str) -> tuple[str, int]:
    """(cheap signature of the dataset's MANO layer — per npz (emb/relative path, size, mtime_ns), the `_input_stat`
    form — and its file count). A reader re-run replaces the files → a new signature."""
    h = hashlib.sha256()
    n = 0
    for emb in dl.EMBODIMENTS:
        d = _dest("mano", ds, emb=emb)
        for f in sorted(d.glob("*/0/trajectory_keypoints.npz")) if d.is_dir() else []:
            st = f.stat()
            n += 1
            h.update(f"{emb}/{f.relative_to(d).as_posix()}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16], n


def hand_dir_stats(ds: str, force: bool = False) -> dict:
    """Circular mean of the MANO wrist→fingertip (mean of 5) xy direction angle — over every clip
    of the dataset. Cached in INTAKE/hand_dir.json with the MANO layer signature it was measured on (`mano_input`):
    the cache is used only while the layer is unchanged — a changed (or unsigned, older) cache is measured again and
    the change is printed ([HAND_DIR]), so a re-made MANO layer never builds trees on the old yaw. No MANO layer
    (taco: V1, cache only) → the cache is the reference value, said so ([HAND_DIR])."""
    idir = _dest("intake", ds)
    cache = idir / "hand_dir.json"
    sig, n_files = _mano_input(ds)
    old = json.loads(cache.read_text()) if cache.exists() else None
    if old is not None and not force:
        if n_files == 0:
            print(
                f"[HAND_DIR] {ds}: no MANO layer — using the cached reference {old['mean_deg']}° (no input to "
                "measure again)",
                flush=True,
            )
            return old
        if old.get("mano_input") == sig:
            return old
    angs = []
    n = 0
    for emb in dl.EMBODIMENTS:
        d = _dest("mano", ds, emb=emb)
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*/0/trajectory_keypoints.npz")):
            z = np.load(f)
            n += 1
            for s in ("right", "left"):
                if f"qpos_finger_{s}" not in z.files:
                    continue
                tips = z[f"qpos_finger_{s}"][:, :, :3]
                wr = z[f"qpos_wrist_{s}"][:, :3]
                if not np.isfinite(tips).all() or np.abs(tips).max() == 0:
                    continue
                v = (tips.mean(1) - wr)[:, :2]
                v = v[np.linalg.norm(v, axis=1) > 1e-3]
                if len(v):
                    angs.append(_circ_mean_deg(np.degrees(np.arctan2(v[:, 1], v[:, 0])))[0])
    if not angs:
        raise RuntimeError(f"[{ds}] no clips in the MANO layer to measure the hand direction")
    mean, spread = _circ_mean_deg(np.array(angs))
    rec = {
        "dataset": ds,
        "mean_deg": round(mean, 2),
        "spread_deg": round(spread, 2),
        "n_clips": n,
        "n_hands": len(angs),
        "definition": "MANO qpos_wrist→mean(qpos_finger tips) xy angle, circular mean of the per-clip circular means",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mano_input": sig,
    }
    if old is not None:
        why = (
            "forced"
            if force
            else (
                "old cache without a fingerprint"
                if "mano_input" not in old
                else "the MANO layer changed after the cache"
            )
        )
        print(
            f"[HAND_DIR] {ds}: {why} → measured again {old['mean_deg']}° → {rec['mean_deg']}° "
            f"(Δ {round(rec['mean_deg'] - old['mean_deg'], 2)}°)",
            flush=True,
        )
    idir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


PER_CLIP_YAW = (
    "hot3d",
    # D5-b: an egocentric world (each recording's headset initial pose = origin · yaw), so a dataset-mean yaw is
    # meaningless
    # (hand-direction spread ±88°) → per-clip yaw. Marker [YAW_CLIP]
)


def ref_dir_stat(ds: str) -> dict:
    """The reference hand direction this dataset is aligned to — a fixed value, never the freshly measured TACO mean
    (owner: align.for_dataset — the dataset generation's declared value; without a declaration the constant, except for
    the datasets built under another value (align.BUILT_OFF_REF), which fail).
    Same shape as a hand_dir_stats record (`mean_deg`) so the callers read one key. Marker [REF_DIR]."""
    ref, src = RD.for_dataset(ds, _dest("intake", ds))
    print(f"{RD.TAG} ds={ds} reference direction {RD.fmt(ref)}° ({src})", flush=True)
    return {"dataset": "taco", "mean_deg": ref, "source": src}


def ref_dir_oracle() -> bool:
    """Measurement ↔ statement: the TACO mean as measured now against the fixed reference. Says so, changes nothing."""
    try:
        m = hand_dir_stats("taco")
    except RuntimeError as e:
        print(f"{RD.TAG} check skipped — cannot measure the TACO hand direction: {e}", flush=True)
        return True
    ok, line = RD.oracle(m["mean_deg"])
    print(line, flush=True)
    return ok


def yaw_for(ds: str) -> tuple[float, dict, dict]:
    t = ref_dir_stat(ds)
    d = hand_dir_stats(ds)
    if ds == "taco":
        return 0.0, d, t
    psi = (t["mean_deg"] - d["mean_deg"] + 180.0) % 360.0 - 180.0
    return float(psi), d, t


def supplements_for(ds: str, tree_name: str, have: set, yaw: float | None = None) -> tuple[list[dict], list[str]]:
    """INTAKE/raw_objects/<tree>__<hand>__<rid>.npz (oakink2/raw_objects.py --export) — a 50 Hz pose
    sequence in the base frame (yaw included). `have` = the part ids (oakink2.parts.rid_of) of the objects the tree
    already carries: a supplement of the same id is a **copy** of a delivered object (a second free body with the same
    mesh and pose — it also made the D1-g part rule see three parts) and is not attached. Returns (attached, the copy
    files left out) — the caller prints [SUPP_DUP] and records them in rules[oakink2.parts.SUPP_KEY].
    A supplement sampled by another rule than raw_sample.RAW_SAMPLE (no `sample` field = the old one: rotation taken
    from the next raw frame, a pose up to 8 ms away from the delivered objects) is refused with the reason — ValueError.
    `yaw` = the scene rotation this tree is built with. Only with the repair switch on (raw_sample.YAW_FIX_ENV) a
    supplement whose recorded rotation (raw_sample.YAW_KEY) is missing or differs is refused the same way; off = as
    before (the field is not read)."""
    fix = RS.yaw_fix_on()
    if fix:
        if yaw is None:
            raise ValueError(f"{RS.YAW_MARK} {tree_name}: enabled but this tree did not receive the scene rotation")
        RS.yaw_mark(f"base_adapter ds={ds}", yaw)
    out, seen, dup = [], set(), []
    for f in _supplement_files(
        ds,
        tree_name,
        # hand ∈ {right, left, part, support}: part = sibling-part supplement (D1-e, contact-irrelevant) · support = a
        # raw
        # object under a main object (D42 — object_roles reads it through coords.support_of). If the same rid arrives by
        # two
        # paths, take it once
    ):
        z = np.load(f, allow_pickle=True)
        hand, rid = _supp_hand_rid(f, tree_name)
        if rid in have:
            dup.append(f.name)
            continue
        if rid in seen:
            continue
        seen.add(rid)
        why = RS.stamp_reason(z["sample"].item() if "sample" in z.files else None, f.name)
        if not why and fix:
            why = RS.yaw_reason(z[RS.YAW_KEY].item() if RS.YAW_KEY in z.files else None, yaw, f.name)
        if why:
            raise ValueError(why)
        out.append(
            {
                "hand": hand,
                "rid": rid,
                "pos": z["pos"],
                "quat": z["quat_wxyz"],
                "contact_frac": float(z["contact_frac"]),
                "mesh_src": str(z["mesh_src"]),
                "file": str(f.relative_to(ROOT)),
                "sample": RS.RAW_SAMPLE,
            }
        )
    return out, dup


# ── Segments, slices, support surface ──────────────────────────────────────
def mano_to_kin_offset(T_mano: int, T_kin: int) -> int:
    """Measured: the object poses agree to 0.0 mm at kin[i]==mano[i] → the filter drops
    the trailing Δ frames. Offset 0."""
    d = T_mano - T_kin
    if 0 <= d <= MAX_T_DIFF:
        return 0
    raise ValueError(f"MANO {T_mano} vs kinematic {T_kin} frame gap {d} — allowed 0..{MAX_T_DIFF}")
