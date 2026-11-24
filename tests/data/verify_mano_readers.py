"""Verify a reader against the delivered MANO layer (the oracle): every key, every frame, signed quaternions.

    python tests/data/verify_mano_readers.py --dataset arctic --raw <data/arctic/raw> --mano-dir <body_models/mano> \
        --delivered <data/arctic/mano/bimanual> --clips <clip> ... [--enumerate] [--json out.json]

Per clip: the reader's Phase-1 key list must equal the delivered key list minus the Phase-3 keys (contact_*,
appended by the contact detector) in order; every key must match shape and dtype; the max |diff| is
reported per key (quaternions compared with their sign — no hemisphere folding); task_info.json must be equal.
--enumerate: the reader's clip set against the delivered directory names (set difference both ways).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

PHASE3_KEYS = ("contact_left", "contact_pos_left", "contact_right", "contact_pos_right")
#: a clip passes when positions match to TOL_POS metres and quaternion components / other values to TOL_QUAT —
#: the float32 round-off of the raw inputs (positions ~1e-7 m; a wrist frame built from float32 joints ~1e-6)
TOL_POS = 1e-6
TOL_QUAT = 5e-6


def compare_clip(mod, a: argparse.Namespace, clip: str, handle) -> dict:
    """Run the reader on one clip and compare it with the delivered layer."""
    arrays, ti = mod.convert(a.raw, a.mano_dir, clip, handle, **getattr(mod, "verify_kwargs", lambda _a: {})(a))
    ddir = Path(a.delivered) / clip
    z = np.load(ddir / "0" / "trajectory_keypoints.npz")
    dti = json.loads((ddir / "task_info.json").read_text())
    dkeys = [k for k in z.files if k not in PHASE3_KEYS]
    row = {
        "clip": clip,
        "frames": int(len(z["timestamps"])),
        "keys_equal": list(arrays) == dkeys,
        "phase3_keys": [k for k in z.files if k in PHASE3_KEYS],
        "diff": {},
        "problems": [],
    }
    if not row["keys_equal"]:
        row["problems"].append(f"keys: reader {list(arrays)} vs delivered {dkeys}")
    for k in dkeys:
        if k not in arrays:
            continue
        x, y = arrays[k], z[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            row["problems"].append(f"{k}: reader {x.shape} {x.dtype} vs delivered {y.shape} {y.dtype}")
            continue
        e = np.abs(x.astype(np.float64) - y.astype(np.float64))
        if k.startswith("qpos_") and e.shape[-1] == 7:
            row["diff"][k + ".pos"] = float(e[..., :3].max()) if e.size else 0.0
            row["diff"][k + ".quat"] = float(e[..., 3:].max()) if e.size else 0.0
        else:
            row["diff"][k] = float(e.max()) if e.size else 0.0
    bad = {k: v for k, v in row["diff"].items() if v > (TOL_POS if k.endswith(".pos") else TOL_QUAT)}
    if bad:
        row["problems"].append(f"above tolerance (pos {TOL_POS} m, other {TOL_QUAT}): {bad}")
    # A second delivery batch (GRAB s7, 120 clips) rewrote <side>_object_convex_dir to
    # <mesh_dir>/convex after decomposition — a later phase, not the reader's; recorded, not failed
    row["phase2_convex"] = []
    for s in ("right", "left"):
        k = f"{s}_object_convex_dir"
        if ti.get(k) is not None and dti.get(k) == f"{ti[k]}/convex":
            row["phase2_convex"].append(s)
            dti[k] = ti[k]
    if ti != dti:
        diff = {k: (ti.get(k), dti.get(k)) for k in set(ti) | set(dti) if ti.get(k) != dti.get(k)}
        row["problems"].append(f"task_info differs: {diff}")
    if list(ti) != list(dti):
        row["problems"].append(f"task_info key order: {list(ti)} vs {list(dti)}")
    row["max_diff"] = max(row["diff"].values()) if row["diff"] else float("nan")
    row["ok"] = not row["problems"]
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["grab", "arctic", "h2o", "oakink2"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--mano-dir", required=True)
    ap.add_argument("--delivered", required=True, help="the delivered mano/bimanual layer")
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--enumerate", action="store_true", help="compare the reader's clip set with the delivery")
    ap.add_argument("--json", help="write the per-clip rows here")
    ap.add_argument("--subject-meshes", help="GRAB tools/subject_meshes directory (grab only)")
    ap.add_argument("--sides", choices=["table", "contact"], help="H2O object side rule (h2o only)")
    a = ap.parse_args(argv)
    mod = importlib.import_module(f"dexengine.data.readers.{a.dataset}")
    clips = mod.enumerate_clips(a.raw) if a.enumerate else getattr(mod, "candidate_clips", mod.enumerate_clips)(a.raw)
    out = {"dataset": a.dataset, "rows": []}
    rc = 0
    if a.enumerate:
        deliv = {p.name for p in Path(a.delivered).iterdir() if (p / "0" / "trajectory_keypoints.npz").is_file()}
        mine = set(clips)
        out["enumerate"] = {
            "reader": len(mine),
            "delivered": len(deliv),
            "reader_only": sorted(mine - deliv),
            "delivered_only": sorted(deliv - mine),
        }
        e = out["enumerate"]
        print(
            f"[ENUM] {a.dataset}: reader {e['reader']} · delivered {e['delivered']} · reader-only "
            f"{len(e['reader_only'])} {e['reader_only'][:4]} · delivered-only {len(e['delivered_only'])} "
            f"{e['delivered_only'][:4]}"
        )
        rc |= int(bool(e["reader_only"] or e["delivered_only"]))
    for c in a.clips:
        if c not in clips:
            print(f"[VERIFY] {a.dataset} {c}: not produced by the reader")
            rc = 1
            continue
        row = compare_clip(mod, a, c, clips[c])
        out["rows"].append(row)
        top = sorted(row["diff"].items(), key=lambda kv: -kv[1])[:3]
        print(
            f"[VERIFY] {a.dataset} {c}: {'OK' if row['ok'] else 'FAIL'} frames={row['frames']} "
            f"max={row['max_diff']:.3g} top={[(k, f'{v:.2g}') for k, v in top]} phase3={len(row['phase3_keys'])} "
            f"phase2_convex={row['phase2_convex']}" + ("" if row["ok"] else f" problems={row['problems']}")
        )
        rc |= int(not row["ok"])
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
