"""Sidecar fingerprint check: read a sidecar's stored fingerprint, compare it with the expected one (pose
tolerance, difference detail) and fail with the regeneration hint (re-exported by contact.fingerprint)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FP_SCHEMA = "sidecar-fp-1"


#: Object pose comparison tolerance (m · quaternion components · rad). Cross-hand noise within a generation ≈2e-16,
#: generation differences ≥1e-2.
POSE_ATOL = 1e-6


LAYER_ALL, LAYER_BASE = "contact_all", "contact_base"


class SidecarFingerprintError(RuntimeError):
    """The sidecar was built from inputs other than the reading base (stale) or its inputs are unknown
    (unknown/missing); consumers fail."""


def read(npz_path: Path) -> dict | None:
    """Stored fingerprint. None without fingerprint fields (= unknown, a version before fingerprints)."""
    with np.load(npz_path, allow_pickle=False) as z:
        if "fp_schema" not in z.files:
            return None
        return {
            "schema": str(z["fp_schema"]),
            "exact": json.loads(str(z["fp_exact"])),
            "pose": z["fp_pose"],
            "input_id": str(z["fp_input_id"]),
            "prov": {
                "built_from": str(z["fp_built_from"]),
                "base_time": str(z["fp_base_time"]),
                "base_adapter": str(z["fp_base_adapter"]),
            },
        }


def pose_close(a: np.ndarray, b: np.ndarray) -> tuple[bool, float, int]:
    """(equal, max position difference m, mismatched frame count). Quaternions are sign-free (q ≡ −q); NaN equals only
    NaN."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return False, float("inf"), -1
    if a.size == 0:
        return True, 0.0, 0

    def close(x, y):
        return np.isclose(x, y, rtol=0.0, atol=POSE_ATOL, equal_nan=True)

    pos_ok = close(a[..., :3], b[..., :3]).all(-1)
    q_ok = close(a[..., 3:7], b[..., 3:7]).all(-1) | close(a[..., 3:7], -b[..., 3:7]).all(-1)
    h_ok = close(a[..., 7], b[..., 7])
    bad = ~(pos_ok & q_ok & h_ok).all(-1)  # (T,)
    d = np.abs(a[..., :3] - b[..., :3])
    dpos = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    return (not bad.any()), dpos, int(bad.sum())


def _diff_detail(se: dict, ee: dict, k: str) -> str:
    if k == "mano":
        a, b = se.get(k) or {}, ee.get(k) or {}
        return "mano:" + ",".join(sk for sk in sorted(set(a) | set(b)) if a.get(sk) != b.get(sk))
    if k in ("code", "T"):
        return f"{k}:{se.get(k)}→{ee.get(k)}"
    if k == "objects":
        sn = [(o["name"], o["obj_id"]) for o in se.get(k) or []]
        en = [(o["name"], o["obj_id"]) for o in ee.get(k) or []]
        return "objects:" + (f"{sn}→{en}" if sn != en else "surface/articulated")
    return k


def compare(stored: dict | None, exp: dict) -> tuple[str, str]:
    """(verdict, reason). verdict ∈ fresh · stale · unknown. stored is a read() result or another base's expected()."""
    if stored is None:
        return "unknown", "no fingerprint (version before fingerprints)"
    if stored.get("schema") != FP_SCHEMA:
        return "unknown", f"fingerprint version {stored.get('schema')} ≠ {FP_SCHEMA}"
    se, ee = stored["exact"], exp["exact"]
    diff = [k for k in sorted(set(se) | set(ee)) if se.get(k) != ee.get(k)]
    if diff:
        return "stale", " · ".join(_diff_detail(se, ee, k) for k in diff)
    ok, dpos, nbad = pose_close(stored["pose"], exp["pose"])
    if not ok:
        return (
            "stale",
            f"object pose generation differs (max position difference {dpos:.4g} m · mismatched frames {nbad}/"
            f"{len(exp['pose'])})",
        )
    return "fresh", ""


def regen_hint(ds: str, layer: str) -> str:
    return (
        f"regenerate: $(paths.py --what retarget-py) dexengine/data/contact/sidecar.py --dataset {ds} --hand "
        "<this hand>" + (" --all-objects" if layer == LAYER_ALL else "")
    )


def check(npz_path: Path, exp: dict, *, ds: str) -> dict:
    """Consumer entry: SidecarFingerprintError unless fresh. Returns the stored fingerprint (read)."""
    npz_path = Path(npz_path)
    layer = exp["exact"]["layer"]
    if not npz_path.is_file():
        raise SidecarFingerprintError(f"[SIDECAR_FP] missing {npz_path.name} — {regen_hint(ds, layer)}")
    st = read(npz_path)
    v, why = compare(st, exp)
    if v != "fresh":
        src = f" · built by hand {st['prov']['built_from']}@{st['prov']['base_time']}" if st else ""
        mine = f"{exp['prov']['built_from']}@{exp['prov']['base_time']}"
        raise SidecarFingerprintError(
            f"[SIDECAR_FP] {v} {npz_path.name}: {why}{src} · reading base {mine} — {regen_hint(ds, layer)}"
        )
    return st
