#!/usr/bin/env python3
"""HOT3D desk-plane levelling table — the only writer of ``configs/datasets/hot3d_up_level.tsv``.

Rule: a recording of a levelled device (``source_frames.LEVEL_DEVICES``) takes the normal of its desk
plane as its up axis. The plane is fitted in the device +z-up world (``source_frames.world_up`` — never the levelled
world, so a rerun does not read its own output) to the rests on the recording's desk level:
  rests      = every rest run of every object (``work_surface.rest_runs``, the desk rule's rest), its floor (lowest
               vertex) and centre (vertex mean) at the run's middle sample, weight sqrt(max(duration s, 0.5));
  desk level = rests within DESK_BAND_M of the recording's table height (``hot3d.table_z_world``);
  fit        = weighted least squares z = a·x + b·y + c, refit FIT_ITERS times on the rests within INLIER_M of
               the plane;
  accepted   = ≥ MIN_RESTS inlier rests of ≥ MIN_OBJECTS objects whose XY spread (smallest singular value of the
               centred centres / sqrt(n)) is ≥ MIN_SPREAD_M. Otherwise the recording is not written (it keeps the device
               axis — no evidence, no rotation).
Usage: $RETARGET_PY dexengine/data/support/hot3d_level.py [--out configs/datasets/hot3d_up_level.tsv] [--limit N]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from dexengine.data.readers import source_frames as SF
from dexengine.data.readers.hot3d_gt import has_ground_truth
from dexengine.data.support import hot3d as H
from dexengine.data.support import work_surface as WS

DESK_BAND_M = 0.03
INLIER_M = 0.01
FIT_ITERS = 3
MIN_RESTS, MIN_OBJECTS, MIN_SPREAD_M = 6, 3, 0.05
COLUMNS = (
    "recording",
    "headset",
    "a",
    "b",
    "tilt_deg",
    "n_rests",
    "n_objects",
    "spread_m",
    "rms_flat_cm",
    "rms_plane_cm",
)


def rest_rows(objs: dict, mesh) -> list[dict]:
    """[{uid, floor, cx, cy, w}] — one per rest run (middle sample)."""
    out = []
    for u, a in objs.items():
        pts = mesh(u)
        for i, j in WS.rest_runs(a, H.STEP_HZ):
            V = WS.verts_at(a, pts, (i + j) // 2)
            dur = float((a[j, 0] - a[i, 0]) / 1e9)
            out.append(
                {
                    "uid": u,
                    "floor": float(V[:, 2].min()),
                    "cx": float(V[:, 0].mean()),
                    "cy": float(V[:, 1].mean()),
                    "w": float(np.sqrt(max(dur, 0.5))),
                }
            )
    return out


def fit_plane(rows: list[dict], tz: float) -> dict | None:
    """Desk plane of one recording from its rest rows and table height (rule above) → {a, b, …} or None."""
    rows = [r for r in rows if abs(r["floor"] - tz) <= DESK_BAND_M]
    if len(rows) < MIN_RESTS or len({r["uid"] for r in rows}) < MIN_OBJECTS:
        return None
    X = np.array([[r["cx"], r["cy"], 1.0] for r in rows])
    z = np.array([r["floor"] for r in rows])
    w = np.array([r["w"] for r in rows])
    keep = np.ones(len(z), bool)
    for _ in range(FIT_ITERS):
        p, *_ = np.linalg.lstsq(X[keep] * w[keep, None], z[keep] * w[keep], rcond=None)
        keep = np.abs(z - X @ p) <= INLIER_M
    xy = X[keep, :2] - X[keep, :2].mean(0)
    spread = float(np.linalg.svd(xy, compute_uv=False)[-1] / np.sqrt(max(int(keep.sum()), 1)))
    n_obj = len({rows[i]["uid"] for i in np.nonzero(keep)[0]})
    if keep.sum() < MIN_RESTS or n_obj < MIN_OBJECTS or spread < MIN_SPREAD_M:
        return None
    rp, rf = (z - X @ p)[keep], z[keep] - z[keep].mean()
    return {
        "a": float(p[0]),
        "b": float(p[1]),
        "tilt_deg": SF.plane_tilt_deg(p[0], p[1]),
        "n_rests": int(keep.sum()),
        "n_objects": n_obj,
        "spread_m": spread,
        "rms_flat_cm": float(np.sqrt((rf**2).mean()) * 100),
        "rms_plane_cm": float(np.sqrt((rp**2).mean()) * 100),
    }


def device_objects(seq: str) -> tuple[dict, str]:
    """Object samples of one recording in the device +z-up world (not levelled) and its headset."""
    headset = json.loads((H.RAW / seq / "metadata.json").read_text()).get("headset")
    return SF.to_z_up(SF.object_samples(H.RAW / seq), SF.world_up(headset)[1]), headset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SF.LEVEL_TABLE))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    seqs = sorted(p.name for p in H.RAW.iterdir() if (p / "metadata.json").is_file())
    rows, n_dev, n_fit = [], 0, 0
    for seq in seqs[: a.limit or None]:
        headset = json.loads((H.RAW / seq / "metadata.json").read_text()).get("headset")
        if headset not in SF.LEVEL_DEVICES or not has_ground_truth(H.RAW / seq):
            continue
        n_dev += 1
        objs, _ = device_objects(seq)
        tz, _diag = H.table_z_world(objs)
        fit = None if tz is None else fit_plane(rest_rows(objs, H.mesh_pts), tz)
        if fit is None:
            continue
        n_fit += 1
        rows.append({"recording": seq, "headset": headset} | fit)
    with open(a.out, "w") as f:
        f.write(
            f"# written by dexengine/data/support/hot3d_level.py (rule in its docstring) · devices {SF.LEVEL_DEVICES}"
            f" · {n_fit} of {n_dev} recordings fitted\n"
        )
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.9g}" if isinstance(r[c], float) else str(r[c]) for c in COLUMNS) + "\n")
    print(f"[HOT3D_LEVEL_TABLE] devices={SF.LEVEL_DEVICES} recordings={n_dev} fitted={n_fit} → {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
