#!/usr/bin/env python3
"""h2o desk plane oracle: fits a per-take desk plane from the source depth (cam4), moves it into the **delivered capture
frame**, and measures tilt and height (D10-e'').

Why: the official h2o code defines no desk and the world frame differs per take (initial cam4 egocentric pose ≈ world,
    measured in `h2o_frame.py`). The camera looks down at the desk, so when world +z differs from the desk normal the
    "desk = horizontal plane" model is structurally wrong, and an object moving across the desk penetrates on one side
        and floats on the other.
Inputs: data/h2o/raw/sample/<subject>/<scene>/<take>/cam4/{depth/*.png, cam_pose/*.txt, cam_intrinsics.txt}   (streamed
extraction)
        INTAKE/h2o_frame.json (world → delivered frame G, `h2o_frame.py --fit`) · mano layer
            <clip>/0/trajectory_keypoints.npz (delivered, fixed world frame) · INTAKE/support_audit.json
Method: ① depth point cloud → cam_pose (cam→world) → world → RANSAC plane (normal within UP_TOL of the world direction
          of capture +z, inliers 1 cm), on frames 0 and 100; the cam_pose convention (forward/inverse) is decided from
              the
          data by the agreement of the two frames' planes (cam2world is right: 0.03–0.5° vs inverse 0.8–16°).
        ② move the plane into the capture frame with G: tilt (normal ↔ +z) and the downhill xy direction, plane height
            at
          the object's rest xy vs INTAKE desk_z, signed distance of the resting object bottom to the plane, lowest hand
          signed distance, and the span of plane z over the xy range the hands covered (= error span of a horizontal
              desk model).
Output: INTAKE/h2o_table_plane.json (the planes the pipeline reads) and the per-take report next to it,
        INTAKE/h2o_table_plane_report.{tsv,json} (REPORT_STEM)
Usage: $RETARGET_PY dexengine/data/support/h2o.py [--subject subject1] [--verbose]     (capped.sh 8G recommended)
The quality_why texts and the decision string are stored in the INTAKE plane file and stay as written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dexengine.core.paths import (
    main_root as _data_root,  # noqa: E402 — owner of the data root (env DEXENGINE_MAIN_ROOT)
)
from dexengine.data.readers import h2o_frame as HF  # (raw IO · G)
from dexengine.data.support import plane as SA  # (path layers)
from dexengine.data.support import plane_audit as SPA  # (obj_verts · qmat)

SAMPLE = _data_root() / "data/h2o/raw/sample"
#: per-take report files, next to the INTAKE plane file (h2o_frame.plane_path)
REPORT_STEM = "h2o_table_plane_report"
INLIER_M = 0.01
UP_TOL_DEG = 30.0
RNG = np.random.default_rng(0)


def load_depth(p: Path) -> np.ndarray:
    from PIL import Image

    d = np.array(Image.open(p)).astype(np.float64)
    return d / 1000.0 if d.max() > 50 else d  # mm → m (h2o depth PNGs are uint16 in mm)


def intrinsics(p: Path):
    v = list(map(float, p.read_text().split()))
    if len(v) >= 4:
        return v[0], v[1], v[2], v[3]  # fx fy cx cy [w h]
    raise ValueError(f"unknown intrinsics format: {p}")


def cloud(depth: np.ndarray, K, stride=3):
    fx, fy, cx, cy = K
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    ok = (z > 0.2) & (z < 3.0)
    x = (xs[ok] - cx) * z[ok] / fx
    y = (ys[ok] - cy) * z[ok] / fy
    return np.stack([x, y, z[ok]], 1)


def ransac_plane(P: np.ndarray, up: np.ndarray, iters=800):
    best = (0, None)
    for _ in range(iters):
        i = RNG.choice(len(P), 3, replace=False)
        a, b, c = P[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(np.dot(n, up)) < np.cos(np.radians(UP_TOL_DEG)):
            continue  # horizontal planes only (normal ≈ up)
        d = -np.dot(n, a)
        dist = np.abs(P @ n + d)
        k = int((dist < INLIER_M).sum())
        if k > best[0]:
            best = (k, (n, d))
    if best[1] is None:
        return None
    n, d = best[1]
    inl = np.abs(P @ n + d) < INLIER_M
    Q = P[inl]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n2 = vt[2]  # full_matrices=True would build an (N×N) U of tens of GB (N≈50k)
    if np.dot(n2, up) < 0:
        n2 = -n2
    return {"n": n2, "c": c, "inliers": int(inl.sum()), "frac": float(inl.mean())}


def plane_z_at(n, c, xy):
    """z on the plane at xy."""
    return c[2] - (n[0] * (xy[..., 0] - c[0]) + n[1] * (xy[..., 1] - c[1])) / n[2]


def _take_planes(take_dir: Path, R, t, up_w) -> list[dict]:
    """RANSAC desk plane per depth frame of cam4, under both cam_pose conventions, in the capture frame."""
    K = intrinsics(take_dir / "cam4/cam_intrinsics.txt")
    planes = []
    for dp in sorted((take_dir / "cam4/depth").glob("*.png")):
        cp = take_dir / "cam4/cam_pose" / f"{dp.stem}.txt"
        if not cp.is_file():
            continue
        Mc = np.array(list(map(float, cp.read_text().split()))).reshape(4, 4)
        P = cloud(load_depth(dp), K)
        for conv, Mw in (("cam2world", Mc), ("world2cam_inv", np.linalg.inv(Mc))):
            Pw = P @ Mw[:3, :3].T + Mw[:3, 3]
            pl = ransac_plane(Pw, up_w)
            if pl is None:
                continue
            n_c = R @ pl["n"]
            c_c = R @ pl["c"] + t
            planes.append(
                {
                    "frame": int(dp.stem),
                    "conv": conv,
                    "n_c": n_c,
                    "c_c": c_c,
                    "inliers": pl["inliers"],
                    "frac": pl["frac"],
                    "tilt_deg": float(np.degrees(np.arccos(np.clip(n_c[2], -1, 1)))),
                }
            )
    return planes


def _reference_plane(planes: list[dict]) -> tuple[dict, dict]:
    """The reference plane (frame 0) and the angle of every other plane to it, per convention."""
    f0 = [p for p in planes if p["frame"] == 0]
    base = f0[0] if f0 else planes[0]  # frame 0 (cam_pose ≈ identity) is the convention-independent reference
    others = [p for p in planes if p["frame"] != base["frame"]]
    conv_agree = {
        p["conv"]: float(np.degrees(np.arccos(np.clip(np.dot(p["n_c"], base["n_c"]), -1, 1)))) for p in others
    }
    return base, conv_agree


def _main_object(kp: dict, ti: dict, obj_root: Path) -> tuple:
    """Pose track and mesh vertices of the first side holding an object (None, None when there is none)."""
    side_obj = next(
        (
            s
            for s in ("right", "left")
            if kp.get(f"qpos_obj_{s}") is not None and np.abs(kp[f"qpos_obj_{s}"][:, :3]).sum() > 0
        ),
        None,
    )
    o = kp[f"qpos_obj_{side_obj}"] if side_obj else None
    md = ti.get(f"{side_obj}_object_mesh_dir") if side_obj else None
    v = SPA.obj_verts(obj_root / Path(md).name) if md else None
    return o, v


def _object_rest_gap(o, v, rests: list, n, c) -> tuple:
    """Median gap of the object's lowest vertex to the plane and median plane z under the object, over rest frames."""
    obj_rest_gap = plane_z_obj = None
    if o is not None and v is not None and rests:
        gaps, zs = [], []
        for r in rests:
            for k in range(r["start"], r["end"], 5):
                if not np.isfinite(o[k]).all():
                    continue
                vw = v @ SPA.qmat(o[k, 3:7]).T + o[k, :3]
                gaps.append(float(((vw - c) @ n).min()))
                zs.append(float(plane_z_at(n, c, o[k, :2])))
        if gaps:
            obj_rest_gap = float(np.median(gaps))
            plane_z_obj = float(np.median(zs))
    return obj_rest_gap, plane_z_obj


def _hand_points(kp: dict):
    """Wrist + fingertip keypoints of the present hands, (T, K, 3), or None."""
    hands = []
    for s in ("left", "right"):
        w = kp[f"qpos_wrist_{s}"][:, :3]
        f = kp[f"qpos_finger_{s}"][:, :, :3]
        if np.abs(w).sum() == 0:
            continue
        hands.append(np.concatenate([w[:, None, :], f], axis=1))
    hp = np.concatenate(hands, axis=1) if hands else None
    return hp


def _hand_gaps(hp, n, c) -> tuple:
    """Lowest and 5th-percentile hand gap to the plane, and the plane z span over the hands' xy range."""
    hand_gap_min = hand_gap_p5 = zspan = None
    if hp is not None:
        g = (hp - c) @ n
        hand_gap_min = float(np.nanmin(g))
        hand_gap_p5 = float(np.nanpercentile(g, 5))
        xy = hp.reshape(-1, 3)[:, :2]
        xy = xy[np.isfinite(xy).all(1)]
        pz = plane_z_at(n, c, xy)
        # how much the plane z varies over the xy range the hands covered = error span of a horizontal desk model
        zspan = float(pz.max() - pz.min())
    return hand_gap_min, hand_gap_p5, zspan


def _pivot(o, rests: list, hp, n, c) -> tuple[list, str]:
    """D10-e'' pivot: the plane point under the object's rest xy (without rest, under the hand centre xy); rotation
    leaves that point fixed, so desk_z = pivot.z."""
    if o is not None and rests:
        oxy = np.concatenate([o[r["start"] : r["end"], :2] for r in rests])
        oxy = oxy[np.isfinite(oxy).all(1)]
        pxy = np.median(oxy, axis=0)
        pivot_source = "object_rest"
    elif hp is not None:
        xy = hp.reshape(-1, 3)[:, :2]
        pxy = np.nanmedian(xy, axis=0)
        pivot_source = "hand_median"
    else:
        pxy = c[:2]
        pivot_source = "plane_centroid"
    pivot = [float(pxy[0]), float(pxy[1]), float(plane_z_at(n, c, np.array(pxy)))]
    return pivot, pivot_source


def _quality_issues(base: dict, conv_agree: dict) -> list[str]:
    """Reasons the reference plane is below quality (few inliers, frames disagree, steep tilt); empty = ok."""
    agree_c2w = conv_agree.get("cam2world")
    q_why = []
    if base["frac"] < 0.30:
        q_why.append(f"inliers {base['frac']:.2f} < 0.30")
    if agree_c2w is not None and agree_c2w > 2.0:
        q_why.append(f"frame 0/100 plane disagreement {agree_c2w:.1f}° > 2°")
    if base["tilt_deg"] > 40.0:
        q_why.append(f"tilt {base['tilt_deg']:.1f}° > 40°")
    return q_why


def analyze_take(take_dir: Path, sa_clips: dict, obj_root: Path, verbose=False):
    """Desk plane of one take from its raw depth, with the object/hand gaps and pivot in the capture frame."""
    subject, scene, take = take_dir.parts[-3:]
    clip = f"{subject}_{scene}__take{take}_cam4"
    mano_dir = SA.P.layer_path("mano", "h2o", emb="bimanual") / clip
    if not (mano_dir / "0/trajectory_keypoints.npz").is_file():
        return {"clip": clip, "error": "no mano"}
    kp = np.load(mano_dir / "0/trajectory_keypoints.npz")
    kp = {k: kp[k] for k in kp.files}
    ti = json.loads((mano_dir / "task_info.json").read_text())
    G = HF.G_of(clip)
    R, t = G[:3, :3], G[:3, 3]
    rec_fit = HF.load_fits()["clips"][clip]
    up_w = R.T @ np.array([0.0, 0.0, 1.0])  # world direction of capture +z (RANSAC candidate axis)
    planes = _take_planes(take_dir, R, t, up_w)
    if not planes:
        return {"clip": clip, "error": "no plane"}
    base, conv_agree = _reference_plane(planes)
    n, c = base["n_c"], base["c_c"]
    # objects and hands (delivered capture frame)
    rec = sa_clips.get(clip, {})
    desk_z = rec.get("desk_z")
    rests = rec.get("rest_intervals") or []
    o, v = _main_object(kp, ti, obj_root)
    obj_rest_gap, plane_z_obj = _object_rest_gap(o, v, rests, n, c)
    hp = _hand_points(kp)
    hand_gap_min, hand_gap_p5, zspan = _hand_gaps(hp, n, c)
    down_dir = -n[:2] / max(np.linalg.norm(n[:2]), 1e-9)  # direction in capture xy in which the desk slopes down
    pivot, pivot_source = _pivot(o, rests, hp, n, c)
    q_why = _quality_issues(base, conv_agree)
    return {
        "clip": clip,
        "fit_rms_mm": rec_fit["rms_mm"],
        "cam_travel_cm_max": rec_fit["cam_travel_cm_max"],
        "pivot_capture": [round(x, 4) for x in pivot],
        "pivot_source": pivot_source,
        "quality_ok": not q_why,
        "quality_why": "; ".join(q_why) or None,
        "plane_frame": base["frame"],
        "plane_conv": base["conv"],
        "inliers": base["inliers"],
        "inlier_frac": round(base["frac"], 3),
        "conv_agree_deg": {k: round(v_, 2) for k, v_ in conv_agree.items()},
        "n_capture": np.round(n, 4).tolist(),
        "c_capture": np.round(c, 4).tolist(),
        "tilt_deg": round(base["tilt_deg"], 2),
        "down_dir_xy": np.round(down_dir, 3).tolist(),
        "desk_z_intake": desk_z,
        "plane_z_at_obj": None if plane_z_obj is None else round(plane_z_obj, 4),
        "diff_plane_minus_intake": None if (plane_z_obj is None or desk_z is None) else round(plane_z_obj - desk_z, 4),
        "obj_rest_gap": None if obj_rest_gap is None else round(obj_rest_gap, 4),
        "hand_gap_min": None if hand_gap_min is None else round(hand_gap_min, 4),
        "hand_gap_p5": None if hand_gap_p5 is None else round(hand_gap_p5, 4),
        "plane_zspan_over_hand_xy": None if zspan is None else round(zspan, 4),
    }


def _parse_args() -> argparse.Namespace:
    """Command line: --subject, --verbose."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def _take_row(td: Path, sa_clips: dict, obj_root: Path, verbose: bool) -> dict:
    """analyze_take of one take; a failing take becomes an error row."""
    try:
        r = analyze_take(td, sa_clips, obj_root, verbose)
    except Exception as e:  # noqa: BLE001  per-take batch: the error is the take's row, the run goes on
        r = {"clip": "/".join(td.parts[-3:]), "error": f"{type(e).__name__}: {str(e)[:120]}"}
    return r


def _write_report(out_json: Path, rows: list[dict]) -> None:
    """Per-take report JSON next to the INTAKE plane file."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(
            {
                "tool": "dexengine/data/support/h2o.py",
                "frame_model": "capture = G · world (h2o_frame.py)",
                "rows": rows,
            },
            fh,
            indent=1,
            ensure_ascii=False,
        )


def _update_planes(rows: list[dict]) -> None:
    """Merge the measured takes into the INTAKE plane file (per-clip plane, pivot and quality) read by the
    pipeline (D10-e'', h2o_frame.tilt_transform). Existing entries stay; only takes measured now are updated
    (accumulates while streaming)."""
    pp = HF.plane_path()
    prev = json.loads(Path(pp).read_text())["clips"] if pp.is_file() else {}
    keep = (
        "n_capture",
        "c_capture",
        "pivot_capture",
        "pivot_source",
        "tilt_deg",
        "down_dir_xy",
        "inlier_frac",
        "conv_agree_deg",
        "plane_frame",
        "quality_ok",
        "quality_why",
        "desk_z_intake",
        "plane_z_at_obj",
        "obj_rest_gap",
        "hand_gap_min",
    )
    for r in rows:
        if "error" in r:
            continue
        prev[r["clip"]] = {**{k: r.get(k) for k in keep}, "version": HF.TILT_VERSION}
    pp.parent.mkdir(parents=True, exist_ok=True)
    with open(pp, "w") as fh:
        json.dump(
            {
                "tool": "dexengine/data/support/h2o.py",
                "version": HF.TILT_VERSION,
                "decision": "D10-e'' (A): normal→+z rotation, pivot = plane "
                "point under the resting object, desk_z = pivot.z",
                "n_clips": len(prev),
                "clips": dict(sorted(prev.items())),
            },
            fh,
            indent=1,
            ensure_ascii=False,
        )
    print(
        f"[h2o_table_plane] INTAKE planes {len(prev)} clips (quality_ok "
        f"{sum(1 for v in prev.values() if v.get('quality_ok'))}) → {pp}"
    )


def _cell(r: dict, k: str) -> str:
    """One TSV cell: empty for None, JSON for dicts and lists, 4 significant digits for floats."""
    return (
        ""
        if r.get(k) is None
        else (
            json.dumps(r[k])
            if isinstance(r[k], (dict, list))
            else (f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]))
        )
    )


def _write_tsv(out_tsv: Path, rows: list[dict]) -> None:
    """Per-take report TSV (one column per measured field)."""
    cols = [
        "clip",
        "tilt_deg",
        "down_dir_xy",
        "inlier_frac",
        "plane_conv",
        "conv_agree_deg",
        "desk_z_intake",
        "plane_z_at_obj",
        "diff_plane_minus_intake",
        "obj_rest_gap",
        "hand_gap_min",
        "hand_gap_p5",
        "plane_zspan_over_hand_xy",
        "fit_rms_mm",
        "cam_travel_cm_max",
        "error",
    ]
    lines = ["\t".join(cols)]
    lines += ["\t".join(_cell(r, k) for k in cols) for r in rows]
    out_tsv.write_text("\n".join(lines) + "\n")


def _print_stats(rows: list[dict]) -> None:
    """One line of tilt / desk_z difference / gap / plane span percentiles over the takes without errors."""
    ok = [r for r in rows if "error" not in r]
    if ok:
        tilt = np.array([r["tilt_deg"] for r in ok])
        gaps = np.array([r["obj_rest_gap"] for r in ok if r.get("obj_rest_gap") is not None])
        hg = np.array([r["hand_gap_min"] for r in ok if r.get("hand_gap_min") is not None])
        span = np.array([r["plane_zspan_over_hand_xy"] for r in ok if r.get("plane_zspan_over_hand_xy") is not None])
        dif = np.array([r["diff_plane_minus_intake"] for r in ok if r.get("diff_plane_minus_intake") is not None])

        def f(x, q):
            return float(np.percentile(x, q)) if len(x) else float("nan")

        print(
            f"[h2o_table_plane] takes={len(rows)} ok={len(ok)} | tilt_deg p50={f(tilt, 50):.1f} p95={f(tilt, 95):.1f} "
            f"max={tilt.max():.1f} | "
            f"plane−intake desk_z m p50={f(dif, 50):+.4f} p5={f(dif, 5):+.4f} p95={f(dif, 95):+.4f} | "
            f"obj_rest_gap m p50={f(gaps, 50):+.4f} | "
            f"hand_gap_min m p50={f(hg, 50):+.4f} p5={f(hg, 5):+.4f} | plane_zspan m p50={f(span, 50):.4f} "
            f"p95={f(span, 95):.4f}"
        )


def main() -> int:
    a = _parse_args()
    p_sa = SA.P.layer_path("intake", "h2o") / "support_audit.json"
    sa_clips = json.loads(Path(p_sa).read_text())["clips"] if p_sa.is_file() else {}
    obj_root = SA.P.layer_path("assets_objects", "h2o")
    takes = sorted({dp.parents[2] for dp in SAMPLE.glob(f"{a.subject or '*'}/*/*/cam4/depth/*.png")})
    rows = []
    for td in takes:
        r = _take_row(td, sa_clips, obj_root, a.verbose)
        rows.append(r)
        if a.verbose:
            print(r)
    out_json = HF.plane_path().with_name(f"{REPORT_STEM}.json")
    out_tsv = out_json.with_suffix(".tsv")
    _write_report(out_json, rows)
    _update_planes(rows)
    _write_tsv(out_tsv, rows)
    _print_stats(rows)
    print(f"[h2o_table_plane] {len(rows)} rows → {out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
