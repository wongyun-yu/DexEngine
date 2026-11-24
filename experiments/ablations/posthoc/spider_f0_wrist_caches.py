#!/usr/bin/env python3
"""Floating-hand result + post-hoc arm IK (paper Sec. E.4), SPIDER source — the wrist caches the arm solver reads.

Input = what the canonical body / wrist tools wrote for the closed sandbox built by `spider_f0_wrists.py`
(`wrist_poses_<side>/<clip>.npz`, rows = lead-in + clip). Output = the same files with ONE rule applied:

  rows [0:lead]  = the canonical f0-stage cache rows, copied (the lead-in is not SPIDER's — its rollout starts at
                   clip frame 0 — and in 19 of the 64 clips the canonical lead-in rotations were written by the
                   in-place lead-in repair, not by FK of the tree, so "FK of the canonical lead-in" would not be
                   the canonical lead-in; measured: position 0 mm, rotation up to 6.38 deg);
  rows [lead:]   = the tool rows of the input tree (SPIDER's wrist path), sign of the whole tail chosen so that
                   the quaternion is continuous across the junction.

`--mode ctl` applies the same rule to the control sandbox (canonical hand columns) and is the gate: the result must
equal the canonical caches within `--tol-mm` / `--tol-deg` in every clip, else the run stops.

  python -m experiments.ablations.posthoc.spider_f0_wrist_caches --mode spider --clips <names> --canon-f0 <F0>
  --tool-out <dir> \
      --lead-report <spider_f0_wrists report.json> --out <dir> --report <json>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

KEYS = ("positions", "quaternions")
SIDES = ("right", "left")


def rot_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(np.sum(a.astype(np.float64) * b.astype(np.float64), axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(d, 0.0, 1.0)))


def load(p: Path) -> dict:
    z = np.load(p)
    if tuple(sorted(z.files)) != tuple(sorted(KEYS)):
        raise SystemExit(f"[SPIDER_WC] {p}: keys {z.files} != {KEYS}")
    out = {k: z[k] for k in KEYS}
    if out["positions"].dtype != np.float32 or out["quaternions"].dtype != np.float32:
        raise SystemExit(f"[SPIDER_WC] {p}: dtype is not float32")
    return out


def _parse_args():
    """Parse the command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("spider", "ctl"))
    ap.add_argument("--clips", required=True)
    ap.add_argument("--canon-f0", required=True, help="canonical f0-stage root (holds wrist_poses_<side>/) — read only")
    ap.add_argument("--tool-out", required=True, help="dir with wrist_poses_<side>/ written by the canonical tools")
    ap.add_argument("--lead-report", required=True, help="report of spider_f0_wrists.py for the same sandbox")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--tol-mm", type=float, default=0.1)
    ap.add_argument("--tol-deg", type=float, default=0.1)
    a = ap.parse_args()
    return a


def _read_inputs(a):
    """Source roots (out must lie outside them), the clip list and the per-clip lead from the lead report."""
    canon, tool, out = Path(a.canon_f0).resolve(), Path(a.tool_out).resolve(), Path(a.out).resolve()
    for src in (canon, tool):
        if out == src or src in out.parents:
            raise SystemExit(f"[SPIDER_WC] --out {out} is inside a source root {src} — refused")
    clips = [ln for ln in Path(a.clips).read_text(encoding="utf-8").splitlines() if ln]
    rep = json.loads(Path(a.lead_report).read_text(encoding="utf-8"))
    if rep["mode"] != a.mode:
        raise SystemExit(f"[SPIDER_WC] lead report is mode={rep['mode']}, asked {a.mode}")
    lead = {r["clip"]: int(r["lead"]) for r in rep["rows"]}
    if set(lead) != set(clips):
        raise SystemExit("[SPIDER_WC] clip set of the lead report != --clips")
    return canon, out, tool, clips, lead


def _joined_poses(canon, s, c, tool, lead):
    """Canonical rows [0:lead] + tool rows [lead:] of one side cache (tail sign continuous at the junction)."""
    cz, tz = load(canon / f"wrist_poses_{s}" / f"{c}.npz"), load(tool / f"wrist_poses_{s}" / f"{c}.npz")
    n, L = cz["positions"].shape[0], lead[c]
    if tz["positions"].shape[0] != n or tz["quaternions"].shape != cz["quaternions"].shape or not 0 <= L < n:
        raise SystemExit(f"[SPIDER_WC] {s} {c}: rows {tz['positions'].shape[0]} vs canonical {n} (lead {L})")
    pos, quat = tz["positions"].copy(), tz["quaternions"].copy()
    flipped = False
    if L:
        if float(np.dot(cz["quaternions"][L - 1].astype(np.float64), quat[L].astype(np.float64))) < 0.0:
            quat[L:] = -quat[L:]
            flipped = True
        pos[:L], quat[:L] = cz["positions"][:L], cz["quaternions"][:L]
    step = np.sum(quat[1:].astype(np.float64) * quat[:-1].astype(np.float64), axis=-1)
    if step.size and float(step.min()) < 0.0:
        raise SystemExit(f"[SPIDER_WC] {s} {c}: quaternion sign is not continuous after the join")
    return cz, L, n, pos, quat, flipped


def _deviations(pos, cz, quat, L):
    """Per-row deviation from the canonical cache and the junction steps (joined and canonical)."""
    dp = np.linalg.norm(pos.astype(np.float64) - cz["positions"].astype(np.float64), axis=-1) * 1e3
    dr = rot_deg(quat, cz["quaternions"])
    jump_mm = float(np.linalg.norm(pos[L].astype(np.float64) - pos[L - 1].astype(np.float64)) * 1e3) if L else 0.0
    jump_deg = float(rot_deg(quat[L], quat[L - 1])) if L else 0.0
    cj_mm = (
        float(np.linalg.norm(cz["positions"][L].astype(np.float64) - cz["positions"][L - 1].astype(np.float64)) * 1e3)
        if L
        else 0.0
    )
    cj_deg = float(rot_deg(cz["quaternions"][L], cz["quaternions"][L - 1])) if L else 0.0
    return dp, dr, jump_mm, jump_deg, cj_mm, cj_deg


def _summary(a, clips, rows, canon, tool, out, lead, worst_mm, worst_deg, gate):
    """The report record."""
    summ = {
        "mode": a.mode,
        "clips": len(clips),
        "files": len(rows),
        "canon_f0": str(canon),
        "tool_out": str(tool),
        "out": str(out),
        "rule": "rows [0:lead] = canonical cache rows · rows [lead:] = tool rows of the input tree (tail sign "
        "chosen for continuity at the junction)",
        "lead_counts": {str(k): sum(lead[c] == k for c in clips) for k in sorted(set(lead.values()))},
        "tails_sign_flipped": sum(r["tail_sign_flipped"] for r in rows),
        "leadin_equal_files": sum(r["leadin_equal"] for r in rows),
        "dev_from_canonical": {
            "mm_max": worst_mm,
            "deg_max": worst_deg,
            "mm_mean_median_over_files": float(np.median([r["dev_mm_mean"] for r in rows])),
            "deg_mean_median_over_files": float(np.median([r["dev_deg_mean"] for r in rows])),
        },
        "junction_step": {
            "mm_max": max(r["junction_step_mm"] for r in rows),
            "deg_max": max(r["junction_step_deg"] for r in rows),
            "canonical_mm_max": max(r["canonical_junction_step_mm"] for r in rows),
            "canonical_deg_max": max(r["canonical_junction_step_deg"] for r in rows),
        },
        "gate": {"tol_mm": a.tol_mm, "tol_deg": a.tol_deg, "passed": gate},
        "rows": rows,
    }
    return summ


def main() -> int:
    a = _parse_args()
    canon, out, tool, clips, lead = _read_inputs(a)
    rows = []
    for s in SIDES:
        (out / f"wrist_poses_{s}").mkdir(parents=True, exist_ok=False)
        for c in clips:
            cz, L, n, pos, quat, flipped = _joined_poses(canon, s, c, tool, lead)
            dp, dr, jump_mm, jump_deg, cj_mm, cj_deg = _deviations(pos, cz, quat, L)
            tmp = out / f"wrist_poses_{s}" / f"{c}.tmp.npz"
            np.savez(tmp, positions=pos, quaternions=quat)
            os.replace(tmp, out / f"wrist_poses_{s}" / f"{c}.npz")
            rows.append(
                {
                    "clip": c,
                    "side": s,
                    "rows": int(n),
                    "lead": L,
                    "tail_sign_flipped": flipped,
                    "leadin_equal": bool(
                        L == 0
                        or (
                            np.array_equal(pos[:L], cz["positions"][:L])
                            and np.array_equal(quat[:L], cz["quaternions"][:L])
                        )
                    ),
                    "dev_mm_mean": float(dp[L:].mean()),
                    "dev_mm_max": float(dp.max()),
                    "dev_deg_mean": float(dr[L:].mean()),
                    "dev_deg_max": float(dr.max()),
                    "junction_step_mm": jump_mm,
                    "junction_step_deg": jump_deg,
                    "canonical_junction_step_mm": cj_mm,
                    "canonical_junction_step_deg": cj_deg,
                }
            )
    for s in SIDES:
        made = sorted(p.stem for p in (out / f"wrist_poses_{s}").glob("*.npz"))
        if made != sorted(clips):
            raise SystemExit(f"[SPIDER_WC] {s}: {len(made)} written vs {len(clips)} asked")
    worst_mm, worst_deg = max(r["dev_mm_max"] for r in rows), max(r["dev_deg_max"] for r in rows)
    gate = None
    if a.mode == "ctl":
        gate = bool(worst_mm <= a.tol_mm and worst_deg <= a.tol_deg)
    summ = _summary(a, clips, rows, canon, tool, out, lead, worst_mm, worst_deg, gate)
    Path(a.report).write_text(json.dumps(summ, indent=1))
    print(
        f"[SPIDER_WC] mode={a.mode} files={len(rows)} lead={summ['lead_counts']} flipped={summ['tails_sign_flipped']} "
        f"lead-in equal {summ['leadin_equal_files']}/{len(rows)} · dev from canonical max {worst_mm:.4f} mm / "
        f"{worst_deg:.4f} deg · junction step max {summ['junction_step']['mm_max']:.2f} mm / "
        f"{summ['junction_step']['deg_max']:.2f} deg (canonical {summ['junction_step']['canonical_mm_max']:.2f} mm / "
        f"{summ['junction_step']['canonical_deg_max']:.2f} deg) · gate={gate} → {out}",
        flush=True,
    )
    if a.mode == "ctl" and not gate:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
