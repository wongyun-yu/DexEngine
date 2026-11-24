#!/usr/bin/env python3
"""Floating-hand result + post-hoc arm IK (paper Sec. E.4), SPIDER row — the sandbox trees whose hand columns are the
SPIDER floating-hand rollout, on the axis the canonical arm solver reads (the f0 stage = lead-in + base clip).

Why this axis: the SPIDER rollout was produced on our BASE tree (its reference ==
data/taco/base/.../trajectory_kinematic .npz exactly, measured 64/64), and the f0 stage is that base clip with 0 or 20
lead-in frames in front (measured: tail == base 64/64). So the rollout maps onto the f0 axis by time alone — no
placement transform, no retiming.

Time rule (written, not fitted):
  base frame i (50 Hz) is time i · 0.02 s. The rollout stores the state AFTER each 0.01 s step: flat index k is
  time (k + 1) · 0.01 s. So  S(0) = base frame 0 (the state the rollout starts from),  S(i ≥ 1) = flat[2 i − 1].
  f0 row (lead + i) hand columns = S(i);  lead-in rows stay canonical;  object columns stay canonical everywhere.
  No lag compensation (the rollout trails the reference by 7–11 steps of 0.01 s — that is part of the rollout).

Modes: `spider` (the row) · `ctl` (nothing replaced — the gate: the canonical tools on this copy must reproduce the
canonical f0 wrist caches).

The output root is a CLOSED sandbox (real files, new inodes; the canonical trees are read only):
  <out>/bimanual/<clip>/{scene.xml, scene_eq.xml, task_info.json, 0/trajectory_kinematic.npz}
Only `trajectory_kinematic.npz` is written into 0/ — it is what dexengine/placement/fk/body_positions.py reads. These
trees are an intermediate for wrist caches only; nothing trains on them.

  python -m experiments.ablations.posthoc.spider_f0_wrists --mode spider --clips <names file> --canon-f0
  <…/f0_stage/processed/taco/sharpa_wave> \
      --base <data root>/taco/base/sharpa_wave/bimanual --spider <…/spider_a_m0p2/processed/taco/sharpa_wave/bimanual> \
      --out <sandbox root> --report <json>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HAND_COLS = 56  # columns: R wrist [0, 6) · R fingers [6, 28) · L wrist [28, 34) · L fingers [34, 56) (objects [56, 70))
WRIST_XYZ = {"right": slice(0, 3), "left": slice(28, 31)}
TREE_FILES = ("scene.xml", "scene_eq.xml", "task_info.json")
ROLL_DT, BASE_HZ = 0.01, 50.0
TOL_T = 1e-6
# flat[0] (0.01 s after the start) vs base frame 0, wrist xyz — the rollout starts at the reference
MAX_F0_GAP_M = 0.02


def one(clip: str, a) -> dict:
    src = Path(a.canon_f0) / "bimanual" / clip
    dst = Path(a.out) / "bimanual" / clip
    if dst.exists():
        raise SystemExit(f"[SPIDER_F0] {dst} exists — refused (the sandbox is built once)")
    z = np.load(src / "0" / "trajectory_kinematic.npz", allow_pickle=True)
    keep = {k: z[k] for k in z.files}
    q = np.array(keep["qpos"], dtype=np.float64)
    hz = float(keep["frequency"])
    if abs(hz - BASE_HZ) > 1e-9:
        raise SystemExit(f"[SPIDER_F0] {clip}: tree frequency {hz} != {BASE_HZ}")
    bq = np.load(Path(a.base) / clip / "0" / "trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    tb = bq.shape[0]
    lead = q.shape[0] - tb
    if lead < 0 or not np.array_equal(q[lead:], bq):
        raise SystemExit(f"[SPIDER_F0] {clip}: the f0 tree is not lead-in + base (lead {lead})")
    row = {"clip": clip, "frames_f0": int(q.shape[0]), "frames_base": int(tb), "lead": int(lead)}
    if a.mode == "spider":
        s = np.load(Path(a.spider) / clip / "0" / "trajectory_mjwp.npz", allow_pickle=True)
        flat = np.asarray(s["qpos"], dtype=np.float64).reshape(-1, s["qpos"].shape[-1])
        t = np.asarray(s["time"], dtype=np.float64).reshape(-1)
        if flat.shape[1] != q.shape[1]:
            raise SystemExit(f"[SPIDER_F0] {clip}: rollout width {flat.shape[1]} != tree {q.shape[1]}")
        if np.abs(t - ROLL_DT * (np.arange(t.size) + 1)).max() > TOL_T:
            raise SystemExit(f"[SPIDER_F0] {clip}: rollout time is not (k + 1) · {ROLL_DT}")
        need = 2 * (tb - 1) - 1
        if flat.shape[0] <= need:
            raise SystemExit(f"[SPIDER_F0] {clip}: rollout has {flat.shape[0]} steps, need index {need}")
        gap0 = max(float(np.linalg.norm(flat[0, c] - bq[0, c])) for c in WRIST_XYZ.values())
        if gap0 > MAX_F0_GAP_M:
            raise SystemExit(f"[SPIDER_F0] {clip}: rollout step 0 is {gap0 * 1e3:.1f} mm from base frame 0")
        hand = np.empty((tb, HAND_COLS))
        hand[0] = bq[0, :HAND_COLS]
        hand[1:] = flat[2 * np.arange(1, tb) - 1, :HAND_COLS]
        canon_hand = q[lead:, :HAND_COLS].copy()
        q[lead:, :HAND_COLS] = hand
        keep["qpos"] = q.astype(keep["qpos"].dtype)
        qv = np.array(keep["qvel"], dtype=np.float64)  # hand dofs are 1-dof joints: qvel columns == qpos columns
        qv[lead:, :HAND_COLS] = np.gradient(hand, 1.0 / hz, axis=0) if tb > 1 else 0.0
        keep["qvel"] = qv.astype(keep["qvel"].dtype)
        row.update(rollout_steps=int(flat.shape[0]), step0_gap_mm=gap0 * 1e3)
        for sd, c in WRIST_XYZ.items():
            d = np.linalg.norm(hand[:, c] - canon_hand[:, c], axis=1) * 1e3
            row[f"wrist_xyz_col_dev_mm_mean_{sd}"] = float(d.mean())
            row[f"wrist_xyz_col_dev_mm_max_{sd}"] = float(d.max())
        fing = np.r_[6:28, 34:56]
        row["finger_dev_rad_p99"] = float(np.quantile(np.abs(hand[:, fing] - canon_hand[:, fing]), 0.99))
    (dst / "0").mkdir(parents=True)
    for f in TREE_FILES:
        shutil.copyfile(src / f, dst / f)
    np.savez(dst / "0" / "trajectory_kinematic.npz", **keep)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("spider", "ctl"))
    ap.add_argument("--clips", required=True)
    ap.add_argument("--canon-f0", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--spider", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    if a.mode == "spider" and not a.spider:
        raise SystemExit("[SPIDER_F0] --mode spider needs --spider")
    out = Path(a.out).resolve()
    for canon in (a.canon_f0, a.base, a.spider):
        if canon and (out == Path(canon).resolve() or Path(canon).resolve() in out.parents):
            raise SystemExit(f"[SPIDER_F0] --out is inside a source root {canon} — refused")
    clips = [ln for ln in Path(a.clips).read_text(encoding="utf-8").splitlines() if ln]
    rows = [one(c, a) for c in clips]
    made = sorted(p.name for p in (out / "bimanual").iterdir())
    if made != sorted(clips):
        raise SystemExit(f"[SPIDER_F0] set mismatch: {len(made)} trees vs {len(clips)} clips")
    links = [str(p) for p in out.rglob("*") if p.is_symlink() or (p.is_file() and p.stat().st_nlink > 1)]
    if links:
        raise SystemExit(f"[SPIDER_F0] sandbox is not closed: {links[:3]}")
    rep = {
        "mode": a.mode,
        "clips": len(rows),
        "lead_counts": {str(k): sum(r["lead"] == k for r in rows) for k in sorted({r["lead"] for r in rows})},
        "time_rule": "f0 row (lead + i) = S(i); S(0) = base frame 0; S(i>=1) = rollout flat[2i-1]; no lag shift",
        "rows": rows,
    }
    Path(a.report).write_text(json.dumps(rep, indent=1))
    msg = f"[SPIDER_F0] mode={a.mode} trees={len(rows)} lead={rep['lead_counts']} closed=yes → {out}"
    if a.mode == "spider":
        g = [r["step0_gap_mm"] for r in rows]
        msg += f" · step-0 gap mm median {np.median(g):.1f} max {max(g):.1f}"
    print(msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
