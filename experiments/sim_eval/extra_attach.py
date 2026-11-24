#!/usr/bin/env python3
"""Resting extras → attached to the main object: an evaluation scene rule (read by the closed loop, DP and GR00T
alike).

Why: in evaluation (session option extra_f0 = extra slots fixed at frame 0) an extra resting on the main object (e.g.
    bread on a plate) stayed in place while only the plate moved, whereas the training data (stage 1) moved the extra
    along to the next reference frame (a train ↔ evaluation mismatch).
Rule (no clip-name lookup, applies to unseen clips): at the reference frame-0 pose (trajectory_kinematic), if at least
  SUP_FRAC of the extra e's bottom points (vertices within BOT_M of e's lowest point, thinned evenly to MAXPTS) meet a
  mesh face of the main object m on the vertical line straight below (−z) within GAP_M (faces up to PEN_M above the
  point count too — mesh overlap), 'e rests on m'. A resting e is rigidly attached to m in the evaluation scene:
  relative pose
  = the reference frame-0 value; before every physics step e = m's current pose ∘ that relative pose. An extra resting
  on two main objects dies (no rule where to attach); a missing mesh dies (no silent skip).
Mesh = paths.main_root() / data_layout.object_path(dataset, mesh_dir) / visual.obj (the judges' mesh_pts path).
Markers: per attachment `[EXTRA_ATTACH] <tree> <extra>→<main> frac=… gap_mm=…`, none `[EXTRA_ATTACH] <tree> none
n_extra=<n>`.
  python -m experiments.sim_eval.extra_attach <tree folder>…   → one verdict per line (numpy only)."""

from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

import numpy as np

MARK = "[EXTRA_ATTACH]"
BOT_M = 0.005  # bottom points = within 5 mm of the lowest point
GAP_M = 0.015  # a main-object face within 1.5 cm straight below
PEN_M = 0.005  # faces up to 5 mm above the point count (mesh overlap)
SUP_FRAC = 0.5  # at least half of the bottom points
MAXPTS = 200


def _paths():
    from dexengine.core import paths as P  # noqa: PLC0415

    return P


def quat_R(q: np.ndarray) -> np.ndarray:
    """wxyz → 3×3 (body → world)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@functools.lru_cache(None)
def _mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    V, F = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("v "):
                V.append([float(t) for t in ln.split()[1:4]])
            elif ln.startswith("f "):
                ix = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
                F += [[ix[0], ix[k], ix[k + 1]] for k in range(1, len(ix) - 1)]
    if not V or not F:
        raise SystemExit(f"{MARK} empty mesh {path}")
    return np.array(V), np.array(F)


def mesh_file(ds: str, mesh_dir: str) -> str:
    P = _paths()
    f = Path(P.main_root()) / P.data_layout.object_path(ds, mesh_dir) / "visual.obj"
    if not f.is_file():
        raise SystemExit(f"{MARK} object mesh missing {f}")
    return str(f)


def _below(P: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per point the highest face z its downward vertical line meets (up to PEN_M above the point) · none = nan."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    v0, v1 = B[:, :2] - A[:, :2], C[:, :2] - A[:, :2]
    d = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(d) > 1e-12
    dd = np.where(ok, d, 1.0)
    out = np.full(len(P), np.nan)
    for i, pt in enumerate(P):
        v2 = pt[:2] - A[:, :2]
        u = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / dd
        w = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / dd
        inside = ok & (u >= 0) & (w >= 0) & (u + w <= 1)
        if not inside.any():
            continue
        z = (A[:, 2] + u * (B[:, 2] - A[:, 2]) + w * (C[:, 2] - A[:, 2]))[inside]
        z = z[z <= pt[2] + PEN_M]
        if len(z):
            out[i] = z.max()
    return out


def support_pairs(tree_dir: str | Path, mesh_of=None) -> list[dict]:
    """One tree → resting extras [{extra, main, frac, gap_mm, p_rel (3,), R_rel (3,3)}] (tree frame, frame 0; relative
    pose =
    e seen from m's body frame). mesh_of(ds, mesh_dir) → visual.obj path (default mesh_file)."""
    td = Path(tree_dir)
    ti = json.loads((td / "task_info.json").read_text())
    ds = str(ti.get("dataset_name") or "")
    if not ds:
        raise SystemExit(f"{MARK} {td}: task_info.dataset_name missing")
    q = np.load(td / "0" / "trajectory_kinematic.npz")["qpos"]
    mof = mesh_of or mesh_file
    W = {}
    for o in ti["objects"]:
        if o.get("role") not in ("main", "extra"):
            continue
        s = int(o["qpos_start"])
        p0, R0 = q[0, s : s + 3].astype(np.float64), quat_R(q[0, s + 3 : s + 7])
        V, F = _mesh(mof(ds, o["mesh_dir"]))
        W[o["name"]] = (o["role"], V @ R0.T + p0, F, p0, R0)
    out = []
    for en, (er, Ve, _f, pe, Re) in W.items():
        if er != "extra":
            continue
        hit = []
        for mn, (mr, Vm, Fm, pm, Rm) in W.items():
            if mr != "main":
                continue
            c = Ve[:, :2].mean(0)
            lo, hi = Vm[:, :2].min(0), Vm[:, :2].max(0)
            if not (np.all(c >= lo) and np.all(c <= hi)):
                continue
            bot = Ve[Ve[:, 2] <= Ve[:, 2].min() + BOT_M]
            if len(bot) > MAXPTS:
                bot = bot[np.linspace(0, len(bot) - 1, MAXPTS).astype(int)]
            g = bot[:, 2] - _below(bot, Vm, Fm)
            frac = float(np.mean(np.isfinite(g) & (g <= GAP_M)))
            if frac >= SUP_FRAC:
                hit.append(
                    {
                        "extra": en,
                        "main": mn,
                        "frac": round(frac, 3),
                        "gap_mm": round(float(np.nanmedian(g)) * 1000, 1),
                        "p_rel": Rm.T @ (pe - pm),
                        "R_rel": Rm.T @ Re,
                    }
                )
        if len(hit) > 1:
            raise SystemExit(
                f"{MARK} {td.name}: extra {en} rests on two main objects {[h['main'] for h in hit]} — no "
                "rule where to attach"
            )
        out += hit
    return out


def extra_names(tree_dir: str | Path) -> list[str]:
    ti = json.loads((Path(tree_dir) / "task_info.json").read_text())
    return [o["name"] for o in ti["objects"] if o.get("role") == "extra"]


def main() -> None:
    for td in sys.argv[1:]:
        pr = support_pairs(td)
        if pr:
            for h in pr:
                print(
                    f"{MARK} {os.path.basename(td.rstrip('/'))} {h['extra']}→{h['main']} frac={h['frac']} "
                    f"gap_mm={h['gap_mm']}"
                )
        else:
            print(f"{MARK} {os.path.basename(td.rstrip('/'))} none n_extra={len(extra_names(td))}")


if __name__ == "__main__":
    main()
