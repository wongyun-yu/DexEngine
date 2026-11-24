#!/usr/bin/env python3
"""The f0 repair happens **1× before judging** — it builds a placement-independent shadow staging.

Canonical order: **f0 → judge → (placement bake → IK) → retiming**.
This script is the first of those steps.

Why this order
    With `placement bake → f0 → IK → retiming`, the judge looks at the wrist **before** f0 while the
    bake solves the wrist **after** f0 — the same placement can pass the judge and fail the bake
    (measured: a clip passed the judge and then missed in the bake by 127 mm). With f0 ahead of the
    judge, judge and bake see the **same trajectory**. judge = bake closes structurally.

Why it may go ahead of placement (no re-measurement needed — measured)
    f0 looks only at the trajectory-frame wrist cache, the finger angles and the object, and the
    desk is the z=0 plane of that frame. The dx/dy/yaw of a placement is a z-invariant rigid
    transform that moves hand and object **together**, so the relative geometry is unchanged, and
    dz (robot_z_off) has no parameter at all in the f0 code path. In an A/B measurement (before and
    after a synthetic placement shift) n_pen, max depth, P and status matched exactly.

What it builds (existing outputs are **read only**) — F0 = paths.f0_root(dataset)
    <F0>/<hand>/<emb>/<clip>/                        ← **shadow copy** of the source tree + f0 repair
    <F0>/<hand>/wrist_poses_{left,right}/<clip>.npz  ← wrist cache + f0
    <F0>/<hand>/LEDGER.tsv                           ← clip · status · P · elapsed (for the count check)

    Source layer: `base` = data/<ds>/base/<hand>/<emb>/<clip> — the only source layer (rule set v2).

    The originals are never touched (the shadow is a copy), so "before f0" stays available for an A/B
    comparison.

Cache · idempotence
    If the clip directory holds `F0_REPAIR_INPLACE.json` (repaired) or `F0_REPAIR.json` (clean, so
    no repair), it is **skipped**. That is, f0 is computed **1×** per (hand, clip), and the judge's
    anchor retries, the bake retries and combo (arm) swaps all reuse the same result.
    Re-solving f0 on every anchor retry would cost ~10 s per side × the number of candidates.
    `--force` recomputes (the shadow copy is deleted and rebuilt).

The repair itself is `f0.repair.main_steps` with `--in_place` argv **as it is** (nothing is
reimplemented) — the shadow tree has no F0_REPAIR.json, so mode=prepend, length T → T+P.

Engine: the **GPU engine is the only one**. `f0/gpu.py --worker` processes
(one CUDA device each, `--gpus` × `--gpu-workers-per-device`) run many clips' generators at once and batch
their objective + Nelder-Mead on the GPU; classification (`finish`), markers and the ledger are here.

Usage ($RETARGET_PY = python -m dexengine.core.paths --what retarget-py)
  $RETARGET_PY dexengine/retarget/f0/prejudge.py --hand <hand> --clips <list> \
       --dataset hocap --emb bimanual --src-layer base         # base layer (the canonical entry)
  DEXENGINE_F0_GPUS=0,1 DEXENGINE_F0_GPU_WORKERS=2 $RETARGET_PY dexengine/retarget/f0/prejudge.py …   # devices
  F0=$(python3 dexengine/core/paths.py --what layer --layer f0_root --dataset taco)
  # judge: python -m dexengine.placement.judge.core --wrist-root $F0/sharpa_wave --traj-root $F0/sharpa_wave/bimanual
  # bake:  scripts/stage2/chain_fast.sh with env DEXENGINE_F0_STAGE=$F0 (required: the f0 shadow root)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Canonical owner of the robot names = dexengine/core/paths.py — no copies
from dexengine.core import blas_pin
from dexengine.core.paths import HAND_SPEC_DIR as SPEC_DIR_OF
from dexengine.core.paths import (
    data_layout,
    layer_path,
    read_clip_list,
    require_retarget_py,
    retarget_py,
    wrist_base_dir,
)

# The cache marker this script leaves (the basis of idempotence). Why it is kept separate from
# f0.repair's ledger: f0 has three branches — "repaired / clean / skipped as shallow" — and the
# third one leaves no ledger. Name owner = placement.judge.f0_key (the region-dump reuse gate reads it too).
from dexengine.placement.judge.f0_key import MARKER  # noqa: E402

# Single owner of the machine-readable completion ledger (file name, marker, schema, validation): that module.
# Consumers never derive counts from the human-readable completion line.
# ledger: stdlib only, so it loads under any interpreter
from dexengine.retarget.f0 import ledger as f0_summary  # noqa: E402
from dexengine.retarget.f0.prejudge_modes import run_f0, sweep_fingerprints
from dexengine.retarget.f0.shadow_cli import (
    _report_upstream_completeness,
    _seed_only,
    resolve_shadow_root,
    shadow_root_lock,
)
from dexengine.retarget.f0.shadow_key import SRC_LAYERS
from dexengine.retarget.f0.shadow_status import (
    SIDES,
)
from dexengine.retarget.f0.shadow_tree import applied_scene


def _add_source_args(ap):
    """Hand, clip, dataset/emb/layer and shadow-root arguments."""
    ap.add_argument("--hand", required=True, choices=sorted(SPEC_DIR_OF))  # no default
    ap.add_argument("--clips", default="", help="one clip per line (spaces allowed)")
    ap.add_argument("--clip", default="", help="a single clip")
    # Source axis (grounding for 9 datasets) — every path is built by the paths/data_layout owner
    ap.add_argument(
        "--dataset",
        required=True,
        choices=sorted(data_layout.DATASETS),
        help="dataset (data_layout.DATASETS); the source, raw-wrist, shadow and mesh layers all follow it",
    )
    ap.add_argument(
        "--emb",
        required=True,
        choices=list(data_layout.EMBODIMENTS),
        help="embodiment (subdirectory of the source layer and of the shadow)",
    )
    ap.add_argument(
        "--src-layer",
        default="base",
        choices=SRC_LAYERS,
        help="source layer: base (the canonical entry; scene xml copied with absolute paths), the only source layer",
    )
    ap.add_argument(
        "--f0-root",
        default="",
        help="shadow root (the paths.f0_root(dataset) location: <here>/<hand spec>/<emb>/<clip>). "
        "Default = paths.f0_root(dataset) (env DEXENGINE_F0_STAGE takes precedence)",
    )
    ap.add_argument(
        "--seed-from",
        default="",
        help="seed and exit (isolated run): copy the clip and wrist shadows of this (canonical) shadow "
        "root to --f0-root as real copies, keep existing ones, do not run f0. Marker [F0_SEED]",
    )
    ap.add_argument(
        "--require-complete", action="store_true", help="stop at once if any upstream input is missing (wrist cache)."
    )


def _add_run_args(ap):
    """Cache, fingerprint, budget and GPU-engine arguments."""
    # Number of admission threads (prep = cache decision, mirroring, fingerprint). Does not affect the output
    # (independent per-clip copies). Default: capacity.plan (cap 16).
    ap.add_argument("--prep-threads", type=int, default=None)
    ap.add_argument(
        "--force",
        action="store_true",
        help="ignore caches and fingerprints and recompute everything (forced regeneration)",
    )
    ap.add_argument(
        "--check-stale",
        action="store_true",
        help="only compare fingerprints, without writing: count clips whose shadow disagrees with "
        "upstream. Without --clips every shadow with a marker is checked",
    )
    ap.add_argument(
        "--stamp-fingerprint",
        action="store_true",
        help="stamp a fingerprint into markers that predate the convention, but only those passing the "
        "`shadow_is_current` structure check (inodes + wrist-cache causal invariant); "
        "the rest are reported stale and not stamped. Does not rerun f0",
    )
    # A passage, not a copy of the value: with the default None nothing is passed and the budget is decided by
    # repair_cfg.SIDE_BUDGET_EVALS alone. Only an explicit value is passed on as is (needed for negative tests, i.e.
    # whether a small budget really triggers SKIP_BUDGET, and for calibration measurements).
    ap.add_argument(
        "--side-budget-evals",
        type=int,
        default=None,
        help="for tests and calibration: passed as is to repair's --side_budget_evals",
    )
    # GPU engine (the only one). The device and worker entry points are the two arguments below; callers (the
    # data.stages step 7, scripts/stage2_ground.sh …) call this tool by argv only, so the environment variables
    # DEXENGINE_F0_GPUS / DEXENGINE_F0_GPU_WORKERS come in as the defaults here (this argparse owns their names and
    # parsing).
    ap.add_argument(
        "--gpus",
        default=os.environ.get("DEXENGINE_F0_GPUS") or "0",
        help="CUDA device indices (comma-separated; default = env DEXENGINE_F0_GPUS, else 0)",
    )
    ap.add_argument("--gpu-workers-per-device", type=int, default=int(os.environ.get("DEXENGINE_F0_GPU_WORKERS") or 1))
    ap.add_argument("--gpu-max-inst", type=int, default=3000, help="max concurrent NM instances per worker")
    ap.add_argument(
        "--gpu-ahead",
        type=int,
        default=600,
        help="clips sent ahead per worker (concurrent NM instances up to ≈ 4~6 × this; cap --gpu-max-inst)",
    )
    ap.add_argument(
        "--python",
        default=retarget_py(),
        help="interpreter of the GPU workers (gpu --worker), with warp, pinocchio and mujoco",
    )


def build_parser():
    ap = argparse.ArgumentParser()
    _add_source_args(ap)
    _add_run_args(ap)
    return ap


def _print_run_markers(a, clips, dst_tree, root, sp, stage_src, wrist_dirs):
    """Print the run header and the applied-value markers ([F0_SRC], [F0_SCENE]); returns the applied scene."""
    print(
        f"[f0_prejudge] hand={a.hand}({sp}) clips {len(clips)} · admission threads {a.prep_threads or 'capacity'}"
        f" · GPU {a.gpus} × workers {a.gpu_workers_per_device}"
        + (" · **check only (no writes)**" if a.check_stale and not a.stamp_fingerprint else "")
        + f"\n  staging (read) {stage_src}\n  raw wrist (read) {wrist_dirs['right'].parent}\n  shadow (write) {root}",
        flush=True,
    )
    # Applied-value marker — one line for where the source axis actually resolved to. A run
    # without this line is treated as void.
    print(
        f"[F0_SRC] dataset={a.dataset} emb={a.emb} layer={a.src_layer} src_root={stage_src} "
        f"shadow={dst_tree} n={len(clips)}",
        flush=True,
    )
    # Scene marker: f0's desk plane depends on the scene (repair -> coords.scene_z_offset). The value is owned by
    # coords.scene_mode (env DEXENGINE_SCENE); here the applied value is read and recorded in the log, the completion
    # ledger and markers. The stage-1 driver compares the completion ledger's `scene` with the plan's expected
    # value. It is not part of the cache key (changing the key convention would invalidate every cache).
    scene = applied_scene()
    print(
        f"[F0_SCENE] scene={scene} env_DEXENGINE_SCENE={os.environ.get('DEXENGINE_SCENE') or '-'} shadow_root={root}",
        flush=True,
    )
    return scene


def main() -> int:
    ap = build_parser()
    a = ap.parse_args()
    # Name of this run. The ledger file name and the marker use the same value, so consumers can pick their own run's
    # ledger among other runs' ledgers in the same directory.
    run_id, run_started = f0_summary.new_run_id(), time.time()
    print(
        f"[F0_BLAS] OPENBLAS_CORETYPE={blas_pin.KERNEL} pinned in the worker env (owner blas_pin)",
        flush=True,
    )
    _env = blas_pin.compute_env()
    print(
        f"[F0_ENV] host={_env['host']} cpu={_env['cpu']} python={_env['python']} "
        + " ".join(f"{k}={v}" for k, v in _env["packages"].items())
        + " blas="
        + ",".join(_env["blas"])
        + " (recorded per shadow in F0_PREJUDGE.json compute_env · owner blas_pin.compute_env)",
        flush=True,
    )

    sweep_only = a.check_stale or a.stamp_fingerprint
    if a.clip:
        spec = a.clip
    elif a.clips:
        spec = a.clips
    elif sweep_only:
        spec = None  # filled below from the full shadow listing
    else:
        print("--clips or --clip required")
        return 2
    # Name normalisation happens **once, at the entrance**, in the owner function (one
    # normal form).
    clips = read_clip_list(spec, dataset=a.dataset, label="f0_prejudge")
    sp = SPEC_DIR_OF[a.hand]  # hand → spec directory
    # Source clip root: layer_path(src_layer)/<hand spec>/<emb>.
    stage_src = layer_path(a.src_layer, a.dataset, hand=sp, emb=a.emb)
    if not stage_src.is_dir():
        raise SystemExit(f"[f0_prejudge] source layer missing: layer={a.src_layer} dataset={a.dataset} → {stage_src}")
    # Wrist RAW cache (1× per hand and clip, independent of the combo) — the per-side directory is
    # built by the owner (paths.wrist_base_dir).
    wrist_dirs = {sd: wrist_base_dir(sp, sd, a.dataset) for sd in SIDES}
    shadow_root = resolve_shadow_root(a.dataset, a.f0_root)
    root = shadow_root / sp
    dst_tree = root / a.emb
    dst_wrist_dirs = {sd: root / wrist_base_dir(sp, sd, a.dataset).name for sd in SIDES}
    if a.seed_from:  # seed only: writes --f0-root, reads the source
        return _seed_only(a, clips, shadow_root, sp)
    dst_tree.mkdir(parents=True, exist_ok=True)
    # only writing runs take the shadow-root lock (check-only runs do not write); the object is held to the end (closing
    # releases it)
    _root_lock = None if (a.check_stale and not a.stamp_fingerprint) else shadow_root_lock(root, run_id)  # noqa: F841
    if sweep_only and not clips:
        clips = sorted(m.parent.name for m in dst_tree.glob(f"*/{MARKER}"))
    a.python = str(a.python)
    if not os.path.isfile(a.python):
        # Do not emit a FAIL_EXC per clip — report **within 10 seconds of the start**.
        if not os.environ.get("RETARGET_PY"):
            require_retarget_py()
        raise SystemExit(f"[f0_prejudge] --python missing: {a.python}")
    scene = _print_run_markers(a, clips, dst_tree, root, sp, stage_src, wrist_dirs)

    # **Count input completeness at the start**
    # Why: a single missing wrist cache must not surface hours later as a single
    # FAIL_MIRROR line. An upstream hole has to be visible **within 10 seconds**.
    # The default is to warn and continue — that clip ends as FAIL_MIRROR later on, which is a **run failure**:
    # stage2 then fails the chunk instead of recording a verdict. With --require-complete it stops at once.
    _report_upstream_completeness(clips, stage_src, wrist_dirs, a)

    # Fingerprint check / backfill mode — f0 is not run
    if sweep_only:
        return sweep_fingerprints(a, clips, stage_src, wrist_dirs, dst_tree, root, run_id, run_started, sp, scene)
    return run_f0(a, clips, stage_src, wrist_dirs, dst_tree, dst_wrist_dirs, root, run_id, run_started, sp, scene)


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.retarget.f0.prejudge")
    sys.exit(main())
