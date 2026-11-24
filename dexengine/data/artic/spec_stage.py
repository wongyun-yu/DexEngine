#!/usr/bin/env python3
"""Articulation spec stage: spec update → USD bake of changed (or unfinished) assets → cooked hull dump → surface points
→
check. Owns this chain. Stage 1 step 2b (oakink2.artic_convert --write, canonicalisation) calls ensure **before**
converting, so rebuilding a dataset makes the spec cover its trees by itself (no manual bootstrap).

  1 spec     update. OakInk2 = source record section (dexengine.data.artic.range_src: hand-independent, measured once
  and fixed; the
             hand's trees are only compared) · arctic = tree joints (published statements fixed, windows never
             shrink). Nothing is written when nothing changed. [ARTIC_SRC] · [ARTIC_SPEC].
  2 targets  changed assets (new, widened window) ∪ unfinished assets (_unfinished, by state).
  3 bake     objects/usd_bake.py --batch --articulated --ids_file --decompose_missing (Isaac); rewrites target USDs.
  4 dump     dexengine/robots/objects/cooked_hull.py --ids_file (Isaac); rewrites dumps whose USD sha changed.
  5 surface  dexengine/data/artic/surface.py --ids --force (stale = dexengine.data.artic.surface.surface_problem).
  6 check    (without Isaac) every spec asset unfinished 0 · cooked dumps current (dexengine.robots.objects.cooked_hull
  --verify-only).
             Any mismatch fails (2b does not convert on top of it).
Isaac entry = ISAAC_PY of scripts/_isaac_env.sh (owner) · device = the first device of the stage-1 GPU list
DEXENGINE_F0_GPUS (the same list as f0; step 7 and this stage are the only GPU users in stage 1 and run in turn). With
zero targets Isaac is not started.
Marker: [ARTIC_ASSETS] ds=… targets N (changed a · unfinished b) · bake/dump/surface rc · check result.
Usage: python3 dexengine/data/artic/spec_stage.py --dataset oakink2 [--dry-run] [--remeasure]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from dexengine.core import paths as P  # noqa: E402
from dexengine.data.artic.spec_collect import PHYS_ONLY, update
from dexengine.data.oakink2 import (
    fixed_merge as FM,  # noqa: E402 — owner of rigid-joint assets (merged mesh, pose assignment)
)

REPO = Path(__file__).resolve().parents[3]
GPU_ENV = "DEXENGINE_F0_GPUS"
TIMEOUT_S = 3600  # limit for one Isaac run (bake, dump); exceeding it fails
TAG = "[ARTIC_ASSETS]"


def _unfinished(ds: str, objs: dict) -> list[str]:
    """Unfinished spec asset ids: no USD · stale cooked dump · stale surface points (counted by state)."""
    from dexengine.data.artic import surface as AS  # noqa: PLC0415
    from dexengine.data.objects import hull as OH  # noqa: PLC0415 — USD path accessor (artifact_paths)

    out = []
    for k, v in objs.items():
        if v.get("dataset") != ds:
            continue
        asset = k.split("/", 1)[1]
        usd = OH.usd_file(ds, asset, True)
        surf = P.layer_path("meshes", ds) / asset / "surface_points.npz"
        if not usd.is_file() or OH.cooked_problem(usd) or AS.surface_problem(v, surf):
            out.append(asset)
    return sorted(out)


def _fixed_missing(ds: str, fixed: dict) -> list[str]:
    """Rigid-joint asset ids missing any of merged mesh · rigid USD · cooked dump (only for oakink2)."""
    if ds != "oakink2":
        return []
    from dexengine.data.objects import decompose as DO  # noqa: PLC0415 — object layer root accessor
    from dexengine.data.objects import hull as OH  # noqa: PLC0415

    root = DO.objects_root(ds)
    return sorted(
        a
        for a in fixed
        if not (root / a / "visual.obj").is_file()
        or not OH.usd_file(ds, a, False).is_file()
        or OH.cooked_problem(OH.usd_file(ds, a, False))
    )


def _build_fixed(ds: str, fixed: dict, ids: list[str]) -> dict:
    """Rigid-joint assets: merged mesh → rigid bake (with decomposition, the rigid path as is) → cooked dump. Any rc≠0
    fails."""
    from dexengine.data.objects import decompose as DO  # noqa: PLC0415

    gpu = (os.environ.get(GPU_ENV) or "").split(",")[0].strip()
    if not gpu:
        raise SystemExit(f"{TAG} ✗ ds={ds}: {len(ids)} rigid-joint assets — GPU device ({GPU_ENV}) is empty")
    for a in ids:
        FM.build_mesh(DO.objects_root(ds), a, fixed[a])
    d = _log_dir()
    ids_file = d / f"{ds}_fixed_ids.txt"
    ids_file.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    rcs = {}
    for name, argv in (
        (
            "rigid bake",
            [
                "dexengine/robots/objects/usd_bake.py",
                "--batch",
                "--dataset",
                ds,
                "--ids_file",
                str(ids_file),
                "--decompose_missing",
                "--headless",
            ],
        ),
        (
            "rigid dump",
            ["dexengine/robots/objects/cooked_hull.py", "--dataset", ds, "--ids_file", str(ids_file), "--headless"],
        ),
    ):
        log = d / f"{ds}_fixed_{argv[0].rsplit('/', 1)[-1].removesuffix('.py')}.log"
        rcs[name] = _run(argv, log, gpu)
        print(f"{TAG} ds={ds} {name} rc={rcs[name]} · rigid-joint assets {len(ids)} · log {log}", flush=True)
        if rcs[name] != 0:
            raise SystemExit(f"{TAG} ✗ ds={ds} {name} failed rc={rcs[name]} — {log}")
    left = _fixed_missing(ds, {a: fixed[a] for a in ids})
    if left:
        raise SystemExit(
            f"{TAG} ✗ ds={ds}: rigid-joint assets still missing mesh · USD · cooked dump after the bake {left[:5]}"
        )
    return rcs


def _log_dir() -> Path:
    from dexengine.data import rules as RR  # noqa: PLC0415

    d = P.run_path("run_log", RR.active_hand(), tool="artic_spec_stage")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run(argv: list[str], log: Path, isaac_gpu: str = "") -> int:
    """One step: output goes to a log file (so Kit output does not flood the 2b log) · limit TIMEOUT_S · returns rc."""
    if isaac_gpu:
        cmd = f'source scripts/_isaac_env.sh && CUDA_VISIBLE_DEVICES={shlex.quote(isaac_gpu)} "$ISAAC_PY" ' + " ".join(
            shlex.quote(a) for a in argv
        )
        argv = ["bash", "-c", cmd]
    else:
        argv = [sys.executable, *argv]
    env = {**os.environ, "DEXENGINE_ISAAC_ENV_QUIET": "1"}
    with open(log, "w", encoding="utf-8") as fh:
        try:
            return subprocess.run(
                argv, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=TIMEOUT_S
            ).returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n{TAG} timeout {TIMEOUT_S}s\n")
            return 124


def _build(ds: str, ids: list[str]) -> dict:
    """Bake → dump → surface; stops and fails at the first rc≠0."""
    gpu = (os.environ.get(GPU_ENV) or "").split(",")[0].strip()
    if not gpu:
        raise SystemExit(f"{TAG} ✗ ds={ds}: {len(ids)} assets to bake — GPU device ({GPU_ENV}) is empty")
    d = _log_dir()
    ids_file = d / f"{ds}_ids.txt"
    ids_file.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    steps = [
        (
            "bake",
            [
                "dexengine/robots/objects/usd_bake.py",
                "--batch",
                "--articulated",
                "--dataset",
                ds,
                "--ids_file",
                str(ids_file),
                "--decompose_missing",
                "--headless",
            ],
            gpu,
        ),
        (
            "dump",
            ["dexengine/robots/objects/cooked_hull.py", "--dataset", ds, "--ids_file", str(ids_file), "--headless"],
            gpu,
        ),
        ("surface", ["dexengine/data/artic/surface.py", "--dataset", ds, "--ids", ",".join(ids), "--force"], ""),
    ]
    rcs = {}
    for name, argv, g in steps:
        log = d / f"{ds}_{argv[0].rsplit('/', 1)[-1].removesuffix('.py')}.log"
        rcs[name] = _run(argv, log, g)
        print(f"{TAG} ds={ds} {name} rc={rcs[name]} · assets {len(ids)} · log {log}", flush=True)
        if rcs[name] != 0:
            raise SystemExit(f"{TAG} ✗ ds={ds} {name} failed rc={rcs[name]} — {log}")
    return rcs


def _verify(ds: str, objs: dict) -> None:
    """Every spec asset: unfinished 0 · cooked dumps current (without Isaac). A mismatch fails."""
    miss = _unfinished(ds, objs)
    if miss:
        raise SystemExit(f"{TAG} ✗ ds={ds}: unfinished spec assets {len(miss)} {miss[:5]}")
    ids = sorted(k.split("/", 1)[1] for k, v in objs.items() if v.get("dataset") == ds)
    d = _log_dir()
    ids_file = d / f"{ds}_verify_ids.txt"
    ids_file.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    log = d / f"{ds}_cooked_verify.log"
    rc = _run(
        ["dexengine/robots/objects/cooked_hull.py", "--dataset", ds, "--ids_file", str(ids_file), "--verify-only"], log
    )
    line = next((ln for ln in log.read_text(encoding="utf-8").splitlines() if "[COOKED_VERIFY] usd=" in ln), "")
    if rc != 0:
        raise SystemExit(f"{TAG} ✗ ds={ds}: cooked dump check rc={rc} {line} — {log}")
    print(
        f"{TAG} ds={ds} check passed · spec assets {len(ids)} · USD and surface points present · {line.strip()}",
        flush=True,
    )


def ensure(ds: str, write: bool = True, remeasure: bool = False) -> dict:
    """One run of the spec stage (module docstring 1–6). Returns {changed, targets, rcs}. write=False = collect, merge
    and
    compute targets only · remeasure = measure the OakInk2 source records again (update; explicit choice only)."""
    r = update(write=write, remeasure=remeasure)
    n_new = sum(v == "new" for v in r["changed"].values())
    n_drop = sum(v.startswith("dropped") for v in r["changed"].values())
    n_phys = sum(v == PHYS_ONLY for v in r["changed"].values())
    print(
        f"[ARTIC_SPEC] hands {len(r['stats'])} · trees {sum(v[0] for v in r['stats'].values())} · tree fits "
        f"{sum(v[1] for v in r['stats'].values())} · fit failures {sum(len(v[2]) for v in r['stats'].values())} · "
        f"new assets {n_new} · removed (disagrees with spec kind) {n_drop} · window changed (widened · "
        "narrowed by physical range) "
        f"{len(r['changed']) - n_new - n_drop - n_phys} · physical range only {n_phys} · "
        f"written {'yes' if r['wrote'] else 'no'} · "
        f"sha {r['sha']}",
        flush=True,
    )
    for k, why in sorted(r["changed"].items()):
        print(f"[ARTIC_SPEC]   {k}: {why}", flush=True)
    fx_pre = f"{ds}/{FM.SECTION}/"
    changed = sorted(
        k.split("/", 1)[1]
        for k, v in r["changed"].items()
        if k.startswith(f"{ds}/") and not k.startswith(fx_pre) and not v.startswith("dropped") and v != PHYS_ONLY
    )
    missing = _unfinished(ds, r["objects"])
    targets = sorted(set(changed) | set(missing))
    fx_targets = sorted(
        {k[len(fx_pre) :] for k in r["changed"] if k.startswith(fx_pre)} | set(_fixed_missing(ds, r["fixed"]))
    )
    print(
        f"{TAG} ds={ds} targets {len(targets)}(changed {len(changed)} · unfinished {len(missing)}) · "
        "rigid-joint assets "
        f"{len(r['fixed']) if ds == 'oakink2' else 0} · targets {len(fx_targets)}",
        flush=True,
    )
    if not write:
        return {"changed": r["changed"], "targets": targets, "fixed_targets": fx_targets, "rcs": {}}
    rcs = _build(ds, targets) if targets else {}
    if fx_targets:
        rcs.update(_build_fixed(ds, r["fixed"], fx_targets))
    _verify(ds, r["objects"])
    return {"changed": r["changed"], "targets": targets, "fixed_targets": fx_targets, "rcs": rcs}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="articulation spec update · asset (USD · cooked dump · surface points) bake · check"
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dry-run", action="store_true", help="collect, merge and targets only; neither writes nor bakes")
    ap.add_argument(
        "--remeasure",
        action="store_true",
        help="measure the OakInk2 source records again (only when the sources changed; converted trees of "
        "assets whose statement changed are rebuilt)",
    )
    a = ap.parse_args()
    ensure(a.dataset, write=not a.dry_run, remeasure=a.remeasure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
