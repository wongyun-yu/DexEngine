#!/usr/bin/env python3
"""Stage-1 retargeting with the active hand: align the retargeting start line of every dataset to the hand that runs.

  taco       generate-xml + ik over the TACO retargeting dataset (``…/processed/taco/mano/bimanual``) after the
             hand-side check (``taco_side_assign --check-processed``: the MANO layer's per-hand objects are the
             measured assignment)
  <dataset>  a stage under ``paths.retarget_stage_root(<hand>)`` whose inputs are links into the delivered dataset
             (``data/<ds>/{mano/<emb>/<clip>, assets/objects}``, marked by ``.stage_links``) plus a copy of the
             canonical rig; generate-xml + ik there, then the stage ``processed/MANIFEST.sha256`` (the ingest reads it)
  hg | all   every registered dataset but taco | taco and every other dataset
Each (dataset, emb) is counted by rule (``reretarget_count``): every clip without output must be explained by the
exclusion ledger (``precompute/<ds>/diag/excluded_clips.txt``, read through ``datasets.retarget_excluded``);
expected = input − explained missing; an unexplained missing clip or a count mismatch makes the run end with rc 4 (it
goes on past failing clips).

Usage:
    python3 -m dexengine.retarget.reretarget [taco|hg|all|<dataset>] [WORKERS] [--hand H] [--scope FILE]
        [--manifest-only | --unstage] [--no-manifest]
  --hand           hand id (default: the active hand, env DEXENGINE_HAND — ``datasets.active_hand``)
  --scope FILE     lines ``<ds>\\t<emb>/<clip>``: retarget only the parent takes of the listed clips and count
                   within them (``paths.scope_takes``)
  --manifest-only  rewrite that one dataset's stage manifest from the current files, then stop
  --unstage        remove that one (non-taco) dataset's stage input links; all of them must point into the delivered
                   dataset, otherwise none is removed (rc 2)
  --no-manifest    do not write the manifest after retargeting (the caller writes it right before ingest)
Heavy: run it under a CPU cap. Re-runnable: ``--skip-existing`` reuses a solve only when its input fingerprint is the
same (``dexengine/retarget/input_fp.py``). One run per (hand, dataset) on a machine (a lock in the log folder).
rc 0 · 2 = bad input or a guard refused · 4 = a count did not match.

Lines the stage-1 driver parses are built from the constants of ``reretarget_stage``: SCOPE (the
requested scope, with MANIFEST_ONLY or UNSTAGE for those modes) and the count line ``… input N − explained n =
expected m → output k`` that ends in OK or MISMATCH.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from dexengine.core import config, layout
from dexengine.core import paths as P
from dexengine.core.repo import REPO_ROOT
from dexengine.data import rules as R
from dexengine.retarget.reretarget_count import count_check
from dexengine.retarget.reretarget_stage import (
    MANIFEST_ONLY,
    SCOPE,
    die,
    ds_src,
    ls_visible,
    read_lines,
    scope_takes,
    unstage,
    write_manifest,
    write_scope,
)

EMBS = ("right", "left", "bimanual")
_LOCKS: list = []  # lock files held until the process ends


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO_ROOT)}


# retargeting and the count rule
def ds_lock(ds: str, log: Path) -> None:
    """No two runs of the same (hand, dataset) on one machine: they share the rig copy, the scope files and the logs."""
    fh = open(log / f".lock_{ds}", "w")  # noqa: SIM115 — the flock lives as long as the process (_LOCKS)
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        die(f"[reretarget] ✗ {ds}: another run holds it ({log}/.lock_{ds})")
    _LOCKS.append(fh)
    print(f"[reretarget] lock {ds} (pid {os.getpid()})", flush=True)


def stamp_inval(ds: str) -> None:
    """Before writing: drop the stage stamps that do not match the current rig and prior (owner data/stages.py)."""
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "dexengine/data/stages.py"), "--invalidate-stage-stamp", "--datasets", ds],
        stdout=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
        env=_env(),
    )
    marks = [ln for ln in r.stdout.splitlines() if ln.startswith("[STAGE_STAMP]")]
    if r.returncode or not marks:
        die(f"[reretarget] ✗ {ds}: stage stamp check failed")
    print("\n".join(marks), flush=True)


def run_phase(dd: Path, ds: str, emb: str, ctx, scope_cur: Path | None) -> bool:
    """generate-xml then ik over one (dataset, emb). A failing clip does not stop the run (the count check names it);
    a phase that had clips but printed no ``[SKIP_EXISTING]`` line makes the run invalid (returns False)."""
    ok = True
    for ph in ("generate-xml", "ik"):
        print(f"[reretarget {time.strftime('%H:%M:%S')}] {ds}/{emb} {ph} start (workers {ctx.workers})", flush=True)
        cmd = [
            ctx.py,
            "-m",
            "dexengine.retarget.batch",
            "--dataset-dir",
            str(dd),
            "--dataset-name",
            ds,
            "--robot-type",
            ctx.spec,
            "--embodiment-type",
            emb,
            "--phase",
            ph,
            "--workers",
            str(ctx.workers),
            "--skip-existing",
        ] + (["--clip-list", str(scope_cur)] if scope_cur else [])
        logf = ctx.log / f"{ds}_{emb}_{ph}.log"
        with open(logf, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=_env()).returncode
        if rc:
            print(
                f"[reretarget] ★ {ds}/{emb} {ph} has failing clips (the count check below names them by set "
                "difference)",
                flush=True,
            )
        text = logf.read_text(encoding="utf-8", errors="replace")
        for ln in [ln for ln in text.splitlines() if "completed!" in ln or "Found" in ln][-2:]:
            print(ln)
        found = re.findall(r"Found ([0-9]+) processed clips", text)
        skip = re.findall(rf"\[SKIP_EXISTING\] phase={re.escape(ph)} .*", text)
        if skip:
            print(f"[reretarget] {ds}/{emb} {skip[-1]}", flush=True)
        elif (found[-1] if found else "0") != "0":
            ok = False
            print(
                f"[reretarget] ✗ {ds}/{emb} {ph}: no [SKIP_EXISTING] line of the skip rule (clips {found[-1]}) — "
                "the reuse verdict cannot be trusted",
                file=sys.stderr,
                flush=True,
            )
    return ok


# the two branches
def retarget_taco(ctx, scope: Path | None) -> bool:
    ds_lock("taco", ctx.log)
    stamp_inval("taco")
    mano = ctx.taco_ds / "processed/taco/mano/bimanual"
    r = subprocess.run(
        [
            ctx.py,
            str(REPO_ROOT / "dexengine/data/readers/taco_side_assign.py"),
            "--raw",
            str(ctx.taco_ds / "raw/taco"),
            "--check-processed",
            str(mano),
        ],
        cwd=REPO_ROOT,
        env=_env(),
    )
    if r.returncode:
        die(
            "[reretarget] ✗ the hand sides of the TACO MANO layer differ from the measured assignment or are missing "
            "— run python -m dexengine.retarget.batch --phase convert (with its producer) → kinematic → decompose → "
            "detect-contact first"
        )
    scope_cur = None
    if scope:
        if scope_takes(scope, "taco") != ["bimanual"]:
            die("[reretarget] ✗ the taco scope emb is not bimanual alone")
        scope_cur = ctx.log / "taco_bimanual.scope"
        write_scope(scope_cur, scope_takes(scope, "taco", "bimanual", mano))
        print(f"[RERETARGET_SCOPE] taco/bimanual {sum(1 for t in read_lines(scope_cur) if t)}/{len(ls_visible(mano))}")
    ok = run_phase(ctx.taco_ds, "taco", "bimanual", ctx, scope_cur)
    return count_check(mano, ctx.taco_ds, "taco", "bimanual", "TACO bimanual", ctx, scope_cur) and ok


def link_inputs(pe: Path, srce: Path, scope_cur: Path | None) -> None:
    """Stage input links of one emb. With a scope the stage is the scope: links into ``srce`` outside it are removed
    and the missing ones in it are made."""
    if scope_cur:
        want = set(read_lines(scope_cur))
        for e in os.scandir(pe):
            if e.is_symlink() and os.readlink(e.path).startswith(f"{srce}/") and e.name not in want:
                os.unlink(e.path)
        for c in read_lines(scope_cur):
            if not os.path.exists(pe / c):
                os.symlink(f"{srce}/{c}", pe / c)
        n = sum(1 for e in os.scandir(pe) if e.is_symlink())
        print(f"[RERETARGET_SCOPE] {pe.parent.parent.name}/{pe.name} stage input links {n}")
        return
    for c in sorted(e.name for e in os.scandir(srce) if not e.name.startswith(".") and e.is_dir()):
        if not os.path.exists(pe / c):
            os.symlink(f"{srce}/{c}", pe / c)


def retarget_dataset(ds: str, ctx, scope: Path | None) -> bool:
    """One non-taco dataset through the stage (module docstring)."""
    ds_lock(ds, ctx.log)
    p = ctx.stage / "processed" / ds
    (p / "mano").mkdir(parents=True, exist_ok=True)
    (p / "assets/robots").mkdir(parents=True, exist_ok=True)
    src = ds_src(ds, ctx.hand)
    for what, d in (
        ("the dataset root", src),
        ("the mano layer (ingest it first)", src / "mano"),
        ("the object assets", src / "assets/objects"),
    ):
        if not d.is_dir():
            die(f"[reretarget] ✗ {ds}: {what} does not exist ({d})")
    sembs: list[str] = []
    if scope:  # scope files per emb before the stage is touched (bad scope = rc 2)
        sembs = scope_takes(scope, ds)
        for e in sembs:
            if not (src / "mano" / e).is_dir():
                die(f"[reretarget] ✗ {ds}: the scope emb {e} is not in the delivered dataset")
            write_scope(ctx.log / f"{ds}_{e}.scope", scope_takes(scope, ds, e, src / "mano" / e))
    stamp_inval(ds)
    # The input links live for the run: stage 1 removes them after ingest and stamps (--unstage); the marker stays
    # until then.
    (p / ".stage_links").write_text(f"src={src} pid={os.getpid()} t={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    if not os.path.exists(p / "assets/objects"):
        os.symlink(str(src / "assets/objects"), p / "assets/objects")
    rig = p / "assets/robots" / ctx.spec
    if os.path.islink(rig) or rig.is_file():
        rig.unlink()
    elif rig.is_dir():
        shutil.rmtree(rig)
    shutil.copytree(P.hand_rig_dir(ctx.spec), rig, symlinks=True)
    ok, seen = True, False
    for emb in EMBS:
        srce = src / "mano" / emb
        if not srce.is_dir():
            continue
        seen, scope_cur = True, None
        if scope:
            if emb not in sembs:
                print(f"[RERETARGET_SCOPE] {ds}/{emb} 0/{len(ls_visible(srce))} — skipped")
                continue
            scope_cur = ctx.log / f"{ds}_{emb}.scope"
            print(f"[RERETARGET_SCOPE] {ds}/{emb} {sum(1 for t in read_lines(scope_cur) if t)}/{len(ls_visible(srce))}")
        pe = p / "mano" / emb
        pe.mkdir(parents=True, exist_ok=True)
        link_inputs(pe, srce, scope_cur)
        ok = run_phase(ctx.stage, ds, emb, ctx, scope_cur) and ok
        ok = count_check(pe, ctx.stage, ds, emb, f"{ds}/{emb}", ctx, scope_cur) and ok
    if not seen:
        die(f"[reretarget] ✗ {ds}: no emb folder ({src}/mano/{{right,left,bimanual}})")
    return ok


def run(a) -> int:
    hand = a.hand or R.active_hand()
    os.environ[R.HAND_ENV] = hand  # the hand owners (stage stamps, exclusion ledger) read it — one hand per run
    one = ""
    if a.what not in ("taco", "hg", "all"):
        if a.what not in R.ALL:
            die(f"[reretarget] ✗ unknown argument: {a.what} (taco|hg|all or one dataset of datasets.ALL)")
        one = a.what
    spec = P.hand_spec_dir(hand)
    ctx = SimpleNamespace(
        hand=hand,
        spec=spec,
        workers=a.workers,
        py=P.retarget_py(),
        taco_ds=P.layer_path("retarget", "taco", hand=spec, emb="bimanual").parents[3],
        stage=P.retarget_stage_root(hand),
    )
    if a.manifest_only:
        if not one:
            die("[reretarget] ✗ --manifest-only takes one dataset as the argument")
        if not (ctx.stage / "processed" / one).is_dir():
            die(f"[reretarget] ✗ {one} is not in the stage: {ctx.stage}/processed/{one}")
        write_manifest(ctx.stage, f"./{one}")
        print(f"{SCOPE}{one} {MANIFEST_ONLY}", flush=True)
        return 0
    if a.unstage:
        if not one:
            die("[reretarget] ✗ --unstage takes one dataset other than taco (taco does not use the stage)")
        return unstage(ctx.stage, one, hand)
    ctx.log = P.run_path("run_log", "" if "run_log" in layout.RUN_SHARED else hand, tool="reretarget")
    ctx.log.mkdir(parents=True, exist_ok=True)
    key = one or a.what
    ds_list = [d for d in R.ALL if d != "taco"] if key in ("hg", "all") else ([] if key == "taco" else [one])
    # The scope line is printed outside both branches, so every run has it (the caller compares it with its request).
    print(f"{SCOPE}{key} list=[{' '.join(ds_list) or 'taco only'}] (requested '{a.what}')", flush=True)
    scope = None
    if a.scope is not None:
        if not a.scope.is_file():
            die(f"[reretarget] ✗ the scope file does not exist: {a.scope}")
        scope = a.scope.resolve()
        print(f"[RERETARGET_SCOPE] on — list {scope}", flush=True)
    ok = True
    if a.what in ("taco", "all"):
        ok = retarget_taco(ctx, scope) and ok
    if ds_list:
        (ctx.stage / "processed").mkdir(parents=True, exist_ok=True)
        for ds in ds_list:
            ok = retarget_dataset(ds, ctx, scope) and ok
        if a.no_manifest:
            print("[reretarget] manifest not written (--no-manifest — stage 1 writes it right before ingest)")
        else:
            write_manifest(ctx.stage, f"./{one}" if one else ".")
    print(f"[reretarget] end {time.strftime('%H:%M:%S')} FAIL={0 if ok else 1}", flush=True)
    if not ok:
        print("[reretarget] ✗ a count did not match — a partial run is not a success", file=sys.stderr)
        return 4
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all", help="taco | hg | all | one dataset (default all)")
    ap.add_argument("workers", nargs="?", type=int, default=8, help="solver workers (default 8)")
    ap.add_argument("--hand", default=None)
    ap.add_argument("--scope", type=Path, default=None)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--manifest-only", action="store_true")
    mode.add_argument("--unstage", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    config.add_data_root_arg(ap)
    a = ap.parse_args(argv)
    config.apply_data_root(a)
    try:
        return run(a)
    except ValueError as e:  # a name the name rule refuses, a bad layer axis …
        print(f"[reretarget] ✗ {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.retarget.reretarget")
    sys.exit(main())
