"""Variant bake run of the placement augment (record.augment.bake owns the inputs, the chain step table, the bake env
and the worker Pool): bake_all bakes every variant of a clip through the Pool, bake_one runs one variant's chain
steps and reads the chain's stop line, clean_outputs clears a variant before a rerun. Part of the augment cache
fingerprint (AUG_CODE_FILES)."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dexengine.record.augment.bake import (
    MARK,
    Pool,
    alias_base,
    bake_env,
    base_kin,
    clip_ctx,
    die,
    seeds,
    step_specs,
    values,
)


# ── one variant ───────────────────────────────────────────────────
def clean_outputs(src, shadow: Path, name: str) -> None:
    """Removes variant outputs (inside the shadow, this variant's name only) so the xr backup reuse
    (wrist_shift.source_of) does not read old outputs."""
    for s in ("right", "left"):
        for d in (Path(shadow) / src.ik_rel[s], Path(shadow) / src.wrist_rel[s]):
            for f in d.glob(f"{name}.*"):
                if f.is_symlink() or not f.is_file():
                    die(f"variant output to remove {f} is not a real file (canonical symlink: name clash)")
                f.unlink()


#: detail of a bake reject without a stop line; otherwise the detail is "<stop line without its head> · <last error>"
NO_STOP_LINE = "no stop line"


def _abort_reason(log: Path) -> str:
    from dexengine.placement.bake.chain import STOP  # the stop-line head (one owner)

    txt = log.read_text(errors="replace") if log.is_file() else ""
    stop = [ln for ln in txt.splitlines() if ln.startswith(STOP)]  # stop line of placement.bake.chain
    why = stop[-1].removeprefix(STOP + " ") if stop else NO_STOP_LINE  # reason = data (variant detail)
    tags = ("[stage_dp]", "[v5retime]", "[xr", "[XR", "[ARM_FLIP", "[DP_PIN]")
    # when a step died with an exception, that exception line is the reason
    err = [ln for ln in txt.splitlines() if re.match(r"^[A-Za-z_.]*(Error|Exception|SystemExit)\b", ln)]
    tail = err or [ln for ln in txt.splitlines() if ln.startswith(tags)]
    return f"{why} · {tail[-1][:200] if tail else ''}".strip(" ·")


def _solver(p: Path) -> str:
    return str(np.load(p, allow_pickle=True).get("solver", "")) if p.is_file() else ""


def bake_one(pool: Pool, src, shadow: Path, rec: dict, seed: dict, xr: bool, v: dict, log_dir: Path) -> dict:
    """One variant through the bake steps -> rec update (status ok | reject · why bake | length · bake record)."""
    from dexengine.placement.bake import chain as SR

    name = rec["name"]
    specs = step_specs(src, shadow, name, seed, xr, v)
    log = Path(log_dir) / f"bake_{name}.log"
    t0 = time.time()
    rc = pool.run(specs, log)
    sec = round(time.time() - t0, 1)
    tree = Path(shadow) / src.trees_rel / name
    ikf = {s: Path(shadow) / src.ik_rel[s] / f"{name}.npz" for s in ("right", "left")}
    b = {
        "rc": rc,
        "sec": sec,
        "steps": len(specs),
        "seed": {s: str(p) for s, p in seed.items()},
        "xr": xr,
        "log": str(log),
    }
    if rc != 0:
        rec.update(status="reject", why="bake", detail=_abort_reason(log), bake=b)
        return rec
    nr, nl, nq = SR.halfwrite_lengths(ikf["right"], ikf["left"], tree / "0/trajectory_kinematic.npz")
    kin = np.load(tree / "0/trajectory_kinematic.npz")
    b.update(
        T=nq,
        T_pre=int(len(kin["retime_tau"])),
        stretch=round(nq / max(len(kin["retime_tau"]), 1), 3),
        solver={s: _solver(ikf[s]) for s in ("right", "left")},
    )
    if min(nr, nl, nq) < 0 or not nr == nl == nq:
        rec.update(status="reject", why="length", detail=f"R={nr} L={nl} qpos={nq}", bake=b)
        return rec
    rec.update(status="baked", bake=b)
    return rec


def _seed_or_reject(sd: dict, r: dict, src, bundle: Path, ctx: dict) -> None:
    """Seeds of one written variant into sd; a canonical cell (k 0) without seeds is a data reject, not a defect."""
    try:
        sd[r["name"]] = seeds(src, int(r["k"]), bundle, ctx)
    except SystemExit as e:
        if int(r["k"]) != 0:
            raise
        # data: no bake (not a reproduction check)
        r.update(status="reject", why="bake", detail=f"canonical cell bake seed missing — {e}")


def bake_all(src, shadow: Path, recs: list[dict], bundle: Path, main: Path, jobs: int | None, log_dir: Path) -> dict:
    """Every variant with status written through the bake steps (a worker pool sized by capacity; `jobs` = an explicit
    option) -> pool record."""
    todo = [r for r in recs if r["status"] == "written"]
    if not todo:
        return {"boot_s": 0.0, "n": 0}
    _z, xr, _f = base_kin(src)
    v = values(src)
    ctx = clip_ctx(src, main)
    sd = {}
    for r in todo:
        _seed_or_reject(sd, r, src, bundle, ctx)
    todo = [r for r in todo if r["status"] == "written"]
    for r in todo:
        clean_outputs(src, shadow, r["name"])
        r["alias"] = alias_base(src, shadow, r["name"])
    env = bake_env(src, ctx["active"], shadow)
    troot = Path(shadow) / src.trees_rel
    from dexengine.record.augment import bake_post as ABP

    ABP.worker(env, "stamp", [str(troot / r["name"]) for r in todo], log_dir)
    from dexengine.core import capacity  # noqa: PLC0415

    with capacity.sized("dexengine.record.augment.bake_run", len(todo), jobs) as cap:
        pool = Pool(cap.workers, log_dir, capacity.child_env(cap.threads, env))
        t0 = time.time()
        try:
            with ThreadPoolExecutor(cap.workers) as ex:
                list(ex.map(lambda r: bake_one(pool, src, shadow, r, sd[r["name"]], xr, v, log_dir), todo))
        finally:
            pool.stop()
    wall = round(time.time() - t0, 1)
    n = len(todo)
    nb = sum(r["status"] == "baked" for r in todo)
    print(
        f"{MARK} {src.tree_name}: variants {n} baked · passed {nb} · wall clock {wall} s · "
        f"boot {pool.boot_s} s · xr={xr} · values {v}",
        flush=True,
    )
    return {
        "boot_s": pool.boot_s,
        "wall_s": wall,
        "n": n,
        "xr": xr,
        "values": v,
        "chunk": ctx["tag"],
        "active": ctx["active"],
        "log_dir": str(log_dir),
    }
