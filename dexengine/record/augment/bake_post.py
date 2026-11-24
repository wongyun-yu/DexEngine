"""Placement augment: output gates of one variant after the bake steps (augment.bake) and the worker command line
(called by augment.place_gen and augment.bake).

Contacts (remap_contacts): variant T differs from canonical -> rows of the canonical contact_mano_<side>.npz move to
  the canonical frame closest to each variant frame's source time (retime_src_idx) and are written as real files
  (record phases require rows = kinematic rows; contact flags and distances are invariant under rigid moves; witness
  point coordinates keep the canonical values, the same content as the old symlinks).
Path oracle (canon_oracle, bake reproduction check): the pv00 (identity + bake seed) outputs (arm cache, T,
  retime_tau, tree qpos) are compared with canonical and **only recorded, never cut** (bake reproduction differences
  are filtered by the rollout success decision; rollouts on every GPU). Differences (qpos, arm, tau max, T diff, 2π
  wrap count) and within/outside tolerance (CANON_TOL · CANON_TAU_REL, a split for the record) go to the variant
  record (variants.json canon_oracle) with one marker line [CANON_ORACLE] record only (not cut). With a different T
  the overlapping front rows are measured.
Placement (place_gate): the chain's place_rule.baked_check (audit, the same function as the bake).
Worker (worker, command line stamp | place): the support stamp and placement find the base layer by tree name
  (alias_base) -> they run as Isaac python child processes of the bake env whose data root is the shadow · one JSON
  per line · count check.
Markers: worker lines [AUG_BAKE_WORKER] {json}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dexengine.core.errors import input_missing
from dexengine.record.augment.bake import ROOT, VARIANT_S, bake_env, die, isaac_py

WORKER_TAG = "[AUG_BAKE_WORKER] "
#: path oracle record split (rad · max arm joint difference · max tree qpos difference): within/outside tolerance is
#: only written to the record, never cut.
CANON_TOL = 1e-2  # rad(0.57°)
CANON_TAU_REL = 1e-5  # max retime_tau difference <= CANON_TAU_REL · max(1, τ end): the same decision


def worker(env: dict, cmd: str, args: list[str], log_dir: Path) -> list[dict]:
    """This file's command line (stamp | place) through the bake env's Isaac python: one JSON per line (marker
    WORKER_TAG) -> that list. rc != 0 or lines != arguments die."""
    log = Path(log_dir) / f"worker_{cmd}.log"
    with open(log, "w") as f:
        r = subprocess.run(
            [isaac_py(env), str(Path(__file__).resolve()), cmd, *args],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=f,
            text=True,
            timeout=VARIANT_S,
        )
    with open(log, "a") as f:  # step markers ([DESK_SUPPORT] · [AERIAL_SUPPORT] …) stay in the record too
        f.write(r.stdout)
    got = [json.loads(ln[len(WORKER_TAG) :]) for ln in r.stdout.splitlines() if ln.startswith(WORKER_TAG)]
    n_want = len([x for x in args if not x.startswith("--")]) - (1 if "--arm" in args else 0)
    if r.returncode or len(got) != n_want:
        die(f"{cmd} worker rc={r.returncode} · results {len(got)}/{n_want}: {log}")
    return got


def place_gate(src, shadow: Path, recs: list, bake: dict) -> None:
    """The chain's placement after bake (place_rule.baked_check, audit, the same function as the bake): only variants
    with status ok, in the bake env."""
    todo = [r for r in recs if r["status"] == "ok"]
    if not todo:
        return
    troot = Path(shadow) / src.trees_rel
    log_dir = Path(bake.get("log_dir") or shadow)
    got = worker(
        bake_env(src, bake["active"], shadow),
        "place",
        ["--arm", src.arm] + [str(troot / r["name"]) for r in todo],
        log_dir,
    )
    by = {g["tree"]: g for g in got}
    for r in todo:
        g = by[str(troot / r["name"])]
        r["place"] = {
            "why": g["why"],
            "short_mm": None if g["short_m"] is None else round(float(g["short_m"]) * 1000, 2),
        }
        if g["why"]:
            r.update(status="reject", why="place", detail=g["why"])


# ── contacts · oracle ───────────────────────────────────────────
def nearest_frames(src_idx_canon, src_idx_var) -> np.ndarray:
    """Variant frame -> the canonical frame with the closest source time (both retime_src_idx, monotonic)."""
    a = np.asarray(src_idx_canon, np.float64)
    b = np.asarray(src_idx_var, np.float64)
    i = np.clip(np.searchsorted(a, b), 1, len(a) - 1)
    return np.where(np.abs(a[i - 1] - b) <= np.abs(a[i] - b), i - 1, i)


def remap_contacts(src, tree: Path) -> list[str]:
    """Canonical contact_mano_<side>.npz -> variant tree time axis (module doc). Canonical rows != canonical T dies."""
    kin = np.load(Path(tree) / "0/trajectory_kinematic.npz")
    Tc = len(src.kin["qpos"])
    has = ("retime_src_idx" in src.kin, "retime_src_idx" in kin.files)
    if has == (False, False):
        # trees without retiming (E5 kinematics: original and variant both on the human time axis): time map =
        # identity; lengths must match
        if len(kin["qpos"]) != Tc:
            die(f"{tree}: T {len(kin['qpos'])} of a variant without retiming != canonical T {Tc} (no identity map)")
        j = np.arange(Tc)
        print(f"[REMAP_CONTACTS] {Path(tree).name} identity time axis (no retime_src_idx on either) T {Tc}", flush=True)
    elif has != (True, True):
        die(
            input_missing(
                f"{tree}: retime_src_idx on one side only (canonical {has[0]} · variant {has[1]}): no time map"
            )
        )
    else:
        j = nearest_frames(src.kin["retime_src_idx"], kin["retime_src_idx"])
    done = []
    for f in sorted((src.tree / "0").glob("contact_mano_*.npz")):
        if f.name.count(".") > 1:  # side files (.pre_rt etc.)
            continue
        cz = dict(np.load(f, allow_pickle=False))
        rows = [k for k, x in cz.items() if np.ndim(x) >= 1 and np.shape(x)[0] == Tc]
        if "contact" not in rows:
            die(f"{f}: contact rows != canonical T {Tc} (time axis)")
        for k in rows:
            cz[k] = np.asarray(cz[k])[j]
        dst = Path(tree) / "0" / f.name
        if dst.is_symlink():
            dst.unlink()
        np.savez(dst, **cz)
        done.append(f.name)
    return done


def _maxdiff(a, b) -> tuple[float, int]:
    """Max absolute difference of the overlapping front rows · number of 2π wrapped fields (|diff| >= π: another joint
    state). Different column counts give (inf, 0)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    if a.shape[1:] != b.shape[1:] or n == 0:
        return float("inf"), 0
    d = np.abs(a[:n] - b[:n])
    return float(d.max()), int((d >= np.pi).sum())


def canon_oracle(src, shadow: Path, name: str) -> dict:
    """pv00 (identity + bake seed) outputs <-> canonical: T (diff dT) · retime_tau · arm cache · max tree qpos
    difference · 2π wrap count · within/outside tolerance.
    Record only; no verdict (cut is always False; the rollout success decision filters)."""
    tree = Path(shadow) / src.trees_rel / name
    kin = np.load(tree / "0/trajectory_kinematic.npz")
    o = {"T": [int(len(kin["qpos"])), int(len(src.kin["qpos"]))], "cut": False}
    o["dT"] = o["T"][0] - o["T"][1]
    o["tau"], _w = _maxdiff(np.asarray(kin["retime_tau"])[:, None], np.asarray(src.kin["retime_tau"])[:, None])
    o["qpos"], wq = _maxdiff(kin["qpos"], src.kin["qpos"])
    o["arm"], wrap = {}, wq
    for s in ("right", "left"):
        a = np.load(Path(shadow) / src.ik_rel[s] / f"{name}.npz")["arm_joint_positions_all"]
        b = np.load(src.main / src.ik_rel[s] / f"{src.tree_name}.npz")["arm_joint_positions_all"]
        o["arm"][s], w = _maxdiff(a, b)
        wrap += w
    o["wrap2pi"] = wrap
    o["worst"] = max([o["qpos"], *o["arm"].values()])
    o["within_tol"] = bool(
        o["dT"] == 0
        and o["worst"] <= CANON_TOL
        and o["tau"] <= CANON_TAU_REL * max(1.0, float(src.kin["retime_tau"][-1]))
    )
    o["why"] = (
        ""
        if o["within_tol"]
        else (f"T {o['T'][0]} != canonical {o['T'][1]} · " if o["dT"] else "")
        + f"max diff {o['worst']:.2e} rad (tolerance {CANON_TOL}) · tau {o['tau']:.2e} · 2π wraps {wrap}"
    )
    return o


def record_oracle(rec: dict, o: dict) -> None:
    """Puts the bake reproduction check result into the variant record with one marker line; status and why are not
    touched (never cut)."""
    rec["canon_oracle"] = o
    print(
        f"[CANON_ORACLE] record only (not cut) {rec.get('name')} · {'within' if o['within_tol'] else 'outside'} "
        f"tolerance · T {o['T'][0]}/{o['T'][1]} (dT {o['dT']:+d}) · qpos {o['qpos']:.2e} · arm R "
        f"{o['arm']['right']:.2e} L {o['arm']['left']:.2e} · tau {o['tau']:.2e} · 2π wraps {o['wrap2pi']}",
        flush=True,
    )


def main(argv: list[str]) -> int:
    """Worker command line: stamp <tree>…: coords.stamp_scene (keep_height) · place --arm A <tree>…:
    place_rule.baked_check. The data root is set by the caller's env (the shadow). Unknown commands or arguments of
    another form are killed by argparse."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stamp", help="support restamp (coords.stamp_scene keep_height)").add_argument("trees", nargs="+")
    pl = sub.add_parser("place", help="placement audit (place_rule.baked_check)")
    pl.add_argument("--arm", required=True)
    pl.add_argument("trees", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "stamp":
        from dexengine.placement.coords import core as coords

        for t in a.trees:
            st = coords.stamp_scene(t, keep_height=True)
            disc = [[round(float(d["x"]), 4), round(float(d["y"]), 4)] for d in (st or {}).get("discs") or []]
            print(WORKER_TAG + json.dumps({"tree": t, "discs": disc}), flush=True)
        return 0
    from dexengine.placement.scene import place_rule as PR

    for t in a.trees:
        why, short = PR.baked_check(t, arm=a.arm)
        print(
            WORKER_TAG
            + json.dumps(
                {"tree": t, "why": why, "short_m": None if short is None else float(short)}, ensure_ascii=False
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
