"""SPIDER + arm rows (spider_arm.data): the separate CPU pass — our arm limits per finished SPIDER tree
(`python -m experiments.policies.spider_arm.data limits ...`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.policies.spider_arm.arm_targets import our_limits
from experiments.policies.spider_arm.spider_load import MARK


def _tree_limits(a, r, P, KP):
    """Load one finished tree (A.6 arm caches, wrist errors, constant idle arm broadcast to T); run our_limits."""
    fl = KP.set_env(r["ds"], r["emb"])
    combo = f"rby1_wuji2{fl['variant']}"
    td = P.trees_dir(combo, r["ds"], r["emb"]) / r["tree"]
    e = json.loads((td / "task_info.json").read_text())["e5_spider"]
    ad = json.loads((P.placement_dir(combo, r["ds"]) / f"{r['tree']}{P.PLACEMENT_SFX}").read_text())
    q, pe = {}, {}
    for s in e["spider_sides"]:
        z = np.load(P.ik_dir(combo, s, r["ds"]) / f"{r['tree']}.npz")
        q[s] = np.asarray(z["arm_joint_positions_all"], np.float64)
        pe[s] = np.asarray(z["wrist_pos_error_all"], np.float64) * 1e3
    T = len(next(iter(q.values())))
    for s in [x for x in ("right", "left") if x not in e["spider_sides"]]:
        f = P.ik_dir(combo, s, r["ds"]) / f"{r['tree']}.npz"
        if not f.is_file() and a.idle_arm_root:
            f = Path(a.idle_arm_root) / "precompute" / r["ds"] / "robots" / combo / f"{f.parent.name}" / f.name
        qa = np.asarray(np.load(f)["arm_joint_positions_all"], np.float64)
        if np.abs(qa - qa[0]).max() > 0:
            raise ValueError(f"idle arm {s} cache is not constant ({f})")
        q[s] = np.repeat(qa[:1], T, axis=0)
    lim = our_limits("rby1", "wuji2", td, q, pe, float(ad["robot_z_off"]), e["spider_sides"])
    return lim


def _print_pass(r, lim):
    """Print the pass line of one tree (penetration and per-side tracking, jump, swivel)."""
    print(
        f"{MARK} [SPIDER_LIMITS] {r['tree']} pass {lim['passes']} · penetration "
        f"{lim['pen_ok']} {lim['pen_reason'][:60]} · "
        + " · ".join(
            f"{s} tracking {v['track_p99_mm']} mm jump {v['jump_max_rad']} rad swivel {v['swivel_deg_05s']}°"
            for s, v in lim["per_side"].items()
        ),
        flush=True,
    )


# ── separate CPU pass: our arm limits per finished SPIDER
# tree
# ────────────────────────────────────────────────────────
def limits_main(argv=None) -> int:
    """python -m experiments.policies.spider_arm.data limits --trees <tsv ds emb variant> --out <jsonl> [--part k/n]
    [--idle-arm-root <kin root>]
    (DEXENGINE_MAIN_ROOT = the root). Reads the tree's arm caches (A.6 solution · kin idle arm) and A.6 wrist errors,
    calls
    our_limits. A render root keeps only the arm caches the render reads — a one-hand tree has no idle-arm
    cache
    there; with --idle-arm-root that side's arm = the kin cache of the same tree under that root (the file prep
    broadcast;
    asserted constant, broadcast to T). Without it a missing cache fails the tree. `variant` = the name the render rows
    carry (col 4 of a kin piece row); its tree dir = variant + paths.TREE_SFX. One line per tree,
    end line [SPIDER_LIMITS] DONE ok=<n> fail=<n>."""
    import argparse  # noqa: PLC0415
    import csv  # noqa: PLC0415
    import traceback  # noqa: PLC0415

    from dexengine.core import paths as P  # noqa: PLC0415
    from experiments.policies.kin_data import prep_shadow as KP  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--idle-arm-root", default="")
    a = ap.parse_args(argv)
    k, n = (int(x) for x in a.part.split("/"))
    with open(a.trees) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if rows and set(rows[0]) != {"ds", "emb", "variant"}:
        raise SystemExit(f"{MARK} [SPIDER_LIMITS] --trees header must be ds emb variant (got {sorted(rows[0])})")
    for r in rows:
        r["tree"] = r["variant"] + P.TREE_SFX
    rows = rows[k::n]
    ok = fail = 0
    with open(a.out, "a") as fh:
        for r in rows:
            try:
                lim = _tree_limits(a, r, P, KP)
                d = {
                    "ds": r["ds"],
                    "emb": r["emb"],
                    "variant": r["variant"],
                    "tree": r["tree"],
                    "ok": True,
                    "our_limits": lim,
                }
                ok += 1
                _print_pass(r, lim)
            except Exception as ex:  # noqa: BLE001 — counted; the end line says it
                fail += 1
                d = {
                    "ds": r["ds"],
                    "emb": r["emb"],
                    "variant": r["variant"],
                    "tree": r["tree"],
                    "ok": False,
                    "err": f"{type(ex).__name__}: {ex}"[:400],
                    "tb": traceback.format_exc()[-3000:],
                }
                print(f"{MARK} [SPIDER_LIMITS] {r['tree']} FAIL {d['err'][:200]}", flush=True)
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"{MARK} [SPIDER_LIMITS] DONE ok={ok} fail={fail} total={len(rows)}", flush=True)
    return 0 if fail == 0 else 3
