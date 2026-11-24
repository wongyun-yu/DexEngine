#!/usr/bin/env python3
"""E5 kinematic data — per cell (task × dataset × hand use): pool clips passing reach, target episodes and shortfall.

Hand use rule (one kinematic tree — prep.py's active hands and the tree's main object):
  one active hand → uni-R | uni-L · two → one main object (obj_id of task_info objects role=main) = bi-shared, more
  than one = bi-2obj. (The comparison with our clip table goes in the table header — ours is based on our trees, so a
  few clips differ.)
Target = our training manifest (--targets counts.tsv, episodes per cell) · pass = reach.py (source placement) · one
clip at most look 5 × (1 source + augment policy.PER_CLIP variants) episodes — variants must pass reach too, so the
real number can be lower (counted when drawn). Output: --out-tsv (cell table) · --clips-tsv (per clip: cell, reach,
pass) · stdout = markdown.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

MARK = "[KIN_CELLS]"
LOOKS = 5


def hand_use(active: list[str], ti: dict) -> str:
    if len(active) == 1:
        return "uni-R" if active[0] == "right" else "uni-L"
    mains = {str(o.get("obj_id")) for o in ti.get("objects") or [] if o.get("role") == "main"}
    return "bi-shared" if len(mains) == 1 else "bi-2obj"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="kinematic_pool.tsv")
    ap.add_argument("--prep", required=True, action="append", help="kin_prep result jsonl glob (repeatable)")
    ap.add_argument("--root", required=True, help="kinematic data root (tree task_info)")
    ap.add_argument("--targets", required=True, help="our training manifest counts.tsv")
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--clips-tsv", required=True)
    return ap.parse_args()


def _read_prep_results(prep_globs: list[str]) -> dict:
    """Successful experiments.policies.kin_data.prep results keyed by (dataset, clip), from every jsonl matching the
    globs."""
    res = {}
    files = sorted({f for g in prep_globs for f in glob.glob(g)})
    for f in files:
        with open(f) as fh:
            for ln in fh:
                if ln.strip():
                    d = json.loads(ln)
                    if d.get("ok"):
                        res[(d["ds"], d["clip"])] = d
    return res


def _clip_rows(a: argparse.Namespace, pool: dict, res: dict) -> tuple[list[dict], set, set]:
    """Per-clip rows (reach checked against the IK cache), plus the solver and placement versions seen."""
    from dexengine.core import paths as P
    from experiments.policies.kin_data import reach as KR

    rows, solvers, plac = [], set(), set()
    for k, p in sorted(pool.items()):
        d = res[k]
        td = Path(a.root) / "data" / d["ds"] / "trees" / d["combo"] / d["emb"] / d["tree"]
        ti = json.loads((td / "task_info.json").read_text())
        ek = ti.get("e5_kin") or {}
        if ek.get("placement") != d.get("placement"):
            raise SystemExit(
                f"{MARK} {d['clip']}: tree placement {ek.get('placement')} ≠ prep result {d.get('placement')}"
            )
        plac.add((d.get("placement"), d.get("offsets_sha256")))
        npz = {s: P.ik_dir(d["combo"], s, d["ds"]) / f"{d['tree']}.npz" for s in d["active"]}
        rr = KR.reach_of(npz)
        if abs(rr["reach"] - float(d["reach"])) > 1e-6:
            raise SystemExit(
                f"{MARK} {d['clip']}: prep result reach {d['reach']} ≠ cache {rr['reach']} (different version)"
            )
        nofb = KR.reach_of(npz, drop_fallback=True)
        solvers.add(rr["solver"])
        rows.append(
            {
                "axis": p["axis"],
                "dataset": d["ds"],
                "hand_use": hand_use(d["active"], ti),
                "clip": d["clip"],
                "emb": d["emb"],
                "tree": d["tree"],
                "reach": d["reach"],
                "reach_pass": int(bool(d["reach_pass"])),
                "reach_nofb": nofb["reach"],
                "reach_pass_nofb": int(nofb["passes"]),
                "n_jump": sum(rr["n_jump"].values()),
                "max_dq_ratio": max(rr["max_dq_ratio"].values()),
                "reach_per_side": json.dumps(d["reach_per_side"]),
                "final": p.get("final", ""),
                "placement_src": d.get("placement_src") or "",
            }
        )
    if len(plac) != 1:
        raise SystemExit(
            f"{MARK} mixed placement versions (placement · offsets table sha) {sorted(plac)} — only one is accepted"
        )
    if len(solvers) != 1:
        raise SystemExit(f"{MARK} mixed solver versions {sorted(solvers)} — prep results of one version only")
    return rows, solvers, plac


def _cell_counts(rows: list[dict]) -> dict:
    """Per-cell counts (pool, passed, jumped, placement source) over the clip rows."""
    cells = {}
    for r in rows:
        c = cells.setdefault(
            (r["axis"], r["dataset"], r["hand_use"]),
            {"pool": 0, "passed": 0, "passed_nofb": 0, "jumped": 0, "pass_jumped": 0, "adopted": 0, "kin_opt": 0},
        )
        c["pool"] += 1
        if r["placement_src"] in ("adopted", "kin_opt"):
            c[r["placement_src"]] += 1
        c["passed"] += r["reach_pass"]
        c["passed_nofb"] += r["reach_pass_nofb"]
        c["jumped"] += int(r["n_jump"] > 0)
        c["pass_jumped"] += int(r["n_jump"] > 0 and r["reach_pass"])
    return cells


def _cell_table(cells: dict, tg: dict, nogate: bool, per_clip: int) -> list[dict]:
    """One row per cell in the counts or the targets: reach shares, episode cap and shortfall."""
    out = []
    for key in sorted(set(cells) | set(tg)):
        c = cells.get(
            key, {"pool": 0, "passed": 0, "passed_nofb": 0, "jumped": 0, "pass_jumped": 0, "adopted": 0, "kin_opt": 0}
        )
        t = tg.get(key, 0)
        cap = (c["pool"] if nogate else c["passed"]) * per_clip
        out.append(
            {
                "axis": key[0],
                "dataset": key[1],
                "hand_use": key[2],
                "target_eps": t,
                "pool_clips": c["pool"],
                "reach_pass": c["passed"],
                "pass_pct": round(100 * c["passed"] / c["pool"], 1) if c["pool"] else 0.0,
                "reach_pass_nofb": c["passed_nofb"],
                "jump_clips": c["jumped"],
                "pass_jump_clips": c["pass_jumped"],
                "pass_jump_pct": round(100 * c["pass_jumped"] / c["passed"], 1) if c["passed"] else 0.0,
                "jump_pct_all": round(100 * c["jumped"] / c["pool"], 1) if c["pool"] else 0.0,
                "adopted_clips": c["adopted"],
                "kin_opt_clips": c["kin_opt"],
                "max_eps": cap,
                "clips_needed": math.ceil(t / per_clip) if t else 0,
                "short_eps_min": max(0, t - cap),
            }
        )
    return out


def _write_tsvs(a: argparse.Namespace, out: list[dict], rows: list[dict]) -> None:
    for path, data in ((a.out_tsv, out), (a.clips_tsv, rows)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(data[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(data)


def _print_table(out: list[dict]) -> None:
    cols = list(out[0])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in out:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


def _print_totals(out: list[dict], rows: list[dict], plac: set, solvers: set, per_clip: int, nogate: bool) -> None:
    tot = {
        k: sum(r[k] for r in out)
        for k in (
            "target_eps",
            "pool_clips",
            "reach_pass",
            "reach_pass_nofb",
            "jump_clips",
            "pass_jump_clips",
            "adopted_clips",
            "kin_opt_clips",
            "max_eps",
            "short_eps_min",
        )
    }
    tot["fill_eps"] = sum(min(r["target_eps"], r["max_eps"]) for r in out)
    pl, osha = plac.pop()
    print(
        f"\ntotal: {tot} · max episodes per clip {per_clip} · solver {solvers.pop()} · placement {pl} · "
        f"offsets table sha {osha}"
    )
    print(
        "reach_pass_nofb = reference value counting the divergence-guard fallback frames as unreached "
        "(that fix's share · approximate)"
    )
    print(
        "jump_clips · pass_jump_clips = clips with a jump (active arm, any frame |Δq| > velocity limit / "
        "trajectory rate · "
        "recorded only — not a filter) · whole pool · among passing clips · pass_jump_pct = share of "
        "episodes with a jump (passing clips = rendered "
        "source placement)"
    )
    if nogate:
        print(
            f"placement pipeline = no filter — max_eps = pool clips × {per_clip} · share of episodes with "
            "a jump (source placement · all clips) = "
            f"{tot['jump_clips']}/{tot['pool_clips']} = {round(100 * tot['jump_clips'] / tot['pool_clips'], 1)} %"
        )
    pj = sum(r["pass_jump_clips"] for r in out)
    print(
        f"share of episodes with a jump (source placement) = {pj}/{tot['reach_pass']} = "
        f"{round(100 * pj / tot['reach_pass'], 1) if tot['reach_pass'] else 0.0} %"
    )
    print(
        f"{MARK} DONE cells {len(out)} · pool {len(rows)} · pass {sum(r['reach_pass'] for r in rows)}", file=sys.stderr
    )


def main() -> int:
    a = _parse_args()
    from dexengine.record.augment import policy as AP

    if Path(os.environ.get("DEXENGINE_MAIN_ROOT", "")).resolve() != Path(a.root).resolve():
        raise SystemExit(
            f"{MARK} DEXENGINE_MAIN_ROOT differs from --root — paths resolves the IK cache location from it"
        )
    per_clip = LOOKS * (1 + int(AP.PER_CLIP))
    with open(a.pool) as fh:
        pool = {(r["ds"], r["clip"]): r for r in csv.DictReader(fh, delimiter="\t") if r["in_pool"] == "1"}
    res = _read_prep_results(a.prep)
    miss = sorted(set(pool) - set(res))
    if miss:
        raise SystemExit(f"{MARK} pool {len(pool)}: no prep result for {len(miss)} — {miss[:3]} (count identity)")
    rows, solvers, plac = _clip_rows(a, pool, res)
    nogate = next(iter(plac))[0] == "pipeline"  # placement pipeline = no filter — cell maximum = every pool clip
    with open(a.targets) as fh:
        tg = {(r["axis"], r["dataset"], r["hand_use"]): int(r["episodes"]) for r in csv.DictReader(fh, delimiter="\t")}
    cells = _cell_counts(rows)
    out = _cell_table(cells, tg, nogate, per_clip)
    _write_tsvs(a, out, rows)
    _print_table(out)
    _print_totals(out, rows, plac, solvers, per_clip, nogate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
