#!/usr/bin/env python3
"""Fine reasons of the clips the placement judge could not accept: applies the classification rule
(``judge_fail_rules.py``, its one owner) to every failed clip and writes the summary (Sec. E.5 drop stages:
``experiments/ablations/placement/drop_stages.py`` reads its ``--out-clips``).

Question: for every clip the judge rejected, at which check did the candidate that got furthest (the best candidate)
fail.

Reads only the files the judge left (the judge logic is not run again):
  · clip status   the folded ledgers ``status_final_<axis>.tsv`` of the hand (``dexengine.data.rules.ledger_files``)
  · reason text   the chunk folders' ``p*/noconfirm_<combo>.tsv`` column 3 (the judge's ``fno.write``) — four row
  forms:
                  ``-  no_volume(...) · best_candidate <stage>/<side>:<met>`` · ``anchorN  confirm_failed(checks N)
                  <stage>/<key>=<n> …``
                  · ``EXHAUSTED|TRUNCATED  coverage a/b candidates … · best_candidate … · reject_tally
                  <stage>/<key>=<n> …`` (the tally = ``rej``)
                  · ``anchorN  final_guard_rejected …``
  · vocabulary    the judge's check order ``_MISS_RANK`` and its tally key tuple (``_tally``), read from the judge's
                  source by AST (``dexengine/placement/judge/``). No copy is kept here: when the judge has a key the
                  groups (``judge_fail_rules.GROUPS``) do not cover, the classifier stops.
The words are the judge records' own vocabulary (on-disk data): keys, regexes and row heads match them.

Two labels per clip (class and fine reason): see ``judge_fail_rules.py``.

Before/after boundary: the first commit that added the best-candidate record (``_MISS_RANK``) to the judge, read from
git (history path ``scripts/placement/judge.py``, the judge's path when it was added); a noconfirm file whose mtime is
later counts as "after".

usage: python -m experiments.scoring.judge_fail_detail --hand <hand> [--root <data checkout>] [--out-json F]
       [--out-clips F] [--oracle <class × axis tsv>]
  --oracle compares every row of an earlier class × axis table (rc 3 = differences, reviewed by a person).
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from dexengine.core import paths as P
from experiments.scoring.judge_fail_rules import CLASS_LABEL, CLASS_ORDER, GROUPS, REACH_KINDS, Rows, classify_rows

REPO = Path(__file__).resolve().parents[2]
#: the judge package whose source holds the vocabulary (``_MISS_RANK`` · ``_tally``)
JUDGE_DIR = REPO / "dexengine/placement/judge"
#: the judge's path in git history when the best-candidate record was added (the boundary commit is looked up there)
JUDGE_HISTORY_PATH = "scripts/placement/judge.py"


#: Table penetration depth bins (mm) — a rejected depth is deeper than the judge's threshold
#: (collision.DESK_SURF_PEN_MM).
DEPTH_EDGES = (0, 10, 20, 50, 100, 200, 10**9)


def vocab() -> dict:
    """The judge's vocabulary read from the judge's source (AST); stops when it is not found (no fallback copy)."""
    rank, keys = None, None
    for src in sorted(JUDGE_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(src.read_text(encoding="utf-8"))):
            # a name (`_MISS_RANK = …`) or an attribute (`self._MISS_RANK = …` inside the Judge class) — the same
            # definition
            if (
                rank is None
                and isinstance(node, ast.Assign)
                and any(
                    (isinstance(t, ast.Name) and t.id == "_MISS_RANK")
                    or (isinstance(t, ast.Attribute) and t.attr == "_MISS_RANK")
                    for t in node.targets
                )
            ):
                rank = ast.literal_eval(node.value)
            if keys is None and isinstance(node, ast.FunctionDef) and node.name == "_tally":
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.For)
                        and isinstance(sub.target, ast.Name)
                        and sub.target.id == "key"
                        and isinstance(sub.iter, ast.Tuple)
                    ):
                        keys = tuple(dict.fromkeys(ast.literal_eval(sub.iter)))
    if not rank or not keys:
        raise SystemExit(
            f"[fail_detail] ✗ judge vocabulary not found under {JUDGE_DIR}: _MISS_RANK={bool(rank)} "
            f"_tally keys={bool(keys)} — the judge's structure changed; fix this classifier first"
        )
    mine = [k for _, _, ks in GROUPS for k in ks]
    if len(mine) != len(set(mine)):
        raise SystemExit(f"[fail_detail] ✗ a key appears twice in the groups: {mine}")
    if set(mine) != set(keys):
        raise SystemExit(
            f"[fail_detail] ✗ groups ↔ judge vocabulary differ — groups only {sorted(set(mine) - set(keys))} · "
            f"judge only {sorted(set(keys) - set(mine))}. Give the new key its group in GROUPS"
        )
    if not set(rank) <= set(keys):
        raise SystemExit(f"[fail_detail] ✗ check-order keys missing from the tally: {sorted(set(rank) - set(keys))}")
    return {"rank": rank, "keys": keys}


def fix_boundary(repo: Path = REPO) -> dict:
    """The first commit that added the best-candidate record to the judge (git owns it)."""
    r = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "log",
            "--reverse",
            "--format=%h %ct %ci",
            "-S",
            "_MISS_RANK",
            "--",
            JUDGE_HISTORY_PATH,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    ln = next((x for x in r.stdout.splitlines() if x.strip()), "")
    if not ln:
        raise SystemExit(
            f"[fail_detail] ✗ boundary commit not found in git (rc {r.returncode}) — without it 'no tally' "
            "and 'reach family' would mix silently"
        )
    h, ct, *ci = ln.split()
    return {"commit": h, "epoch": int(ct), "time": " ".join(ci[:2])}


def statuses_fold(hand: str, root: Path) -> tuple[list[tuple], list[str]]:
    """Clip statuses from the folded ledgers (file list owner: ``dexengine.data.rules.ledger_files``)."""
    from dexengine.data import rules as RR  # noqa: PLC0415
    from dexengine.placement import final_ledger as FL  # noqa: PLC0415 — folded ledger reader

    files = RR.ledger_files("*", hand=hand, root=root)
    out = []
    for f in files:
        ax = RR.ledger_axis(f)
        out.extend(
            (ax, row["clip"], row["status"], row.get("source_tag") or "", row.get("note") or "")
            for row in FL.read_final_rows(f)
        )
    return out, [str(f) for f in files]


def _bin(v: float) -> str:
    for lo, hi in zip(DEPTH_EDGES, DEPTH_EDGES[1:], strict=False):
        if lo <= v < hi:
            return f"{lo}–{hi}" if hi < 10**9 else f"≥{lo}"
    return "?"


def _q(xs, q):
    return round(statistics.quantiles(xs, n=100)[q - 1], 1) if len(xs) >= 2 else (xs[0] if xs else None)


def _depth_summary(depths):
    """Penetration depth quantiles over all axes, its histogram and the per-axis quantiles."""
    alld = [v for vs in depths.values() for v in vs]

    depth = {
        "n": len(alld),
        "p50": _q(alld, 50),
        "p90": _q(alld, 90),
        "max": max(alld) if alld else None,
        "bins": {},
        "by_axis": {},
    }
    labels = [_bin(lo) for lo in DEPTH_EDGES[:-1]]
    for lab in labels:
        depth["bins"][lab] = sum(1 for v in alld if _bin(v) == lab)
    for ax, vs in sorted(depths.items()):
        depth["by_axis"][ax] = {
            "n": len(vs),
            "p50": _q(vs, 50),
            "bins": {lab: sum(1 for v in vs if _bin(v) == lab) for lab in labels},
        }
    return depth


def _axis_summary(per_axis, FAIL_STATES, fine_order, clips):
    """Per-axis summary rows, the all-axes class and fine totals, and the desk-penetration rescue count."""
    axes = []
    for ax in sorted(per_axis):
        A = per_axis[ax]
        axes.append(
            {
                "axis": ax,
                "total": A["total"],
                "status": dict(A["status"]),
                "failed": sum(A["status"][s] for s in FAIL_STATES),
                "cls": {c: A["cls"][c] for c in CLASS_ORDER},
                "fine": {f: A["fine"][f] for f in fine_order},
            }
        )
    for ax_rows in list(clips):  # all-axes keys (ALL|<fine>)
        clips[f"ALL|{ax_rows.split('|', 1)[1]}"] += clips[ax_rows]
    tot_cls = collections.Counter()
    tot_fine = collections.Counter()
    for a in axes:
        tot_cls.update(a["cls"])
        tot_fine.update(a["fine"])
    rescue = sum(a["fine"].get("best:desk_pen", 0) for a in axes)
    return axes, tot_cls, tot_fine, rescue


def build(status_rows: list[tuple], stage2: Path, voc: dict, fx: dict) -> tuple[dict, dict]:
    """Classify every failed clip → (summary, {"<axis>|<fine>": [[clip, status, tag, why]]}). A judge failure = the
    judge found no placement (ledger_states.PLACEMENT_FAILED); the other rejected states are counted, not classified."""
    from dexengine.placement.ledger_states import PLACEMENT_FAILED as FAIL_STATES  # noqa: PLC0415
    from dexengine.placement.ledger_states import REJECTED  # noqa: PLC0415

    OTHER_STATES = tuple(s for s in REJECTED if s not in FAIL_STATES)
    keys = voc["keys"]
    rows = Rows(stage2, fx["epoch"])
    per_axis, clips, depths, vals = classify_rows(status_rows, OTHER_STATES, FAIL_STATES, rows, keys)
    fines = sum((A["fine"] for A in per_axis.values()), collections.Counter())
    rank = voc["rank"]
    fine_order = (
        [f"best:{k}" for k in sorted(rank, key=rank.get)]
        + CLASS_ORDER
        + [f"reach:{k}" for k, _ in REACH_KINDS]
        + list(OTHER_STATES)
    )
    fine_order += sorted(k for k in fines if k not in fine_order)  # an unknown fine reason is kept, not dropped
    depth = _depth_summary(depths)
    val_summary = {k: {"n": len(v), "p50": _q(v, 50), "p90": _q(v, 90), "max": max(v)} for k, v in vals.items()}
    axes, tot_cls, tot_fine, rescue = _axis_summary(per_axis, FAIL_STATES, fine_order, clips)
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "boundary": fx,
        "vocab": {"rank": rank, "keys": list(keys)},
        "class_order": CLASS_ORDER,
        "class_label": CLASS_LABEL,
        "fine_order": fine_order,
        "fine_label": (
            {f"best:{k}": f"best candidate → {k}" for k in rank}
            | CLASS_LABEL
            | {f"reach:{k}": f"reach failed — {lab}" for k, lab in REACH_KINDS}
            | {"GUARD_FAIL": "GUARD_FAIL (bake contract guard)", "SOURCE_DEFECT": "SOURCE_DEFECT (source defect)"}
        ),
        "axes": axes,
        "total": {
            "clips": sum(a["total"] for a in axes),
            "failed": sum(a["failed"] for a in axes),
            "cls": {c: tot_cls[c] for c in CLASS_ORDER},
            "fine": {f: tot_fine[f] for f in fine_order},
        },
        "depth_mm": depth,
        "values": val_summary,
        "rescue_desk": {
            "n": rescue,
            "rule": "EXHAUSTED+UNPROVEN clips whose best candidate failed at table penetration "
            "(desk_pen) — the input of a second rescue pass (collision-avoiding IK proposes new solutions that the "
            "same judge and gate functions check again)",
        },
    }
    return out, dict(clips.items())


def oracle_compare(res: dict, tsv: Path) -> list[str]:
    """Compare every row of an earlier class × axis table → the list of differences."""
    diffs = []
    mine = {a["axis"]: a for a in res["axes"]}
    with open(tsv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ax = row["dataset"]
            if ax == "ALL":
                got = res["total"]["cls"]
                gfail = res["total"]["failed"]
                gtot = res["total"]["clips"]
            else:
                a = mine.get(ax)
                if not a:
                    diffs.append(f"{ax}: axis missing in this run")
                    continue
                got, gfail, gtot = a["cls"], a["failed"], a["total"]
            if int(row["total"]) != gtot:
                diffs.append(f"{ax}: total oracle {row['total']} ≠ {gtot}")
            if int(row["failed(EXH+UNP)"]) != gfail:
                diffs.append(f"{ax}: failed oracle {row['failed(EXH+UNP)']} ≠ {gfail}")
            diffs.extend(
                f"{ax}: {c} oracle {row[c]} ≠ {got.get(c, 0)}"
                for c in CLASS_ORDER
                if c in row and int(row[c]) != got.get(c, 0)
            )
    return diffs


def run(hand: str, root: Path) -> tuple[dict, dict]:
    """Classify the hand's failed clips from its folded ledgers under ``root``."""
    voc = vocab()
    fx = fix_boundary()
    # `stage2` = the folder of the tag folders (the hand's chunk folder)
    stage2 = P.run_path("run_chunks", hand, root=root)
    st, used = statuses_fold(hand, root)
    src = {
        "kind": "fold",
        "files": len(used),
        "note": "folded ledgers status_final_<axis>.tsv (the hand's ledger folder)",
    }
    res, clips = build(st, stage2, voc, fx)
    res.update(hand=hand, root=str(root), source=src)
    return res, clips


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Count the fine reasons of judge-failed clips (the check the best candidate failed)."
    )
    ap.add_argument("--hand", required=True, help="hand id (no default)")
    ap.add_argument(
        "--root",
        default=None,
        help="the checkout holding the data trees (overfit/ · data/); default dexengine.core.paths.main_root()",
    )
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-clips", default="")
    ap.add_argument("--oracle", default="", help="an earlier class × axis tsv — every row compared")
    a = ap.parse_args()
    root = Path(a.root).resolve() if a.root else P.main_root().resolve()
    res, clips = run(a.hand, root)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.out_clips:
        Path(a.out_clips).write_text(json.dumps(clips, ensure_ascii=False) + "\n", encoding="utf-8")
    T = res["total"]
    print(
        f"[fail_detail] hand={a.hand} source={res['source']['kind']}({res['source']['files']} files) "
        f"clips {T['clips']} · failed {T['failed']} · boundary {res['boundary']['commit']} {res['boundary']['time']}"
    )
    print("[fail_detail] classes " + " · ".join(f"{CLASS_LABEL[c]} {T['cls'][c]}" for c in CLASS_ORDER))
    print("[fail_detail] fine " + " · ".join(f"{k} {v}" for k, v in T["fine"].items() if v))
    d = res["depth_mm"]
    print(
        f"[fail_detail] table penetration depth n={d['n']} p50={d['p50']} p90={d['p90']} max={d['max']} mm · "
        f"second-pass rescue candidates {res['rescue_desk']['n']}"
    )
    if a.oracle:
        diffs = oracle_compare(res, Path(a.oracle))
        print(f"[fail_detail] oracle {a.oracle} — {len(diffs)} differences")
        for x in diffs[:40]:
            print(f"  ✗ {x}")
        return 3 if diffs else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
