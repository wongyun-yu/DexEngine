"""Placement-search ablation (paper Sec. E.5 / E.7): the rows, how a row's clips become judge items, how the items
are spread over GPU slots, and which fetched attempt of an item counts.

Rows (judge_extra; every row runs the same downstream chain f0 seed → judge → bake → retime → gate):
  a      fixed     --place-strategy fixed                      main set (128)
  b      random    --place-strategy random                     main set
  c      mapfirst  --place-strategy mapfirst                   main set
  dtime  ours      (canon — no extra flag)                     dtime set (16) — the like-for-like time column
  e      allcells  --place-strategy allcells --place-strategy-cap 4096 --place-strategy-budget-s E_BUDGET   ecells (4)
  canary ours      (canon)                                     canary set (64)
  rnd    randorder --place-strategy randorder                  main set — random order, ours' own budget
  xall   allcells  (the flags of e)                            main set, with --only <clips ours dropped>
  rndt   randtime  --place-strategy-budget-s 1200              main set — random order, no cell cap, ours' time cap
  xfull  allcells  cap above every list, budget 7200 s         --only <clips ours dropped> — the whole grid
  ours128 ours     (canon — no extra flag)                     main set, one clip per item — the time column of ours
  on
                                                               the same clips and item shape as rndt / xfull
Items = one stage-2 chunk (one ledger axis = dataset · emb · variant) of ≤ chunk clips (e: 1 clip).
Assignment = longest-processing-time first over the GPU slots, cost = boot + clips × per-clip estimate (row).
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path

from dexengine.data import rules as RR

E_CAP, E_BUDGET = 4096, 7200
FULL_CAP = 4_000_000  # above every desk-valid list (largest seen 928,800 cells) — xfull judges the whole grid
#: row → (clip set, judge extra flags or '-' for the canonical judge, clips per item). Clips per item (one judge
#: process) is small on purpose — a memory limit of 24 GB cards: on an RTX 3090 one canonical judge's process VRAM
#: grows to the card's limit while torch holds 1.8–5.2 GiB, so a canon item died after 3–6 clips; a fresh process per
#: 2 canon clips resets it. (a)–(c) judge ≤ 12 cells per clip → 4.
ROWS = {
    "a": ("main", "--place-strategy fixed", 4),
    "b": ("main", "--place-strategy random", 4),
    "c": ("main", "--place-strategy mapfirst", 4),
    "dtime": ("dtime", "-", 2),
    "ours128": ("main", "-", 1),
    "e": ("ecells", f"--place-strategy allcells --place-strategy-cap {E_CAP} --place-strategy-budget-s {E_BUDGET}", 1),
    "canary": ("canary", "-", 2),
    "rnd": ("main", "--place-strategy randorder", 2),
    "xall": ("main", f"--place-strategy allcells --place-strategy-cap {E_CAP} --place-strategy-budget-s {E_BUDGET}", 1),
    "rndt": ("main", "--place-strategy randtime --place-strategy-budget-s 1200", 1),
    "xfull": (
        "main",
        f"--place-strategy allcells --place-strategy-cap {FULL_CAP} --place-strategy-budget-s {E_BUDGET}",
        1,
    ),
}
#: per-clip GPU seconds used only to balance the queues (RTX 3090 probes on TACO: a 20–22 · b 14–29 · c 17–23 s ·
#: e 0.22 s/cell × the 4096 cap)
PER_CLIP_S = {
    "a": 22.0,
    "b": 30.0,
    "c": 22.0,
    "e": 1000.0,
    "rnd": 300.0,
    "xall": 1000.0,
    "rndt": 1200.0,
    "xfull": 2400.0,
}
BOOT_S = 150.0  # stage2 + judge boot + bake + gate per item
ROW_TAG = {
    "a": 1,
    "b": 2,
    "c": 3,
    "dtime": 4,
    "e": 5,
    "canary": 6,
    "rnd": 7,
    "xall": 8,
    "rndt": 9,
    "xfull": 0,
    "ours128": 4,
}  # runs/ and sb/ are per row name, so a digit shared with dtime never meets it
#: the drawn clip sets (draw.py output, kept as records)
LISTS = Path(__file__).resolve().parent / "data" / "lists"


def read_set(name):
    """Rows of one drawn clip set (clips_<name>.tsv)."""
    return list(csv.DictReader(open(LISTS / f"clips_{name}.tsv", encoding="utf-8"), delimiter="\t"))


def items_for(row, only=None, chunk=None, name=None):
    """Items of a row. only = clip names to keep (a retry of tech-failed clips) · chunk = clips per item instead of the
    row's · name = the row name written to the queue / STATUS / runs folder (a retry runs as its own row, so its tags
    never meet an earlier attempt's)."""
    setname, extra, row_chunk = ROWS[row]
    chunk = chunk or row_chunk
    by_ax = collections.defaultdict(list)
    rs_all = read_set(setname)
    if only is not None:
        unknown = sorted(set(only) - {r["clip"] for r in rs_all})
        if unknown:
            die(f"--only: {len(unknown)} clips not in set {setname}: {unknown[:3]}")
        rs_all = [r for r in rs_all if r["clip"] in only]
    for r in rs_all:
        by_ax[r["axis"]].append(r)
    out = []
    for ax in sorted(by_ax):
        ds, emb, _ = RR.ledger_axis_parts(ax)
        var = RR.ledger_axis_variant(ax) or "-"
        rs = by_ax[ax]
        for k in range(0, len(rs), chunk):
            part = rs[k : k + chunk]
            tag = f"G1k_{ax}_{ROW_TAG[row]}{len(out):03d}"
            if row in ("dtime", "canary", "ours128"):
                cost = BOOT_S + sum(float(p["prod_wall_s"] or 60) for p in part)
            else:
                cost = BOOT_S + len(part) * PER_CLIP_S[row]
            out.append(
                {
                    "row": name or row,
                    "tag": tag,
                    "ds": ds,
                    "emb": emb,
                    "var": var,
                    "clips": ",".join(p["clip"] for p in part),
                    "extra": extra,
                    "cost": cost,
                    "n": len(part),
                }
            )
    return out


def assign(items: list[dict], slots: list) -> tuple[dict, dict]:
    """Longest-processing-time first: each item (by descending cost) goes to the least-loaded slot (ties → smallest
    slot). Returns (slot → items, slot → total cost)."""
    load = dict.fromkeys(slots, 0.0)
    q = {s: [] for s in slots}
    for it in sorted(items, key=lambda x: -x["cost"]):
        s = min(slots, key=lambda s: (load[s], s))
        q[s].append(it)
        load[s] += it["cost"]
    return q, load


def die(msg):
    raise SystemExit(f"[PLACE_ABL_LAUNCH] ✗ {msg}")


def fetched_items(fetched: Path):
    """The item folders the aggregator may read: those each host's ITEMS.tsv names (fetch wrote it). A ledger outside
    the list (an earlier attempt, a stale pull) dies here instead of being counted."""
    out = []
    for hd in sorted(p for p in Path(fetched).iterdir() if p.is_dir() and not p.name.startswith(".")):
        lst = hd / "ITEMS.tsv"
        if not lst.is_file():
            die(f"{hd}: no ITEMS.tsv — fetch the items' run folders first")
        names = {tuple(ln.split("\t")) for ln in lst.read_text(encoding="utf-8").splitlines() if ln}
        seen = {(q.parent.parent.name, q.parent.name) for q in hd.glob("runs/*/*/status_ledger.tsv")}
        if seen != names:
            die(f"{hd}: ledgers {len(seen)} ≠ ITEMS.tsv {len(names)} — {sorted(seen ^ names)[:3]}")
        out += [hd / "runs" / row / tag for row, tag in sorted(names)]
    return out
