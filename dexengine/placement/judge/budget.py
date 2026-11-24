"""Judge budget-cut record — the one owner (no dependencies at import; only the shell entry calls the status-ledger
reader).

Why: the judge's exhaustive search (judge.exhaust) stops at the per-clip wall-clock budget (`--clip-budget-s`) or at
the cell budget (`--exhaust-max`, which truncates the ranked candidate list). The wall-clock budget is a path where
judge speed changes the result (with the same inputs a faster machine sees more cells), so each cut is kept as a ledger
cell rather than a log line, and summarised as counts per dataset.

Cell values (ledger column COLUMN):
  ""               no record — a judge run from before this record, or a row that never went through the judge
                   (f0 defect · inherited piece)
  "-"              judged and not cut by a budget (no exhaustive search · enough candidates · the list was covered)
  "<kind>:<round>" the exhaustive search was cut by a budget (0 adoptions → UNPROVEN · some adoptions → ADOPTED with
                   fewer bake candidates than asked)
    kind (KINDS)   wall   wall clock — the per-clip time budget ran out, the remaining cells were not seen
                   cell   cell — the ranked list was truncated at --exhaust-max and its head did not fill the candidates
                   flush  flush cap — stopped at --exhaust-flushes (not among the canonical judge flags)
    round (round_token)  r0 = canonical budget (EXHAUST_RETRY `from`) · r1 = retry budget (from × factor) ·
                         b<N> = any other budget N
The judge result (plans · noconfirm · judge input record) is unchanged — only the record file and the marker are added.

Flow: the judge writes `<out>/<FILE_PREFIX><combo>.tsv`, one row (clip · cell) per judged clip, and ends with a TAG
marker (row count · cut count) → the status ledger (placement.ledger) collects the files with read_files into the COLUMN
cell → the fold (placement.status_fold) carries it into the final ledger. Summary = summary(rows) (the stage-2 summary
through this file's shell entry · the fold's per-axis marker).
Shell entry: python3 dexengine/placement/judge/budget.py --summary <status ledger> — one summary line (the ledger is
read with the status-ledger reader) · rc 2 when it is not a ledger or has no budget column (a ledger from before this
record).
"""

import sys

#: Ledger column name — the **last** column of the chunk status_ledger.tsv and of the final status_final_<axis>.tsv
#: (read by header).
COLUMN = "judge_budget"
#: Record file the judge writes in a chunk — `<out>/judge_budget_<combo>.tsv` (clip · cell).
FILE_PREFIX = "judge_budget_"
#: Marker head — judge log (one line per cut clip + one at the end of the run) · status ledger · summary.
TAG = "[JUDGE_BUDGET]"
#: Not cut.
NONE = "-"
#: Cut kinds — name → display label (summary). Order = display order.
KINDS = {"wall": "wall clock", "cell": "cell", "flush": "flush"}


def round_token(base_budget, retry: dict) -> str:
    """The judge's base search budget (`--exhaust-max`, before the one-arm ×2) → round token. retry =
    dexengine.data.rules.EXHAUST_RETRY (passed by the caller — this module does not import the registry)."""
    b = int(base_budget)
    if b == int(retry["from"]):
        return "r0"
    if b == int(retry["from"]) * int(retry["factor"]):
        return "r1"
    return f"b{b}"


def encode(kind, rnd: str) -> str:
    """One cell value — kind None = not cut (NONE). An unknown kind or round raises (on the writing side first)."""
    if kind is None:
        return NONE
    v = f"{kind}:{rnd}"
    parse(v)
    return v


def parse(v: str):
    """Cell value → None (no record "") · (None, None) (NONE) · (kind, round). Outside the vocabulary → ValueError."""
    if v == "":
        return None
    if v == NONE:
        return (None, None)
    kind, sep, rnd = v.partition(":")
    ok_r = rnd in ("r0", "r1") or (rnd[:1] == "b" and rnd[1:].isdigit())
    if not sep or kind not in KINDS or not ok_r:
        raise ValueError(
            f"judge budget cell {v!r} is outside the vocabulary (kinds {list(KINDS)} · "
            f"rounds r0|r1|b<N> · not cut {NONE!r})"
        )
    return (kind, rnd)


def cell_of(row) -> str:
    """Budget cell of one ledger row (dict) — missing column · empty cell = "". Outside the vocabulary → ValueError
    (also when carrying an old ledger)."""
    v = row.get(COLUMN) or ""
    parse(v)
    return v


def read_files(roots, key=str) -> dict:
    """Read every `<FILE_PREFIX>*.tsv` under the judge roots → {clip: cell}. Files are read oldest first, so when a clip
    is in two files the newer judge run wins. key = the caller's clip-name normalisation (join key). A cell outside the
    vocabulary or a row that is not (clip, cell) → ValueError."""
    import glob
    import os

    out: dict = {}
    for R in roots:
        for f in sorted(glob.glob(str(R) + f"/**/{FILE_PREFIX}*.tsv", recursive=True), key=os.path.getmtime):
            with open(f, encoding="utf-8") as fh:
                for n, ln in enumerate(fh, 1):
                    if not ln.strip():
                        continue
                    p = ln.rstrip("\n").split("\t")
                    if len(p) != 2 or not p[0]:
                        raise ValueError(f"judge budget record row is not (clip, cell): {f}:{n}: {ln.rstrip()!r}")
                    parse(p[1])
                    out[key(p[0])] = p[1]
    return out


def summary(rows, label: str = "") -> str:
    """One summary line of the budget cuts of ledger rows (dict — clip · status · COLUMN), counted by kind · round ·
    end state. label = name after the marker (the fold's axis — one line per dataset)."""
    from dexengine.placement.ledger_states import STATES  # noqa: PLC0415 — end-state vocabulary (no dependencies)

    if rows and COLUMN not in rows[0]:
        raise ValueError(f"ledger has no budget column {COLUMN} (a ledger from before this record)")
    none = blank = 0
    by_kind = dict.fromkeys(KINDS, 0)
    by_round: dict = {}
    by_state: dict = {}
    for r in rows:
        p = parse(r.get(COLUMN) or "")
        if p is None:
            blank += 1
        elif p[0] is None:
            none += 1
        else:
            by_kind[p[0]] += 1
            by_round[p[1]] = by_round.get(p[1], 0) + 1
            by_state[r["status"]] = by_state.get(r["status"], 0) + 1
    order = [s for s in STATES if s in by_state] + sorted(set(by_state) - set(STATES), key=str)  # unknown states shown
    kinds = " · ".join(f"{KINDS[k]} {v}" for k, v in by_kind.items())
    rounds = " · ".join(f"{k} {v}" for k, v in sorted(by_round.items())) or "-"
    states = " · ".join(f"{s} {by_state[s]}" for s in order) or "-"
    head = f"{TAG} {label}:" if label else TAG
    return (
        f"{head} budget cut {sum(by_kind.values())}/{len(rows)} rows ({kinds}) · rounds {rounds} · "
        f"states {states} · not cut {none} · no record {blank} (record only · judge result unchanged)"
    )


def main(argv) -> int:
    """Shell entry `--summary <status ledger>` — see the module docstring."""
    import argparse

    ap = argparse.ArgumentParser(prog="budget.py", description="One summary line of the judge budget-cut cells")
    ap.add_argument("--summary", required=True, metavar="TSV", help="status ledger (status_ledger.tsv)")
    a = ap.parse_args(argv)
    from dexengine.placement.ledger import read_status_ledger  # noqa: PLC0415 — the one ledger reader (same directory)

    try:
        print(summary(read_status_ledger(a.summary)))
    except (OSError, ValueError) as e:
        print(f"{TAG} ✗ {e}: {a.summary}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
