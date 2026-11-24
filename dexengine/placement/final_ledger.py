"""Format of the folded ledgers — the final ledger `status_final_<axis>.tsv` and the retired ledger
`ledger_retired.tsv`: their CSV dialect and their one reader.

The writer is `status_fold.py` (with WRITER); the files are located by `datasets.ledger_files` / `ledger_file` /
`retired_ledger_file`. Every reader uses `read_final_rows(path)` — reading column positions with `csv.reader` or
`split("\\t")` breaks silently when a column is added or a cell is quoted.

Standard library only, so any process can read a ledger without importing the pipeline.
"""

import csv

#: Write dialect — tab separated · LF line ends (csv's default CRLF leaves \\r on the last cell for shell readers) ·
#: quoting = csv default (QUOTE_MINIMAL).
WRITER = {"delimiter": "\t", "lineterminator": "\n"}
#: Columns every folded ledger header has.
REQUIRED = ("clip", "status")


def read_final_rows(path) -> list[dict]:
    """Rows of one folded ledger (final or retired) — {column name: value} per row, in file order.

    Same dialect as the writer (WRITER — quoted cells are unquoted). A header-only file (0 rows — the fold recorded
    "empty") gives []. A short row has None in its missing cells (the caller decides whether to drop it).
    A header without clip and status (an empty file included) raises ValueError.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh, delimiter=WRITER["delimiter"])
        rows = list(rd)
        head = list(rd.fieldnames or [])
    missing = [c for c in REQUIRED if c not in head]
    if missing:
        raise ValueError(f"not a folded ledger (columns {missing} missing, header {head}): {path}")
    return rows


#: Retired ledger columns — one row = one verdict row of one chunk that the current rules retired.
RETIRED_COLS = ("axis", "clip", "status", "source_tag", "reason")


def retired_key(row) -> tuple:
    """Identity of a retired row = (axis, clip, source chunk) — one verdict row of one chunk. State and reason are not
    part of it."""
    if isinstance(row, dict):
        return (row["axis"], row["clip"], row["source_tag"])
    return (row[0], row[1], row[3])


def merge_retired(old: list[dict], new: list, live: set, axes: set) -> tuple[list[list[str]], dict]:
    """Merge one fold's retired rows into the retired ledger, which is a cumulative history.

    old  = the current table's rows (read_final_rows) · new = the rows this fold retired ([axis, clip, state, source
    chunk, reason]) · live = identities {(axis, clip, source chunk)} of the live rows of the ledgers folded now ·
    axes = the axes folded now.
    Rules:
      1. Old rows stay — also when their source chunk was deleted, or the same clip was judged again by another
         chunk and is live now (that replacement is the history this table records).
      2. Rows retired now are merged by identity — an existing row takes the new state and reason in place, a new
         row is appended.
      3. Reinstatement: a row of an axis folded now whose identity is live again (the rules changed and that verdict
         row folds again) is removed — one verdict row cannot be both live and retired. The removed identities are
         returned in stats["reinstated"]. Rows of axes not folded now are left alone.
    An old row with missing columns raises ValueError.
    """
    out: dict[tuple, list[str]] = {}
    for r in old:
        miss = [c for c in RETIRED_COLS if r.get(c) is None]
        if miss:
            raise ValueError(f"retired ledger row is missing columns {miss}: {r}")
        out[retired_key(r)] = [r[c] for c in RETIRED_COLS]
    n_old = len(out)
    added = updated = 0
    for r in new:
        k = retired_key(r)
        if k in out:
            updated += int(out[k] != list(r))
        else:
            added += 1
        out[k] = list(r)
    back = sorted(k for k in out if k[0] in axes and k in live)
    for k in back:
        out.pop(k)
    return list(out.values()), {"old": n_old, "added": added, "updated": updated, "reinstated": back}
