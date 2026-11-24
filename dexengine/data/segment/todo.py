"""Pieces to do: the pieces (``_tN`` trees) of this hand's base layer whose later outputs are missing or stale.

The long-clip split driver (``dexengine.data.stage1.split_apply``) passes these pieces to the downstream steps. The
judgement per step and tree is ``stage1.status.tree_todo`` (the same as the stage-1 cells); this module resolves the
piece
list to tree paths, asks it, and writes the result.

  input   --units   lines <ds \\t emb \\t piece> (the driver's snapshot) · --stages step ids (comma list)
  output  --out     the pieces to do: <ds \\t emb \\t piece \\t empty step ids (;)>, input order
  marker  [PIECE_TODO] hand=<hand> ds=<ds> ... per dataset, then one total line
  rc      0 = measured (also with nothing to do) · 2 = input error (missing piece folder, unknown step)

    python -m dexengine.data.segment.todo --hand <hand> --units <tsv> --stages 2d,3,4 --out <tsv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import input_missing


def read_units(path: str) -> list[tuple[str, str, str]]:
    """Snapshot lines -> (ds, emb, piece). A line without three fields is fatal."""
    rows = []
    for i, ln in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not ln:
            continue
        f = ln.split("\t")
        if len(f) < 3 or not all(f[:3]):
            raise SystemExit(f"[piece_todo] ✗ {path}:{i} does not have three fields: {ln!r}")
        rows.append((f[0], f[1], f[2]))
    return rows


def main() -> int:
    from dexengine.data.stage1.status import tree_todo  # noqa: PLC0415  (heavy: the stage table)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hand", required=True, help="hand id (no default)")
    ap.add_argument("--units", required=True, help="piece list <ds\\temb\\tpiece>")
    ap.add_argument("--stages", required=True, help="downstream step ids (comma list, first token of a STAGES name)")
    ap.add_argument("--out", required=True, help="TSV of the pieces to do")
    a = ap.parse_args()
    spec = P.hand_spec_dir(a.hand)
    stages = [s for s in a.stages.split(",") if s]
    if not stages:
        print("[piece_todo] ✗ --stages is empty", file=sys.stderr)
        return 2
    units = read_units(a.units)
    tree_of = {}
    for ds, emb, clip in units:
        t = P.layer_path("base", ds, hand=spec, emb=emb) / clip
        if not (t / "task_info.json").is_file():
            print(
                input_missing(
                    f"[piece_todo] ✗ piece {ds}/{emb}/{clip} has no task_info ({t}) -- the list and the layer disagree"
                ),
                file=sys.stderr,
            )
            return 2
        tree_of[(ds, emb, clip)] = t
    gaps: dict = {}
    for ds in dict.fromkeys(u[0] for u in units):
        mine = [tree_of[u] for u in units if u[0] == ds]
        got = tree_todo(ds, spec, stages, mine)
        gaps.update(got)
        per = {s: sum(1 for t in mine if any(n.split()[0] == s for n in got[t])) for s in stages}
        print(
            f"[PIECE_TODO] hand={a.hand} ds={ds} pieces {len(mine)} · to do {sum(1 for t in mine if got[t])} "
            "· per step " + " ".join(f"{s}={k}" for s, k in per.items()),
            flush=True,
        )
    todo = [(u, gaps[tree_of[u]]) for u in units if gaps[tree_of[u]]]
    with open(a.out, "w", encoding="utf-8") as fh:
        for (ds, emb, clip), names in todo:
            fh.write(f"{ds}\t{emb}\t{clip}\t{';'.join(n.split()[0] for n in names)}\n")
    print(f"[PIECE_TODO] hand={a.hand} total pieces {len(units)} · to do {len(todo)} -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
