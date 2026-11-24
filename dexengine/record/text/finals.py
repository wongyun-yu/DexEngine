"""Stored datasets the text sources read: final (collect-layer) folders, the orig-round finals, why a final episode is
not a source yet (final_wait) and whether a piece-layer render was made from the current tree (tree_fresh). The
reason strings are data (frames_missing)."""

from __future__ import annotations

import glob
import json
from pathlib import Path

from dexengine.core import paths as P


def is_final(root) -> bool:
    """A final dataset folder (data/<ds>/collect/<combo>), not a piece-layer dataset."""
    return Path(root).parent.name == "collect"


def final_wait(e: dict) -> str | None:
    """Why a final-layer episode is not a text source (None = usable): an orig-round episode whose ego view did not
    show the grasp (ego_view.grasp_visible not True) is replaced by the aug round's row 0. The reason is data (it
    lands in frames_missing.tsv)."""
    gv = (e.get("ego_view") or {}).get("grasp_visible")
    if e.get("round") == "orig" and gv is not True:
        return f"outside the original view (grasp_visible {gv}) — waiting for aug row 0"
    return None


def final_datasets(root=None) -> list[Path]:
    """Every final dataset (collect layer, with meta/info.json). root None = the data root + every round root
    (layout.collect_round_roots() — aug finals may live outside the data root); root given = that root only. The same
    real folder (symlink) is listed once."""
    DL = P.data_layout
    roots = (
        [Path(root)]
        if root is not None
        else [P.main_root(), *(Path(r) for r in DL.collect_round_roots().values() if r)]
    )
    out, seen = [], set()
    for r in roots:
        for ds in sorted(DL.DATASETS):
            for g in sorted(glob.glob(str(r / DL.render("collect", ds, combo="*")))):
                d = Path(g)
                if (d / "meta" / "info.json").is_file() and d.resolve() not in seen:
                    seen.add(d.resolve())
                    out.append(d)
    return out


def orig_finals() -> list[Path]:
    """The orig-round final datasets under the data root (aug twins are reached through aug_twin)."""
    return final_datasets(P.main_root())


def tree_fresh(src: dict) -> str | None:
    """Was this piece-layer render made from the current tree? The recorded fingerprint (tree_ref.json) must equal
    the current tree fingerprint (tree_ref.tree_sha at tree_ref.record_root). Reason when not (or no record), None
    when it is. The reason strings are data (frames_missing)."""
    from dexengine.record import piece as PC
    from dexengine.record import tree_ref as TR

    piece = PC.bundle_of(Path(src["root"]))
    if piece is None:
        return f"no tree record — {src['root']} is not a round piece location (<piece>/rounds/<round>/dataset)"
    refs = sorted(piece.glob(f"seq1*/**/{src['tree']}/tree_ref.json"))
    if not refs:
        return "no tree record (tree_ref.json)"
    for f in refs:
        r = json.loads(f.read_text())
        if r.get("state") != "verified" or not r.get("sha"):
            return f"tree record state {r.get('state')}"
        try:
            now = TR.tree_sha(TR.record_root(r, f), r["clip"])
        except ValueError as e:
            return f"no tree — {e}"
        if now != r["sha"]:
            return f"generation differs (render {r['sha'][:12]} ≠ now {now[:12]})"
    return None
