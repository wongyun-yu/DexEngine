"""Write, move and delete guard for the per-hand run roots that campaign cells live in.

A run root of a hand is a directory named after the hand under one of `data_layout.RUN_HAND_ROOTS`
(`overfit/hands/<hand>`, `records/hands/<hand>`). Every write, move and delete the campaign code makes in
those roots goes through this module:

* `assert_writable(path, hand)` — the path must lie in this hand's root, or in a hand-independent run layer
  (`data_layout.RUN_SHARED`) whose declared writer this hand is; anything outside the run layers is refused.
* `guarded_move(src, dst, hand)` — both ends inside this hand's root (never the root itself or above it), the
  destination must not exist (no overwrite), one `os.rename`.
* `guarded_import(src, dst, hand)` — work area -> hand root: the source strictly inside one of
  `data_layout.RUN_WORK_AREAS`, the destination inside this hand's root, no overwrite, `os.rename` only (a move across
  file systems fails instead of falling back to copy + delete).
* `guarded_rmtree(path, hand)` — only paths strictly inside this hand's root; a chunk folder only with the campaign
  that owns it (its CAMPAIGN_OWNER file).

A run whose run tree is its data root (a generation or folder root, paths.run_root_of) writes only to that root's
run layers, never to the code checkout's (`_canon_guard`). Ownership is decided by the root's directory name; no
owner-marker file is read or created.
Refusals raise HandRootError; every completed move and delete prints one `[HAND_ROOT]` line to stderr.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys

from dexengine.core import paths as P

DL = P.data_layout


class HandRootError(RuntimeError):
    """A write, move or delete that the hand-root rules refuse."""


def _say(msg: str) -> None:
    print(f"[HAND_ROOT] {msg}", file=sys.stderr, flush=True)


def _base(root=None) -> pathlib.Path:
    return pathlib.Path(root if root is not None else P.run_root()).resolve()


def _canon_guard(p: pathlib.Path) -> None:
    """Refuse writes into the code checkout's run layers (canonical hand roots and hand-independent layers) from two
    kinds of runs, whose ledgers and audits would overwrite the canonical ones. Measured on the real path (a write
    through a symlink counts too).
      1. a run whose run tree is its data root (a generation or folder root — paths.run_root() != the code tree,
         paths.run_root_of) writes only to that root's run layers;
      2. a code checkout that holds its own data (data_layout.holds_data) only describes that data: a run with
         another, unregistered data root does not write there."""
    wt = pathlib.Path(P.WT).resolve()
    real = pathlib.Path(os.path.realpath(str(p)))
    inside = any(f(x, wt) is not None for x in (p, real) for f in (root_of, shared_layer_of))
    if pathlib.Path(P.run_root()).resolve() != wt:
        if inside:
            raise HandRootError(
                f"{p} is the run layer of code checkout {wt} — a run whose data root is the "
                f"run tree ({P.main_root()}) writes only to that root's run layer "
                f"({P.run_root()})"
            )
        return
    data = pathlib.Path(P.main_root()).resolve()
    if inside and data != wt and DL.root_kind(str(data), str(wt)) != "box" and DL.holds_data(str(wt)):
        raise HandRootError(
            f"{p} is the run layer of code checkout {wt}, which has its own data — a run on "
            f"data root {data} (no generation registered) cannot write here (its ledger "
            "describes only that checkout's data)"
        )


def _abs(path) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(str(path)))


def root_of(path, root=None) -> pathlib.Path | None:
    """The hand root directory a path lies in (None = outside every hand root)."""
    base = _base(root)
    p = _abs(path)
    try:
        rel = p.relative_to(base)
    except ValueError:
        try:
            rel = p.resolve().relative_to(base)
        except ValueError:
            return None
    for key in DL.RUN_HAND_ROOTS:
        head = DL.RUN_LAYERS[key].split("{", 1)[0].rstrip("/").split("/")
        if len(rel.parts) > len(head) and list(rel.parts[: len(head)]) == head:
            return base.joinpath(*rel.parts[: len(head) + 1])
    return None


def shared_layer_of(path, root=None) -> str | None:
    """The hand-independent run layer (longest prefix) a path outside the hand roots lies in; None = not a run layer."""
    base = _base(root)
    p = _abs(path)
    try:
        rel = p.relative_to(base).as_posix()
    except ValueError:
        return None
    best, blen = None, -1
    for k, (tpl, _w) in DL.RUN_SHARED.items():
        tpl = tpl.split("{", 1)[0].rstrip("/")  # a layer with a placeholder: its fixed head
        if (rel == tpl or rel.startswith(tpl + "/")) and len(tpl) > blen:
            best, blen = k, len(tpl)
    return best


def _check_hand(hand: str) -> None:
    if hand not in P.HAND_SPEC_DIR:
        raise HandRootError(f"hand {hand!r} is not a registered hand id (registered {list(P.HAND_IDS)}) — no default")


def assert_writable(path, hand: str, *, combo: str = "", by: str = "", root=None) -> pathlib.Path:
    """Check a path before writing it and return it (absolute). Inside a hand root: that root must be this hand's.
    In a hand-independent layer: this hand must be its declared writer (`exp_v2` = the axis hand). Outside the run
    layers: refused. combo / by are accepted for call compatibility and unused."""
    _check_hand(hand)
    p = _abs(path)
    _canon_guard(p)
    r = root_of(p, root)
    if r is None:
        layer = shared_layer_of(p, root)
        if layer is None:
            raise HandRootError(
                f"{p} is not a run layer (data_layout.RUN_LAYERS, RUN_SHARED) — not written through this guard"
            )
        writer = DL.RUN_SHARED[layer][1]
        if writer == "axis" and hand != DL.AXIS_HAND:
            raise HandRootError(
                f"{p} is {layer} (only the axis hand {DL.AXIS_HAND} writes it) — run hand {hand} cannot"
            )
        return p
    owner = r.name
    if owner != hand:
        raise HandRootError(f"{p} is inside the root of hand {owner} — run hand {hand} cannot write it")
    return p


def _chunk_owner_ok(p: pathlib.Path, hand: str, campaign_dir, root=None) -> None:
    """A chunk folder (`run_chunks/<tag>`) is deleted only when its CAMPAIGN_OWNER names this campaign (two campaigns
    of one hand)."""
    if p.parent != _abs(P.run_path("run_chunks", hand, root=root)):
        return
    if campaign_dir is None:
        raise HandRootError(f"{p} is a chunk folder — deleting it needs --campaign-dir (CAMPAIGN_OWNER check)")
    own = p / "CAMPAIGN_OWNER"
    got = own.read_text(encoding="utf-8").splitlines()[0].strip() if own.is_file() else ""
    want = os.path.realpath(str(campaign_dir))
    if got != want:
        raise HandRootError(f"{p}: CAMPAIGN_OWNER={got or '(none)'} != this campaign {want} — not deleted")


def guarded_rmtree(path, hand: str, *, campaign_dir=None, root=None) -> bool:
    """Delete a path strictly inside this hand's root. False = nothing there (nothing to do)."""
    _check_hand(hand)
    p = _abs(path)
    _canon_guard(p)
    if not p.exists() and not p.is_symlink():
        return False
    r = root_of(p, root)
    if r is None:
        raise HandRootError(
            f"{p} is not inside a hand root — destructive operations only inside hand roots "
            "(not retired roots or data layers)"
        )
    if r.name != hand:
        raise HandRootError(f"{p} belongs to hand {r.name} — run hand {hand} cannot delete it")
    if p == r or r.is_relative_to(p):
        raise HandRootError(f"{p} is a hand root itself (or above it) — roots are not deleted")
    _chunk_owner_ok(p, hand, campaign_dir, root)
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink()
    _say(f"rm hand={hand} {p}")
    return True


def guarded_move(src, dst, hand: str, *, root=None) -> None:
    """Move within this hand's root only, and only when the destination does not exist (no overwrite)."""
    _check_hand(hand)
    s, d = _abs(src), _abs(dst)
    _canon_guard(d)
    rs, rd = root_of(s, root), root_of(d, root)
    for x, r in ((s, rs), (d, rd)):
        if r is None:
            raise HandRootError(f"{x} is not inside a hand root — moves only inside hand roots")
        if r.name != hand:
            raise HandRootError(f"{x} belongs to hand {r.name} — run hand {hand} cannot move it")
        if x == r or r.is_relative_to(x):
            raise HandRootError(f"{x} is a hand root itself (or above it)")
    if not s.exists():
        raise HandRootError(f"nothing to move: {s}")
    if d.exists() or d.is_symlink():
        raise HandRootError(f"destination exists: {d} — not overwritten (no re-creation under the same name)")
    d.parent.mkdir(parents=True, exist_ok=True)
    os.rename(s, d)
    _say(f"mv hand={hand} {s} → {d}")


def guarded_import(src, dst, hand: str, *, root=None) -> None:
    """Work area -> hand root (a campaign cell's finalize): the source strictly inside one of
    `data_layout.RUN_WORK_AREAS` (not the area itself, not above it, not inside a hand root), the destination inside
    this hand's root (not the root itself), only when the destination does not exist. `os.rename` only: across file
    systems it fails (no copy + delete fallback)."""
    _check_hand(hand)
    s, d = _abs(src), _abs(dst)
    _canon_guard(d)
    base = _base(root)
    if root_of(s, root) is not None:
        raise HandRootError(
            f"{s} is inside a hand root — imports come only from work areas (inside a hand root use guarded_move)"
        )
    areas = [base / a for a in DL.RUN_WORK_AREAS.values()]
    if not any(s != a and s.is_relative_to(a) for a in areas):
        raise HandRootError(f"{s} is not inside a work area ({sorted(DL.RUN_WORK_AREAS.values())}) — not imported")
    rd = root_of(d, root)
    if rd is None or rd.name != hand:
        raise HandRootError(f"{d} is not inside the root of hand {hand} — not imported")
    if d == rd or rd.is_relative_to(d):
        raise HandRootError(f"{d} is a hand root itself (or above it)")
    if not s.exists() and not s.is_symlink():
        raise HandRootError(f"nothing to import: {s}")
    if d.exists() or d.is_symlink():
        raise HandRootError(f"destination exists: {d} — not overwritten")
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(s, d)
    except OSError as e:
        raise HandRootError(
            f"import rename failed {s} -> {d}: {e} — across file systems it is not replaced by a copy"
        ) from None
    _say(f"import hand={hand} {s} → {d}")
