"""Stage side of stage-1 retargeting (`reretarget`): the marker constants the campaign driver and `admit` read,
the file helpers, the scope files, the stage manifest and the stage input links (`--manifest-only`, `--unstage`)."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import sys
from pathlib import Path

from dexengine.core import paths as P

#: marker lines the stage-1 driver reads: scope (+ mode) and the count result
SCOPE = "[reretarget] scope ds="
MANIFEST_ONLY = "(manifest only)"
UNSTAGE = "(unstage links)"
MISMATCH = "★ count mismatch"


def die(msg: str, rc: int = 2):
    print(msg, file=sys.stderr, flush=True)
    raise SystemExit(rc)


def canon_set(names, ds: str) -> set[str]:
    """Names in their joinable form (``paths.canonical_clip``); empty lines are skipped."""
    return {P.canonical_clip(n, ds) for n in names if n}


def ls_visible(d: Path) -> list[str]:
    """Visible entries of a folder (empty when it does not exist)."""
    try:
        return sorted(n for n in os.listdir(d) if not n.startswith("."))
    except OSError:
        return []


def read_lines(f: Path) -> list[str]:
    return f.read_text(encoding="utf-8").splitlines()


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scope_takes(scope: Path, ds: str, emb: str | None = None, input_dir: Path | None = None) -> list[str]:
    """The scope's embs of ``ds`` (no ``emb``) or the input folders of ``input_dir`` in scope; a bad scope is rc 2."""
    try:
        return P.scope_takes(scope, ds, emb, input_dir)
    except (OSError, ValueError) as e:
        die(f"[RERETARGET_SCOPE] ✗ {e}")


def write_scope(f: Path, takes: list[str]) -> None:
    f.write_text("\n".join(takes) + "\n")


def ds_src(ds: str, hand: str) -> Path:
    """The delivered dataset root ``data/<ds>`` that the stage input links point into (linking and unlinking use it).
    Derived from the MANO layer without depending on an emb existing (two levels up, as strings)."""
    axes = {"hand": P.hand_spec_dir(P.hand_id(hand)), "emb": "bimanual"}
    with contextlib.suppress(ValueError):
        axes["combo"] = P.combo_for(hand)
    return Path(os.path.dirname(os.path.dirname(str(P.layer_path("mano", ds, **axes)))))


# stage manifest and stage links
def write_manifest(stage: Path, scope: str) -> None:
    """``processed/MANIFEST.sha256`` = content sha256 (links followed) and stage-relative path of every file under
    ``scope`` (``.`` or ``./<ds>``). With ``./<ds>`` only that dataset's lines are replaced (the ingest reads ``<ds>/``
    lines only). Under a lock, written to a temporary file and moved into place."""
    proc = stage / "processed"
    mf, tmp = proc / "MANIFEST.sha256", stage / f".MANIFEST.tmp.{os.getpid()}"
    with open(stage / ".manifest.lock", "w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX)
        except OSError:
            die("[reretarget] ✗ manifest lock failed")
        lines: list[str] = []
        if scope != "." and mf.is_file():
            pre = scope[2:] + "/"
            for ln in read_lines(mf):
                i = ln.find("  ")
                if not (ln[i + 2 :] if i >= 0 else ln[1:]).startswith(pre):
                    lines.append(ln)
        try:
            rels = []
            for root, _dirs, files in os.walk(proc / scope, followlinks=True, onerror=_raise):
                for f in files:
                    p = os.path.join(root, f)
                    if f != "MANIFEST.sha256" and os.path.isfile(p):
                        rels.append(os.path.relpath(p, proc))
            lines += [f"{_sha(proc / r)}  {r}" for r in sorted(rels)]
            tmp.write_text("".join(ln + "\n" for ln in lines))
            os.replace(tmp, mf)
        except OSError:
            tmp.unlink(missing_ok=True)
            die("[reretarget] ✗ manifest write failed")
    n = len(lines) if scope == "." else sum(f"  {scope[2:]}/" in ln for ln in lines)
    print(f"[reretarget] manifest {n} entries (scope {scope}) · whole file {len(lines)}", flush=True)


def _raise(e: OSError):
    raise e


def stage_links(p: Path) -> list[tuple[str, str]]:
    """(path, link text) of every symbolic link under ``p`` (links are not followed)."""
    out = []
    for root, dirs, files in os.walk(p):
        for n in dirs + files:
            q = os.path.join(root, n)
            if os.path.islink(q):
                out.append((q, os.readlink(q)))
    return out


def unstage(stage: Path, ds: str, hand: str) -> int:
    """Remove the stage input links of ``ds`` once ingest and stamps are done. Rule: every link must point into the
    delivered dataset (``<src>/mano/…`` or ``<src>/assets/objects``); if one does not, none is removed (rc 2).
    Outputs are real files and stay; retargeting again links again."""
    p = stage / "processed" / ds
    if not p.is_dir():
        die(f"[reretarget] ✗ {ds} is not in the stage: {p}")
    src = ds_src(ds, hand)
    links = stage_links(p)
    bad = [f"{q}\t{t}" for q, t in links if not t.startswith(f"{src}/mano/") and t != f"{src}/assets/objects"][:3]
    if bad:
        die(
            f"[reretarget] ✗ {ds}: stage links point outside the delivered dataset ({src}) — none removed: "
            + " ".join(bad)
        )
    try:
        for q, _ in links:
            os.unlink(q)
        (p / ".stage_links").unlink(missing_ok=True)
    except OSError:
        die(f"[reretarget] ✗ {ds}: could not remove the links")
    left = len(stage_links(p))
    if left:
        die(f"[reretarget] ✗ {ds}: {left} links are left after removal")
    print(f"{SCOPE}{ds} {UNSTAGE} {len(links)} links removed · 0 left", flush=True)
    return 0
