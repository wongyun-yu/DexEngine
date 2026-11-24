"""Shipped inputs: intermediates the release ships because the public code cannot rebuild them identically from the
raw datasets. The list and each input's content digest live in one place, `configs/release/shipped_inputs.tsv`.

Row fields:
  name      identity of the input (used by its consumer and by the from-raw gate)
  path      its place under the data root (the folder the input is unpacked into)
  glob      the files of that folder that belong to the input, a pathlib glob (other files there are produced by the
            pipeline)
  digest    SHA-256 of the input's checksum file `<path>/SHA256SUMS.<name>` (one `<sha256>  <relative path>` line per
            file, sorted by path, sha256sum format)
  consumer  the module that reads it
  why       why the public code cannot make it (a measurement, not an opinion)

A consumer calls `state(name)`: `verified` when the checksum file is present, its digest equals the registered one and
every listed file has its sha256 (no file of the glob missing or extra); `site` when the folder carries no checksum
file (an unverified data root that holds the inputs' sources). Any other state dies: a checksum file with
another digest, a changed, missing or extra file. The consumer prints `[SHIPPED] <name> state=<state> …`; the from-raw
gate requires `verified` for every input (a verified root is never read as an unverified one).

CLI (repository root):
  python -m dexengine.core.shipped sums <name> --root <folder>     write the checksum file, print the digest
  python -m dexengine.core.shipped verify <name> [--root <folder>] check one input, rc 0 = verified
  python -m dexengine.core.shipped list                            the registry
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.hashing import sha256_file
from dexengine.core.repo import CONFIGS_DIR

REGISTRY = CONFIGS_DIR / "release" / "shipped_inputs.tsv"
HEAD = ("name", "path", "glob", "digest", "consumer", "why")
SUMS_PREFIX = "SHA256SUMS."
VERIFIED, SITE = "verified", "site"
TAG = "[SHIPPED]"


class ShippedInputError(SystemExit):
    """A shipped input that is present but is not the registered one (exits like SystemExit(msg))."""


def rows() -> dict[str, dict]:
    """The registry {name: row}. A malformed table dies."""
    lines = [ln for ln in REGISTRY.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    if not lines or tuple(lines[0].split("\t")) != HEAD:
        raise ShippedInputError(f"{TAG} {REGISTRY}: header must be {HEAD}")
    out: dict[str, dict] = {}
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) != len(HEAD) or not all(f[:4]):
            raise ShippedInputError(f"{TAG} {REGISTRY}: row {ln[:80]!r} needs {len(HEAD)} non-empty fields")
        if f[0] in out:
            raise ShippedInputError(f"{TAG} {REGISTRY}: {f[0]} registered twice")
        out[f[0]] = dict(zip(HEAD, f, strict=True))
    return out


def row(name: str) -> dict:
    r = rows().get(name)
    if r is None:
        raise ShippedInputError(f"{TAG} unknown shipped input {name!r} — registered: {sorted(rows())}")
    return r


def _root(name: str, root: str | os.PathLike | None) -> Path:
    if root is not None:
        return Path(root)
    from dexengine.core.locations.roots import main_root  # noqa: PLC0415 — the data root is read only when needed

    return main_root() / row(name)["path"]


def sums_path(name: str, root: str | os.PathLike | None = None) -> Path:
    return _root(name, root) / f"{SUMS_PREFIX}{name}"


def members(name: str, root: str | os.PathLike | None = None) -> list[str]:
    """Relative paths of the files under the input's folder that match its glob (pathlib glob: `*` stays inside one
    folder level), sorted."""
    base, pat = _root(name, root), row(name)["glob"]
    found = (p for p in base.glob(pat) if p.is_file() and not p.name.startswith(SUMS_PREFIX))
    return sorted(p.relative_to(base).as_posix() for p in found)


def write_sums(name: str, root: str | os.PathLike) -> str:
    """Write the input's checksum file from the files present; returns its digest (the value the registry holds)."""
    lines = [f"{sha256_file(Path(root) / rel)}  {rel}\n" for rel in members(name, root)]
    if not lines:
        raise InputMissingError(f"{TAG} {name}: no file under {root} matches {row(name)['glob']!r}")
    p = sums_path(name, root)
    p.write_text("".join(lines), encoding="utf-8")
    return sha256_file(p)


def state(name: str, root: str | os.PathLike | None = None) -> str:
    """`verified` or `site` (module doc); every other case dies. Prints the marker line."""
    r, base = row(name), _root(name, root)
    if not base.is_dir():
        raise InputMissingError(f"{TAG} {name}: {base} is missing — unpack the shipped input there ({r['why']})")
    sp = sums_path(name, base)
    if not sp.is_file():
        print(f"{TAG} {name} state={SITE} root={base} (no {sp.name}: not a release root)", flush=True)
        return SITE
    got = sha256_file(sp)
    if got != r["digest"]:
        raise ShippedInputError(f"{TAG} {name}: {sp} digest {got[:12]} ≠ registered {r['digest'][:12]}")
    listed = {}
    for ln in sp.read_text(encoding="utf-8").splitlines():
        sha, _, rel = ln.partition("  ")
        listed[rel] = sha
    present = set(members(name, base))
    if missing := sorted(set(listed) - present):
        raise InputMissingError(f"{TAG} {name}: {len(missing)} listed file(s) missing under {base}, e.g. {missing[0]}")
    if extra := sorted(present - set(listed)):
        raise ShippedInputError(f"{TAG} {name}: {len(extra)} file(s) not in {sp.name} under {base}, e.g. {extra[0]}")
    bad = [rel for rel, sha in listed.items() if sha256_file(base / rel) != sha]
    if bad:
        raise ShippedInputError(f"{TAG} {name}: {len(bad)} file(s) differ from {sp.name}, e.g. {bad[0]}")
    print(f"{TAG} {name} state={VERIFIED} digest={got[:12]} files={len(listed)} root={base}", flush=True)
    return VERIFIED


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dexengine.core.shipped", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sums")
    s.add_argument("name")
    s.add_argument("--root", required=True)
    v = sub.add_parser("verify")
    v.add_argument("name")
    v.add_argument("--root", default=None)
    sub.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for r in rows().values():
            print(f"{r['name']}\t{r['path']}\t{r['digest'][:12]}\t{r['why']}")
        return 0
    if a.cmd == "sums":
        print(f"{TAG} {a.name} digest={write_sums(a.name, a.root)}")
        return 0
    return 0 if state(a.name, a.root) == VERIFIED else 1


if __name__ == "__main__":
    sys.exit(main())
