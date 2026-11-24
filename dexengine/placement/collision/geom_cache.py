#!/usr/bin/env python3
"""Content-keyed disk cache of clip-independent geometry — the one owner of the geometry cache directory and its rule.

What it holds: values that are pure functions of file bytes and code — a mesh file's arrays (``mesh_arrays``), a
reduced convex hull (robots.wrist_hull.reduce_hull) and the torso decomposition (torso_split.torso_parts), so a new
process reads them instead of recomputing (trimesh parses, qhull, the decomposition ruler).

Rule (one, for every kind):
  key   = sha256 of (kind · the code identity · the input bytes). The code identity is the syntax (AST without
          docstrings — ``syntax``) of the code that computes the value plus the numeric library versions, so a code
          edit or another interpreter (numpy · scipy · trimesh) never reads an old value; a comment edit keeps it.
          Inputs are bytes (file contents, arrays), never names or modification times.
  value = a tree of dict (str keys) · list · tuple · str · int · float · bool · None · ndarray · numpy scalar, stored
          as one ``.npz`` (arrays + a JSON skeleton, no pickle) — every number reads back bit-identical.
  write = temp file + os.replace (a reader never sees half a file; concurrent writers of one key write the same bytes).
  read  = the stored key must equal the asked key; anything unreadable (truncated, another layout) is recomputed and
          rewritten, with a ``[GEOM_CACHE] corrupt`` line on stderr — never a silent skip and never a wrong value.
Directory: env ``HH_MESH_CACHE`` (default /tmp/hh_mesh_cache) — cross_sets.MESH_CACHE reads it here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

CACHE_ENV = "HH_MESH_CACHE"
CACHE_DEFAULT = "/tmp/hh_mesh_cache"
_REV: dict = {}  # code identity per (kind, files) — clip-independent
_MESH: dict = {}  # file_identity → mesh_arrays value (read-only arrays)
_SERVER_KEEP = ("_REV", "_MESH")


def cache_dir() -> Path:
    """The geometry cache directory (env HH_MESH_CACHE, else /tmp/hh_mesh_cache)."""
    return Path(os.environ.get(CACHE_ENV, CACHE_DEFAULT))


def syntax(src: str) -> str:
    """The AST of a source text with docstrings dropped (comments never reach it) — what a code identity hashes."""
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(src))
    for nd in ast.walk(tree):
        body = getattr(nd, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            nd.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def code_rev(files: tuple, libs: tuple = ("numpy", "scipy", "trimesh")) -> str:
    """Identity of the code computing a value: the syntax of each source file + the versions of ``libs``."""
    k = (tuple(str(f) for f in files), libs)
    if k not in _REV:
        import importlib

        h = hashlib.sha256()
        for f in files:
            h.update(syntax(Path(f).read_text(encoding="utf-8")).encode())
        for lib in libs:
            h.update(f"|{lib}={importlib.import_module(lib).__version__}".encode())
        _REV[k] = h.hexdigest()
    return _REV[k]


def _enc(o, arrs: list):
    if isinstance(o, np.ndarray):
        if o.dtype == object:
            raise TypeError("geom_cache: object arrays are not stored")
        arrs.append(o)
        return {"a": len(arrs) - 1}
    if isinstance(o, np.generic):
        arrs.append(np.asarray(o))
        return {"s": len(arrs) - 1}
    if isinstance(o, tuple):
        return {"t": [_enc(x, arrs) for x in o]}
    if isinstance(o, list):
        return {"l": [_enc(x, arrs) for x in o]}
    if isinstance(o, dict):
        if not all(isinstance(k, str) for k in o):
            raise TypeError("geom_cache: dict keys must be str")
        return {"d": [[k, _enc(v, arrs)] for k, v in o.items()]}
    if o is None or isinstance(o, (bool, int, float, str)):
        return {"v": o}
    raise TypeError(f"geom_cache: cannot store a {type(o).__name__}")


def _dec(o, z):
    ((tag, v),) = o.items()
    if tag == "a":
        return z[f"a{v}"]
    if tag == "s":
        return z[f"a{v}"][()]
    if tag == "t":
        return tuple(_dec(x, z) for x in v)
    if tag == "l":
        return [_dec(x, z) for x in v]
    if tag == "d":
        return {k: _dec(x, z) for k, x in v}
    if tag == "v":
        return v
    raise ValueError(f"geom_cache: unknown tag {tag}")


def _read(f: Path, key: str):
    with np.load(f, allow_pickle=False) as z:
        if str(z["__key"][()]) != key:
            raise ValueError("key mismatch")
        arrs = {k: np.array(z[k]) for k in z.files if k.startswith("a")}
        return _dec(json.loads(str(z["__tree"][()])), arrs)


def _write(f: Path, key: str, value) -> None:
    arrs: list = []
    tree = json.dumps(_enc(value, arrs), ensure_ascii=False, allow_nan=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.name}.{os.uname().nodename}.{os.getpid()}.tmp")  # pods share pids on a volume
    with open(tmp, "wb") as fh:
        np.savez(fh, __key=np.array(key), __tree=np.array(tree), **{f"a{i}": a for i, a in enumerate(arrs)})
    os.replace(tmp, f)


def cached(kind: str, rev: str, parts, compute):
    """``compute()`` through the disk cache under key sha256(kind · rev · parts) — ``parts`` = an iterable of bytes /
    str / ndarray (arrays hash with dtype and shape)."""
    h = hashlib.sha256(f"{kind}|{rev}".encode())
    for p in parts:
        if isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(f"|{a.dtype.str}{a.shape}|".encode())
            h.update(a.tobytes())
        else:
            h.update(b"|" + (p if isinstance(p, bytes) else str(p).encode()))
    key = h.hexdigest()
    f = cache_dir() / f"{kind}_{key[:32]}.npz"
    if f.is_file():
        try:
            return _read(f, key)
        except Exception as e:  # noqa: BLE001 — any unreadable entry is recomputed (never a wrong value)
            print(
                f"[GEOM_CACHE] corrupt {f.name} ({type(e).__name__}: {str(e)[:80]}) — recomputing",
                file=sys.stderr,
                flush=True,
            )
    value = compute()
    try:
        _write(f, key, value)
    except OSError as e:
        print(f"[GEOM_CACHE] write failed {f} ({e}) — using the computed value", file=sys.stderr, flush=True)
    return value


def file_identity(fn) -> tuple:
    """(path, mtime ns, size, inode) — the in-process memo key of a file (a rewrite changes it); the disk key is the
    file's bytes."""
    st = os.stat(fn)
    return str(fn), st.st_mtime_ns, st.st_size, st.st_ino


def mesh_arrays(fn) -> tuple[np.ndarray, np.ndarray, str]:
    """(vertices float64 (n, 3), faces int64 (m, 3), sha256 of the file bytes) of a mesh file as
    ``trimesh.load(fn, force="mesh")`` reads it — the one cached mesh reader of the collider and ruler geometry. Once
    per file per process (file_identity); the arrays are read-only (a caller that changed them in place would change
    every later reader's value — it dies instead)."""
    k = file_identity(fn)
    if k not in _MESH:
        sha = hashlib.sha256(Path(fn).read_bytes()).hexdigest()

        def _load():
            import trimesh

            m = trimesh.load(str(fn), force="mesh")
            return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)

        v, f = cached("mesh", code_rev((__file__,), ("numpy", "trimesh")), (sha, Path(fn).suffix.lower()), _load)
        v.flags.writeable = f.flags.writeable = False
        _MESH[k] = (v, f, sha)
    return _MESH[k]
