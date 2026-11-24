#!/usr/bin/env python3
"""Flatten an instanceable USD (with sublayers) into a single-layer USD.

This is the Stage-4 glue that turns UrdfConverter's instanceable output
(a small root USD referencing configuration/{base,physics,robot,sensor}.usd)
into the single-layer USD expected by the mimic-bake scripts.

Pure pxr — no isaaclab.app / SimulationApp required. Run with any Python
that has usd-core installed, e.g.:

  ../IsaacLab/isaaclab.sh -p dexengine/robots/usd/flatten.py \\
      --input  data/usd_cache/robots/full_robot_gen/<combo>/<combo>_base_test.usd \\
      --output data/usd_cache/robots/full_robot_gen/<combo>/<combo>_flat_test.usd

Relative paths resolve against the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

from pxr import Sdf, Usd

from dexengine.core.repo import REPO_ROOT

_PROJECT_ROOT = str(REPO_ROOT)


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_PROJECT_ROOT, path))


def _assert_safe_output(output_usd_abs: str, allow_commit_write: bool) -> None:
    from dexengine.core.paths import main_root  # noqa: PLC0415 — the data root (env DEXENGINE_MAIN_ROOT)

    usd_cache = os.path.normpath(
        os.path.join(str(main_root()), "data", "usd_cache")
    )  # the cache is under the data root
    fname = os.path.basename(output_usd_abs)
    if output_usd_abs.startswith(usd_cache + os.sep) and "_test" not in fname and not allow_commit_write:
        print(
            f"[error] Refusing to write non-test output to {output_usd_abs} — "
            f"use a filename containing '_test' to prevent overwriting committed USDs, "
            f"or pass --allow-commit-write.",
            file=sys.stderr,
        )
        sys.exit(1)
    if os.path.exists(output_usd_abs):
        print(
            f"[error] Output already exists: {output_usd_abs}. Remove it first.",
            file=sys.stderr,
        )
        sys.exit(1)


def _sublayer_sizes(stage: Usd.Stage) -> list[tuple[str, int]]:
    results = []
    root_layer = stage.GetRootLayer()
    for sub_path in root_layer.subLayerPaths:
        resolved = root_layer.ComputeAbsolutePath(sub_path)
        if os.path.exists(resolved):
            results.append((sub_path, os.path.getsize(resolved)))
        else:
            results.append((sub_path, -1))
    return results


#: A prototype root prim ``Stage.Flatten`` writes for each instancing prototype (and every path that names it).
_PROTO = re.compile(r"Flattened_Prototype_(\d+)(?!\d)")
_PROTO_PREFIX = "Flattened_Prototype_"
#: Marker of the renumbering (counts before = after are the proof that only names moved).
CANON_MARK = "[FLATTEN_CANON]"


def _top_blocks(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Split a usda text into its header lines and its top-level prim blocks ``(name, lines)``: a block starts at a
    column-0 ``def``/``over``/``class`` line and ends at the next column-0 ``}`` (``ExportToString``'s layout)."""
    head: list[str] = []
    blocks: list[tuple[str, list[str]]] = []
    cur: list[str] | None = None
    name = ""
    for ln in text.splitlines():
        if cur is None:
            m = re.match(r'(def|over|class)\b[^"]*"([^"]+)"', ln)
            if m:
                cur, name = [ln], m.group(2)
            elif blocks and not ln.strip():
                continue  # blank separators between blocks are rewritten below
            else:
                if blocks:
                    raise ValueError(f"{CANON_MARK} text outside a top-level prim after the first prim: {ln[:80]!r}")
                head.append(ln)
            continue
        cur.append(ln)
        if ln == "}":
            blocks.append((name, cur))
            cur = None
    if cur is not None:
        raise ValueError(f"{CANON_MARK} unterminated top-level prim {name!r}")
    return head, blocks


def _proto_refs(layer: Sdf.Layer) -> dict[str, list[str]]:
    """Prototype root name → the prim paths whose references name it (internal references, any list-op slot)."""
    out: dict[str, list[str]] = {}

    def visit(path: Sdf.Path) -> None:
        if not path.IsPrimPath():
            return
        spec = layer.GetPrimAtPath(path)
        lo = spec.referenceList
        for item in [*lo.explicitItems, *lo.addedItems, *lo.prependedItems, *lo.appendedItems]:
            p = item.primPath
            if not item.assetPath and p and p.pathString.startswith("/" + _PROTO_PREFIX):
                out.setdefault(p.pathString.split("/")[1], []).append(path.pathString)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return out


def flatten_canonical(stage: Usd.Stage) -> Sdf.Layer:
    """``stage.Flatten()`` with bytes that do not depend on the run.

    Measured: the same input flattened four times to the same path gives four different files
    (``setarch -R`` too); the usda texts differ only in which mesh each ``Flattened_Prototype_N`` holds (no path
    line differs; with the numbers removed the sorted lines are equal). So the prototypes are renumbered by
    content: key = sha256 of the prototype's block with every prototype number removed, then the smallest prim
    path that references it (a tie on both keys is kept only where the order cannot change the text — see the
    tie rule in canonical_layer). The names, every path naming them and the order of the prototype blocks follow
    that key; the layer documentation (``Flatten`` writes the root layer's absolute
    path there) is cleared. Prototype and reference counts before and after are printed (``CANON_MARK``) and must
    match. The returned layer is anonymous (usda text re-imported); the caller exports it."""
    return canonical_layer(stage.Flatten())


def canonical_layer(flat: Sdf.Layer) -> Sdf.Layer:
    """The renumbering of `flatten_canonical` on an already flattened layer (a new anonymous layer; ``flat`` is
    read only — its documentation is not copied)."""
    refs = _proto_refs(flat)
    doc = flat.documentation
    flat.documentation = ""
    try:
        text = flat.ExportToString()
    finally:
        flat.documentation = doc
    head, blocks = _top_blocks(text)
    protos = [(n, b) for n, b in blocks if _PROTO.fullmatch(n)]
    others = [(n, b) for n, b in blocks if not _PROTO.fullmatch(n)]
    keys = {}
    for n, b in protos:
        h = hashlib.sha256(_PROTO.sub(_PROTO_PREFIX + "#", "\n".join(b)).encode()).hexdigest()
        keys[n] = (h, min(refs.get(n, [""])))
    # A tie (same content, same referencing path) is allowed only where the order cannot show: the tied blocks are
    # equal up to their own number and name no other prototype — e.g. the unreferenced copies a stage-3 Flatten keeps
    # of the stage-2 prototypes (measured: 234 prototype roots, 117 referenced, 117 unreferenced copies, two of
    # which share content). Any other tie stops the bake.
    tied: dict[tuple, list[str]] = {}
    for n, k in keys.items():
        tied.setdefault(k, []).append(n)
    by_block = dict(protos)
    for ns in tied.values():
        if len(ns) > 1 and any({m.group(0) for m in _PROTO.finditer("\n".join(by_block[n]))} - {n} for n in ns):
            raise ValueError(
                f"{CANON_MARK} prototypes {ns[:3]} share content and referencing path and name other "
                "prototypes — no canonical order"
            )
    order = sorted(keys, key=keys.get)
    new = {n.removeprefix(_PROTO_PREFIX): str(i + 1) for i, n in enumerate(order)}
    ren = lambda s: _PROTO.sub(lambda m: _PROTO_PREFIX + new[m.group(1)], s)  # noqa: E731
    by_name = dict(protos)
    parts = ["\n".join(ren(x) for x in head)]
    parts += ["\n".join(ren(x) for x in b) for _n, b in others]
    parts += ["\n".join(ren(x) for x in by_name[n]) for n in order]
    text = "\n\n".join(parts) + "\n"
    out = Sdf.Layer.CreateAnonymous(".usda")
    if not out.ImportFromString(text):
        raise ValueError(f"{CANON_MARK} the renumbered text does not parse")
    refs_after = _proto_refs(out)
    n_ref = sum(map(len, refs.values()))
    n_ref_after = sum(map(len, refs_after.values()))
    moved = sum(1 for k, v in new.items() if k != v)
    n_after = len([p for p in out.rootPrims if _PROTO.fullmatch(p.name)])
    print(f"{CANON_MARK} protos {len(protos)} → {n_after} · refs {n_ref} → {n_ref_after} · renamed {moved}")
    if n_after != len(protos) or len(refs_after) != len(refs) or n_ref_after != n_ref:
        raise ValueError(
            f"{CANON_MARK} counts changed: protos {len(protos)} → {n_after} · referenced {len(refs)} → "
            f"{len(refs_after)} · refs {n_ref} → {n_ref_after}"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten an instanceable USD into a single-layer USD.")
    parser.add_argument("--input", type=str, required=True, help="Input instanceable USD.")
    parser.add_argument("--output", type=str, required=True, help="Output flattened USD.")
    parser.add_argument(
        "--allow-commit-write",
        action="store_true",
        default=False,
        help="Allow writing canonical filenames under data/usd_cache/.",
    )
    args = parser.parse_args()

    input_abs = _resolve(args.input)
    output_abs = _resolve(args.output)

    if not os.path.exists(input_abs):
        print(f"[error] input USD not found: {input_abs}", file=sys.stderr)
        sys.exit(1)

    _assert_safe_output(output_abs, args.allow_commit_write)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)

    print(f"[info] input:  {input_abs}")
    print(f"[info] output: {output_abs}")
    input_size = os.path.getsize(input_abs)
    print(f"[info] input root size: {input_size:,} bytes")

    stage = Usd.Stage.Open(input_abs)
    if stage is None:
        print(f"[error] failed to open stage: {input_abs}", file=sys.stderr)
        sys.exit(1)

    sublayers = _sublayer_sizes(stage)
    if sublayers:
        print("[info] sublayers:")
        for sub_path, size in sublayers:
            size_s = f"{size:,}" if size >= 0 else "MISSING"
            print(f"  - {sub_path}: {size_s} bytes")

    flat_layer = flatten_canonical(stage)
    flat_layer.Export(output_abs)

    out_size = os.path.getsize(output_abs)
    print(f"[ok] wrote flattened USD: {output_abs}")
    print(f"[info] output size: {out_size:,} bytes")

    total_sub = sum(s for _, s in sublayers if s >= 0)
    if total_sub:
        print(f"[info] input root + sublayers total: {input_size + total_sub:,} bytes")


if __name__ == "__main__":
    main()
