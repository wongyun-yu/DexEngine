#!/usr/bin/env python3
"""Render overlay manifest — md5 · bytes · bake mark of render overlays, the list the render entry checks spawned overlays against.

Why: stale overlay copies on another machine (95 shells of 611 B) were loaded into renders without
textures. The render entry (core.artifact_paths.overlay_manifest_check) checks each spawned overlay's measured md5
against the list in force (core.artifact_paths.overlay_manifest_rule):
  canonical — a checkout with an operations layer: the list kept in git, written by a scan of the canonical data root with
              this tool and committed; it catches stale copies even on machines without the pack layer.
  data-root — a checkout without it: the list under the data root that its overlay bake writes —
              robots.objects.render_overlay calls record() with the overlays that run wrote (merged by key; no scan,
              so an overlay left on disk by another bake is not taken in and dies until baked again).
  project   — DEXENGINE_OVERLAY_MANIFEST names a project's own list in either tree.
All overlay rules are owned by artifact_paths: name (RENDER_USD_SUFFIX) · bake mark (overlay_bake_mark — objects
without a texture are listed with the mark 'none' and pass) · key (overlay_manifest_key). An overlay without a mark
block dies. Write = temp file → replace (the data root is printed, not stored: a list records no absolute
path). Never edit a list by hand.
Usage: DEXENGINE_MAIN_ROOT=<data root> python3 -m dexengine.robots.objects.overlay_manifest [--check]
     (scan every canonical dataset's object USD layers and write the canonical or project list; --check = compare only, rc 1 if
     any key differs; the data-root list is written by the overlay bake, not by a scan)
"""  # noqa: E501  # the first docstring line is the --help description (argparse reads line 1)

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

from dexengine.core import artifact_paths as TP
from dexengine.core import paths as P
from dexengine.data.objects import usd_places as UP

GENERATOR = "dexengine.robots.objects.overlay_manifest"


def entry(path: str) -> tuple[str, dict]:
    """(key, {md5 · bytes · bake · links}) of one overlay; an overlay without a mark block dies."""
    bm = TP.overlay_bake_mark(path)
    if bm is None:
        raise SystemExit(f"[OVERLAY_MANIFEST] FAIL overlay header has no texture mark block: {path}")
    with open(path, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()
    return TP.overlay_manifest_key(path), {"md5": md5, "bytes": os.path.getsize(path), "bake": bm[0], "links": bm[1]}


def build() -> dict:
    """Scan every canonical dataset's object USD layers (usd_places.usd_roots: rigid usd · articulated usd_artic) under
    the data root, each layer with its own overlay name."""
    lay = P.data_layout
    root = lay.data_abspath("")
    out = {}
    for ds in sorted(d for d, s in lay.DATASETS.items() if s.status == "canon"):
        for art, base in UP.usd_roots(ds):
            for p in sorted(glob.glob(os.path.join(str(base), "*", UP.usd_name(ds, art, render=True)))):
                k, v = entry(p)
                out[k] = v
    if not out:
        raise SystemExit(f"[OVERLAY_MANIFEST] FAIL no overlay under data root {root} — check DEXENGINE_MAIN_ROOT")
    return {
        "generator": GENERATOR,
        "data_root": root,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n": len(out),
        "overlays": out,
    }


def _write(path: str, m: dict) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump({k: v for k, v in m.items() if k != "data_root"}, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def record(paths: list[str]) -> str:
    """Merge the overlays one bake wrote (paths) into the list in force by key and return the marker line. The canonical
    git list is not written by a bake (a scan of the canonical data root writes it); its marker says so."""
    rule, mf = TP.overlay_manifest_rule()
    if rule == "canonical":
        return (
            f"[OVERLAY_MANIFEST] canonical list not written by a bake — after a rebake of the canonical data root "
            f"rerun python3 -m {GENERATOR} and commit · {mf}"
        )
    m = {"generator": GENERATOR, "overlays": {}}
    if os.path.isfile(mf):
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
    for p in paths:
        k, v = entry(p)
        m["overlays"][k] = v
    m.update(
        generator=GENERATOR,
        data_root=P.data_layout.data_abspath(""),
        n=len(m["overlays"]),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    _write(mf, m)
    return f"[OVERLAY_MANIFEST] recorded {len(paths)} · list {m['n']} · {rule} list {mf}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--check", action="store_true", help="do not write; only compare with the current list (rc 1 if different)"
    )
    a = ap.parse_args()
    rule, out = TP.overlay_manifest_rule()
    m = build()
    cnt = {k: sum(1 for v in m["overlays"].values() if v["bake"] == k) for k in ("applied", "solid", "none")}
    if a.check:
        old = {}
        if os.path.isfile(out):
            with open(out) as f:
                old = json.load(f)["overlays"]
        new = m["overlays"]
        diff = sorted(k for k in set(old) | set(new) if old.get(k, {}).get("md5") != new.get(k, {}).get("md5"))
        print(
            f"[OVERLAY_MANIFEST] check root {m['data_root']} · overlays {m['n']} {cnt} · {rule} list {out} · "
            f"differs from list {len(diff)} {diff[:5]}"
        )
        return 1 if diff else 0
    if rule == "data-root":
        print(
            f"[OVERLAY_MANIFEST] FAIL the data-root list ({out}) is written by the overlay bake for the overlays it "
            "wrote — a scan would take in overlays left by another bake; bake again (robots.objects.render_overlay)",
            file=sys.stderr,
        )
        return 2
    _write(out, m)
    print(f"[OVERLAY_MANIFEST] OK overlays {m['n']} {cnt} · root {m['data_root']} · {rule} list {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
