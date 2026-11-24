"""Object USD places of a dataset: the layout layers that hold object USDs and the kind each holds (USD_LAYERS), their
roots, the file names, and the one enumeration of the USDs present. Paths through hull.paths_of (artifact_paths with the
data root = paths.main_root())."""

from __future__ import annotations

import os
from pathlib import Path

from dexengine.core import artifact_paths as AP
from dexengine.core import paths as P
from dexengine.data.objects import hull as OH

#: (layout layer, articulated) of the object USD places — rigid = usd layer, articulated = usd_artic layer. Read by
#: usd_roots and by the tools that walk the layers relative to another root (robots.objects.overlay_manifest ·
#: a layer rename).
USD_LAYERS = (("usd", False), ("usd_artic", True))


def usd_roots(ds: str) -> tuple[tuple[bool, Path], ...]:
    """((articulated, root), …) — the two object USD places (USD_LAYERS) under the data root."""
    t = OH.paths_of(ds)
    lay = P.data_layout
    return tuple((art, Path(lay.data_abspath(lay.render(layer, t.family)))) for layer, art in USD_LAYERS)


def usd_name(ds: str, articulated: bool, render: bool = False) -> str:
    """File name of an object USD of that kind (training USD, or its render overlay with `render`)."""
    f = OH.paths_of(ds).usd_file("x", articulated)
    return os.path.basename(AP.render_usd_file(f) if render else f)


def object_usds(ds: str) -> list[tuple[str, bool, Path]]:
    """[(mesh id, articulated, USD path)] of every object USD present — the one enumeration (cooked hull targets ·
    decomposition audit). The usd layer must exist; the usd_artic layer may be absent (nothing baked yet)."""
    out = []
    for art, root in usd_roots(ds):
        if not root.is_dir():
            if not art:
                raise FileNotFoundError(f"{ds}: usd layer {root} is missing")
            continue
        out.extend((d.name, art, f) for d in sorted(root.iterdir()) if (f := OH.usd_file(ds, d.name, art)).is_file())
    return out
