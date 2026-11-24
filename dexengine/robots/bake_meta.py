"""The bake metadata of a baked robot USD (Isaac's UrdfConverter `config.yaml` next to it) — re-exported by
robots.selection, whose generation check reads it."""

from __future__ import annotations

import pathlib


def bake_meta_file(usd_path: str) -> pathlib.Path:
    """The bake metadata Isaac's UrdfConverter leaves next to the USD (``config.yaml``).
    Readers: ``baked_urdf_path`` (generation check · head camera) and ``dexengine/placement/bundle/run_inputs.py``."""
    return pathlib.Path(usd_path).parent / "config.yaml"


def baked_urdf_path(usd_path) -> pathlib.Path | None:
    """The URDF the USD was baked from (absolute, as the converter recorded it), read from the bake metadata
    ``config.yaml``.

    A USD does not record its own generation (even the stage name is the same for every variant); the metadata's
    ``asset_path`` points at the source URDF and is the only USD↔URDF link left on disk. Reads one line (no yaml
    dependency). None when absent (a copied or post-processed body). Readers: ``_urdf_of_baked_usd`` and
    ``robots.usd.head_camera.body_urdf``.
    """
    cfg = bake_meta_file(usd_path)
    try:
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("asset_path:"):
                return pathlib.Path(line.split(":", 1)[1].strip())
    except OSError:
        return None
    return None
