"""The identity of the triangle ruler (collider_shape) that a cached reading was measured with.

Its owner is the ruler (collider_shape); retarget/f0/repair_sep reads it from here for its cache key.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def ruler_id() -> str:
    """The ruler's identity: `blobs_id` of the modules collider_shape.RULER_MODULES names, in that order."""
    import importlib  # noqa: PLC0415

    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415 — owner of the ruler

    return blobs_id([importlib.import_module(f"dexengine.data.objects.{m}").__file__ for m in CS.RULER_MODULES])


def blobs_id(files: list) -> str:
    """sha1 over the git blob id (sha1) of each file, in the given order: one changed byte in any of them changes it."""
    h = hashlib.sha1()
    for f in files:
        b = Path(f).read_bytes()
        h.update(hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest().encode() + b"\n")
    return h.hexdigest()
