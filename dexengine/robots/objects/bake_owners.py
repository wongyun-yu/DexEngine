"""Owner modules the object USD bake reads (all import without Isaac): the data layout, the canonical decomposition
and the artifact path rules."""

from __future__ import annotations


def _data_layout():
    """The data layout (``dexengine.core.layout``, stdlib-only)."""
    from dexengine.core import layout

    return layout


def _decompose():
    """The single owner of the decomposition parameters and fingerprint (its top-level imports are stdlib-only)."""
    from dexengine.retarget import decompose

    return decompose


def _taco_paths():
    """Path owner (articulated USD file name · asset id): ``dexengine.core.artifact_paths``."""
    from dexengine.core import artifact_paths

    return artifact_paths
