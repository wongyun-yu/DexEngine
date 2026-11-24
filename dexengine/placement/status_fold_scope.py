"""Ledger axes of a fold scope: the keys candidates.axis_sets makes for the datasets in the scope (status_fold)."""

from __future__ import annotations

import os

from dexengine.core import paths as P
from dexengine.data import candidates as TC
from dexengine.data import rules as RR


def axis_keys(scope: set) -> tuple[str, ...]:
    """Ledger axes = the keys candidates.axis_sets actually makes, for the datasets in `scope` only. Only a missing
    base layer (BaseLayerMissing) is skipped; any other failure dies."""
    out = []
    hand = RR.hand_dir()
    for ds in RR.DATASET_PROFILE:
        if ds not in scope:
            continue
        for emb in ("bimanual", "right", "left"):
            if not os.path.isdir(os.path.join(P.main_root(), "data", ds, "base", hand, emb)):
                continue
            try:
                keys = TC.axis_sets(ds, hand, emb).keys()
            except TC.BaseLayerMissing:
                continue
            for k in keys:
                if k not in out:
                    out.append(k)
    return tuple(out)
