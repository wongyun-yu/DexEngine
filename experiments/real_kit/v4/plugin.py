"""Record hooks of the v4 real-kit experiments: bake reuse (augment.place_gen --bake-from) and the approach physics
factory of augment sessions. Enable with DEXENGINE_RECORD_PLUGINS=experiments.real_kit.v4.plugin."""

from __future__ import annotations

import os

from dexengine.record import hooks


def _bake_from(src, shadow, recs, v2, main, log_dir):
    from experiments.real_kit.v4 import bake

    return bake.adopt_all(src, shadow, recs, v2, main, log_dir)


def _approach():
    if not os.environ.get("AUG_V4_APPROACH"):
        return None
    from experiments.real_kit.v4 import phys

    return phys.from_env()


hooks.register("bake", "bake_from", _bake_from)
hooks.register("aug_session", "v4_approach", _approach)
