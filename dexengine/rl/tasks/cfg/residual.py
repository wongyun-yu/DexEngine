# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constants of the residual env cfg: object friction and mass, the gravity ramp length and the PhysX buffer exponent.

The cfg classes are built by :mod:`dexengine.rl.tasks.cfg.residual_factory`. The constants stay in this file because
the baseline prep scripts read ``DEFAULT_OBJECT_MASS_KG`` from it by name (AST, without importing Isaac Lab).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

_OBJ_FRICTION = 2.0  # object contact friction (static = dynamic); the robot spawn has no material field
# ---------------------------------------------------------------------------
# Object mass policy (single source of truth).
# ALL loaded objects (right `object`, left `object_left`, and any future/new
# object), default to DEFAULT_OBJECT_MASS_KG for
# controlled bimanual training/eval. Do NOT scatter hardcoded masses/densities
# in the per-dataset cfgs — pass nothing (inherit the default) or call
# make_object_mass_props(...) here. The runtime --object_mass / --object_mass_left
# overrides (dexengine/rl/cli.py) still win and are logged at env build.
# A clearly-named legacy option (density / explicit mass) remains for debugging.
# ---------------------------------------------------------------------------
DEFAULT_OBJECT_MASS_KG = 0.2
# ---------------------------------------------------------------------------
# Gravity / friction ramp duration (early training curriculum).
# The EFFECTIVE ramp duration is ALWAYS re-resolved at runtime in
# dexengine/rl/train/runner.py train_cfg_overrides as
#     ramp_steps = int((max_iterations * rollouts) * --gravity_ramp_fraction)   # default 0.10
# with start_fraction=0.0, and pinned to ramp_steps=0 / start_fraction=1.0 at eval.
# So the value baked into the EventTerm builders below is never the effective ramp
# on any registered train/eval path. We still set a concrete, fraction-derived
# skeleton (rather than a bare magic number) so the cfg is self-consistent at the
# canonical default horizon.
# ---------------------------------------------------------------------------
CANONICAL_TRAIN_TOTAL_TIMESTEPS = 2000 * 32  # build-time placeholder horizon (2000 x 32); runtime resolves ramps
GRAVITY_RAMP_FRACTION_DEFAULT = 0.10
RAMP_STEPS_BUILD_DEFAULT = int(CANONICAL_TRAIN_TOTAL_TIMESTEPS * GRAVITY_RAMP_FRACTION_DEFAULT)  # 6400


def physx_shift() -> int:
    """Exponent reduction k of the training-scale PhysX GPU buffers (2**26 -> 2**(26 - k), ...)."""
    return 2


def make_object_mass_props(
    mass_kg: float = DEFAULT_OBJECT_MASS_KG, legacy_density: float | None = None, legacy_mass_kg: float | None = None
):
    """Return the canonical object MassPropertiesCfg (default 0.2 kg).

    legacy_density / legacy_mass_kg are debug-only escape hatches (historical
    values density=1000.0 / mass=0.080); leave BOTH None for the standard 0.2 kg.
    """
    if legacy_density is not None:
        return sim_utils.MassPropertiesCfg(density=legacy_density)
    if legacy_mass_kg is not None:
        return sim_utils.MassPropertiesCfg(mass=legacy_mass_kg)
    return sim_utils.MassPropertiesCfg(mass=mass_kg)
