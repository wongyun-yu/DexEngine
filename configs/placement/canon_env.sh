#!/bin/bash
# Shell run environment of the placement drivers (judge, bake, gate). Usage: source configs/placement/canon_env.sh
# Every canonical value (judge search conditions = configs/placement/judge.yaml, robot generation, dz layers,
# penetration limits) lives in code; this file holds only what must be set in the shell.

# Drop asset/data generation overrides an earlier experiment may have left in the shell (a leftover combo variant
# would send judging and baking to a variant world). The code defaults are canonical, so unsetting selects them.
unset DEXENGINE_URDF_OVERRIDE
# Every arm's base override (dexengine.core.embodiment.base_pos_env_name: DEXENGINE_<ARM>_BASE_POS), not one arm's.
for _v in "${!DEXENGINE_@}"; do [[ $_v == *_BASE_POS ]] && unset "$_v"; done
unset _v
unset DEXENGINE_COMBO_USD_VARIANT DEXENGINE_COMBO_VARIANT DEXENGINE_POOL_ROOT

# Cross-repair step of the bake: canonical ON (stage2_ground.sh default; --cross-repair 0 turns it off).
export CANON_CROSS_REPAIR=1

# BLAS thread count must be fixed before numpy is imported, so it cannot move into Python.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
