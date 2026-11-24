#!/bin/bash
# Shell view of the canonical RL recipe table (configs/rl/recipe.yaml, owner dexengine.rl.launch.recipe) for shell
# entries that source it (scripts/_lib.sh) and for dexengine.placement.bundle.run_inputs.
#   CTE_PATH=uni|bi source scripts/_rl_recipe.sh || exit 2     (default uni; DEXENGINE_DATASET/EMBODIMENT required)
# Exports the table column (a non-empty value already in the environment wins), DEXENGINE_COMBO_USD_VARIANT,
# ARM_JOINT_PD and DEXENGINE_MOUNT_TOL, then activates Isaac Lab (scripts/_isaac_env.sh: $ISAACLAB_DIR, $ISAAC_PY).
# On failure DEXENGINE_COMBO_USD_VARIANT is poisoned so that a caller that ignores the return code cannot spawn.
_rr_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if ! _rr_env=$(PYTHONPATH="$_rr_root${PYTHONPATH:+:$PYTHONPATH}" python3 -m dexengine.rl.launch.recipe --shell "${CTE_PATH:-uni}"); then
  export DEXENGINE_COMBO_USD_VARIANT=_canon_train_env_broken
  unset _rr_root _rr_env
  return 4 2>/dev/null || exit 4
fi
eval "$_rr_env"
unset _rr_root _rr_env
export PYTHONFAULTHANDLER=1
DEXENGINE_ISAAC_ENV_QUIET=1 source "$(dirname "${BASH_SOURCE[0]}")/_isaac_env.sh" || { return 2 2>/dev/null || exit 2; }
