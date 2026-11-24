"""Gym ids of the RL task (importing this module registers them; it needs only ``gymnasium``).

All three ids share the env class ``dexengine.rl.tasks.env:ArmResidualTrainEnv`` and the skrl PPO config
``configs/rl/skrl_ppo.yaml``; they differ in the env cfg class (``dexengine.rl.tasks.cfg.envs``):

  DexEngine-ArmResidual-RightUni-v0   one policy: right arm + hand, right object
  DexEngine-ArmResidual-LeftUni-v0    one policy: left arm + hand (right arm parked), left object
  DexEngine-ArmResidual-Bimanual-v0   right and left residual action terms trained jointly

The cfg modules are referenced by string, so registration does not import Isaac Lab. Importing this module also
loads the experiment plugins named in ``DEXENGINE_PLUGINS`` (see ``dexengine.rl.tasks.hooks``) before any cfg is
resolved.
"""

from __future__ import annotations

import gymnasium as gym

from dexengine.core.repo import CONFIGS_DIR
from dexengine.rl.tasks import hooks

ENV_ENTRY_POINT = "dexengine.rl.tasks.env:ArmResidualTrainEnv"
AGENT_YAML_PATH = str(CONFIGS_DIR / "rl" / "skrl_ppo.yaml")

RIGHT_UNI = "DexEngine-ArmResidual-RightUni-v0"
LEFT_UNI = "DexEngine-ArmResidual-LeftUni-v0"
BIMANUAL = "DexEngine-ArmResidual-Bimanual-v0"
GYM_IDS = (RIGHT_UNI, LEFT_UNI, BIMANUAL)

_ENV_CFGS = {
    RIGHT_UNI: "dexengine.rl.tasks.cfg.envs:TacoRHArmResidualEnvCfg",
    LEFT_UNI: "dexengine.rl.tasks.cfg.envs:TacoLeftUniArmResidualEnvCfg",
    BIMANUAL: "dexengine.rl.tasks.cfg.envs:TacoBimanualArmResidualEnvCfg",
}

for _gid, _cfg in _ENV_CFGS.items():
    if _gid not in gym.registry:
        gym.register(
            id=_gid,
            entry_point=ENV_ENTRY_POINT,
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": _cfg, "skrl_cfg_entry_point": AGENT_YAML_PATH},
        )

hooks.load_plugins()
