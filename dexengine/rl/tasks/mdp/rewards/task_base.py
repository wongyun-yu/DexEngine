"""Base of the object task rewards: object asset and reference lookups per side."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from dexengine.rl.tasks.mdp.naming import resolve_palm_body_id, resolve_tip_body_names

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class _TaskRewardBase(ManagerTermBase):
    """Base class for task rewards with object access."""

    #: The scene object a term reads when its params name none — read by __init__ below and by the D16 idle-side sweep
    #: (env_setup.sides._term_object_binding), so the two cannot disagree.
    OBJECT_KEY_DEFAULT = "object_right"

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._side = cfg.params.get("side", "right")
        self._traj_mgr = env.trajectory_manager
        # Which scene rigid object the object-tracking rewards read. Default
        # OBJECT_KEY_DEFAULT ('object_right'); left
        # terms pass object_key='object_left'. (Reference lookups stay keyed by
        # side via ref_state[f'{side}_obj_*'].)
        self._object_key = cfg.params.get("object_key", self.OBJECT_KEY_DEFAULT)

        hand = env.scene[f"{self._side}_hand"]
        self._palm_body_id = resolve_palm_body_id(hand, self._side)

        # Fingertip body IDs (auto-detects dataset vs robot naming)
        tip_names = resolve_tip_body_names(hand, self._side)
        self._tip_body_ids = []
        for name in tip_names:
            try:
                self._tip_body_ids.append(hand.data.body_names.index(name))
            except ValueError:  # noqa: PERF203 — each tip body is looked up alone; a missing one maps to body 0
                self._tip_body_ids.append(0)
        self._tip_body_ids_tensor = torch.tensor(self._tip_body_ids, device=env.device)
