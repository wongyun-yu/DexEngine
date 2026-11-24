"""Owning modules of the path layer (layout, qpos layout, registries) and the hand rig folders."""

from __future__ import annotations

from pathlib import Path

from dexengine.core import layout as data_layout  # the canonical layout tables; this module is their window
from dexengine.core import qpos_layout  # noqa: F401 — owner of the qpos block addresses (it reads the tree's task_info)
from dexengine.core import repo as _repo

PV_SFX_PAT = data_layout.PV_SFX_PAT  # placement-variant suffix — owner data_layout (see variant_name)


# The joint-trajectory keys inside an IK npz — so consumers do not memorise them each time.
IK_JOINT_KEY = "arm_joint_positions_all"
IK_ERR_KEY = "wrist_pos_error_all"
TREE_QPOS_KEY = "qpos"
CONTACT_KEY = "contact"


def hand_rig_dir(spec_dir: str) -> Path:
    """The retargeting rig folder of one hand spec directory (see `dexengine.core.repo.hand_rig_dir`)."""
    return _repo.hand_rig_dir(spec_dir)


# ── Robot namespace ──────────────────────────────────────────────────────────
# Why here: a second copy of the hand → spec-directory table (in another stage's script) silently keeps the
# old value when one is changed.
# Robot-side names are under the same discipline as clip names — this file is the sole owner.


#: Old `_shared/<module>.py` registry files, now package modules.
_REGISTRY_MODULES = {
    "absurd_state": "dexengine.rl.tasks.guards",
    "all_obj_collision": "dexengine.rl.tasks.scene.all_obj_collision",
    "arm_specs": "dexengine.core.arms",
    "bimanual_self_collision_filter": "dexengine.rl.tasks.scene.collision_filter",
    "collision_audit": "dexengine.rl.tasks.scene.collision_check",
    "collision_authoring": "dexengine.rl.tasks.scene.collision_authoring",
    "collision_presets": "dexengine.rl.tasks.scene.collision_presets",
    "combo_articulation": "dexengine.rl.tasks.scene.robot",
    "derived_usd": "dexengine.robots.derived_usd",
    "desk_audit": "dexengine.rl.tasks.scene.desk_check",
    "desk_placement": "dexengine.core.desk",
    "extra_collision": "dexengine.rl.tasks.scene.collision_authoring",
    "extra_slots_env": "dexengine.rl.tasks.scene.extra_slots",
    "floating_hand": "dexengine.core.floating",
    "gravity_wrench": "dexengine.rl.tasks.mdp.gravity_wrench",
    "hand_limits": "dexengine.rl.tasks.scene.asset_checks",
    "hand_looks": "dexengine.core.hand_looks",
    "hand_object_contact": "dexengine.rl.tasks.mdp.hand_object_contact",
    "hand_recipe": "dexengine.rl.launch.hand_recipe",
    "hand_recipe_check": "dexengine.rl.launch.hand_recipe",
    "hand_recipe_ledger": "dexengine.rl.launch.run_dir",
    "hand_specs": "dexengine.core.hands",
    "mount_specs": "dexengine.core.mounts",
    "object_pin": "dexengine.rl.tasks.scene.object_pin",
    "pin_support_gap": "dexengine.rl.tasks.scene.object_pin",
    "robot_selection": "dexengine.robots.selection",
    "shared_config_arm": "dexengine.rl.tasks.cfg.constants",
    "spawn_hold": "dexengine.rl.tasks.scene.spawn_hold",
    "split_artic": "dexengine.rl.tasks.scene.split_artic",
    "support_reset": "dexengine.rl.tasks.scene.supports",
    "task_mdp_env_cfg": "dexengine.rl.tasks.cfg.task",
    "train_arm_env": "dexengine.rl.tasks.env",
    "train_arm_env_cfg": "dexengine.rl.tasks.cfg.arm_ik",
    "train_arm_residual_env_cfg": "dexengine.rl.tasks.cfg.residual",
    "wrist_collider": "dexengine.rl.tasks.scene.asset_checks",
}


def load_registry(module: str):
    """Return the robot registry module that used to be read from `_shared/<module>.py` by file."""
    import importlib

    return importlib.import_module(_REGISTRY_MODULES[module])
