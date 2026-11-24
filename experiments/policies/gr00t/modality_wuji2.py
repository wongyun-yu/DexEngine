"""E5 GR00T N1.7 modality — RB-Y1 + two Wuji 2 hands, 56-D (the RB-Y1 + Wuji 2 joint groups, torso left out).

Read by Isaac-GR00T `gr00t/experiment/launch_finetune.py --modality-config-path <this file>` under NEW_EMBODIMENT.
The group names and their column spans are the dataset's meta/modality.json, written by
experiments/policies/data/convert.py from its GROUPS — the two must agree (the converter is the owner of the order;
this file only names the keys). Actions are absolute joint targets (the converter's action = the target of the
next step), horizon 40 frames at 30 Hz (the base's action_horizon).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ACTION_HORIZON = 40
GROUPS = ["left_arm_joints", "right_arm_joints", "head_joints", "hands_joints"]
print(f"[E5_MODALITY] state = {GROUPS}", flush=True)

register_modality_config(
    {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["camera_ego_left", "camera_ego_right"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=GROUPS),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)),
            modality_keys=GROUPS,
            action_configs=[
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
                for _ in GROUPS
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
    },
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
