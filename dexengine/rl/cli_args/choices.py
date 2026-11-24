"""Choice lists and custom actions of the trainer argv (leaf module: imported by the flag groups and the facade).

Robot choices are derived from the spec registries (plain dicts in dependency-free modules: no Isaac import).
"""

import argparse

from dexengine.core.arms import ARM_REGISTRY
from dexengine.core.hands import FINGER_PENALTY_SPANS, HAND_REGISTRY

TASK_CHOICES = [
    # The single pipeline: reference-base + residual PPO.
    "train",
    "eval",
]

HAND_CHOICES = sorted(HAND_REGISTRY.keys())
#: Values of --finger_penalty_span (owner: dexengine.core.hands.FINGER_PENALTY_SPANS).
FINGER_PENALTY_SPAN_CHOICES = tuple(FINGER_PENALTY_SPANS)
ARM_CHOICES = sorted(ARM_REGISTRY.keys())


class _ArmGainGiven(argparse.Action):
    """Arm gain flags (--arm_kp · --arm_kd · --arm_kp_per_joint · --arm_kd_per_joint): store the value and append
    the flag to ``arm_gains_given`` — passing the default value still counts as given. A body without arm joints
    (floating hand) refuses given arm gains. The value is parsed by ``type`` as usual."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.arm_gains_given = tuple(getattr(namespace, "arm_gains_given", ())) + (f"--{self.dest}",)
