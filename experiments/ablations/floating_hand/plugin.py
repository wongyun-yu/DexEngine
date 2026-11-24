"""Floating-hand plugin: registers the floating-hand body and its inner action with the RL task hooks.

Named by the launcher in ``DEXENGINE_PLUGINS=experiments.ablations.floating_hand.plugin`` (dexengine.rl.tasks.hooks
imports it when the task registry loads). Imports Isaac Lab through the action module, so it loads only inside the
Isaac process.
"""

from dexengine.rl.tasks import hooks
from experiments.ablations.floating_hand import action, body

hooks.register(hooks.FLOATING_HAND, body)
hooks.register(hooks.FLOATING_HAND_ACTION, action.FloatingWristFingersAction)
