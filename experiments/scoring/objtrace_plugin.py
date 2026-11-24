"""Object-pose trace plugin: registers the E1–E3 object-pose recorder (``objtrace_recorder``) with the RL task hooks.

A judge launch that sets ``DEXENGINE_E123_OBJTRACE`` also names this module in ``DEXENGINE_PLUGINS`` (comma-separated
with any other plugin); ``dexengine.rl.train.startup`` loads the plugins and takes the recorder from the hook
``objtrace``. Isaac-free at import.
"""

from dexengine.rl.tasks import hooks
from experiments.scoring import objtrace_recorder

hooks.register(hooks.OBJTRACE, objtrace_recorder)
