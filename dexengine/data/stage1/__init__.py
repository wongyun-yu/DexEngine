"""Stage 1 after retargeting: the ordered base-layer steps of one hand on one dataset.

``table`` holds the step table (``STAGES``: what each step runs and how its progress is counted), ``plan`` turns a
step into concrete argv lists (``emit_run``) and reads the rig and prior identities, ``status`` counts every step's
progress per dataset (the "first empty step" table), ``register`` is the chain that adds a new hand, ``clip_scope``
narrows a run to a clip list and ``split_apply`` applies the long-clip split. The CLI is ``dexengine/data/stages.py``.

Every step calls an existing pipeline module; this package adds no rule of its own. It reads the active hand and arm
from the rule registry (``DEXENGINE_HAND`` / ``DEXENGINE_ARM``, the process-boundary identity inputs).
"""
