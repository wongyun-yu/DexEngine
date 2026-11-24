"""The hand registry: hand id -> HandSpec, read from the hand spec files (configs/robots/hands/<hand>.yaml)."""

from __future__ import annotations

from dexengine.core.embodiment import default_catalog
from dexengine.core.hands.spec import HandSpec

#: Registered hands — one spec file each. A new hand is a hand file (every HandSpec field and its rl_recipe), a mount
#: file per arm it is bolted to (configs/robots/mounts/<arm>+<hand>.yaml) and, for retargeting, a rig asset in
#: dexengine.core.layout.RETARGET_HAND_ASSETS; no code changes (docs/hands.md).
HAND_REGISTRY: dict[str, HandSpec] = dict(default_catalog().hands)
