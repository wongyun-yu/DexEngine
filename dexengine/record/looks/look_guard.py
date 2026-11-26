"""Frame guard: no rendered frame or image leaves a process unless the canonical look was applied to its stage.

The one look function (looks.look_d.apply) calls mark(stage) as its last step; nothing else sets the token (a repository
test checks the call sites). Every frame producer calls require(where) where Kit hands it a frame or where it opens a
file for frames: the record rig's camera read (render.rig.Rig.rgb), the episode writer (render.writer), the random
looks' reference stills (looks.augvis.render) and the pose still (rl.render.pose_check). A violation dies with MARK.

The token belongs to one USD stage of this process: clear() drops it when a Kit-reuse worker opens the next clip's
stage (session.persist_kit), and require compares the token's stage with the current one whenever omni.usd is loaded.
Standard library only (the record writer imports it before Kit starts).
"""

from __future__ import annotations

import os
import sys

MARK = "[LOOK_GUARD]"
#: who to blame in the error: the one place robot videos come from
OWNER = "scripts/stage4_record.py (dexengine.record.*)"
_TOKEN: dict = {}


def stage_key(stage) -> str:
    """Identity of a USD stage within the process (its root layer)."""
    return str(stage.GetRootLayer().identifier)


def mark(stage, look: str) -> str:
    """Set the token for `stage` (called only by looks.look_d.apply). -> marker line."""
    _TOKEN.clear()
    _TOKEN.update(pid=os.getpid(), stage=stage_key(stage), look=look)
    return f"{MARK} canonical look {look} applied · stage {_TOKEN['stage']}"


def clear(why: str) -> str:
    """Drop the token (the stage it was set for is gone). -> marker line."""
    had = bool(_TOKEN)
    _TOKEN.clear()
    return f"{MARK} token cleared ({why}){'' if had else ' · none was set'}"


def _current_stage() -> str | None:
    """Key of the open USD stage when omni.usd is loaded (never imports it)."""
    usd = sys.modules.get("omni.usd")
    if usd is None:
        return None
    stage = usd.get_context().get_stage()
    return stage_key(stage) if stage is not None else None


def require(where: str) -> None:
    """Die unless the canonical look was applied to the current stage in this process."""
    if not _TOKEN:
        raise SystemExit(
            f"{MARK} {where}: frame written without the canonical look — the look is applied only by "
            f"dexengine.record.looks.look_d.apply; robot videos come only from {OWNER}"
        )
    if _TOKEN["pid"] != os.getpid():
        raise SystemExit(f"{MARK} {where}: look token of process {_TOKEN['pid']} read in {os.getpid()}")
    cur = _current_stage()
    if cur is not None and cur != _TOKEN["stage"]:
        raise SystemExit(
            f"{MARK} {where}: the look was applied to stage {_TOKEN['stage']}, the frame comes from stage {cur}"
        )


def applied() -> dict:
    """Copy of the token ({} = not applied)."""
    return dict(_TOKEN)
