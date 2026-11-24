"""The identity of the running process: active arm and hand (process-boundary env DEXENGINE_ARM / DEXENGINE_HAND),
the axis hand of shared outputs, the hand spec directory and the combo name."""

from __future__ import annotations

import os
import sys

from dexengine.core import paths as _paths_mod
from dexengine.core.errors import InputMissingError
from dexengine.core.rules.source import _paths

#: **The single window for the hand axis** — so that "reprocess with xhand under these
#: same rules" is one line.
#: The values are not written out again here: the owner of the spec directory·combo name is
#: `dexengine.core.hands.HAND_REGISTRY` → `dexengine.core.paths.HAND_SPEC_DIR`/`combo_id`.
#: Here we decide only **which hand is running right now**.
#: **`DEXENGINE_HAND` has no default** (the active hand is given explicitly at run time).
#: A silent default hand lets different stages of one pipeline run with different hands — unset or empty is a
#: **SystemExit**.
#:
#: Why a **function** and not a module-level constant: evaluating `ACTIVE_HAND` at import time would kill
#: entry points that have nothing to do with the hand (`--campaign-dir` etc.). The name `ACTIVE_HAND` is kept via a
#: PEP 562 module `__getattr__` of the dexengine.core.datasets facade, so consumers reading `RR.ACTIVE_HAND` keep
#: working but die **at the moment of reading**. Inside the module a global lookup does not call `__getattr__`, so use
#: `active_hand()`.
HAND_ENV = "DEXENGINE_HAND"
_HAND_SAID = False


def active_hand() -> str:
    """Id of the hand running now. Unset or empty `DEXENGINE_HAND` **raises loudly** (no default)."""
    global _HAND_SAID
    h = os.environ.get(HAND_ENV, "").strip()
    if not h:
        raise InputMissingError(
            f"[HAND] {HAND_ENV} is not set — give the active hand **explicitly at run time** (there is no default).\n"
            f"       e.g. {HAND_ENV}=wuji2 python3 {os.path.basename(__file__)} --verify\n"
            f"       registered hands: dexengine/core/paths.py --what hand-spec-dirs"
        )
    if not _HAND_SAID:
        print(f"[HAND] {HAND_ENV}={h}", file=sys.stderr)
        _HAND_SAID = True
    return h


#: ── Hand of the shared dashboard axis = the "canonical hand" ──────────────────
#: `active_hand()` is **the hand running now**; this is **whose numbers the shared outputs hold** — another question.
#: A shared output can hold the numbers of one hand only. One place in code must own that hand's name, so that
#: a run with a non-canonical hand does not overwrite the canonical numbers with its own.
#: This is **not** a silent default hand — it is **the owner name of the one set of shared files**.
#: The running hand is still an explicit input and is never derived from this.
#: `dexengine.core.layout.RETARGET_HAND_ASSETS[…].status == "canon"` does **not** answer this question —
#: several hands are canon. That means "the asset is officially registered", not "the axis hand".
#: One value, one name. Shells read it with the rules CLI `--axis-hand` — keep no copy.
#: The spec directory name is not written here (hand id only) — `dexengine.core.paths.hand_spec_dir()` resolves it.
#: The value **lives in `dexengine.core.layout.AXIS_HAND`**. `datasets.py` (the package) must read the same value
#: but cannot import this file — so it lives in a dependency-free module instead of a copy.
#: Here it is **only read**. If it cannot be read, this dies instead of using a silent default.
def _axis_hand_from_layout() -> str:
    return _paths_mod.data_layout.AXIS_HAND


AXIS_HAND = _axis_hand_from_layout()


def axis_hand() -> str:
    """Hand id of the shared dashboard axis. An unregistered name **raises loudly** (no silent pass)."""
    try:
        _paths().hand_spec_dir(AXIS_HAND)
    except Exception as e:  # noqa: BLE001 — no failure passes
        raise InputMissingError(
            f"[AXIS_HAND] {AXIS_HAND!r} is not a registered hand — cannot tell who owns the shared audit outputs.\n"
            f"            registered: dexengine/core/paths.py --what hand-spec-dirs · cause: {e}"
        ) from e
    return AXIS_HAND


ARM_ENV = "DEXENGINE_ARM"
_ARM_SAID = False


def active_arm() -> str:
    """Active arm — given **explicitly at run time** (no default). Same rule as the hand.

    The campaign entry scripts (campaign/*.sh) export DEXENGINE_ARM explicitly.
    """
    global _ARM_SAID
    a = os.environ.get(ARM_ENV, "").strip()
    if not a:
        raise InputMissingError(
            f"[ARM] {ARM_ENV} is not set — give the active arm **explicitly at run time** (there is no default).\n"
            f"       e.g. {ARM_ENV}=rby1 DEXENGINE_HAND=wuji2 python3 {os.path.basename(__file__)} --combo\n"
            f"       registered combos: dexengine/core/paths.py --what combo-ids"
        )
    if not _ARM_SAID:
        print(f"[ARM] {ARM_ENV}={a}", file=sys.stderr)
        _ARM_SAID = True
    return a


def hand_dir(hand: str = "") -> str:
    """Retargeting hand **spec directory** (hand axis of base/tree paths). Unknown hand raises (no silent fallback)."""
    return _paths().hand_spec_dir(hand or active_hand())


def combo(hand: str = "", variant: str = "") -> str:
    """Combo name `<arm>_<hand><variant>`. Strings are never joined by hand."""
    return _paths().combo_id(active_arm(), hand or active_hand(), variant)
