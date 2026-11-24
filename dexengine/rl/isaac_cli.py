"""The one way to add AppLauncher arguments to an Isaac CLI parser, the env sim device those arguments select, plus
the exit-code hooks of a Kit process. Standard library only (called before Kit starts).

Why. ``AppLauncher.add_app_launcher_args`` first runs ``parse_known_args()`` with -h/--help removed to detect name
clashes, so a parser with required arguments (``required=True`` or a required mutually exclusive group) answers
``--help`` with "the following arguments are required" and rc 2. Here the required marks are lowered only during
that pre-parse and restored afterwards: the caller's ``parse_args()`` still enforces them (rc 2 + usage when
missing), and ``--help`` prints the help including the AppLauncher arguments, rc 0.

    from dexengine.rl.isaac_cli import add_app_launcher_args
    parser.add_argument("--out", required=True)
    add_app_launcher_args(parser)          # instead of AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from dexengine.core.repo import REPO_ROOT


def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Same as ``AppLauncher.add_app_launcher_args(parser)``, except that required arguments do not break its
    pre-parse."""
    from isaaclab.app import AppLauncher  # already imported by the caller (Kit not started yet)

    held = [a for a in parser._actions if a.required] + [g for g in parser._mutually_exclusive_groups if g.required]
    for x in held:
        x.required = False
    try:
        AppLauncher.add_app_launcher_args(parser)
    finally:
        for x in held:
            x.required = True


#: Marker of the env cfg's sim device (`sim_device`; printed by rl/cli.py where the env cfg is built).
SIM_DEVICE_MARK = "[SIM_DEVICE]"


def sim_device(device, launcher_default, cfg_device: str, distributed: bool, local_rank: int) -> tuple[str, str]:
    """The env cfg's sim device and where it came from. AppLauncher's `--device` (`device`; `launcher_default` = that
    flag's parser default) given → that value, as Isaac Lab's train scripts apply it; distributed → this rank's GPU
    (`cuda:<local_rank>`); neither → `cfg_device` unchanged (the cfg's own default). `--device` together with
    distributed raises: each rank owns its GPU, so the flag would be silently overridden."""
    given = device is not None and device != launcher_default
    if distributed:
        if given:
            raise ValueError(
                f"{SIM_DEVICE_MARK} --device {device} with --distributed: each rank runs on cuda:<local_rank>"
            )
        return f"cuda:{local_rank}", f"distributed rank {local_rank}"
    if given:
        return str(device), "--device"
    return cfg_device, "default"


#: The "closing" mark (mark_closing): off by default (the hook behaves as usual); a finished run turns it on right
#: before closing the app.
_CLOSING = {"on": False, "why": ""}
#: Head of that mark's log line — the session log reader (record.augment.run) looks for this text.
CLOSING_MARK = "[KIT_RC] closing-start mark on"
_TREE = REPO_ROOT


def mark_closing(why: str) -> None:
    """Called by a finished run (DONE marker + output verified) right before ``simulation_app.close()``. After it,
    an uncaught exception whose innermost frame is outside our tree (another extension's callback during Kit
    shutdown, e.g. a menu callback of an extension already unloaded) does not change the rc (the hook of
    exit_nonzero_on_uncaught). An exception in our frames, or before the mark, still gives rc 1."""
    _CLOSING.update(on=True, why=why)
    print(f"{CLOSING_MARK} — {why} · after this, exceptions of other extensions keep the rc", flush=True)


def _ours(path: str) -> bool:
    """Is this a frame of our code — inside this tree? A path that does not resolve to a real file (``<string>``,
    ``<frozen …>``) counts as ours (unknown -> rc 1)."""
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return True
    except (OSError, ValueError):
        return True
    return p.is_relative_to(_TREE)


def foreign_at_close(tb) -> bool:
    """True when the closing mark is on and the exception's innermost frame is outside our tree (the only case in
    which the hook keeps the rc)."""
    if not _CLOSING["on"] or tb is None:
        return False
    while tb.tb_next is not None:
        tb = tb.tb_next
    return not _ours(tb.tb_frame.f_code.co_filename)


def exit_nonzero_on_uncaught() -> None:
    """Make an uncaught exception end the process with **rc 1** once Kit is up (call it on the line right after
    ``AppLauncher(...)``).

    Why. The Kit of IsaacLab 0.53.1 prints a module-level exception's traceback and then exits with rc 0 (measured:
    ``raise`` after ``AppLauncher`` -> rc 0; with this hook -> rc 1), so a caller judging by rc would read a broken
    run as a pass. The hook prints the traceback and a ``[KIT_RC]`` line and hard-exits; it does not call
    ``simulation_app.close()``, which would also end the process with rc 0. One exception: after the closing mark
    (mark_closing), an exception whose innermost frame is outside our tree only prints the mark line and returns,
    so a finished run's rc is not overwritten by the shutdown order of other extensions."""

    def _hook(etype, value, tb):
        traceback.print_exception(etype, value, tb)
        if foreign_at_close(tb):
            print(
                f"[KIT_RC] exception of another extension during shutdown {etype.__name__} — after DONE · rc kept "
                f"({_CLOSING['why']})",
                file=sys.stderr,
                flush=True,
            )
            return
        print(
            f"[KIT_RC] uncaught exception {etype.__name__} — hard exit with rc 1 (so the Kit shutdown cannot "
            f"overwrite the rc with 0)",
            file=sys.stderr,
            flush=True,
        )
        sys.stdout.flush()
        os._exit(1)

    sys.excepthook = _hook
