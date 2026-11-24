"""Run identity: code tree, commit, branch and dirty flag stamped into output meta and startup markers.
Used by traj_record (trajectory file meta · [TRAJ_REC] on) and the trajectory replay renderer
([REPLAY_RENDER] startup line)."""

from __future__ import annotations

import os
import subprocess


def _git(args: list[str], cwd: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001 — a meta lookup never stops the run; the error type becomes the value
        return f"?({type(e).__name__})"


def tree_identity() -> dict:
    """Tree, commit, branch and dirty flag (tracked files only) of the checkout this package was imported from."""
    here = os.path.dirname(os.path.abspath(__file__))
    return {
        "module": here,
        "tree": _git(["rev-parse", "--show-toplevel"], here),
        "commit": _git(["rev-parse", "HEAD"], here),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], here),
        "dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"], here)),
    }
