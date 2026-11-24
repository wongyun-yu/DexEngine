"""Locations the paper experiments read and write outside the repository.

The comparison with prior pipelines (Tab. 2), the RL ablations (Tab. 3, Tab. S8) and the baseline runs share one
external workspace folder. It holds what does not belong in git:

    code/<method>/     baseline checkouts at the commits in ``experiments/baselines/TREES.txt``, with their venvs
    data/<method>/     baseline inputs (converted clips, assets, checkpoints) and the staged judge inputs
    logs/              run logs
    results/<row>/     per-row records and score files; the table builders read these
    run/               generated per-clip command files

Its location is one input, the environment variable ``DEXENGINE_E123_ROOT`` (an environment variable because the same
scripts run inside cluster jobs, where the workspace is mounted at another path). There is no default: a script that
needs the workspace and finds the variable unset stops with a message. Shell scripts use
``B="${DEXENGINE_E123_ROOT:?…}"``.

``REPO`` is this repository's root (owner: ``dexengine.core.repo``); experiment code reaches its own files through it
(or through package imports), never through the workspace.
"""

from __future__ import annotations

import os
from pathlib import Path

from dexengine.core.repo import REPO_ROOT as REPO

__all__ = [
    "ENV",
    "REPO",
    "root",
    "code",
    "data",
    "results",
    "logs",
    "LEGACY_RUN_ROOT",
    "LEGACY_ENV_PREFIX",
]

ENV = "DEXENGINE_E123_ROOT"

# ── Names of the layout before the package rename, as they appear in EXISTING runs and reference trees
# ─────────────
#: Trainer checkpoint area of the runs trained before the rename (the V1 runs, the Tab. 2 / Tab. 3 runs,
#: every run of a pinned commit). New runs live under dexengine.core.layout.runs.RUN_CKPT_ROOT.
LEGACY_RUN_ROOT = "logs/skrl/openarm_train_stage2"
#: Prefix of the environment flags at the pinned commits (DEXENGINE_<NAME> there is <this prefix><NAME>).
LEGACY_ENV_PREFIX = "OPENARM_"


def root() -> Path:
    """The experiment workspace folder (``$DEXENGINE_E123_ROOT``); exits with a message when unset."""
    raw = os.environ.get(ENV, "")
    if not raw:
        raise SystemExit(f"{ENV} is not set: point it at the experiment workspace (code/ data/ logs/ results/)")
    return Path(raw).expanduser()


def code(*parts: str) -> Path:
    """A path under the workspace's baseline checkouts (``code/``)."""
    return root().joinpath("code", *parts)


def data(*parts: str) -> Path:
    """A path under the workspace's input data (``data/``)."""
    return root().joinpath("data", *parts)


def results(*parts: str) -> Path:
    """A path under the workspace's results (``results/``)."""
    return root().joinpath("results", *parts)


def logs(*parts: str) -> Path:
    """A path under the workspace's run logs (``logs/``)."""
    return root().joinpath("logs", *parts)
