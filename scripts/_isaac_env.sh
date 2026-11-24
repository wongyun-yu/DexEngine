#!/bin/bash
# The **single owner** of the Isaac Python runtime — IsaacLab location · venv activation · EULA · $ISAAC_PY.
#
# Why a separate file: the training recipe table (`scripts/_rl_recipe.sh`) **requires** the training axes
# `DEXENGINE_DATASET`·`DEXENGINE_EMBODIMENT` (right for training). A tool that needs only the Isaac Python path
# (`scripts/build_reach_map.sh` — a function of arm and hand only) would have to invent an unrelated dataset·emb to
# source that table; an unneeded required input invites fake values, and fake values stay in the records as false
# evidence. The location rule lives **here only**; the training env sources this file.
#
# Use:     source scripts/_isaac_env.sh || exit 2     # → $ISAACLAB_DIR · $ISAAC_PY (export) · venv active
# Marker:  one [ISAAC_ENV] line on stderr (skipped with DEXENGINE_ISAAC_ENV_QUIET=1 — the training recipe writes its
#          own manifest).

# ── IsaacLab location — the rule's owner is scripts/_isaac_py.sh (resolves paths only, no activation). A sibling
# ../IsaacLab does not exist in a worktree (used as is, the launcher dies quietly with rc=127) — that file states the
# candidate order and the marker when nothing is found.
. "$(dirname "${BASH_SOURCE[0]}")/_isaac_py.sh" \
  || { echo "[isaac_env] could not read scripts/_isaac_py.sh" >&2; return 2 2>/dev/null || exit 2; }
ISAACLAB_DIR=$(isaac_lab_dir) || { return 2 2>/dev/null || exit 2; }
export ISAACLAB_DIR
# ── Isaac Python runtime. Activates the venv and answers the EULA prompt that the first Isaac Sim bootstrap asks on
# stdin with a non-interactive value (without it a non-interactive shell dies with "kit bootstrap EOF").
# Launch, render, smoke and pool bakes all use the same env.
source "$ISAACLAB_DIR/env_isaaclab/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
ISAAC_PY="$ISAACLAB_DIR/env_isaaclab/bin/python"; export ISAAC_PY
[ -x "$ISAAC_PY" ] || { echo "[isaac_env] ISAAC_PY is not executable: $ISAAC_PY" >&2; return 2 2>/dev/null || exit 2; }
[ "${DEXENGINE_ISAAC_ENV_QUIET:-0}" = 1 ] || echo "[ISAAC_ENV] ISAACLAB_DIR=$ISAACLAB_DIR ISAAC_PY=$ISAAC_PY" >&2
