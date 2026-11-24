#!/bin/bash
# The single owner of the Isaac Python **location** — resolves only the IsaacLab folder and its venv python path (no
# activation · EULA · export).
#
# Why a separate file: scripts/_isaac_env.sh **activates** the venv (PATH·VIRTUAL_ENV change). Shells that need only
# an interpreter path as a default (the bake/judge drivers' $PYDP·$PYI · $RETARGET_PY of a remote launch) must read the
# same rule without that side effect; per-shell copies of `${ISAACLAB_DIR:-$WT/../IsaacLab}/env_isaaclab/bin/python`
# point at a place that does not exist in a worktree. The rule lives here only; _isaac_env.sh sources this file too,
# resolves the location with the same functions, then activates.
#
# Use:   . scripts/_isaac_py.sh
#        d=$(isaac_lab_dir) || exit 2      # ISAACLAB_DIR as is when set · otherwise searched among the candidates
#        py=$(isaac_py) || exit 2          # "$d/env_isaaclab/bin/python"
#        d=$(isaac_lab_of_tree <tree>)     # the IsaacLab that tree finds without env (sibling — rule ①) · rc 2 if none
#        isaac_lab_of_tree --where <tree>  # that place (present or not — where test fixtures build a stand-in)
# Candidates (only when ISAACLAB_DIR is empty — the first whose env_isaaclab/bin/python is executable, normalised with
# cd·pwd):
#   ① sibling ../IsaacLab of the repo holding this file   ② ../IsaacLab from the current folder
#   ③ ${DEXENGINE_MAIN_ROOT}/../IsaacLab   ④ ../../IsaacLab from this repo's git common dir (a worktree's main .git)
# Not found → rc 2 + one stderr line — a missing path is never returned silently.

_ISAAC_PY_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Is one place an IsaacLab (env_isaaclab/bin/python executable)? Yes → print the normalised path, rc 0. The one
# candidate test.
_isaac_lab_at() {
  [ -x "$1/env_isaaclab/bin/python" ] || return 1
  (cd "$1" && pwd)
  return 0
}

# The IsaacLab a tree finds without env = that tree's sibling (candidate ① below). Cluster pods use this place from the
# unpacked tree without ISAACLAB_DIR — the submitter (paths.cluster_tree) asks this function before submitting.
# --where prints the place whether it exists or not (where test fixtures build).
isaac_lab_of_tree() {
  if [ "${1:-}" = --where ]; then printf '%s\n' "$2/../IsaacLab"; return 0; fi
  local w
  w=$(isaac_lab_of_tree --where "$1")
  _isaac_lab_at "$w" && return 0
  echo "[isaac_py] tree $1 has no sibling IsaacLab ($w) — env_isaaclab/bin/python must be executable" >&2
  return 2
}

isaac_lab_dir() {
  if [ -n "${ISAACLAB_DIR:-}" ]; then printf '%s\n' "$ISAACLAB_DIR"; return 0; fi
  local c gc
  gc=$(cd "$_ISAAC_PY_REPO" 2>/dev/null && cd "$(git rev-parse --git-common-dir 2>/dev/null)" 2>/dev/null && pwd)
  for c in "$(isaac_lab_of_tree --where "$_ISAAC_PY_REPO")" "$(isaac_lab_of_tree --where .)" \
           "$(isaac_lab_of_tree --where "${DEXENGINE_MAIN_ROOT:-.}")" ${gc:+"$(isaac_lab_of_tree --where "$gc/..")"}; do
    _isaac_lab_at "$c" && return 0
  done
  echo "[isaac_py] IsaacLab not found — set ISAACLAB_DIR=<Isaac Lab checkout>. Expected layout:" \
       "\$ISAACLAB_DIR/isaaclab.sh and an executable \$ISAACLAB_DIR/env_isaaclab/bin/python (the Isaac Lab venv)." \
       "Unset, only a sibling ../IsaacLab is searched (of the code tree, the cwd, the data root, the main checkout)" \
       "(tried: $_ISAAC_PY_REPO/../IsaacLab · ../IsaacLab · \${DEXENGINE_MAIN_ROOT}/../IsaacLab · git common dir/../IsaacLab)" >&2
  return 2
}

isaac_py() {
  local d
  d=$(isaac_lab_dir) || return 2
  printf '%s\n' "$d/env_isaaclab/bin/python"
}
