#!/bin/bash
# Pipeline common code: config loading · path resolution · option parsing (source only)
# Use:  source "$(dirname "$0")/_lib.sh"; pipeline_init "$@"; set -- "${PIPELINE_ARGS[@]}"
#   → sets REPO ARM HAND DATASET EMB COMBO ISAACLAB_DIR RETARGET_PY DEXENGINE_MAIN_ROOT PYI GPU SEQ_DIR.
#   Common options: --arm X --hand X --dataset X --emb X --gpu N --seq-dir D --config FILE --data-root D
#   (--main-root D is the same option: both set DEXENGINE_MAIN_ROOT, the data root — the Python CLIs' --data-root)
#   Remaining arguments come back in PIPELINE_ARGS.

# The **single owner** of the keys a config file (configs/pipeline.env · PIPELINE_CONFIG=<file>) accepts.
# pipeline_init below reads them all — a key not listed here is refused and dies with rc 2 (otherwise a typo key such
# as `DATSET=` or a renamed key is exported silently and a different value wins).
PIPELINE_CONFIG_KEYS=(ARM HAND DATASET EMB ISAACLAB_DIR RETARGET_PY DEXENGINE_MAIN_ROOT GPU SEQ_DIR)

# The single owner of config-file parsing — pipeline_init and the config checker both call it (no second parser).
# Output = one `KEY<TAB>VALUE` per line (comments · trailing spaces removed). An unknown key or a line that is not
# KEY=VALUE writes the reason to stderr, rc 2.
pipeline_config_read() { # <config file>
  local cfg=$1 line k v kk known
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}; line=${line%"${line##*[![:space:]]}"}
    [ -n "$line" ] || continue
    [[ "$line" == *=* ]] || { echo "[pipeline] config file $cfg: '$line' is not KEY=VALUE — refused" >&2; return 2; }
    k=${line%%=*}; v=${line#*=}
    known=0; for kk in "${PIPELINE_CONFIG_KEYS[@]}"; do [ "$kk" = "$k" ] && known=1; done
    [ "$known" = 1 ] || {
      echo "[pipeline] config file $cfg: unknown key '$k' — accepted keys: ${PIPELINE_CONFIG_KEYS[*]}" >&2
      echo "           (owner scripts/_lib.sh PIPELINE_CONFIG_KEYS · a key without a consumer is refused)" >&2
      return 2; }
    printf '%s\t%s\n' "$k" "$v"
  done < "$cfg"
}

pipeline_init() {
  REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  local cfg=${PIPELINE_CONFIG:-$REPO/configs/pipeline.env}
  PIPELINE_ARGS=()
  # 0) Help runs without config or inputs — so that `-h` does not die on the required-input check (3b below) or the
  #    training recipe's required-axis check, it goes straight to the script's own parser (`-h|--help` → print the
  #    header, rc 0) without reading the config. This path runs nothing.
  local a0
  for a0 in "$@"; do case "$a0" in -h|--help) PIPELINE_ARGS=(-h); return 0 ;; esac; done
  # 1) take --config from the command line first (it can change the config file)
  local a=("$@") i=0
  while [ $i -lt ${#a[@]} ]; do
    if [ "${a[$i]}" = "--config" ]; then cfg=${a[$((i+1))]}; fi
    i=$((i+1))
  done
  [ -f "$cfg" ] || { echo "[pipeline] config file missing: $cfg" >&2; return 2; }
  # 2) config file → environment variables (an existing environment variable wins). The whole file is parsed
  #    **first** — an unknown key or broken line dies even when the environment already holds the value (a rotten
  #    file is rotten regardless of the environment).
  local kv k v
  kv=$(pipeline_config_read "$cfg") || return 2
  while IFS=$'\t' read -r k v; do
    [ -n "$k" ] || continue
    [ -n "${!k:-}" ] || export "$k=$v"
  done <<< "$kv"
  # 3) command-line options (highest priority)
  i=0
  while [ $i -lt ${#a[@]} ]; do
    case "${a[$i]}" in
      --arm)       export ARM=${a[$((i+1))]}; i=$((i+2)) ;;
      # `--robot` is a retired name; accepting it silently would keep two names — die so the caller is fixed.
      --robot)     echo "[pipeline] --robot is now --arm. Fix the caller." >&2; return 2 ;;
      --hand)      export HAND=${a[$((i+1))]}; i=$((i+2)) ;;
      --dataset)   export DATASET=${a[$((i+1))]}; i=$((i+2)) ;;
      # emb is a common option too — parsed by a stage **after** `pipeline_init`, 3b below would not see it.
      --emb)       export EMB=${a[$((i+1))]}; i=$((i+2)) ;;
      --gpu)       export GPU=${a[$((i+1))]}; i=$((i+2)) ;;
      --seq-dir)   export SEQ_DIR=${a[$((i+1))]}; i=$((i+2)) ;;
      --data-root|--main-root) export DEXENGINE_MAIN_ROOT=${a[$((i+1))]}; i=$((i+2)) ;;
      --config)    i=$((i+2)) ;;
      *)           PIPELINE_ARGS+=("${a[$i]}"); i=$((i+1)) ;;
    esac
  done
  # 3b) Identity inputs (arm · dataset · emb) are stated by the caller — no default (the same policy as the campaign
  #     layer and the training recipe). Empty → die here with rc 2, **naming the accepted names** — dying later in the
  #     training recipe would name variables the caller never wrote (DEXENGINE_DATASET·DEXENGINE_EMBODIMENT) and
  #     not with rc 2 (a sourced file's `${…:?}`). The hand (HAND) is killed by the registry check.
  local miss=()
  [ -n "${ARM:-}" ] || miss+=("arm — --arm <arm> or ARM=<arm> (registered: dexengine/core/paths.py --what combo-ids)")
  [ -n "${DATASET:-}" ] || miss+=("dataset — --dataset <ds> or DATASET=<ds> (dexengine/core/paths.py --what datasets)")
  # EMB is the only accepted name for emb — the training-layer name (DEXENGINE_EMBODIMENT) is **derived** from EMB
  # below (like DEXENGINE_DATASET).
  [ -n "${EMB:-}" ] || miss+=("embodiment — --emb <emb> or EMB=<emb> (dexengine.core.layout.EMBODIMENTS)${DEXENGINE_EMBODIMENT:+ · DEXENGINE_EMBODIMENT=$DEXENGINE_EMBODIMENT in the environment is a training-layer name and is not read here}")
  if [ ${#miss[@]} -gt 0 ]; then
    echo "[pipeline] required inputs missing (no default — config file $cfg has no value either):" >&2
    printf '  - %s\n' "${miss[@]}" >&2
    return 2
  fi
  # 4) derivation · normalisation
  export COMBO="${ARM}_${HAND}"
  _abs() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$REPO/$1" ;; esac; }
  export DEXENGINE_MAIN_ROOT=$(_abs "${DEXENGINE_MAIN_ROOT:-.}")
  # The sibling-directory contract (../IsaacLab) resolves against the repo; without a sibling (e.g. a worktree) the
  # data checkout's sibling is used.
  _sibling() { # $1=relative default
    if [ -e "$REPO/$1" ]; then printf '%s' "$REPO/$1"
    elif [ -e "$DEXENGINE_MAIN_ROOT/$1" ]; then printf '%s' "$DEXENGINE_MAIN_ROOT/$1"
    else printf '%s' "$REPO/$1"; fi
  }
  # The Isaac location is owned by scripts/_isaac_py.sh and the runtime (venv · EULA · ISAAC_PY) by scripts/_isaac_env.sh
  # (searched repo → main → git common dir; dies when not found). The retargeting python is owned by paths.retarget_py (the main checkout's
  # .venv). Nothing is searched again here — a third searcher fails to find `../IsaacLab` in a worktree and the chain
  # dies.
  case "${ISAACLAB_DIR:-}" in "") ;; ../*) ISAACLAB_DIR=$(_sibling "$ISAACLAB_DIR"); export ISAACLAB_DIR ;; *) ISAACLAB_DIR=$(_abs "$ISAACLAB_DIR"); export ISAACLAB_DIR ;; esac
  # Set the axes **first** — the training recipe requires both (exporting them after the source kills every
  # pipeline entry).
  export DEXENGINE_DATASET=$DATASET
  # 3b above checked emb against the caller's input (--emb · EMB=) — no default. The training-layer name is derived
  # here, once.
  export DEXENGINE_EMBODIMENT=$EMB
  CTE_PATH=${CTE_PATH:-bi} source "$REPO/scripts/_rl_recipe.sh" || return 2
  export PYI="$ISAAC_PY"
  case "${RETARGET_PY:-}" in
    "") RETARGET_PY=$(env -u RETARGET_PY python3 "$REPO/dexengine/core/paths.py" --what retarget-py) || return 2 ;;
    ../*) RETARGET_PY=$(_sibling "$RETARGET_PY") ;;
    *) RETARGET_PY=$(_abs "$RETARGET_PY") ;;
  esac
  export RETARGET_PY
  # The data path (SEQ_DIR) is relative to DEXENGINE_MAIN_ROOT, which holds the data/ tree (in a worktree: the main
  # checkout).
  if [ -n "${SEQ_DIR:-}" ]; then
    case "$SEQ_DIR" in /*) ;; *) SEQ_DIR="$DEXENGINE_MAIN_ROOT/$SEQ_DIR" ;; esac
    export SEQ_DIR
  fi
  export DEXENGINE_ARM=$ARM DEXENGINE_HAND=$HAND
  # The dataset · embodiment axes are exported too (DEXENGINE_DATASET·DEXENGINE_EMBODIMENT above) — otherwise stage 2
  # of a run goes to `--dataset hocap` while stage 3 alone silently diverges to another axis.
  echo "[pipeline] repo=$REPO robot=$ARM hand=$HAND dataset=$DATASET combo=$COMBO gpu=${GPU:-?}"
  echo "[AXIS] DEXENGINE_DATASET=$DEXENGINE_DATASET DEXENGINE_EMBODIMENT=$DEXENGINE_EMBODIMENT"
  echo "[pipeline] isaaclab=$ISAACLAB_DIR retarget_py=$RETARGET_PY main_root=$DEXENGINE_MAIN_ROOT"
}

pipeline_require() { # path existence check: pipeline_require <what> <path>…
  local what=$1; shift
  local p
  for p in "$@"; do
    [ -e "$p" ] || { echo "[pipeline] $what missing: $p" >&2; return 2; }
  done
}

pipeline_registry_check() { # ARM / HAND / their mount registered? (CPU only, no Isaac)
  python3 - "$REPO" "$ARM" "$HAND" <<'PY'
import sys
repo, arm, hand = sys.argv[1:4]
sys.path.insert(0, repo)
from dexengine.core.arms import ARM_REGISTRY as A
from dexengine.core.hands import HAND_REGISTRY as H
from dexengine.core.mounts import MOUNT_REGISTRY as M
bad = [x for x, r in ((arm, A), (hand, H)) if r.get(x) is None]
if bad or M.get((arm, hand)) is None:
    print(f"[pipeline] not registered: {bad or (arm, hand)} — a new arm or hand goes into dexengine.core.arms / "
          "dexengine.core.hands / dexengine.core.mounts", file=sys.stderr)
    sys.exit(2)
m = M[(arm, hand)]
# A provisional mount is marked in the run log: every output made with it must say so.
print(f"[pipeline] registry OK: arm={arm} hand={hand} mount={m.ee_to_wrist_offset}"
      + (" ★ MOUNT_PROVISIONAL — this run's outputs use a provisional mount" if getattr(m, "provisional", False) else ""))
PY
}
