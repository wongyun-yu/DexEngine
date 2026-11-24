# Clip-name normalisation for shell joins — `canon <dataset>` (source only).
#
# Pass both sides of a name join (comm / join / set comparison) through this function. The rule lives in
# dexengine.core.paths.canonical_clip; this is its shell entry. The dataset argument is required (no default) —
# `paths.py --what canon` refuses to run without `--dataset`.
#
# usage:  . scripts/_clip_canon.sh
#         canon "$DATASET" < names.txt        # one name per line → one joinable clip_id per line
_CLIP_CANON_PATHS_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dexengine/core/paths.py"
canon() {
  python3 "$_CLIP_CANON_PATHS_PY" --what canon \
    --dataset "${1:?canon <dataset> — the dataset argument is required}"
}
