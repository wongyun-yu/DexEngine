#!/bin/bash
# FlashDexRetarget TACO import + kinematic retargeting to Sharpa Wave for the 64 clips (the authors'
# scripts/retarget/taco.sh, unchanged; a file of the FlashDexRetarget checkout $F, external) on CPU. The Isaac Sim
# object assets (steps usd · sdf) are left to the first GPU job (training).
# Raw = data/fdr/taco_raw64 (the 64 clips' TACO release files · UNITS.tsv) -> data/fdr/taco64.
# usage: retarget.sh <cpu workers>   (each coacd worker takes about 6-9 GB of RAM)
# Log logs/fdr_retarget.log · rc logs/fdr_retarget.rc · the pack's clip count vs 64 is printed at the end.
set -uo pipefail
WK=${1:?cpu workers}
B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"
F=$B/code/flashdexretarget; RAW=$B/data/fdr/taco_raw64; OUT=$B/data/fdr/taco64
L=$B/logs/fdr_retarget.log
{
  echo "[FDR_RT] $(date -Is) code $(git -C "$F" rev-parse --short HEAD) raw $RAW ($(grep -c . "$RAW/UNITS.tsv") units) → $OUT"
  source "$F/scripts/env.sh" || exit 5
  python -c "import chumpy, smplx" || { echo "[FDR_RT] FAIL env lacks chumpy/smplx (setup.sh)"; exit 6; }
  # a failed earlier import leaves human_demo/ with only .hydra/, which the pipeline reads as 'has clips'
  [ -d "$OUT/human_demo" ] && ! ls "$OUT/human_demo" | grep -q '\.npz$' && { echo "[FDR_RT] FAIL $OUT/human_demo has no clip — remove $OUT first"; exit 7; }
  echo "[FDR_RT] workers=$WK"
  cd "$F" && CUDA_VISIBLE_DEVICES= ROBOTS=sharpa timeout 14400 bash scripts/retarget/taco.sh "$RAW" "$OUT" steps.usd=false steps.sdf=false workers="$WK"
  rc=$?
  n=$(ls "$OUT/human_demo" 2>/dev/null | grep -c '\.npz$'); k=$(ls "$OUT/sharpa/retargeted" 2>/dev/null | grep -c '\.npz$')
  echo "[FDR_RT] $(date -Is) rc=$rc · imported $n · retargeted $k · motion.pt $( [ -s "$OUT/sharpa/motion.pt" ] && echo yes || echo no) (of 64)"
  exit $rc
} > "$L" 2>&1
rc=$?; echo "rc=$rc" > "$B/logs/fdr_retarget.rc"; exit $rc
