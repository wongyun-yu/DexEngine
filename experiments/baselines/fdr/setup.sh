#!/bin/bash
# FlashDexRetarget environment: the authors' scripts/setup.sh (FlashDexRetarget checkout, external) unchanged, with the
# conda env as a prefix inside the code dir (<workspace>/code/flashdexretarget/.conda311, so it can be copied to a
# cluster like maniptrans/.conda38). The robot USD conversion (needs Isaac Sim + a GPU) is skipped here (--skip_assets)
# and done by the first GPU job (training).
# Log logs/install_fdr.log · rc logs/install_fdr.rc (one line 'rc=<n>').
set -uo pipefail
B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"
F=$B/code/flashdexretarget; P=$F/.conda311; L=$B/logs/install_fdr.log
source "$HOME/miniconda3/etc/profile.d/conda.sh" || { echo "rc=5" > "$B/logs/install_fdr.rc"; exit 5; }
{
  echo "[FDR_SETUP] $(date -Is) prefix $P · code $(git -C "$F" rev-parse --short HEAD)"
  [ -x "$P/bin/python" ] || conda create -p "$P" python=3.11 -y || exit 6
  conda activate "$P" && pip install --upgrade pip || exit 7
  conda deactivate   # setup.sh finds the env by 'conda env list | awk $1' — an active env's line starts with '*'
  conda env list | awk '{print $1}' | grep -qx "$P" || { echo "[FDR_SETUP] FAIL prefix env not listed"; exit 8; }
  bash "$F/scripts/setup.sh" --env "$P" --skip_assets || exit $?
  # the MANO release pickles (mano_v1_2) hold chumpy arrays: smplx's MANO loader needs chumpy to unpickle them; the same
  # pin as our retarget env, --no-deps as setup.sh asks
  conda activate "$P" && pip install --no-deps --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da" \
    && python -c "import chumpy, smplx; print('[FDR_SETUP] chumpy', chumpy.__version__)" || exit 9
} > "$L" 2>&1
rc=$?; echo "rc=$rc" > "$B/logs/install_fdr.rc"; echo "[FDR_SETUP] $(date -Is) rc=$rc" >> "$L"; exit $rc
