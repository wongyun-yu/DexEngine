#!/bin/bash
# ManipTrans command lines (the one owner of these lines) -> <workspace>/run/maniptrans_stage1_imitator_cmds.txt and
# <workspace>/run/maniptrans_cmds.txt. Stages: (1) imitator pre-training, once per hand, pool = pretrain_pool.txt
# (1,084 CHORD TACO clips, the 64 eval clips excluded), cap 36 h; (2) mano2dexhand retarget of each eval clip, both hands;
# (3) per-clip residual (BiH, 4096 envs) under a 2 h training budget; (4) the evaluation recorder (64 worlds, failure
# termination off) -> experiments/scoring/score.py motion (time scale 1.0: the 60 Hz rollout is real time).
# Residual budget = the authors' README values: num_envs=4096 and early_stop_epochs=100 (a minimum epoch count, then the
# code's reward-slope stop) under the 2 h budget (common/capbin/budget: a budget hit is not TRIM, the best checkpoint is
# evaluated). Imitator = the author scale num_envs=<imitator envs>, converge or the 36 h cap (a cap hit is TRIM).
# The env is exported for the whole line: as a `VAR=… cmd` prefix it would reach only the first command, and the later
# ones need the conda PATH (ninja: torch loads gymtorch through it) and LD_LIBRARY_PATH (gym_38.so needs libpython3.8.so).
# LD_LIBRARY_PATH: conda lib first, the platform's path kept after it. The scorer (another interpreter) runs with it cleared.
# usage: gen_cmds.sh <imitator envs: 8192 = author scale | 4096 = when 8192 does not fit the card> [note for the header]
set -euo pipefail
IMIT_ENVS=${1:?imitator envs 8192|4096}; IMIT_NOTE=${2:-}
case $IMIT_ENVS in 8192|4096) ;; *) echo "[GEN_MT] FAIL imitator envs $IMIT_ENVS"; exit 2;; esac
B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$HERE/../../.." && pwd)
M=$B/code/maniptrans; E=$M/.conda38; OUT=$B/run
ENV="cd $M && export PATH=$E/bin:\$PATH LD_LIBRARY_PATH=$E/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} CUDA_VISIBLE_DEVICES=\${GPU:-0} &&"
POOL=$(paste -sd, "$HERE/pretrain_pool.txt")
CLIPS=$R/experiments/scoring/clips/cand_B0r_random_chordfull.tsv
REC="env PYTHONPATH=$R\${PYTHONPATH:+:\$PYTHONPATH} $E/bin/python $R/experiments/baselines/maniptrans/eval_record.py"
SCORE_PY=$B/code/chord_v2d/robotic_grounding/flash_chord/.venv/bin/python
mkdir -p "$OUT" $B/logs/maniptrans $B/results/maniptrans
{
echo "# stage1_imitator (one per hand, the two hands can run on two GPUs) · num_envs=$IMIT_ENVS${IMIT_NOTE:+ · $IMIT_NOTE}"
for S in RH LH; do s=$(echo $S | tr A-Z a-z)
echo "$ENV timeout 129600 $E/bin/python main/rl/train.py task=DexHandImitator dexhand=sharpa_wave side=$S headless=true num_envs=$IMIT_ENVS test=false randomStateInit=true seed=42 dataIndices=[$POOL] experiment=e123_imitator_sharpa_$s > $B/logs/maniptrans/imitator_$s.log 2>&1"
done
} > $OUT/maniptrans_stage1_imitator_cmds.txt
: > $OUT/maniptrans_cmds.txt
tail -n +2 "$CLIPS" | cut -f1,2 | while IFS=$'\t' read -r c d; do
  IMR="\$(ls -t $M/runs/e123_imitator_sharpa_rh__*/nn/e123_imitator_sharpa_rh.pth | head -1)"; IML="\$(ls -t $M/runs/e123_imitator_sharpa_lh__*/nn/e123_imitator_sharpa_lh.pth | head -1)"
  CK="\$(ls -t $M/runs/e123_${c}__*/nn/e123_${c}.pth | head -1)"
  echo "$ENV timeout 1800 $E/bin/python main/dataset/mano2dexhand.py --data_idx $d --dexhand sharpa_wave --side right --headless --iter 4000 > $B/logs/maniptrans/$c.retarget_r.log 2>&1 && timeout 1800 $E/bin/python main/dataset/mano2dexhand.py --data_idx $d --dexhand sharpa_wave --side left --headless --iter 4000 > $B/logs/maniptrans/$c.retarget_l.log 2>&1 && budget 7200 $E/bin/python main/rl/train.py task=ResDexHand dexhand=sharpa_wave side=BiH headless=true num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true seed=42 dataIndices=[$d] rh_base_model_checkpoint=$IMR lh_base_model_checkpoint=$IML early_stop_epochs=100 actionsMovingAverage=0.4 experiment=e123_$c > $B/logs/maniptrans/$c.train.log 2>&1; timeout 1800 $REC $B/results/maniptrans/$c.npz -- task=ResDexHand dexhand=sharpa_wave side=BiH headless=true num_envs=64 test=true randomStateInit=false seed=42 dataIndices=[$d] rh_base_model_checkpoint=$IMR lh_base_model_checkpoint=$IML actionsMovingAverage=0.4 checkpoint=$CK > $B/logs/maniptrans/$c.eval.log 2>&1&& env -u LD_LIBRARY_PATH PYTHONPATH=$R DEXENGINE_E123_ROOT=$B $SCORE_PY -s -P -m experiments.scoring.score motion --rollout $B/results/maniptrans/$c.npz --motion-parquet $B/data/chord_hf_all_taco/$d.parquet --time-scale 1.0 --out $B/results/maniptrans/$c.json" >> $OUT/maniptrans_cmds.txt
done
n=$(wc -l < $OUT/maniptrans_cmds.txt); [ "$n" = 64 ] || { echo "[GEN_MT] FAIL $n"; exit 1; }
echo "[GEN_MT] OK stage1=2 per_clip=$n pool=$(wc -l < "$HERE/pretrain_pool.txt")"
