#!/bin/bash
# DexMachina (extended to N objects) · one Tab. 2 TACO clip · author recipe = dexmachina/examples/train_rl.sh flags
# (only --clip / --hand / -exp differ; -B 4096 as in the released script — the paper's 12,000 envs need > 32 GB).
# usage: train_clip.sh <clip key e.g. e123c000> <T60 frames> <num_envs> <gpu> [max_epochs]
#   env DM_TRAIN_TIMEOUT = training wall limit in s (default 345600 = 96 h, a hang guard: the full recipe needs ~117 h on a
#   5090) · DM_TRAIN_LIMIT = timeout (a cap: TRIM if hit) | budget (the row's declared budget, common/capbin/budget:
#   judged by what the run left) — the row runner passes the row budget minus its margin with DM_TRAIN_LIMIT=budget
# writes: logs/dexmachina/train/<key>_B<n>_E<epochs>/{train.log,rc,vram.tsv,epochs.tsv,start,end,commit,pid,limit}
#   (pid = this script) · epochs.tsv = "<unix s> <epoch>" each time rl-games prints an epoch line (2 s poll)
#   rc = DONE rc=0 · BUDGET rc=124 (stopped at its declared budget) · FAIL rc=N (any other end, a timeout cap included)
set -u
KEY=${1:?key}; T=${2:?frames}; NB=${3:?num_envs}; GPU=${4:?gpu}; ME=${5:-5000}
TT=${DM_TRAIN_TIMEOUT:-345600}; LIM=${DM_TRAIN_LIMIT:-timeout}
case $LIM in timeout|budget) ;; *) echo "[TRAIN] FAIL DM_TRAIN_LIMIT=$LIM (timeout|budget)"; exit 9;; esac
B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"; PY=$B/code/dexmachina/env/bin/python
O=$B/logs/dexmachina/train/${KEY}_B${NB}_E${ME}; mkdir -p $O; echo $$ > $O/pid
cd $B/code/dexmachina/dexmachina || { echo "FAIL cd" > $O/rc; exit 9; }
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline WANDB_SILENT=true TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
date +%s > $O/start; git -C .. rev-parse HEAD > $O/commit 2>/dev/null || cat ../E123_COMMIT > $O/commit; echo "$TT" > $O/timeout_s; echo "$LIM" > $O/limit; echo "[TRAIN] limit=$LIM ${TT}s"
( while sleep 5; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i $GPU | sed "s/^/$(date +%s), /"; done ) > $O/vram.tsv 2>/dev/null &
VP=$!
( last=""; while sleep 2; do e=$(grep -a "fps total" $O/train.log 2>/dev/null | tail -1 | sed -n 's/.*epoch: \([0-9]*\)\/.*/\1/p')
  [ -n "$e" ] && [ "$e" != "$last" ] && { echo "$(date +%s) $e" >> $O/epochs.tsv; last=$e; }; done ) &
EP=$!
"$LIM" "$TT" $PY rl/train_rl_games.py -B $NB -obf -obt --max_epochs $ME \
    --actuate_object --retarget_name para --horizon 16  -imw 0.5 --gain_mode all --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01  -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
    --clip ${KEY}-0-${T}-s01-u01 -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp e123 --hand sharpa_wave_hand > $O/train.log 2>&1
rc=$?
kill $VP $EP 2>/dev/null; date +%s > $O/end
if [ $rc -eq 0 ]; then echo "DONE rc=0" > $O/rc
elif [ "$LIM" = budget ] && [ $rc -eq 124 ]; then echo "BUDGET rc=124" > $O/rc
else echo "FAIL rc=$rc" > $O/rc; fi
