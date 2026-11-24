#!/bin/bash
# Per-clip command files of SPIDER, FlashCHORD and EgoEngine (the one owner of these command lines):
# clip list (TSV: clip, chord_dir, ...) -> <out_dir>/{spider,chord,egoengine}_cmds.txt, one line per clip.
# The scorer (experiments/scoring/score.py) runs in the FlashCHORD venv for every method: it calls CHORD's released
# metrics (flash_chord.evaluation.metrics, numpy 2). Its chord mode resolves object URDFs through experiments.workspace,
# so that call runs it as a module from the repo (-s -P: no user site, no cwd on sys.path); the other modes are
# self-contained and run with -I.
# usage: gen_cmds.sh <clips.tsv> <out_dir>      (the Tab. 2 list: experiments/scoring/clips/cand_B0r_random_chordfull.tsv)
set -euo pipefail
B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$HERE/../../.." && pwd)
LIST=$1; OUT=$2; mkdir -p "$OUT"
SCORE=$R/experiments/scoring/score.py
FC=$B/code/chord_v2d/robotic_grounding/flash_chord; P=$B/data/chord/assets/human_motion_data/taco/processed
SPEED="\$(python3 -I $R/experiments/baselines/chord/motion_speed.py $FC/src/flash_chord/configs/experiment/sharpa_flash_sac.yaml)"
: > "$OUT/spider_cmds.txt"; : > "$OUT/chord_cmds.txt"; : > "$OUT/egoengine_cmds.txt"
tail -n +2 "$LIST" | cut -f1,2 | while IFS=$'\t' read -r c d; do
  echo "cd $B/code/spider && CUDA_VISIBLE_DEVICES=\${GPU:-0} timeout 7200 .venv/bin/python examples/run_mjwp.py +override=taco_sharpa_origin task=$c dataset_dir=$B/data/spider > $B/logs/spider/$c.log 2>&1; echo rc=\$? >> $B/logs/spider/$c.log; env -u LD_LIBRARY_PATH $FC/.venv/bin/python -I $SCORE spider --clip-dir $B/data/spider/processed/taco/sharpa_wave/bimanual/$c --sim-dt 0.01 --out $B/results/spider/$c.json" >> "$OUT/spider_cmds.txt"
  # scripts/* in the CHORD line are files of the FlashCHORD checkout ($FC, external), not this tree's.
  # CHORD training timeout 6 h is a hang guard, not a budget (CHORD's own constraint overflow can halve its rate on a
  # clip); a capped training is TRIM = FAIL
  O=$B/results/chord/$c; PQ="$P/sequence_id=$d/robot_name=sharpa_wave"
  echo "mkdir -p $O && cd $FC && CUDA_VISIBLE_DEVICES=\${GPU:-0} timeout 21600 .venv/bin/python scripts/train_flash_sac.py experiment=sharpa_flash_sac \"task.parquet='$PQ'\" reset.seed=42 logging.mode=disabled output_dir=$O/train > $B/logs/chord/$c.train.log 2>&1 && CUDA_VISIBLE_DEVICES=\${GPU:-0} timeout 3600 .venv/bin/python scripts/evaluate_policy.py \"evaluation.checkpoint=$O/train/policy_250003456.safetensors\" evaluation.world_count=64 \"evaluation.metrics_output=$O/metrics.json\" \"evaluation.object_trajectories_output=$O/objtraj.parquet\" > $B/logs/chord/$c.eval.log 2>&1; PYTHONPATH=$R DEXENGINE_E123_ROOT=$B $FC/.venv/bin/python -s -P -m experiments.scoring.score chord --traj $O/objtraj.parquet --motion-parquet $PQ/motion_v1.parquet --time-scale $SPEED --out $O/score.json" >> "$OUT/chord_cmds.txt"
  echo "cd $B/code/egoengine && CUDA_VISIBLE_DEVICES=\${GPU:-0} timeout 10800 .venv-action/bin/python action/examples/run_mjwp_mixed.py --dataset-dir $B/data/egoengine --dataset-name taco --robot-type sharpa_wave --embodiment-type bimanual --active-hand bimanual --task $c --data-id 0 --seed 42 --device cuda:0 > $B/logs/egoengine/$c.log 2>&1; echo rc=\$? >> $B/logs/egoengine/$c.log; env -u LD_LIBRARY_PATH $FC/.venv/bin/python -I $SCORE egoengine --clip-dir $B/data/egoengine/processed/taco/sharpa_wave/bimanual/$c --sim-dt 0.01 --out $B/results/egoengine/$c.json" >> "$OUT/egoengine_cmds.txt"
done
n=$(tail -n +2 "$LIST" | wc -l); for m in spider chord egoengine; do k=$(wc -l < "$OUT/${m}_cmds.txt"); [ "$k" = "$n" ] || { echo "[GEN_CMDS] FAIL $m $k != $n"; exit 1; }; done
echo "[GEN_CMDS] OK $n clips → $OUT"
