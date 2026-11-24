#!/bin/bash
# CHORD's released TACO motions from Hugging Face: only motion_v1.parquet of the 1,215 sequences (about 0.9 GB), for the
# ManipTrans imitator pre-training pool (maniptrans/build_pretrain_pool.sh). -> data/chord_hf_all_taco/<sequence>.parquet
# The sequence list is experiments/scoring/clips/chord_hf_taco_dirs.txt. Log: logs/fetch_chord_all.log (ends with DONE).
B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$HERE/../../.." && pwd)
D=$B/data/chord_hf_all_taco; mkdir -p "$D"
sed 's#^taco/##' "$R/experiments/scoring/clips/chord_hf_taco_dirs.txt" | xargs -P 8 -I{} sh -c '[ -s "$0/{}.parquet" ] || curl -sfL -m 180 "https://huggingface.co/datasets/nvidia/video-to-data-robot-dexterity-task-library-and-dataset/resolve/main/taco/{}/motion_v1.parquet" -o "$0/{}.parquet" || echo "FAIL {}"' "$D" > $B/logs/fetch_chord_all.log 2>&1
echo "DONE files=$(ls $D | wc -l)/1215 size=$(du -sh $D | cut -f1)" >> $B/logs/fetch_chord_all.log
