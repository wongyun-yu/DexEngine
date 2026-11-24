#!/bin/bash
# FlashCHORD inputs, last step: the support surfaces of the 64 sequences with CHORD's own reconstruct_support_surfaces.py
# -> data/chord/assets/human_motion_data/taco/reconstructed_stage/*_support.usda. Log: logs/chord_support.log.
B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"
RG=$B/code/chord_v2d/robotic_grounding; L=$B/logs; P=$B/data/chord/assets/human_motion_data/taco/processed
cd $RG && CUDA_VISIBLE_DEVICES="" HUMAN_MOTION_DATA_DIR=$B/data/chord/assets/human_motion_data PYTHONPATH=$RG/source/robotic_grounding \
  timeout 1500 $B/code/chord_v2d/.venv-prep/bin/python scripts/reconstruct_support_surfaces.py --input_dir $P --dataset taco --robot_name sharpa_wave > $L/chord_support.log 2>&1
echo "[CHORD_SUPPORT] rc=$? usda=$(ls $B/data/chord/assets/human_motion_data/taco/reconstructed_stage/*_support.usda 2>/dev/null | wc -l)/64" >> $L/chord_support.log
