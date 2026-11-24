#!/bin/bash
# ManipTrans imitator pre-training pool: CHORD's released TACO clips with the full schema (MANO columns) minus the 64 eval
# clips (compared by full sequence id and by the date_idx key), written to pretrain_pool.txt with the leak proof
# pretrain_leak_proof.json next to this script; then the pool's CHORD support surfaces (CHORD's own
# reconstruct_support_surfaces.py) and the per-sequence lowest support top z -> data/maniptrans/support_top.json (read by the
# ManipTrans TACO loader).
# Run after chord/fetch_hf_motions.sh has written data/chord_hf_all_taco. Log: logs/pretrain_pool.log.
B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$HERE/../../.." && pwd)
L=$B/logs; D=$B/data/chord_hf_all_taco; RG=$B/code/chord_v2d/robotic_grounding
EVAL=$R/experiments/scoring/clips/cand_B0r_random_chordfull.tsv
$RG/flash_chord/.venv/bin/python -I - "$D" "$EVAL" "$HERE" >> $L/pretrain_pool.log 2>&1 <<'PY'
import sys, os, glob, csv, json, re, pyarrow.parquet as pq
D, EVAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
allp = sorted(glob.glob(D + "/*.parquet")); full, robot = [], []
for p in allp:
    (full if "mano_right_joints" in pq.read_schema(p).names else robot).append(os.path.basename(p)[:-8])
ev = [r["chord_dir"] for r in csv.DictReader(open(EVAL), delimiter="\t")]
evk = {re.search(r"(\d{8}_\d+)$", s).group(1) for s in ev}
pool = [s for s in full if s not in set(ev) and re.search(r"(\d{8}_\d+)$", s).group(1) not in evk]
inter_id = sorted(set(pool) & set(ev)); inter_key = sorted({re.search(r"(\d{8}_\d+)$", s).group(1) for s in pool} & evk)
open(OUT + "/pretrain_pool.txt", "w").write("\n".join(pool) + "\n")
proof = {"rule": "pool = CHORD-public TACO full-schema (has MANO) minus eval B0r 64; compared by full sequence id AND by date_idx key",
         "chord_files": len(allp), "full_schema": len(full), "robot_only": len(robot), "eval": len(ev), "pool": len(pool),
         "intersection_by_sequence_id": inter_id, "intersection_by_date_idx": inter_key,
         "eval_all_full_schema": all(s in set(full) for s in ev)}
json.dump(proof, open(OUT + "/pretrain_leak_proof.json", "w"), indent=1)
print("[POOL]", json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in proof.items() if k != "rule"}))
assert not inter_id and not inter_key and len(pool) + len(ev) <= len(full)
PY
# the pool's support surfaces (CHORD's script reads HUMAN_MOTION_DATA_DIR/taco/processed/sequence_id=*/robot_name=*/):
# link the pool parquets into that layout
P=$B/data/maniptrans/pool_hmd/taco/processed; mkdir -p $P
while IFS= read -r s; do mkdir -p "$P/sequence_id=$s/robot_name=sharpa_wave"; ln -sfn "$D/$s.parquet" "$P/sequence_id=$s/robot_name=sharpa_wave/motion_v1.parquet"; done < "$HERE/pretrain_pool.txt"
cd $RG && CUDA_VISIBLE_DEVICES="" HUMAN_MOTION_DATA_DIR=$B/data/maniptrans/pool_hmd PYTHONPATH=$RG/source/robotic_grounding \
  timeout 5400 $B/code/chord_v2d/.venv-prep/bin/python scripts/reconstruct_support_surfaces.py --input_dir $P --dataset taco --robot_name sharpa_wave > $L/pool_support.log 2>&1
echo "[POOL_SUPPORT] rc=$? usda=$(ls $B/data/maniptrans/pool_hmd/taco/reconstructed_stage/*_support.usda 2>/dev/null | wc -l)/$(wc -l < "$HERE/pretrain_pool.txt")" >> $L/pretrain_pool.log
$B/code/chord_v2d/.venv-prep/bin/python -I - "$B" "$R/experiments/baselines/chord/support_tops.py" >> $L/pretrain_pool.log 2>&1 <<'PY'
import sys, json, subprocess
B, TOPS = sys.argv[1], sys.argv[2]; out = {}
for d in (f"{B}/data/chord/assets/human_motion_data/taco/reconstructed_stage", f"{B}/data/maniptrans/pool_hmd/taco/reconstructed_stage"):
    subprocess.run([sys.executable, "-I", TOPS, d, f"{B}/data/maniptrans/_tops_tmp.json"], check=True)
    out.update(json.load(open(f"{B}/data/maniptrans/_tops_tmp.json")))
json.dump(out, open(f"{B}/data/maniptrans/support_top.json", "w"), indent=0, sort_keys=True)
print("[POOL_TOPS]", len(out))
PY
echo "DONE" >> $L/pretrain_pool.log
