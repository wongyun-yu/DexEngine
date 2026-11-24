#!/bin/bash
# Fake Isaac Lab entry: `isaaclab.sh -p <script> <trainer argv>`. Emulates the trainer's files and log markers for the
# orchestration oracle. Scenario: FAKE_SCENARIO = ok | oom_once | fail | noeval | mem_restart. Calls are appended to
# $FAKE_CALLS ("exp=… task=… resume=<ckpt basename|-> resume_it=…").
shift   # -p
script=$1; shift
exp=""; mi=""; task=""; resume="-"; rit="-"; prev=""
for a in "$@"; do
  case "$prev" in
    --exp_name) exp=$a ;; --max_iterations) mi=$a ;; --task) task=$a ;;
    --resume_from) resume=$(basename "$a") ;; --resume_iteration) rit=$a ;;
  esac
  prev=$a
done
[ "$task" = eval ] && exp=${exp:-eval}
echo "exp=$exp task=$task resume=$resume resume_it=$rit script=$(basename "$script")" >> "$FAKE_CALLS"
[ "$task" = train ] || { echo "fake eval"; exit 0; }
n=$(grep -c "task=train" "$FAKE_CALLS")
# the trainer's checkpoint area = its run tree's work_ckpt (rl.train.env_build: paths.work_area), not the cwd
ck=$(python3 -m dexengine.core.paths --what work-area --area work_ckpt)/$exp/checkpoints || exit 2
repo=$(python3 -c 'from dexengine.core.repo import REPO_ROOT; print(REPO_ROOT)') || exit 2
case "$(realpath -m "$ck")" in "$repo"/*) echo "fake trainer: $ck is inside the code checkout $repo — refused"; exit 3 ;; esac
mkdir -p "$ck"
roll=32; start=$([ "$rit" = - ] && echo 0 || echo "$rit")
echo "  Max iterations: $mi"; echo "  Eval interval: ${FAKE_EVAL_INTERVAL:-200}"
stop=$mi; die=""
case "${FAKE_SCENARIO:-ok}" in
  oom_once) [ "$n" = 1 ] && { stop=800; die="CUDA out of memory. Tried to allocate 2.00 GiB"; } ;;
  mem_restart) [ "$n" -le 2 ] && { stop=$(( start + 400 )); die="[MEM_RESTART] iter $(( start + 400 )) free 2.1 GB < 3 GB"; } ;;
  fail) echo "Traceback: boom"; exit 1 ;;
esac
for ((it = start + 50; it <= stop; it += 50)); do
  : > "$ck/agent_$(( it * roll )).pt"
  printf '\r  3%%|3   | %d/51200 [40:10<20:01:00,  1.45s/it]\n[CKPT_SAVED] agent_%d.pt iter=%d\n' $(( it * roll - 1 )) $(( it * roll )) "$it"
done
if [ -n "$die" ]; then echo "$die"; exit 1; fi
echo "  Training Complete"; echo "  Checkpoints: $ck"
exit 0
