#!/bin/bash
# ManipTrans imitator-pool retarget (the authors' order: mano2dexhand for the whole imitator pool, both hands, before the
# imitator). One call = one GPU running P items at once; an item = <side>:<data_idx>[:<cap s>].
# The item line is the production retarget line (the residual rows' own: mano2dexhand.py --dexhand sharpa_wave --side <s>
# --headless --iter <IT> under timeout <TO>); common/mkcmds.py passes IT/TO/PY/CODE read from run/maniptrans_cmds.txt and
# each item's cap (mkcmds.item_cap, scaled by the clip's motion frames; TO when none is given).
# An item whose pkl already exists is skipped: the authors' script writes the pkl only at its very end.
# usage: retarget_batch.sh <P> <logdir> <IT> <TO> <PY> <CODE> <side:idx[:cap s]>...
# writes <logdir>/items.tsv (idx side rc seconds frames · one line per item) · <logdir>/<idx>.<r|l>.log · gpu samples, and
# prints  [MT_RTBATCH] P=… n=… ok=… fail=… skipped=… wall=… item_s_med=… item_s_max=… frames_med=… peak_mib=… per_proc_mib_max=…
# rc 0 only when every item's pkl exists afterwards.
set -u
P=${1:?P}; LD=${2:?logdir}; IT=${3:?iter}; TO=${4:?timeout}; PY=${5:?python}; CODE=${6:?code dir}; shift 6
[ $# -ge 1 ] || { echo "[MT_RTBATCH] FAIL no items"; exit 2; }
# the pkl dir = the loader's own constant (taco_chord_dataset_dexhand.TACO_RETARGET_DIR): its two assignment statements
# are evaluated here, because importing that module needs isaacgym before torch
RD=$("$PY" - "$CODE/main/dataset/taco_chord_dataset_dexhand.py" 2>&1 <<'PYRD'
import ast, os, sys
env = {"os": os}
for n in ast.parse(open(sys.argv[1]).read()).body:
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("_B", "TACO_RETARGET_DIR") for t in n.targets):
        exec(compile(ast.Module(body=[n], type_ignores=[]), sys.argv[1], "exec"), env)
print(env["TACO_RETARGET_DIR"])
PYRD
)
case "$RD" in /*/data/maniptrans/retargeted) ;; *) echo "[MT_RTBATCH] FAIL cannot read TACO_RETARGET_DIR from $CODE: $RD"; exit 2;; esac
mkdir -p "$LD"; : > "$LD/items.tsv"; : > "$LD/gpu.csv"; : > "$LD/apps.csv"
( while :; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 >> "$LD/gpu.csv" 2>/dev/null
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i 0 >> "$LD/apps.csv" 2>/dev/null; sleep 2; done ) &
SMP=$!
one() {   # side:idx[:cap]
  local s=${1%%:*} idx=${1#*:} cap=$TO t0 rc f k
  case $idx in *:*) cap=${idx##*:}; idx=${idx%:*};; esac
  [[ "$cap" =~ ^[1-9][0-9]*$ ]] || { echo "$idx	$s	9	0	0	$cap" >> "$LD/items.tsv"; return; }
  case $s in right) k=rh;; left) k=lh;; *) echo "$idx	$s	9	0	0" >> "$LD/items.tsv"; return;; esac
  f="$RD/mano2sharpa_wave_$k/$idx.pkl"
  if [ -s "$f" ]; then echo "$idx	$s	skip	0	-" >> "$LD/items.tsv"; return; fi
  t0=$(date +%s.%N)
  ( cd "$CODE" && export PATH="$(dirname "$PY"):$PATH" LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      CUDA_VISIBLE_DEVICES=${GPU:-0} && timeout "$cap" "$PY" main/dataset/mano2dexhand.py --data_idx "$idx" --dexhand sharpa_wave \
      --side "$s" --headless --iter "$IT" ) > "$LD/$idx.${k:0:1}.log" 2>&1
  rc=$?; [ "$rc" = 0 ] && [ ! -s "$f" ] && rc=8
  fr=$([ -s "$f" ] && "$PY" -c 'import pickle,sys; print(pickle.load(open(sys.argv[1],"rb"))["opt_dof_pos"].shape[0])' "$f" 2>/dev/null || echo -)
  echo "$idx	$s	$rc	$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%d", b-a}')	$fr	$cap" >> "$LD/items.tsv"
}
export -f one; export LD RD IT TO PY CODE
T0=$(date +%s)
printf '%s\n' "$@" | xargs -d '\n' -P "$P" -I{} bash -c 'one "$1"' _ {}
W=$(( $(date +%s) - T0 )); kill "$SMP" 2>/dev/null; wait "$SMP" 2>/dev/null
n=$#; ok=0; miss=0
for it in "$@"; do s=${it%%:*}; idx=${it#*:}; idx=${idx%%:*}; k=rh; [ "$s" = left ] && k=lh; [ -s "$RD/mano2sharpa_wave_$k/$idx.pkl" ] && ok=$((ok+1)) || miss=$((miss+1)); done
st=$(awk -F'\t' '$3!="skip"{print $4}' "$LD/items.tsv" | sort -n | awk '{a[NR]=$1} END{if(NR) printf "%s %s", a[int((NR+1)/2)], a[NR]; else printf "- -"}')
fm=$(awk -F'\t' '$5!="-"&&$3!="skip"{print $5}' "$LD/items.tsv" | sort -n | awk '{a[NR]=$1} END{print (NR? a[int((NR+1)/2)] : "-")}')
pk=$(cut -d, -f1 "$LD/gpu.csv" | sort -n | tail -1); pp=$(cut -d, -f2 "$LD/apps.csv" | sort -n | tail -1)
sk=$(awk -F'\t' '$3=="skip"' "$LD/items.tsv" | wc -l); fl=$(awk -F'\t' '$3!="skip"&&$3!="0"' "$LD/items.tsv" | wc -l)
echo "[MT_RTBATCH] P=$P n=$n ok=$ok fail=$fl skipped=$sk missing=$miss wall=${W}s item_s_med=${st% *} item_s_max=${st#* } frames_med=$fm peak_mib=${pk:-?} per_proc_mib_max=${pp:-?}"
[ "$miss" = 0 ]
