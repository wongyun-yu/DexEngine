#!/bin/bash
# DexMachina card speed probe: the author recipe on clip e123c000 (same code, B=4096,
# same clip as the reference RTX 5090 run) for 20 epochs on the row's card. Steady s/epoch = wall between the epoch-10 and epoch-20 lines
# (env build, kernel compile and the first epochs excluded) = the window of the RTX 5090 figure (epochs 10->20).
# Run on a card other than the reference to convert the RL cap to that card.
# usage: dexmachina_fps_row.sh <gpu>  -> <workspace>/logs/dexmachina/fps/c000_E20/fps.json (+ an [FPS] line)
set -u
GPU=${1:?gpu}; B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"; PY=$B/code/dexmachina/env/bin/python
DM=$(cd "$(dirname "$0")/../../baselines/dexmachina" && pwd) || exit 2   # the DexMachina row scripts
CARD=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" | head -1)
# rl-games names its run dir without the epoch count: an e123c000 B4096 dir already here (the reference run) would be written into
X=$(ls -d "$B"/code/dexmachina/dexmachina/logs/rl_games/sharpa_wave_hand/sharpa-e123_e123c0000-238-s01-u01_B4096_* 2>/dev/null | head -1)
[ -z "$X" ] || { echo "[FPS] FAIL a c000 B4096 run dir exists ($X) — this row runs only on a fresh tree (a cluster job)"; exit 2; }
echo "[FPS] start c000 B4096 20 epochs card=$CARD"
# the card ratio is defined on the original recipe: a matched row (DEXM_MATCH=1) probes with the switch off
# (e123c000 is not a 0.2 kg clip, and the probe has no cap) — DEXM_MATCH unset here = the probe as every earlier row ran it
env -u DEXM_MATCH DM_TRAIN_LIMIT=timeout DM_TRAIN_TIMEOUT=5400 bash "$DM/train_clip.sh" e123c000 238 4096 "$GPU" 20
O=$B/logs/dexmachina/train/e123c000_B4096_E20; F=$B/logs/dexmachina/fps/c000_E20; mkdir -p "$F"
grep -q "^DONE" "$O/rc" || { echo "[FPS] FAIL train $(cat "$O/rc")"; tail -n 5 "$O/train.log"; exit 3; }
"$PY" - "$O" "$CARD" "$F/fps.json" <<'PY'
import json, sys
o, card, out = sys.argv[1:]
ep = {}
for l in open(o + "/epochs.tsv"):
    t, e = l.split(); ep.setdefault(int(e), int(t))      # first time each epoch line was seen (2 s poll)
assert 10 in ep and 20 in ep, sorted(ep)
s = (ep[20] - ep[10]) / 10
d = dict(card=card, clip="e123c000", num_envs=4096, epochs=20, s_per_epoch=round(s, 2), window="epochs 10->20",
         t10=ep[10], t20=ep[20], poll_s=2, train_s=int(open(o + "/end").read()) - int(open(o + "/start").read()),
         commit=open(o + "/commit").read().strip())
json.dump(d, open(out, "w"), indent=1)
print("[FPS]", json.dumps(d))
PY
