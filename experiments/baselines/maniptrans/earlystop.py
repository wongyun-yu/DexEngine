#!/usr/bin/env python3
"""ManipTrans imitator early stop, applied from outside the run.

The authors' rule lives in <maniptrans>/lib/rl/base.py (`if epoch_num >= self.early_stop_epochs:` -> slope of
reward_history over the last 8 · 32 · 64 epochs, all <= 1e-4 -> 'EARLY STOP!'), with the imitator default
early_stop_epochs = 500 (main/cfg/rl_train/DexHandImitatorPPO.yaml `${resolve_default:500,…}`). The imitator runs carry
a top-level early_stop_epochs of 9999999999999 in config.yaml, which resolve_default prefers, so the rule never fires
in-run. This tool reads the rule's numbers from those two files (it stops if their lines changed) and the run's reward
history from its tensorboard events (rewards/iter = the mean_rewards[0] the run appends to reward_history), and prints
the first epoch at which the rule fires.
usage: earlystop.py <events.out.tfevents…> --code <maniptrans checkout>   -> one [MT_EARLYSTOP] line (exit 0 fired · 1
not)
(needs tensorboard + scipy, e.g. the EgoEngine venv)"""

import argparse
import pathlib
import re
import sys

import numpy as np
from scipy.stats import linregress


def find(lines, pat, what):
    hits = [(i + 1, m) for i, ln in enumerate(lines) if (m := re.search(pat, ln))]
    if len(hits) != 1:
        sys.exit(f"[MT_EARLYSTOP] FAIL rule source changed: {what} — {len(hits)} matches for {pat!r}")
    return hits[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--code", required=True, help="the ManipTrans checkout (<workspace>/code/maniptrans)")
    a = ap.parse_args(argv)
    C = pathlib.Path(a.code)
    base = (C / "lib/rl/base.py").read_text().splitlines()
    yml = (C / "main/cfg/rl_train/DexHandImitatorPPO.yaml").read_text().splitlines()
    l_gate, _ = find(base, r"if epoch_num >= self\.early_stop_epochs:", "base.py gate")
    l_h, mh = find(base, r"for horizon in \[([0-9, ]+)\]", "base.py horizons")
    l_s, ms = find(base, r"if slope > ([0-9.e-]+):", "base.py slope threshold")
    l_e, me = find(yml, r"early_stop_epochs: \$\{resolve_default:(\d+),", "imitator yaml early_stop_epochs default")
    H = [int(x) for x in mh.group(1).split(",")]
    TH = float(ms.group(1))
    E0 = int(me.group(1))
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(a.events, size_guidance={"scalars": 0})
    ea.Reload()
    ev = sorted((e.step, e.value) for e in ea.Scalars("rewards/iter"))
    ep = np.array([s for s, _ in ev])
    r = np.array([v for _, v in ev])
    src = (
        f"base.py:{l_gate},{l_h},{l_s} · DexHandImitatorPPO.yaml:{l_e} (early_stop_epochs {E0} · horizons {H} "
        f"· slope ≤ {TH:g})"
    )
    fired = None
    for k in range(len(r)):
        if ep[k] < E0 or k + 1 < max(H):
            continue
        sl = [linregress(np.arange(h), r[k + 1 - h : k + 1]).slope for h in H]
        if all(s <= TH for s in sl):
            fired = (int(ep[k]), sl)
            break
    last = f"last epoch {int(ep[-1])} · reward {r[-1]:.2f} · best {r.max():.2f} @ {int(ep[r.argmax()])}"
    if fired:
        print(
            f"[MT_EARLYSTOP] FIRED at epoch {fired[0]} slopes {', '.join(f'{s:.2e}' for s in fired[1])} · "
            f"{last} · rule {src}"
        )
        sys.exit(0)
    sl = [linregress(np.arange(h), r[-h:]).slope for h in H] if len(r) >= max(H) else []
    print(f"[MT_EARLYSTOP] not fired · {last} · slopes now {', '.join(f'{s:.2e}' for s in sl)} · rule {src}")
    sys.exit(1)


if __name__ == "__main__":
    main()
