#!/usr/bin/env python3
"""DexMachina one-time setup of one clip, measured from the run's own logs (the Tab. 2 time = steady GPU-minutes per
clip on L40S = wall time minus the one-time setup the run's own log measures; evaluation not included). The per-clip
work that produces the policy = retarget + contact precompute + training (to the cap); the
evaluation is not in it. Each setup term is read from timestamped lines (the [Genesis] logger's HH:MM:SS, local time):
  retarget  parallel_retarget.log: the first [Genesis] line -> the first [Genesis] line after 'Compiling simulation
            kernels...' (Genesis init · scene build · kernel compile — the next stamped line is the first sim step)
  contacts  map_contacts.log: the first [Genesis] line -> 'Compiling simulation kernels...' (Genesis init · scene
  build)
  training  train dir: 'start' (unix s, written at launch) -> the first [Genesis] line of train.log (imports · demo
  load
  ·
            scene build · kernel compile: everything before the environment's first step)
Not measured — left inside the steady time (no stamped line bounds them): the contacts kernel compile (the next stamped
line is the exit), and for retarget / contacts the python import before the first [Genesis] line and the kernel-cache
write after 'Exiting Genesis'. Walls: retarget / contacts = their rc 'OK sec=N' (the row's retarget and
contact-mapping steps), training = end - start.
usage: onetime.py <logs/dexmachina/clips/<key>> <logs/dexmachina/train/<key>_B4096_E5000> <out json>"""

import json
import os
import re
import sys
import time
from pathlib import Path

STAMP = re.compile(r"^\[Genesis\] \[(\d\d):(\d\d):(\d\d)\]")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def stamped(path):
    """[(seconds of day, text)] of the [Genesis] lines, in file order."""
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = ANSI.sub("", line).rstrip("\n")
            m = STAMP.match(line)
            if m:
                out.append((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]), line))
    if not out:
        raise SystemExit(f"[ONETIME] FAIL {path}: no [Genesis] stamped line")
    return out


def span(a, b):  # b after a on a clock of day (a stage never runs a day)
    return (b - a) % 86400


def rc_sec(path):
    m = re.match(r"OK sec=(\d+)", Path(path).read_text().strip())
    if not m:
        raise SystemExit(f"[ONETIME] FAIL {path}: not 'OK sec=N'")
    return int(m[1])


def main():
    clip_dir, train_dir, out = sys.argv[1:4]
    key = os.path.basename(clip_dir.rstrip("/"))
    st = {}
    r = stamped(os.path.join(clip_dir, "parallel_retarget.log"))
    ic = next((i for i, (_, t) in enumerate(r) if "Compiling simulation kernels" in t), None)
    if ic is None or ic + 1 >= len(r):
        raise SystemExit(f"[ONETIME] FAIL {key} retarget: no stamped line after 'Compiling simulation kernels'")
    st["retarget"] = {
        "wall_s": rc_sec(os.path.join(clip_dir, "retarget.rc")),
        "setup_s": span(r[0][0], r[ic + 1][0]),
        "setup_from": r[0][1][:90],
        "setup_to": r[ic + 1][1][:90],
    }
    c = stamped(os.path.join(clip_dir, "map_contacts.log"))
    cc = next((i for i, (_, t) in enumerate(c) if "Compiling simulation kernels" in t), None)
    if cc is None:
        raise SystemExit(f"[ONETIME] FAIL {key} contacts: no 'Compiling simulation kernels' line")
    st["contacts"] = {
        "wall_s": rc_sec(os.path.join(clip_dir, "contacts.rc")),
        "setup_s": span(c[0][0], c[cc][0]),
        "setup_from": c[0][1][:90],
        "setup_to": c[cc][1][:90],
    }
    t0 = int(Path(os.path.join(train_dir, "start")).read_text().split()[0])
    t1 = int(Path(os.path.join(train_dir, "end")).read_text().split()[0])
    g = stamped(os.path.join(train_dir, "train.log"))
    lt = time.localtime(t0)
    s0 = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
    d = span(s0, g[0][0])
    if d >= 3600:  # the logger's clock and the start file disagree (another time zone?) — not a setup time
        raise SystemExit(
            f"[ONETIME] FAIL {key} training: first [Genesis] line {d} s after 'start' (clock or zone mismatch)"
        )
    st["training"] = {
        "wall_s": t1 - t0,
        "setup_s": d,
        "setup_from": f"start file {time.strftime('%H:%M:%S', lt)}",
        "setup_to": g[0][1][:90],
    }
    rec = {
        "key": key,
        "rule": "tools/dexmachina/onetime.py (docstring)",
        "stages": st,
        "wall_s": sum(v["wall_s"] for v in st.values()),
        "setup_s": sum(v["setup_s"] for v in st.values()),
        "unmeasured": [
            "contacts kernel compile",
            "retarget/contacts python import before the first [Genesis] line",
            "retarget/contacts kernel-cache write at exit",
        ],
        "logs": {"clip": os.path.abspath(clip_dir), "train": os.path.abspath(train_dir)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(out + ".tmp", out)
    print(
        f"[ONETIME] {key} wall {rec['wall_s']} s · setup {rec['setup_s']} s ("
        + " · ".join(f"{k} {v['setup_s']}/{v['wall_s']}" for k, v in st.items())
        + f") -> {out}"
    )


if __name__ == "__main__":
    main()
