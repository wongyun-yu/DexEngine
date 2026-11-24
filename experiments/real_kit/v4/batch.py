"""v4 full run (many clips): `run all` for every clip of the table, one per GPU lane, with a host RAM gate and an event
file.

Table (--table TSV, header clip campaign bundle cells ks): one clip per row · ks = comma list of variant numbers (all
variants in the set) · cells = the judge cell root that made the clip's v2 (parent of <campaign>/<bundle>). The table
is data — this script holds no values. Lanes (--gpus 0,1): one clip at a time per GPU; rows appended to the table while
running are launched too (TABLE_ADD). Before a new clip it waits for MemAvailable ≥ --ram-gb and root disk free ≥
--disk-gb (both required). Event file <root>/events.log (appended): START · DONE <clip> rc passing/all variants · FAIL
<clip> rc reason · RAM_WAIT · DISK_WAIT · ALL_DONE (read by the watcher). Marker [AUG_V4_BATCH]. The caller exports
PYTHONPATH with this code tree's root — else death (children inherit it).
  python -m experiments.real_kit.v4.batch --main <data root> --table T --root R --v2-shadow S --gpus 0,1 --ram-gb 70
      --disk-gb 30 --all-obj-collision 1 [--job-timeout 21600]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T = Path(__file__).resolve().parents[3]
MARK = "[AUG_V4_BATCH]"


def mem_avail_gb() -> float:
    for ln in Path("/proc/meminfo").read_text().splitlines():
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 1024 / 1024
    raise SystemExit(f"{MARK} /proc/meminfo has no MemAvailable")


def clip_result(root: Path, r: dict) -> tuple[int, int, int, int]:
    """One finished clip → (variants passing the post gate, rolled-out variants, variants passing the join gate, table
    ks count).
    """
    rj = sorted((root / "clips" / f"{r['clip']}_pl_rt").glob("roll_*/retry.json"))
    ok = tot = gate = 0
    if rj:
        v = json.loads(rj[-1].read_text())["variants"]
        ok, tot = sum(bool(x["post_ok"]) for x in v.values()), len(v)
        gate = join_gate_ok(rj[-1].parent, v)
    return ok, tot, gate, len([k for k in r["ks"].split(",") if k.strip()])


def join_gate_ok(rd: Path, variants: dict) -> int:
    """Per variant: does the v4_metrics.json row of the round that passed the post gate (ok_round) pass the join stop
    gate (join_gate)? → count.
    """
    n = 0
    for name, st in variants.items():
        if not st.get("post_ok"):
            continue
        rr = next(x for x in st["rounds"] if x["round"] == st["ok_round"])
        mj = rd / f"r{rr['round']}_s{rr['seed']}" / "v4_metrics.json"
        rows = json.loads(mj.read_text())["rows"] if mj.is_file() else []
        n += any(row["name"] == name and row.get("join_gate") is True for row in rows)
    return n


def _parse_args():
    """Parse the command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--v2-shadow", required=True, type=Path)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--ram-gb", required=True, type=float)
    ap.add_argument("--disk-gb", required=True, type=float)
    ap.add_argument("--job-timeout", type=int, default=21600)
    ap.add_argument("--main", type=Path, required=True, help="canonical data root passed to every run")
    ap.add_argument(
        "--all-obj-collision",
        required=True,
        choices=("0", "1"),
        help="RL switch all_obj_collision passed to every run (v4 set = 1)",
    )
    a = ap.parse_args()
    return a


def _batch_start(a, rows, event):
    """GPU list, the run environment (this tree must be on PYTHONPATH) and the BATCH_START event."""
    gpus = [g for g in a.gpus.split(",") if g.strip()]
    if T not in {Path(x).resolve() for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x}:
        raise SystemExit(
            f"{MARK} PYTHONPATH does not hold this code tree {T} — another tree's dexengine would be imported"
            " (the caller exports it)"
        )
    env = dict(os.environ, AUG_V4="1")
    n_var = sum(len([k for k in r["ks"].split(",") if k.strip()]) for r in rows)
    event(
        "BATCH_START",
        f"clips {len(rows)} · variants {n_var} · GPU {gpus} · RAM gate {a.ram_gb} GB · "
        f"disk gate {a.disk_gb} GB · tree {T} · ALL_OBJ={a.all_obj_collision}",
    )
    return gpus, env


def _take_new_rows(read_table, seen, todo, rows, event):
    """Launch rows appended to the table while running: add them to the queue and the row list."""
    new = [
        r for r in read_table() if r["clip"] not in seen
    ]  # rows appended to the table while running are launched too
    for r in new:
        seen.add(r["clip"])
        todo.append(r)
        rows.append(r)
        event("TABLE_ADD", r["clip"], f"variants {len([k for k in r['ks'].split(',') if k.strip()])}")


def _reap(run, a, event, fails):
    """Collect finished or timed-out runs (DONE needs every variant passed and gated); return the failure count."""
    for g, (r, p, t0, log) in list(run.items()):
        rc = p.poll()
        if rc is None:
            if time.time() - t0 > a.job_timeout:
                p.kill()
                event("FAIL", r["clip"], "rc=timeout", f"{a.job_timeout}s", log)
                fails += 1
                del run[g]
            continue
        del run[g]
        ok, tot, gate, n_ks = clip_result(a.root, r)
        if rc == 0 and tot == n_ks and ok == tot and gate == tot:
            event(
                "DONE",
                r["clip"],
                f"rc={rc}",
                f"variants {ok}/{n_ks} · join gate {gate}/{tot}",
                f"{time.time() - t0:.0f}s",
            )
        else:
            fails += 1
            event(
                "FAIL",
                r["clip"],
                f"rc={rc}",
                f"variants passed {ok}/{tot} · join gate {gate}/{tot} · table {n_ks}",
                log,
            )
    return fails


def _run_cmd(a, r, g):
    """The per-clip run command."""
    cmd = [
        sys.executable,
        "-m",
        "experiments.real_kit.v4.run",
        "all",
        "--root",
        str(a.root),
        "--main",
        str(a.main),
        "--campaign",
        r["campaign"],
        "--bundle",
        r["bundle"],
        "--clip",
        r["clip"],
        "--cells",
        r["cells"],
        "--v2-shadow",
        str(a.v2_shadow),
        "--ks",
        r["ks"],
        "--gpu",
        g,
        "--all-obj-collision",
        a.all_obj_collision,
    ]
    return cmd


def main() -> int:
    a = _parse_args()
    a.main = a.main.resolve()
    need = {"clip", "campaign", "bundle", "cells", "ks"}

    def read_table() -> list[dict]:
        rows_ = list(csv.DictReader(a.table.open(), delimiter="\t"))
        if not rows_ or need - set(rows_[0]):
            raise SystemExit(f"{MARK} table {a.table} needs the header {need}")
        if len({r["clip"] for r in rows_}) != len(rows_):
            raise SystemExit(f"{MARK} a clip appears twice in the table")
        return rows_

    rows = read_table()
    a.root.mkdir(parents=True, exist_ok=True)
    ev = a.root / "events.log"

    def event(*x):
        line = f"{time.strftime('%F %T')} " + " ".join(str(v) for v in x)
        with ev.open("a") as f:
            f.write(line + "\n")
        print(MARK, line, flush=True)

    gpus, env = _batch_start(a, rows, event)
    todo = list(rows)
    run: dict[str, tuple] = {}
    fails = 0
    seen = {r["clip"] for r in rows}
    while todo or run:
        _take_new_rows(read_table, seen, todo, rows, event)
        fails = _reap(run, a, event, fails)
        free = [g for g in gpus if g not in run]
        if todo and free:
            av = mem_avail_gb()
            if av < a.ram_gb:
                event("RAM_WAIT", f"{av:.0f} GB < {a.ram_gb}")
                time.sleep(60)
                continue
            dk = shutil.disk_usage(a.root).free / 1024**3
            if dk < a.disk_gb:
                event("DISK_WAIT", f"{dk:.0f} GB < {a.disk_gb}")
                time.sleep(60)
                continue
            r = todo.pop(0)
            g = free[0]
            log = a.root / "logs" / f"{r['clip']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = _run_cmd(a, r, g)
            p = subprocess.Popen(cmd, cwd=str(T), env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)
            run[g] = (r, p, time.time(), log)
            event("START", r["clip"], f"GPU {g}", f"pid {p.pid}", f"MemAvailable {av:.0f} GB")
            continue
        time.sleep(30)
    event("ALL_DONE", f"failed {fails} · clips {len(rows)}")
    return 0 if not fails else 6


if __name__ == "__main__":
    sys.exit(main())
