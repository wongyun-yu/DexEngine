#!/usr/bin/env python3
"""f0 completion ledger: the single owner of the machine-readable f0 run summary.

Why a separate module: counts derived from the human-readable completion line break (e.g. a byte-based `cut` that
splits a multi-byte character leaves half a byte in the value), and several shell consumers (scripts/stage2_ground.sh,
the stage-1 driver) would each count differently. This module owns the file name, marker string, schema,
writing, reading and validation. The producer (prejudge.py) calls `write()`; shell consumers run it with
`--log` and get the values. No two places assemble the same name.

The reader is stdlib only: `scripts/stage2_ground.sh` runs it with a bare `python3`, where importing `paths` or the
prejudge would fail.

Why the file name differs per run: several runs write into one `<f0 shadow root>/<hand spec>/` at the same time (runs
with different emb, bimanual/right/left, and several stage2 chunks on the same canonical f0 root). A fixed name would
let a later run overwrite an earlier one, so a correct run would die with "not my ledger". The name therefore carries
`run_id`; consumers read the path from the marker printed in their own run's log and never assemble the name. There is
no `latest` symlink (it would be a second path to the same thing).

Contract
--------
  ledger    <f0 shadow root>/<hand spec>/f0_summary.<run_id>.json   (same directory as LEDGER.tsv)
  marker    [F0_SUMMARY] run=… ok=… total=… budget_skip=… rc=… json=…   (ASCII only)
  reader    python3 -m dexengine.retarget.f0.ledger --log <f0 log>
            -> one line `ok<TAB>total<TAB>budget_skip<TAB>rc<TAB>json`; missing or inconsistent -> rc != 0.
            `--field run_failure` = run-failure count from the non-judgeable split marker [F0_FAILCLASS] (counting
            rule owned by shadow_status, line format by failclass_line); run mode only, checked against the ledger
            total.

The human-readable completion line stays as it is; the values are owned by this JSON alone.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

#: Schema version. Bump it when a column is removed or changes meaning (adding columns keeps it).
SCHEMA = 1

#: Marker prefix. ASCII only, so a consumer that extracts values from the line never meets a multi-byte character.
MARKER = "[F0_SUMMARY]"

#: File name rule (owned here only).
_NAME_FMT = "f0_summary.{run_id}.json"
NAME_GLOB = "f0_summary.*.json"

#: Characters allowed in a run_id (safe in both file names and markers).
_RUN_ID_OK = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_T")


def new_run_id() -> str:
    """Name of this run: time (human-readable) + pid + random suffix (avoids collisions within one second)."""
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def summary_path(root, run_id: str) -> Path:
    """Ledger path (the only owner of the name)."""
    if not run_id or any(ch not in _RUN_ID_OK for ch in run_id):
        raise ValueError(f"run_id cannot be used in a file name: {run_id!r}")
    return Path(root) / _NAME_FMT.format(run_id=run_id)


def repo_commit(tree) -> str:
    """HEAD commit of the tree. Empty string when git is missing or this is not a tree (never raises)."""
    try:
        out = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def marker_line(d: dict, path) -> str:
    """ASCII log line. `json=` comes last, so a path with spaces does not shift the earlier fields."""
    return (
        f"{MARKER} run={d['run_id']} ok={d['ok']} total={d['total']} "
        f"budget_skip={d['budget_skip_sides']} rc={d['rc']} json={path}"
    )


#: Non-judgeable split marker. The counting rules (data defect, run failure) belong to shadow_status (is_data_defect,
#: is_run_failure); this module owns only the line format and its reading (ASCII keys, leading prefix).
FAILCLASS_MARKER = "[F0_FAILCLASS]"
FAILCLASS_KEYS = ("data_defect", "run_failure", "judgeable", "held")


def failclass_line(data_defect: int, run_failure: int, judgeable: int, held: int) -> str:
    return (
        f"{FAILCLASS_MARKER} data_defect={data_defect} run_failure={run_failure} judgeable={judgeable} held={held} "
        "(owner shadow_status)"
    )


def failclass(log_path, d: dict) -> dict:
    """Last [F0_FAILCLASS] line of the log -> {data_defect, run_failure, judgeable}, checked against this run's ledger
    d (from load). Raises when the run is not in run mode (sweeps do not split), the line is missing, or the three
    counts do not sum to total."""
    if d.get("mode") != "run":
        raise SystemExit(
            f"[f0_summary] non-judgeable split is counted for run-mode runs only — mode={d.get('mode')!r} ({log_path})"
        )
    hit, mk = None, FAILCLASS_MARKER.encode()
    with open(log_path, "rb") as fh:
        for raw in fh:
            if mk in raw:
                hit = raw[raw.index(mk) :].decode("ascii", "replace")
    if hit is None:
        raise SystemExit(
            f"[f0_summary] marker '{FAILCLASS_MARKER}' not in the log — non-judgeable split unknown: {log_path}"
        )
    kv = dict(t.split("=", 1) for t in hit.split() if "=" in t)
    try:
        out = {k: int(kv[k]) for k in FAILCLASS_KEYS}
    except (KeyError, ValueError):
        raise SystemExit(f"[f0_summary] malformed marker: {hit.strip()}") from None
    if sum(out.values()) != d.get("total"):
        raise SystemExit(
            f"[f0_summary] marker {out} sum ≠ ledger total {d.get('total')} — a line from another run: {log_path}"
        )
    return out


def build(
    *, run_id: str, rows: list, ok_statuses, budget_skip_sides: int, started: float, finished: float, tree, **meta
) -> dict:
    """Build the summary from the rows; counts are taken here once.

    `ok` and `fail_by_kind` are both derived from `by_status`, so no hand-subtracted category can drift from them.
    """
    by_status: dict[str, int] = {}
    for r in rows:
        st = str(r.get("status", "?"))
        by_status[st] = by_status.get(st, 0) + 1
    okset = set(ok_statuses)
    ok = sum(n for st, n in by_status.items() if st in okset)
    fail_by_kind = {st: n for st, n in sorted(by_status.items()) if st not in okset}
    total = len(rows)
    d = {
        "schema": SCHEMA,
        "run_id": run_id,
        "ok": ok,
        "total": total,
        "by_status": dict(sorted(by_status.items())),
        "fail_by_kind": fail_by_kind,
        "budget_skip_sides": int(budget_skip_sides),
        "no_mark": int(by_status.get("FAIL_NO_BUDGET_MARK", 0)),
        "clips_failed": [str(r.get("clip", "")) for r in rows if str(r.get("status", "?")) not in okset],
        "stale_regenerated": sum(1 for r in rows if r.get("stale")),
        "started": time.strftime("%FT%T", time.localtime(started)),
        "finished": time.strftime("%FT%T", time.localtime(finished)),
        "elapsed_s": round(finished - started, 1),
        "tree": str(tree),
        "commit": repo_commit(tree),
        "rc": 0 if ok == total else 1,
    }
    d.update(meta)
    return d


def write(root, d: dict) -> Path:
    """Write atomically (temp + os.replace), so a consumer never reads a half-written file."""
    p = summary_path(root, d["run_id"])
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
    return p


# reader side
def parse_marker(log_path) -> dict:
    """Fields of the last marker line in the log; raises when there is none.

    The log may not be valid UTF-8 (a dying worker can leave half a byte), so it is read as bytes and only the marker
    line is decoded as ASCII; this is why the marker is ASCII only.
    """
    p = Path(log_path)
    if not p.is_file():
        raise SystemExit(f"[f0_summary] log missing: {p}")
    hit = None
    mk = MARKER.encode()
    with open(p, "rb") as fh:
        for raw in fh:
            if mk in raw:
                hit = raw[raw.index(mk) :].decode("ascii", "replace").strip()
    if hit is None:
        raise SystemExit(
            f"[f0_summary] the producer wrote no completion ledger — marker '{MARKER}' not in the log: {p}\n"
            f"  (the prejudge died midway or is an old version without the marker; the run is invalid.)"
        )
    out: dict[str, str] = {}
    body = hit[len(MARKER) :].strip()
    # `json=` comes last: everything after it is the path (spaces allowed).
    j = body.find("json=")
    if j < 0:
        raise SystemExit(f"[f0_summary] marker has no json=: {hit}")
    out["json"] = body[j + len("json=") :].strip()
    for tok in body[:j].split():
        k, sep, v = tok.partition("=")
        if sep:
            out[k] = v
    for need in ("run", "ok", "total", "budget_skip", "rc"):
        if need not in out:
            raise SystemExit(f"[f0_summary] marker has no {need}=: {hit}")
    return out


def load(log_path) -> dict:
    """Marker -> read the JSON and check that marker and file belong to the same run.

    Ledgers of other runs sit in the same directory; a file the log points to that belongs to another run carries
    someone else's values, so this raises instead of using it.
    """
    m = parse_marker(log_path)
    jp = Path(m["json"])
    if not jp.is_file():
        raise SystemExit(f"[f0_summary] ledger named by the marker is missing: {jp} (log {log_path})")
    try:
        d = json.loads(jp.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise SystemExit(f"[f0_summary] cannot read the ledger: {jp} — {type(e).__name__}: {e}") from e
    if str(d.get("run_id")) != m["run"]:
        raise SystemExit(
            f"[f0_summary] ledger is not from this run: marker run={m['run']} ≠ file run_id={d.get('run_id')} ({jp})"
        )
    # Measured against stated counts: a ledger whose sums disagree carries no values.
    by = d.get("by_status") or {}
    if sum(by.values()) != d.get("total"):
        raise SystemExit(
            f"[f0_summary] inconsistent ledger: by_status sum {sum(by.values())} ≠ total {d.get('total')} ({jp})"
        )
    if d.get("ok", -1) + sum((d.get("fail_by_kind") or {}).values()) != d.get("total"):
        raise SystemExit(f"[f0_summary] inconsistent ledger: ok + fail_by_kind ≠ total ({jp})")
    # Marker and file must state the same counts (for consumers that only read the marker).
    for k_m, k_j in (("ok", "ok"), ("total", "total"), ("budget_skip", "budget_skip_sides"), ("rc", "rc")):
        if str(d.get(k_j)) != m[k_m]:
            raise SystemExit(f"[f0_summary] marker and ledger differ: {k_m}={m[k_m]} ≠ {k_j}={d.get(k_j)} ({jp})")
    d["_path"] = str(jp)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(
        description="f0 completion ledger reader — shell consumers get the values only through this command."
    )
    ap.add_argument("--log", required=True, help="f0 log of the run (the file with the marker)")
    ap.add_argument("--field", default="", help="print this field only (default: one TSV line)")
    a = ap.parse_args()
    d = load(a.log)
    if a.field in FAILCLASS_KEYS:
        print(failclass(a.log, d)[a.field])
        return 0
    if a.field:
        v = d.get(a.field)
        if v is None:
            print(f"[f0_summary] no such field: {a.field}", file=sys.stderr)
            return 3
        print(v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False))
        return 0
    print(f"{d['ok']}\t{d['total']}\t{d['budget_skip_sides']}\t{d['rc']}\t{d['_path']}")
    return 0


def split_failures(argv: list[str]) -> int:
    """``ledger.py split-failures <f0 shadow root> <clip list> <hand spec dir> --run-failed <file>`` — one chunk's f0
    non-judgeable clips, split into the two kinds (what stage2 reads after f0).

    The meaning of each status is owned by the prejudge (`is_judgeable` · `is_data_defect`); this applies it to the
    clips of one chunk, reading this hand's f0 ledger only (`<f0 root>/<hand spec dir>/LEDGER.tsv` — another hand's
    ledger would lend its FAIL rows to this hand). The last row of a clip is its current status (the ledger is
    append-only).

      stdout            declared data defects (DATA_DEFECT), one clip per line, in input order → <OUT>/f0_failed.txt
                        → status_ledger SOURCE_DEFECT
      --run-failed F    run failures (every other non-judgeable status): `clip <TAB> status <TAB> exception tail` —
                        always written (empty when none). stage2 fails the chunk when it is not empty.
      --held F          held clips (HOLD_OBJ — the f0 object repair refused a side): `clip <TAB> status <TAB> reason` —
                        always written. stage2 leaves them out of judging; the placement ledger records them HELD.
    A clip absent from the ledger is neither (the judge's preflight reports the missing shadow).
    """
    import os

    # not stdlib-only: this subcommand alone needs the prejudge's status rules
    from dexengine.retarget.f0.shadow_status import LEDGER_NAME, is_data_defect, is_held, is_judgeable, ledger_last

    ap = argparse.ArgumentParser(prog="ledger.py split-failures")
    ap.add_argument("f0_root")
    ap.add_argument("clips")
    ap.add_argument("hand_dir")
    ap.add_argument("--run-failed", required=True)
    ap.add_argument("--held", required=True)
    a = ap.parse_args(argv)
    with open(a.clips, encoding="utf-8") as fh:
        want = [ln.rstrip("\n") for ln in fh if ln.strip()]
    led = os.path.join(a.f0_root, a.hand_dir, LEDGER_NAME)
    if not os.path.isfile(led):
        sys.exit(
            f"[stage2] this hand's f0 ledger is missing: {led} — f0 did not run for this hand (another hand's "
            "ledger is never read)"
        )
    last = ledger_last(led, set(want))  # last row = current status (parser owner: the prejudge)
    run_failed, held = [], []
    for c in want:  # keep the input order
        st, note = last.get(c, (None, ""))
        if st is None or is_judgeable(st):
            continue
        if is_data_defect(st):
            print(c)
        elif is_held(st):
            held.append(f"{c}\t{st}\t{note[-300:]}\n")
        else:
            run_failed.append(f"{c}\t{st}\t{note[-300:]}\n")
    for path, rows in ((a.run_failed, run_failed), (a.held, held)):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("".join(rows))
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["split-failures"]:
        sys.exit(split_failures(sys.argv[2:]))
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, str):  # message to stderr, rc 3
            print(e.code, file=sys.stderr)
            sys.exit(3)
        raise
