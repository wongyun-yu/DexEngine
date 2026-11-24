#!/usr/bin/env python3
"""One E5 kinematic piece — count check of a finished render job (<work>/rounds/<round>/dataset) → final merge → one
ledger row; and `ready`, the completeness proof of the final root.
- piece = a few clips of one dataset (rows tsv: ds emb clip variants). Stored once at the end of the job (collect.py) —
  the smaller the piece, the less is lost.
- check (sets): job rc 0 · info.total_episodes = clips × the round's look count (orig = D 1 + RANDOM_PER_TRAJ) · the
clip
  set of episodes.jsonl = the piece's clip set (exactly that count per clip) · parquet count · two-eye mp4 count =
  episode count. Any mismatch → FAIL, no merge.
- merge = canonical store merge append (lock piece.final_lock, idempotent piece-id); final location =
  <final root>/data/<ds>/collect/rby1_wuji2 (the converter's input contract).
- ledger = <ledger> (tsv, append only): time piece lane state episodes retries note. Retries = the [COLLECT_RETRY] lines
  of driver.log (counted even when the automatic retry recovered).
  python -m experiments.policies.kin_data.piece merge --rows <tsv> --work <job dir> --round orig --final-root <exo-on
      root> --exo0-root <exo-off root> --piece-id <id> --lane <name> --job-rc <rc> --ledger <tsv>
  python -m experiments.policies.kin_data.piece ready --final-root <root> --clips <tsv (dataset clip …)> --per-clip 5
      [--subset <tsv> …] [--ledger <q1 ledger>]
`ready`: every dataset folder passes the store merge verify, the episode clip set = the listed clip set (nothing
missing, nothing extra) and every clip has the expected count → only then <final root>/KIN_READY.json is written (any
mismatch: not written, rc 1). Each --subset list (e.g. a pairing subset) is proven the same way; '#' lines are headers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from dexengine.record import episodes as EPI
from experiments.textio import read_lines

MARK = "[KIN_PIECE]"
READY_MARK = "[KIN_READY]"
ROOT = Path(__file__).resolve().parents[3]


def _pythonpath() -> str:
    """This tree first, then the caller's PYTHONPATH (it may carry the store's pyarrow)."""
    return os.pathsep.join([str(ROOT), *filter(None, [os.environ.get("PYTHONPATH", "")])])


COMBO = "rby1_wuji2"
CAMS = (
    "observation.images.ego_left",
    "observation.images.ego_right",
)  # eyes that must exist (converter input) — other video keys are stated by info
TREE_SFX = "_pl_rt"


def looks_per_traj(rnd: str) -> int:
    """Looks per trajectory — orig = D 1 + random (only D with DEXENGINE_COLLECT_LOOKS=D) · aug = D 1 + random (the
    canonical aug has no D-only branch).
    """
    if rnd not in ("orig", "aug"):
        raise SystemExit(f"{MARK} round {rnd!r} — orig or aug only")
    return 1 if (rnd == "orig" and EPI.looks_mode() == EPI.LOOK_D) else 1 + EPI.RANDOM_PER_TRAJ


VAR_ORIG_ENV = "KIN_VAR_ORIG_D"  # owner = this file (read by collect.py) · default empty = unchanged
VAR_ORIG_MARK = "[KIN_VAR_ORIG_D]"


def var_orig_d() -> bool:
    """Dense L1 kinematic pairs (default off): 1 = record the row's source + variants in one record session and draw one
    look-D episode **per variant** in the orig round (the source is recorded, not drawn — as dense_render.py: one
    variant = one orig-round stage2_clip). The stored clips are the drawn variants (merge keys per variant, so variants
    of one source split over pieces never overlap). On, it accepts only round orig with DEXENGINE_COLLECT_LOOKS=D.
    Unknown values
    die."""
    v = os.environ.get(VAR_ORIG_ENV, "")
    if v not in ("", "1"):
        raise SystemExit(f"{MARK} {VAR_ORIG_ENV}={v!r} — empty or 1 only")
    return v == "1"


def want_of(rows: list[dict], rnd: str) -> Counter:
    """The piece's expected set — orig = per source tree · aug = per row variant (comma list of variant clip names),
    the
    round's look count each. With KIN_VAR_ORIG_D=1 orig is per variant too (the source is not drawn)."""
    if rnd == "aug" or (rnd == "orig" and var_orig_d()):
        return Counter(
            {
                (v if v.endswith(TREE_SFX) else v + TREE_SFX): looks_per_traj(rnd)
                for r in rows
                for v in (r.get("variants") or "").split(",")
                if v
            }
        )
    return Counter({r["clip"] + TREE_SFX: looks_per_traj(rnd) for r in rows})


def exo_env(ds_dir: Path) -> str:
    """The exo value the piece's (or final's) video keys state — the merge runs with the render's env (store VIDEO_KEYS
    reads it).
    """
    vk = (json.loads((ds_dir / "meta/info.json").read_text()).get("collect") or {}).get("video_keys") or []
    return "1" if "observation.images.exo" in vk else "0"


def ledger(path: Path, piece: str, lane: str, state: str, eps: int, retries: int, note: str) -> None:
    with open(path, "a") as f:
        f.write(f"{time.strftime('%FT%T')}\t{piece}\t{lane}\t{state}\t{eps}\t{retries}\t{note}\n")


def check(rows: list[dict], ds_dir: Path, rnd: str) -> int:
    want = want_of(rows, rnd)
    info = json.loads((ds_dir / "meta/info.json").read_text())
    n = int(info["total_episodes"])
    if n != sum(want.values()):
        raise ValueError(f"total_episodes {n} ≠ expected {sum(want.values())}")
    got = Counter(
        json.loads(line)["clip"] for line in (ds_dir / "meta/episodes.jsonl").read_text().splitlines() if line.strip()
    )
    if got != want:
        raise ValueError(
            f"clip set differs — missing {sorted(set(want) - set(got))[:3]} · extra "
            f"{sorted(set(got) - set(want))[:3]} · "
            f"count differs {[c for c in want if got.get(c) not in (None, want[c])][:3]}"
        )
    npq = len(list((ds_dir / "data").glob("chunk-*/episode_*.parquet")))
    vk = list((info.get("collect") or {}).get("video_keys") or [])
    if not set(CAMS) <= set(vk):
        raise ValueError(f"video_keys {vk} lack {CAMS}")
    nmp = {c: len(list((ds_dir / "videos").glob(f"chunk-*/{c}/episode_*.mp4"))) for c in vk}
    if npq != n or any(v != n for v in nmp.values()):
        raise ValueError(f"parquet {npq} · mp4 {nmp} ≠ episodes {n}")
    return n


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    for k in ("rows", "work", "final-root", "exo0-root", "ledger"):
        m.add_argument(
            f"--{k}", type=Path, required=True
        )  # exo0-root = the final root next to it for exo-off pieces (not mixed)
    for k in ("round", "piece-id", "lane"):
        m.add_argument(f"--{k}", required=True)
    m.add_argument("--job-rc", type=int, required=True)
    r = sub.add_parser("ready")
    r.add_argument("--final-root", type=Path, required=True)
    r.add_argument("--clips", type=Path, required=True)
    r.add_argument("--per-clip", type=int, required=True)
    r.add_argument("--ledger", type=Path)
    r.add_argument("--subset", type=Path, action="append", default=[])
    return ap.parse_args()


def _run_append(cmd: list[str], ds_dir: Path):
    """Run the canonical store merge with the piece's exo value (another value in the final dies in the canonical
    check); returns the process and its last output line."""
    p = subprocess.run(
        [sys.executable, "-m", "dexengine.record.store.merge", *cmd],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": _pythonpath(), "DEXENGINE_COLLECT_EXO": exo_env(ds_dir)},
    )
    tail = (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]
    print(f"{MARK} dexengine.record.store.merge {cmd[0]} rc {p.returncode} · {tail[0][:200]}", flush=True)
    return p, tail


def merge(a) -> int:
    """`merge`: check one piece (one dataset) and append it to the final root; one ledger row either way."""
    with open(a.rows) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    dss = {r["ds"] for r in rows}
    drv = a.work / "driver.log"
    retries = sum(1 for line in read_lines(drv, errors="replace") if "[COLLECT_RETRY]" in line) if drv.is_file() else -1
    if len(dss) != 1:
        ledger(a.ledger, a.piece_id, a.lane, "FAIL", 0, retries, f"datasets {sorted(dss)} — a piece holds one dataset")
        return 2
    ds = dss.pop()
    if a.job_rc != 0:
        ledger(a.ledger, a.piece_id, a.lane, "FAIL", 0, retries, f"job rc {a.job_rc} · {a.work}")
        print(f"{MARK} FAIL {a.piece_id} job rc {a.job_rc}", flush=True)
        return 1
    ds_dir = a.work / "rounds" / a.round / "dataset"
    try:
        n = check(rows, ds_dir, a.round)
    except (ValueError, OSError, KeyError) as e:
        ledger(a.ledger, a.piece_id, a.lane, "FAIL", 0, retries, f"check {e}")
        print(f"{MARK} FAIL {a.piece_id} check {e}", flush=True)
        return 1
    ex = exo_env(ds_dir)
    final = (a.final_root if ex == "1" else a.exo0_root) / "data" / ds / "collect" / COMBO
    # merge = one append (input check and journal post-check are append's). The full final verify runs once in `ready`
    # — per piece it would grow with the final's size and stall the lane meanwhile.
    for cmd in (["append", "--final", str(final), "--piece", str(ds_dir), "--piece-id", a.piece_id],):
        p, tail = _run_append(cmd, ds_dir)
        if p.returncode != 0:
            ledger(
                a.ledger, a.piece_id, a.lane, "FAIL", n, retries, f"merge {cmd[0]} rc {p.returncode} · {tail[0][:160]}"
            )
            return 1
    ledger(
        a.ledger,
        a.piece_id,
        a.lane,
        "DONE" if ex == "1" else "DONE_EXO0",
        n,
        retries,
        f"{ds} clips {len(rows)} → {final}",
    )
    print(f"{MARK} DONE {a.piece_id} {ds} clips {len(rows)} · episodes {n} · retries {retries} → {final}", flush=True)
    return 0


def main() -> int:
    a = _parse_args()
    if a.cmd == "ready":
        return ready(a)
    return merge(a)


def _clip_keys(path: Path) -> set:
    """(dataset, clip) keys of a clip list (tsv; '#' lines are comments)."""
    with open(path) as fh:
        rows = csv.DictReader([x for x in fh if not x.startswith("#")], delimiter="\t")
    return {(r["dataset"], r["clip"]) for r in rows}


def _verify_all(dirs: list[Path]) -> dict:
    """Run the canonical final verify on every dataset folder in parallel; returns {dir: (video keys, process)}."""
    # final verify = the canonical merge_check.verify_final (under the lock) without the frame check (frames=False):
    # each append's post-check already counted the frames of the files it placed in the final; decoding everything again
    # takes far longer.
    code = (
        "import json,sys; from pathlib import Path; from dexengine.record import piece as CP; "
        "from dexengine.record.store import merge_check as MC\n"
        "d=Path(sys.argv[1])\nwith CP.final_lock(d):\n    r=MC.verify_final(d, frames=False)\n"
        "print(json.dumps(r, ensure_ascii=False)); sys.exit(0 if r['ok'] else 1)"
    )

    def verify(d: Path):
        vk = (json.loads((d / "meta/info.json").read_text()).get("collect") or {}).get("video_keys") or []
        return vk, subprocess.run(
            [sys.executable, "-c", code, str(d)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env={
                **os.environ,
                "PYTHONPATH": _pythonpath(),
                "DEXENGINE_COLLECT_EXO": "1" if "observation.images.exo" in vk else "0",
            },
        )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(
        max_workers=len(dirs) or 1
    ) as ex:  # one per dataset folder (different locks) — run together
        res = dict(zip(dirs, ex.map(verify, dirs), strict=True))
    return res


def _scan_datasets(dirs: list[Path], res: dict) -> tuple[dict, Counter, list[str]]:
    """Per-dataset summary, episode count per (dataset, clip), and the verify/superseded problems."""
    per_ds, got, bad = {}, Counter(), []
    for d in dirs:
        ds = d.parts[-3]
        vk, p = res[d]
        eps = [json.loads(x) for x in (d / "meta/episodes.jsonl").read_text().splitlines() if x.strip()]
        sup = d / "meta/superseded.jsonl"
        n_sup = len([x for x in sup.read_text().splitlines() if x.strip()]) if sup.is_file() else 0
        c = Counter((ds, e["clip"].removesuffix(TREE_SFX)) for e in eps)
        got.update(c)
        info = json.loads((d / "meta/info.json").read_text())
        per_ds[ds] = {
            "exo": "observation.images.exo" in vk,
            "clips": len(c),
            "episodes": len(eps),
            "total_episodes": int(info["total_episodes"]),
            "superseded": n_sup,
            "verify_rc": p.returncode,
            "verify_tail": ((p.stdout + p.stderr).strip().splitlines() or [""])[-1][:200],
        }
        if p.returncode != 0:
            bad.append(f"{ds} verify rc {p.returncode}")
        if n_sup:
            bad.append(f"{ds} superseded {n_sup}")
    return per_ds, got, bad


def _subset_checks(a, want: set, got: Counter, bad: list[str]) -> tuple[dict, list[str]]:
    """Coverage of each subset list against the final root; problems appended to bad."""
    subs = {}
    for s in a.subset:
        sw = _clip_keys(s)
        sm, so = sorted(sw - set(got)), sorted(k for k in sw if k in got and got[k] != a.per_clip)
        subs[s.name] = {
            "path": str(s),
            "sha256": hashlib.sha256(s.read_bytes()).hexdigest(),
            "clips": len(sw),
            "missing": len(sm),
            "per_clip_off": len(so),
            "episodes": sum(got.get(k, 0) for k in sw),
            "outside_list": len(sw - want),
        }
        bad += [f"subset {s.name} missing {len(sm)} {sm[:3]}"] * bool(sm) + [
            f"subset {s.name} per clip ≠ {a.per_clip}: {so[:3]}"
        ] * bool(so)
        bad += [f"subset {s.name} outside the list {len(sw - want)}"] * bool(sw - want)
    return subs, bad


def _write_ready(a, doc: dict, bad: list[str], got: Counter, want: set, per_ds: dict, subs: dict) -> int:
    """Write KIN_READY.json when there is no problem; print the verdict."""
    if bad:
        print(f"{READY_MARK} FAIL — KIN_READY not written: {bad[:5]}", flush=True)
        print(json.dumps(doc, ensure_ascii=False)[:600])
        return 1
    tmp = a.final_root / "KIN_READY.json.tmp"
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    tmp.rename(a.final_root / "KIN_READY.json")
    print(
        f"{READY_MARK} OK clips {len(got)}/{len(want)} · episodes {sum(got.values())} · datasets "
        f"{len(per_ds)} · subsets "
        + " · ".join(f"{k} missing 0 ({v['clips']} clips · episodes {v['episodes']})" for k, v in subs.items())
        + " → "
        f"{a.final_root / 'KIN_READY.json'}",
        flush=True,
    )
    return 0


def ready(a) -> int:
    """`ready`: prove the final kinematic root complete and write KIN_READY.json (module docstring)."""

    want = _clip_keys(a.clips)
    dirs = sorted((a.final_root / "data").glob(f"*/collect/{COMBO}"))
    res = _verify_all(dirs)
    per_ds, got, bad = _scan_datasets(dirs, res)
    miss, extra = sorted(want - set(got)), sorted(set(got) - want)
    off = sorted(k for k, v in got.items() if v != a.per_clip)
    bad += [f"missing {len(miss)} {miss[:3]}"] * bool(miss) + [f"extra {len(extra)} {extra[:3]}"] * bool(extra)
    bad += [f"per clip ≠ {a.per_clip}: {len(off)} {off[:3]}"] * bool(off)
    subs, bad = _subset_checks(a, want, got, bad)
    doc = {
        "kind": "e5_kin_orig",
        "subsets": subs,
        "written": time.strftime("%FT%T%z"),
        "final_root": str(a.final_root),
        "combo": COMBO,
        "clips_list": str(a.clips),
        "clips_list_sha256": hashlib.sha256(a.clips.read_bytes()).hexdigest(),
        "want_clips": len(want),
        "got_clips": len(got),
        "episodes": sum(got.values()),
        "per_clip": a.per_clip,
        "missing": len(miss),
        "extra": len(extra),
        "per_clip_off": len(off),
        "datasets": per_ds,
        "ledger": str(a.ledger) if a.ledger else None,
        "verify": "collect_merge_check.verify_final(frames=False) · frames = post-check of the piece append",
        "round": "orig",
        "cameras": ["ego_left", "ego_right"],
        "exo": sorted({v.get("exo") for v in per_ds.values()}),
    }
    return _write_ready(a, doc, bad, got, want, per_ds, subs)


if __name__ == "__main__":
    sys.exit(main())
