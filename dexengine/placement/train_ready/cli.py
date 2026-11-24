"""Command line of the trainability survey: arguments, the T5 cache location, stdout discipline for the JSON modes,
the bake-input id files, the re-judge list and the human table."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from dexengine.core import paths as P
from dexengine.data import rules as RR
from dexengine.placement.train_ready import assets as A
from dexengine.placement.train_ready import judge_fp as JF
from dexengine.placement.train_ready import survey as SV
from dexengine.placement.train_ready import trees as TR


def write_stale(path: str) -> None:
    """`<ds>\\t<ledger axis>\\t<clip>\\t<reason>` — reason = T4 · T4-b · T5-base[:unknown] · T5:<diff> ·
    T5-judge:<parts>|unknown · T5-desk:<fields> · static_desk.STALE_REASON:<object>:<overhang %>.
    remaining_to_judge --t4 reads columns 0 and 2 only."""
    pathlib.Path(path).write_text("".join(f"{d}\t{e}\t{c}\t{r}\n" for d, e, c, r in SV.STALE))


def _stdout_to_stderr() -> int:
    """Point fd 1 at fd 2 and return a copy of the old fd 1 (child processes too — redirect_stdout only moves print)."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    return saved


def _stdout_restore(saved: int) -> None:
    sys.stdout.flush()
    os.dup2(saved, 1)
    os.close(saved)


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trainability survey (T1 USD · T2 promotion)")
    ap.add_argument("--ids-dir", default=None, help="write the missing obj_ids to <dir>/<ds>.txt (USD) · <ds>.mesh.txt")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument(
        "--json-clips",
        action="store_true",
        help="JSON {rows, clips} with per-clip verdicts; --json is unchanged",
    )
    ap.add_argument(
        "--stale-out",
        default=None,
        help="write T4 (old roles) · T5 (old setup) · T5-judge (judge input differs / absent) clips as "
        "`<ds>\\t<ledger axis>\\t<clip>\\t<reason>` — the re-judge queue's input",
    )
    ap.add_argument(
        "--t5-workers",
        type=int,
        default=None,
        help="processes that precompute T5 cache misses (default: capacity.plan · 1 = in the loop, same result); "
        "the full recompute after a "
        "rule change is bound by articulated D20 trees (~2 s each)",
    )
    ap.add_argument(
        "--t5-cache",
        default="",
        help="T5 input-fingerprint cache file (empty = hand root logs/train_ready/t5_setup_cache.json · 'off' = none); "
        "results do not depend on it (the key is the rule code bytes + every input's mtime and size)",
    )
    ap.add_argument(
        "--datasets",
        default="",
        help="comma list — measure only these datasets' ledgers (a narrowed campaign passes its scope; empty = every "
        "ledger of the hand); unknown names die (RR.datasets_scope). Marker [TRAIN_READY] scope=… (stderr)",
    )
    return ap.parse_args()


def _write_ids(ids_dir: str, rows: list[dict]) -> None:
    os.makedirs(ids_dir, exist_ok=True)
    for r in rows:
        for suffix, kinds in ((".txt", {"usd"}), (".mesh.txt", {"surface", "keypoints"})):
            ids = [k for k, v in r["missing"].items() if kinds & set(v)]
            if ids:
                with open(os.path.join(ids_dir, r["dataset"] + suffix), "w") as f:
                    f.write("\n".join(ids) + "\n")


def _stale_summary(stale_out: str) -> None:
    from dexengine.placement.scene import static_desk as SD  # noqa: PLC0415 — static-desk reason head owner

    st = SV.STALE
    n5 = sum(1 for s in st if s[3].startswith("T5:"))
    nj = sum(1 for s in st if s[3].startswith("T5-judge:"))
    nu = sum(1 for s in st if s[3] == JF.JUDGE_UNKNOWN)
    ns = sum(1 for s in st if s[3].startswith(SD.STALE_REASON + ":"))
    nd = sum(1 for s in st if s[3].startswith("T5-desk:"))
    print(
        f"[train_ready] to re-judge {len(st)} (T4 old roles {len(st) - n5 - nj - ns - nd} · T5 old setup {n5}"
        f" · T5-judge judge input differs {nj - nu} · no fingerprint (unknown) {nu} · static object on the desk {ns}"
        f" · T5-desk desk fields {nd})"
        f" → {stale_out}",
        file=sys.stderr,
    )


def _verdict(r: dict) -> str:
    envk = sorted({k for v in r["missing"].values() for k in v if k.startswith("env_")})
    if envk:
        return f"before env wiring ({','.join(envk)}) — left out of the training total" + (
            "" if r.get("variant") else " (articulated clips inside the canonical axis)"
        )
    if r["ready"]:
        return "trainable"
    if r["promotable"]:
        return "T2 promotion only (to status=canon)"
    if r["missing"]:
        ex = [f"{k}({','.join(v)})" for k, v in list(r["missing"].items())[:2]]
        return f"T1 assets missing: {', '.join(ex)}{' and more' if len(r['missing']) > 2 else ''}"
    if r.get("stale_roles"):
        return f"T4 old-role trees {r['stale_roles']} — re-judge and rebake (assets complete)"
    if r.get("stale_setup"):
        return f"T5 old-setup trees {r['stale_setup']} — re-judge and rebake"
    if r.get("setup_unevaluable"):
        return f"T5 setup not evaluable {r['setup_unevaluable']} — the base statement must be fixed"
    return "no tree on the ledger axis (another layout)"


def _table(rows: list[dict]) -> None:
    """The human table and the total line."""
    print(f"{'dataset':12}{'status':7}{'adopt':>7}{'assets':>8}{'no_tree':>9}{'missing':>9}  verdict")
    for r in rows:
        verdict = _verdict(r)
        print(
            f"{r['dataset']:12}{r['status']:7}{r['adopted']:>7}{r['covered']:>8}"
            f"{r['no_tree']:>9}{len(r['missing']):>9}  {verdict}"
            + (f" · excluded by the current first pass {r['stale_excluded']}" if r.get("stale_excluded") else "")
            + (
                f" · T4 old-role trees {r['stale_roles']}"
                if r.get("stale_roles") and not verdict.startswith("T4")
                else ""
            )
            + (
                f" · T5 old-setup trees {r['stale_setup']}"
                if r.get("stale_setup") and not verdict.startswith("T5 old")
                else ""
            )
            + (
                f" · T5 not evaluable {r['setup_unevaluable']}"
                if r.get("setup_unevaluable") and not verdict.startswith("T5 setup")
                else ""
            )
            + (f" · T5-judge judge input differs {r['stale_judge']}" if r.get("stale_judge") else "")
            + (f" · T5-judge no fingerprint (unknown) {r['judge_unknown']}" if r.get("judge_unknown") else "")
            + (f" · T5-judge not evaluable {r['judge_unevaluable']}" if r.get("judge_unevaluable") else "")
        )
    # The trainable total is per clip and per tree: a covered clip is trainable regardless of the other clips of its
    # dataset. A variant axis counts once its env gate is wired (RR.AERIAL_ENV_WIRED / ARTIC_ENV_WIRED).
    tot_a = sum(r["adopted"] for r in rows)
    tot_c = sum(r["covered"] for r in rows)
    tot_r = sum(r["covered"] for r in rows if r["status"] == "canon")
    tot_v = sum(r["adopted"] - r["covered"] for r in rows if r.get("variant"))
    tot_s = sum(r.get("stale_roles", 0) for r in rows)
    tot_5 = sum(r.get("stale_setup", 0) for r in rows)
    tot_e = sum(r.get("setup_unevaluable", 0) for r in rows)
    tot_j = sum(r.get("stale_judge", 0) + r.get("judge_unknown", 0) for r in rows)
    print(
        f"\ntotal adopted {tot_a} · assets complete {tot_c} · **trainable now (per clip, env gate included)** {tot_r}"
        f" · variant axes incomplete {tot_v}"
        f" · env gate AERIAL={int(RR.AERIAL_ENV_WIRED)} ARTIC={int(RR.ARTIC_ENV_WIRED)}"
        + (f" · T4 old-role trees (back after a re-judge) {tot_s}" if tot_s else "")
        + (f" · T5 old-setup trees (back after a re-judge) {tot_5}" if tot_5 else "")
        + (f" · T5 setup not evaluable {tot_e}" if tot_e else "")
        + (f" · T5-judge re-judge (input differs · unknown) {tot_j}" if tot_j else "")
    )


def main() -> int:
    a = _args()
    scope = RR.scope_arg(a.datasets)  # the received names are checked by their owner (RR.scope_arg)
    if a.t5_cache != "off":
        if a.t5_cache:
            TR.T5_CACHE["path"] = a.t5_cache
        else:
            TR.T5_CACHE["path"] = str(
                P.run_path("run_log", RR.active_hand(), root=A.REPO, tool="train_ready") / "t5_setup_cache.json"
            )
            TR.T5_CACHE["guard"] = True  # inside the hand root — owner marker checked before writing
    SV.T5_WORKERS = None if a.t5_workers is None else max(1, int(a.t5_workers))
    t0 = time.time()
    # In --json · --json-clips mode stdout is one JSON document only: output during the computation (child processes,
    # other modules' markers such as coords' [DESK_Z]) goes to stderr at the fd level, or the consumer's json.loads
    # breaks.
    saved = _stdout_to_stderr() if (a.json or a.json_clips) else None
    try:
        rows = SV.survey(P.data_layout, scope)
        TR.cache_save()
        TR.t5_marker(t0)
        JF.t5j_marker(t0)
        JF.t5b_marker()
        JF.t5d_marker()
    finally:
        if saved is not None:
            _stdout_restore(saved)
    if a.ids_dir:
        _write_ids(a.ids_dir, rows)
    if a.json_clips:
        print(json.dumps({"rows": rows, "clips": SV.CLIPS}, ensure_ascii=False, indent=1))
        return 0
    if a.json:
        # Machine-readable: stdout carries the JSON only — returning here keeps the human table from being appended.
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        if a.stale_out:
            write_stale(a.stale_out)
        return 0
    if a.stale_out:
        write_stale(a.stale_out)
        _stale_summary(a.stale_out)
        return 0
    _table(rows)
    return 0
