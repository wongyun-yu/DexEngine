"""Spec mode and final tally of the record driver (record.driver); called by the driver (this module does not run on
its own).
  spec  --spec <jsonl> --spec-id <id> (one record render lane job = one spec; written by the lane specs command): the
        spec owns hand, round, clips, owner sha and tag (= spec_id) (lane pilot_spec; giving --hand, --round, --tag,
        --owner, --owner-sha, --order or --clip-subset as well is rc 12). Ledger = the frozen copy the spec points at
        (lane owner_path) · --only = the spec bundle · spec clips ⊆ the queue bundle's clips · without --gpus 0 (dies
        when CUDA_VISIBLE_DEVICES has a comma) · bundle cap = the spec budget_s (the one home of the time).
  rc    run_rc(outcome of the bundles targeted by this run; --only when given): 0 DONE · 10 PARTIAL (bundles not
        started) · 11 augment failed · 12 input missing (spec, ledger file) · 13 owner sha · 1 otherwise (a FAILED
        bundle is never 0).
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import shutil
import sys
from pathlib import Path

from dexengine.core import paths as P  # clip_name
from dexengine.core.errors import is_input_missing
from dexengine.record import hooks  # spec-mode provider (kind driver_spec)
from dexengine.record import piece as PC  # round place part tail (part_key)
from dexengine.record.driver_check import bdir_name, mem_available_gb, now, say, write_atomic

PLAIN_NEED = ("hand", "owner", "owner_sha", "tag", "order", "gpus", "round")


def _jobspec():
    """The job-spec provider of spec mode: the hooks kind ``driver_spec`` (name "lane"), registered by a
    plugin. Spec mode without it dies with rc 12 (INPUT_MISSING)."""
    hooks.load_plugins()
    sp = hooks.get("driver_spec", "lane")
    if sp is None:
        print(
            f"[COLLECT_PILOT] ✗ --spec needs a job-spec provider (hooks kind driver_spec, name lane): name its "
            f"plugin module in {hooks.PLUGINS_ENV}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(CODE["INPUT_MISSING"])
    return sp


def resolve(a) -> None:
    """Input resolution: in spec mode (--spec) pilot_spec decides from the spec (owner arguments given as well or a
    bundle mismatch = rc 12) · otherwise every old required argument must be present. Marker [COLLECT_PILOT] spec …
    with the applied values."""
    a.spec_mode = bool(a.spec or a.spec_id)
    if not a.spec_mode:
        if miss := [k for k in PLAIN_NEED if getattr(a, k) in (None, "")]:
            need = ["--" + k.replace("_", "-") for k in miss]
            print(
                f"[COLLECT_PILOT] ✗ arguments {need} missing (required outside spec mode)", file=sys.stderr, flush=True
            )
            raise SystemExit(2)  # usage error = rc 2 as argparse
        # outside a spec = one bundle = the round place as it is (no part)
        a.part = ""
        return
    if not (a.spec and a.spec_id):
        _jobspec().die("--spec and --spec-id only together", _jobspec().RC_INPUT)
    r = _jobspec().pilot_spec(a)
    a.spec_row, a.hand, a.round, a.tag = r["spec"], r["hand"], r["round"], r["tag"]
    a.owner_sha, a.owner, a.spec_clips, a.only = r["owner_sha"], r["owner"], r["clips"], r["only"]
    a.order = r["spec"]["dataset"]  # spec_queue appends the ledger's other datasets (queue rule)
    a.part = PC.part_key(r["spec"])  # sibling parts split the round place (piece.part_key)
    if not a.gpus:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if "," in cvd:
            _jobspec().die(
                f"no --gpus and CUDA_VISIBLE_DEVICES={cvd} lists several GPUs: name the job's one GPU",
                _jobspec().RC_INPUT,
            )
        a.gpus = "0"
    say(
        f"spec {a.spec_id} sha {_jobspec().CCOL.spec_sha(a.spec_row)[:12]} · hand {a.hand} · round {a.round} · bundle "
        f"{a.only} · clips {len(a.spec_clips)} · expected {len(a.spec_row['expect'])} · ledger {a.owner} · GPU "
        f"{a.gpus} · round place part {a.part or 'none (nparts 1)'} · "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '-')}"
    )


def row0_of(a, clips: list[str]) -> list[str]:
    """List handed to the job for the owner of the off-screen original decision (render.fov owner_set): clips with a
    row-0 place in the aug spec's expect ('<aug 0>', lane aug_slot(0); the handing side --row0-clips sets it from the
    orig round decision) ∩ this bundle's clips. Outside a spec, orig round or no row 0 = [] (bundle command
    byte-identical)."""
    row = getattr(a, "spec_row", None)
    if not row or row.get("round") != "aug":
        return []
    r0 = {P.clip_name(c) for c, t, *_l in row.get("expect", []) if t == _jobspec().aug_slot(0)}
    return sorted(P.clip_name(c) for c in clips if P.clip_name(c) in r0)


def spec_timeout(a) -> None:
    """Spec-mode bundle cap = the spec budget_s alone (within the job limit budget + setup the bundle ends first and
    leaves its rc)."""
    a.bundle_timeout_base = 0
    a.bundle_timeout_per_clip = math.ceil(a.spec_row["budget_s"] / len(a.spec_clips))
    say(
        f"bundle cap = spec budget_s {a.spec_row['budget_s']} s (clips {len(a.spec_clips)} x "
        f"{a.bundle_timeout_per_clip} s)"
    )


def need_file(p) -> None:
    """The owner ledger file must exist (plain and spec mode): rc 12 (INPUT_MISSING) otherwise."""
    if not Path(p).is_file():
        print(f"[COLLECT_PILOT] ✗ owner ledger {p} is missing", file=sys.stderr, flush=True)
        raise SystemExit(CODE["INPUT_MISSING"])


def owner_gate(fn, *args) -> None:
    """Owner ledger check (record.sources); dying on a mismatch is rc 13 (OWNER_SHA)."""
    try:
        fn(*args)
    except SystemExit as e:
        print(e, file=sys.stderr, flush=True)
        raise SystemExit(CODE["OWNER_SHA"]) from None


def spec_order(a) -> list[str]:
    """The queue rule requires every dataset of the ledger in the order; in spec mode the spec dataset first."""
    if a.spec_mode:
        from dexengine.rl import acceptance as CO  # ledger reader; lazy: rl loads plugin modules at import time

        a.order += "," + ",".join(sorted({r["dataset"] for r in CO.read_ledger(a.ledger)} - {a.order}))
    return [x for x in a.order.split(",") if x]


def spec_queue(a, queue: list[dict]) -> list[dict]:
    """Spec mode: the one spec bundle from the queue, only the spec clips (⊆ that bundle's clips, else rc 12)."""
    if not a.spec_mode:
        return queue
    hit = [b for b in queue if bdir_name(b) in a.only]  # a.only = {spec bundle} (a set from check_env)
    have = {P.clip_name(c) for c in hit[0]["clips"]} if len(hit) == 1 else set()
    out = sorted({P.clip_name(c) for c in a.spec_clips} - have) if len(hit) == 1 else []
    if len(hit) != 1 or out:
        _jobspec().die(
            f"spec bundle {a.only} appears {len(hit)} times in the queue · spec clips outside the queue "
            f"{out[:3] or '-'}",
            _jobspec().RC_INPUT,
        )
    return [dict(hit[0], clips=sorted(a.spec_clips))]


def outcome(run_dir: Path, queue: list[dict], only: set[str]) -> tuple[list[dict], int]:
    """Bundles targeted by this run (--only when given) -> (result.json of finished bundles, count not started); the
    first word of state decides."""
    results, left = [], 0
    for b in queue:
        if only and bdir_name(b) not in only:
            continue
        d = Path(run_dir) / "bundles" / bdir_name(b)
        st = (d / "state").read_text().split("\n", 1)[0] if (d / "state").is_file() else ""
        if st.startswith(("DONE", "FAILED")):
            rj = d / "result.json"
            results.append(json.loads(rj.read_text()) if rj.is_file() else {"state": st.split()[0], "reason": st})
        else:
            left += 1
    return results, left


#: job rc table (the same codes the job launchers report)
RC_NAMES = {0: "DONE", 10: "PARTIAL", 11: "AUG_FAILED", 12: "INPUT_MISSING", 13: "OWNER_SHA"}

CODE = {name: code for code, name in RC_NAMES.items()}
BROKEN = 1
#: input-missing failure reasons (rc 12): the root line carries core.errors.INPUT_MISSING (written by
#: InputMissingError and input_missing). Only the label of a failed bundle depends on it (INPUT_MISSING vs broken rc 1;
#: nothing branches on 12). Reasons stored before the marker existed carry no marker; this reader takes the
#: marker only.
ORDER = (CODE["OWNER_SHA"], CODE["INPUT_MISSING"], BROKEN, CODE["AUG_FAILED"], CODE["PARTIAL"], CODE["DONE"])


def bundle_rc(res: dict) -> int:
    """rc of one finished bundle (result.json): DONE = 0 (reasoned exclusions are expected) · failures by the rule
    above."""
    from dexengine.record import bundle as C

    if res.get("state") == "DONE":
        return CODE["DONE"]
    root = (res.get("reason") or "").split(" ⇐ ")[-1]
    if is_input_missing(root):
        return CODE["INPUT_MISSING"]
    if res.get("rc") == C.RC_AUG_FAILED or res.get("aug_failed"):
        return CODE["AUG_FAILED"]
    return BROKEN


def run_rc(results: list[dict], left: int) -> int:
    """One driver run: results = result.json of the finished bundles this run targeted · left = bundles not
    started."""
    codes = {bundle_rc(r) for r in results}
    if left > 0:
        codes.add(CODE["PARTIAL"])
    return next((c for c in ORDER if c in codes), CODE["DONE"])


def name(rc: int) -> str:
    return RC_NAMES.get(rc, "BROKEN" if rc == BROKEN else f"rc {rc}")


def write(S, final: str = "") -> None:
    """progress.json of the operations place (60 s and at the end): bundles, clips, sizes, times, failures and workers
    per dataset."""
    a = S.a
    per = {ds: collections.Counter() for ds in a.order}
    fails = []
    for b in S.queue:
        c = per[b["dataset"]]
        c["bundles"] += 1
        c["target_clips"] += len(b["clips"])
        st = S.state(b)
        rj = S.run / "bundles" / bdir_name(b) / "result.json"
        r = json.loads(rj.read_text()) if rj.is_file() and not st.startswith("RUNNING") else {}
        if st.startswith("DONE"):
            c["done_bundles"] += 1
            c["merged_bundles"] += (r.get("merge") or {}).get("mode") == "on"
            c["done_clips"] += r.get("target", 0)
            for k in ("episodes", "frames", "video_frames", "bytes_dataset", "bytes_total"):
                c[k] += r.get(k, 0)
            c["excluded"] += len(r.get("excluded", {}))
            c["sec"] += r.get("sec", 0)
        elif st.startswith("FAILED"):
            c["failed_bundles"] += 1
            c["failed_clips"] += len(b["clips"])
            c["sec_failed"] += r.get("sec", 0)
            fails.append({"bundle": bdir_name(b), "dataset": b["dataset"], "reason": r.get("reason", st)[:300]})
        elif st.startswith("RUNNING"):
            c["running_bundles"] += 1
    out = {}
    for ds, c in per.items():
        row = dict(c)
        if c["done_bundles"]:
            row["sec_per_bundle"] = round(c["sec"] / c["done_bundles"], 1)
            row["sec_per_clip"] = round(c["sec"] / max(1, c["done_clips"]), 1)
            row["gb_dataset"] = round(c["bytes_dataset"] / 1e9, 3)
            row["gb_total"] = round(c["bytes_total"] / 1e9, 3)
        out[ds] = row
    tot = collections.Counter()
    for c in per.values():
        tot.update(c)
    T = dict(tot)
    if tot["done_bundles"]:
        T["sec_per_clip"] = round(tot["sec"] / max(1, tot["done_clips"]), 1)
        T["gb_total"] = round(tot["bytes_total"] / 1e9, 3)
    T["pending_bundles"] = len(S.queue) - tot["done_bundles"] - tot["failed_bundles"] - tot["running_bundles"]
    doc = {
        "hand": a.hand,
        "combo": a.combo,
        "tag": a.tag,
        "started": a.started,
        "updated": now(),
        "deadline": dt.datetime.fromtimestamp(a.deadline_ts).astimezone().isoformat(timespec="seconds"),
        "driver": {"pid": S.me, "state": final or ("stopping: " + S.stop_reason if S.stop_reason else "running")},
        "tree": str(a.tree),
        "commit": S.tree0.split("|")[0],
        "ledger": str(a.ledger),
        "ledger_sha256": a.ledger_sha,
        "data_root": str(a.main),
        "data_dirs": {ds: str(P.layer_path("collect", ds, combo=a.combo)) for ds in a.order},
        "merge": a.merge,
        "round": a.round,
        "work_glob": P.run_glob("run_collect_work", a.hand, root=a.main),
        "order": a.order,
        "min_rate": a.min_rate,
        "gpus": a.gpus,
        "exclusive_gpus": sorted(a.exclusive),
        "workers": S.workers,
        "datasets": out,
        "totals": T,
        "failed": fails,
        "mem_available_gb": round(mem_available_gb(), 1),
        "disk_free_gb": round(shutil.disk_usage(a.main).free / 1e9, 1),
    }
    write_atomic(S.run / "progress.json", json.dumps(doc, indent=1, ensure_ascii=False))
