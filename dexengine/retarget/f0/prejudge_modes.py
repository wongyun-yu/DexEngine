"""The two modes of the f0 shadow staging (`prejudge.main` resolves the inputs and calls one of them).

- `sweep_fingerprints` (--check-stale / --stamp-fingerprint): compare or stamp the cache fingerprints; f0 is not run.
- `run_f0`: run f0 on the GPU engine for every clip, append the per-clip ledger, print the completion lines and write
  the machine-readable completion ledger.
"""

from __future__ import annotations

import json
import time

from dexengine.core.repo import REPO_ROOT as _REPO
from dexengine.placement.judge.f0_key import MARKER
from dexengine.retarget.f0 import ledger as f0_summary
from dexengine.retarget.f0.shadow_gpu import run_gpu
from dexengine.retarget.f0.shadow_key import FP_DESIGN, FP_DESIGN_V1, WHY_NO_FINGERPRINT, cache_verdict
from dexengine.retarget.f0.shadow_status import (
    DATA_DEFECT,
    HELD,
    LEDGER_NAME,
    OK_STATUSES,
    SourceDefectError,
    failclass_marker,
    is_ok_status,
)
from dexengine.retarget.f0.shadow_tree import shadow_is_current


def sweep_fingerprints(a, clips, stage_src, wrist_dirs, dst_tree, root, run_id, run_started, sp, scene) -> int:
    """Fingerprint check / backfill: returns the rc of the sweep completion ledger (1 when any clip is stale)."""
    fresh = stale_n = stamped = unverif = nomark = 0
    # This path also writes the machine-readable ledger. The contract with consumers is "rc=0 means a ledger
    # exists", and it must hold at every exit of the function: one exit slipping out quietly would stop the caller
    # with "the producer did not prove completion" on the day that exit is taken (`data.stages --emit-run 7` does
    # not pass this flag, but "it does not happen today" is not the contract).
    sweep_rows = []
    for c in clips:
        mk = dst_tree / c / MARKER
        # Same decision function as prep (cache_verdict: v2 content key, strict v1 migration); f0 does not run here.
        # A key that cannot be measured is a run failure, not 'stale' (FAIL_EXC; only declared defects are
        # DATA_DEFECT), as in prep.
        try:
            verdict, why, key_now, j = cache_verdict(c, stage_src, wrist_dirs, dst_tree, a.dataset, a.src_layer)
        except Exception as e:  # noqa: BLE001
            st = DATA_DEFECT if isinstance(e, SourceDefectError) else "FAIL_EXC"
            print(f"  {st}(key computation {type(e).__name__}: {str(e)[:80]}) {c}", flush=True)
            sweep_rows.append({"clip": c, "status": st})
            continue
        if verdict == "NOMARK":
            nomark += 1
            sweep_rows.append({"clip": c, "status": "NOMARK"})
            continue
        if verdict == "FRESH":
            fresh += 1
            sweep_rows.append({"clip": c, "status": "FRESH"})
            continue
        if verdict == "STALE" and why == WHY_NO_FINGERPRINT:
            # marker predating the convention
            why2 = shadow_is_current(c, stage_src, wrist_dirs, dst_tree, a.src_layer)
            if why2:
                print(f"  STALE(structure check failed: {why2}) {c}", flush=True)
                stale_n += 1
                sweep_rows.append({"clip": c, "status": "STALE_STRUCT"})
                continue
            verdict, why = "MIGRATE", "pre-convention marker · structure check passed"
        if verdict == "STALE":
            print(f"  STALE({why}) {c}", flush=True)
            stale_n += 1
            sweep_rows.append({"clip": c, "status": "STALE"})
        elif a.stamp_fingerprint:  # MIGRATE -> write the v2 key (shadow output unchanged)
            j = dict(j, input_key=key_now, fp_design=FP_DESIGN, migrated_from=FP_DESIGN_V1, migrate_why=why)
            mk.write_text(json.dumps(j, ensure_ascii=False, indent=1))
            stamped += 1
            sweep_rows.append({"clip": c, "status": "STAMPED"})
        else:
            unverif += 1
            sweep_rows.append({"clip": c, "status": "UNVERIFIED"})
    print(
        f"[f0_prejudge] fingerprint check ({FP_DESIGN}) {len(clips)} clips — fresh {fresh} · "
        f"**stale {stale_n}** · "
        f"v2 key stamped {stamped} · migratable (unrecorded) {unverif} · no marker {nomark} · "
        f"key failures {sum(1 for r in sweep_rows if r['status'] in ('FAIL_EXC', DATA_DEFECT))}",
        flush=True,
    )
    # success set = everything not stale -> rc = `1 if stale_n else 0`
    sweep_sum = f0_summary.build(
        run_id=run_id,
        rows=sweep_rows,
        ok_statuses=("FRESH", "STAMPED", "UNVERIFIED", "NOMARK"),
        budget_skip_sides=0,
        started=run_started,
        finished=time.time(),
        tree=_REPO,
        mode="sweep",
        hand=a.hand,
        hand_spec=sp,
        dataset=a.dataset,
        emb=a.emb,
        src_layer=a.src_layer,
        shadow_root=str(root),
        scene=scene,
        prep_threads=a.prep_threads,
    )
    print(f0_summary.marker_line(sweep_sum, f0_summary.write(root, sweep_sum)), flush=True)
    return sweep_sum["rc"]


def _solve_and_ledger(a, clips, dst_tree, dst_wrist_dirs, root, stage_src, wrist_dirs):
    """Run f0 on every clip through the GPU engine, appending one ledger row per clip as it completes."""
    rows = []
    led = root / LEDGER_NAME
    if not led.exists():
        # bskip and evals come before note: note must be the last column so free text never shifts columns. The reader
        # (stage2_ground.sh) uses only p[0] and p[1], so adding columns is safe.
        led.write_text("clip\tstatus\tP\tsec\tstale\tbskip\tevals\tnote\n")

    # One clip's exception must not kill the whole run: prep exceptions go through `_guard_prep`; worker exceptions and
    # deaths become per-clip FAIL_EXC / FAIL_GPU_WORKER rows from run_gpu. The ledger is appended per clip immediately.
    for r in run_gpu(clips, a, stage_src, wrist_dirs, dst_tree, dst_wrist_dirs):
        rows.append(r)
        print(
            f"  {r['status']:<13} P={r['P']:<3} {r['t']:6.1f}s  {r['clip']}"
            + (f"  [{r['log']}]" if not is_ok_status(r["status"]) else ""),
            flush=True,
        )
        # **Append immediately, per clip** — even if the run dies, the record up to here survives.
        with open(led, "a") as fh:
            fh.write(
                f"{r['clip']}\t{r['status']}\t{r['P']}\t{r['t']:.1f}\t"
                f"{r.get('stale', '')}\t{r.get('bskip', '-')}\t"
                f"{r.get('evals', '')}\t{r['log']}\n"
            )
    return led, rows


def _print_completion(clips, led, rows):
    """Print the human completion line, the failure-class marker and the [F0_CACHE] marker; returns the
    budget-skipped side count."""

    def _n(s):
        return sum(1 for r in rows if r["status"] == s)

    ok = sum(1 for r in rows if is_ok_status(r["status"]))
    _re = sum(1 for r in rows if r.get("stale"))
    # **Every** status is counted — a missing status makes the sum disagree and that clip vanishes
    # from the summary. When a new status is added, this line grows with it.
    _other = (
        len(clips)
        - ok
        - _n("REFUSED_BIG")
        - _n("REPAIR_FAIL")
        - _n("FAIL_NOLEDGER")
        - _n("FAIL_NO_BUDGET_MARK")
        - _n(DATA_DEFECT)
        - _n(HELD)
    )
    # The completion line prints the number of sides dropped on budget directly. Clip statuses alone hide this gap when
    # one hand was repaired: the same `completed 50/50 (repaired 48 · shallow 2)` from two runs can come with different
    # output md5. It is printed even when 0 (its absence means the run is suspect).
    _bskip = sum(int(r.get("bskip", 0) or 0) for r in rows)
    print(
        f"[f0_prejudge] completed {ok}/{len(clips)} (repaired {_n('REPAIRED')} · clean "
        f"{_n('CLEAN')} · shallow-skipped {_n('SHALLOW')} · cached {_n('CACHED')}"
        f" · refused (oversized) {_n('REFUSED_BIG')} · repair failed {_n('REPAIR_FAIL')}"
        f" · **budget-skipped sides {_bskip}**"
        f" · no marker {_n('FAIL_NO_BUDGET_MARK')}"
        f" · unexpected {_n('FAIL_NOLEDGER')}"
        f" · data defects {_n(DATA_DEFECT)}"
        f" · held (object repair refused) {_n(HELD)}"
        + (f" · unclassified {_other}" if _other else "")
        + f") · regenerated after upstream change {_re} · ledger {led}",
        flush=True,
    )
    print(failclass_marker(rows), flush=True)  # a nonzero run-failure count makes stage2 fail the chunk
    # cache-key marker: proof that this run decided with the v2 content key + v1 migrations + pass-through refreshes
    _mig = sum(int(r.get("migrated", 0) or 0) for r in rows)
    _npt = sum(int(r.get("passthrough", 0) or 0) for r in rows)
    print(
        f"[F0_CACHE] key={FP_DESIGN} hits {_n('CACHED')} (v1→v2 migrated {_mig}) · re-solved {_re} · "
        f"pass-through refreshed {_npt}",
        flush=True,
    )
    return _bskip


def run_f0(a, clips, stage_src, wrist_dirs, dst_tree, dst_wrist_dirs, root, run_id, run_started, sp, scene) -> int:
    """f0 over every clip (GPU engine): per-clip ledger rows, completion lines, the completion ledger and its rc."""
    led, rows = _solve_and_ledger(a, clips, dst_tree, dst_wrist_dirs, root, stage_src, wrist_dirs)

    _bskip = _print_completion(clips, led, rows)

    # machine-readable completion ledger: the values are owned by this JSON alone
    # The line above is for humans. Consumers (scripts/stage2_ground.sh, the stage-1 driver) never derive counts from
    # it; they read the ledger the marker below points to.
    if len(rows) != len(clips):
        # a worker silently dropping results dies here, so a partial completion never looks like success
        raise SystemExit(
            f"[f0_prejudge] count mismatch: ledger rows {len(rows)} ≠ requested clips {len(clips)} — the run is invalid"
        )
    summary = f0_summary.build(
        run_id=run_id,
        rows=rows,
        ok_statuses=OK_STATUSES,
        budget_skip_sides=_bskip,
        started=run_started,
        finished=time.time(),
        tree=_REPO,
        mode="run",
        hand=a.hand,
        hand_spec=sp,
        dataset=a.dataset,
        emb=a.emb,
        src_layer=a.src_layer,
        shadow_root=str(root),
        scene=scene,
        ledger=str(led),
        prep_threads=a.prep_threads,
        engine="gpu",
        gpus=a.gpus,
        gpu_workers=a.gpu_workers_per_device,
    )
    jpath = f0_summary.write(root, summary)
    print(f0_summary.marker_line(summary, jpath), flush=True)
    return summary["rc"]
