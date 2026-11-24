"""PARK E-clip oracle — a report, not a gate: it never changes a judge value.

The PARK rule (`dexengine.rl.judge.park`) counts the |dz| of parked-hand-touch (E) samples as 0 in gate A. If the
cut-time state of those samples hurt the next batch, the harm could only show as a lower real-batch completion rate of
that E clip, because the judge keeps the same env <-> clip pairing in every batch (static assignment). So the oracle
compares, per clip, the completion-rate change Δ between the warm-up batch (batch 0, no earlier cut) and the real
batches (after a cut), on E clips against non-E clips:
  value = mean(Δ | E clips) − mean(Δ | non-E clips) [pp] · 95 % interval = cluster (clip) bootstrap, BOOT draws, seed
  SEED. A lower bound < LOW_PP gives status "low": re-check that dataset's judge (report only; the judge sidecar does
  not read this line).
  E clip = in a majority of batches, a majority of that clip's samples on either side are E (`Park.e_rows` per batch).
  Without static assignment (single-object one-hand bundles: the pairing changes every batch) the harm would land on
  any clip and cannot be seen, so the status is "none" with a `why`. The evaluation loop says which assignment it used.
Unit: one judge = one dataset (DEXENGINE_DATASET). Several judges of one dataset are pooled with this file's CLI from
their judge records (`park_e` of `<judge>_clips.json`, copied there by judge/metrics.py); the cluster is then
(record, clip).
Marker: `[PARK_E_ORACLE] {json}`, one line at the end of a judge (PARK on, not a probe); its reader is
judge/metrics.py `PARK_E_TAG`. The warm-up results are collected only here; the clip accumulators and the AGGREGATE of
the judge loop are unchanged.
CLI: python3 park_e_oracle.py <judge>_clips.json …  → one pooled line (rc 2 = a record without park_e). Standard
library only.
"""

from __future__ import annotations

import json
import random
import sys

TAG = "[PARK_E_ORACLE] "
LOW_PP = -1.5  # lower-bound threshold [pp]: the lower end of the judge's ±1.5 pp noise band
BOOT = 4000
SEED = 42


def e_clips(batches: list[dict]) -> set[int]:
    """Per batch {(clip, side): (E samples k, samples n)} → the E clips (in a majority of batches, a majority of the
    clip's samples on either side are E)."""
    votes: dict[int, int] = {}
    for rows in batches:
        for c in {c for (c, _side), (k, n) in rows.items() if 2 * k >= n}:
            votes[c] = votes.get(c, 0) + 1
    return {c for c, v in votes.items() if 2 * v >= len(batches)}


def delta_ci(rows: list[tuple]) -> dict:
    """[(cluster key, E?, Δ pp)] → {n, n_e, delta, ci95, low_pp, status}; the interval resamples cluster keys with
    replacement. Without E or without non-E rows the status is "none" (nothing to measure)."""
    by: dict = {}
    for key, e, d in rows:
        by.setdefault(key, []).append((bool(e), float(d)))
    keys = sorted(by)

    def stat(ks):
        e = [d for k in ks for x, d in by[k] if x]
        n = [d for k in ks for x, d in by[k] if not x]
        return sum(e) / len(e) - sum(n) / len(n) if e and n else None

    out = {
        "n": len(keys),
        "n_e": sum(any(x for x, _ in by[k]) for k in keys),
        "delta": stat(keys),
        "ci95": None,
        "low_pp": LOW_PP,
        "status": "none",
    }
    if out["delta"] is None:
        return out
    r = random.Random(SEED)
    bs = sorted(v for v in (stat([r.choice(keys) for _ in keys]) for _ in range(BOOT)) if v is not None)
    lo, hi = bs[int(0.025 * len(bs))], bs[min(int(0.975 * len(bs)), len(bs) - 1)]
    out.update(delta=round(out["delta"], 3), ci95=[round(lo, 3), round(hi, 3)], status="low" if lo < LOW_PP else "ok")
    return out


class Collector:
    """Collects, for one judge (one run_eval_balanced call), the per-clip [completed, evaluated] of the warm-up and
    the real batches and the E rows of each batch. With on=False every method does nothing (PARK off, or a probe).
    static = the judge assignment pins each env to one clip."""

    def __init__(self, clip_names: list[str], on: bool, static: bool):
        self.names, self.on, self.static = clip_names, on, bool(static)
        self.acc: dict = {True: {}, False: {}}  # warm-up? → clip → [completed, evaluated]
        self.e_batches: list[dict] = []
        self.n = {True: 0, False: 0}

    def add(self, warmup: bool, clip_counts: dict, e_rows: dict) -> None:
        if not self.on:
            return
        acc = self.acc[bool(warmup)]
        for c, (comp, early) in clip_counts.items():
            a = acc.setdefault(int(c), [0, 0])
            a[0] += int(comp)
            a[1] += int(comp) + int(early)
        self.e_batches.append(dict(e_rows))
        self.n[bool(warmup)] += 1

    def rows(self) -> list[tuple[str, bool, float]]:
        """[(clip name, E?, Δ = real − warm-up completion rate in pp)] for the clips measured in both."""
        e = e_clips(self.e_batches)
        out = []
        for c in sorted(set(self.acc[True]) & set(self.acc[False])):
            (wc, wt), (rc, rt) = self.acc[True][c], self.acc[False][c]
            if wt and rt:
                out.append((self.names[c], c in e, round(100.0 * (rc / rt - wc / wt), 3)))
        return out

    def report(self) -> dict | None:
        if not self.on:
            return None
        rows = self.rows()
        d = dict(
            delta_ci(rows),
            warm_batches=self.n[True],
            real_batches=self.n[False],
            rows=[[k, int(e), x] for k, e, x in rows],
            static=self.static,
        )
        if not self.static:
            d.update(
                status="none",
                why="non-static assignment — env ↔ clip pairs change every batch, so the E clips' harm cannot be seen",
            )
        print(TAG + json.dumps(d, ensure_ascii=False), flush=True)
        return d


def collector_for(clip_names: list[str] | None, num_clips: int, *, static: bool) -> Collector:
    """An active collector when PARK is on and this is not a probe, else an inactive one (no PARK means no E).
    static = the judge assignment."""
    from dexengine.rl.judge.park import enabled, probe_steps

    names = list(clip_names) if clip_names else [f"clip_{i}" for i in range(num_clips)]
    return Collector(names, on=enabled() and not probe_steps(), static=static)


def main(argv: list[str] | None = None) -> int:
    """Several judge records (one dataset) → one pooled line. A record without park_e dies (it is never silently
    dropped)."""
    files = sys.argv[1:] if argv is None else argv
    if not files:
        print("usage: park_e_oracle.py <judge>_clips.json …", file=sys.stderr)
        return 2
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            pe = json.load(fh).get("park_e")
        if not pe or "rows" not in pe:
            print(
                f"[PARK_E_ORACLE] {f}: the judge record has no park_e — not the record of a PARK-on judge",
                file=sys.stderr,
            )
            return 2
        rows += [((f, k), bool(e), float(x)) for k, e, x in pe["rows"]]
    print(TAG + json.dumps(dict(delta_ci(rows), files=len(files)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
