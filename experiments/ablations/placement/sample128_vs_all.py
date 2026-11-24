#!/usr/bin/env python3
"""Placement-search ablation (paper Sec. E.5): the 128-clip Wuji 2 sample against the whole production record.

Reads the sample list (draw.py's clips_main.tsv; column prod_status = the production ledger verdict at draw time) and
the folded Wuji 2 ledgers <root>/overfit/hands/wuji2/ledger/status_final_*.tsv. Row "ours" of the ablation takes
accept / stretch from this same production record, so both numbers come from one judge under one condition; the only
difference is which clips are counted. Writes one JSON.

    python -m experiments.ablations.placement.sample128_vs_all <data root> <clips_main.tsv> <out.json>
"""

import collections
import csv
import glob
import json
import math
import os
import sys
import time


def main(root, lst, out):
    led = {}
    for f in sorted(glob.glob(f"{root}/overfit/hands/wuji2/ledger/status_final_*.tsv")):
        ax = os.path.basename(f)[len("status_final_") : -4]
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                led[(ax, r["clip"])] = r["status"]
    with open(lst, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    mis = [
        (r["axis"], r["clip"], r["prod_status"], led.get((r["axis"], r["clip"])))
        for r in rows
        if led.get((r["axis"], r["clip"])) != r["prod_status"]
    ]

    def ds_of(ax):
        return ax.split("-")[0]

    allc, alla = collections.Counter(), collections.Counter()
    for (ax, _), s in led.items():
        allc[ds_of(ax)] += 1
        alla[ds_of(ax)] += s == "ADOPTED"
    sc, sa = collections.Counter(), collections.Counter()
    for r in rows:
        sc[r["dataset"]] += 1
        sa[r["dataset"]] += r["prod_status"] == "ADOPTED"
    n, k, N, K = len(rows), sum(sa.values()), len(led), sum(alla.values())
    p = K / N
    exp = sum(sc[d] * alla[d] / allc[d] for d in sc)  # kept count the sample would have at each dataset's own rate
    var = sum(sc[d] * (alla[d] / allc[d]) * (1 - alla[d] / allc[d]) for d in sc)
    res = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "hand": "wuji2",
        "sample_list": lst,
        "sample_n": n,
        "sample_kept": k,
        "sample_kept_pct": round(100 * k / n, 2),
        "sample_status": dict(collections.Counter(r["prod_status"] for r in rows)),
        "all_n": N,
        "all_kept": K,
        "all_kept_pct": round(100 * p, 2),
        "sample_vs_ledger_now_mismatch": mis,
        "expected_kept_in_sample_at_dataset_rates": round(exp, 1),
        "sd_of_that_count": round(math.sqrt(var), 2),
        "z": round((k - exp) / math.sqrt(var), 2),
        "per_dataset": {
            d: {
                "sample_n": sc[d],
                "sample_kept": sa[d],
                "all_n": allc[d],
                "all_kept": alla[d],
                "all_kept_pct": round(100 * alla[d] / allc[d], 1),
            }
            for d in sorted(sc)
        },
        "reading": "Same judge record and same conditions for both numbers (Tab. 4 row 'ours' = production verdicts "
        "of the sample). "
        "The gap is which clips were drawn: the sample is stratified by dataset (seed 42), uniform inside a dataset.",
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print({k2: v for k2, v in res.items() if k2 not in ("per_dataset", "reading", "sample_list")})
    for d, v in res["per_dataset"].items():
        print(d, v)


if __name__ == "__main__":
    main(*sys.argv[1:4])
