"""Write clips60.tsv (clip, dataset, embodiment, task) for the 60 E5 clips from the frozen pick lists.

python -m experiments.policies.spider_arm.clips60 --picks <dir with *_pick20*.tsv> --data <data root> <out.tsv>
"""

import argparse
import csv
import glob
import os
import sys

PICKS = (("pnp", "pnp_pick20.tsv"), ("pour", "pour_pick20_v2.tsv"), ("sweep", "sweep_pick20_v2.tsv"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--picks", required=True, help="folder of the frozen pick lists")
    ap.add_argument("--data", required=True, help="data root (data/<dataset>/base/wuji2/<embodiment>/<clip>)")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    rows = []
    for task, f in PICKS:
        with open(f"{a.picks}/{f}") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["state"] != "pick":
                    continue
                hits = glob.glob(f"{a.data}/{r['dataset']}/base/wuji2/*/{r['clip']}")
                if len(hits) != 1:
                    sys.exit(f"[CLIPS60] FAIL {r['clip']}: {len(hits)} trees")
                rows.append([r["clip"], r["dataset"], os.path.basename(os.path.dirname(hits[0])), task])
    if len(rows) != 60 or len({r[0] for r in rows}) != 60:
        sys.exit(f"[CLIPS60] FAIL {len(rows)} rows, {len({r[0] for r in rows})} unique")
    with open(a.out, "w") as fh:
        fh.write("clip\tdataset\tembodiment\ttask\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print(f"[CLIPS60] OK 60 rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
