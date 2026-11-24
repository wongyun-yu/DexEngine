#!/usr/bin/env python3
"""Floating-hand result + post-hoc arm IK (paper Sec. E.4), (c-full) open-loop replay: ADD-AUC aggregate.

Inputs under <dir>: c_replay/{cfull,cpos,cctl}/clips/<clip>.json (the E1–E3 scorer on the zero-residual replays of the
c-full trees, the one-run post-hoc arm and the canonical control) and c_posthoc_arm_ik_limits_pos.json (the arm-IK
limits of the one-run arm); <list> = the clips that have a c-full arm path. Writes <out>/cfull_replay_add_auc.json and
<out>/cfull_replay_per_clip.tsv (default <out> = <dir>).

    python -m experiments.ablations.posthoc.cfull_replay_agg <dir> <list> [--out-dir <dir>]
"""

import argparse
import json
import statistics as st
from pathlib import Path


def load(R, n):
    """Scored clips of one replay variant (clip name without the `_pl_rt` suffix → score record)."""
    out = {}
    for f in sorted((R / "c_replay" / n / "clips").glob("*.json")):
        j = json.loads(f.read_text())
        out[j["clip"].replace("_pl_rt", "")] = j
    return out


def summ(d, keep):
    ks = [k for k in d if k in keep]

    def f(key):
        return [d[k][key] for k in ks]

    return {
        "n": len(ks),
        "add_auc_best_mean": round(st.fmean(f("add_auc_best")), 4),
        "add_auc_best_median": round(st.median(f("add_auc_best")), 4),
        "add_auc_pooled_mean": round(st.fmean(f("add_auc")), 4),
        "spider_sr_mean": round(st.fmean(f("spider_sr")), 4),
        "maniptrans_sr_mean": round(st.fmean(f("maniptrans_sr")), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("list", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    R, out_dir = a.dir, a.out_dir or a.dir
    with open(a.list) as fh:
        L42 = [ln.strip() for ln in fh if ln.strip()]
    cf, cp, cc = load(R, "cfull"), load(R, "cpos"), load(R, "cctl")
    bad = [k for k, v in cf.items() if "error" in v]
    S = set(cf)
    names42 = {x.replace("_pl_rt", "") for x in L42}
    print(
        "cfull clips",
        len(cf),
        "errors",
        len(bad),
        "in cpos",
        len(S & set(cp)),
        "in cctl",
        len(S & set(cc)),
        "list42 match",
        len(S & names42),
        sorted(names42 - S)[:2],
        sorted(S)[:1],
    )
    lim = {
        r["clip"].replace("_pl_rt", ""): r
        for r in json.loads((R / "c_posthoc_arm_ik_limits_pos.json").read_text())["rows"]
    }
    p35 = {k for k, r in lim.items() if r["pass_limits"]}
    out = {
        "what": "(c-full) floating-hand wrists + canonical whole-clip arm solve + canonical retiming, open-loop "
        "zero-residual replay; scored with the E1-E3 scorer",
        "n_with_path": len(cf),
        "n_total": 64,
        "cfull_on_42": summ(cf, S),
        "cctl_canonical_on_same_42": summ(cc, S),
        "cpos_one_run_on_same_42": summ(cp, S),
        "cfull_on_42_and_one_run_pass": summ(cf, S & p35),
        "cpos_one_run_on_42_and_pass": summ(cp, S & p35),
        "n_42_and_one_run_pass": len(S & p35),
        "cctl_all_64": summ(cc, set(cc)),
        "cpos_all_64": summ(cp, set(cp)),
        "cfull_zero_filled_64": round(sum(v["add_auc_best"] for v in cf.values()) / 64, 4),
        "note": "cfull_zero_filled_64 counts the 22 clips without an arm path as 0 — a choice, not a measurement",
    }
    (out_dir / "cfull_replay_add_auc.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    with open(out_dir / "cfull_replay_per_clip.tsv", "w") as f:
        f.write(
            "clip\tcfull_add_auc_best\tcfull_add_auc\tcfull_spider_sr\tcctl_add_auc_best\tcpos_add_auc_"
            "best\tone_run_pass_limits\n"
        )
        for k in sorted(cf):
            f.write(
                f"{k}\t{cf[k]['add_auc_best']:.4f}\t{cf[k]['add_auc']:.4f}\t{cf[k]['spider_sr']:.4f}\t"
                f"{cc[k]['add_auc_best']:.4f}\t{cp[k]['add_auc_best']:.4f}\t{int(k in p35)}\n"
            )
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, v["n"], v["add_auc_best_mean"], v["add_auc_best_median"], v["spider_sr_mean"])
        elif k.startswith(("cfull_zero", "n_")):
            print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
