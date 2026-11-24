#!/usr/bin/env python3
"""Held-out transfer probe table (RL ablation): scored judges of a trained checkpoint on 64 held-out clips.

Inputs: <results>/<alias>/<clip>.json (the E1–E3 scorer in reference-only object mode; default <results> =
$DEXENGINE_E123_ROOT/results/ours_rl_ho) and the clip lists experiments/scoring/clips/E123_{HELDOUT,CLIPS}_v1.tsv.
One row per alias with the ablation table's best-of-64 rule: ADD-AUC = mean over the 64 clips of add_auc_best (a clip
with no score = 0) · SP / MP = clips with ≥ 1 of 64 rollouts passing (sp_k / mp_k ≥ 1).
Strata (descriptive): seen_triplet
47 · unseen_triplet 17 · any untrained object 34. The two criteria were fixed before the results.
Writes <out>/heldout_table.{json,md} (default <out> = <results>).

    python -m experiments.ablations.rl_arm.heldout.ho_table [--results <dir>] [--out-dir <dir>]
"""

import argparse
import csv
import json
import pathlib

from experiments import workspace

CLIPS = workspace.REPO / "experiments/scoring/clips"
ALIAS = {
    "a0_ho": "HO_a0r21_ho64_pol_jf1600r2",
    "zero_ho": "HO_a0r21_ho64_zero_jf1600r2",
    "a4_ho": "HO_a4r14_ho64_pol_jf1600r2",
    "zero_k64": "HO_a0r21_k64_zero_jf1600r2",
}
#: in-distribution values the criteria compare against (training 64 clips · RTX 3090 · ablation_table.json)
IN = {"a0_k64": 0.8964, "a4_k64": 0.8379, "noise": 0.004}


def rd(p):
    return list(csv.DictReader((ln for ln in open(p) if not ln.startswith("#")), delimiter="\t"))


def strata():
    """Clip sets: the held-out strata and the training 64."""
    HO = rd(CLIPS / "E123_HELDOUT_v1.tsv")
    K64 = [r["clip"] for r in rd(CLIPS / "E123_CLIPS_v1.tsv")]
    assert len(HO) == 64 and len({r["clip"] for r in HO}) == 64 and len(set(K64)) == 64
    S = {
        "all": [r["clip"] for r in HO],
        "seen_triplet": [r["clip"] for r in HO if r["stratum"] == "seen_triplet"],
        "unseen_triplet": [r["clip"] for r in HO if r["stratum"] == "unseen_triplet"],
        "untrained_object": [r["clip"] for r in HO if r["right_obj_trained"] == "0" or r["left_obj_trained"] == "0"],
    }
    assert [len(v) for v in S.values()] == [64, 47, 17, 34], [len(v) for v in S.values()]
    return S, K64


def score(R, alias, clips):
    """Best-of-64 row of one alias over `clips`."""
    d = R / alias
    js = {c: json.loads((d / f"{c}.json").read_text()) for c in clips if (d / f"{c}.json").is_file()}

    def g(c, k):
        return float(js[c][k]) if c in js and js[c].get(k) is not None else 0.0

    return {
        "n": len(clips),
        "scored": len(js),
        "add_auc_best": round(sum(g(c, "add_auc_best") for c in clips) / len(clips), 4),
        "sp_clips": sum(g(c, "sp_k") >= 1 for c in clips),
        "mp_clips": sum(g(c, "mp_k") >= 1 for c in clips),
    }


def build(R) -> dict:
    """Rows of every alias and the two criteria (C1 transfer · C2 the contact arm loses more held out)."""
    S, K64 = strata()
    out = {
        "rows": {
            k: {s: score(R, a, S[s] if k != "zero_k64" else K64) for s in (S if k != "zero_k64" else ["all"])}
            for k, a in ALIAS.items()
        },
        "in_dist": IN,
    }
    r = out["rows"]
    full = all(r[k]["all"]["scored"] == 64 for k in r)
    gap_in = round(IN["a0_k64"] - r["zero_k64"]["all"]["add_auc_best"], 4)
    gap_ho = round(r["a0_ho"]["all"]["add_auc_best"] - r["zero_ho"]["all"]["add_auc_best"], 4)
    d_in = round(IN["a0_k64"] - IN["a4_k64"], 4)
    d_ho = round(r["a0_ho"]["all"]["add_auc_best"] - r["a4_ho"]["all"]["add_auc_best"], 4)
    out["criteria"] = {
        "complete": full,
        "gap_in": gap_in,
        "gap_ho": gap_ho,
        "c1_transfer": full and gap_ho >= 0.05 and gap_ho >= 0.5 * gap_in,
        "d_in": d_in,
        "d_ho": d_ho,
        "c2_contact_more": full and d_ho - d_in >= 0.03,
    }
    return out


def markdown(out) -> str:
    L = [
        "| row | stratum | clips | scored | best ADD-AUC | SPIDER clips | ManipTrans clips |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, ss in out["rows"].items():
        for s, x in ss.items():
            L.append(
                f"| {k} | {s} | {x['n']} | {x['scored']} | {x['add_auc_best']:.4f} | {x['sp_clips']} | "
                f"{x['mp_clips']} |"
            )
    c = out["criteria"]
    L += [
        "",
        f"complete={c['complete']} · gap_in={c['gap_in']} · gap_ho={c['gap_ho']} · C1 transfer={c['c1_transfer']} · "
        f"d_in={c['d_in']} · d_ho={c['d_ho']} · C2 contact-more={c['c2_contact_more']}",
    ]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--results",
        type=pathlib.Path,
        default=None,
        help="scored held-out judges (default $DEXENGINE_E123_ROOT/results/ours_rl_ho)",
    )
    ap.add_argument("--out-dir", type=pathlib.Path, default=None, help="default: --results")
    a = ap.parse_args(argv)
    R = a.results or workspace.results("ours_rl_ho")
    out_dir = a.out_dir or R
    out = build(R)
    (out_dir / "heldout_table.json").write_text(json.dumps(out, indent=1))
    md = markdown(out)
    (out_dir / "heldout_table.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
