#!/usr/bin/env python3
"""Where the clips that did not become feasible were dropped, per hand (placement funnel; no GPU, counts read from
judge records).

Reads only: the per-hand folded ledgers `overfit/hands/<hand>/ledger/status_final_<axis>.tsv` (status · judge_budget)
and the output of the classification owner `experiments/scoring/judge_fail_detail.py` (--out-clips: per clip, the check
at which the candidate that got furthest fell). Nothing is re-judged and no rule is restated: the stage of a clip is
(ledger status, budget-cut kind, owner's fine reason).

  python -m experiments.ablations.placement.drop_stages --root <data checkout> --raw <dir with fail_clips_<hand>.tsv> \
      --out <dir>     → <out>/b1_drop_stages.{json,tsv}

The fine-reason keys and the labels in ROWS carry the judge's Korean record words (parsed byte for byte from the
classifier output and written unchanged into the JSON).
"""

import argparse
import collections
import csv
import glob
import json
import os
import subprocess
import time
from pathlib import Path

HANDS = (("sharpa_wave", "SharpA"), ("xhand_pro", "XHand Pro"), ("dg5f_s", "Tesollo"), ("wuji2", "Wuji 2"))
# paper stage of a dropped clip. Column of the stop = ledger status + budget cut; row = owner's fine reason.
STOP = (
    ("all_tested", "every candidate tested, none passed (ledger EXHAUSTED)"),
    ("cap_candidates", "stopped at the candidate cap (ledger UNPROVEN, judge_budget cell:* — cap 1,024)"),
    ("cap_time", "stopped at the time cap (ledger UNPROVEN, judge_budget wall:* — 1,200 s)"),
    ("cap_unmarked", "ledger UNPROVEN with no budget-cut mark"),
)
ROWS = (
    (
        "screen_none",
        "screening leaves no candidate (no_volume · no_place_spot · no_receiver_spot)",
        ("reach:no_volume", "reach:no_place_spot", "reach:no_receiver_spot"),
    ),
    (
        "screen_cover",
        "candidates rejected for arm-reach coverage (IK pool cover; no candidate reached a check)",
        ("D_gate_pool_cov",),
    ),
    (
        "reach_frames",
        "no candidate reached the checks: frames unreachable / early stop (early_stop · frame_infeasible)",
        ("C_track_reach",),
    ),
    (
        "nocheck",
        "no candidate reached a check, reason row only (reach:EXHAUSTED · reach:TRUNCATED)",
        ("reach:EXHAUSTED", "reach:TRUNCATED"),
    ),
    ("ik_track", "arm IK: wrist tracking error (error)", ("best:error",)),
    ("retime_stretch", "retiming: length ratio (stretch)", ("best:stretch",)),
    ("retime_speed", "retiming: speed ratio (speed_ratio)", ("best:speed_ratio",)),
    ("ik_jump", "arm IK: joint jump (jump)", ("best:jump",)),
    ("ik_flip", "arm IK: arm flip (arm_flip)", ("best:arm_flip",)),
    ("pen_desk", "penetration: table (desk_pen)", ("best:desk_pen",)),
    ("pen_torso", "penetration: torso (torso_pen)", ("best:torso_pen",)),
    ("ik_swivel", "arm IK: swivel swing (swivel)", ("best:swivel",)),
    (
        "final_in_judge",
        "judge's final guard / final table check (final_guard_rejected · final_desk_check_rejected)",
        ("reach:final_guard_rejected", "reach:final_desk_check_rejected", "reach:v2_port_guard_rejected"),
    ),
)


def _read_ledger(a, hand):
    """Final ledger status and judge budget per (axis, clip) of one hand."""
    st, bud = {}, {}
    files = sorted(glob.glob(f"{a.root}/overfit/hands/{hand}/ledger/status_final_*.tsv"))
    for f in files:
        ax = os.path.basename(f)[len("status_final_") : -4]
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                k = (ax, r["clip"])
                if k in st:
                    raise SystemExit(f"duplicate {k}")
                st[k], bud[k] = r["status"], (r.get("judge_budget") or "-")
    return bud, st, files


def _classify_drops(a, hand, st, bud, fine2row):
    """Count the EXHAUSTED/UNPROVEN clips per reason row and stop column (classifier rows must match the ledger)."""
    fc = json.loads(Path(f"{a.raw}/fail_clips_{hand}.tsv").read_text(encoding="utf-8"))
    tab = {r[0]: collections.Counter() for r in ROWS}
    seen, other = set(), collections.Counter()
    for key, rows in fc.items():
        ax, fine = key.split("|", 1)
        if ax == "ALL":  # the owner also writes an all-axes copy of the same rows
            continue
        for clip, s, _tag, _best in rows:
            k = (ax, clip)
            if st.get(k) != s:
                raise SystemExit(f"{hand} {k}: ledger {st.get(k)} != classifier {s}")
            if s not in ("EXHAUSTED", "UNPROVEN"):
                continue
            if k in seen:
                raise SystemExit(f"{hand} {k} twice")
            seen.add(k)
            stop = (
                "all_tested"
                if s == "EXHAUSTED"
                else (
                    "cap_candidates"
                    if bud[k].startswith("cell:")
                    else "cap_time"
                    if bud[k].startswith("wall:")
                    else "cap_unmarked"
                )
            )
            if fine not in fine2row:
                other[(fine, stop)] += 1
                continue
            tab[fine2row[fine]][stop] += 1
    if other:
        raise SystemExit(f"{hand}: fine reasons outside the table: {dict(other)}")
    return tab, seen


def _record(a, res):
    """The output record (what, reading, the per-hand results)."""
    rev = subprocess.run(
        ["git", "-C", a.root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "what": "clips not kept by the feasibility stage, by where they were dropped",
        "root": a.root,
        "root_commit": rev,
        "classifier": "experiments/scoring/judge_fail_detail.py (owner of the fine reason)",
        "stop_columns": dict(STOP),
        "reason_rows": {r[0]: r[1] for r in ROWS},
        "reading": (
            "Rows = the check at which the candidate that got furthest fell (judge order: tracking error → "
            "length ratio → speed ratio → jump → table penetration → swivel swing). Columns = why the search "
            "stopped. A clip under cap_* was NOT proven infeasible: the search stopped at the budget."
        ),
        "cannot_split": [
            "audit: the ledgers hold no separate 'audit' state — post-judge rejections are GUARD_FAIL "
            "(bake/gate physical guard) and SOURCE_DEFECT (input defect); listed under after_judge",
            "retiming ratio: no dropped clip has its furthest candidate stopped at the length-ratio or "
            "speed-ratio check (both rows are 0); per-candidate ratio rejections exist only as tallies "
            "inside the reason strings and are not a per-clip stage",
        ],
        "hands": res,
    }
    return out


def _write_tsv(a, res):
    """Write the TSV and print the per-hand lines."""
    with open(f"{a.out}/b1_drop_stages.tsv", "w", encoding="utf-8") as f:
        f.write("hand\tsection\trow\tall_tested\tcap_candidates\tcap_time\tcap_unmarked\ttotal\n")
        for hand, name in HANDS:
            h = res[hand]
            f.write(
                f"{name}\tledger\tclips\t\t\t\t\t{h['clips_in_ledger']}\n{name}\tledger\tkept\t\t\t\t\t{h['kept']}\n"
            )
            for r in ROWS:
                v = [h["table"][r[0]][s[0]] for s in STOP]
                f.write(f"{name}\tjudge\t{r[0]}\t" + "\t".join(map(str, v)) + f"\t{sum(v)}\n")
            v = [h["stop_totals"][s[0]] for s in STOP]
            f.write(f"{name}\tjudge\tTOTAL\t" + "\t".join(map(str, v)) + f"\t{sum(v)}\n")
            for k, n in h["after_judge"].items():
                f.write(f"{name}\tafter_judge\t{k}\t\t\t\t\t{n}\n")
            v = h["stop_totals"]
            print(
                f"{name}: ledger {h['clips_in_ledger']} kept {h['kept']} ({h['kept_pct']} %) dropped "
                f"{h['dropped']} | all tested "
                f"{v['all_tested']} · candidate cap {v['cap_candidates']} · time cap {v['cap_time']} · "
                f"unmarked {v['cap_unmarked']} | "
                f"after judge {h['after_judge']}"
            )
            print(
                "   "
                + " · ".join(
                    f"{r[0]} {sum(h['table'][r[0]].values())} ({h['table'][r[0]]['all_tested']} all-tested)"
                    for r in ROWS
                )
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fine2row = {f: r[0] for r in ROWS for f in r[2]}
    res = {}
    for hand, name in HANDS:
        bud, st, files = _read_ledger(a, hand)
        status = collections.Counter(st.values())
        tab, seen = _classify_drops(a, hand, st, bud, fine2row)
        n_pf = status["EXHAUSTED"] + status["UNPROVEN"]
        if len(seen) != n_pf:
            raise SystemExit(f"{hand}: classified {len(seen)} != ledger EXHAUSTED+UNPROVEN {n_pf}")
        tot = sum(status.values())
        kept = status["ADOPTED"]
        res[hand] = {
            "name": name,
            "ledger_files": len(files),
            "clips_in_ledger": tot,
            "kept": kept,
            "kept_pct": round(100 * kept / tot, 2),
            "dropped": tot - kept,
            "status": dict(status),
            "stop_totals": {s[0]: sum(tab[r[0]][s[0]] for r in ROWS) for s in STOP},
            "table": {r[0]: {s[0]: tab[r[0]][s[0]] for s in STOP} for r in ROWS},
            "after_judge": {
                "GUARD_FAIL": status["GUARD_FAIL"],
                "SOURCE_DEFECT": status["SOURCE_DEFECT"],
                "BAKE_ERROR": status["BAKE_ERROR"],
            },
        }
    out = _record(a, res)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/b1_drop_stages.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return _write_tsv(a, res)


if __name__ == "__main__":
    main()
