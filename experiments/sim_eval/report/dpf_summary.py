#!/usr/bin/env python3
"""DP final-checkpoint evaluation table, three tasks × training sets: counted directly from the result JSONs
(<level>.json of each evaluation batch).
Cells = L0 successes/scenes · L1 successes/scenes · per level 'clips with at least one success / clips in the list ·
/20' · invalid (by reason).
Input = spec TSV (task · who (ours|kin) · run · level · result JSON path — a level split over several batches has
several rows; a scene in two batches is resolved by verdicts.overlap). A scene judged in no batch (verdict outside
success/fail/invalid) is not a value: it is written as '(unjudged k)' in its cell and the whole table stays
'unfinished'.
Lists = <lists>/<task>_<l0|l1|l2>.txt — a result scene set different from the list is reported, never merged silently.
Train-set check: per list L0 ⊆ keys and L1 ∩ keys = ∅ (keys = train_keys over the training sets' episodes.jsonl under
--train-root); a violation keeps the table 'unfinished'. Only old results of scenes listed in
<lists>/removed_not_train.txt (must be outside the training set) are dropped from the counts (and reported).
The texts written to the .md / .json / unjudged.tsv outputs and the last [DPF_SUM] line are read by the paper tables.
Usage: python -m experiments.sim_eval.report.dpf_summary <spec.tsv> <out.md> --lists DIR --sent-dir DIR
  --train-root DIR"""

import argparse
import collections
import csv
import datetime as _dt
import json
import os
import re
from pathlib import Path

from experiments.sim_eval import train_keys as TKEYS
from experiments.sim_eval.report import verdicts as JVM  # owner of the batch-overlap rule (explicit supersedes column)
from experiments.sim_eval.report.verdicts import JV  # owner of the judged values — anything else is unjudged
from experiments.textio import read_lines


def _parse_args(argv):
    """Command line: spec, out, the list / sent / train-set folders and the who filter."""
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument(
        "--lists", required=True, help="folder holding <task>_<level>.txt, removed_not_train.txt, lists.sha256"
    )
    ap.add_argument(
        "--sent-dir", required=True, help="L1 loop folder holding <task>_sent.txt (scenes submitted then removed)"
    )
    ap.add_argument("--train-root", required=True, help="root holding the training sets' <set>/meta/episodes.jsonl")
    ap.add_argument("--who", default="ours,kin", help="comma list of the who column to tabulate (SPIDER block: spider)")
    a = ap.parse_args(argv)
    return a


def _read_spec(SPEC):
    """Spec TSV -> result JSONs per (task, who, run, level), plus the supersedes rule and its counters."""
    parts = collections.defaultdict(list)
    SUP = JVM.supersedes(SPEC)
    SUPN = collections.Counter()  # (task, who, level, new tag, old tag) → superseded scene count
    rep = collections.Counter()
    with open(SPEC) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            parts[(r["task"], r["who"], r["run"], r["level"])].append(r["json"])
    return parts, SUP, SUPN, rep


def _read_train_sets(a, LD):
    """Training-set keys, the removed-not-train scenes and the L1 loop's sent records per task."""
    KEYS = {tk: TKEYS.keys(tk, a.train_root) for tk in ("lift", "pour", "sweep")}
    RMF = LD + "/removed_not_train.txt"
    REMOVED = {x.strip() for x in read_lines(RMF) if x.strip()} if os.path.exists(RMF) else set()
    dropped, tsok = collections.Counter(), []
    # results outside the list are not counted. Explained ones are in the L1 loop's sent records (<sent dir>/
    # <task>_sent.txt — old scenes that left the list after submission) → one 'reference — outside the list' line
    # (does not block completion). Outside-list results without a sent record (all of L0, which has none) are
    # unexplained extras = mismatch (blocks completion).
    SW = a.sent_dir
    SENT = {
        tk: {os.path.basename(os.path.dirname(x.strip())) for x in read_lines(f"{SW}/{tk}_sent.txt") if x.strip()}
        if os.path.exists(f"{SW}/{tk}_sent.txt")
        else set()
        for tk in ("lift", "pour", "sweep")
    }
    return KEYS, REMOVED, dropped, tsok, SENT


def _read_part(lv, NOTES2, notes, LD, LIST, tk, js, run, SUP, rep, who, SUPN):
    """Read one cell list and its judge files into scene -> record."""
    NT = NOTES2 if lv == "L2" else notes
    with open(f"{LD}/{LIST[tk]}_{lv.lower()}.txt") as fh:
        want = [
            os.path.basename(os.path.dirname(x.strip())) for x in fh if x.strip() and not x.startswith("#")
        ]  # '#' = L2 list header
    sc = {}
    for j in js:
        d = json.loads(Path(j).read_text())
        assert d["level"] == lv and d["run"] == run, (j, d["level"], d["run"])
        for s in d["scenes"]:
            o = sc.get(s["scene"])
            if o is not None:
                win, old = JVM.overlap(
                    o["verdict"], o["_json"], s["verdict"], j, run, lv, SUP
                )  # ValueError on a conflict
                if old is None:
                    rep[(tk, who, lv)] += 1
                else:
                    SUPN[(tk, who, lv, JVM.tag_of(j) if win == "new" else JVM.tag_of(o["_json"]), old)] += 1
                if win == "old":
                    continue
            sc[s["scene"]] = dict(s, _json=j)
    return NT, want, sc


def _check_part(want, lv, KEYS, tk, NT, NAME, who, TS2, tsok, sc, REMOVED, dropped, SENT, XREF):
    """Check the cell against its list (keys, removed, missing, extra scenes); return the ok scenes."""
    tbad = TKEYS.violations(want, lv, KEYS[tk][0])
    if tbad:
        NT.append(f"{NAME[tk]} {who} {lv}: train-set check failed {len(tbad)} (e.g. {tbad[0]})")
    else:
        (TS2 if lv == "L2" else tsok).append((tk, who, lv))
    for x in [x for x in sc if x in REMOVED and x not in want]:
        assert TKEYS.key_of(x) not in KEYS[tk][0], f"{x} of removed_not_train is in the training set"
        del sc[x]
        dropped[(tk, who, lv)] += 1
    miss = sorted(set(want) - set(sc))
    extra = sorted(set(sc) - set(want))
    xok = [x for x in extra if lv == "L1" and x in SENT[tk]]
    xbad = [x for x in extra if x not in set(xok)]
    if xok:
        XREF.append(f"{NAME[tk]} {who} {lv} {len(xok)}")
    if miss or xbad:
        NT.append(
            f"{NAME[tk]} {who} {lv}: list {len(want)} · results {len(sc)} · missing {len(miss)} · "
            f"unexplained outside the list {len(xbad)}" + (f" (e.g. {xbad[0]})" if xbad else "")
        )
    ok = [
        sc[x] for x in want if x in sc and sc[x]["verdict"] == JVM.JV_SUCCESS
    ]  # count list scenes only (old results outside the list never enter the numerator)
    return ok


def _record_part(clip, want, ok, sc, lv, UNJ2, UNJ, tk, who, run, NT, NAME, RES2, res):
    """Record the cell result, its unjudged scenes and its other verdicts."""
    clips = sorted({clip(s) for s in want})
    okc = sorted({clip(s["scene"]) for s in ok})
    inv = collections.Counter(
        re.sub(r" \(.*\)$", "", s["why"] or "?")
        for s in (sc[x] for x in want if x in sc)
        if s["verdict"] == JVM.JV_INVALID
    )
    oth = collections.Counter(s["verdict"] for s in (sc[x] for x in want if x in sc) if s["verdict"] not in JV)
    (UNJ2 if lv == "L2" else UNJ).extend(
        (tk, who, run, lv, x, sc[x]["verdict"], sc[x]["_json"]) for x in want if x in sc and sc[x]["verdict"] not in JV
    )
    if oth:
        NT.append(
            f"{NAME[tk]} {who} {lv}: unjudged {sum(oth.values())} ("
            + " · ".join(f"{k} {v}" for k, v in sorted(oth.items()))
            + ")"
        )
    (RES2 if lv == "L2" else res)[(tk, who, lv)] = {
        "n": len(want),
        "got": len(sc),
        "ok": len(ok),
        "clips": len(clips),
        "okc": len(okc),
        "inv": inv,
        "oth": oth,
        "run": run,
        "vd": {x: (sc[x]["verdict"], sc[x]["why"] or "") for x in want if x in sc},
    }


def _tally(
    parts, LD, LIST, NAME, SUP, SUPN, rep, KEYS, REMOVED, SENT, res, notes, RES2, NOTES2, UNJ2, TS2, UNJ, dropped, tsok
):
    """Count every (task, who, run, level) cell against its list; fills the result, note and unjudged tables."""

    def clip(sc):
        return sc.rsplit("_pv", 1)[0]

    XREF = []
    for (tk, who, run, lv), js in parts.items():
        NT, want, sc = _read_part(lv, NOTES2, notes, LD, LIST, tk, js, run, SUP, rep, who, SUPN)
        ok = _check_part(want, lv, KEYS, tk, NT, NAME, who, TS2, tsok, sc, REMOVED, dropped, SENT, XREF)
        _record_part(clip, want, ok, sc, lv, UNJ2, UNJ, tk, who, run, NT, NAME, RES2, res)
    return XREF


def _table_state(res, notes, WHOS, tsok, LD, LIST, NAME):
    """Table state: cells complete, lists at 100 scenes, L0 done, train-set check passed."""
    # L0 and L1: 100 scenes per task (kin and ours share the list file) — 'final' only when both are 100/100
    NEED = 100
    nlist = {
        (tk, lv): len([x for x in read_lines(f"{LD}/{LIST[tk]}_{lv.lower()}.txt") if x.strip()])
        for tk in ("lift", "pour", "sweep")
        for lv in ("L0", "L1")
    }
    badlv = {(n.split(" ")[0], n.split(" ")[2].rstrip(":")) for n in notes}  # (task name, level) with a mismatch
    cells_ok = (
        all((tk, w, lv) in res for tk in ("lift", "pour", "sweep") for w in WHOS for lv in ("L0", "L1")) and not notes
    )
    final = cells_ok and all(v == NEED for v in nlist.values())
    l0done = all((tk, w, "L0") in res for tk in ("lift", "pour", "sweep") for w in WHOS) and not any(
        lv == "L0" for _, lv in badlv
    )
    tsall = len(tsok) == len(res) and bool(res)
    state = (
        "final"
        if final
        else "unfinished "
        + " · ".join(
            f"{NAME[tk]} L0 {nlist[(tk, 'L0')]}/{NEED} · L1 {nlist[(tk, 'L1')]}/{NEED}"
            for tk in ("lift", "pour", "sweep")
        )
    )
    return NEED, cells_ok, final, l0done, tsall, state


def _md_table(state, cells_ok, SPEC, RULE, res, WHOS, NAME, invref):
    """Markdown head and the L0/L1 rows; collects the invalid-run notes per row into invref."""
    L = [
        "# three tasks × two runs — DP last-checkpoint evaluation (judge v2 · ground-truth replay frame)",
        "",
        f"**state: {state}**"
        + ("" if cells_ok else " · results do not all match the list (see 'unfinished · mismatches' below)"),
        "",
        f"generator `experiments.sim_eval.report.dpf_summary` · spec `{SPEC}` · numbers counted directly from the "
        "result JSONs.",
        "",
        f"count: **{RULE}**",
        "",
        "| task | run | L0 successes/scenes | L1 successes/scenes | L0 success clips / listed clips · /20 "
        "| L1 success clips / listed clips · /20 |",
        "|---|---|---|---|---|---|",
    ]
    for tk in ("lift", "pour", "sweep"):
        for who in WHOS:
            a, b = res.get((tk, who, "L0")), res.get((tk, who, "L1"))

            def cell(x):
                return (
                    (
                        f"{x['ok']}/{x['n']}"
                        + ("" if x["got"] == x["n"] else f" (results {x['got']})")
                        + (f" (unjudged {sum(x['oth'].values())})" if x["oth"] else "")
                    )
                    if x
                    else "not yet"
                )

            def cc(x):
                return f"{x['okc']}/{x['clips']} · {x['okc']}/20" if x else "not yet"

            inv = collections.Counter()
            for lv, x in (("L0", a), ("L1", b)):
                if x:
                    for k, v in x["inv"].items():
                        inv[f"{lv} {k}"] += v
            ivs = " · ".join(f"{k} {v}" for k, v in sorted(inv.items())) or "0"
            L.append(f"| {NAME[tk]} | {who} | {cell(a)} | {cell(b)} | {cc(a)} | {cc(b)} |")
            invref.append(f"{NAME[tk]} {who} {ivs}")
    return L


def _md_notes(L, invref, res, NAME, dropped, tsok, KEYS, rep, SUPN, XREF, notes):
    """Markdown note lines under the L0/L1 table (appended to L)."""
    L += ["", "note — runs that were invalid (counted as failures above · by reason): " + " / ".join(invref)]
    L += ["", "runs: " + " · ".join(sorted({f"{NAME[k[0]]} {k[1]} = {v['run']}" for k, v in res.items()}))]
    if dropped:
        L += [
            "",
            "dropped old results (scenes outside the training set, removed from L0): "
            + " · ".join(f"{NAME[k[0]]} {k[1]} {k[2]} {v}" for k, v in sorted(dropped.items())),
        ]
    if len(tsok) == len(res) and res:
        L += [
            "",
            f"train-set check = passed ({len(tsok)} lists · L0 ⊆ keys · L1 ∩ keys = 0 · keys = "
            "training-set episodes.jsonl "
            "(e5_source.clip, look) · train_keys.py · " + " · ".join(v[1] for v in KEYS.values()) + ")",
        ]
    if rep:
        L += [
            "",
            "superseded (an unjudged scene judged by a later batch): "
            + " · ".join(f"{NAME[k[0]]} {k[1]} {k[2]} {v}" for k, v in sorted(rep.items())),
        ]
    if SUPN:  # explicit supersession (the superseded batch tag is recorded as 'superseded: <old tag>')
        L += [
            "",
            "explicit supersession (spec supersedes column — a new batch beats the old batch of the same scene): "
            + " · ".join(
                f"{NAME[k[0]]} {k[1]} {k[2]} {k[3]} scenes {v} (superseded: {k[4]})" for k, v in sorted(SUPN.items())
            ),
        ]
    if XREF:
        L += [
            "",
            "note — outside the list "
            + " · ".join(XREF)
            + " (not counted · reason: old scenes the L1 loop submitted and "
            "that later left the list — recorded in <task>_sent.txt · in neither numerator nor denominator)",
        ]
    if notes:
        L += ["", "unfinished · mismatches: " + " / ".join(notes)]


def _md_l2(L, LD, LIST, NAME, WHOS, RES2, TS2, NOTES2):
    """Markdown L2 section (appended to L); returns the L2 list sizes."""
    # L2 (unseen recordings, original placement): same judge, same counting, separate lines — list <task>_l2.txt
    # ('#' header = rule)
    L2N = {
        tk: len([x for x in read_lines(f"{LD}/{LIST[tk]}_l2.txt") if x.strip() and not x.startswith("#")])
        if os.path.exists(f"{LD}/{LIST[tk]}_l2.txt")
        else 0
        for tk in ("lift", "pour", "sweep")
    }
    if any(L2N.values()):
        L += [
            "",
            "## L2 — unseen recordings · original placement "
            f"({' · '.join(f'{NAME[t]} {n}' for t, n in L2N.items())} scenes per task "
            "· same judge · same count)",
            "",
            "| task | run | L2 successes/scenes | L2 success clips / listed clips | invalid (counted as failures) |",
            "|---|---|---|---|---|",
        ]
        for tk in ("lift", "pour", "sweep"):
            for who in WHOS:
                x = RES2.get((tk, who, "L2"))
                c2 = (
                    (
                        f"{x['ok']}/{x['n']}"
                        + ("" if x["got"] == x["n"] else f" (results {x['got']})")
                        + (f" (unjudged {sum(x['oth'].values())})" if x["oth"] else "")
                    )
                    if x
                    else "not yet"
                )
                L.append(
                    f"| {NAME[tk]} | {who} | {c2} | "
                    f"{(str(x['okc']) + '/' + str(x['clips'])) if x else 'not yet'} | "
                    f"{(' · '.join(f'{k} {v}' for k, v in sorted(x['inv'].items())) or '0') if x else '—'} "
                    "|"
                )
        if TS2:
            L += [
                "",
                f"L2 train-set check = passed {len(TS2)} cells (per recording · training recordings ∩ L2 "
                "recordings = 0)",
            ]
        if NOTES2:
            L += ["", "L2 unfinished · mismatches: " + " / ".join(NOTES2)]
    return L2N


def _write_json(
    SPEC, OUT, state, final, cells_ok, NEED, RULE, XREF, notes, LD, L2N, NOTES2, TS2, RES2, WHOS, NAME, res, SUPN
):
    """<out name>.json — the same cell values as the markdown table (the machine-readable copy)."""

    def _cell(x):
        return (
            None
            if not x
            else {
                "ok": x["ok"],
                "n": x["n"],
                "got": x["got"],
                "unjudged": sum(x["oth"].values()),
                "okc": x["okc"],
                "clips": x["clips"],
                "invalid": dict(sorted(x["inv"].items())),
            }
        )

    JD = {
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "experiments.sim_eval.report.dpf_summary",
        "spec": SPEC,
        "md": OUT,
        "state": state,
        "final": final,
        "cells_ok": cells_ok,
        "need": NEED,
        "rule": RULE,
        "outside_list": XREF,
        "notes": notes,
        "lists_sha256": Path(f"{LD}/lists.sha256").read_text().split("\n")
        if os.path.exists(f"{LD}/lists.sha256")
        else [],
        "l2": {
            "lists": L2N,
            "notes": NOTES2,
            "train_check_ok": len(TS2),
            "done": bool(any(L2N.values()))
            and all((tk, w, "L2") in RES2 for tk in ("lift", "pour", "sweep") if L2N[tk] for w in WHOS)
            and not NOTES2,
        },
        "rows": [
            {
                "task": NAME[tk],
                "who": who,
                "L0": _cell(res.get((tk, who, "L0"))),
                "L1": _cell(res.get((tk, who, "L1"))),
                "L2": _cell(RES2.get((tk, who, "L2"))),
            }
            for tk in ("lift", "pour", "sweep")
            for who in WHOS
        ],
    }
    JD["superseded"] = [
        {"task": k[0], "who": k[1], "level": k[2], "new_tag": k[3], "old_tag": k[4], "scenes": v}
        for k, v in sorted(SUPN.items())
    ]
    JP = os.path.splitext(OUT)[0] + ".json"
    Path(JP + ".tmp").write_text(json.dumps(JD, ensure_ascii=False, indent=1))
    os.replace(JP + ".tmp", JP)


def _write_unjudged(OUT, UNJ, UNJ2):
    """unjudged.tsv (and unjudged_l2.tsv) next to the markdown output."""
    UF = os.path.join(os.path.dirname(OUT), "unjudged.tsv")
    ut = f"{UF}.{os.getpid()}.tmp"
    with open(ut, "w") as fh:
        fh.write("task\twho\trun\tlevel\tscene\tverdict\tjson\n")
        fh.writelines("\t".join(r) + "\n" for r in sorted(UNJ))
    os.replace(ut, UF)
    if UNJ2 or os.path.exists(os.path.join(os.path.dirname(OUT), "unjudged_l2.tsv")):
        U2 = os.path.join(os.path.dirname(OUT), "unjudged_l2.tsv")
        u2 = f"{U2}.{os.getpid()}.tmp"
        with open(u2, "w") as fh:
            fh.write("task\twho\trun\tlevel\tscene\tverdict\tjson\n")
            fh.writelines("\t".join(r) + "\n" for r in sorted(UNJ2))
        os.replace(u2, U2)


def main(argv=None) -> None:
    """Build the table from the spec; writes <out>.md, <out>.json and unjudged.tsv next to it."""
    a = _parse_args(argv)
    SPEC, OUT, LD = a.spec, a.out, a.lists
    LIST = {"pour": "pour", "sweep": "sweep", "lift": "lift"}
    NAME = {"lift": "P&P", "pour": "Pour", "sweep": "Sweep"}

    parts, SUP, SUPN, rep = _read_spec(SPEC)
    res, notes = {}, []
    WHOS = tuple(
        a.who.split(",")
    )  # SPIDER block: --who spider with its own spec and out dir — the main table never sees it
    # L2 cells are collected apart — L0/L1 counts, state and lines are unaffected (L2 mismatches only go to the L2
    # lines and never block 'final')
    RES2, NOTES2, UNJ2, TS2 = {}, [], [], []
    # unjudged scenes (in the list, judged in no batch) → <out folder>/unjudged.tsv — the L1 loop's resubmit input
    UNJ = []
    KEYS, REMOVED, dropped, tsok, SENT = _read_train_sets(a, LD)
    XREF = _tally(
        parts,
        LD,
        LIST,
        NAME,
        SUP,
        SUPN,
        rep,
        KEYS,
        REMOVED,
        SENT,
        res,
        notes,
        RES2,
        NOTES2,
        UNJ2,
        TS2,
        UNJ,
        dropped,
        tsok,
    )
    NEED, cells_ok, final, l0done, tsall, state = _table_state(res, notes, WHOS, tsok, LD, LIST, NAME)
    invref = []
    # the counting rule, one place (md and JSON carry the same text)
    RULE = (
        "successes / listed scenes (target 100) — invalid runs (start jump, blow-up, … removed by the "
        "judge) count as failures "
        "(kin and ours alike) · denominator unchanged · no scene replacement. Only unjudged scenes (judge failed · "
        "unjudgeable · session failed) leave the table 'unfinished'"
    )
    L = _md_table(state, cells_ok, SPEC, RULE, res, WHOS, NAME, invref)
    _md_notes(L, invref, res, NAME, dropped, tsok, KEYS, rep, SUPN, XREF, notes)
    L2N = _md_l2(L, LD, LIST, NAME, WHOS, RES2, TS2, NOTES2)
    Path(OUT + ".tmp").write_text("\n".join(L) + "\n")
    os.replace(OUT + ".tmp", OUT)

    _write_json(
        SPEC, OUT, state, final, cells_ok, NEED, RULE, XREF, notes, LD, L2N, NOTES2, TS2, RES2, WHOS, NAME, res, SUPN
    )
    _write_unjudged(OUT, UNJ, UNJ2)
    print(
        f"[DPF_SUM] {'final' if final else 'unfinished'} {OUT} · train-set check {'passed' if tsall else 'failed'} "
        f"· L0 results {'done' if l0done else 'not yet'} · cells {len(res)}/12 · {state}"
        + ("" if not notes else f" · {' / '.join(notes)[:300]}")
    )


if __name__ == "__main__":
    main()
