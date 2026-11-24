"""Judged values of the DP final-checkpoint evaluation and the batch-overlap rule — one owner, dependency free.

JV = the judged verdict values (success · fail · invalid, as stored in the judge records). Any other verdict
(session failed · judge failed · unjudgeable …) is unjudged: it is counted as 'unfinished' and resubmitted by the L1
loop.
Reader: dpf_summary.py.
Batch overlap (the same scene in two batches of one run and level): explicit supersession first — the spec row's
supersedes column (comma list of old batch tags) makes that batch win. Without it, a judged batch beats an unjudged
one; two judged or two unjudged batches raise. Batches naming each other raise. Specs without a supersedes column
read as before."""

import csv
import os
import re

JV_SUCCESS, JV_FAIL, JV_INVALID = "success", "fail", "invalid"  # = levels.SUCCESS · FAIL · INVALID
JV = (JV_SUCCESS, JV_FAIL, JV_INVALID)


def tag_of(json_path: str) -> str:
    """Result JSON path → batch tag (the tag of its <run>_<6-digit step>[_<tag>] folder; '' when none)."""
    b = os.path.basename(os.path.dirname(json_path))
    m = re.search(r"_\d{6}(?:_(.+))?$", b)
    if not m:
        raise ValueError(f"batch folder name {b!r} is not of the form <run>_<step>[_<tag>]")
    return m.group(1) or ""


def supersedes(spec_path: str) -> dict:
    """Spec → {(run, level, new tag): {old tags…}}; rows with an empty supersedes column give nothing."""
    out = {}
    with open(spec_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            v = (r.get("supersedes") or "").strip()
            if v:
                old = {x.strip() for x in v.split(",") if x.strip()}
                k = (r["run"], r["level"], tag_of(r["json"]))
                if k in out and out[k] != old:
                    raise ValueError(f"spec {spec_path}: two supersedes values for {k}: {out[k]} vs {old}")
                out[k] = old
    return out


def overlap(o_verdict: str, o_json: str, s_verdict: str, s_json: str, run: str, lv: str, sup: dict) -> tuple:
    """The same scene in two batches (o = read first, s = read now) → ('old'|'new', superseded old tag | None).
    Without explicit supersession the tag is None (a judged batch replaces an unjudged one). Conflicts raise
    ValueError."""
    to, ts = tag_of(o_json), tag_of(s_json)
    so, ss = sup.get((run, lv, to), set()), sup.get((run, lv, ts), set())
    if to in ss and ts in so:
        raise ValueError(f"batches {to} and {ts} are recorded as superseding each other")
    if to in ss:
        return "new", to
    if ts in so:
        return "old", ts
    ja, jb = o_verdict in JV, s_verdict in JV
    if ja == jb:
        raise ValueError(
            f"batch overlap ({o_json} {o_verdict} vs {s_json} {s_verdict}) — no explicit supersession "
            "(spec supersedes column)"
        )
    return ("old" if ja else "new"), None
