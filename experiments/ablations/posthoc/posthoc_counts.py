#!/usr/bin/env python3
"""Post-hoc arm IK of a floating-hand wrist path — per-clip outcome of the judge on the production cell, as counts.

Reads one judge run of `scripts/stage2_ground.sh --judge-only` made with `--place-strategy adopted` (the cell is the
production cell; only the wrist path differs) and classifies every clip:

  adopted                         the judge accepted the cell (arm solve + its limits + exact-mesh guard + desk xy)
  solver_ok_post_check_rejected   rejected only by the exact-mesh guard / desk-xy check, which run after the arm solve
                                  passed its own limits
  solver_failed                   the full-frame arm solve failed one of its own limits (reason in reject_reasons)
  gate_failed                     rejected by the sparse whole-clip gate before the full arm solve

`solver_ok` = adopted + solver_ok_post_check_rejected. The rule is the one of the (c-full) run
(cfull_posthoc_counts.json) — this file is its parametrized form, so two floating
sources are counted by the same code. `--cross name=<tsv>:<column>` adds a per-clip cross-table against another
per-clip class table (same clip set required).

  python -m experiments.ablations.posthoc.posthoc_counts --run-dir <runs/<name>/<tag>> --variant spider --title "..."
  --what "..." --flags "..." \
      --commit <sha> --branch <b> --machine "<m>" --started "<t>" --ended "<t>" --out <prefix> \
      [--cross canonical=<b2 tsv>:v0 --cross ours_floating=<cfull tsv>:cf]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The judge writes its reject tally in record words (parsed here byte for byte): a noconfirm detail holds
# "reject_tally <phase>/<reason>=<n> …"; phase "confirm" = the full-frame confirm (arm solve), any other phase = the
# sparse gate.
POST = {
    "guard",
    "desk_xy",
}  # confirm reasons that run after the arm solve passed its limits: exact-mesh guard / desk xy
CLASSES = ("adopted", "solver_ok_post_check_rejected", "solver_failed", "gate_failed")
AD_RE = re.compile(r"\[PLACE_STRATEGY\] (\S+) strategy=adopted .*adopted=([01])")
TA_RE = re.compile(
    r"\[TRACK_ABLATE_CLIP\] (\S+) off=(\S*) graft_tried=(\d+) graft_taken=(\d+) ball_tried=(\d+) "
    r"ball_taken=(\d+)"
)


def classify(run_dir: Path, combo: str):
    clips = [ln for ln in (run_dir / "clips_judge.txt").read_text(encoding="utf-8").splitlines() if ln]
    log = (run_dir / "judge.log").read_text(encoding="utf-8", errors="replace").splitlines()
    ad, ta = {}, {}
    for ln in log:
        m = AD_RE.search(ln)
        if m:
            ad[m.group(1)] = int(m.group(2))
        m = TA_RE.search(ln)
        if m:
            ta[m.group(1)] = tuple(int(x) for x in m.groups()[2:])  # the last line of the clip wins
    head = [ln[ln.index("[TRACK_ABLATE]") :] for ln in log if "[TRACK_ABLATE] " in ln][:1]
    nc = {}
    ncf = run_dir / "p0" / f"noconfirm_{combo}.tsv"
    for ln in ncf.read_text(encoding="utf-8").splitlines() if ncf.is_file() else []:
        c, st, det = ln.split("\t")[:3]
        tl = det.split("reject_tally ")[1] if "reject_tally " in det else ""
        nc[c] = (st, {k: int(n) for k, n in re.findall(r"(\S+?/\S+?)=(\d+)", tl)}, det)
    if set(ad) != set(clips):
        raise SystemExit(f"[POSTHOC_COUNTS] marker set {len(ad)} != clips {len(clips)}")
    if {c for c in clips if not ad[c]} != set(nc):
        raise SystemExit("[POSTHOC_COUNTS] not-adopted set != noconfirm list")
    cls, reasons, why = {}, {}, {}
    for c in clips:
        if ad[c]:
            cls[c], why[c] = "adopted", ""
            continue
        _, tl, det = nc[c]
        if not tl:
            raise SystemExit(f"[POSTHOC_COUNTS] {c}: no reject tally: {det}")
        ph = {k.split("/")[0] for k in tl}
        rs = {k.split("/")[1] for k in tl}
        if ph == {"confirm"} and rs <= POST:
            cls[c] = "solver_ok_post_check_rejected"
        elif ph == {"confirm"}:
            cls[c] = "solver_failed"
        else:
            cls[c] = "gate_failed"
        why[c] = ";".join(tl)
        for k in tl:
            reasons[k] = reasons.get(k, 0) + 1
    return clips, cls, why, reasons, ta, head


def _parse_args():
    """Parse the command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--combo", default="rby1_sharpa_wave")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--what", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--started", required=True)
    ap.add_argument("--ended", required=True)
    ap.add_argument("--out", required=True, help="output prefix (<prefix>.json · <prefix>.tsv)")
    ap.add_argument("--cross", action="append", default=[], help="name=<per-clip tsv>:<class column> (repeatable)")
    a = ap.parse_args()
    return a


def _variant_row(cls, ta, clips, a, d, head, reasons):
    """Class counts and track-marker totals of this variant."""
    n = {k: sum(x == k for x in cls.values()) for k in CLASSES}
    g = [ta.get(c) for c in clips]
    row = dict(
        variant=a.variant,
        tag=d.name,
        what=a.what,
        flags=a.flags,
        header=head,
        clips=len(clips),
        **n,
        solver_ok=n["adopted"] + n["solver_ok_post_check_rejected"],
        reject_reasons=reasons,
        clips_with_track_marker=sum(x is not None for x in g),
        graft_tried=sum(x[0] for x in g if x),
        graft_taken=sum(x[1] for x in g if x),
        ball_tried=sum(x[2] for x in g if x),
        ball_taken=sum(x[3] for x in g if x),
        clips_graft_taken=sum(bool(x and x[1]) for x in g),
        clips_ball_taken=sum(bool(x and x[3]) for x in g),
    )
    return n, row


def _cross_tables(a, clips, cls):
    """Per-clip class of each --cross run against this one (contingency table and both/only/neither counts)."""
    other, cross = {}, {}
    for spec in a.cross:
        name, rest = spec.split("=", 1)
        path, col = rest.rsplit(":", 1)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        hd = lines[0].split("\t")
        if col not in hd:
            raise SystemExit(f"[POSTHOC_COUNTS] --cross {name}: no column {col} in {path}")
        i = hd.index(col)
        o = {ln.split("\t")[0]: ln.split("\t")[i] for ln in lines[1:] if ln}
        if set(o) != set(clips):
            raise SystemExit(f"[POSTHOC_COUNTS] --cross {name}: clip set differs ({len(o)} vs {len(clips)})")
        bad = set(o.values()) - set(CLASSES)
        if bad:
            raise SystemExit(f"[POSTHOC_COUNTS] --cross {name}: unknown classes {sorted(bad)}")
        other[name] = o
        t = {}
        for c in clips:
            k = f"{name}:{o[c]} | {a.variant}:{cls[c]}"
            t[k] = t.get(k, 0) + 1

        def ok(x):
            return x in ("adopted", "solver_ok_post_check_rejected")

        cross[name] = {
            "source": f"{path}:{col}",
            "table": t,
            "adopted_both": sum(o[c] == "adopted" and cls[c] == "adopted" for c in clips),
            "adopted_only_other": sum(o[c] == "adopted" and cls[c] != "adopted" for c in clips),
            "adopted_only_this": sum(o[c] != "adopted" and cls[c] == "adopted" for c in clips),
            "adopted_neither": sum(o[c] != "adopted" and cls[c] != "adopted" for c in clips),
            "solver_ok_both": sum(ok(o[c]) and ok(cls[c]) for c in clips),
            "solver_ok_only_other": sum(ok(o[c]) and not ok(cls[c]) for c in clips),
            "solver_ok_only_this": sum(not ok(o[c]) and ok(cls[c]) for c in clips),
            "solver_ok_neither": sum(not ok(o[c]) and not ok(cls[c]) for c in clips),
        }
    return cross, other


def _record(a, row, cross):
    """The output record (run identity, definitions, the variant row, the cross tables)."""
    out = {
        "what": a.title,
        "commit": a.commit,
        "branch": a.branch,
        "machine": a.machine,
        "started": a.started,
        "ended": a.ended,
        "defs": {
            "adopted": "the judge accepted the production cell (arm solve + its limits + exact-mesh guard + desk xy)",
            "solver_ok": "adopted + clips rejected only by the exact-mesh guard / desk-xy check, which run "
            "after the arm solve passed",
            "solver_failed": "the full-frame arm solve of the production cell failed one of its own limits "
            "(reason in reject_reasons)",
            "gate_failed": "rejected by the sparse whole-clip gate before the full arm solve",
            "cross": "per-clip class of another run on the same clips (name:class | this:class → clips)",
        },
        "variants": [row],
        "cross": cross,
    }
    return out


def main() -> int:
    a = _parse_args()
    d = Path(a.run_dir)
    clips, cls, why, reasons, ta, head = classify(d, a.combo)
    n, row = _variant_row(cls, ta, clips, a, d, head, reasons)
    cross, other = _cross_tables(a, clips, cls)
    out = _record(a, row, cross)
    pre = Path(a.out)
    pre.with_suffix(".json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(pre.with_suffix(".tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(["clip", a.variant, f"{a.variant}_reasons", *other]) + "\n")
        for c in sorted(clips):
            f.write("\t".join([c, cls[c], why[c], *(other[k][c] for k in other)]) + "\n")
    print(
        f"[POSTHOC_COUNTS] {a.variant} clips={len(clips)} adopted={n['adopted']} solver_ok={row['solver_ok']} "
        f"post={n['solver_ok_post_check_rejected']} solver_failed={n['solver_failed']} gate={n['gate_failed']} "
        f"graft={row['graft_taken']}/{row['graft_tried']} ball={row['ball_taken']}/{row['ball_tried']} "
        f"reasons={reasons}",
        flush=True,
    )
    for k, v in cross.items():
        print(
            f"[POSTHOC_COUNTS] cross {k}: adopted both={v['adopted_both']} only {k}={v['adopted_only_other']} "
            f"only {a.variant}={v['adopted_only_this']} neither={v['adopted_neither']} · solver_ok both="
            f"{v['solver_ok_both']} only {k}={v['solver_ok_only_other']} only {a.variant}={v['solver_ok_only_this']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
