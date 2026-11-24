"""The classification rule for the clips the placement judge could not accept, and its one owner: the class groups
(keys = the judge's ``_tally`` vocabulary, order = priority), the reach-family row kinds, the parsers of the judge's
reason text, ``Rows`` (the chunk folders' noconfirm rows) and ``classify_clip`` / ``classify_rows``, which give every
failed clip its class and fine reason. ``judge_fail_detail.py`` checks the judge's vocabulary against these groups,
applies the rule to every failed clip and writes the summary.

Two labels per clip:
  · class — a tally key or the best-candidate key that falls in a group gives that group; priority = GROUPS order.
    With neither: "no best candidate (reach family)" for files written after the judge recorded best candidates,
    "no tally" for files before it, "no rows" when the clip has no row.
  · fine — the best candidate's check name (judge order: error → stretch → speed ratio → jump → table penetration →
    swivel), else the class; for the reach family, the row kind it ended on (REACH_KINDS).
"""

from __future__ import annotations

import collections
import glob
import os
import re
from pathlib import Path

from dexengine.placement.judge.exhaust import (
    REJECT_TALLY,  # head of the rejection tally (the groups read what precedes it)
)

#: Groups — this file owns them. Order = priority. Keys = the judge's ``_tally`` vocabulary (table / torso penetration,
#: guard, early penetration · tracking error, stretch, speed ratio, jump, arm flip, swivel · early stop, frame
#: unreachable · cover, none after policy, no candidate · table footprint). ``vocab()`` stops when the judge's
#: vocabulary and the union of these keys differ.
GROUPS = (
    ("A_collision", "table / torso penetration", ("desk_pen", "torso_pen", "guard", "early_pen")),
    (
        "B_late_noncoll",
        "late non-collision (tracking quality)",
        ("error", "stretch", "speed_ratio", "jump", "arm_flip", "swivel"),
    ),
    ("C_track_reach", "confirmed unreachable", ("early_stop", "frame_infeasible")),
    ("D_gate_pool_cov", "IK pool coverage short", ("coverage", "none_after_polish", "no_candidate")),
    ("E_footprint", "table footprint (xy)", ("desk_xy",)),
)
NOBEST_POST = ("G_post_nobest_reach", "no best candidate (reach family)")
NOTALLY_PRE = ("F_notally_pre", "before the best-candidate record — no tally")
NODATA = ("NODATA", "no chunk folder / rows")
CLASS_ORDER = [g[0] for g in GROUPS] + [NOBEST_POST[0], NOTALLY_PRE[0], NODATA[0]]
CLASS_LABEL = {g[0]: g[1] for g in GROUPS} | dict([NOBEST_POST, NOTALLY_PRE, NODATA])
_KV = re.compile(r"(sparse|confirm|select)/([^=\s]+)=(\d+)")
_BEST = re.compile(r"best_candidate (\S+)")
#: The value inside a best-candidate text — the judge's f-string forms (e.g. ``desk_pen{mm:.0f}mm``).
_VAL = {
    "desk_pen": re.compile(r"desk_pen([\d.]+)mm"),
    "torso_pen": re.compile(r"torso_pen([\d.]+)mm"),
    "error": re.compile(r"error_p\d+=([\d.]+)mm"),
    "swivel": re.compile(r"swivel([\d.]+)°"),
    "jump": re.compile(r"jump([\d.]+)rad"),
}
#: Clips with neither a tally nor a best candidate (after the boundary): the row KIND is the reason (the judge's
#: ``fno.write`` heads). The first match wins (later judge stages first); an unknown head is kept as ``other:<head>``.
REACH_KINDS = (
    ("final_guard_rejected", "final guard rejected (the same function as the bake guard)"),
    ("final_desk_check_rejected", "final table check rejected (left after the pivot change)"),
    ("v2_port_guard_rejected", "v2 transplant guard rejected"),
    ("TRUNCATED", "budget reached (TRUNCATED — not a proof)"),
    ("EXHAUSTED", "exhausted (EXHAUSTED — no candidate reached a check)"),
    ("no_place_spot", "no place to set the object"),
    ("no_receiver_spot", "no place for the receiving object"),
    ("no_volume", "no volume (IK pool common cover 0 — no anchor candidate)"),
    ("confirm_failed", "confirm failed (tally 0)"),
)


def _keyof(k: str, keys: tuple) -> str:
    for key in sorted(keys, key=len, reverse=True):
        if k.startswith(key):
            return key
    return "other:" + k[:10]


class Rows:
    """The noconfirm rows of the chunk folders — read once per tag."""

    def __init__(self, stage2: Path, t_fix: int):
        self.stage2, self.t_fix, self.cache = stage2, t_fix, {}

    def get(self, tag: str) -> dict:
        if tag not in self.cache:
            d = collections.defaultdict(list)
            for nf in sorted(glob.glob(str(self.stage2 / tag / "p*" / "noconfirm_*.tsv"))):
                post = os.path.getmtime(nf) > self.t_fix
                with open(nf, encoding="utf-8", errors="replace") as fh:
                    for ln in fh:
                        p = ln.rstrip("\n").split("\t")
                        if len(p) >= 3:
                            d[p[0]].append((p[1], p[2], post))
            self.cache[tag] = d
        return self.cache[tag]


def classify_clip(rr: list, keys: tuple) -> dict:
    """One clip's noconfirm rows → class · fine · best candidate · tally."""
    if not rr:
        return {"cls": NODATA[0], "fine": NODATA[0], "best": "", "best_key": "", "val": None, "tally": "", "rej": ""}
    cnt, rej = collections.Counter(), collections.Counter()
    best = ""
    for _kind, txt, _p in rr:
        txt, _, tail = txt.partition(REJECT_TALLY)  # the rejection tally is recorded apart (no effect on class/fine)
        rej.update({k: int(n) for k, _, n in (t.rpartition("=") for t in tail.split()) if n.isdigit()})
        for stg, k, n in _KV.findall(txt.split("best_candidate")[0]):
            cnt[f"{stg}/{_keyof(k, keys)}"] += int(n)
        m = _BEST.search(txt)
        if m:
            best = m.group(1)  # the later row wins — the judge's _BEST never drops in rank within a clip
    post = any(x[2] for x in rr)
    have = {k.split("/", 1)[1] for k in cnt}
    bk = _keyof(best.split(":", 1)[1], keys) if ":" in best else ""
    cls = None
    for gid, _lab, ks in GROUPS:
        if have & set(ks) or bk in ks:
            cls = gid
            break
    if cls is None:
        cls = NOBEST_POST[0] if post else NOTALLY_PRE[0]
    val = float(m.group(1)) if (m := _VAL[bk].search(best) if bk in _VAL else None) else None
    fine = f"best:{bk}" if bk and not bk.startswith("other:") else cls
    last = rr[-1][1][:120]
    if cls == NOBEST_POST[0]:  # reach family — the row kind it ended on (class unchanged, fine split)
        heads = [(k, t) for k, t, _p in rr]
        fine = next(
            (f"reach:{key}" for key, _lab in REACH_KINDS if any(k == key or t.startswith(key) for k, t in heads)),
            "reach:other:" + rr[-1][1][:12],
        )
    return {
        "cls": cls,
        "fine": fine,
        "best": best,
        "best_key": bk,
        "val": val,
        "post": post,
        "last": last,
        **{n: " ".join(f"{k}={v}" for k, v in c_.most_common()) for n, c_ in (("tally", cnt), ("rej", rej))},
    }


def classify_rows(status_rows, OTHER_STATES, FAIL_STATES, rows, keys):
    """Per axis the class and fine-reason counts, per key its clips, and the depths and values of the failed clips."""
    per_axis = collections.defaultdict(
        lambda: {
            "total": 0,
            "status": collections.Counter(),
            "cls": collections.Counter(),
            "fine": collections.Counter(),
        }
    )
    clips: dict[str, list] = collections.defaultdict(list)
    depths = collections.defaultdict(list)
    vals = collections.defaultdict(list)
    for ax, clip, st, tag, note in status_rows:
        A = per_axis[ax]
        A["total"] += 1
        A["status"][st] += 1
        if st in OTHER_STATES:
            A["fine"][st] += 1
            clips[f"{ax}|{st}"].append([clip, st, tag, note[:120]])
            continue
        if st not in FAIL_STATES:
            continue
        c = classify_clip(rows.get(tag).get(clip, []) if tag else [], keys)
        A["cls"][c["cls"]] += 1
        A["fine"][c["fine"]] += 1
        why = (
            c["best"] or c["tally"][:120] or c["rej"][:120] or c.get("last", "")
        )  # an exhaust with no check: its tally
        clips[f"{ax}|{c['fine']}"].append([clip, st, tag, why])
        if c["best_key"] in _VAL and c["val"] is not None:
            vals[c["best_key"]].append(c["val"])
            if c["best_key"] == "desk_pen":
                depths[ax].append(c["val"])
    return per_axis, clips, depths, vals
