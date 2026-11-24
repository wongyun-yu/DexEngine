"""OakInk2 scenes with two or more merged instances (clips showing two or more articulated objects). Called by the
pairing rule (oakink2.artic_fit.part_clusters) and the converter (oakink2.artic_convert).

Rule: when at most one instance moved its joint in this clip (the single-instance decision
oakink2.artic_fit._one_instance gives two_clusters), that one stays articulated as usual and the others are fixed as one
body in their recorded pose (rigid joint, the fixed_only path: body + all other parts). A single instance's exclusion or
error becomes the scene's reason as is (first by root order).
Exclusion words: multi_moving (two or more instances moved) · multi_fixed_drifts (a fixed instance with a jointed part
lies outside the rigid-joint band at its recorded pose (clip median); band and statistics owned by
oakink2.fixed_merge.rigid) · a fixed instance of attached parts only outside the band gets the word of the
single-instance rigid joint (oakink2.fixed_merge.moved_word). The exclusion messages reach the ledger as written.
Conversion: rigid joints first, the joint last (joint addresses belong to the final model), chained so the tree is
written once · tree record rules.D1g = the joint step's record (else the first rigid joint) + multi (step records) · the
conversion ledger TSV has one line per instance.
Marker [ARTIC_MULTI] <clip> moving <instance|none> · fixed <instances…> (per tree).
"""

from __future__ import annotations

import numpy as np

from dexengine.data.oakink2 import fixed_merge as FM  # noqa: E402 — owner of the rigid-joint band and statistics
from dexengine.data.oakink2 import parts as OP  # noqa: E402
from dexengine.data.oakink2.artic_fit import PartExcluded, _one_instance  # noqa: E402 — the single-instance decision

TAG = "[ARTIC_MULTI]"


def decide(inst: dict[str, list[dict]], q: np.ndarray) -> dict:
    """Merged instances {root: [part objects]} (two or more) → {status: multi, instances: [roots…], moving: decision of
    the moving instance or None, fixed: [fixed_only-shaped decisions…]}, or an exclusion (PartExcluded)."""
    decs = {r: _one_instance(r, v, q) for r, v in sorted(inst.items())}
    moving = [r for r, d in decs.items() if d["status"] == "two_clusters"]
    names = " · ".join(f"{OP.name_of(r)}({len(v)})" for r, v in sorted(inst.items()))
    if len(moving) > 1:
        raise PartExcluded(
            "multi_moving",
            f"merged instances {names} with a moving joint: "
            + " · ".join(OP.name_of(r) for r in moving)
            + " — one articulated body per tree",
        )
    fixed = []
    for r, d in decs.items():
        if r in moving:
            continue
        cl = d["clusters"][:1]  # one_cluster and fixed_only both have one body cluster (jointed part included)
        rec = FM.record(cl[0], q)
        if not FM.rigid(rec):
            why = (
                f"of the merged instances {names}, {OP.name_of(r)} is outside the fixed-joint band at the "
                "recorded pose (clip median)"
            )
            if d["status"] == "fixed_only":  # attached parts only: the same word as a single-instance rigid joint
                raise PartExcluded(FM.moved_word(rec), why)
            raise PartExcluded(
                "multi_fixed_drifts", why + " — the joint did not move, yet it cannot be fixed as one body"
            )
        fixed.append(
            {
                "status": "fixed_only",
                "instance": r,
                "same_grid": d["same_grid"],
                **({"levers": d["levers"]} if d.get("levers") else {}),
                "clusters": cl,
            }
        )
    return {"status": "multi", "instances": list(decs), "moving": decs[moving[0]] if moving else None, "fixed": fixed}


def instances(pc: dict) -> list[dict]:
    """part_clusters decision → instance decisions (the fixed ones, the moving one last); one instance → [that decision]
    ·
    no parts = []."""
    if pc["status"] == "multi":
        return [*pc["fixed"], *([pc["moving"]] if pc["moving"] else [])]
    return [] if pc["status"] == "no_parts" else [pc]


def plans(pc: dict, decide_one) -> dict:
    """Conversion decision: decide_one per instance; if any is not converted, its decision (reason) is the scene's,
    otherwise {multi: [decisions…], moving: root or None, fixed: [roots…]}."""
    out = []
    for p in instances(pc):
        d = decide_one(p)
        if "status" in d:
            return d
        out.append(d)
    return {
        "multi": out,
        "moving": pc["moving"]["instance"] if pc["moving"] else None,
        "fixed": [p["instance"] for p in pc["fixed"]],
    }


def _name(root: str) -> str:
    return f"{OP.name_of(root)}({root})"


def mark(clip: str, dec: dict) -> str:
    return f"{TAG} {clip} moving {_name(dec['moving']) if dec['moving'] else 'none'} · fixed " + " · ".join(
        _name(r) for r in dec["fixed"]
    )


def merged(done: list[dict]) -> dict:
    """Step records → tree row (per tree; count: convert_version.tally): one as is, several = the joint step's record
    (else the first rigid joint) + multi."""
    if len(done) == 1:
        return done[0]
    return {**next((r for r in done if r["kind"] != "fixed"), done[0]), "multi": done}


def stamp(res: dict) -> dict:
    """Tree row → the body of rules.D1g (clip fields dropped, in sub-records too)."""
    return {
        k: ([{kk: vv for kk, vv in x.items() if kk != "clip"} for x in v] if k == "multi" else v)
        for k, v in res.items()
        if k != "clip"
    }


def instance_rows(rows: list[dict]) -> list[dict]:
    """Tree rows → instance rows (a multi row becomes its sub-records); read by the conversion ledger TSV and the
    canonical marker."""
    return [x for r in rows for x in (r.get("multi") or [r])]
