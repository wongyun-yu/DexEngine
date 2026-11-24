"""Source label, case and hands of one clip record (text.sources.record): the per-dataset label readers (SOURCE_RULES;
an unknown dataset has no rule), the stored case under the current handover gate and the active hands."""

from __future__ import annotations

import re


def clip_head(clip: str) -> str:
    return clip.partition("__")[0]


# ───────────────────── source label per dataset ─────────────────────
def src_taco(ti: dict, clip: str) -> dict:
    head = clip_head(clip)
    parts = head.split("_")
    f = {"verb": parts[0].replace("-", " "), "tool": parts[1], "target": parts[2]} if len(parts) == 3 else {}
    return {"kind": "triplet", "text": head, "fields": f}


def src_oakink2(ti: dict, clip: str) -> dict:
    r = ti.get("reference") or {}
    if not r.get("desc"):
        return {"kind": "none", "text": None, "fields": {"why": f"reference.source={r.get('source')}"}}
    return {
        "kind": "sentence",
        "text": r["desc"],
        "fields": {"primitive": r.get("primitive_key"), "active_hands": r.get("active_hands")},
    }


def src_h2o(ti: dict, clip: str) -> dict:
    acts = sorted((ti.get("reference") or {}).get("actions") or [], key=lambda x: -float(x.get("overlap_s") or 0))
    if not acts:
        return {"kind": "none", "text": None, "fields": {}}
    return {
        "kind": "action_label",
        "text": acts[0]["action"],
        "fields": {"actions": [{"action": x["action"], "overlap_s": x.get("overlap_s")} for x in acts]},
    }


def src_label(kind: str):
    def f(ti: dict, clip: str) -> dict:
        r = ti.get("reference") or {}
        lab = r.get("label")
        return {"kind": kind if lab else "none", "text": lab, "fields": {"clip_head": clip_head(clip)}}

    return f


def src_dexcanvas(ti: dict, clip: str) -> dict:
    tail = clip.partition("__")[2]
    m = re.match(r"([a-z]+)(\d*)_(\d+)", tail)
    f = {"object": m.group(1), "variant": m.group(2), "type_code": m.group(3)} if m else {"raw": tail}
    return {"kind": "object_and_type_code", "text": tail, "fields": f}


def src_none(ti: dict, clip: str) -> dict:
    return {"kind": "none", "text": None, "fields": {}}


SOURCE_RULES = {
    "taco": src_taco,
    "oakink2": src_oakink2,
    "h2o": src_h2o,
    "grab": src_label("intent"),
    "arctic": src_label("intent"),
    "hrdexdb": src_label("object_names"),
    "dexcanvas": src_dexcanvas,
    "hocap": src_none,
    "hot3d": src_none,
}


# ───────────────────── case (handover) ─────────────────────
def case_now(ti: dict) -> tuple[str | None, list[dict]]:
    """Stored case (task_info.case) -> (case name under the current not-a-handover gate data.cases
    dexengine.data.cases.gate.regate, remaining handovers [{unit, from, to, time_s}]). Handover window = receiving
    hand's grasp start .. giving hand's release (case.grasp_s); the window goes to the prompt through the record field
    handover."""
    from dexengine.data.cases import gate as CGATE
    from dexengine.data.cases import hand_object_assign as H

    case, objs = ti.get("case") or {}, ti.get("objects") or []
    if not case.get("case"):
        return case.get("case"), []
    name, g = CGATE.regate(case, H.units_of(objs)[1], objs)
    out = []
    if "HANDOVER" in name:
        share = case.get("share") or {}
        for u in case.get("used") or []:
            r = (case.get("grasp_s") or {}).get(u) or {}
            if (
                (share.get(u) or {}).get("kind") != "handover"
                or CGATE.is_out(g.get(u, []))
                or not (r.get("right") and r.get("left"))
            ):
                continue
            gv, rv = ("right", "left") if r["right"][0][0] < r["left"][0][0] else ("left", "right")
            out.append(dict(unit=u, **{"from": f"{gv} hand", "to": f"{rv} hand"}, time_s=[r[rv][0][0], r[gv][-1][1]]))
    return name, out


# ───────────────────── hands ─────────────────────
def hands_of(ti: dict) -> str:
    act = sorted(ti.get("active_sides") or [])
    return {("right",): "right", ("left",): "left", ("left", "right"): "both"}.get(tuple(act), "unspecified")
