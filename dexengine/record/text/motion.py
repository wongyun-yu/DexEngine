"""Measured motion of a text input record.

1. Motion summary (motion_summary): per object net displacement, lift, rotation and "ends near"; per articulated part
   (drawer, lid, door) the joint angle range. Joint units come from the joint name tail (JOINT_UNITS: hinge = rad ->
   degrees, slide = m -> cm); an unknown tail dies.
2. Hand-contact facts (attach_facts -> motion.hand_contact): per object, aggregated over every shipped episode of the
   clip, read from the facts tables in env COLLECT_TEXT_FACTS_DIR (<dataset>.jsonl, one row per episode: per-slot
   lift, travel, rotation, joint range and hand x slot contact intervals). The prompt shows them and rules R22-R31
   read them. A text serves all episodes of its source clip (original and placement variants), so per object:
     episodes · lift_cm · move_cm · rot_deg · joint [min, max] (joint in the record's joint unit)
     lifted all|some|none (episodes with a measured lift onset, lift >= check.LIFT_CM)
     touched_by (hands touching in every episode) · touched_some (in some episodes only)
     lifted_by (hands touching within [onset - ONSET_S, onset + JOIN_S] of every lifted episode) · lifted_by_some
     held_at_start (hands touching within START_S in every episode)
     handover '<giver>→<receiver>' (every episode) | 'some' — a handover = after the lift, one hand alone >= SOLO_S,
       (only both-hands overlap in between), then the other hand alone >= SOLO_S; a gap where no hand touches is a
       put-down and re-grasp, not a handover
     air_start all|some|none (the final episodes of that key carry support discs: it starts in the air)
     end_xy_cm · end_z_cm [min, max] (end pose - start pose: horizontal distance, height) · end_up all|some|none
       (a hand touches within END_S of the end and the end height >= FACTS_LIFT_CM: still held up at the end)
"""

from __future__ import annotations

import functools
import json
import math
import os
from functools import cache
from pathlib import Path

from dexengine.record import phases as PH
from dexengine.record.text.api import die

MOVE_CM, ROT_DEG = PH.MOVE_M * 100, PH.ROT_DEG  # "moved" thresholds (owner: record.phases)
NEAR_CM = 12.0  # end within this distance of another object's centre = "ends near"
#: joint name tail -> (unit, scale, motion threshold).
JOINT_UNITS = {"hinge": ("deg", math.degrees(1.0), 30.0), "slide": ("cm", 100.0, 5.0)}


def quat_angle(q0, q1) -> float:
    """Angle in degrees between two unit quaternions."""
    d = abs(sum(a * b for a, b in zip(q0, q1, strict=True)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def motion_summary(tree: Path, ti: dict) -> dict | None:
    """Object and joint motion of a tree from 0/trajectory_kinematic.npz (object pose = qpos[:, start:start+7],
    xyz + wxyz); None without a trajectory."""
    import numpy as np

    f = tree / "0" / "trajectory_kinematic.npz"
    if not f.exists():
        return None
    q = np.load(f)["qpos"]
    dt = float(ti.get("ref_dt") or 0.0)
    rows, ends = [], {}
    for o in ti.get("objects") or []:
        s = o.get("qpos_start")
        if s is None:
            continue
        p = q[:, s : s + 3].astype(float)
        r = q[:, s + 3 : s + 7].astype(float)
        disp = (p[-1] - p[0]) * 100
        lift = float((p[:, 2] - p[0, 2]).max() * 100)
        rot_end = quat_angle(r[0], r[-1])
        rot_max = max(quat_angle(r[0], x) for x in r[:: max(1, len(r) // 50)])
        moved = float(np.linalg.norm(disp)) > MOVE_CM or rot_max > ROT_DEG or lift > MOVE_CM
        ends[o["obj_id"]] = p[-1]
        rows.append(
            {
                "obj": o["obj_id"],
                "moved": moved,
                "net_cm": [round(float(x), 1) for x in disp],
                "lift_max_cm": round(lift, 1),
                "rot_end_deg": round(rot_end),
                "rot_max_deg": round(rot_max),
            }
        )
    for row in rows:
        if not row["moved"]:
            continue
        me = ends[row["obj"]]
        near = sorted((float(np.linalg.norm(me - v)) * 100, k) for k, v in ends.items() if k != row["obj"])
        if near and near[0][0] < NEAR_CM:
            row["ends_near"] = {"obj": near[0][1], "cm": round(near[0][0], 1)}
    js = joint_motion(ti, q)
    return dict(duration_s=round(len(q) * dt, 2) if dt else None, objects=rows, **({"joints": js} if js else {}))


def joint_motion(ti: dict, q) -> list[dict]:
    """Per articulated part {joint, kind, range, net, unit, moved, hand} from the qpos_layout rows of kind 'joint'
    ([] without any). hand = the side that moves the part (the object's articulation.parts.moving_role; None when
    unknown)."""
    side = {}
    for o in ti.get("objects") or []:
        a = o.get("articulation") or {}
        role = (a.get("parts") or {}).get("moving_role") or []
        if a.get("joint"):
            side[a["joint"]] = role[1] if len(role) > 1 and role[1] in ("left", "right") else None
    out = []
    for j in ti.get("qpos_layout") or []:
        if j.get("kind") != "joint":
            continue
        kind = str(j["name"]).rsplit("_", 1)[-1]
        if kind not in JOINT_UNITS:
            raise SystemExit(f"[COLLECT_TEXT_SRC] FAIL unknown joint tail {j['name']!r} — {sorted(JOINT_UNITS)}")
        unit, k, thr = JOINT_UNITS[kind]
        v = [float(x) * k for x in q[:, int(j["start"])]]
        rng = max(v) - min(v)
        out.append(
            {
                "joint": j["name"],
                "kind": kind,
                "range": round(rng, 1),
                "net": round(v[-1] - v[0], 1),
                "unit": unit,
                "moved": rng >= thr,
                "hand": side.get(j["name"]),
            }
        )
    return out


def rec_joints(rec: dict) -> list:
    """Joint motion of a record: its joints field when present with a hand per row, otherwise measured from its
    tree (tree_joints; records of clips without joints carry no joints field), else the stored field."""
    mo = rec.get("motion") or {}
    if _current(mo):
        return list(mo["joints"] or [])
    t = tree_joints(str(rec.get("tree") or ""))
    return list(t) if t is not None else list(mo.get("joints") or [])


def _current(mo: dict) -> bool:
    """The joints field is present and every row has a hand field."""
    return "joints" in mo and all("hand" in j for j in mo["joints"] or [])


@cache
def tree_joints(tree: str) -> tuple:
    """Joint motion measured at a tree folder; None without the tree or its trajectory."""
    import numpy as np

    t = Path(tree)
    if not (t / "task_info.json").is_file() or not (t / "0" / "trajectory_kinematic.npz").is_file():
        return None
    ti = json.loads((t / "task_info.json").read_text(encoding="utf-8"))
    return tuple(joint_motion(ti, np.load(t / "0" / "trajectory_kinematic.npz")["qpos"]))


# ───────────────────────── hand-contact facts ─────────────────────────
FACTS_ENV = "COLLECT_TEXT_FACTS_DIR"  # input path; read in the sources child process (process boundary)
#: Lift threshold the facts tables were built with (onset times are measured at it): must equal check.LIFT_CM.
FACTS_LIFT_CM = 2.0
#: Time grid and windows (s): hands at the lift onset = [onset - ONSET_S, onset + JOIN_S].
STEP_S, SOLO_S, ONSET_S, JOIN_S, START_S, END_S = 0.02, 0.3, 0.3, 1.0, 0.1, 0.1
HANDS = ("left", "right")


@functools.cache
def _facts_index(fdir: str, ds: str) -> dict:
    """{text key: [per-episode facts {obj_id: object facts}]} of one dataset's facts table."""
    from dexengine.record.text import order as ORD

    f = Path(fdir) / f"{ds}.jsonl"
    if not f.is_file():
        die(f"facts table missing: {f} (the dataset file under {FACTS_ENV})")
    out: dict = {}
    for ln in f.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            k = ORD.ep_clip({"selection": {"source_clip": r.get("source_clip")}, "clip": r["clip"]})
            out.setdefault(k, []).append(_facts_episode(r))
    return out


def facts_dir() -> str:
    """The facts table folder (env FACTS_ENV, required)."""
    from dexengine.record.text import check as CHK

    d = os.environ.get(FACTS_ENV)
    if not d or not Path(d).is_dir():
        die(f"{FACTS_ENV} is not a folder ({d!r}): name the facts table folder explicitly (no default)")
    if CHK.LIFT_CM != FACTS_LIFT_CM:
        die(f"text lift threshold {CHK.LIFT_CM} != facts threshold {FACTS_LIFT_CM}: rebuild the facts tables")
    return d


def _runs(st: list[int]) -> list[tuple[int, float, float]]:
    """Run-length encoding of a state list: [(state, start_s, duration_s)]."""
    out, i = [], 0
    while i < len(st):
        j = i
        while j + 1 < len(st) and st[j + 1] == st[i]:
            j += 1
        out.append((st[i], i * STEP_S, (j - i + 1) * STEP_S))
        i = j + 1
    return out


def _handover_pairs(held: dict, lift_on: float | None) -> list[str]:
    """After the lift: one hand alone -> (overlap) -> the other hand alone, as '<giver>→<receiver>'."""
    if lift_on is None:
        return []
    n = len(held["left"])
    st = [
        3 if held["left"][i] and held["right"][i] else 1 if held["left"][i] else 2 if held["right"][i] else 0
        for i in range(n)
    ]
    rs, got = _runs(st), []
    for a, (s, _, d) in enumerate(rs):
        if s not in (1, 2) or d < SOLO_S:
            continue
        b = a + 1
        while b < len(rs) and rs[b][0] == 3:
            b += 1
        if b < len(rs) and rs[b][0] in (1, 2) and rs[b][0] != s and rs[b][2] >= SOLO_S and rs[b][1] >= lift_on:
            got.append(f"{HANDS[s - 1]}→{HANDS[rs[b][0] - 1]}")
    return got


def _facts_episode(r: dict) -> dict:
    """One facts row -> {obj_id: facts} (two slots on the same object — a shared object — merge their contacts)."""
    n = int(round(float(r["duration_s"]) / STEP_S)) + 1
    out: dict = {}
    for s, v in r["slots"].items():
        o = out.setdefault(v["obj_id"], dict(v, held={h: [False] * n for h in HANDS}))
        for c in r["contacts"]:
            if c["slot"] == s:
                a, b = int(round(c["start_s"] / STEP_S)), int(round(c["end_s"] / STEP_S))
                for i in range(max(0, a), min(n, b + 1)):
                    o["held"][c["hand"]][i] = True
    for o in out.values():
        h, lo = o.pop("held"), o["lift_on_s"]
        o["hands"] = [x for x in HANDS if any(h[x])]
        w = (int(round((lo - ONSET_S) / STEP_S)), int(round((lo + JOIN_S) / STEP_S))) if lo is not None else (0, -1)
        o["lift_hands"] = [x for x in HANDS if any(h[x][max(0, w[0]) : w[1] + 1])]
        o["start_hands"] = [x for x in HANDS if any(h[x][: int(round(START_S / STEP_S)) + 1])]
        o["end_hands"] = [x for x in HANDS if any(h[x][-(int(round(END_S / STEP_S)) + 1) :])]
        o["handover"] = _handover_pairs(h, lo)
    return out


def _span(vals: list) -> list | None:
    vals = [v for v in vals if v is not None]
    return [min(vals), max(vals)] if vals else None


def _tri(flags: list[bool]) -> str:
    return "all" if all(flags) else "some" if any(flags) else "none"


def aggregate_facts(eps: list[dict], name_of: dict, joint_scale: float | None, air: str) -> list[dict]:
    """Per-episode facts -> the record's hand_contact rows (objects with a record name only, in name order)."""
    out = []
    for oid, name in name_of.items():
        vs = [e[oid] for e in eps if oid in e]
        if not vs:
            continue
        lifted = [v["lift_on_s"] is not None for v in vs]
        every = [set(v["hands"]) for v in vs]
        lb = [set(v["lift_hands"]) for v, f in zip(vs, lifted, strict=True) if f]
        ho = [v["handover"][0] if v["handover"] else None for v in vs]
        row = {
            "obj": name,
            "episodes": len(vs),
            "lifted": _tri(lifted),
            "lift_cm": _span([v["lift_max_cm"] for v in vs]),
            "move_cm": _span([v["xy_max_cm"] for v in vs]),
            "rot_deg": _span([v["rot_max_deg"] for v in vs]),
            "touched_by": sorted(set.intersection(*every)),
            "touched_some": sorted(set.union(*every) - set.intersection(*every)),
            "held_at_start": sorted(set.intersection(*[set(v["start_hands"]) for v in vs])),
            "air_start": air,
            "end_xy_cm": _span([v["xy_end_cm"] for v in vs]),
            "end_z_cm": _span([v["z_end_cm"] for v in vs]),
            "end_up": _tri([bool(v["end_hands"]) and v["z_end_cm"] >= FACTS_LIFT_CM for v in vs]),
        }
        if lb:
            row["lifted_by"] = sorted(set.intersection(*lb))
            row["lifted_by_some"] = sorted(set.union(*lb) - set.intersection(*lb))
        if all(ho) and len(set(ho)) == 1:
            row["handover"] = ho[0]
        elif any(ho):
            row["handover"] = "some"
        if joint_scale is not None and any(v["joint_range"] is not None for v in vs):
            row["joint"] = _span(
                [round(v["joint_range"] * joint_scale, 1) if v["joint_range"] is not None else None for v in vs]
            )
        out.append(row)
    return out


@functools.cache
def _air_index(roots: tuple[str, ...]) -> dict:
    """(dataset, text key) -> per final episode: does it carry support discs (supports)."""
    from dexengine.record.text import order as ORD

    out: dict = {}
    for root in roots:
        f = Path(root) / "meta" / "episodes.jsonl"
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                e = json.loads(ln)
                out.setdefault((Path(root).parts[-3], ORD.ep_clip(e)), []).append(bool(e.get("supports")))
    return out


def attach_facts(rec: dict, finals) -> dict:
    """Copy of the record with motion.hand_contact (unchanged when the facts table has no row for the clip; the
    joint column only when the record has one joint kind)."""
    eps = _facts_index(facts_dir(), rec["dataset"]).get(rec["clip"])
    if not eps:
        return rec
    joints = (rec.get("motion") or {}).get("joints") or []
    kinds = {j.get("kind") for j in joints}  # facts joint values are in degrees -> convert to the record's unit
    scale = JOINT_UNITS[next(iter(kinds))][1] / JOINT_UNITS["hinge"][1] if len(kinds) == 1 else None
    air = _tri(_air_index(tuple(sorted(str(f) for f in finals))).get((rec["dataset"], rec["clip"])) or [False])
    name_of = {str(o.get("id")): o["name"] for o in rec.get("objects") or [] if o.get("id") is not None}
    mo = dict(rec.get("motion") or {})
    mo["hand_contact"] = aggregate_facts(eps, name_of, scale, air)
    return dict(rec, motion=mo)


def inputs(rec: dict, finals) -> dict:
    """Finish the measured fields of a text input record: drop objects absent from the render
    (vocab.drop_absent), then attach the hand-contact facts."""
    from dexengine.record.text import vocab as CN

    return attach_facts(CN.drop_absent(rec, finals), finals)
