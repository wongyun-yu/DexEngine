"""Oracle gate for zero touched samples of the record stage-1 recorder.

rollout.recorder CollectRecorder.finalize calls oracle_gate when the fingertip filter sensors stay 0 over the whole
rollout. Witness = the
partner-object channels (hand_object_contact view: object = sensor, hand links = filter, a PhysX view independent
of the fingertip sensor wiring) per link max |object force| (all envs, all samples). Gates:
  ORACLE_UNTOUCHED  the object channels are also 0 everywhere: no contact to prove (untouched_ok).
  ORACLE_BUMP       every partner object with force was never touched by the human hand holding it in the human
                    data (contact sidecar GT 0 frames, phases.human_touch): a non-fingertip link bumped an object
                    that is not manipulated. Meta collect.oracle_gate = {rule, channels: {ch: {max_n, links,
                    human_frames}}}.
  ORACLE_NONTIP     a manipulated object the human touched has force, but the witness is 0 at the fingertip link
                    columns (the oracle pair bodies, the links the fingertip sensors cover) over every sample: the
                    fingertips never touched the object, so fingertip sensors 0 is the right reading (non-fingertip
                    links carried the load). The channel record adds tip_links (the fingertip links, all 0) and
                    load_links ([link, max N], largest first).
Anything else (a fingertip link column of the witness has force while the fingertip sensors read 0) is a wiring
fault and dies, naming those fingertip links and values. Unreadable human contact, fingertip columns or link names
(no clip, root, side, sidecar or pair) and a NaN anywhere in the witness (checked before all three gates) also
die. With several clips per session the human contact is summed; the witness is per session (no clip axis).
"""

from __future__ import annotations

import math
from pathlib import Path

from dexengine.record.rollout.recorder import MARK

ORACLE_UNTOUCHED, ORACLE_BUMP, ORACLE_NONTIP = "ORACLE_UNTOUCHED", "ORACLE_UNTOUCHED_BUMP", "ORACLE_NONTIP"


def untouched_ok(wit: dict) -> bool:
    """Partner-object channels exist and are 0 over all links and samples (ORACLE_UNTOUCHED); {} is not."""
    return bool(wit) and all(v == 0.0 for v in wit.values())


def oracle_nontip(v: list, tips: list | None, names: list | None, head: str) -> dict:
    """One human-touched manipulated-object channel: v = witness max per link, tips = fingertip link columns, names =
    link names -> the ORACLE_NONTIP channel record (max_n and links mean what they mean for BUMP). Force (or NaN) in a
    fingertip column dies (wiring fault); unreadable columns or names die."""
    if (
        not tips
        or not names
        or len(names) != len(v)
        or not all(0 <= j < len(v) for j in tips)
        or any(math.isnan(x) for x in v)
    ):
        raise RuntimeError(
            f"{head} · cannot read the fingertip link columns (columns {tips} · names {len(names or [])} · links "
            f"{len(v)} · NaN check)"
        )
    hot = {names[j]: round(v[j], 4) for j in sorted(set(tips)) if v[j] != 0.0}
    if hot:
        raise RuntimeError(
            f"{head} · witness (object-side PhysX view) fingertip link force {hot}: the fingertips touched but the "
            "fingertip sensors read 0 = wiring fault"
        )
    load = sorted(((names[j], round(x, 4)) for j, x in enumerate(v) if x > 0.0), key=lambda t: (-t[1], t[0]))
    return {
        "max_n": round(max(v), 4),
        "links": len(load),
        "tip_links": [names[j] for j in sorted(set(tips))],
        "load_links": [list(t) for t in load],
    }


def oracle_verdict(wit: dict, human, tips: dict | None = None, names: dict | None = None) -> tuple[str, dict]:
    """wit = {partner-object channel: [max |object force| per link]}, human(channel) = frames in which the human hand
    holding that object touched it, tips = {channel: fingertip link columns}, names = {channel: link names} -> (gate
    name, per-channel record); dies when the gate does not pass."""
    nan = sorted(ch for ch, v in wit.items() if any(math.isnan(x) for x in v))
    if nan:  # before all three gates: Python max drops a NaN that is not first, which passed as untouched
        raise RuntimeError(f"{MARK} NaN in the oracle witness — channels {nan}: cannot judge zero touched samples")
    flat = {ch: max(v) if v else 0.0 for ch, v in wit.items()}
    if untouched_ok(flat):
        return ORACLE_UNTOUCHED, {}
    if not wit:
        raise RuntimeError(f"{MARK} oracle has no partner-object channel — cannot judge zero touched samples")
    rule, rec = ORACLE_BUMP, {}
    for ch, v in sorted(wit.items()):
        if flat[ch] == 0.0:
            continue
        frames = int(human(ch))
        if frames != 0:
            head = (
                f"{MARK} oracle saw no touched sample — the object force is unproven · partner-object "
                f"hand-link force max {flat} · {ch} = {frames} human contact frames (manipulated "
                f"object)"
            )
            rule, rec[ch] = ORACLE_NONTIP, oracle_nontip(v, (tips or {}).get(ch), (names or {}).get(ch), head)
            rec[ch]["human_frames"] = frames
            continue
        rec[ch] = {"max_n": round(flat[ch], 4), "links": sum(x > 0.0 for x in v), "human_frames": 0}
    return rule, rec


def human_frames(rec, ch: str) -> int:
    """Recorder rec's partner-object channel ch -> human contact frames of the hands holding it (channel sides), summed
    over clips; read from rec.traj_mgr.cfg.data_dir/<clip> via record.phases.human_touch."""
    from dexengine.record import phases as PH  # noqa: PLC0415

    sides = next((c.get("sides") for c in rec._chs if c["name"] == ch), None)
    root = getattr(getattr(getattr(rec, "traj_mgr", None), "cfg", None), "data_dir", None)
    clips = list(getattr(rec, "clip_names", None) or [])
    if not sides or not root or not clips:
        raise RuntimeError(
            f"{MARK} {ch}: cannot read human contact (sides {sides} · root {root} · clips "
            f"{len(clips)}): cannot judge zero touched samples"
        )
    return sum(int(PH.human_touch(Path(root) / c, s).sum()) for c in clips for s in sides)


def oracle_gate(rec, wit_t: dict) -> str:
    """finalize's zero-touched-samples branch: wit_t = {channel: per-link max (tensor)} -> gate name, meta and marker
    (dies when it does not pass). Fingertip link columns = the recorder's oracle pairs (rec._oracle entries (sensor
    key, sensor, channel, link column)) · link names = the channel names."""
    wit = {ch: [float(x) for x in v.reshape(-1).cpu()] for ch, v in wit_t.items()}
    tips = {ch: [j for _k, _s, c, j in rec._oracle if c == ch] for ch in wit}
    names = {c["name"]: list(c["names"]) for c in rec._chs if c["name"] in wit}
    rule, chs = oracle_verdict(wit, lambda ch: human_frames(rec, ch), tips, names)
    rec.meta_in.setdefault("collect", {})["oracle_gate"] = {"rule": rule, "channels": chs}
    if rule == ORACLE_UNTOUCHED:
        print(
            f"{MARK} oracle zero touched samples — partner objects {sorted(wit)} hand-link force 0 (all samples, "
            f"hand_object_contact witness) · nothing to prove ({ORACLE_UNTOUCHED} · gate passed)",
            flush=True,
        )
    elif rule == ORACLE_BUMP:
        print(
            f"{MARK} oracle zero touched samples — "
            + " · ".join(f"{ch} max {r['max_n']:.4g} N · links {r['links']}" for ch, r in chs.items())
            + " · 0 human contact frames for that hand — a non-fingertip link bumped an object that is not "
            f"manipulated; force stays in the object channel ({ORACLE_BUMP} · gate passed)",
            flush=True,
        )
    else:
        for ch, r in chs.items():
            load = " · ".join(f"{n} {x:.4g} N" for n, x in r.get("load_links", []))
            print(
                f"{MARK} oracle zero touched samples — {ch} {r['human_frames']} human contact frames · witness "
                f"fingertip links {r.get('tip_links', '(not human-touched)')} force 0 (all samples) · links that "
                f"carried the load {r['links']}: {load or '-'} — fingertip sensors 0 is the right reading "
                f"({ORACLE_NONTIP} · gate passed)",
                flush=True,
            )
    return rule
