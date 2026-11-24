"""Checker rules R24-R31: reject wording that the hand-contact facts (record field motion.hand_contact, built by
motion.attach_facts) do not back. Called by rules_qa.qa_problems; records without facts are not judged.

A text serves every shipped episode of its source clip (original and placement variants): a claim is rejected when
it is false in **any** episode; a requirement holds only for what happened in **every** episode. Objects whose
episodes disagree (lifted in some) reject both lift and slide words and pass only neutral words (move); the prompt
tells the model to write 'lifted some' that way.
  R24 a slide word (slide, push, drag, shove) names an object that is lifted (in any episode), the same text field
      has no lift word for it, and it is not a tool stroke (across, over, along ... — 'drag the knife across the
      plate'). Also rejected: an object that starts in the air (ends LIFT_CM or more below its start) and a move
      along the surface (SURFACE_RE).
  R25 a joint moved in every episode (joint min >= the motion.JOINT_UNITS threshold) but no field has a part-motion
      word (PART_RE).
  R26 a press word (press, click, pump, staple) or an open/close word whose object is that object (not 'close to',
      not a hand word) names an object without a joint.
  R27 (a) the text says it is held up in the air at the start but some episode has no support disc; (b) an object
      held from the start (held_at_start) is said to be lifted / raised while some episode does not lift it, or
      picked up while some episode lifts it less than PICK_HELD_CM.
  R28 hand roles — (a) 'both hands' + a lift word on an object whose lifted_by lacks one of the hands; (b) '<side>
      hand <verb> the <object>' / '<object> with the <side> hand' where that hand does not touch the object in some
      episode.
  R30 moved to a new spot (RELOC_RE) while some episode ends closer than RELOC_CM (horizontal end distance).
  R31 end state — (a) still held up at the end (HELD_UP_RE + END_RE in one comma segment) while some episode is not
      held up at the end (end_up != all); (b) set down (SET_DOWN_RE) while some episode is still held above LIFT_CM
      at the end (end_up != none). An object that starts and ends in the air cannot be judged (start height
      unknown); prompt rule 16 covers it.
Handover (R4 · R9) lives in rules_scene. Words are closed classes (rules, not lookup tables).
"""

from __future__ import annotations

import re

from dexengine.record.text import nouns as NN
from dexengine.record.text import qa_words as QW
from dexengine.record.text import rules_qa as QA
from dexengine.record.text import scene_words as SW

SLIDE = (
    "slide",
    "slides",
    "sliding",
    "slid",
    "push",
    "pushes",
    "pushing",
    "pushed",
    "drag",
    "drags",
    "dragging",
    "dragged",
    "shove",
    "shoves",
    "shoving",
    "shoved",
)
LIFTING = (
    "lift",
    "lifts",
    "lifting",
    "lifted",
    "raise",
    "raises",
    "raising",
    "raised",
    "pick",
    "picks",
    "picking",
    "picked",
    "carry",
    "carries",
    "carrying",
    "carried",
)
PART_RE = re.compile(
    r"\b(?:open\w*|clos(?:e|es|ed|ing)|shut\w*|swing\w*|swung|rais\w*|lower\w*|flip\w*|fold\w*|"
    r"unfold\w*|turn\w*|rotat\w*|twist\w*|lift\w*|pull\w*\s+(?:\w+\s+){0,3}?(?:out|open|apart)|"
    r"push\w*\s+(?:\w+\s+){0,3}?(?:in|closed|shut|back)|slid\w*\s+(?:\w+\s+){0,3}?(?:out|open|in|closed))\b"
)
#: Words that move a part by pressing (open and close are left out: they mostly name a hand or a distance — 'close
#: the hand around it', 'close to where it started'; squeeze is left out: it comes from a dataset label, H2O 'squeeze
#: lotion').
ACT = (
    "press",
    "presses",
    "pressing",
    "pressed",
    "click",
    "clicks",
    "clicking",
    "pump",
    "pumps",
    "pumping",
    "staple",
    "staples",
    "stapling",
)
#: Followers that make a press word a force, not a part motion ('press it down onto', 'press down on the chips');
#: controls (button, switch ...) may be pressed even when the model has no joint ('Press the button on the lamp').
NOT_ACT_NEXT = ("down", "onto", "into", "against", "on")
CONTROL = ("button", "buttons", "switch", "switches", "key", "keys", "trigger", "lever")
#: R26 open/close words — only when directly followed by the object (it, them, a name); 'close to', 'closer' and
#: 'close the hand around' are distance and hand talk.
OPEN_CLOSE = ("open", "opens", "opened", "opening", "close", "closes", "closed", "closing", "shut", "shuts", "shutting")
NOT_OBJ_NEXT = ("to", "by", "around", "in", "on", "against", "with", "up", "over", "together")
#: R30 new-spot threshold (cm, minimum horizontal end distance). 0.8-1.1 cm is "the same spot"; 3.5-4.7 cm is a new
#: spot (rules_qa.STILL_MOVE_CM = 5 cm would reject that).
RELOC_CM = 2.0
RELOC_RE = re.compile(
    r"\b(?:to|into|onto|in|at)\s+(?:a|another|its|their|the)\s+(?:new|different|other)\s+"
    r"(?:spot|place|position|location|area|site)\b|\b(?:to|in)\s+another\s+(?:spot|place|position|"
    r"location)\b|\brelocat\w*|\belsewhere\b|\b(?:move|moves|moved|moving|shift|shifts|shifted|"
    r"shifting|slide|slides|slid|sliding)\s+(?:it|them|the\s+\w+(?:\s+\w+)?)\s+over(?=\s*$|\s+(?:and|"
    r"then|while|before|after|with)\b)"
)
HELD_UP_RE = re.compile(
    r"\b(?:hold|holds|holding|held|keep|keeps|keeping|kept)\s+(?:\w+\s+){0,3}?"
    r"(?:up|raised|aloft|lifted|in the air|in midair|off the table)\b"
)
END_RE = re.compile(r"\b(?:at the end|in the end|by the end|until the end|still|finally|ends?|ending)\b")
SET_DOWN_RE = re.compile(
    r"\b(?:set|sets|setting|put|puts|putting|place|places|placing|placed|lower|lowers|lowering|"
    r"lowered)\s+(?:\w+\s+){0,3}?(?:down|on the table|onto the table|on the surface|"
    r"onto the surface)\b"
)
START_RE = re.compile(r"\b(?:already|initially|at the start|from the start|at first|to begin with|at the beginning)\b")
UP_RE = re.compile(
    r"\b(?:held up|holding up|held aloft|in the air|off the table|above the table|lifted|raised|"
    r"held in mid ?air|in midair)\b"
)
PICK_UP = ("pick", "picks", "picking", "lift", "lifts", "lifting", "raise", "raises", "raising")
#: Minimum lift (cm, every episode) for 'pick up' of an object held from the start: 2.3-2.9 cm is ambiguous, correct
#: cases rise 7.2 cm or more. lift / raise only check 'lifted all': the measurement cannot separate a held object that
#: rises 2-4 cm from one that starts in the air, so that threshold is a prompt recommendation (rule 16) only.
PICK_HELD_CM = 5.0
NEG = ("without", "not", "never", "no", "nor")
#: Tool stroke — a slide word followed by one of these is dragging a held tool over another object.
STROKE = ("across", "over", "along", "through", "inside", "within", "around", "against")
#: R24 moving along a surface (the object stays on the surface — 'moved along the surface').
SURFACE_RE = re.compile(r"\b(?:along|across|over)\s+(?:the\s+)?(?:surface|table|tabletop|counter|desk)\b")
SPAN = 6


def _hc(rec: dict) -> list[dict]:
    return (rec.get("motion") or {}).get("hand_contact") or []


def verb_moved(rec: dict, cls: str) -> bool:
    """The verb class (check.VERB_CLASS) is true in every replay by the facts — right even when the source summary
    (motion.objects, one episode) is under the threshold but every shipped replay is above it."""
    from dexengine.record.text import check as CHK

    ok = {
        "lift": lambda o: o["lift_cm"][0] >= CHK.LIFT_CM,
        "move": lambda o: o["move_cm"][0] >= CHK.MOVE_CM_MIN or o["lift_cm"][0] >= CHK.LIFT_CM,
        "rotate": lambda o: o["rot_deg"][0] >= CHK.ROT_DEG_MIN,
    }[cls]
    return any(ok(o) for o in _hc(rec))


def label_due(rec: dict, verb: str) -> bool:
    """Must the text keep the label verb? For a motion verb (VERB_CLASS) only when the facts show that motion in every
    replay (a requirement holds only for every replay); without facts or for a non-motion verb, yes."""
    from dexengine.record.text import check as CHK

    cls = CHK.VERB_CLASS.get(verb)
    return not (cls and _hc(rec)) or verb_moved(rec, cls)


def hands_all(rec: dict) -> str | None:
    """Hands touching in every replay (union of the facts' touched_by): left, right, both, or None. When the input
    hands are 'both' but only one hand touches in every replay, 'both' is true in some replays only."""
    s = {h for o in _hc(rec) for h in o.get("touched_by") or []}
    return {"left": "left", "right": "right", "left right": "both"}.get(" ".join(sorted(s)))


def _after(ws: list[str], i: int) -> list[str]:
    nxt = ws[i + 1 : i + 1 + SPAN]
    return nxt[: next((k for k, x in enumerate(nxt) if x in QW.CLAUSE_WORDS), len(nxt))]


def _target(hc: list[dict], nxt: list[str]) -> list[dict]:
    """Objects named by the words after a verb; with no name and a single object, that object (it, them)."""
    return QA._named(hc, nxt) or (hc if len(hc) == 1 else [])


def _neg(ws: list[str], i: int) -> bool:
    return bool(set(ws[max(0, i - 2) : i]) & set(NEG))


def fact_problems(rec: dict, out: dict) -> list[str]:
    """Reasons from R24-R31 (none for a record without facts)."""
    hc = _hc(rec)
    if not hc:
        return []
    from dexengine.record.text import rules_scene as RU

    alls = QA.fold(rec, RU._all_texts(out))
    return (
        _slide_lifted(rec, hc, alls)
        + _part_missing(rec, hc, alls)
        + _act_rigid(rec, hc, alls)
        + _start_claims(rec, hc, alls)
        + _hand_roles(hc, alls)
        + _end_claims(rec, hc, alls)
    )


def _slide_lifted(rec: dict, hc: list[dict], alls: list[str]) -> list[str]:
    """R24 — slide words (SLIDE + the object after it) and moves along a surface (SURFACE_RE + the object of that
    clause, else the single main object). A tool stroke = a stroke word (STROKE) followed by another object's name
    ('drag the knife across the plate'); 'along the surface' is not a stroke."""
    from dexengine.record.text import check as CHK
    from dexengine.record.text import rules_scene as RU

    bad = set()

    def off(o, w):
        down = o["end_z_cm"] and o["end_z_cm"][0] <= -CHK.LIFT_CM  # ends lower than it started = not on the surface
        if (o["lifted"] != "none" or down) and o["obj"] not in lifted_here and not o.get("joint"):
            bad.add(f"{w}→{o['obj']}(lift {o['lift_cm'][1]:g} cm · end height {o['end_z_cm'][0]:g} cm)")

    for t in alls:
        ws = RU._ws(t)
        lifted_here = {
            o["obj"] for i, w in enumerate(ws) if w in LIFTING and not _neg(ws, i) for o in _target(hc, _after(ws, i))
        }
        for i, w in enumerate(ws):
            nxt = _after(ws, i)
            stroke = any(x in STROKE and QA._named(hc, nxt[k + 1 :]) for k, x in enumerate(nxt))
            if (
                w in SLIDE
                and not _neg(ws, i)
                and nxt[:1] != ["down"]
                and not stroke
                and not set(nxt[:2]) & set(QW.NOT_THING)
                and not set(nxt[:4]) & set(CONTROL)
                and not _part_of(rec, nxt)
            ):
                for o in _target(hc, nxt):
                    off(o, w)
        for c in RU.CLAUSE_RE.split(t.lower()):
            if c and SURFACE_RE.search(c) and not set(RU._ws(c)) & set(NEG):
                for o in QA._named(hc, RU._ws(c)) or _main(rec, hc):
                    off(o, SURFACE_RE.search(c).group())
    return (
        [
            f"R24 push words {sorted(bad)} ↔ that object is lifted or in the air from the start — the object is off "
            "the surface in this clip "
            "(lifted, or carried down from above the surface); say lift, carry or move it, not slide or push it"
        ]
        if bad
        else []
    )


def _part_missing(rec: dict, hc: list[dict], alls: list[str]) -> list[str]:
    from dexengine.record.text import motion as MOT

    kinds = {j.get("kind") for j in (rec.get("motion") or {}).get("joints") or []}
    if len(kinds) != 1:
        return []
    unit, _, thr = MOT.JOINT_UNITS[next(iter(kinds))]
    moved = [o for o in hc if o.get("joint") and o["joint"][0] >= thr]
    if not moved or any(PART_RE.search(t.lower()) for t in alls):
        return []
    got = [f"{o['obj']} {o['joint'][0]:g}–{o['joint'][1]:g} {unit}" for o in moved]
    return [
        f"R25 the articulated part moved (every episode ≥ {thr:g} {unit}: {got}) but no part-motion words — say "
        "how the part "
        "moves (open or close the lid or door, pull the drawer out, swing it open); do not write only grasp, hold "
        "and let go"
    ]


def _act_rigid(rec: dict, hc: list[dict], alls: list[str]) -> list[str]:
    from dexengine.record.text import rules_scene as RU

    if QA._joints(rec):
        return []  # a clip with an articulated part: the words are about that part
    bad = set()
    for t in alls:
        ws = RU._ws(t)
        for i, w in enumerate(ws):
            nxt = _after(ws, i)
            if (
                w in OPEN_CLOSE
                and not _neg(ws, i)
                and nxt
                and nxt[0] not in NOT_OBJ_NEXT
                and not set(nxt[:3]) & set(QW.NOT_THING + CONTROL)
            ):
                bad |= {
                    f"{w}→{o['obj']}"
                    for o in (QA._named(hc, nxt[:4]) or (_main(rec, hc) if nxt[0] in ("it", "them") else []))
                }
            if w not in ACT or _neg(ws, i) or set(nxt) & set(NOT_ACT_NEXT + CONTROL):
                continue
            bad |= {f"{w}→{o['obj']}" for o in _target(hc, nxt)}
    return (
        [
            f"R26 joint-motion words {sorted(bad)} ↔ that object has no joint — it has no moving part in this clip; "
            "describe "
            "only the measured motion (lift, tilt, set down), not pressing, opening or stapling"
        ]
        if bad
        else []
    )


def _start_claims(rec: dict, hc: list[dict], alls: list[str]) -> list[str]:
    from dexengine.record.text import rules_scene as RU

    why = []
    if any(o["air_start"] != "all" for o in hc):
        up = sorted(
            {t[:60] for t in alls for c in RU.CLAUSE_RE.split(t.lower()) if START_RE.search(c) and UP_RE.search(c)}
        )
        if up:
            why.append(
                f"R27 words saying it starts held in the air {up} ↔ no support — the object rests on the table "
                "at the "
                "start; do not say it is already held up or in the air"
            )
    bad = set()
    for ws in (RU._ws(seg) for t in alls for seg in re.split(r"[,;.]", t)):
        for i, w in enumerate(ws):
            nxt = _after(ws, i)
            if (
                w not in PICK_UP
                or _neg(ws, i)
                or (w.startswith("pick") and "up" not in nxt)
                or set(nxt[:2]) & set(QW.NOT_THING)
            ):  # 'lift the hand off the lamp' is the hand
                continue

            def held(o, w=w):  # called inside this iteration only
                low = o["lift_cm"][0] < PICK_HELD_CM and w.startswith("pick")
                return o["held_at_start"] and (o["lifted"] != "all" or low)

            c = QA.refs(rec, hc, ws[:i], ws[i + 1 : i + 1 + SPAN])  # only when every candidate is held from the start
            if c and all(held(o) for o in c):
                bad |= {f"{w}→{o['obj']}({o['lift_cm'][0]:g} cm)" for o in c}
    if bad:
        why.append(
            f"R27 picks up an object held from the start {sorted(bad)} ↔ in the hand from the start and rises < "
            f"{PICK_HELD_CM:g} cm — "
            "it is already in the hand at the start; do not say it is picked up (say lift or raise only if it "
            "rises in every replay); say it is held and moved"
        )
    return why


def _hand_roles(hc: list[dict], alls: list[str]) -> list[str]:
    from dexengine.record.text import rules_scene as RU

    both, one = set(), set()
    seen = [w for o in hc for w in set(NN.toks(o["obj"]))]
    toks = {w: o for o in hc for w in NN.toks(o["obj"]) if len(w) >= 3 and seen.count(w) == 1}  # one object's word
    for t in alls:
        for c in RU.CLAUSE_RE.split(t.lower()):
            ws = RU._ws(c)
            if (RU._has("both hands", ws) or RU._has("two hands", ws)) and set(ws) & set(PICK_UP):
                i = next(k for k, x in enumerate(ws) if x in PICK_UP)
                for o in _target(hc, ws[i + 1 :]) or _target(hc, ws):
                    if o["lifted"] != "none" and set(o.get("lifted_by") or []) != {"left", "right"}:
                        both.add(f"{o['obj']} lifted_by={o.get('lifted_by')}")
            skip = QA.place_mask(ws, set(seen), QW.MOVE_PREP)  # 'out of the base': base is not the hand's object
            for side, w in RU.held_by(ws, toks, skip):
                o = toks[w]
                if side not in o["touched_by"]:
                    one.add(f"{side} hand→{o['obj']}(touched_by={o['touched_by']})")
    why = []
    if both:
        why.append(
            f"R28 lifted with both hands ↔ hands touching at lift start {sorted(both)} — say which hand lifts "
            "it first and when "
            "the other hand joins"
        )
    if one:
        why.append(
            f"R28 hand ↔ touched object {sorted(one)} — that hand does not touch that object (in every shipped "
            "replay); give each object to the hand that touches it"
        )
    return why


def _part_of(rec: dict, ws: list[str]) -> bool:
    """In a record whose joint moved, the verb's object is a part noun (rules_scene.PART): 'open the door', 'lift the
    lid', 'turn the crank' are part words, not moves of the body (not judged by R24, R30, R31)."""

    return bool(set(ws) & set(SW.PART)) and any(j.get("moved") for j in QA._joints(rec))


def _main(rec: dict, hc: list[dict]) -> list[dict]:
    """Object named by a pronoun (it, them): the record's main object when there is one, else the single facts
    object."""
    main = {o.get("name") for o in rec.get("objects") or [] if o.get("role") == "main"}
    mains = [o for o in hc if o["obj"] in main]
    return mains if len(mains) == 1 else hc if len(hc) == 1 else []


def _all_bad(c: list, bad, fmt) -> set:
    """The candidates, only when every candidate violates the claim (one plausible target that fits passes)."""
    return {fmt(o) for o in c} if c and all(bad(o) for o in c) else set()


def _span(o: dict) -> str:
    return f"{o['obj']}({o['end_xy_cm'][0]:g}–{o['end_xy_cm'][1]:g} cm)"


def _state(o: dict) -> str:
    return f"{o['obj']}(end_up {o['end_up']})"


def _end_claims(rec: dict, hc: list[dict], alls: list[str]) -> list[str]:
    """R30 · R31 — forms per comma segment; targets = rules_qa.refs (names in the form -> every name before it -> all
    facts objects), rejected only when every candidate violates. The destination or the support of a put-down
    (another object after the form — 'down on top of the tripod') is not a target: a put-down onto a support ends
    higher than it started, so the set-down form is not judged when another object follows it."""
    from dexengine.record.text import rules_scene as RU

    reloc, up, down = set(), set(), set()
    for t in alls:
        for sm in re.finditer(r"[^,;.]+", t.lower()):
            seg, pre = sm.group(), RU._ws(t.lower()[: sm.start()])  # earlier segments of this text ('…, lifted

            def pick(m, inner, seg=seg, pre=pre):  # and carried to a new spot') — called in this iteration only
                before = RU._ws(seg[: m.start()])
                if set(before[-4:]) & set(NEG) or _part_of(rec, before[-3:] + inner):  # 'without moving it to …'
                    return []
                return QA.refs(rec, hc, pre + before, inner)  # all candidates — rejected only if all violate

            for m in RELOC_RE.finditer(seg):
                reloc |= _all_bad(pick(m, []), lambda o: o["end_xy_cm"][0] < RELOC_CM, _span)
            if END_RE.search(seg):
                for m in HELD_UP_RE.finditer(seg):
                    up |= _all_bad(pick(m, RU._ws(m.group())), lambda o: o["end_up"] != "all", _state)
            for m in SET_DOWN_RE.finditer(seg):
                if not QA._named(hc, RU._ws(seg[m.end() :])[:SPAN]):
                    down |= _all_bad(pick(m, RU._ws(m.group())[1:]), lambda o: o["end_up"] != "none", _state)
    why = []
    if reloc:
        why.append(
            f"R30 move-to-a-new-spot words ↔ end distance < {RELOC_CM:g} cm {sorted(reloc)} — it ends where it "
            "started (in "
            "at least one replay); do not say it is moved to a new spot"
        )
    if up:
        why.append(
            f"R31 held-at-the-end words ↔ not held up in the hand at the end {sorted(up)} — at the end it is "
            "not still held "
            "up in the hand in every replay; say where it ends (set down, resting)"
        )
    if down:
        why.append(
            f"R31 set-down words ↔ still held up at the end {sorted(down)} — it is still held up in the hand at "
            "the "
            "end (in at least one replay); do not say it is set down or placed on the table"
        )
    return why
