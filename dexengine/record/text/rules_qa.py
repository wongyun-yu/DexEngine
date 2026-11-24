"""Checker rules R16-R23: reject wording that the measured metadata does not back. Called by
rules_scene.v3_problems; also the target resolution shared by these rules and rules_facts (refs, fold, place_mask).

  R16 an articulated part moved in this replay (a measured manipulate phase and a joint range >= the motion
      threshold of motion.JOINT_UNITS) but the instruction has no part-motion verb (ART_VERBS) — e.g. a drawer pulled
      out written as 'Hold the drawer frame steady'. Records without a joints field are measured from their tree.
  R16 hand — the part is moved by one hand (joint hand) but a clause with a part-only verb says 'both hands' or the
      other hand.
  R17 a size word (slightly, a little ...) within NEAR words of a turn verb in one clause (CLAUSE_RE) while the
      largest rotation of a moved object is >= ROT_BIG_DEG ('slide it slightly and rotate it' passes: two clauses).
  R18 the source of a pour verb ('... from the <object>') is an object that did not move in this clip.
  R19 the carried thing is not in the scene — (a) a single object without articulated parts but a take-out / put-in
      form (take ... out, put ... in/into; a place noun after it (LOC), as in 'put it down in a new spot', passes);
      (b) a carry verb whose object is a placeholder (PLACEHOLDER — 'carry the item': a source sentence leaking in);
      (c) no moved object and no moved joint but a carry verb (CARRY / MOVE).
  R20 an object held from the start (no measured approach or grasp phase, the object ends >= HELD_DROP_CM below its
      start and its lift stays under the threshold) is said to be picked up — the opposite of its end state.
  R21 the text names an object missing from the render (objects_absent — removed from the input by
      vocab.drop_absent).
  R22 a lift word (carry, lift, raise, pick up — negation or a hand word after it passes) names an object lifted
      less than LIFT_CM, not lowered from the air and without joint motion.
  R23 a stay-still clause (steady, hold ... still, keep/stay ... in place) names an object (by name when there are
      two or more objects) that is lifted >= LIFT_CM or slides >= STILL_MOVE_CM (turning in place passes).
Thresholds and words are closed classes (rules, not lookup tables); their owner is qa_words.
"""

from __future__ import annotations

import re

from dexengine.record.text import nouns as NN
from dexengine.record.text.qa_words import (
    ART_VERBS,
    BODY_WORDS,
    CARRY,
    CLAUSE_WORDS,
    HELD_DROP_CM,
    LIFT_SPAN,
    LIFT_WORDS,
    MOVE,
    NEAR,
    NEG,
    NOT_THING,
    PART_ONLY,
    PICK,
    PICK_SPAN,
    PLACE_PREP,
    PLACEHOLDER,
    POUR_RE,
    PUT_IN,
    ROT_BIG_DEG,
    SMALL,
    STEADY_RE,
    STILL_MOVE_CM,
    STOP_VERBS,
    TAKE_OUT,
    TURN,
)


def _joints(rec: dict) -> list:
    from dexengine.record.text import motion as MOT

    return MOT.rec_joints(rec)


def _near(ws: list[str]) -> bool:
    """A size word within NEAR words of a turn verb ('turning it slightly', 'a slight turn')."""
    turn = [j for j, w in enumerate(ws) if w in TURN]
    small = [i for p in SMALL for i in range(len(ws)) if ws[i : i + len(p.split())] == p.split()]
    return any(abs(i - j) <= NEAR for i in small for j in turn)


def _joint_problems(moved, out, why):
    """R16 hand: the hand that moves an articulated part is the hand the text names for it."""
    from dexengine.record.text import rules_scene as RU  # lazy: rules_scene imports this module's callers

    for j in moved:
        if j.get("hand") in ("left", "right"):
            other = "right" if j["hand"] == "left" else "left"
            bad = sorted(
                {
                    w
                    for ws in RU._side_clauses(out)
                    if set(ws) & set(PART_ONLY)
                    for w in ("both hands", f"{other} hand")
                    if RU._has(w, ws)
                }
            )
            if bad:
                why.append(
                    f"R16 hand — the hand moving the part ({j['joint']}) is {j['hand']} but {bad} — "
                    f"say the {j['hand']} hand "
                    "moves the part; the other hand only holds the body"
                )


def _still_problems(rec, alls, why):
    """R18 pour source: text may pour only from an object that moves in this clip."""
    still = set().union(
        *[NN.toks(o["name"]) for o in rec.get("objects") or [] if o.get("moved") is False],
        *[NN.toks(n) for n in rec.get("scene_objects") or []],
    )  # render-scene objects are still
    live = set().union(*[NN.toks(o["name"]) for o in rec.get("objects") or [] if o.get("moved")])
    src = sorted(
        {
            w
            for t in alls
            for m in POUR_RE.finditer(str(t).lower())
            for w in m.group(1).split()
            if w in still and w not in live
        }
    )
    if src:
        why.append(
            f"R18 pour source {src} does not move in this clip — pour only from the object that is tipped in this clip"
        )


def qa_problems(rec: dict, out: dict) -> list[str]:
    """Reasons from R16-R23, then the fact rules (rules_facts R24-R31)."""
    from dexengine.record.text import rules_facts as FR
    from dexengine.record.text import rules_scene as RU  # lazy: rules_scene imports this module's callers

    why, alls = [], fold(rec, RU._all_texts(out))
    ins = set(RU._ws(out.get("instruction")))
    moved = [j for j in _joints(rec) if j.get("moved")]
    if moved and "manipulate" in RU.observed(rec) and not ins & set(ART_VERBS):
        got = [f"{j['joint']} {j['range']}{j['unit']}" for j in moved]
        why.append(
            f"R16 articulated part moves {got} but the instruction has no part motion — say the part motion "
            "(pull the drawer out, "
            "open or close the lid or door); do not write only 'hold it steady'"
        )
    _joint_problems(moved, out, why)
    mo = [o for o in (rec.get("motion") or {}).get("objects") or [] if o.get("moved")]
    rot = max((float(o.get("rot_max_deg") or 0) for o in mo), default=0.0)
    cl = [RU._ws(c) for t in alls for c in RU.CLAUSE_RE.split(str(t).lower()) if c]
    if rot >= ROT_BIG_DEG and any(_near(ws) for ws in cl):
        why.append(
            f"R17 size words (slightly · a little) + turning ↔ measured max rotation {rot:.0f}° ≥ "
            f"{ROT_BIG_DEG:.0f}° — "
            "the object "
            "turns a lot; do not call the turn slight"
        )
    _still_problems(rec, alls, why)
    scene = [o for o in rec.get("objects") or [] if not str(o.get("role") or "").startswith("supplemented")]
    if (
        len(scene) == 1
        and not _joints(rec)
        and (
            bad := sorted({m.group(0) for t in alls for rx in (TAKE_OUT, PUT_IN) for m in rx.finditer(str(t).lower())})
        )
    ):
        why.append(
            f"R19 only one object but a take-out/put-in form {bad} — there is no container in this scene; keep "
            "the label "
            "verb without 'out' or 'in' ('Take the cocoa box and lift it up')"
        )
    if ph := sorted({m.group(0) for t in alls for m in PLACEHOLDER.finditer(str(t).lower())}):
        why.append(
            f"R19 carries an unnamed thing {ph} — nothing else is carried in this clip; name only the listed objects"
        )
    if (rec.get("motion") or {}).get("objects") and not mo and not moved and (mv := _carry_words(alls)):
        why.append(
            f"R19 no object moves but a carry verb {mv} — nothing moves in this clip; describe the reach, "
            "grasp or "
            "hold only"
        )
    why += _held_pick(rec, out) + _absent_named(rec, alls) + _lift_unlifted(rec, alls) + _steady_moved(rec, alls)
    return why + FR.fact_problems(rec, out)


def _carry_words(alls: list[str]) -> list[str]:
    """Words that carry a thing (R19 c): every carry / transfer; move only before it, them or the <non-hand word>."""
    from dexengine.record.text import rules_scene as RU

    got = set()
    for ws in (RU._ws(t) for t in alls):
        for i, w in enumerate(ws):
            nxt = ws[i + 1 : i + 3]
            if set(ws[max(0, i - 2) : i]) & {"without", "not", "never", "no", "nor"}:  # 'without moving it' = negated
                continue
            if w in CARRY or (
                w in MOVE
                and nxt
                and (nxt[0] in ("it", "them") or (nxt[0] == "the" and len(nxt) > 1 and nxt[1] not in NOT_THING))
            ):
                got.add(w)
    return sorted(got)


def _held_pick(rec: dict, out: dict) -> list[str]:
    """R20 — an object held from the start is said to be picked up (its end state is the opposite)."""
    from dexengine.record.text import check as CHK
    from dexengine.record.text import rules_scene as RU

    obs = RU.observed(rec)
    if not obs or obs & {"approach", "grasp"}:
        return []
    names = {
        o["obj"]
        for o in (rec.get("motion") or {}).get("objects") or []
        if o.get("moved")
        and float((o.get("net_cm") or [0, 0, 0])[2]) <= -HELD_DROP_CM
        and float(o.get("lift_max_cm") or 0) < CHK.LIFT_CM
    }
    objs = (rec.get("motion") or {}).get("objects") or []
    bad = set()
    for t in fold(rec, RU._all_texts(out)):
        for sent in re.split(r"[.;,]", str(t).lower()):
            ws = RU._ws(sent)
            for i, w in enumerate(ws):
                if w in PICK and names:
                    tail = []
                    for x in ws[i + 1 : i + 1 + PICK_SPAN]:
                        if x in STOP_VERBS:
                            break
                        tail.append(x)
                    c = refs(rec, objs, ws[:i], tail, back=False)  # target = names after the verb only
                    if c and all(o["obj"] in names for o in c):
                        bad |= {o["obj"] for o in c}
    return (
        [
            f"R20 says it picks up an object held from the start {sorted(bad)} — it is already held at the start and "
            "is set down; "
            "say it is lowered or put down, not picked up"
        ]
        if bad
        else []
    )


def _absent_named(rec: dict, alls: list[str]) -> list[str]:
    """R21 — names of objects missing from the render: the full name, or a tail of two or more words with the leading
    modifiers removed ('whole milk carton' -> 'milk carton'). Words inside a kept or scene object name pass."""
    here = " | ".join(
        [str(o.get("name", "")).lower() for o in rec.get("objects") or []]
        + [str(n).lower() for n in rec.get("scene_objects") or []]
    )
    t = " ".join(str(x) for x in alls).lower()
    hit = set()
    for n in rec.get("objects_absent") or []:
        ws = n.lower().split()
        tails = [" ".join(ws[k:]) for k in range(len(ws) - 1)] if len(ws) > 1 else ws
        if any(
            not re.search(r"\b" + re.escape(x) + r"\b", here) and re.search(r"\b" + re.escape(x) + r"\b", t)
            for x in tails
        ):
            hit.add(n)
    hit = sorted(hit)
    return (
        [f"R21 object not in the render {hit} — {', '.join(hit)} is not in this scene; name only the listed objects"]
        if hit
        else []
    )


def _fact(rec: dict, name) -> dict | None:
    """The object's hand-contact facts row (motion.hand_contact), or None."""
    return next((f for f in (rec.get("motion") or {}).get("hand_contact") or [] if f["obj"] == name), None)


def _named(objs: list, ws) -> list:
    """Objects named by the words: an object all of whose name words (>= 3 letters) appear; otherwise by a word that
    only one object has (a shared word like 'bottle' does not name another bottle)."""

    tk = [(o, {w for w in NN.toks(str(o.get("obj", ""))) if len(w) >= 3}) for o in objs]
    full = [o for o, t in tk if t and t <= set(ws)]
    if full:  # both match -> only the longer name ('test tube holder')
        return [o for o, t in tk if o in full and not any(t < u for p, u in tk if p in full)]
    seen = [w for _, t in tk for w in t]
    return [o for o, t in tk if {w for w in t if seen.count(w) == 1} & set(ws)]


def _lift_unlifted(rec: dict, alls: list[str]) -> list[str]:
    """R22 — a lift word on an object that was not lifted: every object named within LIFT_SPAN words after the verb
    (all moved objects when none is named) stays under LIFT_CM, was not lowered from the air (net z > -HELD_DROP_CM)
    and has no joint motion."""
    from dexengine.record.text import check as CHK
    from dexengine.record.text import rules_scene as RU

    mo = [o for o in (rec.get("motion") or {}).get("objects") or [] if o.get("moved")]
    if not mo or any(j.get("moved") for j in _joints(rec)) or any(o.get("lift_max_cm") is None for o in mo):
        return []  # records without a measured lift are not judged

    def flat(o):  # with facts: some shipped episode is not lifted (rules_facts' criterion)
        dz, f = float((o.get("net_cm") or [0, 0, 0])[2]), _fact(rec, o.get("obj"))
        up = f["lifted"] == "all" if f else float(o.get("lift_max_cm") or 0) >= CHK.LIFT_CM
        return not up and dz > -HELD_DROP_CM

    got = set()
    for ws in (RU._ws(seg) for t in alls for seg in re.split(r"[,;.]", str(t))):
        for i, w in enumerate(ws):
            nxt = ws[i + 1 : i + 1 + LIFT_SPAN]
            nxt = nxt[: next((k for k, x in enumerate(nxt) if x in CLAUSE_WORDS), len(nxt))]  # within the clause
            if (
                w not in LIFT_WORDS
                or set(ws[max(0, i - 2) : i]) & set(NEG)
                or set(nxt[:2]) & set(NOT_THING)
                or (w.startswith("pick") and "up" not in nxt)
            ):
                continue
            named = refs(rec, mo, ws[:i], ws[i + 1 : i + 1 + LIFT_SPAN])  # unmoved objects cannot be judged
            if named and all(flat(o) for o in named):
                got.add(f"{w}→{'/'.join(str(o.get('obj')) for o in named)}")
    return (
        [
            f"R22 lift words {sorted(got)} ↔ that object's max lift < {CHK.LIFT_CM:g} cm — the object is pushed or "
            "slid along "
            "the surface, not lifted; say slide, push or move, not carry, lift or pick up"
        ]
        if got
        else []
    )


def _steady_moved(rec: dict, alls: list[str]) -> list[str]:
    """R23 — an object said to stay still moved (per clause; without a name and with a single object, that object)."""
    from dexengine.record.text import check as CHK
    from dexengine.record.text import rules_scene as RU

    objs = (rec.get("motion") or {}).get("objects") or []
    joint = bool(_joints(rec))

    def moved(f):  # moved in any shipped episode (max); an object held from the start and not
        if f["held_at_start"] and f["lift_cm"][1] < CHK.LIFT_CM:  # lifted further is 'held', not moved
            return False
        return f["lift_cm"][1] >= CHK.LIFT_CM or f["move_cm"][1] >= STILL_MOVE_CM

    def big(o):  # several objects with one name: only when all of them moved
        fs = [f for f in (rec.get("motion") or {}).get("hand_contact") or [] if f["obj"] == o.get("obj")]
        if fs:
            return all(moved(f) for f in fs)
        n = o.get("net_cm") or [0, 0]
        flat = max(abs(float(n[0])), abs(float(n[1])))
        return float(o.get("lift_max_cm") or 0) >= CHK.LIFT_CM or flat >= STILL_MOVE_CM

    bad = set()
    for t in alls:
        for c in RU.CLAUSE_RE.split(str(t).lower()):
            if not c or not STEADY_RE.search(c):
                continue
            ws = RU._ws(c)
            # two or more objects: the named ones; a single object: only when the clause names the body of an
            # articulated object (base, body ...) — otherwise it means 'carry it without shaking'
            one = joint and len(objs) == 1 and set(ws) & set(BODY_WORDS)
            named = _named(objs, ws) if len(objs) > 1 else (objs if one else [])
            for o in named:
                if big(o):
                    bad.add(str(o.get("obj")))
    why = f"R23 still words ↔ moved object {sorted(bad)} (lift ≥ {CHK.LIFT_CM:g} cm · move ≥ {STILL_MOVE_CM:g} cm) — "
    return (
        [why + f"{', '.join(sorted(bad))} moves in this clip; do not say it is held steady, still or in place"]
        if bad
        else []
    )


# ───────────────────────── target resolution (R20 · R22 · R27 · R28 · R30 · R31) ─────────────────────────
def place_mask(ws: list[str], vocab: set, preps: tuple = PLACE_PREP) -> set[int]:
    """Positions of the name phrases (determiners + name words) after a place preposition (preps)."""

    got, i = set(), 0
    while i < len(ws):
        if ws[i] not in preps:
            i += 1
            continue
        i += 1 + (ws[i] == "out" and ws[i + 1 : i + 2] == ["of"])
        while i < len(ws) and ws[i] in NN.DET:
            i += 1
        while i < len(ws) and ws[i] in vocab:
            got.add(i)
            i += 1
    return got


def strip_places(ws: list[str], vocab: set) -> list[str]:
    """The words without the place name phrases (place_mask)."""
    m = place_mask(ws, vocab)
    return [w for k, w in enumerate(ws) if k not in m]


def mentions(objs: list, ws: list[str]) -> list:
    """Every object the words name = at each verbatim occurrence the longest name ('test tube' inside 'test tube
    holder' does not count; a separate 'the pencil' is pencil) ∪ names all of whose words appear (any order: 'the
    milk carton' <-> carton milk, short words like 'oj' included) ∪ a word only one object has ('the bottle cap' <->
    gray-green porcelain bottle cap)."""
    from dexengine.record.text import rules_scene as RU

    at = [
        (i, i + len(n), o)
        for o in objs
        if (n := RU._ws(o.get("obj")))
        for i in range(len(ws))
        if ws[i : i + len(n)] == n
    ]
    got = [o for i, j, o in at if not any(a <= i and j <= b and b - a > j - i for a, b, _ in at)]
    tk = [(o, set(NN.toks(str(o.get("obj", ""))))) for o in objs]
    full = [o for o, t in tk if t and t <= set(ws)]
    got += [o for o, t in tk if o in full and not any(t < u for p, u in tk if p in full)]
    seen = [w for _, t in tk for w in t if len(w) >= 3]
    got += [o for o, t in tk if {w for w in t if len(w) >= 3 and seen.count(w) == 1} & set(ws)]
    return [o for k, o in enumerate(got) if all(o is not x for x in got[:k])]


def refs(rec: dict, objs: list, before: list[str], after: list[str], back: bool = True) -> list:
    """Every object the claim may name (rules reject only when all candidates violate the claim). Order: names after
    the verb (up to a clause word) -> names after 'and/or + verb' ('picking up and turning the pot') -> every name
    before the verb in the same segment (pronoun, no name) -> all objs. Place phrases are removed (strip_places). A
    candidate slot naming an object outside objs (metadata or scene name — not measurable) gives []. back=False:
    names after the verb only (pronouns are not resolved — R20)."""

    mine = {str(o.get("obj")) for o in objs}
    other = [
        n
        for n in {str(o.get("name") or "") for o in rec.get("objects") or []}
        | set(map(str, rec.get("scene_objects") or []))
        if n and n not in mine
    ]
    vocab = {w for n in [*mine, *other] for w in NN.toks(n)}
    k = next((k for k, x in enumerate(after) if x in CLAUSE_WORDS), len(after))
    spans = [after[:k]]
    if after[k : k + 1] in (["and"], ["or"]) and set(after[:k]) <= set(NN.PARTICLE):  # two verbs, one object
        rest = after[k + 2 :]
        spans.append(rest[: next((j for j, x in enumerate(rest) if x in CLAUSE_WORDS), len(rest))])
    for span in [*spans, before] if back else spans:
        ws = strip_places(list(span), vocab)
        if any((t := set(NN.toks(n))) and t <= set(ws) for n in other):
            return []
        if got := mentions(objs, ws):
            return got
    return list(objs) if back else []


def fold(rec: dict, texts: list) -> list[str]:
    """Glue another object's longer name that contains a facts name ('test tube' ⊂ 'test tube holder') into one
    word when the longer name has no facts row, so it is not read as the shorter object ('grip the test tube
    holder' would trigger R28 / R22)."""
    from dexengine.record.text import rules_scene as RU

    facts = [RU._ws(f["obj"]) for f in (rec.get("motion") or {}).get("hand_contact") or []]
    names = {str(o.get("name") or "") for o in rec.get("objects") or []} | set(map(str, rec.get("scene_objects") or []))
    longer = sorted(
        (
            w
            for w in map(RU._ws, names)
            if w not in facts
            and any(len(f) < len(w) and any(w[i : i + len(f)] == f for i in range(len(w))) for f in facts)
        ),
        key=len,
        reverse=True,
    )
    if not longer:
        return [str(t) for t in texts]
    pat = re.compile(r"\b(" + "|".join(r"\s+".join(w) for w in longer) + r")\b", re.I)
    return [pat.sub(lambda m: "".join(m.group(0).split()), str(t)) for t in texts]
