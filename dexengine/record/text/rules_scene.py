"""Checker scene rules: reject wording that the measurements, the case or the source text do not back. Called by
check.validate_one; every rule rejects (a rejected clip is generated again).

  R1  a segment (segment set) whose phase table has no manipulate phase must not use a manipulation verb
      (verb = the verb field, else the first instruction word; non-manipulation verbs = closed class NON_MANIP).
  R2  a relation word (RELATION) followed within REL_SPAN words (same clause) by a side object (role extra) word in
      the instruction or a paraphrase — backed when the source text has that relation word, or the object is main
      (measured motion), support (measured support role) or near the end position of a moved object (motion
      ends_near; threshold motion.NEAR_CM).
  R3  instruction, paraphrases and detailed name different part nouns (PART, closed class: lid <-> door).
  R4 · R9 handover (below): the facts win; the case decides only without facts.
  R5  table / camera / body directions (prompt rule 4) in any text field including phase texts: preposition + the +
      left/right (not followed by a hand or arm word), move preposition + the + middle/center/front/back/edge (not
      followed by 'of', which is object-relative), leftward/rightward, from left to right, and body directions
      (BODY_DIR). Passes when the source text has the same form.
  R13 · R14 · R15 (v3_problems), R16-R23 (rules_qa), R24-R31 (rules_facts), R6-R11 (source_wording_problems).
Closed word classes and thresholds live in scene_words and qa_words.
"""

from __future__ import annotations

import re

from dexengine.core.errors import InputMissingError
from dexengine.record.text import nouns as NN
from dexengine.record.text import qa_words as QW
from dexengine.record.text.scene_words import (
    ARCTIC_PARTS,
    BODY,
    BODY_DIR,
    CENTER,
    CONTACT,
    CONTENTS,
    CUT_OBJ,
    HANDOVER_DETS,
    HANDOVER_VERB_ONLY,
    HANDOVER_WORDS,
    IRREGULAR,
    MOVE_PREP,
    NET_BACK_CM,
    NON_MANIP,
    PART,
    POSITION,
    PROMPT_LISTS,
    REL_SPAN,
    RELATION,
    SCREW_VERB,
    SIDE,
    SIDE_PREP,
    TACO_OUT,
    TACO_PUT,
)
from dexengine.record.text.vocab import META_RE, PROMPT_TEXTS


def _ws(s) -> list[str]:
    return re.findall(r"[a-z]+", str(s or "").lower())


def _has(phrase: str, ws: list[str]) -> bool:
    p = phrase.split()
    return any(ws[i : i + len(p)] == p for i in range(len(ws) - len(p) + 1))


def observed(rec: dict) -> set[str]:
    """Phases measured in this replay — the other_replay rows of the request phase table (filled for other replays)
    are excluded (R1, R10)."""
    return {str(p.get("phase")) for p in rec.get("phases") or [] if not p.get("other_replay")}


def _texts(out: dict) -> list[str]:
    return [str(out.get("instruction") or ""), *map(str, out.get("paraphrases") or [])]


def _all_texts(out: dict) -> list[str]:
    t = [*_texts(out), str(out.get("short") or ""), str(out.get("detailed") or "")]
    for x in out.get("subtasks") or []:
        if isinstance(x, dict):
            t += [str(x.get("instruction") or ""), *map(str, x.get("paraphrases") or [])]
    return t


def _clauses(texts) -> list[list[str]]:
    return [_ws(c) for t in texts for c in re.split(r"[,;.]", str(t).lower())]


def handover_claims(texts) -> list[str]:
    """Handover phrases found in the text fields (sorted, unique)."""
    got = set()
    for ws in _clauses(texts):
        for p in HANDOVER_WORDS:
            n = len(p.split())
            at = [i for i in range(len(ws) - n + 1) if ws[i : i + n] == p.split()]
            if at and (p not in HANDOVER_VERB_ONLY or any(i == 0 or ws[i - 1] not in HANDOVER_DETS for i in at)):
                got.add(p)
    return sorted(got)


def handover_state(rec: dict) -> tuple[str, str]:
    """(all | some | none, evidence text). With facts (motion.hand_contact) the measurement wins: any object with a
    handover '<giver>→<receiver>' in every episode = all, only 'some' = some, else none. Without facts the case
    decides (HANDOVER in the case name)."""
    hc = (rec.get("motion") or {}).get("hand_contact") or []
    if hc:
        ho = [(o["obj"], o["handover"]) for o in hc if o.get("handover")]
        s = "all" if any(h != "some" for _, h in ho) else "some" if ho else "none"
        return s, "measured " + (" · ".join(f"{n} {h}" for n, h in ho) or "no handover")
    case = str(rec.get("case") or "")
    return ("all" if "HANDOVER" in case.upper() else "none"), f"case {case or 'none'}"


def handover_problems(rec: dict, out: dict) -> list[str]:
    """R4 a handover (all) but no handover phrase in instruction, paraphrases or detailed. R9 no handover (none or
    some) but a handover phrase in any text field (phase texts included)."""
    s, why = handover_state(rec)
    if s == "all":
        if not handover_claims(_texts(out) + [str(out.get("detailed") or "")]):
            return [
                f"R4 handover ({why}) but no handover words — one hand passes the object to the other hand in this "
                "clip; say "
                "so in the instruction (e.g. 'the right hand hands it to the left hand')"
            ]
    elif said := handover_claims(_all_texts(out)):
        return [
            f"R9 no handover ({why}) but handover words {said} — no hand passes the object to the other hand in "
            "every replay; do not say it is passed"
        ]
    return []


def _mask(ws: list[str], names: list[list[str]]) -> list[str]:
    """Replace object name word sequences (left shoe) by the single word 'obj' so left/right inside a name is not
    read as a direction (R5)."""
    out, i = [], 0
    while i < len(ws):
        n = next((n for n in names if ws[i : i + len(n)] == n), None)
        out.append("obj" if n else ws[i])
        i += len(n) if n else 1
    return out


def direction_words(out: dict, names=()) -> list[str]:
    """R5 — table / camera / body direction forms in all text fields (sorted, unique). names = object names (a
    direction word inside a name is not a direction — 'toward the left shoe')."""
    got = set()
    seqs = sorted((w for n in names if set(w := _ws(n)) & set(SIDE + CENTER)), key=len, reverse=True)
    for ws in (_mask(c, seqs) for c in _clauses(_all_texts(out))):
        got |= {b for b in BODY_DIR if _has(b, ws)}
        for i, w in enumerate(ws):
            nx = ws[i + 3] if i + 3 < len(ws) else ""
            if w in ("leftward", "leftwards", "rightward", "rightwards"):
                got.add(w)
            elif w == "from" and ws[i + 1 : i + 2] and ws[i + 1] in SIDE and ws[i + 2 : i + 3] == ["to"]:
                got.add(" ".join(ws[i : i + 4]))
            elif ws[i + 1 : i + 2] in (["the"], ["its"], ["their"]) and ws[i + 2 : i + 3]:
                k = ws[i + 2]
                if (w in SIDE_PREP and k in SIDE and nx not in BODY) or (w in MOVE_PREP and k in CENTER and nx != "of"):
                    got.add(" ".join(ws[i : i + 3]))
    return sorted(got)


def _words_of(names) -> set[str]:
    return {w for n in names for w in _ws(str(n).replace("_", " ")) if len(w) >= 3}


def _backed(rec: dict) -> set[str]:
    """Object names that back a relation word: main, support, and objects near a moved object's end (ends_near)."""
    near = [(m.get("ends_near") or {}).get("obj") for m in (rec.get("motion") or {}).get("objects") or []]
    return {o.get("name") for o in rec.get("objects") or [] if o.get("role") in ("main", "support")} | {
        n for n in near if n
    }


def fill_prompt(text: str) -> str:
    """Fill the prompt's {{NAME}} slots from PROMPT_LISTS (word lists), the thresholds and PROMPT_TEXTS; dies when a
    slot is left unfilled."""
    for k, v in PROMPT_LISTS.items():
        text = text.replace("{{" + k + "}}", ", ".join(f'"{w}"' for w in v))
    from dexengine.record.text import check as CHK  # threshold owners (lazy: they import this module)
    from dexengine.record.text import motion as MOT
    from dexengine.record.text import rules_facts as FR

    j = MOT.JOINT_UNITS
    for k, v in {
        **PROMPT_VALUES,
        "LIFT_CM": CHK.LIFT_CM,
        "STILL_MOVE_CM": QW.STILL_MOVE_CM,
        "JOINT_DEG": j["hinge"][2],
        "JOINT_CM": j["slide"][2],
        "PICK_HELD_CM": FR.PICK_HELD_CM,
    }.items():
        text = text.replace("{{" + k + "}}", f"{v:g}")
    for k, v in PROMPT_TEXTS.items():
        text = text.replace("{{" + k + "}}", v)
    if left := re.findall(r"\{\{(\w+)\}\}", text):
        raise InputMissingError(f"[TEXT_RULES] prompt slots {sorted(set(left))} have no value — PROMPT_LISTS")
    return text


def scene_problems(rec: dict, out: dict) -> list[str]:
    """Reasons from R1-R5, then v3_problems (R13-R31); duplicates removed, order kept."""
    why = []
    phases = observed(rec)
    verb = (_ws(out.get("verb")) or _ws(out.get("instruction")) or [""])[0]
    if rec.get("segment") and phases and "manipulate" not in phases and verb not in {w.split()[0] for w in NON_MANIP}:
        why.append(f"R1 no manipulate phase in the segment ({sorted(phases)}) — manipulation verb {verb!r}")
    src = _ws((rec.get("source") or {}).get("text"))  # R2 applies at every confidence level
    objs = _words_of(o.get("name") for o in rec.get("objects") or [] if o.get("role") == "extra") - _words_of(
        _backed(rec)
    )
    for t in _texts(out):
        for clause in re.split(r"[,;.]", t.lower()):
            ws = _ws(clause)
            for r in RELATION:
                p = r.split()
                at = [i for i in range(len(ws) - len(p) + 1) if ws[i : i + len(p)] == p]
                if any(set(ws[i + len(p) : i + len(p) + REL_SPAN]) & objs for i in at) and not _has(r, src):
                    why.append(
                        f"R2 relation words {r!r} + object without support — {t[:60]} — relate to another "
                        "object only "
                        "when the source text, its support role or its measured end position backs it"
                    )
                    break
    fields = [
        (k, set(_ws(t)) & set(PART))
        for k, t in [
            ("instruction", out.get("instruction")),
            *[(f"paraphrase{i}", p) for i, p in enumerate(out.get("paraphrases") or [])],
            ("detailed", out.get("detailed")),
        ]
    ]
    named = [(k, s) for k, s in fields if s]
    if any(not (a & b) for i, (_, a) in enumerate(named) for _, b in named[i + 1 :]):
        why.append(f"R3 part noun mismatch {[(k, sorted(s)) for k, s in named]}")
    why += handover_problems(rec, out)
    src = _ws((rec.get("source") or {}).get("text"))
    if bad := [d for d in direction_words(out, [o.get("name") for o in rec.get("objects") or []]) if not _has(d, src)]:
        why.append(
            f"R5 desk/camera/body-frame direction words {bad} (prompt rule 4) — no directions; say where it "
            "goes "
            "relative to a given object, or 'to a new spot'"
        )
    why += v3_problems(rec, out)
    return [w for i, w in enumerate(why) if w not in why[:i]]


# ── R13 · R14 · R15 ──
#: (Degree and number words are not checked: correct texts quote the measured slightly / cm / degrees; the prompt
#: rule 'do not embellish degree or numbers' covers them.)
#: R14 flip words <-> measured rotation: a flip only when a moved object rotates FLIP_DEG or more.
FLIP_DEG = 120.0
FLIP_RE = re.compile(
    r"\b(turn|turns|turned|turning)\b(?:\s+\w+){0,3}?\s+over\b|\bflip(s|ped|ping)?\b|"
    r"\bupside\s+down\b|\binvert(s|ed|ing)?\b"
)
#: Prompt generation number, written into manifest.json and every text row (prompt_gen). Raise it when the rules
#: (prompt or checker) change so that existing rows must be regenerated.
PROMPT_GEN = 3


#: Threshold values for the prompt's {{NAME}} slots (fill_prompt).
PROMPT_VALUES = {"FLIP_DEG": FLIP_DEG}
#: R13 hand assignment: clauses are also split at these words (two hands' tasks in one sentence).
CLAUSE_RE = re.compile(r"[,;.]|\b(?:while|and|as|then|whereas|before|after)\b")


def _side_clauses(out: dict) -> list[list[str]]:
    t = [*_texts(out), str(out.get("detailed") or "")]
    for x in out.get("subtasks") or []:
        if isinstance(x, dict):
            t += [str(x.get("instruction") or ""), *map(str, x.get("paraphrases") or [])]
    return [_ws(c) for s in t for c in CLAUSE_RE.split(str(s).lower()) if c]


def _contact(ws: list[str]) -> bool:
    """The word run (hand -> object, or before the object) has a contact verb with no position word after it."""
    k = max(
        (
            i
            for i, w in enumerate(ws)
            if w in IRREGULAR
            or any(
                w
                in (
                    v,
                    v + "s",
                    v + "es",
                    v + "ed",
                    v + "d",
                    v + "ing",
                    v[:-1] + "ing",
                    v + v[-1] + "ing",
                    v + v[-1] + "ed",
                    v[:-1] + "ies",
                    v[:-1] + "ied",
                )
                for v in CONTACT
            )
        ),
        default=None,
    )
    return k is not None and not set(ws[k + 1 :]) & set(POSITION)


def held_by(ws: list[str], side: dict, skip: set = frozenset()) -> list[tuple[str, str]]:
    """(hand side, object word that hand holds or touches) in one clause — '<side> hand <verb> the <object>' takes the
    first object word after the hand; '... <object> with|in|using the <side> hand' takes the nearest object word
    before the hand (other objects of the clause are not looked at — in 'the right hand moves the brush over the pan'
    the pan is not the right hand's object). Without a contact verb (CONTACT), or for an object after a position
    word, it is not a claim (_contact — 'the left hand stays by the pan')."""
    got = []
    for i in range(len(ws) - 1):
        if ws[i] in ("left", "right") and ws[i + 1] in ("hand", "hands"):
            back = i >= 2 and ws[i - 1] in ("the", "its") and ws[i - 2] in ("with", "in", "using", "by")
            at = range(i - 3, -1, -1) if back else range(i + 2, len(ws))
            k = next((j for j in at if ws[j] in side and (back or j not in skip)), None)  # skip = place phrases
            if k is not None and _contact(ws[:k] if back else ws[i + 2 : k]):
                got.append((ws[i], ws[k]))
    return got


def v3_problems(rec: dict, out: dict) -> list[str]:
    """R13 hand assignment <-> metadata hands and object sides, R14 flip <-> measured rotation, R15 talk about the
    source, labels or metadata; then rules_qa (R16-R31)."""

    why, alls = [], _all_texts(out)
    h, cl = rec.get("hands"), _side_clauses(out)
    if h in ("left", "right"):
        other = "right" if h == "left" else "left"
        if any(_has(f"{other} hand", ws) or _has(f"{other} hands", ws) for ws in cl):
            why.append(f"R13 hand assignment — meta hand {h} but {other} hand — only the {h} hand acts in this clip")
    elif h == "both":
        side = {}
        for o in rec.get("objects") or []:
            if o.get("side") in ("left", "right"):
                for w in NN.toks(o["name"]) | {t for a in o.get("aliases", []) for t in NN.toks(a)}:
                    side.setdefault(w, set()).add(o["side"])
        bad = sorted({f"{s} hand + {w}" for ws in cl for s, w in held_by(ws, side) if s not in side[w]})
        if bad:
            why.append(
                f"R13 hand assignment ↔ object side (meta objects side) {bad} — say which hand holds which "
                "object only as "
                "the metadata gives it"
            )
    mo = [o for o in (rec.get("motion") or {}).get("objects") or [] if o.get("moved")]
    if (
        mo
        and any(FLIP_RE.search(str(t).lower()) for t in alls)
        and max(float(o.get("rot_max_deg") or 0) for o in mo) < FLIP_DEG
    ):
        why.append(
            f"R14 flip words ↔ measured max rotation {max(float(o.get('rot_max_deg') or 0) for o in mo):.0f}° < "
            f"{FLIP_DEG:.0f}° — say 'rotate' or 'tilt', not 'turn over' or 'flip'"
        )
    if meta := sorted({m.group(0).lower() for t in alls for m in META_RE.finditer(str(t))}):
        why.append(
            f"R15 talk of source · label · meta {meta} — describe only the clip; never mention the source, "
            "its labels or the metadata"
        )
    from dexengine.record.text import rules_qa as QA

    return why + QA.qa_problems(rec, out)


# ── R6 · R7 · R8 · R10 · R11: words the measurement, the source text or the object list do not back ──
def _net(rec: dict, name: str) -> float | None:
    for m in (rec.get("motion") or {}).get("objects") or []:
        if m.get("obj") == name and m.get("net_cm") is not None:
            return sum(float(x) ** 2 for x in m["net_cm"]) ** 0.5
    return None


def source_wording_problems(rec: dict, out: dict, vocab: frozenset[str]) -> list[str]:
    """R6 TACO label copied literally, R7 objects only in the source text, R8 ARCTIC 'use' part words, R10
    manipulation verb without motion in this segment, R11 contents words."""

    why, ds, src = [], rec.get("dataset"), rec.get("source") or {}
    f, alls, ins = src.get("fields") or {}, _all_texts(out), set(_ws(out.get("instruction")))
    allw = set(_ws(" ".join(alls)))
    objw = set().union(
        *[NN.toks(o["name"]) | {t for a in o.get("aliases", []) for t in NN.toks(a)} for o in rec.get("objects") or []],
        *[NN.toks(n) for n in rec.get("scene_objects") or []],
    )
    if ds == "taco":
        v, tool = str(f.get("verb") or ""), str(f.get("tool") or "")
        net = next(
            (
                n
                for o in rec.get("objects") or []
                if tool and tool in o["name"] and (n := _net(rec, o["name"])) is not None
            ),
            None,
        )
        said_out = any(_has(p, w) for w in _clauses(alls) for p in ("put out", "puts out"))
        put_in = set(TACO_PUT) & ins and {"in", "into", "inside"} & ins and not set(TACO_OUT) & ins
        low = " ".join(str(t).lower() for t in alls)
        bad = (
            "cut"
            if v == "cut" and CUT_OBJ.search(low)
            else "put out"
            if v == "put out" and said_out
            else "screw"
            if v == "screw" and SCREW_VERB.search(low)
            else "put in"
            if v == "put in" and put_in and net is not None and net < NET_BACK_CM
            else None
        )
        if bad:
            why.append(
                f"R6 literal TACO label {bad!r} — do not copy the label verb; say the physical motion"
                + (" (the tool goes in and comes back out)" if bad == "put in" else "")
            )
    if src.get("kind") not in ("intent", "action_label"):
        srcw = NN.toks(str(src.get("text") or "")) - NN.toks(str(f.get("verb") or ""))
        said = {
            w
            for ws, st in NN.clause_words([out.get("instruction"), out.get("detailed")])  # noun slots only —
            for k, w in enumerate(ws)
            if not NN.not_noun_slot(ws, k, vocab, st)
        }  # 'Scoop with' = verb
        if absent := sorted(w for w in said & set(vocab) if w not in objw and w in srcw):
            why.append(f"R7 object not in the scene {absent} — only in the source text, not in this clip's object list")
    if ds == "arctic" and str(src.get("text")) == "use" and (got := sorted(allw & set(ARCTIC_PARTS))):
        why.append(f"R8 ARCTIC use part words {got} — the label is only 'use'; do not name parts")
    mo, verb = rec.get("motion") or {}, (_ws(out.get("verb")) or _ws(out.get("instruction")) or [""])[0]
    if (
        "manipulate" in observed(rec)
        and mo.get("objects")
        and not any(o.get("moved") for o in mo["objects"])
        and "ART" not in str(rec.get("case") or "")
        and verb not in {w.split()[0] for w in NON_MANIP}
    ):
        why.append(
            f"R10 manipulation verb {verb!r} but no object moves in this segment — describe what happens in "
            "this segment"
        )
    if got := sorted(w for w in allw & set(CONTENTS) if w not in objw):
        why.append(f"R11 content words {got} — nothing is inside; describe the tool motion only")
    return why
