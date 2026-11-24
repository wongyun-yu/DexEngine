"""Output contract of the clip instruction: the JSON schema (structured output, output_config.format) and the checks
(body words, length, objects, hands, English, phase texts = the request's phase table in order). SCHEMA is the one
owner of the schema; validate_one reads it and requests send api_schema(SCHEMA). Word and noun reading: nouns.
"""

from __future__ import annotations

import json

from dexengine.record.phases import PHASES
from dexengine.record.text import motion as MOT
from dexengine.record.text import vocab as VOC
from dexengine.record.text.nouns import foreign_objects, object_ok, unsupported, words

#: Output schema (a model input: the descriptions are part of the request). validate_one reads it; requests send
#: api_schema(SCHEMA) as structured output.
SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string", "description": "imperative task instruction, 4-20 words"},
        "paraphrases": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
            "description": "four different wordings of the same instruction, each 4-20 words",
        },
        "short": {"type": "string", "description": "2-6 word task name, e.g. 'pour from teapot'"},
        "detailed": {
            "type": "string",
            "description": "1-2 sentences: what is done, with which hand(s), to which objects",
        },
        "verb": {"type": "string", "description": "main action verb in base form"},
        "objects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "object names used in the text, each matching a given object (natural form allowed)",
        },
        "hands": {"type": "string", "enum": ["left", "right", "both", "unspecified"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence": {
            "type": "array",
            "items": {"type": "string", "enum": ["source_text", "objects", "motion", "dataset_note", "frames"]},
        },
        "notes": {"type": "string", "description": "short reason when confidence is not high, else empty"},
        "subtasks": {
            "type": "array",
            "description": "one entry per row of the given phase table, same order",
            "items": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string", "enum": list(PHASES)},
                    "instruction": {
                        "type": "string",
                        "description": "imperative instruction for this phase only, 3-15 words",
                    },
                    "paraphrases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "two different wordings, each 3-15 words",
                    },
                },
                "required": ["phase", "instruction", "paraphrases"],
            },
        },
    },
    "required": [
        "instruction",
        "paraphrases",
        "short",
        "detailed",
        "verb",
        "objects",
        "hands",
        "confidence",
        "evidence",
        "notes",
        "subtasks",
    ],
}
WORDS_MIN, WORDS_MAX, SHORT_MAX = 4, 20, 6
SUB_MIN, SUB_MAX = 3, 15
#: Words the texts must not use: no body (robot or human) is mentioned. Matched per lower-case word.
BANNED = ("robot", "robotic", "human", "person", "people", "man", "woman", "operator", "agent", "gripper", "mano")
#: Non-ASCII punctuation the model likes -> ASCII (same meaning: curly quotes, dashes, ellipsis, no-break space).
PUNCT = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
}


# ───────────────────────── checks ─────────────────────────
def normalize(out: dict) -> dict:
    """ASCII punctuation in the text fields (before the checks; the stored value too). Other non-ASCII stays and
    the checks reject it."""

    def n(t):
        for k, v in PUNCT.items():
            t = t.replace(k, v)
        return t.strip()

    o = dict(out)
    for k in ("instruction", "short", "detailed", "notes"):
        if isinstance(o.get(k), str):
            o[k] = n(o[k])
    if isinstance(o.get("paraphrases"), list):
        o["paraphrases"] = [n(t) if isinstance(t, str) else t for t in o["paraphrases"]]
    if isinstance(o.get("subtasks"), list):

        def sub(x):
            if not isinstance(x, dict):
                return x
            ins = x.get("instruction")
            return dict(
                x,
                instruction=n(ins) if isinstance(ins, str) else ins,
                paraphrases=[n(t) if isinstance(t, str) else t for t in x.get("paraphrases") or []],
            )

        o["subtasks"] = [sub(x) for x in o["subtasks"]]
    return o


def subtask_problems(rec: dict, subs) -> list[str]:
    """Phase texts: the same phases in the same order as the request phase table (phases.full_table — always the
    five PHASES), per phase one instruction + two paraphrases (SUB_MIN-SUB_MAX words, no body words, English)."""
    from dexengine.record import phases as PH  # same full_table that build_dir puts into the request

    want = [p["phase"] for p in PH.full_table(rec.get("phases") or [])]
    if not isinstance(subs, list) or any(not isinstance(x, dict) for x in subs):
        return ["subtasks shape"]
    got = [x.get("phase") for x in subs]
    if got != want:
        return [f"phases {got} ≠ phase table {want}"]
    why = []
    for x in subs:
        texts = [x.get("instruction"), *(x.get("paraphrases") or [])]
        if (
            len(x.get("paraphrases") or []) != 2
            or any(not isinstance(t, str) for t in texts)
            or len({t.strip().lower() for t in texts}) != 3
        ):
            why.append(f"{x.get('phase')}: not 2 paraphrases, or repeated text")
            continue
        for t in texts:
            n = len(words(t))
            if not SUB_MIN <= n <= SUB_MAX:
                why.append(f"{x['phase']}: length {n} words — {t[:60]}")
            if any(w in BANNED for w in words(t)):
                why.append(f"{x['phase']}: body word — {t[:60]}")
            if not t.isascii():
                why.append(f"{x['phase']}: not English — {t[:60]}")
    return why


#: Plausibility check (no API): output verb <-> measured motion (source summary motion.objects, or the facts —
#: rules_facts.verb_moved). Thresholds from the measured distribution of moved objects: lift 2 cm (above the 0-1 cm
#: noise), horizontal move 1 cm (the phases.MOVE_M motion definition) or a lift, rotation 10 degrees. With a label,
#: the label verb must appear in the text.
LIFT_CM, MOVE_CM_MIN, ROT_DEG_MIN = 2.0, 1.0, 10.0
VERB_CLASS = {
    "lift": "lift",
    "raise": "lift",
    "pick": "lift",
    "hoist": "lift",
    "move": "move",
    "carry": "move",
    "slide": "move",
    "push": "move",
    "pull": "move",
    "transfer": "move",
    "bring": "move",
    "drag": "move",
    "shift": "move",
    "relocate": "move",
    "rotate": "rotate",
    "turn": "rotate",
    "twist": "rotate",
    "flip": "rotate",
    "spin": "rotate",
    "tilt": "rotate",
    "pivot": "rotate",
}
GENERIC_LABEL = ("use", "grab", "inspect", "offhand", "hold", "pass", "none", "grasp", "touch")


def plausible(rec: dict, out: dict) -> list[str]:
    """Reasons where the output disagrees with the measurement or the label. Verb = class of the verb field (else
    the first instruction word); passes when any moved object exceeds that class's threshold (clips without a motion
    summary are not verb-checked)."""
    from dexengine.record.text import rules_facts as FR

    why = []
    verb = (words(str(out.get("verb") or "")) or words(str(out.get("instruction") or "")) or [""])[0]
    cls = VERB_CLASS.get(verb)
    mo = [o for o in (rec.get("motion") or {}).get("objects", []) if o.get("moved")]
    if cls and rec.get("motion"):

        def flat(o):
            n = o.get("net_cm") or [0, 0]
            return max(abs(n[0]), abs(n[1]))

        ok = {
            "lift": lambda o: (o.get("lift_max_cm") or 0) >= LIFT_CM,
            "move": lambda o: flat(o) >= MOVE_CM_MIN or (o.get("lift_max_cm") or 0) >= LIFT_CM,
            "rotate": lambda o: (o.get("rot_max_deg") or 0) >= ROT_DEG_MIN,
        }[cls]
        if (
            not any(ok(o) for o in mo)
            and not FR.verb_moved(rec, cls)
            and not any(j.get("moved") for j in MOT.rec_joints(rec))
        ):
            why.append(
                f"verb {verb}({cls}) ↔ no measured motion (lift ≥ {LIFT_CM} cm · move ≥ {MOVE_CM_MIN} cm · "
                f"rotation ≥ {ROT_DEG_MIN}°, as applies)"
            )
    src = rec.get("source") or {}
    labeled = src.get("kind") in ("intent", "action_label")
    label = str((src.get("fields") or {}).get("verb") or (src.get("text") if labeled else "") or "")
    lv = [w for w in words(label.replace("-", " ").replace("_", " ")) if len(w) >= 3 and w not in GENERIC_LABEL]
    if lv and FR.label_due(rec, lv[0]):
        said = " ".join(words(" ".join(str(out.get(k) or "") for k in ("instruction", "short", "verb", "detailed"))))
        hit = VOC.label_words(src) or (lv[0][:4],)
        if not any(w in said for w in hit):
            why.append(
                f"label verb {lv[0]!r} missing from the text (dataset label {label!r} · accepted word pieces "
                f"{list(hit)})"
            )
    return why


def validate_one(rec: dict, out: dict, vocab: frozenset[str] = frozenset()) -> list[str]:
    """Rejection reasons of one output for its input record ([] = passes). The reason strings are data: they are
    stored in rejects.jsonl and sent back to the model as retry_why (retry.HINT matches their prefixes)."""
    why = []
    miss = [k for k in SCHEMA["required"] if k not in out]
    if miss:
        return [f"missing fields {miss}"]
    texts = [out["instruction"], *out["paraphrases"]]
    if len(out["paraphrases"]) != 4 or len({t.strip().lower() for t in texts}) != 5:
        why.append("not 4 paraphrases, or repeated text")
    for t in texts:
        n = len(words(t))
        if not WORDS_MIN <= n <= WORDS_MAX:
            why.append(f"length {n} words — {t[:60]}")
        if any(w in BANNED for w in words(t)):
            why.append(f"body word — {t[:60]}")
        if not t.isascii():
            why.append(f"not English — {t[:60]}")
    if not 1 <= len(words(out["short"])) <= SHORT_MAX:
        why.append("short length")
    if any(w in BANNED for w in words(out["detailed"])):
        why.append("body word in detailed")
    if rec.get("objects"):
        bad = [o for o in out["objects"] if not object_ok(o, rec)]
        if bad:
            why.append(f"object not given {bad}")
        if not out["objects"]:
            why.append("no objects")
    h = rec.get("hands")
    from dexengine.record.text import rules_facts as FR  # hands = the input hands or those touching in every replay

    if h in ("left", "right", "both") and out["hands"] not in (h, "unspecified", FR.hands_all(rec)):
        why.append(f"hands {out['hands']} ≠ input {h}")
    from dexengine.record.text import rules_scene as RULES

    return (
        why
        + unsupported(rec, out)
        + foreign_objects(rec, out, vocab)
        + plausible(rec, out)
        + subtask_problems(rec, out["subtasks"])
        + RULES.scene_problems(rec, out)
        + RULES.source_wording_problems(rec, out, vocab)
    )


#: Constraints structured output does not accept: dropped from the request only (validate_one checks the counts).
API_DROP = ("minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum", "multipleOf")


def api_schema(s):
    """SCHEMA -> structured-output schema (additionalProperties false on every object, unsupported constraints
    removed)."""
    if isinstance(s, list):
        return [api_schema(x) for x in s]
    if not isinstance(s, dict):
        return s
    o = {k: api_schema(v) for k, v in s.items() if k not in API_DROP}
    if o.get("type") == "object":
        o["additionalProperties"] = False
    return o


def output(msg: dict) -> dict | None:
    """Response message -> output fields (the JSON of the first text block); None on refusal, truncation, non-JSON or
    non-object."""
    if msg.get("stop_reason") in ("refusal", "max_tokens"):
        return None
    txt = next((c.get("text") for c in msg.get("content", []) if c.get("type") == "text"), None)
    try:
        out = json.loads(txt) if txt else None
    except ValueError:
        return None
    return out if isinstance(out, dict) else None
