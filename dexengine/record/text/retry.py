"""Retry reasons sent back to the model when a rejected clip is generated again (record field retry_why, prompt
request.retry_note). Caller: generate.cmd_redo -> for_redo.

Reasons = this round's rejection reasons ∪ the reasons of every earlier attempt of the same (dataset, clip) (texts
and rejects rows of the sibling rounds) re-checked with the current record and the current checker
(check.validate_one, vocabulary from the same rounds' records). Sending only the last attempt makes the model
oscillate between two rules; re-checking earlier attempts with the current rules keeps outdated wording out. Order:
current reasons, then earlier attempts (newest first); each reason once; prompt-generation reasons ('P ...', not
checker wording) are dropped; at most RETRY_MAX lines. Earlier outputs missing a SCHEMA field are not re-checked.
The reason strings are the checker's own wording; HINT matches their prefixes.
"""

from __future__ import annotations

from pathlib import Path

from dexengine.record.text import nouns as NN

RETRY_MAX = 12


def _read(f: Path) -> list[dict]:
    import json

    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.is_file() else []


def _history(rounds: Path, keys: set) -> dict:
    """{(dataset, clip): [earlier outputs, newest first]} — round folder names sort in build order
    (r133 < r133_redo < r135)."""
    out: dict = {}
    if not keys or not Path(rounds).is_dir():
        return out
    for d in sorted(p for p in Path(rounds).iterdir() if p.is_dir()):
        for f in ("texts.jsonl", "rejects.jsonl"):
            for r in _read(d / f):
                if (r.get("dataset"), r.get("clip")) in keys:
                    out.setdefault((r["dataset"], r["clip"]), []).append(r)
    return {k: v[::-1] for k, v in out.items()}


def _vocab(rounds: Path, recs: list) -> frozenset:

    rs = [r for f in sorted(Path(rounds).glob("*/records.jsonl")) for r in _read(f)] if Path(rounds).is_dir() else []
    return NN.object_vocab([*rs, *recs])


def _obj(rec: dict, w: str) -> dict:
    """The facts object a reason refers to (the longest name inside the reason text, else the main object, else the
    first one); {} without facts."""
    hc = (rec.get("motion") or {}).get("hand_contact") or []
    named = sorted((o for o in hc if o["obj"] in w), key=lambda o: -len(o["obj"]))
    main = [o for o in hc for m in rec.get("objects") or [] if m.get("role") == "main" and m.get("name") == o["obj"]]
    return (named or main or hc or [{}])[0]


def _form(rec: dict, o: dict, verb: str = "") -> str:
    """A correct sentence from the facts — '<hand> hold(s) the <object> and <motion> it, then <end>' (only what is
    true in every replay; an end that differs between replays is left out)."""
    from dexengine.record.text import check as CHK
    from dexengine.record.text import rules_facts as FR

    name = o.get("obj") or next((m.get("name") for m in rec.get("objects") or [] if m.get("role") == "main"), "it")
    hands = o.get("held_at_start") or o.get("lifted_by") or o.get("touched_by") or []
    act = (
        "lift"
        if o["lifted"] == "all" and not o["held_at_start"]
        else "turn"
        if o["rot_deg"][0] >= CHK.ROT_DEG_MIN
        else "raise"
        if o["lifted"] == "all"
        else "move"
        if o["move_cm"][0] >= CHK.MOVE_CM_MIN
        else ""
    )
    act = verb or act  # a reason that keeps the label verb uses that verb
    xy = o["end_xy_cm"]
    end = (
        "keep it held up"
        if o["end_up"] == "all"
        else "put it back where it was"
        if xy[1] < FR.RELOC_CM
        else "set it down at a new spot"
        if o["end_up"] == "none" and xy[0] >= FR.RELOC_CM
        else ""
    )
    fr, _, to = str(o.get("handover") or "").partition("→")  # handover in every replay ('some' is not said)
    if to:
        hands = [fr]
    give = f", give it to the {to} hand" if to else ""
    tail = "".join(x for x in (f" and {act} it" if act else "", give, f", then {end}" if end else ""))
    if not hands:
        return f"Hold the {name}{tail}"
    who = "both hands hold" if len(hands) == 2 else f"the {hands[0]} hand holds"
    third = {
        "keep": "keeps",
        "put": "puts",
        "set": "sets",
        "lift": "lifts",
        "turn": "turns",
        "raise": "raises",
        "move": "moves",
        "give": "gives",
    }
    return f"{who} the {name}" + " ".join(
        third.get(t, t + "s" if verb and t == verb else t) if len(hands) == 1 else t for t in tail.split(" ")
    )


#: Reason prefix (the checker's wording) -> lead of the correct form (English, sent to the model); _form fills the
#: form from the facts. Reasons without a matching prefix are sent alone.
HINT = (
    ("R4 ", "say the pass, e.g. 'the right hand hands it to the left hand'"),
    (
        "length ",
        "every instruction and paraphrase has 4-20 words; drop manner words, keep the motion and the end, e.g. '{f}'",
    ),
    ("manipulate: length", "each step sentence is short; e.g. 'Turn the {n}.'"),
    ("not in meta", "name objects only with the given names (no extra nouns); call it 'the {n}', e.g. '{f}'"),
    ("object not given", "the objects field lists only the given object names, e.g. ['{n}']"),
    ("label verb", "keep the dataset label verb {v} in the instruction (as the motion allows), e.g. '{f}'"),
    ("hands ", "the hands field is the input hands or the hands that touch it in every replay, e.g. '{f}'"),
    ("verb ", "use a verb that the measured motion shows, e.g. '{f}'"),
    ("R", "a sentence that fits the measured facts, e.g. '{f}'"),
)


def hint(rec: dict, w: str) -> str:
    """One reason + a correct form built from the facts ('<reason> → <lead with the form>')."""
    head = next((t for h, t in HINT if w.startswith(h)), "")
    if not head or "e.g." in w:
        return w
    o = _obj(rec, w)
    if not o:  # no facts: no form is invented
        return w
    if not o.get("touched_by"):  # an object no hand touches (a pan the tool works on):
        return (
            f"{w} → the {o['obj']} is not held; say it moves a little as the tool works on it, not that it stays still"
            if w.startswith("R23")
            else w
        )  # a held-form would make R28 and R13 fight
    v = w.split("'")[1] if w.count("'") >= 2 else ""
    f = _form(rec, o, v if w.startswith("label verb") else "")
    return f"{w} → {head.format(f=f, n=o.get('obj', 'object'), v=repr(v))}"


def reasons(rec: dict, now: list, prior: list, vocab: frozenset) -> list[str]:
    """Current reasons ∪ earlier attempts re-checked with the current checker (order, dedup, P and the cap as in the
    module docstring), each line with its correct form (hint)."""
    from dexengine.record.text import check as CHK

    req = CHK.SCHEMA["required"]
    seen, got = set(), []
    for w in [*now, *(w for o in prior if all(k in o for k in req) for w in CHK.validate_one(rec, o, vocab))]:
        if not str(w).startswith("P ") and w not in seen:
            seen.add(w)
            got.append(w)
    return [hint(rec, w) for w in got[:RETRY_MAX]]


def for_redo(src: Path) -> dict[str, list[str]]:
    """Rejected rows of one round -> {custom_id: this round's reasons ∪ earlier attempts} (earlier attempts = the
    sibling rounds of src)."""
    from dexengine.record.text import generate as CTX

    src = Path(src)
    rej = _read(src / "rejects.jsonl")
    recs = {CTX.custom_id(r): r for r in _read(src / "records.jsonl")}
    hist = _history(src.parent, {(r["dataset"], r["clip"]) for r in rej})
    vocab = _vocab(src.parent, []) if rej else frozenset()
    return {
        r["custom_id"]: reasons(recs[r["custom_id"]], r["why"], hist.get((r["dataset"], r["clip"]), []), vocab)
        for r in rej
    }
