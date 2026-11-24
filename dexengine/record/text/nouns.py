"""Word and noun-slot reading of the text checker (check.validate_one): tokens, the metadata vocabulary, words the
metadata must back (material, color, scene), object-name slots, foreign objects and object-name matching."""

from __future__ import annotations

import itertools
import re

from dexengine.record.text.vocab import CONTAINER, NOT_OBJECT

#: Stop words dropped by toks (function words, not object names).
STOP = {"the", "a", "an", "of", "to", "from", "in", "into", "on", "with", "and", "one", "another", "position", "other"}


# ───────────────────────── words ─────────────────────────
def words(s: str) -> list[str]:
    return [w.strip(".,;:!?\"'()").lower() for w in s.split() if w.strip(".,;:!?\"'()")]


#: Words rejected unless the metadata (object names, aliases, source text) has them — prompt rule 3 (material,
#: color) and rule 4 (table / camera references). Matched per word in every text field.
MATERIAL = (
    "wooden",
    "wood",
    "metal",
    "metallic",
    "plastic",
    "glass",
    "ceramic",
    "porcelain",
    "paper",
    "cardboard",
    "rubber",
    "steel",
    "iron",
    "aluminum",
    "leather",
    "fabric",
    "cotton",
    "silicone",
    "stone",
    "marble",
)
COLOR = (
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
    "silver",
    "gold",
    "golden",
    "beige",
    "violet",
    "cyan",
    "magenta",
    "turquoise",
    "navy",
)
SCENE = ("table", "tabletop", "desk", "countertop", "camera", "pedestal", "platform")
#: The render support (pedestal under an object held in the air — 'set it down on the stand') as a noun: 'stand' is
#: also a verb ('stand it upright'), so it counts as a scene word only after a determiner or modifier.
STAND_HEAD = ("the", "a", "an", "its", "small", "little", "round", "wooden", "metal", "tall")


def toks(s: str) -> set[str]:
    """Word tokens (underscores and hyphens split): digit-only words and stop words removed; the form without
    trailing digits is added ('cactus3' -> cactus, 'cube1' -> cube)."""
    ws = {w for w in words(s.replace("_", " ").replace("-", " ")) if not w.isdigit() and w not in STOP}
    return ws | {w.rstrip("0123456789") for w in ws if len(w.rstrip("0123456789")) >= 3}


def meta_tokens(rec: dict) -> set[str]:
    """Words the metadata says: object names, aliases, source text (e.g. OakInk2 '... tool rack'), source
    actions."""
    given = set()
    for o in rec.get("objects", []):
        given |= toks(o["name"])
        given |= {t for al in o.get("aliases", []) for t in toks(al)}
    src = rec.get("source") or {}
    given |= toks(str(src.get("text") or ""))
    for x in (src.get("fields") or {}).get("actions") or []:
        given |= toks(str(x.get("action") or ""))
    return given


def unsupported(rec: dict, out: dict) -> list[str]:
    """Material, color and table/camera words in the text fields that the metadata does not have (rules 3, 4)."""
    texts = [out.get("instruction"), *(out.get("paraphrases") or []), out.get("short"), out.get("detailed")]
    for x in out.get("subtasks") or []:
        if isinstance(x, dict):
            texts += [x.get("instruction"), *(x.get("paraphrases") or [])]
    given = meta_tokens(rec)
    acr = {g for g in given if g.isalpha() and 2 <= len(g) <= 3}
    got = set()
    for t in texts:
        if not isinstance(t, str):
            continue
        ws = words(t)  # an expanded metadata acronym ('orange juice' <-> 'oj') is backed by the metadata
        cover = {
            i
            for k in range(len(ws))
            for g in acr
            if "".join(x[:1] for x in ws[k : k + len(g)]) == g
            for i in range(k, k + len(g))
        }
        cover |= {
            i
            for k in range(len(ws))
            for m in (2, 3)
            if len(ws[k : k + m]) == m and "".join(ws[k : k + m]) in given
            for i in range(k, k + m)
        }
        got |= {w.rstrip("s") if w.rstrip("s") in SCENE else w for i, w in enumerate(ws) if i not in cover}
        got |= {
            "pedestal"
            for i, w in enumerate(ws)
            if w in ("stand", "stands") and i and ws[i - 1] in STAND_HEAD and "stand" not in given
        }
    why = []
    for kind, vocab in (("material", MATERIAL), ("color", COLOR), ("desk/camera frame", SCENE)):
        bad = sorted(w for w in got & set(vocab) if w not in given)
        if bad:
            why.append(f"not in meta: {kind} words {bad}")
    return why


def object_vocab(recs) -> frozenset[str]:
    """Object vocabulary = union of object name and alias words of the given records (built from the metadata each
    time): alphabetic, >= 3 letters (no id forms), without NOT_OBJECT and the material, color and scene words
    (unsupported checks those)."""
    out = set()
    for r in recs:
        for o in r.get("objects", []):
            out |= toks(o["name"]) | {t for al in o.get("aliases", []) for t in toks(al)}
    return frozenset(
        w for w in out if w.isalpha() and w.isascii() and len(w) >= 3 and w not in NOT_OBJECT + MATERIAL + COLOR + SCENE
    )


#: Closed word classes for the part-of-speech slot test (function words, not object names): subject and relative
#: pronouns / determiners and possessives.
SUBJECT = ("i", "you", "he", "she", "it", "we", "they", "which", "who")
DET = (
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "its",
    "their",
    "his",
    "her",
    "each",
    "both",
    "another",
    "other",
    "every",
    "either",
    "neither",
    "some",
    "any",
    "one",
)


def one_off(a: str, g: str) -> bool:
    """Word a (>= 6 letters) minus one letter is inside metadata word g — a one-letter spelling difference of an
    official name (gamecontroler <-> controller)."""
    return len(a) >= 6 and any(a[:i] + a[i + 1 :] in g for i in range(len(a)))


def not_noun_slot(ws: list[str], k: int, heads=frozenset(), start: bool = False) -> bool:
    """ws[k] (a word of one clause, split at commas and periods) is grammatically not an object noun slot (not an
    object mention even when it is in the vocabulary). A participle (-ing) is a noun only after a determiner and
    before an object name word (heads: 'the standing frame'); otherwise a modifier ('set it down standing'). Any
    other word right after a determiner is a noun. Otherwise a verb slot: sentence start (imperative — 'Scoop out
    ...', or after a leading adverb LEAD — 'Then brush ...'), after a subject pronoun, right after 'to', followed by a
    determiner, an object pronoun, be/not or a particle ('Brush the box', 'dish out'), a one-word clause, VERB_PREP."""
    prev = ws[k - 1] if k else ""
    nxt = ws[k + 1] if k + 1 < len(ws) else ""
    if ws[k].endswith("ing"):
        return prev not in DET or bool(nxt) and nxt not in heads
    if prev in DET:
        return False
    return (
        start
        and (k == 0 or k == 1 and prev in LEAD)
        or prev in SUBJECT
        or prev == "to"
        or nxt in DET
        or nxt in PARTICLE
        or len(ws) == 1
        or nxt in VERB_PREP
        and (k == 0 or prev in ("and", "then", "or"))
        or nxt in ("be", "not", "it", "them")
    )


#: Particles (closed class): right after a word without a determiner the word is a phrasal verb (scoop out).
LEAD = ("then", "first", "next", "finally", "now", "carefully", "gently", "slowly")  # leading adverbs
PARTICLE = ("up", "down", "out", "off", "over", "in", "into", "onto", "away", "back", "through", "around", "aside")
#: A clause-initial word, or one right after and / then / or, directly followed by a tool or place preposition is a
#: verb ('..., scoop with it', 'and scoop along'); after a name word it is a compound noun ('tea powder with').
VERB_PREP = ("with", "along", "inside", "across", "from", "against", "toward", "towards")


def clause_words(texts) -> list[tuple[list[str], bool]]:
    """Word lists per clause (split at commas, semicolons, colons) with a sentence-start flag — the input of
    not_noun_slot (R7 reads it too)."""
    return [
        (words(c), j == 0)
        for t in texts
        if isinstance(t, str)
        for sent in re.split(r"[.!?]", t)
        for j, c in enumerate(re.split(r"[,;:]", sent))
    ]


def foreign_objects(rec: dict, out: dict, vocab: frozenset[str]) -> list[str]:
    """A vocabulary object word in the text fields that the clip's metadata does not have (an invented object —
    rule 3). Natural names pass: the word inside a metadata word (phone ⊂ cellphone) or the reverse, plurals."""
    texts = [out.get("instruction"), *(out.get("paraphrases") or []), out.get("short"), out.get("detailed")]
    for x in out.get("subtasks") or []:
        if isinstance(x, dict):
            texts += [x.get("instruction"), *(x.get("paraphrases") or [])]
    given = meta_tokens(rec)
    seqs = clause_words(texts)

    def after_meta(ws, k):  # the previous word is a metadata object word
        prev = ws[k - 1].removesuffix("'s") if k else ""
        return bool(prev) and any(prev == g or prev in g for g in given if len(g) >= 3)

    def of_meta(ws, k):  # '<container> of (the) <object>' — its packaging
        j = k + 2
        while j < len(ws) and ws[j] in DET:
            j += 1
        return (
            k + 1 < len(ws)
            and ws[k + 1] == "of"
            and j < len(ws)
            and any(ws[j] == g or ws[j] in g for g in given if len(g) >= 3)
        )

    # a container noun introduced anywhere as '<object> <container>' or '<container> of <object>' is the same object
    # when used alone later ('the bottle')
    heads = vocab | given | set(NOT_OBJECT)  # object name words (heads for the participle test)
    introduced = {
        w for ws, _ in seqs for k, w in enumerate(ws) if w in CONTAINER and (after_meta(ws, k) or of_meta(ws, k))
    }
    bad = set()
    for ws, start in seqs:
        for k, w in enumerate(ws):
            stems = (w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w)  # plurals only
            base = next((b for b in stems if b in vocab), None) if len(w) >= 3 else None
            if (
                not base
                or any(base in g or g in base or one_off(base, g) for g in given if len(g) >= 3)
                or not_noun_slot(ws, k, heads, start)
            ):
                continue
            if base in CONTAINER and (after_meta(ws, k) or of_meta(ws, k) or base in introduced):  # containers
                continue
            bad.add(base)
    return [f"not in meta: object words {sorted(bad)}"] if bad else []


#: Size adjectives (closed class): a glued name minus one of them equal to the output name is the same object (GRAB
#: cubemedium <-> cube).
SIZE = ("small", "medium", "large", "big", "little", "tall", "short", "long", "mini", "tiny", "huge")


def object_ok(name: str, rec: dict) -> bool:
    """Does the output object name match a given object: one or more shared words (natural names pass:
    'plate_019' <-> 'plate')."""
    given = meta_tokens(rec)
    got = toks(name)
    ws = [w for w in words(name.replace("_", " ").replace("-", " ")) if w not in STOP]
    glued = {"".join(x) for x in itertools.permutations(ws)} if len(ws) <= 3 else {"".join(ws)}  # 'small cylinder'
    sized = any(
        g != t and (g.startswith(t) and g[len(t) :] in SIZE or g.endswith(t) and g[: -len(t)] in SIZE)
        for g in given
        for t in got
        if len(t) >= 3
    )  # 'cube' <-> 'cubemedium' (size removed)
    return (
        bool(given & got)
        or bool(got & {w + "s" for w in given})
        or bool({w.rstrip("s") for w in got} & given)
        or bool(glued & given)
        or bool({g.rstrip("s") for g in glued} & given)
        or sized
        or any(one_off(x, g) for x in glued | got for g in given)
    )
