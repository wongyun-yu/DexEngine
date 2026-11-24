"""Closed word classes, patterns and thresholds of the checker rules R16-R23 (rules_qa) and of their target
resolution (place prepositions). Rules, not lookup tables: each entry is a closed class."""

from __future__ import annotations

import re

#: Part-motion verbs (closed class: opening, closing, pulling, turning a part; includes operate/use for the ARCTIC
#: label 'use').
ART_VERBS = (
    "open",
    "opens",
    "opening",
    "close",
    "closes",
    "closing",
    "shut",
    "shuts",
    "pull",
    "pulls",
    "pulling",
    "push",
    "pushes",
    "pushing",
    "slide",
    "slides",
    "sliding",
    "swing",
    "swings",
    "lift",
    "lifts",
    "lifting",
    "raise",
    "raises",
    "raising",
    "lower",
    "lowers",
    "lowering",
    "fold",
    "folds",
    "unfold",
    "turn",
    "turns",
    "turning",
    "rotate",
    "rotates",
    "rotating",
    "twist",
    "twists",
    "flip",
    "flips",
    "tilt",
    "tilts",
    "press",
    "presses",
    "pressing",
    "operate",
    "operates",
    "operating",
    "use",
    "uses",
    "using",
    "work",
    "works",
    "draw",
    "draws",
    "extend",
    "extends",
    "retract",
    "retracts",
    "squeeze",
    "squeezes",
    "crank",
    "spin",
)
#: R16 hand — verbs that only move a part (lift, turn, tilt, use also move a whole body: 'lift the laptop with both
#: hands').
PART_ONLY = (
    "open",
    "opens",
    "opening",
    "close",
    "closes",
    "closing",
    "shut",
    "shuts",
    "pull",
    "pulls",
    "pulling",
    "push",
    "pushes",
    "pushing",
    "slide",
    "slides",
    "sliding",
    "swing",
    "swings",
    "fold",
    "folds",
    "unfold",
    "draw",
    "draws",
    "extend",
    "extends",
    "retract",
    "retracts",
    "squeeze",
    "squeezes",
    "crank",
)
#: R17 — size words, turn verbs (same clause) and the rotation threshold (degrees). 'small' and 'minor' are left out:
#: they modify objects ('the small cylinder').
SMALL = ("slightly", "slight", "a little", "a bit", "barely")
TURN = (
    "turn",
    "turns",
    "turning",
    "turned",
    "rotate",
    "rotates",
    "rotating",
    "rotation",
    "twist",
    "twists",
    "twisting",
)
ROT_BIG_DEG, NEAR = 60.0, 3
#: R18 — pour verb (closed class) followed by 'from the <object>'.
POUR_RE = re.compile(
    r"\b(?:pour|pours|pouring|tip|tips|tipping|empty|empties|emptying)\b[^,;.]*?\bfrom\s+(?:the|a|an)"
    r"\s+((?:[a-z]+\s+){0,2}[a-z]+)"
)
#: R19 — take-out / put-in forms (out, in/into/inside within three words of the verb; 'in place' and hand words are
#: place talk and excluded).
LOC = r"(?:spot|place|position|location|area|side|hand|hands)"
PLACEHOLDER = re.compile(
    r"\b(?:carry|carries|carrying|move|moves|moving|transfer|transfers|bring|brings|take|takes)\s+"
    r"(?:[a-z]+\s+){0,2}?(?:the|an?|some)\s+(?:items?|things?|stuff|something)\b"
)
#: R19 (c) — carry and transfer always move a thing; move only when followed by it, them or the <non-hand word>
#: ('move the hand away' is the hand).
CARRY = ("carry", "carries", "carrying", "carried", "transfer", "transfers", "transferring", "transferred")
MOVE = ("move", "moves", "moving", "moved")
NOT_THING = ("hand", "hands", "fingers", "finger", "arm", "arms", "palm", "thumb")
#: R20 — pick verbs (closed class) and the held-from-start threshold (cm of net drop; the lift ceiling is the verb
#: check's LIFT_CM).
PICK = ("pick", "picks", "picking", "lift", "lifts", "lifting", "raise", "raises", "grab", "grabs", "grabbing")
STOP_VERBS = (
    "move",
    "moves",
    "set",
    "sets",
    "place",
    "places",
    "put",
    "puts",
    "carry",
    "carries",
    "lower",
    "lowers",
    "slide",
    "slides",
    "push",
    "pushes",
    "pull",
    "pulls",
    "turn",
    "turns",
    "rotate",
    "rotates",
    "hold",
    "holds",
    "release",
    "releases",
    "bring",
    "brings",
    "tilt",
    "tilts",
    "then",
    "while",
)
HELD_DROP_CM, PICK_SPAN = 5.0, 8
TAKE_OUT = re.compile(r"\b(?:take|takes|taking|took|get|gets)\s+(?:[a-z]+\s+){0,3}?out\b")
PUT_IN = re.compile(
    r"\b(?:put|puts|putting|place|places|placing|insert|inserts)\s+(?:[a-z]+\s+){0,3}?"
    r"(?:in|into|inside)"
    rf"\b(?!\s+(?:[a-z]+\s+){{0,3}}?{LOC}\b)"
)


LIFT_WORDS = (
    "carry",
    "carries",
    "carrying",
    "carried",
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
)
NEG = ("without", "not", "never", "no", "nor")
STEADY_RE = re.compile(
    r"\b(?:steady|steadies|steadying|steadied|stabiliz\w*|(?:hold|holds|holding|keep|keeps|keeping|"
    r"stay|stays|staying|remain|remains|remaining)\s+(?:\w+\s+){0,3}?(?:still|steady|in place|"
    r"stationary|motionless))\b"
)
STILL_MOVE_CM = 5.0
BODY_WORDS = ("base", "body", "bottom", "frame", "housing", "stand")
LIFT_SPAN = 6
CLAUSE_WORDS = ("while", "as", "then", "before", "after", "and", "whereas", "until")


#: Names after these words are not the moved object but its origin, destination or neighbour ('out of the mug',
#: 'from the drawer', 'into the rack', 'carry it to the tripod', 'near the plate', 'tilt it over the mug').
PLACE_PREP = (
    "out",
    "from",
    "into",
    "off",
    "onto",
    "inside",
    "in",
    "on",
    "to",
    "toward",
    "towards",
    "near",
    "beside",
    "over",
    "across",
    "above",
    "under",
    "against",
    "behind",
    "around",
)
#: Masked in hand claims (R28): only origin, destination and neighbour of a move ('rests on / against the pan' is a
#: contact claim).
MOVE_PREP = ("out", "from", "into", "onto", "to", "toward", "towards", "off", "inside", "near", "beside")
