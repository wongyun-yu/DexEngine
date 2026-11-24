"""Closed word classes of the checker scene rules (rules_scene): manipulation, relation, part, side, contact,
position, handover and body-direction words, the R6-R11 source-wording words, and the prompt's word-list slots
(PROMPT_LISTS)."""

from __future__ import annotations

import re

#: Non-manipulation verbs (closed class: the hand only reaches, grasps, holds, lets go, withdraws).
NON_MANIP = (
    "reach",
    "approach",
    "grasp",
    "grab",
    "grip",
    "hold",
    "touch",
    "release",
    "let go",
    "retreat",
    "withdraw",
    "rest",
    "keep",
    "steady",
    "support",
    "pinch",
    "clasp",
    "press",
    "tap",
)
#: Relation words (position relative to another object), matched as word sequences (lower case, single spaces).
RELATION = (
    "toward",
    "towards",
    "on top of",
    "beside",
    "next to",
    "alongside",
    "near",
    "close to",
    "by",
    "in front of",
    "around",
    "against",
)
#: The target of a relation word = within this many words after it (determiners, modifiers + an object name;
#: adverbs like 'turn it around and ...' are not caught).
REL_SPAN = 4
#: Part nouns (closed class: moving or attached parts of an object).
PART = (
    "lid",
    "door",
    "cap",
    "cover",
    "handle",
    "knob",
    "drawer",
    "lever",
    "hinge",
    "flap",
    "tray",
    "button",
    "blade",
    "trigger",
    "spout",
    "nozzle",
    "latch",
    "panel",
    "top",
    "crank",
    "screen",
)
#: R5 direction word classes (closed classes).
SIDE, CENTER = ("left", "right"), ("middle", "center", "centre", "front", "back", "edge")
SIDE_PREP = ("to", "toward", "towards", "onto", "on", "from", "at")
MOVE_PREP = ("to", "toward", "towards", "onto", "into")
BODY = ("hand", "hands", "arm", "arms", "palm", "thumb", "finger", "fingers", "wrist", "one", "ones")
#: Verbs of a hand -> object contact claim (held_by, R13, R28) — closed class (grip, support, move by hand). It is a
#: claim only when such a verb sits between the hand and the object (or before the object) with no position word
#: (POSITION) after it: 'the left hand stays by the pan', 'reaches toward the pan' are positions; 'rests on the pan',
#: 'holds the pan steady' are contact. Tool verbs (brush, stir, cut) are left out: the tool touches, not the hand.
CONTACT = (
    "grip",
    "grasp",
    "grab",
    "hold",
    "press",
    "steady",
    "support",
    "pinch",
    "clasp",
    "clutch",
    "touch",
    "rest",
    "stay",
    "remain",
    "keep",
    "brace",
    "lift",
    "pick",
    "raise",
    "lower",
    "carry",
    "take",
    "push",
    "pull",
    "slide",
    "drag",
    "turn",
    "rotate",
    "twist",
    "tilt",
    "flip",
    "open",
    "close",
    "squeeze",
    "shake",
    "move",
    "set",
    "put",
    "place",
    "use",
    "catch",
    "receive",
    "release",
    "let",
)
IRREGULAR = ("held", "took", "taken", "caught", "kept")
POSITION = (
    "by",
    "near",
    "beside",
    "next",
    "close",
    "toward",
    "towards",
    "over",
    "above",
    "behind",
    "around",
    "away",
    "from",
    "at",
    "to",
    "alongside",
    "below",
    "under",
    "beneath",
)

# ── handover (one hand -> the other) ──
#: Handover phrases, matched as word sequences in one clause (split at commas and periods). The prompt's
#: {{HANDOVER}} slot reads this tuple (PROMPT_LISTS).
HANDOVER_WORDS = (
    "hand it over",
    "hands it over",
    "handed it over",
    "handing it over",
    "hand them over",
    "hands them over",
    "hand it to",
    "hands it to",
    "handed it to",
    "handing it to",
    "hand it off",
    "hands it off",
    "pass it",
    "passes it",
    "passed it",
    "passing it",
    "pass them",
    "passes them",
    "give it to",
    "gives it to",
    "gave it to",
    "giving it to",
    "to the other hand",
    "into the other hand",
    "to the left hand",
    "to the right hand",
    "into the left hand",
    "into the right hand",
    "from one hand to the other",
    "from the other hand",
    "from the right hand to the left",
    "from the left hand to the right",
    "handover",
    "hand to hand",
    "switch hands",
    "switches hands",
    "swap",
    "swaps",
    "swapped",
    "swapping",
    "hand over",
    "hands over",
    "handed over",
    "hand off",
)
#: Verb-only phrases count only when the previous word is not a determiner or number (HANDOVER_DETS): 'with both
#: hands over the bowl' is a hand position, not a handover.
HANDOVER_VERB_ONLY = ("hand over", "hands over", "handed over", "hand off")
HANDOVER_DETS = ("the", "a", "both", "two", "his", "her", "their", "its", "my", "your", "with", "of")
#: The prompt's {{NAME}} slots for word lists (fill_prompt) — the prompt holds no copy of these lists.
PROMPT_LISTS = {"RELATION": RELATION, "NON_MANIP": NON_MANIP, "HANDOVER": HANDOVER_WORDS, "CONTACT": CONTACT}


#: R5 body / viewer directions ('in front of you', 'nudge it forward').
BODY_DIR = (
    "forward",
    "forwards",
    "backward",
    "backwards",
    "toward you",
    "towards you",
    "toward yourself",
    "towards yourself",
    "away from you",
    "in front of you",
    "behind you",
    "closer to you",
)


# ── R6 · R7 · R8 · R10 · R11: words the measurement, the source text or the object list do not back ──
TACO_PUT = ("put", "puts", "place", "places", "insert", "inserts")
TACO_OUT = ("out", "back", "return", "returns", "remove", "removes", "lift", "lifts", "take", "takes", "withdraw")
NET_BACK_CM = 5.0  # R6 'put in': the tool's net travel below this = put in and taken back out
#: R6 literal 'cut' / 'screw' = the label verb taking an object directly ('cut the plate', 'cut into the pan',
#: 'screw the screwdriver into the box'). The verb as a manner ('in a cutting motion', 'as if driving a screw')
#: passes.
CUT_OBJ = re.compile(r"\bcut(?:s|ting)?\s+(?:into|through|up|the|a|an|it|them|its|across)\b")
SCREW_VERB = re.compile(r"(?<!\ba )\bscrew(?:s|ing|ed)?\b(?!driver)")
ARCTIC_PARTS = ("knob", "knobs", "lever", "levers", "button", "buttons", "dial", "switch", "handle", "trigger")
CONTENTS = ("contents", "content", "residue", "residues", "ingredients", "leftovers", "crumbs", "food", "liquid")
