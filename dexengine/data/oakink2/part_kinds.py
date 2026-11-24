#!/usr/bin/env python3
"""OakInk2 per-object joint spec (semantics): the joint kind and partner (parent) of every part in the official part
tree (oakink2.parts).

Why: deciding a part pair's joint kind from clip motion gives the same object different kinds per clip (bottle cap =
hinge · apple half = hinge · drawer = both hinge and slide · lamp = hinge). The kind follows what the thing is (a
microwave door is a hinge, the joint is where the parts meet, stated explicitly). Motion only **verifies** the kind
(oakink2.artic_fit.kind_check, which fails with a reason on a mismatch).

Kinds (part → parent part):
  body       the instance's main body (no parent, one per instance)
  revolute   hinge: one-axis rotation against the parent (door · screen · scissor blade · tong arm · crank handle)
  prismatic  slide: one-axis translation against the parent (drawer)
  fixed      rigid joint: one body with the parent (one object whose connecting geometry is missing, like a lamp head ·
             sub-parts)
  lever      a hinged lever on the body = a second joint (microwave door lever; measured: 46 of 54 trees where it moved
             hinge against the body · rotation residual ≤0.1° · 15–65°). The converter emits only one joint, so the
             lever is **attached fixed to the body in its rest pose**, but only in clips where the lever surface moves
             within LEVER_TIP_M of the rest pose (a single press) and the lever does not push the jointed part
             (LEVER_DRIVE_SHARE). Other clips (pulling it far, lever pushing the door open) are excluded.
  deformable a part joined to its parent by a bending link (lamp head; the neck bends). Clips where the link stays still
             attach it fixed (same band as fixed); clips where it bends are excluded.
  free       an independent object that comes off (bottle · toothpaste cap · pen cap · box lid · cut fruit · bread ·
             paper). Not merged; an ordinary object for the extra rule (extra_slots).
Rule: a part of the official tree missing from the table is an error (an unseen part is never treated silently as an
independent object).
"""

from __future__ import annotations

import argparse

from dexengine.data.oakink2 import parts as OP

KINDS = ("body", "revolute", "prismatic", "fixed", "lever", "deformable", "free")
JOINT_KINDS = ("revolute", "prismatic")
#: Attached kind → exclusion reason word for trees where that part moved outside its band. **The kind owns the word**
#: (one line = one decision; turning the part's kind back to fixed changes the reason to attach_moves).
ATTACH_MOVED = {
    "fixed": "attach_moves",  # stated fixed but moves on its own: spec and source disagree
    "lever": "two_joint_excluded",  # two joints are excluded
    "deformable": "deformable_excluded",  # deformables are excluded
}
ATTACH_KINDS = tuple(ATTACH_MOVED)
#: **Closed vocabulary** of the reason words for excluding a tree because of an articulated object → plain-language
#: label.
#: Every word the merge rule (oakink2.artic_fit), the converter, rigid joints and the articulated-support rule of
#: candidate
#: selection (dexengine.data.candidates.SUPPORT_ARTIC) emit is here; unknown words fail (exclude_label). The exclusion
#: ledger, candidate selection, the prep page and dashboards read this one table. Only the words are written to the
#: ledger; the labels are display text.
EXCLUDE_WORDS = {
    "two_joint_excluded": "object with two joints (a microwave lever pulled far, or a lever that opens the door)",
    "deformable_excluded": "lamp with a bending neck",
    "attach_moves": "a part that should be rigid with its body moves on its own",
    "kind_mismatch": "the part moves unlike the joint kind of its object",
    "multi_moving": "scene with two or more articulated objects: more than one moves",
    "multi_fixed_drifts": (
        "scene with two or more articulated objects: an object to be fixed rigid drifts from its recorded pose"
    ),
    "canon_misfit": "the canonical joint cannot express this clip's motion",
    "rest_off_joint": "the part's rest pose lies outside the canonical joint's range",
    "no_joint_spec": "part pair without a canonical joint yet",
    "fk_oracle": "the part pose rebuilt from the joint differs from the source",
    "support_articulated": (
        "support object is articulated (a drawer etc.) — support baking and the judge's support "
        "measure handle rigid bodies only"
    ),
}
#: Lever band = **tip displacement**: upper bound on how far the lever surface (mesh vertices) travels from the rest
#: pose
#: (attachment representative). Not measured as rotation/translation of the mesh origin (an origin far from the hinge
#: inflates 15° to a 65 mm origin move). Basis: measured distribution over 222 microwave trees: lever at rest ≤ 19 mm
#: (median 2) · single press 14–23 mm · pulled far ≥ 29 mm; the value sits in the 23–29 mm gap. Read only by
#: oakink2.attach.lever_in_band; pairing (part_clusters) and the converter FK oracle
#: (oakink2.artic_convert.convert_tree) use it.
LEVER_TIP_M = 0.025
#: Separates clips where the lever pushes the jointed part: when the share of the jointed part's motion during which the
#: lever is turned away from rest is at least this, the lever opened it (hand opened the door ≤ 0.2 · lever pushed the
#: door ≥ 0.9; the middle of the gap). With the lever fixed, the jointed part in such a clip becomes a reference moving
#: with nobody touching it.
LEVER_DRIVE_SHARE = 0.5
#: Slide rail play band: how far a prismatic jointed part tilts and sags on its rail. Applies to slides only (hinges and
#: rigid joints keep the D1-g thresholds). Read only by oakink2.artic_fit.band; kind check, canonical comparison
#: (converter · spec stage · trainability · training entry) and the FK oracle use it. It covers the rotation and
#: off-rail motion against the canonical slide in clips that push only the drawer with the frame still, and leaves out
#: clips that lift the whole chest while the two parts' tracking drifts apart. Paired with drawer–frame self-collision
#: off (with collision on, the tilt inside the band would jam). Basis: the D1-g articulation rule.
SLIDE_PLAY_ROT_DEG = 8.0
SLIDE_PLAY_POS_M = 0.015


def _o(n: int, i: int) -> str:
    return f"O02@{n:04d}@{i:05d}"


def _fixed_all(body: str, *rest: str) -> dict:
    return {body: ("body", None), **dict.fromkeys(rest, ("fixed", body))}


def _spec() -> dict[str, tuple[str, str | None]]:
    """Part rid → (kind, parent rid). Instance roots are not listed (a root that is itself the object is not a part)."""
    s: dict[str, tuple[str, str | None]] = {}
    s.update(_objects_3_to_29())
    s.update(_objects_32_to_60())
    s.update(_objects_90_to_206())
    s.update(_one_piece_objects())
    return s


def _objects_3_to_29() -> dict[str, tuple[str, str | None]]:
    """Toothpaste, teapot, bottles, box, drawer, pens, markers and pencil."""
    s: dict[str, tuple[str, str | None]] = {}
    s[_o(3, 5)] = ("body", None)  # toothpaste: tube · cap
    s[_o(3, 4)] = ("free", _o(3, 5))
    s.update(
        {
            _o(13, 5): ("body", None),
            _o(13, 4): ("free", _o(13, 5)),  # teapot: lid comes off
            **{_o(13, i): ("fixed", _o(13, 5)) for i in (7, 8, 9)},
        }
    )
    for root in range(3, 28, 3):  # 9 bottles: cap · body
        s[_o(15, root - 1)] = ("body", None)
        s[_o(15, root - 2)] = ("free", _o(15, root - 1))
    s[_o(18, 2)] = ("body", None)  # box: removable lid (remove_lid · put_on_lid clips, position residual 17–92 cm)
    s[_o(18, 1)] = ("free", _o(18, 2))
    s.update(
        {
            _o(19, 1): ("body", None),
            _o(19, 2): ("prismatic", _o(19, 1)),  # drawer: frame · drawer
            _o(19, 16): ("fixed", _o(19, 2)),
            _o(19, 17): ("fixed", _o(19, 2)),
        }
    )
    for body, cap, subs in ((1, 2, (14, 15)), (8, 9, (19, 20)), (11, 12, (21, 22))):  # pen · marker: cap
        s.update(
            {
                _o(29, body): ("body", None),
                _o(29, cap): ("free", _o(29, body)),
                **{_o(29, i): ("fixed", _o(29, body)) for i in subs},
            }
        )
    s.update(_fixed_all(_o(29, 16), _o(29, 17), _o(29, 18)))  # pencil
    return s


def _objects_32_to_60() -> dict[str, tuple[str, str | None]]:
    """Tongs, scissors, stapler, microwave (door and lever), pencil sharpener, lamp, laptop, paper and book."""
    s: dict[str, tuple[str, str | None]] = {}
    for n in (32, 35):  # tongs · scissors: two-arm hinge
        s.update(
            {
                _o(n, 1): ("body", None),
                _o(n, 2): ("revolute", _o(n, 1)),
                _o(n, 7): ("fixed", _o(n, 1)),
                _o(n, 8): ("fixed", _o(n, 1)),
                _o(n, 9): ("fixed", _o(n, 2)),
                _o(n, 10): ("fixed", _o(n, 2)),
            }
        )
    s[_o(37, 2)] = ("body", None)  # stapler: press-arm hinge
    s[_o(37, 1)] = ("revolute", _o(37, 2))
    s.update(
        {
            _o(39, 1): ("body", None),
            _o(39, 2): ("revolute", _o(39, 1)),  # microwave: door · lever
            **{_o(39, i): ("fixed", _o(39, 1)) for i in (5, 6, 7)},
        }
    )
    s[_o(39, 4)] = ("lever", _o(39, 1))  # the lever is declared on this one line; if it moves → ATTACH_MOVED["lever"]
    s[_o(44, 1)] = ("body", None)  # pencil sharpener: crank handle
    s[_o(44, 2)] = ("revolute", _o(44, 1))
    s[_o(45, 1)] = ("body", None)  # lamp: base · head (bending neck link)
    # the head is declared on this one line; if it bends → ATTACH_MOVED["deformable"]
    s[_o(45, 2)] = ("deformable", _o(45, 1))
    s[_o(53, 2)] = ("body", None)  # laptop: screen hinge
    s[_o(53, 1)] = ("revolute", _o(53, 2))
    s[_o(56, 1)] = ("body", None)  # two sheets of paper
    s[_o(56, 2)] = ("free", _o(56, 1))
    s[_o(60, 1)] = ("body", None)  # book: two-half spine hinge
    s[_o(60, 2)] = ("revolute", _o(60, 1))
    return s


def _objects_90_to_206() -> dict[str, tuple[str, str | None]]:
    """Cut fruit, bread, test-tube tongs and alcohol lamp."""
    s: dict[str, tuple[str, str | None]] = {}
    for n in (90, 92):  # apple · pear: cut halves
        s[_o(n, 1)] = ("body", None)
        s[_o(n, 2)] = ("free", _o(n, 1))
    s.update({_o(94, 1): ("body", None), **{_o(94, i): ("free", _o(94, 1)) for i in (2, 3, 4)}})  # bread
    s.update(
        {
            _o(203, 1): ("body", None),
            _o(203, 2): ("revolute", _o(203, 1)),  # test-tube tongs: two-arm hinge
            _o(203, 4): ("fixed", _o(203, 1)),
            _o(203, 5): ("fixed", _o(203, 1)),
            _o(203, 6): ("fixed", _o(203, 2)),
            _o(203, 7): ("fixed", _o(203, 2)),
        }
    )
    s.update(
        {
            _o(206, 2): ("body", None),
            _o(206, 1): ("free", _o(206, 2)),  # alcohol lamp: lid comes off
            _o(206, 4): ("fixed", _o(206, 2)),
            _o(206, 5): ("fixed", _o(206, 2)),
        }
    )
    return s


def _one_piece_objects() -> dict[str, tuple[str, str | None]]:
    """One-piece objects: sub-parts never loaded separately; if loaded, they are bound into one body."""
    s: dict[str, tuple[str, str | None]] = {}
    for body, *rest in (
        (_o(10, 4), _o(10, 5)),
        (_o(20, 3), _o(20, 4)),
        (_o(20, 5), _o(20, 6)),
        (_o(25, 2), _o(25, 3)),
        (_o(26, 3), _o(26, 2)),
        (_o(28, 3), _o(28, 2)),
        (_o(30, 4), _o(30, 3)),
        (_o(31, 4), _o(31, 3)),
        (_o(38, 2), _o(38, 3), _o(38, 4)),
        (_o(54, 2), _o(54, 3)),
        (_o(55, 4), _o(55, 3)),
        (_o(55, 6), _o(55, 5)),
        (_o(200, 2), _o(200, 3)),
        (_o(204, 2), _o(204, 3), _o(204, 4)),
        ("C02@0010@00001", "C02@0010@00002"),
        ("C02@0014@00001", "C02@0014@00002", "C02@0014@00003"),
        ("C02@0043@00001", "C02@0043@00002"),
        ("C02@0051@00002", "C02@0051@00001"),
        ("S02@0010@00001", "S02@0010@00002"),
        ("S02@0010@00003", "S02@0010@00004"),
        ("S02@0010@00005", "S02@0010@00006"),
        ("S02@0020@00002", "S02@0020@00001"),
    ):
        s.update(_fixed_all(body, *rest))
    return s


SPEC = _spec()


def kind_of(rid: str) -> tuple[str, str | None]:
    """Official part rid → (kind, parent rid). ValueError when it is not a part, or a part missing from the table."""
    if not OP.is_part(rid):
        raise ValueError(f"[PART_KINDS] {rid} is not an official part")
    if rid not in SPEC:
        raise ValueError(
            f"[PART_KINDS] part {rid}({OP.name_of(rid)}) has no joint kind in the spec — "
            f"add it to dexengine/data/oakink2/part_kinds.py"
        )
    return SPEC[rid]


def exclude_label(word: str) -> str:
    """Exclusion reason word → plain-language label. Words outside the vocabulary fail (no reason of unknown meaning
    reaches ledgers or pages)."""
    if word not in EXCLUDE_WORDS:
        raise SystemExit(
            f"[PART_KINDS] ✗ unknown exclusion reason word {word!r} — vocabulary = {sorted(EXCLUDE_WORDS)} · add new "
            "words to dexengine/data/oakink2/part_kinds.py EXCLUDE_WORDS with a label"
        )
    return EXCLUDE_WORDS[word]


#: Whether a D1g exclusion of a parent passes on to the pieces cut from it. True = a verdict on the object itself (the
#: object's kind, a mismatch with the spec; it holds for every piece using that object) → inherited · False = a verdict
#: that depends on the interval (fit, pose, oracle, scene motion) → re-measured on the piece window. **Only decided
#: words**
#: are listed; a new word fails until it is decided (piece_inherits · check; never set silently either way).
PIECE_INHERIT = {
    "deformable_excluded": True,
    "kind_mismatch": True,
    "two_joint_excluded": True,
    "attach_moves": True,
    "no_joint_spec": True,
    "canon_misfit": False,
    "multi_moving": False,
    "multi_fixed_drifts": False,
    "rest_off_joint": False,
    "fk_oracle": False,
    # whether a main object rests on an articulated support differs per piece window; the predicate
    # (dexengine.data.candidates) runs per piece
    "support_articulated": False,
}


def piece_inherits(word: str) -> bool:
    """Exclusion reason word → inherited by pieces (PIECE_INHERIT). Words outside the vocabulary or not yet decided fail
    (ValueError; the converter counts the tree as an error row)."""
    exclude_label(word)
    if word not in PIECE_INHERIT:
        raise ValueError(
            f"[PART_KINDS] ✗ whether exclusion reason {word!r} passes on to pieces is not decided (decided = "
            f"{sorted(PIECE_INHERIT)}) — decide it, then add it to "
            "oakink2_part_kinds.PIECE_INHERIT"
        )
    return PIECE_INHERIT[word]


def check_inherit() -> list[str]:
    """Check of the inheritance table (part of check): its words are in the vocabulary and every vocabulary word is
    classified."""
    return [f"inherit word outside the vocabulary {w}" for w in sorted(PIECE_INHERIT) if w not in EXCLUDE_WORDS] + [
        f"reason word without an inherit decision {w} (add it to PIECE_INHERIT once decided)"
        for w in sorted(EXCLUDE_WORDS)
        if w not in PIECE_INHERIT
    ]


def body_of(root: str) -> str:
    """Instance root → main-body part rid (exactly one in the spec)."""
    got = [r for r, (k, _p) in SPEC.items() if k == "body" and OP.root_of(r) == root]
    if len(got) != 1:
        raise ValueError(f"[PART_KINDS] root {root}({OP.name_of(root)}) has {len(got)} main bodies — needs exactly one")
    return got[0]


def merges(rid: str) -> bool:
    """Whether a part is merged (a part that is not free); otherwise it is an ordinary object."""
    return OP.is_part(rid) and kind_of(rid)[0] != "free"


def check() -> list[str]:
    """Check of the spec itself: every official part is in the table · parents are parts of the same root · one main
    body
    per root · at most one jointed part per instance (converter contract: one joint per articulated object). Returns the
    problems (empty = pass)."""
    tree, rev, _desc, _inst = OP._load()
    bad = [f"not in the table {c}({OP.name_of(c)})" for c in sorted(rev) if c not in SPEC]
    bad += [f"not a part {r}" for r in sorted(SPEC) if r not in rev]
    bad += [
        f"reason word outside the vocabulary {k} → {w}"
        for k, w in sorted(ATTACH_MOVED.items())
        if w not in EXCLUDE_WORDS
    ]
    bad += check_inherit()
    for r, (k, p) in sorted(SPEC.items()):
        if k not in KINDS:
            bad.append(f"unknown kind {r}: {k}")
        if (k == "body") != (p is None):
            bad.append(f"body/parent mismatch {r}: {k} · {p}")
        if p is not None and (p not in SPEC or OP.root_of(p) != OP.root_of(r)):
            bad.append(f"parent is not a part of the same root {r} → {p}")
    for root in sorted({OP.root_of(c) for c in rev}):
        try:
            body_of(root)
        except ValueError as e:
            bad.append(str(e))
        nj = [r for r, (k, _p) in SPEC.items() if k in JOINT_KINDS and OP.root_of(r) == root]
        if len(nj) > 1:
            bad.append(f"more than one jointed part {root}: {nj}")
    return bad


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    probs = check()
    print(
        f"[PART_KINDS] parts {len(SPEC)} · kinds "
        + " · ".join(f"{k} {sum(v[0] == k for v in SPEC.values())}" for k in KINDS)
        + f" · problems {len(probs)}"
    )
    for p in probs:
        print("  ✗", p)
    raise SystemExit(1 if probs else 0)
