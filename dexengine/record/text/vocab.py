"""Words of the text stage: dataset object id -> human name, render-scene object names, and the closed word tables
the checker and the prompt read (TACO label verbs, non-object words, container nouns, meta-talk patterns).

Object names: OakInk2 = the official part tree, HO-Cap = the official object list, TACO = the dataset's own
verb_tool_target labels. An id without a name dies (that clip gets no text).
"""

from __future__ import annotations

import functools
import glob
import json
import re
from pathlib import Path

from dexengine.core import stdio
from dexengine.record.text.api import die

#: HO-Cap official object list (IRVLUTD/HO-Cap config/hocap_info.yaml `object_descriptors`; the sibling
#: hocap_info.SOURCE.json records source, version and sha256), relative to the data root.
HOCAP_INFO = "hocap/raw/hocap_info.yaml"


@functools.cache
def hocap_names(data_root: str) -> dict[str, str]:
    """HO-Cap G code -> name ('G05_2' -> 'body lotion'), read from the official list only."""
    f = Path(data_root) / HOCAP_INFO
    if not f.is_file():
        die(f"HO-Cap object list missing: {f} (put the official hocap_info.yaml there)")
    names, on = {}, False
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("object_descriptors:"):
            on = True
            continue
        m = re.match(r"^\s+(G\d+_\d+):\s*(\S+)\s*$", line) if on else None
        if on and not m and not line.startswith(" "):
            break
        if m:
            names[m[1]] = m[2].replace("_", " ")
    if len(names) < 64:
        die(f"HO-Cap object list too short: {len(names)} < 64 in {f}")
    return names


@functools.cache
def taco_names(data_root: str) -> dict[str, str]:
    """TACO object id -> name. TACO has no official id list, so names are solved from the dataset's own labels
    (base clip folders verb_tool_target__seq with their two object ids, all hands): each id starts with the
    intersection of {tool, target} over its clips; an id solved in one clip fixes the other id of that clip to the
    remaining name (fixed point). Dies when an id stays unsolved (0 or 2+ names) or disagrees with a clip label."""
    rows = []
    for f in sorted(glob.glob(str(Path(data_root) / "taco" / "base" / "*" / "bimanual" / "*" / "task_info.json"))):
        head = Path(f).parent.name.split("__")[0].split("_")
        if len(head) != 3:
            die(f"TACO folder name is not verb_tool_target: {f}")
        names = {head[1].replace("-", " "), head[2].replace("-", " ")}
        rows.append((names, [str(o["obj_id"]) for o in json.loads(Path(f).read_text())["objects"]]))
    if not rows:
        die(f"no TACO base records under {data_root}/taco/base: cannot solve names")
    cand: dict[str, set] = {}
    for names, ids in rows:
        for i in ids:
            cand[i] = cand.get(i, set(names)) & names
    changed = True
    while changed:
        changed = False
        for names, ids in rows:
            for x in ids:
                for y in ids:
                    other = names - cand[x]
                    if x != y and len(cand[x]) == 1 and len(names) == 2 and other <= cand[y] and cand[y] != other:
                        cand[y], changed = set(other), True
    bad = {k: sorted(v) for k, v in cand.items() if len(v) != 1}
    if bad:
        die(f"TACO id names unsolved: {len(bad)} {dict(list(bad.items())[:5])}")
    out = {k: next(iter(v)) for k, v in cand.items()}
    off = [ids for names, ids in rows if {out[i] for i in ids} - names]
    if off:
        die(f"TACO id names disagree with the clip labels in {len(off)} clips")
    return out


#: An object named like a body part is confused with the actor; ' model' is appended (GRAB's official 'hand' is a
#: hand-shaped object).
BODY_WORDS = ("hand", "hands", "arm", "arms", "finger", "fingers", "body", "head", "foot", "leg")


def readable(ds: str, obj_id: str, clip: str, data_root: str) -> tuple[str, list[str]]:
    """Human-readable object name and flags. oakink2 = official part description, hocap = official object list,
    taco = name solved from the labels (taco_names); those three die when the id has no name."""
    if ds == "oakink2":  # official part tree (object_part_tree, part_desc)
        from dexengine.data.oakink2 import parts as OP

        try:
            rid = OP.rid_of(obj_id)
            name = OP.name_of(rid)
            base = re.sub(r"@F[0-9A-F]+$", "", rid)  # id with a part tail (@F<hex>) -> a part of that object
            if name == rid and base != rid and OP.name_of(base) != base:
                name = OP.name_of(base) + " part"
        except Exception as e:  # noqa: BLE001 — any failure means the same thing: no name
            die(f"oakink2 object id {obj_id!r}: cannot read its name ({clip}) — {type(e).__name__}: {e}")
        if name == rid:
            die(f"oakink2 object id {obj_id!r} has no description in the official part tree ({clip})")
        return name, []
    if ds in ("hocap", "taco"):
        names = (hocap_names if ds == "hocap" else taco_names)(str(data_root))
        if obj_id not in names:
            die(f"{ds} object id {obj_id!r} has no name in the list ({clip}); this clip gets no text")
        return names[obj_id], []
    if ds == "hot3d" and obj_id.split("_")[0] == "aria":  # HOT3D Aria glasses (official name aria_*, no description)
        return "pair of glasses", []
    name = " ".join(re.sub(r"(?<=[a-z]{3})\d+$", "", w) for w in obj_id.replace("_", " ").split())  # cactus3 -> cactus
    if any(w in BODY_WORDS for w in name.lower().split()):
        name += " model"
    return name, []


# ── render-scene objects (what the video shows) ──
def placed_ids(e: dict) -> list[str]:
    """Object ids placed in the render scene of one final episodes.jsonl row (extras_visual.placed, written by the
    render stage). Dropped objects are not in the render."""
    return [str(p["obj_id"]) for p in ((e.get("extras_visual") or {}).get("placed") or []) if p.get("obj_id")]


@functools.cache
def absent_index(roots: tuple[str, ...]) -> dict[str, frozenset[str]]:
    """{clip key: object ids missing from the render} = union over every episode of the final datasets of
    extras_visual.dropped whose reason is not scene.extras.DRAWN_ELSEWHERE. One clip's text serves all its episodes
    (original, placement variants, hands), so one missing render is enough. Keeps only keys and ids per row."""
    from dexengine.record.scene import extras as X
    from dexengine.record.text import order as ORD

    out: dict[str, set] = {}
    for root in roots:
        with open(Path(root) / "meta" / "episodes.jsonl", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    e = json.loads(ln)
                    out.setdefault(ORD.ep_clip(e), set()).update(
                        str(d["obj_id"])
                        for d in (e.get("extras_visual") or {}).get("dropped") or []
                        if d.get("obj_id") and d.get("reason") not in X.DRAWN_ELSEWHERE
                    )
    return {k: frozenset(v) for k, v in out.items()}


def drop_absent(rec: dict, finals) -> dict:
    """Remove objects that are not in the render from the text input (the source tree lists more objects than the
    render shows: off the desk, overlapping, hidden). Removed names go to objects_absent (rule R21 rejects a text
    that names them). Moved and main objects are kept (a render defect) and counted in absent_kept."""
    gone = absent_index(tuple(sorted(str(f) for f in finals))).get(rec["clip"], frozenset())
    obj = rec.get("objects") or []
    cut = [o for o in obj if str(o.get("id")) in gone and not o.get("moved") and o.get("role") != "main"]
    keep = [o for o in obj if o not in cut]
    names = {o["name"] for o in cut}
    mo = dict(rec.get("motion") or {})
    if "objects" in mo:
        mo["objects"] = [m for m in mo["objects"] if m.get("obj") not in names]
    return dict(
        rec,
        objects=keep,
        motion=mo,
        objects_absent=sorted(names | set(rec.get("objects_absent") or [])),
        absent_kept=sorted(str(o.get("id")) for o in keep if str(o.get("id")) in gone),
    )


def scene_names(ds: str, obj_ids: list[str], clip: str, data_root, known=()) -> tuple[list[str], int]:
    """Readable names of render-scene objects (allowed by R7, still objects for R18) and the number of ids that had
    no name (those are not allowed — conservative)."""

    out, bad = [], 0
    for oid in obj_ids:
        try:
            with stdio.capture():
                n = readable(ds, oid, clip, str(data_root))[0]
        except SystemExit:
            bad += 1
            continue
        if n not in known and n not in out:
            out.append(n)
    return out, bad


#: Episode meta fields that determine the render: render job (piece_id, round), clip, look, chosen variant, scene
#: objects, camera, hand, length. Text fields (tasks, text, source_label, subtasks — retext rewrites them without
#: rendering) and indices are excluded.
RENDER_KEYS = (
    "clip",
    "piece_id",
    "round",
    "look",
    "render_look",
    "render_preset",
    "selection",
    "extras_visual",
    "camera_pose",
    "hand_usd",
    "length",
    "video_frames",
)


def ep_sig(e: dict) -> str:
    """Fingerprint of an episode's render (RENDER_KEYS only; a missing field = None). It changes when the final
    re-renders the clip or picks another episode, and survives a retext (frame file names and the record's
    frames_sig carry it)."""
    import hashlib

    body = json.dumps({k: e.get(k) for k in RENDER_KEYS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(body.encode()).hexdigest()[:12]


# ── closed word tables (checker + prompt) ──
#: TACO verb (first field of the triplet label) -> (gloss for the prompt, word stems that count as keeping the label
#: verb). The prompt ({{TACO_VERBS}} in datasets.json taco) and the label-verb check (label_words) read this one
#: table, so a text written as the prompt says is never rejected for not copying the label literally.
_RUB = "brush or rub the target with the tool (a roller is rolled over it)"
_TILT = "a bowl or plate tool is tilted over the target as if putting or tipping something "
TACO_VERBS = {
    "cut": (
        "move the tool's edge across the target in a cutting or slicing motion "
        "(with a spoon or spatula never 'cut the plate')",
        ("cut", "slic", "saw"),
    ),
    "scrape-off": ("scrape across the target with the tool", ("scrap",)),
    "skim-off": ("skim across the target with the tool", ("skim",)),
    "brush": (_RUB, ("brush", "rub", "roll")),
    "dust": (_RUB, ("dust", "brush", "rub", "roll")),
    "stir": ("stir with the tool inside the target", ("stir",)),
    "stir-fry": ("stir with the tool inside the target", ("stir",)),
    "smear": ("smear the tool across the target", ("smear",)),
    "pour-in-some": ("tilt the tool (a bowl or teapot) over the target as if pouring", ("pour",)),
    "empty": ("tilt the tool (a bowl or teapot) over the target as if emptying it", ("empt", "pour")),
    "put-in": (
        "move the tool into the target; " + _TILT + "in",
        ("put", "into", "inside", "scoop", "pour", "empt", "tip"),
    ),
    "put-out": (
        "scoop out of the target with the tool and lift it out; " + _TILT + "out",
        ("put", "scoop", "out of", "pour", "empt", "tip"),
    ),
    "hit": ("tap the target with the tool", ("tap", "hit", "strik")),
    "screw": ("turn the screwdriver against the target as if driving a screw", ("screw",)),
    "measure": ("hold the tool against the target as if measuring it", ("measur",)),
}
#: Text for the prompt's {{NAME}} slots (rules_scene.fill_prompt). It lands inside a JSON string: no double quotes,
#: no backslashes.
PROMPT_TEXTS = {"TACO_VERBS": "; ".join(f"'{v}' = {d}" for v, (d, _) in TACO_VERBS.items())}


def label_words(src: dict) -> tuple | None:
    """Accepted word stems of a TACO triplet label's verb (TACO_VERBS); None for other sources (the checker then
    uses the label's first word). An unknown TACO verb dies."""
    if src.get("kind") != "triplet":
        return None
    v = "-".join(str((src.get("fields") or {}).get("verb") or "").split())  # field 'put out' -> key 'put-out'
    if v not in TACO_VERBS:
        raise SystemExit(f"[TEXT_RULES] unknown TACO verb {v!r}: add its gloss and accepted words to TACO_VERBS")
    return TACO_VERBS[v][1]


#: Words removed from the object vocabulary: size, position, shape, part and generic words that do not name another
#: object by themselves.
NOT_OBJECT = (
    "big",
    "little",
    "large",
    "small",
    "left",
    "right",
    "tall",
    "narrow",
    "circular",
    "conical",
    "raw",
    "attached",
    "base",
    "body",
    "hand",
    "handle",
    "lid",
    "cap",
    "light",
    "power",
    "work",
    "test",
    "kitchen",
    "french",
    "mouth",
    "cavity",
    "frame",
    "holder",
    "case",
    "container",
    "mesh",
    "hub",
    "food",
    "fruit",
    "toy",
    "stick",
    "roll",
    "wash",
    "washing",
    "watering",
    "top",
    "bottom",
    "side",
    "part",
    "piece",
    "whole",
    "mix",
    "model",
)


#: Container and packaging nouns: right after a given object name they are part of that object ('lotion bottle').
CONTAINER = (
    "bottle",
    "can",
    "jar",
    "tube",
    "box",
    "carton",
    "bag",
    "pack",
    "packet",
    "tin",
    "tub",
    "pouch",
    "canister",
    "container",
    "case",
    "cup",
    "sachet",
    "spray",
    "dispenser",
)


#: R15 talk about the source, labels or metadata (the text describes its input instead of the clip: 'No source
#: label is given, so the task is inferred from the motion', 'The source label says ...', 'the metadata lists both
#: hands'). An object label ('the label on the bottle') is not matched: the bare word 'label' is not caught. Also
#: covered: recording / measurement / summary nouns ('recording', 'measured motion', 'phase table', 'segment
#: index'), position of the segment ('segment of a longer', 'first segment of'), 'listed as', 'according to the
#: <input>', 'the data shows'. 'during this segment' passes; the TACO verb 'measure' ('as if measuring') passes —
#: only 'measured motion|as' and 'measurably' match. Segment-scope talk (SEG_SCOPE — 'This segment covers only the
#: carrying part', 'the clip shows only the grasp'): a segment reference and a scope word in one sentence.
SEG_REF = (
    r"(?:this (?:segment|clip|portion|excerpt)|the (?:segment|portion|excerpt)|(?:in|of|during) the clip|"
    r"the clip (?:shows|covers|is|only|ends|begins))"
)
SEG_SCOPE = (
    r"(?:only|covers?|covered|shown|described|depicts?|depicted|captures?|captured|contains?|includes?|"
    r"focus(?:es)?)"
)
META_RE = re.compile(
    r"\b(?:metadata|object list|inferred from|the source\b|source (?:label|text|sentence)s?|"
    r"(?:no|the) (?:source )?labell?(?:ed)? (?:action|is|says|was|given)|no labell?(?:ed)?\b|"
    r"labell?ed action|matching the label|recordings?\b|"
    r"measured (?:motion|movement|rotation|trajectory|data|values?|as)\b|measurably|"
    r"measurements? (?:shows?|indicates?|suggests?|says?)\b|motion summary|"
    r"phase table|segment index|(?:segment|part|portion) of (?:a|the) "
    r"(?:longer|larger|full|whole|same|"
    r"original)|(?:first|second|third|last|final|earlier|later|next|previous) (?:segment|portion) of|"
    r"listed (?:as|in|among)|according to the (?:source|label|data|metadata|summary|measurement|"
    r"motion)|"
    r"data (?:shows?|indicates?|suggests?)|"
    rf"{SEG_REF}\b[^.;]*\b{SEG_SCOPE}\b|\b{SEG_SCOPE}\b[^.;]*\b{SEG_REF})",
    re.I,
)
