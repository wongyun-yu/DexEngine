"""Reference-source readers of the D12c quality step (filtering.py): per-dataset source records (TACO legacy
trees, OakInk2 program info, H2O action labels, GRAB/ARCTIC intents) and their verdicts. HAND is the run's hand spec
directory, set by filtering's --hand; filtering's verdict functions and its input fingerprint read these readers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.data.cases import (
    # keys and read entry of the assignment record (per-hand contact · no contact · motion without contact)
    assign_version as AR,
)
from dexengine.data.readers import taco_source

dl = P.data_layout


ROOT = P.main_root()


HAND = ""  # hand spec directory; main sets it from --hand (no default)
#: "no source" reasons the source readers return; they enter the stored reference and the D12c input fingerprint
NO_LEGACY_TREE = "no legacy tree"
NAME_FORM_MISMATCH = "name form mismatch"
NO_OFFICIAL_LABELS = "no official labels (the test split labels are not public)"
SOURCE_REASONS = ("NO_LEGACY_TREE", "NAME_FORM_MISMATCH", "NO_OFFICIAL_LABELS")


def mains_by_hand(ti):
    out = {}
    for o in ti["objects"]:
        if o["role"] == "main":
            for s in ["right", "left"] if o["side"] == "both" else [o["side"]]:
                out[s] = o["obj_id"]
    return out


# Source readers of the reference material; the verdict functions and the input fingerprint (`input_fingerprint`) read
# through **the same functions**. Returns the source record (it enters the fingerprint as is) or a string (reason for no
# source; the verdict becomes source "none").
def _no_source(what, where):
    raise InputMissingError(
        f"[D12c] {what} reference source {where} is missing or empty — a missing root is not 'none'"
    )


def _taco_legacy(ti):
    """Whole task_info of the TACO source tree (measured grasps); location owner data_layout.taco_source_root."""
    lroot = ROOT / dl.taco_source_root()
    lp = (lroot if lroot.is_dir() else _no_source("taco legacy", lroot)) / ti["parent_clip"] / "task_info.json"
    taco_source.checked(lroot)  # a shipped pack is verified; the tree form is unverified (same task_info files)
    if not lp.exists():
        return NO_LEGACY_TREE
    return json.loads(Path(lp).read_text())


def ref_taco(ti):
    L = _taco_legacy(ti)
    if isinstance(L, str):
        return {"source": "none", "note": L}
    exp = {"right": os.path.basename(L["right_object_mesh_dir"]), "left": os.path.basename(L["left_object_mesh_dir"])}
    obs = mains_by_hand(ti)
    return {
        "source": "taco_legacy_grasp_assign",
        "expected": exp,
        "observed": obs,
        "agree": obs == exp,
        "side_assign": L.get("side_assign"),
    }


_OAK = {}


def _oak_index():
    """{hash12: [(key, range | None, description)]}; order = program_info key order (the same list as the intake rule
    oakink2.required.measure_root). Ranges (extension) and descriptions (desc_info) are looked up by key name; some
    sequences have different key sets, so reading by position would point at the wrong primitive. Keys missing from
    extension get range None (the verdict uses only the key's active hands)."""
    if _OAK:
        return _OAK
    from dexengine.data.oakink2 import (
        raw_objects as RO,  # owner of the clip key, the `_sNNN` rule and the source location
    )

    pi = RO.RAW / "program/program_info"
    E = RO.RAW / "program_extension/extension"
    D = RO.RAW / "program/desc_info"
    for f in sorted(pi.glob("*.json")) or _no_source("oakink2", pi):
        h = re.search(r"seq__([0-9a-f]+)__", f.name).group(1)[:12]
        ext = json.loads((E / f.name).read_text()) if (E / f.name).exists() else {}
        desc = json.loads((D / f.name).read_text()) if (D / f.name).exists() else {}
        _OAK[h] = [(k, ext.get(k), desc.get(k, {}).get("seg_desc")) for k in json.loads(f.read_text())]
    return _OAK


def _oak_entry(ti):
    """The tree's program entry (key, range, description). Clip key = the source clip (RO.intake_clip; segments `_segNN`
    and D36 pieces map to the source clip) · stage = its `_sNNN` (RO.SEQ_KEY) = program_info key order. Fails when the
    key cannot be decided or the stage is not in the program (the same gate as the intake rule; never passed silently as
    source none)."""
    from dexengine.data.oakink2 import raw_objects as RO

    m = RO.SEQ_KEY.fullmatch(RO.intake_clip(ti))
    idx = _oak_index()
    h, k = m.group(3), int(m.group(4))
    if h not in idx or k >= len(idx[h]):
        raise InputMissingError(
            f"[D12c] oakink2 {ti.get('clip_id')}: stage {k} not in program_info ({h}: "
            f"{len(idx.get(h, ()))} stages) — does it come from the same source as the intake rule?"
        )
    return idx[h][k]


def ref_oakink2(ti):
    e = _oak_entry(ti)
    if isinstance(e, str):
        return {"source": "none", "note": e}
    key, ranges, desc = e
    lh, rh = eval(key)  # "((2587, 3711), None)" → (lh, rh)
    active = [s for s, r in (("right", rh), ("left", lh)) if r is not None]
    inter = AR.record(ti, HAND)["quality"]["contact_frac"]
    obs = [s for s in ti["hands"] if inter.get(s, 0) > 0]
    agree = all(s in obs for s in active if s in ti["hands"])  # active hands should touch
    return {
        "source": "oakink2_program",
        "desc": desc,
        "primitive_key": key,
        "active_hands": active,
        "observed_touching": obs,
        "agree": agree,
        "ranges": ranges,
    }


INTENT_HANDS = {"offhand": 2, "takepicture": 2}  # grab: only intents that clearly use both hands give an expectation


def ref_intent(ds, ti):
    c = ti["parent_clip"]
    if ds == "grab":
        it = c.split("__", 1)[1].split("_")[1] if "__" in c else c
    elif ds == "arctic":
        it = c.split("-", 1)[1].split("_")[1] if "-" in c else c
    else:
        it = c.split("__", 1)[1].split("_")[1] if "__" in c else c
    inter = AR.record(ti, HAND)["quality"]["contact_frac"]
    n = sum(1 for v in inter.values() if v > 0)
    exp = INTENT_HANDS.get(it) if ds == "grab" else (2 if (ds == "arctic" and it == "use") else None)
    return {
        "source": f"{ds}_intent_in_name",
        "label": it,
        "expected_hands_touching": exp,
        "observed_hands_touching": n,
        "agree": (exp is None or n >= exp),
    }


# h2o: official action_labels (github taeinkwon/h2odataset, 30 fps frame ranges + action ids).
H2O_ACT = {
    0: "background",
    1: "grab book",
    2: "grab espresso",
    3: "grab lotion",
    4: "grab spray",
    5: "grab milk",
    6: "grab cocoa",
    7: "grab chips",
    8: "grab cappuccino",
    9: "place book",
    10: "place espresso",
    11: "place lotion",
    12: "place spray",
    13: "place milk",
    14: "place cocoa",
    15: "place chips",
    16: "place cappuccino",
    17: "open lotion",
    18: "open milk",
    19: "open chips",
    20: "close lotion",
    21: "close milk",
    22: "close chips",
    23: "pour milk",
    24: "take out espresso",
    25: "take out cocoa",
    26: "take out chips",
    27: "take out cappuccino",
    28: "put in espresso",
    29: "put in cocoa",
    30: "put in cappuccino",
    31: "apply lotion",
    32: "apply spray",
    33: "read book",
    34: "read espresso",
    35: "spray spray",
    36: "squeeze lotion",
}


H2O_NO_CONTACT_VERBS = ("apply", "background")  # actions that do not touch an object (from the official verb list)


_H2O = {}


def _h2o_labels():
    """path(subjectS/XY/N) → [(action_id, start_frame30, end_frame30, end_frame_total)]. The test split (subject4) has
    no labels."""
    if _H2O:
        return _H2O
    for f in sorted((ROOT / dl.render("raw", "h2o") / "action_labels").glob("action_*.txt")) or _no_source(
        "h2o", "action_labels"
    ):
        with open(f) as fh:
            for line in fh:
                p = line.split()
                if len(p) == 7 and p[1].startswith("subject"):
                    _H2O.setdefault(p[1], []).append((int(p[2]), int(p[3]), int(p[4]), int(p[6])))
    return _H2O


def _h2o_entry(ti):
    """The tree's official labels (path key, label list)."""
    m = re.match(r"^(subject\d)_(\w\d)__take(\d+)_cam4$", ti["parent_clip"])
    if not m:
        return NAME_FORM_MISMATCH
    key = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    labs = _h2o_labels().get(key)
    if not labs:
        return NO_OFFICIAL_LABELS
    return key, labs


def ref_h2o(ti):
    e = _h2o_entry(ti)
    if isinstance(e, str):
        return {"source": "none", "note": e}
    key, labs = e
    sg = ti["segment"]
    a, b = sg["start"], sg["end"]  # our 50 Hz frames
    acts = []
    for act, s0, e0, _tot in labs:
        k0, k1 = s0 * 50 / 30, e0 * 50 / 30
        ov = max(0.0, min(b, k1) - max(a, k0))
        if ov > 0:
            acts.append({"action": H2O_ACT[act], "overlap_s": round(ov / 50, 2)})
    dur_off = abs(labs[0][3] / 30 - ti["provenance"]["T_full"] / 50)
    contact_expected = any(x["action"].split()[0] not in H2O_NO_CONTACT_VERBS and x["overlap_s"] >= 0.5 for x in acts)
    inter = AR.record(ti, HAND)["quality"]["contact_frac"]
    observed = any(v > 0 for v in inter.values())
    return {
        "source": "h2o_action_labels",
        "path": key,
        "actions": acts,
        "expected_contact": contact_expected,
        "observed_contact": observed,
        "agree": (contact_expected == observed) if acts else True,
        "duration_official_s": round(labs[0][3] / 30, 2),
        "duration_ours_s": round(ti["provenance"]["T_full"] / 50, 2),
        "duration_agree": dur_off < 0.1,
        "label": ", ".join(x["action"] for x in acts) or "(outside the labelled intervals)",
    }


def ref_hrdexdb(ti):
    obj = ti["parent_clip"].split("__")[0]
    obs = mains_by_hand(ti)
    return {
        "source": "hrdexdb_object_in_name",
        "label": obj,
        "observed": obs,
        "agree": all(v == obj for v in obs.values()),
    }
