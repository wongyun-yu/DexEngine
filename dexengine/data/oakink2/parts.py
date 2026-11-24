#!/usr/bin/env python3
"""OakInk2 part identity, read only from the official release
`object_affordance/{object_part_tree,part_desc,instance_id}.json` (D1-d).

Why: the second field of an object id `O02@NNNN@PPPPP` is a **category group**, not an instance (`O02@0202@00001/2` =
two
test tubes, `O02@0017` = two plates). Reading "same number = part pair" would group separate objects as parts. Part
relations exist only in the part tree OakInk2 releases (root instance → child parts; 216 ids · 141 parts · 53 roots).

API (all rids look like `O02@0039@00002`):
  rid_of(obj_id)      tree/delivery obj_id (`right_o02_0039_00002`, `raw_o02_0039_00002`, `o02_0039_00002`) → rid
  is_part(rid)        whether it is someone's child in the official tree (= a part of an instance)
  root_of(rid)        root instance rid for a part, else None   ·  same_instance(a, b)
  siblings(rid)       all other parts of the same root (sub-parts included), excluding itself
  name_of(rid)        official English name (part_desc)
Source: dexengine.core.layout raw layer `data/oakink2/raw/hf/object_affordance/`; without the json files, only those 4
json files are extracted from `object_affordance.tar` in the same folder (models are not extracted).
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from functools import lru_cache
from pathlib import Path

from dexengine.core import paths as P  # noqa: E402

JSONS = ("object_part_tree.json", "part_desc.json", "instance_id.json", "affordance_label.json")


def affordance_root() -> Path:
    """Location of the official part tree json (dexengine.core.layout raw layer); read by the opener (affordance_dir)
    and the statement (input_files)."""
    return P.layer_path("raw", "oakink2") / "hf" / "object_affordance"


def input_files() -> list[Path]:
    """The four json files this module opens (or that must exist; affordance_dir checks all four): the statement of the
    dataset-level inputs (coords.dataset_files collects them into the judge transfer list and the RL layer)."""
    return [affordance_root() / j for j in JSONS]


def affordance_dir() -> Path:
    d = affordance_root()
    if not all((d / j).is_file() for j in JSONS):
        tar = d.parent / "object_affordance.tar"
        if not tar.is_file():
            raise FileNotFoundError(
                f"OakInk2 official part tree missing: {d} (or {tar}) — download HF kelvin34501/OakInk-v2 "
                "object_affordance.tar"
            )
        with tarfile.open(tar) as tf:
            tf.extractall(
                d.parent, members=[m for m in tf.getmembers() if m.name in {f"object_affordance/{j}" for j in JSONS}]
            )
        print(f"[oakink2_parts] {tar.name}: extracted {len(JSONS)} json files → {d}", flush=True)
    return d


@lru_cache(maxsize=1)
def _load():
    d = affordance_dir()
    tree = json.loads((d / "object_part_tree.json").read_text())
    desc = json.loads((d / "part_desc.json").read_text())
    inst = set(json.loads((d / "instance_id.json").read_text()))
    rev = {ch: p for p, chs in tree.items() for ch in chs}
    return tree, rev, desc, inst


#: Version and tree marker key of the supplement copy rule (same part id = a copy of the delivery;
#: base_layer.hand_dir.supplements_for
#: does not load it). Only trees whose content the rule changed (a copy actually dropped) record this version in
#: task_info
#: rules[SUPP_KEY]; the generation of the hand-shared layer (dexengine.data.contact.fingerprint.GEN_RULES) uses it to
#: tell the new side,
#: and a tree with a copy dropped but no marker is stale (base_layer.supplements.supplement_mark_reason). Bump it when
#: the rule changes.
SUPP_KEY, SUPP_VERSION = "SUPP", "2026-10-12a"
#: Owners of the supplement file tags (<tree>__<tag>__<rid>.npz): required-object categories (oakink2.required ·
#: scene_requirement) · supports (support) · sibling parts (part, oakink2.raw_objects). Any other tag is an old
#: supplement
#: without an owner (right · left of the old contact branch); base_adapter refuses to load it (supp_tag) and
#: oakink2.required --export retires it (retire_unowned).
SUPP_TAGS = ("moved", "touched", "carried", "receiver", "support", "part")


def supp_tag(tag: str, name: str) -> str:
    """Check a supplement tag; ValueError for a tag without an owner (no ownerless supplement is loaded silently)."""
    if tag not in SUPP_TAGS:
        raise ValueError(f"supplement tag without an owner {tag!r}: {name} — oakink2.required --export retires it")
    return tag


def rid_of(obj_id: str) -> str:
    """`right_o02_0039_00002` / `raw_o02_0039_00002` / `o02_0039_00002` / `O02@0039@00002` → `O02@0039@00002`."""
    s = re.sub(r"^(right|left|raw|supp)_", "", obj_id)
    return s.upper().replace("_", "@")


def is_part(rid: str) -> bool:
    return rid in _load()[1]


def root_of(rid: str) -> str | None:
    rev = _load()[1]
    if rid not in rev:
        return None
    x = rid
    while x in rev:
        x = rev[x]
    return x


def same_instance(a: str, b: str) -> bool:
    ra, rb = root_of(a), root_of(b)
    return ra is not None and ra == rb


def descendants(rid: str) -> list[str]:
    tree = _load()[0]
    out = []
    for ch in tree.get(rid, []):
        out += [ch] + descendants(ch)
    return out


def siblings(rid: str) -> list[str]:
    r = root_of(rid)
    return [s for s in descendants(r) if s != rid] if r else []


def name_of(rid: str) -> str:
    return _load()[2].get(rid, {}).get("obj_name", rid)


if __name__ == "__main__":  # self-check: microwave, test tubes
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    for q in ("right_o02_0039_00001", "raw_o02_0039_00002", "o02_0202_00001", "O02@0017@00002"):
        r = rid_of(q)
        print(
            q,
            "→",
            r,
            name_of(r),
            "| part:",
            is_part(r),
            "root:",
            root_of(r),
            "| siblings:",
            [name_of(s) for s in siblings(r)],
        )
    assert same_instance("O02@0039@00001", "O02@0039@00002") and not same_instance("O02@0202@00001", "O02@0202@00002")
    print("[oakink2_parts] OK")
