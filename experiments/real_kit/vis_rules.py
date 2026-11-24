#!/usr/bin/env python3
"""Real-print set rule check — values are read only from print_rules.json (the chain builds it from the augvis
catalog's PRINT_KEEP and PRINT_SET_DROP; no copy). check_episode: a print look's (_pgreen | _pblue | _ptex) meta
obj_keep must equal the table — returns the reason for old pieces or clips outside the table. verify <set>
<print_rules.json>: FAIL when the
finished set's meta/episodes.jsonl holds a dropped variant; last line [VISRULES] OK|FAIL."""

import json
import os
import re
import sys
from pathlib import Path

PRINT_LOOK = re.compile(r"r\d+_s\d+_p(?:green|blue|tex)")
# single-colour looks (OBJ_COLORS, r0_s02_green form) — checked only when the rules json has color_keep (else outside
# the rule, unchanged). v4 keeps the augvis catalog's color_keep objects (--color-keep of the dense render) unpainted
# in these looks; the set build checks it here.
COLOR_LOOK = re.compile(r"r\d+_s\d+_(?:red|green|blue)")


def load(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    keep = {}
    for c, o in d["keep"]:
        keep.setdefault(c, []).append(o)
    if not keep or not d["drop_variants"]:
        raise SystemExit(f"[VISRULES] FAIL empty rules {path}")
    ck = None
    if (
        "color_keep" in d
        # [[clip, object id], …] — built by the chain from the augvis catalog color_keep (value owner there; an empty
        # list checks that nothing is kept in single-colour looks)
    ):
        ck = {}
        for c, o in d["color_keep"]:
            ck.setdefault(c, []).append(o)
        ck = {c: sorted(v) for c, v in ck.items()}
    return {"keep": {c: sorted(v) for c, v in keep.items()}, "drop": set(d["drop_variants"]), "color_keep": ck}


def check_episode(base_clip: str, variant: str, look: str, spec: dict, rules: dict):
    if variant in rules["drop"]:
        return f"{variant} is a dropped variant"
    if rules.get("color_keep") is not None and COLOR_LOOK.fullmatch(look):
        want, got = rules["color_keep"].get(base_clip, []), spec.get("obj_keep") or []
        if got != want:
            return (
                f"{variant} {look}: plain-colour look meta obj_keep {got} ≠ color_keep {want} (painted kept "
                "object or widening)"
            )
        return None
    if not PRINT_LOOK.fullmatch(look):
        return None
    want, got = rules["keep"].get(base_clip, []), spec.get("obj_keep") or []
    if got != want:
        return f"{variant} {look}: meta obj_keep {got} ≠ table {want} (old render or widening)"
    return None


def verify(set_dir: str, rules_path: str) -> int:
    rules = load(rules_path)
    with open(os.path.join(set_dir, "meta/episodes.jsonl")) as fh:
        eps = [json.loads(x) for x in fh if x.strip()]
    bad = sorted({e["e5_source"]["clip"] for e in eps} & rules["drop"])
    n_var = len({e["e5_source"]["clip"] for e in eps})
    if bad or not eps:
        print(f"[VISRULES] FAIL dropped variants in the set {bad} · episodes {len(eps)}")
        return 1
    print(f"[VISRULES] OK episodes {len(eps)} · variants {n_var} · dropped variants 0/{len(rules['drop'])}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "verify":
        raise SystemExit("usage: python -m experiments.real_kit.vis_rules verify <set> <print_rules.json>")
    sys.exit(verify(sys.argv[2], sys.argv[3]))
