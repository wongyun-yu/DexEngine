"""MJCF contact pairs re-made after collision geoms are replaced — one rule for the start-state (a) inputs
(prep_start_a.transform) and the E5 SPIDER inputs (spider_arm.prep_inputs.transform). Standard library only."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def remake_pairs(contact, newmap):
    """Every <pair> on a replaced geom -> one pair per new geom (attributes copied, named <a>__<b>, each (a, b)
    once), appended to <contact>; newmap = {old geom: new geoms}. Returns (new pairs, seen (a, b))."""
    pairs, seen = [], set()
    for p in list(contact.findall("pair")):
        g1, g2 = p.get("geom1"), p.get("geom2")
        if g1 not in newmap and g2 not in newmap:
            continue
        contact.remove(p)
        for a in newmap.get(g1, [g1]):
            for b in newmap.get(g2, [g2]):
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                q = dict(p.attrib)
                q["geom1"], q["geom2"], q["name"] = a, b, f"{a}__{b}"
                pairs.append(q)
    for q in pairs:
        ET.SubElement(contact, "pair", q)
    return pairs, seen
