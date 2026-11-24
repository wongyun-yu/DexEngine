"""Every code pointer of the rules registry (owner, scope source and threshold `path:symbol`) resolves in this tree."""

from __future__ import annotations

from dexengine.data import rules as RR


def test_every_rule_pointer_resolves():
    bad = []
    for r in RR.resolve():
        refs = [("owner", r.owner)] + [(label, ref) for label, ref in r.thresholds]
        bad += [f"{r.id} {kind}: {ref}" for kind, ref in refs if not RR._sym(ref)[1]]
        if r.scope_from:
            try:
                RR.scope_of(r)
            except ValueError as e:
                bad.append(f"{r.id} scope: {e}")
    assert not bad, "registry pointers that do not resolve:\n" + "\n".join(bad)
