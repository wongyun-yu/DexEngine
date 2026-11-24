"""The 'not a handover' gate on bimanual shared-object cases (cases.gate)."""

from dexengine.data.cases import gate as G

GIVE, RECV = [[0.44, 1.5]], [[1.18, 2.76]]  # left gives, right receives · overlap 0.32 s · end 3.6 s


def test_unit_gate_conditions():
    ok = {"left": GIVE, "right": RECV}
    assert G.unit_gate(ok, 3.6, False) == []
    assert any(x.startswith("①") for x in G.unit_gate({"left": GIVE, "right": [[0.5, 2.76]]}, 3.6, False))
    assert any(x.startswith("②") for x in G.unit_gate({"left": [[0.44, 3.5]], "right": RECV}, 3.6, False))
    assert any(x.startswith("③") for x in G.unit_gate(ok, 3.6, True))
    assert any(x.startswith("④") for x in G.unit_gate({"left": GIVE, "right": [[1.6, 2.76]]}, 3.6, False))
    w = G.unit_gate({"left": GIVE, "right": []}, 3.6, False)
    assert len(w) == 1 and not G.is_out(w)  # one hand without an interval: cannot measure, keep


def test_art_units_and_regate():
    objs = [
        {"name": "drawer"},
        {"name": "drawer_lid", "articulation_group": {"base": "drawer", "role": "part"}},
        {"name": "cup"},
        {"name": "box", "articulation": {"type": "hinge"}},
    ]
    members = {"drawer": ["drawer", "drawer_lid"], "cup": ["cup"], "box": ["box"]}
    assert G.art_units(members, objs) == {"drawer", "box"}

    def case(name, used, grasp, kind="handover"):
        return {
            "case": name,
            "used": used,
            "T": 180,
            "fps": 50.0,
            "share": {u: {"kind": kind} for u in used},
            "grasp_s": grasp,
        }

    m, objs = {"u": ["u"], "v": ["v"]}, [{"name": "u"}, {"name": "v"}]
    good, bad = {"u": {"left": GIVE, "right": RECV}}, {"u": {"left": GIVE, "right": [[0.5, 2.76]]}}
    assert G.regate(case("BI_HANDOVER", ["u"], good), m, objs)[0] == "BI_HANDOVER"
    n, g = G.regate(case("BI_HANDOVER+ART1", ["u"], bad), m, objs)
    assert n == "BI_SHARED_MIXED+ART1" and "u" in g
    n, _ = G.regate(case("BI_HANDOVER_PLUS", ["u", "v"], dict(bad, v={"left": GIVE, "right": RECV})), m, objs)
    assert n == "BI_SHARED_PLUS"
    assert G.regate(case("BI_SHARED_SIM", ["u"], bad, "sim"), m, objs)[0] == "BI_SHARED_SIM"
