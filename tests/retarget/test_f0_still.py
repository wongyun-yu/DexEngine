"""The f0 object repair's pinned stills (repair_still, repair_bodies): exact measure on the true shape, the grid solve,
the still's thresholds, the bodies it adds, and the identity of a side without stills."""

from __future__ import annotations

import types

import numpy as np
import pytest

from dexengine.data.objects import collider_shape as CS
from dexengine.retarget.f0 import repair_bodies as RB
from dexengine.retarget.f0 import repair_obj as RO
from dexengine.retarget.f0 import repair_still as RST
from tests.imports import own_module

_BOX_F = np.array(
    [
        [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
        [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],
    ]
)  # fmt: skip


def _box(c, h, inward=False):
    """(V, F) of an axis-aligned box (half sizes h), faces wound outward (inward: reversed)."""
    h = np.broadcast_to(np.asarray(h, float), (3,))
    V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * h + c
    return V, (_BOX_F[:, ::-1] if inward else _BOX_F)


def _cube(c, h):
    return np.array(
        [[c[0] + sx * h, c[1] + sy * h, c[2] + sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )


def _strict(c=(0.0, 0.0, 0.0), h=0.01, piece=None):
    """A one-piece hand (a 2 cm cube on link right_tip, or `piece` points, link frame = world at its centre c)."""
    s = types.SimpleNamespace(side="right", parts={"right_tip": [_cube((0, 0, 0), h) if piece is None else piece]})
    s.links = [("right_tip", 0, None, None)]
    return s, {"right_tip": (np.eye(3), np.asarray(c, float))}


def _slab_still():
    """A 10 cm × 10 cm × 4 cm slab, top face at z = 0, at the world origin (a pinned still on triangles)."""
    V, F = _box((0.0, 0.0, -0.02), (0.05, 0.05, 0.02))
    return RST.StillBody("plate", "touched_solid", CS.Triangles(V, F), np.eye(4))


def _hollow_still():
    """A closed hollow shell (outer 10 cm cube, inner 8 cm cavity wound inward) — a bowl's cavity: its convex hull is
    solid, its triangles hold an empty inside."""
    Vo, Fo = _box((0, 0, 0), 0.05)
    Vi, Fi = _box((0, 0, 0), 0.04, inward=True)
    return RST.StillBody("bowl", "receiver", CS.Triangles(np.r_[Vo, Vi], np.r_[Fo, Fi + 8]), np.eye(4))


def test_the_measure_reads_the_true_shape_both_ways():
    """A finger cube 3 mm into the slab reads 3 mm; the same cube in a hollow cavity reads 0 while the cavity's convex
    hull swallows it (the convex ruler would repair a hand that touches nothing)."""
    s, LT = _strict((0.0, 0.0, 0.01 - 0.003))
    d, ln = RST.depth(_slab_still(), s, LT)
    assert d == pytest.approx(0.003, abs=1e-9) and ln == "right_tip"
    s, LT = _strict((0.0, 0.0, 0.0))
    assert RST.depth(_hollow_still(), s, LT) == (0.0, "")
    hull = CS.Parts([_box((0, 0, 0), 0.05)[0]])
    assert hull.depth(_cube((0, 0, 0), 0.01)) > 0.03


@pytest.mark.parametrize("still_mm,opened", [(1.5, True), (0.9, False)])
def test_a_still_opens_above_the_slot_threshold(capsys, still_mm, opened):
    """The still's thresholds are the slot's (repair_obj.THRESHOLDS_MM reads SLOT_REPAIR_MM / SLOT_RESIDUAL_MM)."""
    assert RO.THRESHOLDS_MM["still"] == (RO.SLOT_REPAIR_MM, RO.SLOT_RESIDUAL_MM)
    sat = RO.Sat([("main", "main", 0.0, ""), ("plate", "still", still_mm / 1000.0, "right_tip")])
    rep = {"side": "right"}
    assert RO.gate(rep, sat, "", False) is opened
    assert "still:plate=" in capsys.readouterr().out
    bad = RO.failed_checks(0.0, "", 0.0, 0.0, RO.Sat([("main", "main", 0.0, ""), ("plate", "still", 0.0012, "t")]))
    assert bad == "still plate SAT 1.20mm > 1mm (t)"


def _projection(monkeypatch, still, c0, piece=None):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    S = own_module("dexengine.retarget.f0.repair_obj_solve")
    nf = S.HC.nd - 6
    s, _LT = _strict(c0, piece=piece)
    main = RO.CookedObject([_cube((1.0, 0, 0), 0.02)], "m")  # far away
    at = RB.BodySet(main, 0, [], "d", [still]).at(np.r_[1.0, 0, 0, 1, 0, 0, 0])
    still.build_grid("right", [_cube(c0, 0.01)])
    box = (np.full(nf, -9.0), np.full(nf, 9.0))
    P = S.Projection(s, at, np.eye(3), np.zeros(3), (np.zeros(3), np.eye(3), np.zeros(nf)), -1.0, box, [])
    P.link_T = lambda dq: [(np.eye(3), np.asarray(c0, float) + np.asarray(dq[:3], float))]
    return S, s, P


def test_a_planted_frame0_overlap_is_projected_out_on_the_grid(monkeypatch):
    """A finger 4 mm into the slab: the grid rows move it out to the still's target, and the exact measure accepts the
    result (≤ the still's residual)."""
    still = _slab_still()
    c0 = np.array([0.0, 0.0, 0.01 - 0.004])
    S, s, P = _projection(monkeypatch, still, c0)
    assert P.res_mm.tolist() == [RO.OBJ_RESIDUAL_MM, RO.SLOT_RESIDUAL_MM]
    assert RST.depth(still, s, {"right_tip": (np.eye(3), c0)})[0] == pytest.approx(0.004, abs=1e-9)
    r_t = np.minimum(RO.OBJ_TOUCH_MM, P.res_mm * S.STRICT_MARGIN_SCALE) / 1000.0
    dq = P.solve(r_t)
    after = RST.depth(still, s, {"right_tip": (np.eye(3), c0 + dq[:3])})[0]
    assert after * 1000.0 <= RO.SLOT_RESIDUAL_MM and np.linalg.norm(dq[:3]) < 0.006


def test_a_thin_wall_through_a_finger_is_read_from_the_wall_side(monkeypatch):
    """A 1 mm plate through the middle of a finger cube: no cube vertex is inside the plate, the plate's surface samples
    inside the cube carry the row (exit along the cube's face), and the exact measure accepts the result."""
    V, F = _box((0.0, 0.0, 0.0), (0.05, 0.05, 0.0005))
    still = RST.StillBody("tray", "touched", CS.Triangles(V, F), np.eye(4))
    c0 = np.array([0.0, 0.0, 0.004])
    S, s, P = _projection(monkeypatch, still, c0)
    r_t = np.minimum(RO.OBJ_TOUCH_MM, P.res_mm * S.STRICT_MARGIN_SCALE) / 1000.0
    rows = P.rows(np.zeros(S.HC.nd), r_t)[0]
    assert rows and RST.depth(still, s, {"right_tip": (np.eye(3), c0)})[0] > 0.005
    dq = P.solve(r_t)
    assert RST.depth(still, s, {"right_tip": (np.eye(3), c0 + dq[:3])})[0] * 1000.0 <= RO.SLOT_RESIDUAL_MM


def _ball(r=0.01, n=300):
    """n points on a sphere of radius r (a Fibonacci lattice): a dense convex piece, its hull ~2n planes."""
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1.0 - 2.0 * i / n), np.pi * (1.0 + 5.0**0.5) * i
    return r * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def _same_rows(a, b):
    return len(a) == len(b) and all(
        x[0] == y[0] and x[1] == y[1] and np.array_equal(x[2], y[2]) and np.array_equal(x[3], y[3])
        for x, y in zip(a, b, strict=True)
    )


def test_the_wall_gate_gives_the_full_plane_rows(monkeypatch):
    """The wall term's gate (gate_planes: the piece's hull planes facing 26 directions, a polytope holding the piece)
    leaves out only samples that set no row: a dense piece (a 300-point ball, ~600 hull planes) through a 1 mm plate,
    from 12 mm deep to clear, reads the rows of the full plane test on every sample (a gate that passes all) bit for
    bit. A gate cut 1 mm into the piece (one that does not hold it) changes them — the comparison sees a wrong gate."""
    V, F = _box((0.0, 0.0, 0.0), (0.05, 0.05, 0.0005))
    still = RST.StillBody("tray", "touched", CS.Triangles(V, F), np.eye(4))
    S, _s, P = _projection(monkeypatch, still, np.zeros(3), piece=_ball())
    e = P.pieces[0][2]
    assert len(P.gates) == 1 and len(P.gates[0]) <= 26 < len(e)
    r_t = np.minimum(RO.OBJ_TOUCH_MM, P.res_mm * S.STRICT_MARGIN_SCALE) / 1000.0
    gate, full = list(P.gates), [np.array([[0.0, 0.0, 0.0, -1.0]])]
    cut = [P.gates[0] + np.array([0.0, 0.0, 0.0, 0.001])]
    seen, differs = 0, False
    for dz in np.linspace(-0.012, 0.012, 25):
        dq = np.zeros(S.HC.nd)
        dq[:3] = (0.003, -0.002, dz)
        P.gates = gate
        got = P.rows(dq, r_t)[0]
        P.gates = full
        assert _same_rows(got, P.rows(dq, r_t)[0]), dz
        P.gates = cut
        differs |= not _same_rows(got, P.rows(dq, r_t)[0])
        seen += bool(got)
    assert 0 < seen < 25 and differs


def test_a_sample_reads_the_same_plane_values_in_any_batch():
    """repair_still.plane_values gives each sample the same values, bit for bit, alone, in any subset of a batch and in
    the whole batch — the property that keeps the wall gate's cut from moving a residual (a matmul row's rounding
    follows the row count; on this BLAS one sample alone already reads different ulps)."""
    rng = np.random.default_rng(42)
    e = np.c_[rng.normal(size=(600, 3)), 0.02 * rng.normal(size=600)]
    e[:, :3] /= np.linalg.norm(e[:, :3], axis=1, keepdims=True)
    x = 0.02 * rng.normal(size=(300, 3))
    full = RST.plane_values(x, e)
    for i in range(0, 300, 7):
        assert np.array_equal(RST.plane_values(x[i : i + 1], e)[0], full[i]), i
    for _ in range(50):
        k = rng.random(300) < rng.uniform(0.02, 0.6)
        assert np.array_equal(RST.plane_values(x[k], e), full[k])


def test_the_deepest_sample_is_the_exact_one(monkeypatch):
    """repair_still.deepest gives the sample and plane values that plane_values over the whole batch gives (the first
    on ties), bit for bit, on batches holding an exact copy placed before the deepest sample and a copy moved by
    ~1e-16 m — also when the matmul screen is off by up to PLANE_TIE_M / 4. Off by up to 4 PLANE_TIE_M the screen
    drops the first copy and the pick moves: the margin is what makes it exact."""
    rng = np.random.default_rng(7)
    e = np.c_[rng.normal(size=(600, 3)), 0.02 * rng.normal(size=600)]
    e[:, :3] /= np.linalg.norm(e[:, :3], axis=1, keepdims=True)

    def ref(x):
        v = RST.plane_values(x, e)
        j = int(np.argmin(v.max(1)))
        return j, v[j]

    xs = []
    for _ in range(60):
        x = 0.02 * rng.normal(size=(int(rng.integers(3, 150)), 3))
        j = ref(x)[0]
        x = np.insert(x, int(rng.integers(0, j + 1)), x[j], axis=0)  # an exact copy before the deepest sample
        xs.append(np.insert(x, int(rng.integers(0, len(x) + 1)), x[j] + 1e-16 * rng.normal(size=3), axis=0))
    fast = RST._plane_values_fast

    def picks():
        return [RST.deepest(x, e) for x in xs]

    def same(got):
        return all(g[0] == r[0] and np.array_equal(g[1], r[1]) for g, r in zip(got, map(ref, xs), strict=True))

    assert same(picks())
    assert sum(len(x) * len(e) > RST.SCREEN_MIN_READS for x in xs) > 40  # most batches go through the screen
    for scale, want in ((0.25, True), (4.0, False)):
        s = scale * RST.PLANE_TIE_M
        monkeypatch.setattr(
            RST, "_plane_values_fast", lambda x, e, s=s: fast(x, e) + rng.uniform(-s, s, (len(x), len(e)))
        )
        assert same(picks()) is want, scale


def test_the_grid_filled_in_bricks_reads_the_whole_grid(monkeypatch):
    """A grid fills a brick of nodes the first time a read reaches it (repair_still.Grid): reads anywhere — cells that
    straddle brick borders on one, two or three axes included — give the values of the grid filled at once, bit for
    bit, and a node no read reached stays unfilled."""
    monkeypatch.setattr(RST, "GRID_BRICK", 4)
    V, F = _box((0.0, 0.0, -0.02), (0.05, 0.05, 0.02))
    ruler = CS.Triangles(V, F)
    lo, hi = np.array([-0.02, -0.02, -0.03]), np.array([0.02, 0.025, 0.01])
    g = RST.Grid(ruler, lo, hi)
    full = RST.Grid(ruler, lo, hi)
    P = np.stack(np.meshgrid(*full.ax, indexing="ij"), -1).reshape(-1, 3)
    full.S[...] = ruler.signed(P).reshape(tuple(full.n))
    full.filled[...] = True
    assert not g.filled.any()
    rng = np.random.default_rng(0)
    for Q in (lo + rng.random((40, 3)) * 0.004, lo + rng.random((400, 3)) * (hi - lo)):
        a, b = g.at(Q), full.at(Q)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert g.filled.any() and not g.filled.all()
    assert np.array_equal(
        g.S[g.filled.repeat(4, 0).repeat(4, 1).repeat(4, 2)[: g.n[0], : g.n[1], : g.n[2]]],
        full.S[g.filled.repeat(4, 0).repeat(4, 1).repeat(4, 2)[: g.n[0], : g.n[1], : g.n[2]]],
    )


def test_a_hand_in_a_cavity_is_not_touched(monkeypatch):
    """A finger inside the hollow still's cavity: no still row, so the projection leaves it where it is."""
    still = _hollow_still()
    S, s, P = _projection(monkeypatch, still, np.zeros(3))
    r_t = np.minimum(RO.OBJ_TOUCH_MM, P.res_mm * S.STRICT_MARGIN_SCALE) / 1000.0
    assert P.rows(np.zeros(S.HC.nd), r_t)[0] == []
    assert np.array_equal(P.solve(r_t), np.zeros(S.HC.nd))


def test_stills_are_pinned_rows_whose_hand_collides(monkeypatch):
    """Only STATIC_SPAWN rows with the training preset's hand channel ON are stills: a SLOT row and a still whose hand
    channel is OFF are not."""
    from dexengine.data.objects import extra_slots as XS
    from dexengine.rl.tasks.scene import collision_presets as CP

    rows = [
        {"name": "lid", "cls": XS.SLOT, "kind": "moving"},
        {"name": "rack", "cls": XS.STATIC_SPAWN, "kind": "touched"},
        {"name": "tray", "cls": XS.STATIC_SPAWN, "kind": "still_off"},
    ]
    monkeypatch.setattr(
        CP, "extra_collide", lambda kind, full=None, ds=None: {"hand": "OFF" if kind == "still_off" else "ON"}
    )
    ti = {"objects": [{"name": n, "qpos_start": 7 * i} for i, n in enumerate(("lid", "rack", "tray"))]}
    got = RB.hand_on_stills(rows, ti, "oakink2")
    assert [(o["name"], r["kind"]) for o, r in got] == [("rack", "touched")]


def test_a_still_joins_the_key_only_where_it_stands():
    """The policy line (the f0 key's object part) carries the still's values only for a side with stills."""
    assert RO.policy_line(True) == RO.policy_line(True, False) and "still" not in RO.policy_line(True)
    line = RO.policy_line(True, True)
    assert "still open>1.0" in line and f"grid={RST.GRID_STEP_M!r}" in line and "ruler=" in line


@pytest.mark.parametrize(
    "others,opened",
    [
        ([("plate", "still", 0.0015, "t")], True),  # a still opens: the object branch lifts the hand
        ([("plate", "still", 0.0009, "t")], False),  # under its open threshold: the desk repair keeps the side
        ([("lid", "slot", 0.003, "t")], False),  # a slot alone does not move a desk-deep side off the desk repair
    ],
)
def test_a_desk_deep_side_opens_only_for_a_still(capsys, others, opened):
    """On a desk-deep side the gate opens only for a still over its open threshold (a main object 8 mm deep does not
    open it); desk_start then gives frame 0 the desk repair's floor and cap, and nothing on any other side."""
    rep = {"side": "right"}
    assert RO.gate(rep, RO.Sat([("main", "main", 0.008, "t"), *others]), "", True) is opened
    assert rep.get("obj_desk_start", False) is opened
    assert ("desk-deep start" in capsys.readouterr().out) is opened
    assert RO.desk_start(rep, 0.1, 0.0, 0.003) == (pytest.approx((0.103, 0.0)) if opened else None)
    assert RO.desk_start({"side": "right"}, 0.1, 0.0, 0.003) is None


def test_a_desk_deep_start_is_accepted_by_the_desk_rule():
    """Frame 0 of a desk-deep start passes only at the desk repair's margin (not 'no deeper than the start')."""
    assert RO.failed_checks(0.0, "", 0.0004, 0.022) == ""
    assert RO.failed_checks(0.0, "", 0.0004, 0.022, None, 0.0) == "desk 0.400mm > 0mm (desk-deep start)"
    assert RO.failed_checks(0.0, "", 0.0, 0.022, None, 0.0) == ""


def test_a_hand_in_the_desk_and_a_still_is_lifted_out_of_both(monkeypatch):
    """A finger 4 mm into a slab still and 6 mm below the desk plane: with the desk-deep start's floor (desk plane +
    clearance) the projection clears both — the still by the exact measure, the desk at every hull vertex."""
    still = _slab_still()
    c0 = np.array([0.0, 0.0, 0.01 - 0.004])
    S, s, P = _projection(monkeypatch, still, c0)
    desk_z, clearance = 0.002, 0.003
    P.floor = RO.desk_start({"obj_desk_start": True}, desk_z, 0.0, clearance)[0]
    r_t = np.minimum(RO.OBJ_TOUCH_MM, P.res_mm * S.STRICT_MARGIN_SCALE) / 1000.0
    dq = P.solve(r_t)
    c1 = c0 + dq[:3]
    assert RST.depth(still, s, {"right_tip": (np.eye(3), c1)})[0] * 1000.0 <= RO.SLOT_RESIDUAL_MM
    assert (c1[2] - 0.01) >= desk_z  # the cube's lowest vertex is above the desk plane
