"""Rule S-levels (opt-in): **observed supports carry what rests on them, at their measured height** — a disc or pedestal
stands only where no observed support is under the object.

Observed support = a tracked static object the training scene stands at its measured (frame-0) pose and that collides
with the main object: a static extra slot of kind receiver whose receiving partner is that main object (the
measurement `extra_slots_geom.receivers` — stands on at frame 0 · a rest interval · the end, or arrives; owner of the
row `dexengine/data/objects/extra_slots.py`). Such a body may stand at any height (a box on the desk, a tray, a shelf
object), so the supports of the scene are multi-level: the desk, the observed supports, and synthetic discs/pedestals
only where nothing observed is under the object. Today a rest disc or start disc is dropped only under a **stated**
support object (role support, `support_kind.rest_on_support`); with this rule a rest disc (at its rest frame) or a start
disc (frame 0)
whose main object stands on one of its observed supports (`support_kind.on_support_object`, the same test) is not raised
either — two bodies would carry one object, the disc piercing the observed one. Pedestals of fixed objects
(`static_fixed.rests_on`) and end pedestals (`end_on_object`) already yield to a spawned object under them.
Datasets that do not opt in: nothing changes (compute_desk_support byte-identical).
"""

from __future__ import annotations

LEVELS_RULE = "S-levels"
TAG = "[S_LEVELS]"


def on(ti: dict) -> bool:
    """Is S-levels on for this tree's dataset (opt-in, rules registry)? A dataset outside the registry is off."""
    from dexengine.data import rules as RR  # noqa: PLC0415

    ds = str(ti.get("dataset_name") or "")
    return ds in RR.ALL and RR.rule_on(LEVELS_RULE, ds)


def observed_supports(rows: list[dict], mains: set) -> dict:
    """{main object name: [observed support names]} from the extra-slot rows (`extra_slots.clip_rows`): static spawns
    of kind receiver whose receiving partner (`recv_by`) is that main object."""
    out: dict = {}
    for r in rows:
        if r.get("cls") == "STATIC_SPAWN" and r.get("kind") == "receiver" and r.get("recv_by") in mains:
            out.setdefault(str(r["recv_by"]), []).append(str(r["name"]))
    return out


def carried_by_observed(discs: list[dict], frame_of, verts_at, sup: dict, test) -> tuple[list, list]:
    """(kept, dropped) disc rows: a row is dropped when its main object `obj`, at `frame_of(row)`, stands on one of its
    observed supports at frame 0 (`test(V, [support vertices])`). `verts_at(name, frame)` = capture vertices."""
    kept, dropped = [], []
    for d in discs:
        names = sup.get(str(d["obj"]), [])
        on_obs = bool(names) and test(verts_at(str(d["obj"]), int(frame_of(d))), [verts_at(n, 0) for n in names])
        (dropped if on_obs else kept).append(d)
    return kept, dropped


def drop_carried(tree_dir, ti: dict, q, rest: list, starts: list) -> tuple[list, list, list]:
    """compute_desk_support's step (only when `on`): (rest discs kept, start discs kept, dropped rows)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.placement.coords.objects import _placers  # noqa: PLC0415
    from dexengine.placement.coords.support_kind import on_support_object  # noqa: PLC0415

    by = {str(o.get("name")): o for o in ti.get("objects") or []}
    mains = {n for n, o in by.items() if o.get("role") == "main"}
    sup = observed_supports(XS.clip_rows(str(tree_dir), ti, q, strict=False), mains)
    if not sup or not (rest or starts):
        return list(rest), list(starts), []
    placed, _ = _placers(tree_dir, ti, q)

    def verts_at(name, f):
        return placed(by[name], f)

    rk, rd = carried_by_observed(rest, lambda d: d["frame"], verts_at, sup, on_support_object)
    sk, sd = carried_by_observed(starts, lambda d: 0, verts_at, sup, on_support_object)
    if rd or sd:
        print(
            f"{TAG} {getattr(tree_dir, 'name', tree_dir)}: observed supports carry {len(rd)} rest · {len(sd)} start "
            f"discs — not raised ({[(d['obj'], d['kind']) for d in rd + sd]} on {sup})",
            flush=True,
        )
    return rk, sk, rd + sd
