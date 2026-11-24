#!/usr/bin/env python3
"""The capsule table generator of arm_proxy — fit · bake · split into base and append tables · --write / --check /
--show. The fit rule is arm_proxy's module doc; the table it writes is arm_proxy.TABLE (+ one append file per new hand).

Usage (the retargeting python — the mesh stack — from the checkout root):
  python -m dexengine.placement.collision.capsule_fit --write | --check | --show
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.placement.collision import arm_proxy as AP
from dexengine.placement.collision.arm_proxy import (
    CHECK_TOL_M,
    FIT_DIRS,
    FIT_GRID,
    FIT_SAMPLE_M,
    JOINT_AXIS_MIN_M,
    RADIUS_QUANTILE,
    RADIUS_RULE,
    REBAKE,
    SCHEMA,
    TABLE,
    _load,
    _table,
    append_table,
    arm_key,
    file_hash,
    hand_key,
    src_digest,
    table_hash,
)


def _fib(n: int):
    import numpy as np

    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def _fit_axis(v, c, d, U, hH) -> dict:
    """Ends and radius on the line c + t·d minimising mean |support error| (l1: r = median gap)."""
    import numpy as np

    tv = (v - c) @ d
    du, base0 = U @ d, U @ c
    g = np.linspace(tv.min(), tv.max(), FIT_GRID)
    best = (np.inf, 0.0, 0.0, 0.0)
    for i, ta in enumerate(g):
        tb = g[i:]
        base = base0[None, :] + np.maximum(ta * du[None, :], tb[:, None] * du[None, :])  # (nb, dirs)
        r = np.median(hH[None, :] - base, axis=1)
        e = np.abs(base + r[:, None] - hH[None, :]).mean(1)
        j = int(np.argmin(e))
        if e[j] < best[0]:
            best = (float(e[j]), float(ta), float(tb[j]), float(r[j]))
    _, ta, tb, r = best
    r = max(r, 1e-3)
    err = base0 + np.maximum(ta * du, tb * du) + r - hH
    return {"p": c + d * 0.5 * (ta + tb), "d": d, "r": r, "L": max(tb - ta, 1e-4), "err": err}


def fit(v, child_off) -> dict:
    """The capsule of one hull (vertices v, link frame) — the fit rule of arm_proxy's module doc."""
    import numpy as np

    from dexengine.placement.collision.arm_surface import hull_samples
    from dexengine.robots import wrist_hull as G

    v = np.asarray(v, float)
    smp = hull_samples(v, G.hull_triangles(v), FIT_SAMPLE_M)
    U = _fib(FIT_DIRS)
    hH = (U @ v.T).max(1)
    c = smp.mean(0)
    X = smp - c
    _w, E = np.linalg.eigh(X.T @ X / len(X))
    cands = [(f"pca{i}", E[:, i] * (1.0 if E[np.argmax(np.abs(E[:, i])), i] >= 0 else -1.0)) for i in (2, 1, 0)]
    if np.linalg.norm(child_off) > JOINT_AXIS_MIN_M:
        cands.insert(0, ("joint", np.asarray(child_off, float) / np.linalg.norm(child_off)))
    best = None
    for name, d in cands:
        f = _fit_axis(v, c, d, U, hH)
        if best is None or np.abs(f["err"]).mean() < np.abs(best[1]["err"]).mean():
            best = (name, f)
    name, f = best
    r = max(float(np.quantile(f["r"] - f["err"], RADIUS_QUANTILE)), 1e-3)  # the gap = r − err
    e = f["err"] + (r - f["r"])
    return {
        "p": [float(x) for x in f["p"]],
        "d": [float(x) for x in f["d"]],
        "r": r,
        "L": float(f["L"]),
        "axis": name,
        "src": src_digest(v),
        "err_l1_mm": float(np.abs(e).mean() * 1e3),
        "err_p5_mm": float(np.percentile(e, 5) * 1e3),
        "err_p95_mm": float(np.percentile(e, 95) * 1e3),
        "err_max_mm": float(e.max() * 1e3),
        "err_min_mm": float(e.min() * 1e3),
    }


def child_offset(urdf, link: str):
    """First (by joint name) child-joint origin of ``link`` with a non-zero offset, in the link frame; 0 if none."""
    import xml.etree.ElementTree as ET

    import numpy as np

    offs = []
    for j in ET.parse(str(urdf)).getroot().findall("joint"):
        if j.find("parent").get("link") == link:
            o = j.find("origin")
            xyz = np.array([float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            if np.linalg.norm(xyz) > 0:
                offs.append((j.get("name"), xyz))
    return sorted(offs, key=lambda t: t[0])[0][1] if offs else np.zeros(3)


def bake() -> dict:
    """The table now: every arm with a collider revision (core.arms.ARM_SIM_COLLIDER_REV) × its sides × revision hulls,
    and every registered hand of that arm × side × finger group (hand_surface.hand_tree — the bodies the torso guard
    reads)."""
    from dexengine.core import arms as AR
    from dexengine.placement.collision import hand_surface as HS
    from dexengine.placement.collision import sim_collider as SC

    bodies: dict = {}
    for arm in sorted(AR.ARM_SIM_COLLIDER_REV):
        spec = AR.ARM_REGISTRY[arm]
        hands = SC.arm_hands(arm)
        if not hands:
            raise InputMissingError(
                f"[COLL_PROXY] ✗ arm {arm}: no registered hand — no URDF to read the arm shapes from"
            )
        for side in spec.sides:
            urdf = SC.combo_urdf(arm, hands[0])
            for b in SC.bodies(arm, side, urdf):
                if not isinstance(b, SC.Capsule):
                    bodies[arm_key(arm, b.link)] = fit(b.verts, child_offset(urdf, b.link))
            for hand in hands:
                u = SC.combo_urdf(arm, hand)
                ht = HS.hand_tree(str(u), SC.arm_link_names(arm, side)[-1], HS.finger_names(hand, side))
                for gi, g in enumerate(ht["groups"]):
                    if gi and g["verts"] is not None:
                        bodies[hand_key(hand, side, g["root"])] = fit(g["verts"], [0.0, 0.0, 0.0])
    return {
        "schema": SCHEMA,
        "radius_rule": RADIUS_RULE,
        "fit": {
            "dirs": FIT_DIRS,
            "grid": FIT_GRID,
            "sample_m": FIT_SAMPLE_M,
            "joint_axis_min_m": JOINT_AXIS_MIN_M,
            "radius_quantile": RADIUS_QUANTILE,
        },
        "bodies": dict(sorted(bodies.items())),
    }


def _text(t: dict) -> str:
    """The table file text (floats rounded to 9 decimals, sorted keys)."""

    def rnd(x):
        if isinstance(x, float):
            return round(x, 9)
        if isinstance(x, list):
            return [rnd(y) for y in x]
        if isinstance(x, dict):
            return {k: rnd(y) for k, y in x.items()}
        return x

    return json.dumps(rnd(t), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def compare(old: dict, new: dict) -> list[str]:
    """Differences that make the table stale: header · body set · src digests · any fitted number > CHECK_TOL_M (axis
    names included)."""
    out = [
        f"{k}: {old.get(k)!r} ≠ {new.get(k)!r}" for k in ("schema", "radius_rule", "fit") if old.get(k) != new.get(k)
    ]
    ob, nb = old.get("bodies", {}), new.get("bodies", {})
    out += [f"only in the table {k}" for k in sorted(set(ob) - set(nb))] + [
        f"only now {k}" for k in sorted(set(nb) - set(ob))
    ]
    for k in sorted(set(ob) & set(nb)):
        o, n = ob[k], nb[k]
        if o["src"] != n["src"] or o["axis"] != n["axis"]:
            out.append(f"{k}: src/axis {o['src']}/{o['axis']} ≠ {n['src']}/{n['axis']}")
            continue
        dev = max(
            abs(x - y)
            for f in ("p", "d", "r", "L")
            for x, y in zip(*((o[f], n[f]) if isinstance(o[f], list) else ([o[f]], [n[f]])), strict=False)
        )
        if dev > CHECK_TOL_M:
            out.append(f"{k}: fitted value difference {dev:.2e} > {CHECK_TOL_M:g}")
    return out


def main(argv=None) -> int:
    """--write | --check | --show (--check: rc 0 same · 1 stale — a hull changed, a body came/went, the fit moved >
    CHECK_TOL_M)."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="bake and write TABLE (atomic)")
    g.add_argument("--check", action="store_true", help="rebake and compare with TABLE — rc 0 same · 1 stale")
    g.add_argument("--show", action="store_true", help="radius · length · axis · support-function error per body")
    a = ap.parse_args(argv)
    if a.show:
        t = all_bodies(TABLE, _table())
        for k, e in t.items():
            print(
                f"{k:60s} r {e['r'] * 1e3:5.1f} L {e['L'] * 1e3:6.1f} {e['axis']:5s} error L1 {e['err_l1_mm']:4.1f} "
                f"p5/p95 {e['err_p5_mm']:+5.1f}/{e['err_p95_mm']:+5.1f} min/max {e['err_min_mm']:+5.1f}/"
                f"{e['err_max_mm']:+5.1f} mm"
            )
        print(f"[COLL_PROXY] table {TABLE.name} bodies {len(t)} hash {table_hash()}")
        return 0
    files = planned(TABLE, *split(bake(), hands_of(_table()["bodies"]) if TABLE.is_file() else set()))
    if a.write:
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        for path, t in files:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(_text(t), encoding="utf-8")
            tmp.replace(path)
        AP._STATE.clear()
        print(
            "[COLL_PROXY] table written "
            + " · ".join(f"{p.name} bodies {len(t['bodies'])} hash {file_hash(p, AP._STATE)}" for p, t in files)
        )
        return 0
    diff = [
        f"{p.name}: {d}"
        for p, t in files
        for d in (
            compare(json.loads(p.read_text(encoding="utf-8")), json.loads(_text(t))) if p.is_file() else ["no table"]
        )
    ]
    for d in diff[:20]:
        print(f"[COLL_PROXY] ✗ {d}")
    print(
        f"[COLL_PROXY] check {'OK' if not diff else 'STALE'} bodies {sum(len(t['bodies']) for _, t in files)} "
        f"differences {len(diff)} (rebake: {REBAKE})"
    )
    return 1 if diff else 0


# ── base table + append tables ──


def hands_of(bodies: dict) -> set[str]:
    """The hands whose bodies a table holds."""
    return {k.split("/")[1] for k in bodies if k.startswith("hand/")}


def split(new: dict, keep: set[str]) -> tuple[dict, dict[str, dict]]:
    """One baked table → (base table, {append hand: append table}). Base table = the arms + the hands in keep (keep
    empty — the first bake, no base table yet — every hand). The bodies of existing hands stay in the base table, so
    their identity never changes; a new hand goes to its own append table."""
    head = {k: v for k, v in new.items() if k != "bodies"}
    base, app = dict(head, bodies={}), {}
    for k, v in new["bodies"].items():
        h = k.split("/")[1] if k.startswith("hand/") else None
        if h is None or not keep or h in keep:
            base["bodies"][k] = v
        else:
            app.setdefault(h, dict(head, bodies={}))["bodies"][k] = v
    return base, app


def planned(table: Path, base: dict, app: dict[str, dict]) -> list[tuple[Path, dict]]:
    """(place, table) pairs to write · compare — the base table + one per append. An append file the bake does not
    produce (a base-table hand's · an unregistered hand's — two statements) on disk dies."""
    stray = sorted(
        f.name for f in table.parent.glob(f"{table.stem}_*.json") if f.stem[len(table.stem) + 1 :] not in app
    )
    if stray:
        raise SystemExit(f"[COLL_PROXY] ✗ append tables the bake does not produce {stray} — delete them")
    return [(table, base)] + [(append_table(table, h), t) for h, t in sorted(app.items())]


def all_bodies(table: Path, base: dict) -> dict:
    """The bodies of the base table + every append table (--show)."""
    out = dict(base["bodies"])
    for f in sorted(table.parent.glob(f"{table.stem}_*.json")):
        out.update(_load(f)["bodies"])
    return out


if __name__ == "__main__":
    sys.exit(main())
