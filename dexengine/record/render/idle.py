"""Idle hand measurement (render session): for one-hand bundles, check that the opposite hand of an idle-hand clip is
in the render body and measure where it is.

The render body USD of the one-hand launcher is the two-hand body (the idle hand's joints are in the
  trajectory entity robot too, and the render draws the values with which the stage-1 rollout held that hand at its
  idle pose; no render-only hand is added).
Idle hand = bodies of the robot articulation whose names start with '<side>_' (arm links are 'link_<side>_arm_*') on
  the side that is not active (record meta sides). Measured on the first and middle video frames: those bodies'
  visual mesh vertices (USD, PhysX pose -> env) vs the desk top box (render.body_geom.desk_box, signed distance) · the
  nearest vertex distance to the body visual meshes (render.body.body_boxes) · the depth below the top of base (the
  lump lying on the floor; vertical lines of the lowest IDLE_SAMPLE vertices vs base triangles, since vertex distances
  miss a wide top) · the lowest z. Meta idle_hand · marker [COLLECT_IDLE].
  What to do on overlap is a human decision: it does not die.
"""

from __future__ import annotations

from dexengine.core.errors import InputMissingError

MARK = "[COLLECT_IDLE]"
IDLE_SAMPLE = 4000  # idle hand vertices checked for base penetration (lowest first)
SEED_STRIDE = 64  # every n-th idle hand vertex seeds the search bound of nearest_body_mesh


def log_problems(text: str) -> list[str]:
    return [] if f"{MARK} " in text else [f"idle hand marker '{MARK}' missing"]


def skipped(clip: str, why: str) -> None:
    """Marker line of a load whose idle hand is not measured (render.episode.load_traj camera_only): the line says that
    and why, so the session log keeps the marker (log_problems) without claiming a measurement."""
    print(f"{MARK} idle hand not measured for {clip}: {why}", flush=True)


def nearest_body_mesh(trees: dict, H) -> tuple[str, float]:
    """(body mesh, distance) of the body visual mesh nearest to the points H [N, 3]: the same result as the full search
    `near = {n: tree.query(H)[0].min()}; n_b = min(near, key=near.get)` (the first mesh in order wins a tie), found
    with a bounded search. A strided subsample of H gives an upper bound b (a real point pair, so b >= the true
    minimum); each mesh is then queried with distance_upper_bound just above the running bound. A point whose nearest
    vertex lies inside the bound gets the same exact distance as the full search, the point pair at a mesh's minimum
    lies inside the bound whenever that mesh can still win, and a mesh is replaced only by a strictly smaller
    distance — so name and distance equal the full search's bit for bit (tests/record/test_idle_nearest.py)."""
    import numpy as np

    seed = H[::SEED_STRIDE]
    bound = min(float(tr.query(seed)[0].min()) for tr in trees.values())
    best_n, best = None, np.inf
    for n, tr in trees.items():
        d = tr.query(H, distance_upper_bound=bound * (1 + 1e-9) + 1e-12)[0]
        m = float(d.min())
        if m < best:
            best_n, best = n, m
            bound = min(bound, m)
    if best_n is None:
        raise RuntimeError(f"{MARK} nearest body mesh: no mesh within the seed bound {bound} (a real pair) — cannot be")
    return best_n, best


def _link_points(stage, b, ids: list[int]) -> tuple[list, list, list]:
    """Bodies among the ids of articulation b that have visuals ('visuals' children) -> (ids, per-body vertex lists
    [N, 3] in the body frame, names of bodies without visuals)."""
    from pxr import Usd

    from dexengine.record.render import body as BODY
    from dexengine.record.render import body_geom as BG

    root = stage.GetPrimAtPath(BODY._root_path(b))
    prims = {p.GetName(): p for p in Usd.PrimRange(root) if p.GetChild("visuals")}
    got, out, bare = [], [], []
    for i in ids:
        name = b.asset.body_names[i]
        P = BG.mesh_points(prims[name].GetChild("visuals"), prims[name]) if name in prims else []
        if len(P):
            got.append(i)
            out.append(P)
        else:
            bare.append(name)
    return got, out, bare


def _step_rows(steps, write, sel, org, trees, boxes, base, bare, aerial, dbox):
    """Per step of the trajectory: the idle hand's mesh points against the robot bodies, the desk and the base mesh."""
    import numpy as np

    import isaaclab.utils.math as mu

    from dexengine.record.render import body_geom as BG

    rows = []
    for t in steps:
        write(t)
        H = []
        for b, ids, pts, _nb in sel:
            b.asset.update(1e-6)
            pos = (b.asset.data.body_pos_w[0, ids] - org).detach().cpu().numpy().astype(float)
            rot = mu.matrix_from_quat(b.asset.data.body_quat_w[0, ids]).detach().cpu().numpy().astype(float)
            H += [P @ rot[k].T + pos[k] for k, P in enumerate(pts)]
        H = np.concatenate(H)
        n_b, d_b = nearest_body_mesh(trees, H)
        low = H[np.argsort(H[:, 2])[:IDLE_SAMPLE]]  # lowest vertices (penetration candidates)
        bp, bf = boxes[base][2], boxes[base][3]
        depth = BG.top_under(low, bp[bf]) - low[:, 2]
        rows.append(
            {
                "step": int(t),
                "links": sum(len(x[1]) for x in sel),
                "vertices": len(H),
                "no_visual": bare,
                "desk_gap_m": None if aerial else round(float(BG.box_sdist(H, dbox).min()), 4),
                "body_min_dist_m": round(d_b, 4),
                "body_mesh": n_b,
                "base_penetration_m": round(float(max(depth.max(), 0.0)), 4),
                "base_penetrating": int(np.sum(depth > 0)),
                "sampled": len(low),
                "z_min_m": round(float(H[:, 2].min()), 4),
            }
        )
    return rows


def measure(raw_env, binds, tf, desk: dict, steps: list[int], write, boxes: dict) -> dict | None:
    from scipy.spatial import cKDTree

    import omni.usd

    from dexengine.core.qpos_layout import SIDES  # owner of the side names
    from dexengine.record.render import body as BODY
    from dexengine.record.render import body_geom as BG

    active = {s["side"] for s in tf.meta["sides"]}
    robots = [b for b in binds if b.kind == "artic" and not b.name.startswith("object")]
    idle = [
        s for s in SIDES if s not in active and any(n.startswith(f"{s}_") for b in robots for n in b.asset.body_names)
    ]
    if not idle:
        print(f"{MARK} no idle hand (active {sorted(active)})", flush=True)
        return None
    stage = omni.usd.get_context().get_stage()
    org = raw_env.scene.env_origins[0]
    aerial = bool(desk.get("aerial_desk_dz"))
    dbox = None if aerial else BG.desk_box(desk)
    trees = {n: cKDTree(bx[2]) for n, bx in boxes.items()}
    base = next(k for k, v in BODY.chain()["mesh"].items() if v == "base")
    out = {}
    for s in idle:
        sel = [(b, [i for i, n in enumerate(b.asset.body_names) if n.startswith(f"{s}_")]) for b in robots]
        sel = [(b, *_link_points(stage, b, ids)) for b, ids in sel if ids]
        bare = [n for *_x, nb in sel for n in nb]
        if not any(ids for _b, ids, _pp, _nb in sel):
            raise InputMissingError(
                f"{MARK} idle hand {s} bodies have no visuals at all ({bare[:4]}): they are not drawn"
            )
        rows = _step_rows(steps, write, sel, org, trees, boxes, base, bare, aerial, dbox)
        out[s] = rows
    print(
        f"{MARK} idle hand {idle} · "
        + " · ".join(
            f"{s} step {r['step']}: links {r['links']} · desk top clearance {r['desk_gap_m']} m · body vertex min "
            f"distance {r['body_min_dist_m']} m ({r['body_mesh']}) · below the base top "
            f"{r['base_penetrating']}/{r['sampled']} max "
            f"{r['base_penetration_m']} m · lowest z {r['z_min_m']} m"
            for s, rr in out.items()
            for r in rr
        ),
        flush=True,
    )
    # rule is stored in the meta (data)
    return {
        "sides": idle,
        "frames": out,
        "rule": "vertices — desk = top-board box signed distance · body = nearest-vertex distance · "
        "base = depth below the top surface (vertical line)",
    }
