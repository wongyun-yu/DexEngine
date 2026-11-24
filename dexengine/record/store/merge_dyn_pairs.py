"""S-dyn-pairs drop (b) in the final merge: the delivered visual overlap of the both-dynamic pairs whose collision the
scene turned off, measured on the delivered object poses before the final lock is taken (merge.measure_piece, as
J-self-pen: store.merge_self_pen). A trajectory deeper than scene_requirement.TRUE_SHAPE_OVERLAP_M on the visual meshes
(data/objects/dyn_pair_check.output_check) leaves the merge with all its episodes (looks), and the pieces line records
it (`dyn_pair_dropped`); a piece with nothing left still gets its line.

Statement (episode line field LINE_KEY, written by the store from the tree the episode was recorded from,
`line_statement`): {dataset, pairs: [{a, b, f0_m}] (the scene's pair, dyn_pairs.scene_pair of that tree, named by its
delivered columns object_<side>), objects: {column: {name, obj_id, mesh_dir, articulation}}}. A line without it (a
piece stored before this rule) fails the merge: its pairs are not known.
Reading: per trajectory (the episodes of one episode clip) each distinct pose table once (the looks of one rollout
share it): `observation.object_pose.<entity>` and, for an articulated object, `observation.object_joint_pos.<entity>`;
the trajectory's value is the deepest of its tables. A pair whose object the episode does not record (object_present
false) is not in the delivered scene, and the marker says so.
Callers: store.merge.measure_piece (measure, with merge_self_pen.fingerprint covering the bytes) · store.merge
.merge_piece (plan, after J-self-pen) · store.merge_check (line_problems) · store.form.form_episode (line_statement).
Markers: [DYN_PAIRS] output per trajectory · [DYN_PAIRS_SUM] per piece.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from dexengine.record.store import merge_overlap as MO
from dexengine.record.store.merge_check import DATA, die

RULE = "S-dyn-pairs-output"
LINE_KEY = "dyn_pairs"
POSE, JOINT = "observation.object_pose.", "observation.object_joint_pos."
SUM = "[DYN_PAIRS_SUM]"


def line_statement(meta: dict, clip: str) -> dict:
    """The episode line's statement (module doc) from the tree of record meta env.sequence_dir + clip: the scene's one
    pair (dyn_pairs.scene_pair: its right and left main objects), named by the delivered columns object_<side>
    (store.form.main_object), with those objects' obj_id, mesh_dir and articulation. A tree that cannot be found
    raises (the pairs would be unknown). One read per tree (cached)."""
    from dexengine.record import tree_ref as TR  # noqa: PLC0415

    seq = (meta.get("env") or {}).get("sequence_dir")
    tree, why = TR.meta_tree(seq, clip)
    if tree is None:
        raise ValueError(f"[DYN_PAIRS] {clip}: no tree for the pair-off statement (env.sequence_dir={seq!r} · {why})")
    return json.loads(_tree_statement(str(tree)))


@functools.lru_cache(maxsize=256)
def _tree_statement(tree: str) -> str:
    from dexengine.core import stdio  # noqa: PLC0415
    from dexengine.data.objects import dyn_pairs as DP  # noqa: PLC0415
    from dexengine.record.store import form as FM  # noqa: PLC0415

    ti = json.loads((Path(tree) / "task_info.json").read_text(encoding="utf-8"))
    q = np.load(Path(tree) / "0" / "trajectory_kinematic.npz")["qpos"]
    with stdio.capture():
        p = DP.scene_pair(tree, ti, q)
    out = {"dataset": str(ti["dataset_name"]), "pairs": [], "objects": {}}
    if p is not None:
        objs = {str(o["name"]): o for o in ti["objects"]}
        a, b = (FM.main_object(p["sides"][n]) for n in (p["a"], p["b"]))
        out["pairs"].append({"a": a, "b": b, "f0_m": p["f0_m"]})
        for e, n in ((a, p["a"]), (b, p["b"])):
            out["objects"][e] = {"name": n, **{k: objs[n].get(k) for k in ("obj_id", "mesh_dir", "articulation")}}
    return json.dumps(out)


def _table(parquet, ents: list) -> tuple:
    """(poses {entity: (T,4,4)}, joints {entity: (T,)}, key) of one episode file's object columns."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    from dexengine.data.objects import extra_slots_geom as XG  # noqa: PLC0415

    have = set(pq.read_schema(str(parquet)).names)
    cols = [c for e in ents for c in (POSE + e, JOINT + e) if c in have]
    t = pq.read_table(str(parquet), columns=cols)
    arr = {c: np.asarray(t.column(c).to_pylist(), float) for c in cols}
    poses = {e: XG.pose_mats(arr[POSE + e], 0) for e in ents if POSE + e in arr}
    joints = {e: arr[JOINT + e].reshape(len(arr[JOINT + e]), -1)[:, 0] for e in ents if JOINT + e in arr}
    h = hashlib.sha256(b"".join(np.ascontiguousarray(arr[c]).tobytes() for c in sorted(arr))).hexdigest()
    return poses, joints, h


def measure(pc: dict) -> dict:
    """{episode clip: (worst result, [piece episode positions])} of a read piece, in episode order; result =
    {depth_m, frame, pair, absent, verdict} (no pair: depth 0, keep)."""
    from dexengine.data.objects import dyn_pair_check as DC  # noqa: PLC0415

    by: dict[str, list[int]] = {}
    for k, e in enumerate(pc["eps"]):
        by.setdefault(e["clip"], []).append(k)
    seen, out = {}, {}
    for clip, ks in by.items():
        worst = {"depth_m": 0.0, "frame": -1, "pair": "", "absent": [], "verdict": DC.KEEP}
        for k in ks:
            e = pc["eps"][k]
            if LINE_KEY not in e:
                die(f"episode {e['episode_index']} ({clip}): no {LINE_KEY} statement — stored before S-dyn-pairs")
            st = e[LINE_KEY]
            here = [
                p for p in st["pairs"] if all((e.get("object_present") or {}).get(x, True) for x in (p["a"], p["b"]))
            ]
            absent = [f"{p['a']}×{p['b']}" for p in st["pairs"] if p not in here]
            if not here:
                worst["absent"] = sorted(set(worst["absent"]) | set(absent))
                continue
            poses, joints, h = _table(pc["files"][k][DATA], sorted({x for p in here for x in (p["a"], p["b"])}))
            h = (h, json.dumps({**st, "pairs": here}, sort_keys=True))  # one reading per pose table and statement
            if h not in seen:
                res = DC.output_check(st["dataset"], st["objects"], [(p["a"], p["b"]) for p in here], poses, joints)
                top = max(res["pairs"], key=lambda r: r["depth_m"])
                seen[h] = {"depth_m": top["depth_m"], "frame": top["frame"], "pair": f"{top['a']}×{top['b']}"}
                seen[h]["verdict"] = DC.verdict(top["depth_m"])
            if not worst["pair"] or seen[h]["depth_m"] > worst["depth_m"]:  # a reading at or under the cap is a bound
                worst = {**seen[h], "absent": worst["absent"]}
            worst["absent"] = sorted(set(worst["absent"]) | set(absent))
        out[clip] = (worst, ks)
    return out


def marker(clip: str, res: dict) -> str:
    """`[DYN_PAIRS] output <clip>: a×b 6.2 mm frame 40 -> drop|keep` (no pair-off pair: says so)."""
    from dexengine.data.objects import dyn_pair_check as DC  # noqa: PLC0415

    if not res["pair"]:
        tail = f" (not in the scene: {' · '.join(res['absent'])})" if res["absent"] else ""
        return f"[DYN_PAIRS] output {clip}: no pair-off pair{tail} -> {res['verdict']}"
    got = f"{res['pair']} {DC.reading(res['depth_m'])} frame {res['frame']}"
    return f"[DYN_PAIRS] output {clip}: {got} -> {res['verdict']}"


def plan(pc: dict, piece_id: str, got: dict) -> tuple[dict, list]:
    """(the piece without its dropped trajectories, the pieces-line records of the dropped ones) from `got` (measure:
    positions of the piece as measured, recorded in piece_eps). The piece may already be a subset (J-self-pen ran
    first): only its clips are judged and kept by clip. Prints the markers."""
    here = {e["clip"] for e in pc["eps"]}
    dropped = []
    for clip, (res, ks) in got.items():
        if clip not in here:
            continue
        print(marker(clip, res), flush=True)
        if res["verdict"] == "drop":
            why = f"{res['pair']} visual overlap {res['depth_m'] * 1e3:.1f} mm at frame {res['frame']}"
            dropped.append({"clip": clip, "rule": RULE, "why": why, "statement": res, "piece_eps": ks})
    gone = {d["clip"] for d in dropped}
    keep = [k for k, e in enumerate(pc["eps"]) if e["clip"] not in gone]
    print(
        f"{SUM} {piece_id} — trajectories {len(here)} · dropped {len(dropped)} · episodes kept {len(keep)} / "
        f"{len(pc['eps'])} · recorded in the pieces line dyn_pair_dropped",
        flush=True,
    )
    if not dropped:
        return pc, []
    return MO.keep_eps(pc, keep), dropped


def line_problems(r: dict) -> list:
    """A pieces line's dyn_pair_dropped records: each a drop of this rule, its clip not among the line's episodes."""
    from dexengine.data.objects import scene_requirement as SR  # noqa: PLC0415

    bad = []
    for d in r.get("dyn_pair_dropped") or []:
        st = d.get("statement") or {}
        if d.get("rule") != RULE or st.get("verdict") != "drop" or not d.get("piece_eps"):
            bad.append(f"piece {r['piece_id']}: dyn_pair_dropped record {d.get('clip')} is not a {RULE} drop")
        elif not float(st.get("depth_m", 0.0)) > SR.TRUE_SHAPE_OVERLAP_M:
            bad.append(f"piece {r['piece_id']}: dyn_pair_dropped record {d.get('clip')} is not over the line")
        if d.get("clip") in set(r.get("episode_clips") or []):
            bad.append(f"piece {r['piece_id']}: dyn_pair_dropped clip {d.get('clip')} is in the final")
    return bad
