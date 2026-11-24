"""Inputs of the E5 offline judge: scenes from the kinematic tree, the session reference.npz, the rollout episode.npz
and the data-episode contact forces, plus the data-root and HOME-map path resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.sim_eval.judges.geometry import HZ, quat_R

MARK = "[E5_JUDGE]"


def set_data_root(root: str) -> Path:
    """Point the data-root resolver (dexengine.core.paths.main_root, env DEXENGINE_MAIN_ROOT) at one explicit root, so
    the extra-slot rule (extra_slots → coords.input_path) and the mesh reads here see the same tree. No default and no
    guessing: an empty root or one without data/ raises."""
    if not root:
        raise SystemExit(f"{MARK} --data-root is empty — the data root is a required input")
    r = Path(root).resolve()
    if not (r / "data").is_dir():
        raise SystemExit(f"{MARK} --data-root {r}/data does not exist")
    os.environ["DEXENGINE_MAIN_ROOT"] = str(r)
    P = _paths()
    got = P.main_root()
    if got != r:
        raise SystemExit(f"{MARK} paths.main_root() {got} != --data-root {r}")
    print(f"{MARK} data root {r} (paths.main_root)", flush=True)
    return r


def _paths():
    """The path owner (data root, data layout)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    return P


class Scene:
    """Object name → p (T,3) · R (T,3,3); force (T,) per object, or None for a reference scene. Rows at HZ."""

    def __init__(self, objs: dict, force: dict | None, label: str):
        self.o, self.force, self.label = objs, force, label
        self.T = len(next(iter(objs.values()))["p"])

    def padded(self, n: int) -> Scene:
        """The scene with its last frame repeated n more rows — the session's reference clock stays on the last frame
        after the clip ends."""
        o = {
            k: dict(
                v,
                p=np.concatenate([v["p"], np.repeat(v["p"][-1:], n, 0)]),
                R=np.concatenate([v["R"], np.repeat(v["R"][-1:], n, 0)]),
            )
            for k, v in self.o.items()
        }
        return Scene(o, None, self.label + f"+pad{n}")

    def static(self) -> Scene:
        """The scene frozen at frame 0 (the frame-0 control of judgeability): same length, zero force (no grasp)."""
        o = {
            k: dict(v, p=np.repeat(v["p"][:1], self.T, 0), R=np.repeat(v["R"][:1], self.T, 0))
            for k, v in self.o.items()
        }
        return Scene(o, {k: np.zeros(self.T) for k in self.o}, "frame0")


def task_objects(tree_clip: Path) -> tuple[dict, list[dict]]:
    """(task_info, its objects) of a tree clip."""
    ti = json.loads((tree_clip / "task_info.json").read_text())
    return ti, ti["objects"]


def kin_scene(tree_clip: Path, objs: list[dict]) -> tuple[Scene, float]:
    """Kinematic reference scene (tree frame) from 0/trajectory_kinematic.npz, and its rate."""
    z = np.load(tree_clip / "0" / "trajectory_kinematic.npz")
    q = z["qpos"]
    hz = float(z["frequency"]) if "frequency" in z.files else HZ
    o = {}
    for x in objs:
        s = x["qpos_start"]
        o[x["name"]] = {
            "meta": x,
            "p": q[:, s : s + 3].astype(np.float64),
            "R": quat_R(q[:, s + 3 : s + 7].astype(np.float64)),
        }
    return Scene(o, None, "reference_kin"), hz


def entity_map(tree_clip: Path, ti: dict) -> dict:
    """task_info object name → session physics-scene entity name(s). Main object = object_<side> (a shared object with
    side=both is object_right, else object_left); extras = extra_obj_<k> with k the order of the slot rule owner
    dexengine.data.objects.extra_slots.slot_rows (SLOT and STATIC_SPAWN only — render-only background objects are not
    in
    the physics scene)."""
    from dexengine.data.objects import extra_slots as XS

    rows, routed = XS.slot_rows(str(tree_clip), ti)
    m = {r["name"]: f"extra_obj_{k}" for k, r in enumerate(rows)}
    for x in ti["objects"]:
        if x.get("role") == "main":
            sd = x.get("side")
            m[x["name"]] = ["object_right", "object_left"] if sd == "both" else [f"object_{sd}"]
    return m


def env_from_tree(envo: dict, kin: Scene) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid transform (R, t) tree frame → env-local frame, from the frame-0 pose of a reference main object in
    reference.npz and in the kinematic tree. Fit error = max position gap of the main objects over all frames (~0 when
    reference.npz is the same trajectory moved rigidly). Raises without a main object."""
    if not envo:
        raise SystemExit(f"{MARK} no reference main object — cannot fix the tree → env frame")
    n = sorted(envo)[0]
    Re, pe, Rk, pk = envo[n]["R"][0], envo[n]["p"][0], kin.o[n]["R"][0], kin.o[n]["p"][0]
    R = Re @ Rk.T
    t = pe - R @ pk
    err = max(float(np.linalg.norm(v["p"] - (kin.o[m]["p"] @ R.T + t), axis=1).max()) for m, v in envo.items())
    return R, t, err


def roll_scene(ep: Path, objs: list[dict], emap: dict, kin: Scene, envo: dict, C: dict) -> tuple[Scene, dict]:
    """Rollout scene: objects with a scene entity take the recorded states; objects outside the physics scene
    (render-only or outside the slots) take the kinematic reference pose moved into the env frame (the policy cannot
    move them; their names go to from_reference). A main object without an entity raises."""
    z = np.load(ep)
    ents = {e["name"]: e for e in json.loads(str(z["entities"]))}
    meta = json.loads(str(z["meta"]))
    st = z["states"].astype(np.float64)
    o, miss, from_ref = {}, [], []
    Rt, tt, fit = env_from_tree(envo, kin)
    if fit > C["frame_fit_tol_m"]:
        raise SystemExit(
            f"{MARK} tree → env frame fit error {fit:.4f} m > {C['frame_fit_tol_m']} — reference.npz is not "
            "the kinematic reference moved rigidly"
        )
    for x in objs:
        c = emap.get(x["name"])
        c = [c] if isinstance(c, str) else (c or [])
        en = next((e for e in c if e in ents), None)
        if en is None:
            if x.get("role") == "main":
                raise SystemExit(
                    f"{MARK} entity {c} of main object {x['name']} is not in the rollout (have {sorted(ents)})"
                )
            miss.append((x["name"], c, x.get("tag")))
            k = kin.o[x["name"]]
            T = st.shape[0]
            kp = np.concatenate([k["p"], np.repeat(k["p"][-1:], max(0, T - len(k["p"])), 0)])[:T]
            kR = np.concatenate([k["R"], np.repeat(k["R"][-1:], max(0, T - len(k["R"])), 0)])[:T]
            o[x["name"]] = {"meta": x, "entity": None, "p": kp @ Rt.T + tt, "R": np.einsum("ij,tjk->tik", Rt, kR)}
            from_ref.append(x["name"])
            continue
        lo = ents[en]["lo"]
        o[x["name"]] = {"meta": x, "entity": en, "p": st[:, lo : lo + 3], "R": quat_R(st[:, lo + 3 : lo + 7])}
    # Recorder version 2 (loop.common.RECORDER): meta.held = the object entity name of force row hand_obj_force[:, k],
    # held_sides = the hand (side) of that row. Version 1 stored the side in held, so forces could not be matched.
    if meta.get("recorder") != 2:
        raise SystemExit(
            f"{MARK} recorder version {meta.get('recorder')} — only version 2 (meta.held = entity names) "
            "is judged; re-run the session"
        )
    f = z["hand_obj_force"]  # (T, held, links) — force magnitude of each hand link on that object
    held = meta["held"]
    if f.shape[1] != len(held) or any(h not in ents for h in held):
        raise SystemExit(f"{MARK} meta.held {held} does not match hand_obj_force {f.shape} / entities {sorted(ents)}")
    force = {}
    for x in objs:
        idx = [k for k, h in enumerate(held) if h == o.get(x["name"], {}).get("entity")]
        # shared object = the same entity on two rows (right, left): sum of both hands (grasp = either, release = both)
        if idx:
            force[x["name"]] = sum(f[:, k, :].sum(axis=-1) for k in idx)
    return Scene(o, force, f"rollout_{meta['mode']}"), {
        "meta": meta,
        "missing": miss,
        "from_reference": from_ref,
        "frame_fit_m": round(fit, 6),
    }


def env_ref(run: Path, objs: list[dict], kin: Scene) -> tuple[dict, dict]:
    """Main objects of reference.npz (env-local frame) → {name: dict(p, R)}, plus a relative-motion check against the
    kinematic npz (oracle)."""
    z = np.load(run / "reference.npz")
    out, chk = {}, {}
    for x in objs:
        if x.get("role") != "main" or x.get("side") not in ("right", "left", "both"):
            continue
        sd = (
            "right" if x["side"] == "both" else x["side"]
        )  # shared object = the right reference (same order as entity_map)
        p = z[f"{sd}_obj_pos"].astype(np.float64)
        R = quat_R(z[f"{sd}_obj_quat"].astype(np.float64))
        k = kin.o[x["name"]]
        if len(p) != len(k["p"]):
            raise SystemExit(f"{MARK} reference.npz {x['side']} has {len(p)} frames != kinematic {len(k['p'])}")
        dz = float(np.abs((p[:, 2] - p[0, 2]) - (k["p"][:, 2] - k["p"][0, 2])).max())
        dd = float(np.abs(np.linalg.norm(p - p[0], axis=1) - np.linalg.norm(k["p"] - k["p"][0], axis=1)).max())
        out[x["name"]] = {"meta": x, "p": p, "R": R}
        chk[x["name"]] = {"rise_maxdiff_m": round(dz, 5), "disp_maxdiff_m": round(dd, 5)}
    return out, chk


_PTS: dict = {}


def mesh_pts(ds: str, x: dict, n: int) -> np.ndarray:
    """n surface points (body frame, seed 0) of the object's visual mesh, cached."""
    import trimesh

    k = (ds, x["mesh_dir"], n)
    if k not in _PTS:
        P = _paths()
        f = P.main_root() / P.data_layout.object_path(ds, x["mesh_dir"]) / "visual.obj"
        if not f.is_file():
            raise SystemExit(f"{MARK} object mesh missing {f}")
        m = trimesh.load(f, force="mesh")
        _PTS[k] = trimesh.sample.sample_surface(m, n, seed=0)[0]
    return _PTS[k]


def map_home(path: str, home_map: str) -> Path:
    """Map an absolute job path (under the HOME of the machine that built the job) to this machine. home_map =
    '<build-machine HOME>=<its copy on this machine>' (e.g. <build HOME>=<this HOME>). A different prefix raises."""
    if "=" not in home_map:
        raise SystemExit(f"{MARK} --home-map {home_map!r} — expected '<build HOME>=<local copy>'")
    a, b = home_map.split("=", 1)
    rel = Path(path).relative_to(a) if Path(path).is_relative_to(a) else None
    if rel is None:
        raise SystemExit(f"{MARK} job path {path} is outside the build HOME {a}")
    return Path(b) / rel


def tree_clip_of(job: dict, home_map: str) -> Path:
    """job.seq_dir (build-machine path: the source tree, or the prepared shadow of a placement variant) / job.clip,
    on this machine."""
    tc = map_home(job["seq_dir"], home_map) / job["clip"]
    if not (tc / "task_info.json").is_file():
        raise SystemExit(f"{MARK} tree {tc}/task_info.json missing")
    return tc


def data_object_force(row: dict) -> tuple[dict, float, str]:
    """Hand–object force of the data episode: per entity [T] = sum over links of |F| (the same quantity as the rollout
    hand_obj_force), the force rate hz and the file. force_object.<entity>.force_n is [T, link, 3] (xyz world), so
    summing components directly would cancel; the magnitude is taken per link (record.store.force.link_mag).
    A missing file or rate raises."""
    from dexengine.record.rollout import recorder as REC  # noqa: PLC0415
    from dexengine.record.store import force as CFO  # noqa: PLC0415

    f = (row.get("files") or {}).get("forces")
    if not f or not os.path.isfile(f):
        raise SystemExit(f"{MARK} data force record missing {f} — cannot measure Put release 'auto'")
    info = json.loads((Path(row["root_dir"]) / "meta" / "info.json").read_text())
    hz = float(info["collect"]["force_rate_hz"])
    forces, _keys = CFO.load_forces(np.load(f, allow_pickle=True))
    out = {}
    for stem, (fn, names) in forces.items():
        kind, _, ent = stem.partition(".")
        if kind == REC.KIND_OBJECT and ent:
            out[ent] = CFO.link_mag({stem: (fn, names)})[1].sum(axis=1)
    if not out:
        raise SystemExit(f"{MARK} {f} has no {REC.KIND_OBJECT}.<entity>.force_n")
    return out, hz, f
