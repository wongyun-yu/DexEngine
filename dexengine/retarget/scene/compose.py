"""One clip, one rig: load inputs, assemble the scene spec, write `scene.xml`, `scene_eq.xml` and `task_info.json`."""

from __future__ import annotations

import json
import os
import re
import warnings

import mujoco
import numpy as np

from dexengine.retarget import mjutil
from dexengine.retarget.ik.prior import emb_sides
from dexengine.retarget.scene import build as B
from dexengine.retarget.scene.objects import extra_objects, primary_object

CONTACT_KEYS = ("contact_right", "contact_left", "contact_pos_right", "contact_pos_left")
EQ_SOLREF = [0.02, 1.0]
EQ_SOLIMP = [0.0, 1.0, 100.0, 0.5, 2.0]
RIG_FILE = {"bimanual": "bimanual.xml", "right": "right.xml", "left": "left.xml"}


def _contact_positions(keypoints_file: str, sides: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Per side, the (5, 3) mean contact points of the five fingertips in the object frame (zeros without keys)."""
    out = {s: np.zeros((5, 3)) for s in ("right", "left")}
    with np.load(keypoints_file) as z:
        missing = [k for k in CONTACT_KEYS if k not in z.files]
        if missing:
            warnings.warn(
                f"[SCENE] {keypoints_file}: no contact keys {missing}; object track sites at the origin", stacklevel=2
            )
            return out
        for s in sides:
            out[s] = np.asarray(z[f"contact_pos_{s}"], dtype=float)[:5]
    return out


def _rebase_rig_files(spec: mujoco.MjSpec, rig_dir: str, assets_dir: str) -> None:
    """Rewrite rig mesh/texture files relative to the dataset assets folder (the scene's meshdir/texturedir)."""
    for items, subdir in ((spec.meshes, spec.meshdir), (spec.textures, spec.texturedir)):
        for item in items:
            if not item.file:
                continue
            path = item.file if os.path.isabs(item.file) else os.path.join(rig_dir, subdir or "", item.file)
            item.file = os.path.relpath(os.path.abspath(path), assets_dir)


def fix_materials(xml: str, bindings: dict[str, str]) -> str:
    """Write `texture="<tex>"` and `texrepeat="1 1"` into each listed `<material>` element of the serialised XML.

    MuJoCo 3.4.1 writes these materials without their texture binding; the scene contract carries it.
    """
    for mat, tex in bindings.items():
        pat = re.compile(r'<material\b[^>]*\bname="' + re.escape(mat) + r'"[^>]*?(/?)>')
        m = pat.search(xml)
        if m is None:
            raise RuntimeError(f"[SCENE] material {mat!r} missing from the serialised scene")
        tag = m.group(0)
        tag = re.sub(r'\s(texture|texrepeat)="[^"]*"', "", tag)
        end = "/>" if tag.endswith("/>") else ">"
        tag = tag[: -len(end)].rstrip() + f' texture="{tex}" texrepeat="1 1"' + end
        xml = xml[: m.start()] + tag + xml[m.end() :]
    return xml


def _add_track_equalities(spec: mujoco.MjSpec, model: mujoco.MjModel) -> int:
    """Connect equalities `track*` → `ref*` for every compiled site that has a `ref` partner (site order)."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "" for i in range(model.nsite)]
    present = set(names)
    n = 0
    for name in names:
        if "track" not in name:
            continue
        partner = name.replace("track", "ref")
        if partner not in present:
            continue
        spec.add_equality(
            name=f"{name}_equality_constraint",
            type=mujoco.mjtEq.mjEQ_CONNECT,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=name,
            name2=partner,
            data=np.zeros(11),
            solref=EQ_SOLREF,
            solimp=EQ_SOLIMP,
        )
        n += 1
    return n


def _check_bodies(model: mujoco.MjModel, task_info: dict, sides: tuple[str, ...]) -> None:
    expected = [f"{s}_object" for s in sides if task_info.get(f"{s}_object_mesh_dir")]
    if not expected:
        raise RuntimeError("[SCENE] task_info declares no object mesh for the embodiment's sides")
    missing = [b for b in expected if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) < 0]
    if missing:
        raise RuntimeError(f"[SCENE] scene lacks the object bodies task_info declares: {missing}")


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def build_scene(
    dataset_dir: str,
    dataset_name: str,
    task: str,
    data_id: int,
    robot_type: str,
    embodiment_type: str,
    use_visual_mesh_as_collision: bool = False,
) -> str:
    """Build and write the scene files of one clip for one rig; returns the scene folder."""
    sides = emb_sides(embodiment_type)
    dataset_dir = os.path.abspath(dataset_dir)
    ds_root = os.path.join(dataset_dir, "processed", dataset_name)
    assets_dir = os.path.join(ds_root, "assets")
    rig_dir = os.path.join(assets_dir, "robots", robot_type)
    rig_file = os.path.join(rig_dir, RIG_FILE[embodiment_type])
    if not os.path.isfile(rig_file):
        raise FileNotFoundError(f"[SCENE] rig file missing: {rig_file}")
    mano_dir = mjutil.processed_data_dir(dataset_dir, dataset_name, "mano", embodiment_type, task, data_id)
    task_info_file = os.path.join(os.path.dirname(mano_dir), "task_info.json")
    with open(task_info_file) as f:
        task_info = json.load(f)
    out_dir = os.path.dirname(
        mjutil.processed_data_dir(dataset_dir, dataset_name, robot_type, embodiment_type, task, data_id)
    )
    os.makedirs(os.path.join(out_dir, str(data_id)), exist_ok=True)
    contact_pos = _contact_positions(os.path.join(mano_dir, "trajectory_keypoints.npz"), sides)

    spec = mujoco.MjSpec.from_file(rig_file)
    _rebase_rig_files(spec, rig_dir, assets_dir)
    rel_assets = os.path.relpath(assets_dir, out_dir) + "/"
    spec.meshdir = rel_assets
    spec.texturedir = rel_assets
    spec.modelfiledir = out_dir + os.sep
    B.add_world(spec)

    assets = B.SceneAssets(spec, assets_dir)
    primaries = {s: primary_object(dataset_dir, task_info, s, use_visual_mesh_as_collision) for s in sides}
    extras = extra_objects(dataset_dir, task_info)
    for s in sides:
        assets.primary_visual(s, primaries[s])
    for i, obj in extras:
        assets.extra(i, obj)
    for s in sides:
        assets.primary_parts(s, primaries[s])

    extra_geoms = [B.add_extra_body(spec, i, obj, assets) for i, obj in extras]
    placed: dict[str, B.ObjectGeoms | None] = {"right": None, "left": None}
    for s in sides:
        if primaries[s].has_body:
            placed[s] = B.add_primary_body(spec, s, primaries[s], assets, contact_pos[s])
    B.add_pairs(spec, extra_geoms, placed["right"], placed["left"], bimanual=embodiment_type == "bimanual")

    # compile before the first serialisation: the writer then emits each <mesh> with name and file only (an
    # uncompiled spec also writes a `content_type`), so scene.xml and scene_eq.xml share one format
    model = spec.compile()
    scene_xml = fix_materials(spec.to_xml(), assets.material_textures)
    _check_bodies(model, task_info, sides)
    n_eq = _add_track_equalities(spec, model)
    eq_xml = fix_materials(spec.to_xml(), assets.material_textures)

    _write_atomic(os.path.join(out_dir, "scene.xml"), scene_xml)
    _write_atomic(os.path.join(out_dir, "scene_eq.xml"), eq_xml)
    info = dict(task_info)
    info["robot_type"] = robot_type
    _write_atomic(os.path.join(out_dir, "task_info.json"), json.dumps(info, indent=2))
    print(
        f"[SCENE] {dataset_name}/{task} {robot_type} {embodiment_type}: bodies {model.nbody} geoms {model.ngeom} "
        f"pairs {model.npair} mocap {model.nmocap} track equalities {n_eq} -> {out_dir}"
    )
    return out_dir
