"""One Tab. 2 TACO clip (our SPIDER-form clip folder, the same input SPIDER and EgoEngine got) -> DexMachina's input
files
(DexMachina extended to N rigid objects by the port patches under patches/).

Reused, not re-derived: the SharpA kinematic retarget (0/trajectory_kinematic.npz qpos: 28 right + 28 left hand dofs,
right/left object free poses), its fingertip contacts (contact, contact_pos), the TACO object meshes + convex parts
and the object mass/inertia of the SPIDER scene (scene.xml, MuJoCo density 1000 over the convex parts).

Written (DexMachina asset tree, clip key K, subject s01, use 01, hand sharpa_wave_hand):
  assets/taco_e123/objects/<id>/<id>.urdf (+ visual.obj, convex/*.obj copies)   one rigid single-link object
  assets/arctic/processed/s01/K_use_01.npy       their demo format + `objects` {states, specs, table_height}
  assets/retargeter_results/sharpa_wave_hand/s01/K_use_01_vector.npy   their kinematic-retarget format (hand_qpos ...)
  assets/sharpa_wave_hand/retarget_config.yaml   (urdf paths for map_contacts; dex_retargeting itself is not run)
  <out>/K_adapter.json                            provenance, conversion checks, frame/time map

Conventions:
  * time: DexMachina steps one demo frame per sim step (dt 1/60 s). The SPIDER-form reference is 50 Hz, so it is
    resampled to 60 Hz (frame k <-> original-motion time tau = k/60 s): positions and joint values linear, rotations
    slerp, contact flags/positions from the nearest 50 Hz frame. T60 = floor(t_end*60)+1.
  * wrist: SPIDER base = 3 slides + 3 hinges on the flange body, world rotation R = Rx(a) Ry(b) Rz(c) R0.
    add_wrist_dof.py chain = tx ty tz (root frame) then roll(z) pitch(x) yaw(-y): R = Rz(r) Rx(p) Ry(-y).
    So (r, p, -y) = intrinsic ZXY Euler of R; unwrapped over time.
  * objects in env order [right_object, left_object]; contact part id = n_parts - link index (their ARCTIC rule:
    part 2 = first link 'bottom', part 1 = 'top'), i.e. right_object -> 2, left_object -> 1.
  * table: TACO frame z 0 = desk (task_info.support.desk_z) = Genesis ground plane. Their fell-off rule (object below
    TABLE_HEIGHT 0.6 m, i.e. 0.35 m under the ARCTIC support top 0.95 m) becomes desk_z - 0.35.
Runs with this repository on PYTHONPATH (prep_clips.sh writes both calls):
usage: <python with mujoco>=3.3> -s -B taco_to_dexmachina.py --dump-inertia <clip>/scene.xml <out>/inertia.json
       python taco_to_dexmachina.py --inertia-json <out>/inertia.json --clip-dir <spider clip dir> --key e123c000 --dm
       <code/dexmachina/dexmachina> --out <dir>
       [--fk-check]   (Genesis FK of the converted dofs vs SPIDER fingertip/palm positions; needs a GPU)
variant (variant.py, e.g. m0p2 = every object at our fixed mass; none = the original rows):
       <spider python> -s -B taco_to_dexmachina.py --dump-inertia <clip>/scene.xml <out>/inertia_m0p2.json
       --mass-variant m0p2
         (mass = the owner's value via common/prep_mass_0p2.py owner_mass · inertia = scene x M/m_scene · ipos iquat
         unchanged ·
          the scene values kept as mass_scene / inertia_scene)
       then the adapter as above with --inertia-json <that file> --key <the variant's key>: the inertia record names the
       variant,
         so its objects go to the variant's own asset root (never over the original URDFs) and the key must be the
         variant's
"""

import argparse
import ast
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from experiments.baselines.dexmachina.fk_check import fk_check
from experiments.baselines.dexmachina.scene_inertia import dump_inertia

R0_WXYZ = None  # SPIDER base body quat at zero pose, read from the MJCF


def lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def wrist_to_dm(q6, r0_wxyz):
    """SPIDER (x,y,z,a,b,c) -> DexMachina (tx,ty,tz,roll,pitch,yaw) per frame (T,6)."""
    R0 = Rot.from_quat(np.asarray(r0_wxyz)[[1, 2, 3, 0]])
    Rs = Rot.from_euler("XYZ", q6[:, 3:6]) * R0  # intrinsic X then Y then Z == Rx(a) Ry(b) Rz(c)
    zxy = Rs.as_euler("ZXY")  # R = Rz(r) Rx(p) Ry(q)
    zxy = np.unwrap(zxy, axis=0)
    out = np.concatenate([q6[:, :3], zxy[:, :1], zxy[:, 1:2], -zxy[:, 2:3]], axis=1)
    return out, Rs


def resample(t_src, t_dst, x, kind="lin"):
    if kind == "lin":
        return np.stack([np.interp(t_dst, t_src, x[:, i]) for i in range(x.shape[1])], axis=1)
    if kind == "quat":  # wxyz
        r = Slerp(t_src, Rot.from_quat(x[:, [1, 2, 3, 0]]))(t_dst).as_quat()
        return r[:, [3, 0, 1, 2]]
    if kind == "near":
        idx = np.clip(np.round(t_dst / (t_src[1] - t_src[0])).astype(int), 0, len(t_src) - 1)
        return x[idx]
    raise ValueError(kind)


def _put(path, data, shared):
    """shared (variant assets): one object id is written by every clip that uses it, possibly at once (prep in parallel)
    —
    write to a temp name and rename (atomic); a file already there must hold the same bytes (one object = one URDF)."""
    assert shared
    if os.path.exists(path):
        if Path(path).read_bytes() != data:
            raise SystemExit(f"[ADAPTER] FAIL {path} exists with other content (one object id, two inertials?)")
        return
    tmp = f"{path}.tmp{os.getpid()}"
    Path(tmp).write_bytes(data)
    os.replace(tmp, path)


def write_object(dm, oid, inert_rec, body, spider_assets, out_rel):
    """URDF for one TACO object = SPIDER scene body: its convex parts as collision, visual.obj, MuJoCo's
    mass/inertia.
    """
    r = inert_rec[body]
    mass, ipos, iquat, inert = r["mass"], np.array(r["ipos"]), np.array(r["iquat"]), np.array(r["inertia"])
    rpy = Rot.from_quat(np.asarray(iquat)[[1, 2, 3, 0]]).as_euler("xyz")
    src = os.path.join(spider_assets, "objects", oid)
    dst = os.path.join(dm, "assets", out_rel)
    os.makedirs(os.path.join(dst, "convex"), exist_ok=True)
    mv = inert_rec.get("mass_variant")
    parts = sorted(f for f in os.listdir(os.path.join(src, "convex")) if f.endswith(".obj"))
    if mv:
        for f in ["visual.obj"] + [os.path.join("convex", p) for p in parts]:
            _put(os.path.join(dst, f), Path(os.path.join(src, f)).read_bytes(), True)
    else:
        shutil.copyfile(os.path.join(src, "visual.obj"), os.path.join(dst, "visual.obj"))
        for f in parts:
            shutil.copyfile(os.path.join(src, "convex", f), os.path.join(dst, "convex", f))
    # the parts the SPIDER scene simulates must be exactly these files
    n_scene = r["n_collision_meshes"]
    assert n_scene == len(parts), f"{oid}: scene has {n_scene} collision meshes, folder {len(parts)}"
    coll = "\n".join(
        '    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh '
        f'filename="convex/{f}"/></geometry></collision>'
        for f in parts
    )
    head = (
        (
            f"meshes = SPIDER clip assets (copied), mass/inertia = SPIDER scene body '{body}'\n"
            "     (MuJoCo, density 1000 over the convex parts)"
        )
        if not mv
        else (
            f"meshes = SPIDER clip assets (copied), variant {mv['variant']}: mass = {mv['mass_kg']:.9g} kg "
            f"({mv['owner_name']}),\n"
            "     inertia = SPIDER scene (MuJoCo, density 1000 over the convex parts; "
            f"{r['mass_scene']:.9g} kg) x M/m_scene, ipos iquat unchanged"
        )
    )
    urdf = f'''<?xml version="1.0"?>
<!-- TACO object {oid} for DexMachina (E1-E3): {head} -->
<robot name="taco_{oid}">
  <link name="body">
    <inertial>
      <origin xyz="{ipos[0]:.9g} {ipos[1]:.9g} {ipos[2]:.9g}" rpy="{rpy[0]:.9g} {rpy[1]:.9g} {rpy[2]:.9g}"/>
      <mass value="{mass:.9g}"/>
      <inertia ixx="{inert[0]:.9g}" ixy="0" ixz="0" iyy="{inert[1]:.9g}" iyz="0" izz="{inert[2]:.9g}"/>
    </inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="visual.obj"/></geometry></visual>
{coll}
  </link>
</robot>
'''
    if mv:
        _put(os.path.join(dst, f"{oid}.urdf"), urdf.encode(), True)
    else:
        Path(os.path.join(dst, f"{oid}.urdf")).write_text(urdf)
    return {"urdf": f"{out_rel}/{oid}.urdf", "mesh": f"{out_rel}/visual.obj", "mass": mass, "n_convex": len(parts)}


def _hand_qpos(a, inert_rec, q, t50, t60):
    """Per hand: the clip's 28 qpos columns → the DexMachina 6-dof wrist + finger dofs in Genesis actuated order,
    resampled 50 → 60 Hz (and the wrist rotation round-trip check)."""
    dm_q, names_dm, wrist_check = {}, {}, {}
    sides = {"right": (0, "R_"), "left": (28, "L_")}
    for s, (off, pre) in sides.items():
        jn, adr = inert_rec["joint_names"], inert_rec["jnt_qposadr"]
        mj_names = [n for n, ad in zip(jn, adr, strict=False) if off <= ad < off + 28]
        assert [adr[jn.index(n)] for n in mj_names] == list(range(off, off + 28)), f"{s}: scene qpos order"
        r0 = inert_rec["base_quat"][s]
        assert len(mj_names) == 28 and mj_names[:6] == [f"{s}_pos_{c}" for c in "xyz"] + [f"{s}_rot_{c}" for c in "xyz"]
        qs = q[:, off : off + 28]
        w6, Rs = wrist_to_dm(qs[:, :6], r0)
        # check: rebuild R from the DexMachina chain
        Rb = Rot.from_euler("ZXY", np.stack([w6[:, 3], w6[:, 4], -w6[:, 5]], 1))
        wrist_check[s] = float(np.max((Rb.inv() * Rs).magnitude()))
        vals = {
            f"{pre}forearm_{n}_link_joint": w6[:, i] for i, n in enumerate(["tx", "ty", "tz", "roll", "pitch", "yaw"])
        }
        for i, n in enumerate(mj_names[6:]):
            vals[n] = qs[:, 6 + i]
        # Genesis actuated order of the 6-dof URDF (BFS) -- read from the URDF without Genesis: same as printed by
        # sharpa_collision_groups.py; checked again in --fk-check against the entity
        order = json.loads(Path(os.path.join(a.out, f"actuated_{s}.json")).read_text())
        assert sorted(order) == sorted(vals), f"{s}: dof names differ {set(order) ^ set(vals)}"
        q50 = np.stack([vals[n] for n in order], 1)
        dm_q[s] = resample(t50, t60, q50, "lin")
        names_dm[s] = order
    return dm_q, names_dm, wrist_check, sides


def _objects(a, ti, q, layout, t50, t60, T60, inert_rec, spider_assets, obj_root):
    """Object states resampled to 60 Hz and the object assets (URDF + mesh) with their specs."""
    objs = {}
    for name, (st, _w) in (("right_object", layout["right_object"]), ("left_object", layout["left_object"])):
        objs[name] = {
            "obj_trans": resample(t50, t60, q[:, st : st + 3], "lin").astype(np.float32),
            "obj_quat": resample(t50, t60, q[:, st + 3 : st + 7], "quat").astype(np.float32),
            "obj_arti": np.zeros((T60, 0), np.float32),
        }
    # objects: assets
    specs, oinfo = {}, {}
    for name in objs:
        oid = os.path.basename(ti[f"{name.split('_')[0]}_object_mesh_dir"])
        info = write_object(a.dm, oid, inert_rec, name, spider_assets, f"{obj_root}/{oid}")
        oinfo[name] = dict(info, obj_id=oid)
        specs[name] = {
            "name": name,
            "urdf_path": info["urdf"],
            "part_mesh_fnames": {"body": info["mesh"]},
            "base_init_pos": objs[name]["obj_trans"][0].tolist(),
            "base_init_quat": objs[name]["obj_quat"][0].tolist(),
        }
    return objs, specs, oinfo


def _contacts(ti, objs, contact, cpos, t50, t60, T50, T60):
    """Table height and the raw human contacts in their ARCTIC world_coord form, from the fingertip contacts."""
    sup = lit(ti["support"])
    desk_z = float(sup.get("desk_z", 0.0))
    table_height = desk_z - (0.95 - 0.6)
    # raw human contacts (their ARCTIC field contacts.<side>: (T, 50, [x y z part_id]) world) from the fingertip
    # contacts
    names_obj = list(objs)
    part_id = {n: len(names_obj) - i for i, n in enumerate(names_obj)}
    c60 = resample(t50, t60, contact.astype(np.float32), "near") > 0.5
    p60 = resample(t50, t60, cpos.reshape(T50, -1), "near").reshape(T60, -1, 3)
    wc = {"contact_threshold": np.array(0.0)}
    for s, slot0, own in (("right", 0, "right_object"), ("left", 5, "left_object")):
        raw = np.zeros((T60, 50, 4), np.float32)
        val = np.zeros((T60, 50), bool)
        raw[:, :5, :3] = p60[:, slot0 : slot0 + 5]
        raw[:, :5, 3] = np.where(c60[:, slot0 : slot0 + 5], part_id[own], 0)
        val[:, :5] = c60[:, slot0 : slot0 + 5]
        wc[f"contacts.{s}"], wc[f"valid_contacts.{s}"] = raw, val
        wc[f"joints.{s}"] = np.zeros(
            (T60, 21, 3), np.float32
        )  # MANO joints: not in our clip; unused (--load_retargeter)
        wc[f"contact_links_{s}"] = np.zeros(
            (T60, 16, 4), np.float32
        )  # MANO-link contacts: replaced by --use_retarget_contact
    return table_height, names_obj, part_id, c60, wc


def _write_retarget_config(a, names_dm, mv):
    """retarget_config.yaml (hand-level, shared with the original rows: a variant only checks it)."""
    # retarget_config.yaml (their template; only urdf_path is read on our path -- map_contacts.py)
    yml = [
        "# SharpA Wave for DexMachina (E1-E3). Kinematic retargeting is NOT done by dex_retargeting here: the SharpA",
        "# kinematic retarget of the SPIDER-form clip is reused (taco_to_dexmachina.py -> retargeter_results).",
        "low_pass_alpha: 1.0",
        "scaling_factor: 1.0",
    ]
    for s in ("left", "right"):
        yml += [
            f"{s}:",
            f"  urdf_path: {s}_sharpa_wave_6dof.urdf",
            f"  target_origin_link: {s}_hand_C_MC",
            "  human_origin_idx: 0",
            "  target_finger_links: "
            f"[{', '.join(f'{s}_{f}_fingertip' for f in ('thumb', 'index', 'middle', 'ring', 'pinky'))}]",
            "  human_finger_idxs: [16, 17, 18, 19, 20]",
            f"  target_joint_names: [{', '.join(names_dm[s])}]",
            f"  wrist_body: {s}_hand_C_MC",
        ]
    ycfg = os.path.join(a.dm, "assets", "sharpa_wave_hand", "retarget_config.yaml")
    if not mv:
        Path(ycfg).write_text("\n".join(yml) + "\n")
    elif (
        Path(ycfg).read_text() != "\n".join(yml) + "\n"
    ):  # hand-level, shared with the original rows: a variant never rewrites it
        raise SystemExit(
            f"[ADAPTER] FAIL {ycfg} differs from this clip's (a variant does not rewrite the shared hand config)"
        )


def _parse_args():
    """Command line of the adapter."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-inertia", nargs=2, metavar=("SCENE_XML", "OUT_JSON"))
    ap.add_argument("--clip-dir")
    ap.add_argument("--key")
    ap.add_argument("--dm")
    ap.add_argument("--out")
    ap.add_argument("--inertia-json", help="output of --dump-inertia for this clip's scene.xml")
    ap.add_argument("--fk-check", action="store_true")
    ap.add_argument("--mass-variant", help="with --dump-inertia: the variant's object mass (variant.py)")
    a = ap.parse_args()
    return a


def _write_demo(objs, names_obj, wc, T60, specs, table_height, ti, a, T50):
    """The DexMachina demo file (ARCTIC processed layout) of the clip."""
    first = objs[names_obj[0]]
    demo = {
        "world_coord": wc,
        "params": {
            "obj_trans": first["obj_trans"],
            "obj_quat": first["obj_quat"],
            "obj_arti": np.zeros(T60, np.float32),
        },
        "objects": {"states": objs, "specs": specs, "table_height": table_height},
        "source": {
            "clip": ti["clip_id"],
            "clip_dir": os.path.realpath(a.clip_dir),
            "resample": "50 Hz -> 60 Hz",
            "T50": T50,
            "T60": T60,
        },
    }
    pdir = os.path.join(a.dm, "assets", "arctic", "processed", "s01")
    os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, f"{a.key}_use_01.npy"), demo, allow_pickle=True)


def _write_retarget_results(sides, names_dm, dm_q, T60, a):
    """The per-side retargeter result file of the clip."""
    rr = {}
    for s in sides:
        widx = [names_dm[s].index(n) for n in names_dm[s] if "forearm" in n]
        rr[s] = {
            "hand_qpos": dm_q[s].astype(np.float32),
            "wrist_qpos": dm_q[s][:, widx].astype(np.float32),
            "wrist_idxs": np.tile(np.array(widx), (T60, 1)),
            "actuated_dof_names": names_dm[s],
            "actuated_dof_idxs": None,
        }
    rdir = os.path.join(a.dm, "assets", "retargeter_results", "sharpa_wave_hand", "s01")
    os.makedirs(rdir, exist_ok=True)
    np.save(os.path.join(rdir, f"{a.key}_use_01_vector.npy"), rr, allow_pickle=True)


def _adapter_record(
    a,
    ti,
    T50,
    T60,
    dt50,
    oinfo,
    part_id,
    table_height,
    wrist_check,
    c60,
    dm_q,
    names_dm,
    sides,
    mv,
    obj_root,
    inert_rec,
):
    """The adapter record: the clip facts, the checks and the variant objects."""
    rec = {
        "key": a.key,
        "clip": ti["clip_id"],
        "T50": T50,
        "T60": T60,
        "dt50": dt50,
        "tau_of_frame": "k/60 s",
        "clip_string": f"{a.key}-0-{T60}-s01-u01",
        "objects": oinfo,
        "part_id": part_id,
        "table_height": table_height,
        "wrist_rot_roundtrip_max_rad": wrist_check,
        "contact_frames": {s: int(v.any(1).sum()) for s, v in (("right", c60[:, :5]), ("left", c60[:, 5:]))},
        "pitch_abs_max_deg": {
            s: round(float(np.degrees(np.abs(dm_q[s][:, names_dm[s].index(f"{p}forearm_pitch_link_joint")]).max())), 2)
            for s, (_o, p) in sides.items()
        },
    }
    if mv:
        rec["mass_variant"] = dict(
            mv,
            obj_root=obj_root,
            objects={
                n: {"obj_id": o["obj_id"], "mass": inert_rec[n]["mass"], "mass_scene": inert_rec[n]["mass_scene"]}
                for n, o in oinfo.items()
            },
        )
    return rec


def _say_adapter(rec):
    """One ADAPTER OK line with the record key facts."""
    print(
        "[ADAPTER] OK "
        + json.dumps(
            {
                k: rec[k]
                for k in (
                    "key",
                    "clip",
                    "T60",
                    "clip_string",
                    "wrist_rot_roundtrip_max_rad",
                    "pitch_abs_max_deg",
                    "contact_frames",
                )
            }
        )
    )


def main():
    a = _parse_args()
    if a.dump_inertia:
        return dump_inertia(*a.dump_inertia, variant=a.mass_variant)
    assert not a.mass_variant, (
        "--mass-variant goes with --dump-inertia (the adapter reads the variant from the inertia record)"
    )
    inert_rec = json.loads(Path(a.inertia_json).read_text())
    assert a.key.isalnum(), "their clip strings split on '-' and file names on '_' -> key must be alphanumeric"
    mv = inert_rec.get("mass_variant")
    obj_root = "taco_e123/objects"
    if mv:  # the variant's own key space and asset root (variant.py): never the original rows' files
        from experiments.baselines.dexmachina import variant as dmv

        V = dmv.get(mv["variant"])
        assert a.key.startswith(V["key_prefix"]) and a.key[len(V["key_prefix"]) :].isdigit(), (a.key, V["key_prefix"])
        obj_root = V["obj_root"]
    ti = json.loads(Path(os.path.join(a.clip_dir, "task_info.json")).read_text())
    z = np.load(os.path.join(a.clip_dir, "0", "trajectory_kinematic.npz"))
    bp = dict(np.load(os.path.join(a.clip_dir, "0", "body_positions.npz"), allow_pickle=True))
    q, contact, cpos = z["qpos"], z["contact"].astype(bool), z["contact_pos"]
    layout = {d["name"]: (d["start"], d["width"]) for d in lit(ti["qpos_layout"])}
    assert ti["embodiment_type"] == "bimanual" and ti["robot_type"] == "sharpa_wave" and q.shape[1] == ti["nq"] == 70
    assert layout["right_object"] == (56, 7) and layout["left_object"] == (63, 7), layout
    dt50 = float(ti["ref_dt"])
    T50 = q.shape[0]
    t50 = np.arange(T50) * dt50
    T60 = int(np.floor(t50[-1] * 60.0 + 1e-9)) + 1
    t60 = np.arange(T60) / 60.0
    spider_assets = os.path.realpath(os.path.join(a.clip_dir, "..", "..", "..", "assets"))
    dm_q, names_dm, wrist_check, sides = _hand_qpos(a, inert_rec, q, t50, t60)
    objs, specs, oinfo = _objects(a, ti, q, layout, t50, t60, T60, inert_rec, spider_assets, obj_root)
    table_height, names_obj, part_id, c60, wc = _contacts(ti, objs, contact, cpos, t50, t60, T50, T60)
    _write_demo(objs, names_obj, wc, T60, specs, table_height, ti, a, T50)
    # kinematic retarget in their retargeter_results format
    _write_retarget_results(sides, names_dm, dm_q, T60, a)
    _write_retarget_config(a, names_dm, mv)
    rec = _adapter_record(
        a,
        ti,
        T50,
        T60,
        dt50,
        oinfo,
        part_id,
        table_height,
        wrist_check,
        c60,
        dm_q,
        names_dm,
        sides,
        mv,
        obj_root,
        inert_rec,
    )
    if a.fk_check:
        rec["fk_check"] = fk_check(a.dm, dm_q, names_dm, bp, t50, t60)
    with open(os.path.join(a.out, f"{a.key}_adapter.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    _say_adapter(rec)
    return None


if __name__ == "__main__":
    main()
