"""The ``--dump-inertia`` step of ``taco_to_dexmachina.py``: mass, inertia and collision-mesh count of the two objects
of a SPIDER scene (read with MuJoCo), and a variant's fixed object mass applied to them (variant.py)."""

import json


def mass_variant(res, variant):
    """the variant's object mass (variant.py): mass = the owner's value, read by name through common/prep_mass_0p2.py
    owner_mass() (whose LABEL must be this variant's name); principal moments x M/m_scene (same shape, uniform density);
    ipos / iquat unchanged; the scene's values kept beside them."""
    from experiments.baselines.common import prep_mass_0p2 as mod

    if variant != mod.LABEL:
        raise SystemExit(f"[ADAPTER] FAIL --mass-variant {variant}: the mass owner reader is labelled {mod.LABEL}")
    M = mod.owner_mass()
    for body in ("right_object", "left_object"):
        r = res[body]
        m0, i0 = r["mass"], list(r["inertia"])
        assert m0 > 0 and all(x > 0 for x in i0), (body, m0, i0)
        r.update(mass_scene=m0, inertia_scene=i0, mass=M, inertia=[x * (M / m0) for x in i0])
    res["mass_variant"] = {
        "variant": variant,
        "mass_kg": M,
        "owner": mod.OWNER,
        "owner_name": "DEFAULT_OBJECT_MASS_KG",
        "rule": "mass = M; inertia = inertia_scene x M/mass_scene (same shape, uniform density); ipos, iquat unchanged",
    }


def dump_inertia(scene_xml, out_json, variant=None):
    """read-only: mass/inertia/collision-mesh count of right_object/left_object in the SPIDER scene (needs a MuJoCo that
    parses it -- the dexmachina env pins mujoco 3.2.5 for Genesis, so this runs under another interpreter)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(scene_xml)
    res = {"mujoco": mujoco.__version__}
    for body in ("right_object", "left_object"):
        b = m.body(body).id
        n = sum(
            1
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == b
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and m.geom(g).name != f"{body}_visual"
        )
        res[body] = {
            "mass": float(m.body_mass[b]),
            "ipos": m.body_ipos[b].tolist(),
            "iquat": m.body_iquat[b].tolist(),
            "inertia": m.body_inertia[b].tolist(),
            "n_collision_meshes": int(n),
        }
    res["joint_names"] = [m.joint(j).name for j in range(m.njnt)]
    res["jnt_qposadr"] = [int(m.jnt_qposadr[j]) for j in range(m.njnt)]
    res["base_quat"] = {s: m.body_quat[m.body(f"{s}_hand_C_MC").id].tolist() for s in ("right", "left")}
    if variant:
        mass_variant(res, variant)
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[ADAPTER] inertia -> {out_json}")
