"""Object mass and rigid-body properties: the articulated-object link-mass split, collision / rigid props of a spawned
object and the per-object mass override (split from env_setup.objects)."""

from __future__ import annotations

#: Articulated link name contract: the two links robots/objects/usd_bake.py --articulated bakes.
#: Other names or counts fail.
ARTIC_LINK_NAMES = ("base_link", "moving_link")


def _artic_distribute_link_mass(prim_path_regex: str, key: str, total_kg: float, ctx_label: str) -> None:
    """Distribute the articulated total mass `total_kg` over the links (base_link · moving_link) by the **USD bake mass
    ratio**.

    Why here (after the spawn, before the physics view is created, at USD level): on the rigid path `spawn.mass_props`
    goes through `modify_mass_properties` (apply_nested), which writes the same value on **every link** with a MassAPI,
    so an articulation got twice the total (0.398 kg). A runtime `set_masses` is undone at the first reset, because
    `randomize_object_mass_at_reset` restores from `obj.data.default_mass` (= the USD value when the physics view was
    created). Written to the USD, default_mass becomes the distributed value, and mass DR scales every body by one
    factor, keeping ratios and the total.
    The same mechanism as the rigid path (mass only; PhysX derives the inertia from the shape).
    The spawn function returns after creating every env (MultiAssetSpawnerCfg uses Sdf.CopySpec copies), so this walks
    **every env prim**.
    Marker `[ARTIC][mass] <key>: base=… moving=… total=… kg (ratio from USD; envs=N distinct_ratios=K)`.
    """
    from pxr import UsdPhysics

    import isaaclab.sim as _su
    from isaaclab.sim.utils import find_matching_prims

    obj_prims = find_matching_prims(prim_path_regex)
    if not obj_prims:
        raise RuntimeError(f"[ARTIC][mass] {key}: no spawned object prim — {prim_path_regex}")
    distinct: dict[tuple, int] = {}
    n_written = 0
    for obj in obj_prims:
        links = {c.GetName(): c for c in obj.GetChildren() if c.HasAPI(UsdPhysics.MassAPI)}
        if tuple(sorted(links)) != tuple(sorted(ARTIC_LINK_NAMES)):
            raise RuntimeError(
                f"[ARTIC][mass] {key}: MassAPI links {sorted(links)} of {obj.GetPath()} ≠ "
                f"{ARTIC_LINK_NAMES} — objects.usd_bake --articulated contract broken"
            )
        usd_m = {n: float(UsdPhysics.MassAPI(links[n]).GetMassAttr().Get() or 0.0) for n in ARTIC_LINK_NAMES}
        if any(v <= 0.0 for v in usd_m.values()):
            raise RuntimeError(f"[ARTIC][mass] {key}: {obj.GetPath()} link USD mass ≤ 0 {usd_m}")
        tot = sum(usd_m.values())
        sig = tuple(round(usd_m[n], 6) for n in ARTIC_LINK_NAMES)
        distinct[sig] = distinct.get(sig, 0) + 1
        for n in ARTIC_LINK_NAMES:
            _su.modify_mass_properties(
                links[n].GetPath().pathString, _su.MassPropertiesCfg(mass=total_kg * usd_m[n] / tot)
            )
        # Output-level check: read the values just written back from the USD and check that they sum to the total mass
        back = sum(float(UsdPhysics.MassAPI(links[n]).GetMassAttr().Get()) for n in ARTIC_LINK_NAMES)
        if abs(back - total_kg) > 1e-6:
            raise RuntimeError(f"[ARTIC][mass] {key}: {obj.GetPath()} read-back sum {back:.6f} ≠ {total_kg}")
        n_written += 1
    if n_written != len(obj_prims):
        raise RuntimeError(f"[ARTIC][mass] {key}: written {n_written} ≠ prims {len(obj_prims)}")
    b0, m0 = next(iter(distinct))
    print(
        f"[ARTIC][mass] {key}: base={total_kg * b0 / (b0 + m0):.4f} moving={total_kg * m0 / (b0 + m0):.4f} "
        f"total={total_kg:.4f} kg (ratio from USD; envs={n_written} distinct_ratios={len(distinct)})",
        flush=True,
    )
    if len(distinct) > 1:
        for (b, mv), cnt in distinct.items():
            print(
                f"[ARTIC][mass] {key}:   usd base={b:.6f} moving={mv:.6f} → base={total_kg * b / (b + mv):.4f} "
                f"moving={total_kg * mv / (b + mv):.4f} (envs {cnt})",
                flush=True,
            )


def object_collision_props(_su):
    """Collision props of a dynamic scene object (the main objects): contact and rest offsets, torsional patch."""
    return _su.CollisionPropertiesCfg(
        contact_offset=0.005,
        rest_offset=0.0,
        torsional_patch_radius=0.005,
        min_torsional_patch_radius=0.001,
    )


def object_rigid_props(_su):
    """Rigid body props of a dynamic scene object (the main objects): dynamic · gravity by the ramp wrench
    (disable_gravity) · velocity caps · the shared depenetration velocity."""
    from dexengine.rl.tasks.cfg.constants import MAX_DEPENETRATION_VELOCITY as _mdv

    return _su.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        disable_gravity=True,
        max_linear_velocity=50.0,
        max_angular_velocity=100.0,
        max_depenetration_velocity=_mdv,
    )


def apply_object_mass(cfg, args_cli, ctx_label):
    """Task-5 single place for object mass at runtime (train + eval).

    Applies the single-source DEFAULT_OBJECT_MASS_KG (0.2 kg) to BOTH `object` and
    (bimanual) `object_left`, overridable PER SIDE by --object_mass / --object_mass_left.
    Unset -> 0.2 kg (the controlled default); passing a value is the legacy/USD-mass
    escape hatch. Logs the mass set for each object. Replaces the previously-scattered
    per-side override blocks so no object silently keeps a density-derived mass.

    A-4: articulated (ArticulationCfg) slots do not use `spawn.mass_props` (it attaches to every link, 2× in total); the
    spawn function is wrapped and distributes per link after the spawn (`_artic_distribute_link_mass`). The rigid path
    stays byte-identical.
    """
    # Single owner of the mass value (A-4); no copy, no fallback: unreadable fails.
    import isaaclab.sim as _su
    from isaaclab.assets import ArticulationCfg

    from dexengine.rl.tasks.cfg.residual import DEFAULT_OBJECT_MASS_KG

    _idle_key = f"object_{args_cli.idle_side}" if str(getattr(args_cli, "idle_side", "") or "") else None

    def _set(key, cli_val):
        if not hasattr(cfg.scene, key) or getattr(cfg.scene, key) is None:
            return
        if key == _idle_key:  # D16 idle side: apply_idle_side removes this placeholder prim — no mass, no log line
            return
        m = float(cli_val) if cli_val is not None else float(DEFAULT_OBJECT_MASS_KG)
        src = "CLI override" if cli_val is not None else f"single-source default {DEFAULT_OBJECT_MASS_KG}kg"
        ent = getattr(cfg.scene, key)
        if isinstance(ent, ArticulationCfg):
            ent.spawn.mass_props = None  # blocks the per-link path: apply_nested would attach m to every link
            _orig_func = ent.spawn.func

            def _spawn_then_distribute(prim_path, spawn_cfg, *a, **k):
                prim = _orig_func(prim_path, spawn_cfg, *a, **k)
                _artic_distribute_link_mass(prim_path, key, m, ctx_label)
                return prim

            ent.spawn.func = _spawn_then_distribute
            print(
                f"[{ctx_label}] object mass: {key} = {m} kg ({src}) — [ARTIC] per-link distribution (after the spawn, "
                f"USD mass ratio)"
            )
            return
        ent.spawn.mass_props = _su.MassPropertiesCfg(mass=m)
        print(f"[{ctx_label}] object mass: {key} = {m} kg ({src})")

    _active = "object_left" if getattr(args_cli, "left_uni", 0) else "object_right"
    _set(_active, getattr(args_cli, "object_mass", None))
    if not getattr(args_cli, "left_uni", 0):  # left_uni: object_left IS the active slot above
        _set("object_left", getattr(args_cli, "object_mass_left", None))
