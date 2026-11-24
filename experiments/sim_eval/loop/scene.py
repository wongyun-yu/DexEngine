"""Scene look of the closed loop: render-only prim shift for world-moving scenes and object paint (obj_color /
obj_keep)."""

from __future__ import annotations

from experiments.sim_eval.loop.common import _say
from experiments.sim_eval.session_cfg import value as _cfg

PHYS_APIS = ("PhysicsCollisionAPI", "PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI", "PhysxContactReportAPI")


SKIP_TYPES = ("Scope", "Material", "Shader", "PhysicsScene", "PhysicsCollisionGroup", "")


class RenderShift:
    """Closed-loop render of a world-moving scene (aerial world_shift). Physics stays in stage-1 coordinates (robot root
    = base + dz, objects and discs unchanged); only the **physics-free render-only prims** are moved by z + dz for the
    capture and moved back — PhysX state is neither read nor written. The relative geometry equals the record render
    rule (collect_body.shifted — binding root z − dz; background, supports, extras and outside cameras in the render
    frame): (render-only + dz) − physics = render-only − (physics − dz).
    Render-only = children of /World and /World/envs/env_0 that are not binding roots and carry no physics API below
    them (a prim mixing physics is descended into with the same rule) — a rule, not a name list. Rendered collision
    planes
    (physics, cannot move) are hidden during the capture."""

    def __init__(self, S, T, body: dict):
        from dexengine.record.render import desk_stretch as DS

        self.S, self.binds = S, T["binds"]
        if DS.world_moved(body):
            self.dz, self.why = (
                float(body["root_z_offset_m"]),
                f"{body['method']} dz {float(body['root_z_offset_m']):+.4f}",
            )
        else:
            self.dz, self.why = None, None
        self.n, self.ops, self.hide = 0, [], []
        if self.dz is not None:
            self._collect()

    def _collect(self) -> None:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        st = omni.usd.get_context().get_stage()
        roots = [
            b.asset.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace(".*", "0") for b in self.binds
        ]

        def phys(p) -> bool:  # true if any physics API or physics type (joint, collision group) is below
            return any(
                any(a in q.GetAppliedSchemas() for a in PHYS_APIS) or str(q.GetTypeName()).startswith("Physics")
                for q in Usd.PrimRange(p)
            )

        def visit(p, out, hide):
            pth = str(p.GetPath())
            if (
                any(pth == r or pth.startswith(r + "/") or r.startswith(pth + "/") for r in roots)
                or pth == "/World/envs"
            ):
                if pth in ("/World/envs", "/World/envs/env_0"):
                    for c in p.GetChildren():
                        visit(c, out, hide)
                return
            if p.GetTypeName() in SKIP_TYPES and not p.GetChildren():
                return
            if not phys(p):
                if UsdGeom.Xformable(p):
                    out.append(p)
                return
            if (
                UsdGeom.Gprim(p)
                and UsdGeom.Imageable(p).ComputePurpose() in ("default", "render")
                and UsdGeom.Imageable(p).ComputeVisibility() != "invisible"
            ):
                hide.append(
                    p
                    # rendered geometry with physics attached (e.g. the floor collision plane) — not moved, hidden
                    # during the capture
                )
            for c in p.GetChildren():
                visit(c, out, hide)

        prims, hide = [], []
        for c in st.GetPrimAtPath("/World").GetChildren():
            visit(c, prims, hide)
        for p in prims:
            xf = UsdGeom.Xformable(p)
            op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "e5RenderShift")
            op.Set(Gf.Vec3d(0, 0, 0))
            xf.SetXformOpOrder(
                [op] + [o for o in xf.GetOrderedXformOps() if o.GetOpName() != op.GetOpName()], xf.GetResetXformStack()
            )
            self.ops.append((str(p.GetPath()), op))
        self.hide = [(str(p.GetPath()), UsdGeom.Imageable(p)) for p in hide]
        _say(
            f"[E5_WORLD] {len(self.ops)} render-only prims at capture z {self.dz:+.4f} · hidden physics "
            f"geometry {[h for h, _ in self.hide]} · "
            f"moved {[p for p, _ in self.ops]}"
        )

    def describe(self) -> str:
        if self.dz is None:
            return "[E5_WORLD] no world shift"
        return f"[E5_WORLD] render-only prims moved for capture {self.why}"

    def capture(self, warm):
        from pxr import Gf

        if self.dz is None:
            self.n += self.dz is not None
            return self.S.rig.capture(warm)
        try:
            for _p, op in self.ops:
                op.Set(Gf.Vec3d(0, 0, self.dz))
            for _p, im in self.hide:
                im.MakeInvisible()
            cap = self.S.rig.capture(warm)
        finally:
            for _p, op in self.ops:
                op.Set(Gf.Vec3d(0, 0, 0))
            for _p, im in self.hide:
                im.MakeVisible()
        self.n += 1
        return cap


def _paint_objects(S, T, look: str, variant: str = "") -> None:
    """obj_color=<name> (default off): measure the closed loop with the paint of the data look r0_*_red.
    Value = augvis catalog OBJ_COLORS · paint = augvis scene paint_objects · roots = manipulated objects + physics
    extra slots (hidden z ≤ -5 excluded) + the original scene's extra objects — place pad, robot, desk and room are
    not painted. Marker [OBJ_COLOR]. obj_keep=print (default off): the objects of augvis catalog PRINT_KEEP keep their
    original look — marker [PRINT_KEEP] per session (also with nothing to keep); the kept objects must match the scene
    roots exactly and
    not be bound to the paint material after painting, else it dies. The post-paint state is measured and logged."""
    name = _cfg("obj_color", "")
    keep_mode = _cfg("obj_keep", "")
    if keep_mode not in ("", "print"):
        raise SystemExit(f"[PRINT_KEEP] obj_keep={keep_mode!r} — print only")
    if keep_mode and not name:
        raise SystemExit("[PRINT_KEEP] obj_keep needs obj_color")
    if not name:
        return
    import omni.usd

    from dexengine.record.looks.augvis import catalog as AC
    from dexengine.record.looks.augvis import scene as SC
    from dexengine.record.render import body as BODY

    if name not in AC.OBJ_COLORS:
        raise SystemExit(f"[OBJ_COLOR] obj_color={name!r} — one of {sorted(AC.OBJ_COLORS)}")
    if look != "D":
        raise SystemExit(f"[OBJ_COLOR] obj_color applies to look-D scenes only — this scene's look is {look}")
    binds, st = T["binds"], T["st"]
    objs = [BODY._root_path(b) for b in binds if b.name.startswith("object")]
    slots = [
        BODY._root_path(b)
        for b in binds
        if b.name.startswith("extra_obj_") and float(st[:, b.root_cols.start + 2].max()) > -5.0
    ]
    extras = [r["prim"] for r in (T.get("extras") or {}).get("placed", []) if r.get("prim")]
    roots = objs + slots + extras
    oc = AC.OBJ_COLORS[name]
    stage, kept, keep = omni.usd.get_context().get_stage(), [], []
    if keep_mode:
        keep = AC.print_keep(variant)  # a different name form dies
        ids = {p: SC.prim_obj_id(stage, p) for p in objs + slots}
        ids.update({x["prim"]: x.get("obj_id") for x in (T.get("extras") or {}).get("placed", []) if x.get("prim")})
        kept = [p for p in roots if ids.get(p) in keep]
        if sorted(str(ids[p]) for p in kept) != sorted(keep):
            raise SystemExit(f"[PRINT_KEEP] {variant} kept objects {keep} ≠ scene roots {ids}")
        roots = [p for p in roots if p not in kept]
    r = (
        SC.paint_objects(stage, roots, oc, f"D_{name}")
        if roots or not keep_mode
        else {"roots": 0, "meshes": 0, "material": None}
    )  # a clip whose roots PRINT_KEEP all removed (spam_can__1)
    print(
        f"[OBJ_COLOR] D_{name} rgb={list(oc)} · objects {r['roots']} (manipulated {len(objs)} · slots "
        f"{len(slots)} · extras {len(extras)}) · meshes {r['meshes']} · "
        f"material {r['material']} · roots {[str(x).rsplit('/', 1)[-1] for x in roots]}",
        flush=True,
    )
    if keep_mode:
        from pxr import Usd, UsdGeom, UsdShade

        meshes = painted = 0
        for p in kept:
            for q in Usd.PrimRange(stage.GetPrimAtPath(p), Usd.TraverseInstanceProxies()):
                if q.IsA(UsdGeom.Mesh):
                    meshes += 1
                    mat = UsdShade.MaterialBindingAPI(q).ComputeBoundMaterial()[0]
                    painted += bool(mat) and str(mat.GetPath()).endswith("_objcolor")
        if keep and (not meshes or painted):
            raise SystemExit(f"[PRINT_KEEP] {variant} kept roots {kept} meshes {meshes} · paint material {painted}")
        print(
            f"[PRINT_KEEP] {variant} objects {keep} · roots {[str(x).rsplit('/', 1)[-1] for x in kept]} · "
            f"meshes {meshes} · paint material {painted} · "
            f"other roots {[str(x).rsplit('/', 1)[-1] for x in roots]}",
            flush=True,
        )
