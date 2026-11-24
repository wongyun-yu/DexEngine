"""Measurement and apply layer (_Run) of the record render visual augment: the random looks of the per-clip render
session (render.clip_session) inherit it through looks.look_run.LookRun (constructor and trajectory context there).
It puts variations on top of look D (axes of augvis.catalog: top, floor and wall materials, panel intensity, colour
temperature, WB, distractors; the scene-side functions are augvis.scene) and checks light convergence (LOOK.settle,
the record canon). Physics, trajectory, head pose and exposure do not change.

Scene measurements (once, on D):
  object pixels  depth difference (> MASK_DZ_M) with the object prims hidden and shown -> left-eye object pixels ->
                 mean D render colour (the object colour of the contrast rule)
  top pixels     left-eye depth back to world, on the top surface (|z − top| < DESK_DZ_M · inside the top) -> mean
                 D render colour
  swept volume   top grid height field (augvis.scene Swept) of robot link origins over every STRIDE step +
                 segments between joint pairs (USD PhysicsJoint body0/1) · objects · supports (disc + post); a
                 distractor needs top + height + CLEAR_M < height field over its whole footprint.
  occlusion      a distractor cylinder crossed by a segment from the left eye (every step) or the outside camera to
                 the object centre or the active hand centre drops that place.
Markers [AUG_VIS] ….
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dexengine.record.looks import look_d as LOOK
from dexengine.record.looks.augvis import allow as AL
from dexengine.record.looks.augvis import catalog as AC
from dexengine.record.looks.augvis import sample as AS  # owner of the sampler (sample_set · card rotation)
from dexengine.record.looks.augvis import scene as SC
from dexengine.record.looks.augvis import swept as SW
from dexengine.record.render import body as BODY
from dexengine.record.render import cameras as CAMS
from dexengine.record.render import rig as RIG

MARK = AC.MARK
STRIDE = 2
MASK_DZ_M, DESK_DZ_M = 0.005, 0.004
WARM = RIG.STILL_WARMUP * 2


class _Run:
    """Scene, trajectory and camera state of one session (one clip); constructor and render order are the subclass's
    (looks.look_run.LookRun)."""

    # ── trajectory writing · capture ──
    def write(self, t: int) -> None:
        self.TR.write_states(self.scene, self.binds, self.P[t : t + 1], self.eids)
        self.raw_env.sim.physics_sim_view.update_articulations_kinematic()

    def capture_at(self, fr: int, n: int = WARM) -> dict:
        self.write(self.steps[fr])
        return self.rig.capture(n)

    @staticmethod
    def save(img, p: Path) -> None:
        from PIL import Image

        h, w = img.shape[:2]
        Image.fromarray(img).resize((960, round(960 * h / w)), Image.LANCZOS).save(p, quality=85)

    # ── swept volume · sight lines ──
    def build_swept(self) -> None:
        import numpy as np

        t0 = time.time()
        stage, st, rig, desk = self.stage, self.st, self.rig, self.desk
        arts = [b.asset for b in self.binds if b.kind == "artic" and not b.name.startswith("object")]
        pairs = {id(a): SW.joint_pairs(stage, a) for a in arts}
        if not all(pairs.values()):
            raise SystemExit(f"{MARK} robot joint pairs not found in the USD {[len(v) for v in pairs.values()]}")
        orad = [SW.object_radius(stage, BODY._root_path(b)) for b in self.objs]
        sw = SW.Swept(desk) if not self.aerial else None
        sight_steps = list(range(0, st.shape[0], 10)) + [st.shape[0] - 1]
        sightlines = []
        # with exo off (render.cameras EXO_ENV=0) cams_off turned that camera's render product off; reading its data
        # (Camera.data -> annotator readback) dies after a few renders with Warp memcpy_d2d invalid argument (measured
        # on the first random look of an aug round). Then the exo place = the value calibrate put (rig.exo_pos_w, the
        # input of the same set_world_poses).
        exo_cam = rig.cams["collect_exo_front"] if CAMS.exo_on() else None
        for t in range(0, st.shape[0], STRIDE) if sw else []:
            self.write(t)
            for a in arts:
                a.update(1e-6)
                B = a.data.body_pos_w[0].detach().cpu().numpy().astype(float)
                pts = [B]
                for i0, i1 in pairs[id(a)]:
                    n = max(2, int(np.linalg.norm(B[i1] - B[i0]) / SW.SEG_M) + 1)
                    pts.append(B[i0] + np.linspace(0, 1, n)[:, None] * (B[i1] - B[i0]))
                sw.add(np.concatenate(pts), SW.R_LINK)
            for b, r in zip(self.objs, orad, strict=True):
                b.asset.update(1e-6)
                sw.add(b.asset.data.root_pos_w[0:1].detach().cpu().numpy().astype(float), r)
        if sw:
            for pts, r in SW.support_samples((self.look["supports"] or {}).get("discs", []), desk, desk["origin"]):
                sw.add(pts, r)
            rig.render(2)
            exo_p = (
                exo_cam.data.pos_w[0].detach().cpu().numpy().astype(float)
                if exo_cam is not None
                else np.asarray(rig.exo_pos_w, float)
            )
            for t in sight_steps:
                self.write(t)
                eye = rig.pose("collect_ego_left")[0]
                tg = [self.per["object"][t][i] for i in range(len(self.per["object"][t]))]
                if len(self.per["hand"][t]):
                    tg.append(np.asarray(self.per["hand"][t]).mean(0))
                for q in tg:
                    sightlines += [(eye, np.asarray(q, float)), (exo_p, np.asarray(q, float))]
        self.t_swept = time.time() - t0
        self.sw = sw
        self.ctx = {"desk": desk, "swept": sw, "sightlines": sightlines}
        if sw:  # height field · sight lines (for pictures and audits)
            np.savez_compressed(
                self.out / "swept.npz",
                x=sw.x,
                y=sw.y,
                h=sw.h,
                top=desk["top_z"],
                sight=np.array([[*e, *t] for e, t in sightlines], float),
            )
        kind = "aerial" if self.aerial else "desk"
        print(
            f"{MARK} scene {self.clip} · {kind} · swept spheres {sw.n if sw else 0} · free top at height 0.15 m "
            f"{sw.free_frac() if sw else None} · sight lines {len(sightlines)} · {self.t_swept:.1f} s",
            flush=True,
        )

    # ── measurement tools ──
    def back_project(self, depth, eye, q):
        import numpy as np

        from dexengine.record.render import framing as CF

        k = self.k
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = np.asarray(depth, float)
        pc = np.stack([(u - k["cx"]) / k["fx"] * z, (v - k["cy"]) / k["fy"] * z, z], -1).reshape(-1, 3)
        return (pc @ CF.quat_R(q).T + np.asarray(eye)).reshape(H, W, 3)

    def desk_mask(self, depth, eye, q, excl):
        import numpy as np

        desk = self.desk
        if self.aerial:
            return np.zeros(depth.shape, bool)
        Pw = self.back_project(depth, eye, q)
        m = np.isfinite(depth) & (depth > 0) & (np.abs(Pw[..., 2] - desk["top_z"]) < DESK_DZ_M)
        m &= (Pw[..., 0] >= desk["near_x"]) & (Pw[..., 0] <= desk["far_x"])
        m &= np.abs(Pw[..., 1] - desk["cy"]) <= desk["half_y"]
        return m & ~excl

    def converge(self) -> dict:
        """Light convergence = the record canon LOOK.settle (load + light transport flow; rules and thresholds owned
        by look_d) alone."""
        d = LOOK.settle(self.rig)
        return {
            "settle_rounds": d["load_rounds"],
            "conv": d["drift"],
            "conv_ok": not d["drift_cap"],
            "settle_s": d["s"],
            "settle_renders": d["renders"],
        }

    # ── on D: object pixels · top pixels · allowed list ──
    def measure_base(self) -> None:
        import numpy as np

        from pxr import UsdGeom

        stage, frames = self.stage, self.frames
        self.conv0 = self.converge()
        print(f"{MARK} reference convergence {self.conv0}", flush=True)
        self.base = {}
        for fr in frames:
            cap = self.capture_at(fr)
            eye, q = self.rig.pose("collect_ego_left")[:2]
            masks = {}
            for nm, paths in (("om", self.obj_paths), ("rm", self.robot_paths)):
                for p in paths:
                    UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeInvisible()
                hid = self.capture_at(fr)
                for p in paths:
                    UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeVisible()
                d0, d1 = np.nan_to_num(cap["depth_m"], posinf=99), np.nan_to_num(hid["depth_m"], posinf=99)
                masks[nm] = np.abs(d0 - d1) > MASK_DZ_M
            om, rm = masks["om"], masks["rm"] & ~masks["om"]
            dm = self.desk_mask(cap["depth_m"], eye, q, om | rm)
            self.base[fr] = {
                "depth": cap["depth_m"],
                "om": om,
                "rm": rm,
                "dm": dm,
                "eye": eye,
                "q": q,
                "img": cap["egoL"],
            }
            # look D as it is before variations (before materials are rebound), for the D reproduction check
            (self.out / "D0").mkdir(exist_ok=True)
            for cam in ("egoL", "exo") if CAMS.exo_on() else ("egoL",):  # exo off = left eye only
                self.save(cap[cam], self.out / "D0" / f"{cam}_{fr:03d}.jpg")
        # object and top colour = mean over the object and top pixels of three frames (start, middle, end)
        # (in some clips a hand hides the object in the middle)
        b = self.base
        opx = np.concatenate([b[f]["img"][b[f]["om"]] for f in frames]) if frames else np.zeros((0, 3))
        dpx = np.concatenate([b[f]["img"][b[f]["dm"]] for f in frames]) if frames else np.zeros((0, 3))
        obj_rgb = (opx.mean(0) / 255).round(4).tolist() if len(opx) > 50 else None
        desk_rgb = (dpx.mean(0) / 255).round(4).tolist() if len(dpx) > 200 else None
        names = [o.name for o in self.objs] + [str(Path(o.asset.cfg.prim_path).name) for o in self.objs]
        self.scene_info = {
            "aerial": self.aerial,
            "clip": self.clip,
            "object_names": names,
            "object_rgb": obj_rgb,
            "desk_rgb_d": desk_rgb,
            "look_d_albedo": self.man["look_d"],
            "object_px": len(opx),
            "desk_px": len(dpx),
        }
        self.allow = AL.allowed(self.scene_info, self.man)
        self.cnt = AS.count(self.allow)
        print(
            f"{MARK} scene measurement · object pixels {len(opx)} colour {obj_rgb} · top pixels {len(dpx)} colour "
            f"{desk_rgb} · allowed {json.dumps({a: len(v) for a, v in self.allow['allowed'].items()})} · "
            f"excluded tops {self.allow['excluded'].get('desk')} · "
            f"discrete variations {self.cnt['total']}",
            flush=True,
        )

    # ── applying variations ──
    def paint_roots(self) -> list[str]:
        """Roots painted by flat colour object looks: this class has the manipulated objects (objs). The record session
        (looks.look_run) adds the original scene's extra objects."""
        return [BODY._root_path(b) for b in self.objs]

    def root_obj_ids(self) -> dict:
        """Paint root -> object id; only sessions that use PRINT_KEEP (looks.look_run) know it."""
        raise SystemExit("[PRINT_KEEP] this session does not know object ids")

    def print_split(self, spec) -> tuple[list, list]:
        """(roots to paint, roots to keep). No obj_keep = (paint_roots, []), byte-identical. Every kept object must have
        exactly one root, else it dies."""
        roots = self.paint_roots()
        keep = spec.get("obj_keep")
        if not keep:
            return roots, []
        ids = self.root_obj_ids()
        kept = [r for r in roots if ids.get(r) in keep]
        if sorted(str(ids[r]) for r in kept) != sorted(keep):
            raise SystemExit(f"[PRINT_KEEP] {spec.get('name')} kept objects {keep} != scene roots {ids}")
        return [r for r in roots if r not in kept], kept

    def apply_spec(self, spec):
        """Applies a variation (materials, light, distractors; no convergence). render.clip_session uses this path too
        (convergence is its render's)."""
        d = self.desk
        desk_pt = (0.5 * (d["near_x"] + d["far_x"]), d["cy"], d["top_z"] + float(d.get("aerial_desk_dz") or 0.0))
        ok = SC.apply_materials_lights(self.stage, spec, self.man, self.root, self.aerial, desk_pt)
        placed, dropped = (
            SW.place_distractors(self.stage, spec, self.ctx, self.man, self.root) if not self.aerial else ([], [])
        )
        oc = spec.get("obj_color")  # flat colour object look (default none = nothing below this line)
        if oc is not None or getattr(self, "_obj_painted", False):
            roots = self.print_split(spec)[0]
            r = (
                SC.paint_objects(
                    self.stage, roots, oc, str(spec.get("name", "x")).replace("-", "_"), rough=spec.get("obj_rough")
                )
                # a clip whose roots PRINT_KEEP removed entirely (spam_can__1)
                if roots or not spec.get("obj_keep")
                else {"roots": 0, "meshes": 0, "material": None}
            )
            self._obj_painted = oc is not None
            print(
                f"[OBJ_COLOR] {spec.get('name')} rgb={oc} · objects {r['roots']} · meshes {r['meshes']} · "
                f"material {r['material']} · roots {[str(x).rsplit('/', 1)[-1] for x in roots]}",
                flush=True,
            )
        ot = spec.get("obj_tint")  # original texture of the print look (default none = nothing below this line)
        if ot is not None or getattr(self, "_obj_tinted", False):
            roots = self.print_split(spec)[0]
            # a clip with all roots removed = restore only
            r = SC.tint_objects(self.stage, roots, ot if roots or not spec.get("obj_keep") else None)
            self._obj_tinted = ot is not None
            print(
                f"[OBJ_TINT] {spec.get('name')} tint={ot} · objects {r['roots']} · meshes {r['meshes']} · textures "
                f"{r['texture']} · constants {r['constant']} · MDL {r['mdl']} · untinted {r['untinted']} "
                f"{json.dumps(r['why'], ensure_ascii=False)} · "
                f"roots {[str(x).rsplit('/', 1)[-1] for x in roots]}",
                flush=True,
            )
        # objects not painted (PRINT_KEEP): the real state after paint and tint is measured and printed. No marker
        if spec.get("obj_keep"):
            roots, kept = self.print_split(spec)
            s = SC.kept_state(self.stage, kept)
            if not s["meshes"] or s["painted"] or s["tinted"]:
                raise SystemExit(f"[PRINT_KEEP] {spec.get('name')} kept roots {kept} state {s}")

            def short(rs):
                return [str(x).rsplit("/", 1)[-1] for x in rs]

            print(
                f"[PRINT_KEEP] {spec.get('name')} objects {json.dumps(spec['obj_keep'])} · roots {short(kept)} · "
                f"meshes {s['meshes']} · paint material {s['painted']} · tint {s['tinted']} · other roots "
                f"{short(roots)}",
                flush=True,
            )
        if spec.get("obj_noise") is not None:
            print(f"[OBJ_NOISE] {spec.get('name')} {json.dumps(spec['obj_noise'], ensure_ascii=False)}", flush=True)
        print(AS.spec_line(spec, placed) + f" dropped={json.dumps([d['key'] for d in dropped])}", flush=True)
        return ok, placed, dropped
