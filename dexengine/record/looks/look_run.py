#!/usr/bin/env python3
"""Random look layer (LookRun) of the per-clip render session; called by render.clip_session (used only inside the
Isaac session). It uses augvis.render _Run's canonical measurement and application (measure_base · build_swept ·
apply_spec) as they are, on top of the session's trajectory context. allow (the allowed list) is measured on the
original trajectory only (measure · record.episodes allow_traj): the premise of cards that do not depend on the
round."""

from __future__ import annotations

import json
import time
from pathlib import Path

from dexengine.core.errors import InputMissingError

#: file that hands the allow measured by the head session (record.clip per_look: choose · measure) to the drawing
#: sessions (under <out>/_look/): drawing sessions do not run the measure renders (original f0 · middle · end
#: convergence + hidden renders), which left the original object place in the RTX cache and printed it as a
#: translucent ghost on the next episode frames (augment A/B · 2 choosing sessions · 1 fixed session).
ALLOW_FILE = "allow.json"
ALLOW_MARK = "[LOOK_ALLOW]"


class _LookRunSteps:
    """The LookRun steps that need nothing from the factory scope (bind, measure, painted, spare); LookRun puts
    this class before augvis.render._Run in its bases, as if the methods were written in its own body."""

    def bind(self, T: dict) -> None:
        from dexengine.record.render import body as BODY

        self.clip, self.tf, self.st, self.binds, self.P = T["clip"], T["tf"], T["st"], T["binds"], T["P"]
        self.steps, self.per, self.look = T["steps"], T["per"], T["look"]
        self.objs = [b for b in self.binds if b.name.startswith("object")]
        self.obj_paths = [BODY._root_path(b) for b in self.objs]
        # roots painted by flat colour object looks = manipulated objects + physics extra slots + this trajectory's
        # original scene extra objects (visual prims placed by scene.extras; not distractors)
        self.extras_placed = (T.get("extras") or {}).get("placed", [])
        self.extra_paths = [r["prim"] for r in self.extras_placed if r.get("prim")]
        # physics extra slots (extra_obj_*, e.g. the swept plate of brush_brush_plate, once left unpainted).
        # Empty slots sit at the hidden pose (z <= -5, object_pin.reset_discs rule) and are not painted
        self.slot_paths = [
            BODY._root_path(b)
            for b in self.binds
            if b.name.startswith("extra_obj_") and float(self.st[:, b.root_cols.start + 2].max()) > -5.0
        ]
        self.robot_paths = [
            BODY._root_path(b) for b in self.binds if b.kind == "artic" and not b.name.startswith("object")
        ]
        self.frames = sorted({0, len(self.steps) // 2, len(self.steps) - 1})
        self.out = self.out0 / self.clip
        self.out.mkdir(parents=True, exist_ok=True)

    def measure(self, T: dict, plan: dict) -> dict:
        """allow and deck only in the head session · on the original (allow_traj) · look D · canonical exposure
        (LREV.LOOK_REV); otherwise it dies."""
        from dexengine.record import episodes as EPI
        from dexengine.record.looks import rev as LREV
        from dexengine.record.render.clip_session import MARK  # owner of the session marker

        # drawing sessions = no measure renders (ALLOW_FILE)
        if self.no_measure or T["clip"] != EPI.allow_traj(plan):
            raise SystemExit(
                f"{MARK} {ALLOW_MARK} allow only in the head session on the original "
                f"{EPI.allow_traj(plan)}: loaded trajectory {T['clip']} · drawing session "
                f"{self.no_measure}"
            )
        if abs(LREV.exposure_now() - LREV.exposure_s(0.0)) > 1e-7:
            raise SystemExit(
                f"{MARK} allow measurement exposure {LREV.exposure_now()} != canonical "
                f"{LREV.exposure_s(0.0)}: the cards would split"
            )
        t0 = time.perf_counter()
        self.bind(T)
        self.measure_base()
        args = {k: int(v) for k, v in plan["deck"].items()}
        self.rows = self.AS.sample_set(self.allow, plan["look_key"], **args)
        return {
            "how": "augvis_render._Run.measure_base → augvis_catalog.allowed · augvis_sample.sample_set("
            "allow, plan.look_key, **plan.deck)",
            "traj": T["clip"],
            "key": plan["look_key"],
            "args": args,
            "scene": self.scene_info,
            "allowed": {a: len(v) for a, v in self.allow["allowed"].items()},
            "measure_s": round(time.perf_counter() - t0, 2),
        }

    @staticmethod
    def painted(spec: dict, ep: dict) -> dict:
        """Flat colour object looks (dense_render DENSE_OBJ_COLORS): the one place the original card and the
        brightness gate spare cards both pass. Slot as it is · name = <slot>_<colour>. Episodes without a colour
        keep the spec."""
        # PRINT_KEEP; none = byte-identical
        keep = {"obj_keep": [str(o) for o in ep["obj_keep"]]} if ep.get("obj_keep") else {}
        # original texture of the print look (DENSE_PRINT_LOOKS); meta = obj_tint · obj_noise
        if ep.get("obj_tint") is not None:
            return dict(
                spec,
                name=ep["look"],
                obj_tint=[float(c) for c in ep["obj_tint"]],
                obj_noise=ep["obj_noise"],
                **keep,
            )
        if ep.get("obj_color") is None:
            return spec
        if ep.get("obj_noise") is not None:  # flat colour of the print look; meta adds obj_rough · obj_noise
            return dict(
                spec,
                name=ep["look"],
                obj_color=[float(c) for c in ep["obj_color"]],
                obj_rough=float(ep["obj_rough"]),
                obj_noise=ep["obj_noise"],
                **keep,
            )
        # keep of flat colour looks (DENSE_COLOR_KEEP); none = byte-identical
        return dict(spec, name=ep["look"], obj_color=[float(c) for c in ep["obj_color"]], **keep)

    def spare(self, T: dict, ep: dict, plan: dict, k: int) -> tuple[dict, dict]:
        """Spare card k of the brightness gate: the spare deck of the same allow with the key
        '<look_key>|floor|r<row>c<column>' (LOOK_REV.random_tries cards). From the k-th only the room and light
        axes are taken; the row's distractors, place seed and name stay those of the original card (depth once per
        row). Returns (spec, applied info) after applying; light convergence happens once on the drawing side
        (looks.rev settle_gate)."""
        from dexengine.record.looks import rev as LREV
        from dexengine.record.render.clip_session import MARK  # owner of the session marker

        n = int(LREV.LOOK_REV["random_tries"])
        orig = self.rows[ep["row"]][ep["col"]]
        key = f"{plan['look_key']}|floor|r{ep['row']}c{ep['col']}"
        if not 0 <= k < n:
            raise SystemExit(f"{MARK} spare card {k}: outside 0-{n - 1}")
        spec = dict(self.AS.sample_set(self.allow, key, n_looks=n, n_rows=1)[0][k], spare={"k": k, "key": key})
        # per row · the name is the original card's (absent there = absent on spares)
        for kk in ("name", "distractors", "place_seed"):
            if kk in orig:
                spec[kk] = orig[kk]
            else:
                spec.pop(kk, None)
        spec = self.painted(spec, ep)
        t = time.perf_counter()
        ok, placed, dropped = self.apply_spec(spec)
        return spec, {
            "apply_s": round(time.perf_counter() - t, 2),
            "placed": [d["key"] for d in placed],
            "dropped": dropped,
            "ok": ok,
        }


def look_run_cls():
    from dexengine.record.looks.augvis import render as AR
    from dexengine.record.render.clip_session import MARK  # owner of the session marker (the caller, already loaded)

    class LookRun(_LookRunSteps, AR._Run):
        """Random looks: augvis.render's canonical measurement and application (measure_base · build_swept ·
        apply_spec) as they are, on the session's trajectory context (only the constructor differs: trajectory,
        cameras and body were already set up by render.episode load_traj). allow = measure_base before the first random
        episode (still look D) -> augvis.allow.allowed · spec = AS.sample_set(allow, plan look_key, **plan
        deck)[row][column]; a name other than the episode look dies."""

        def __init__(self, S, out: Path):
            import omni.usd

            from dexengine.rl.eval import traj_record as TR

            self.TR, self.raw_env, self.scene, self.eids, self.rig = TR, S.env, S.scene, S.eids, S.rig
            self.stage = omni.usd.get_context().get_stage()
            self.root = AR.SC.assets_root()
            self.man = json.loads((self.root / "manifest.json").read_text())
            self.desk, self.aerial, self.k, self.out0 = S.rig.desk(), bool(S.rig.desk_dz), S.cal["zed"], out
            self.clip, self.rows, self.no_measure = None, None, False

        def paint_roots(self) -> list[str]:
            return list(self.obj_paths) + list(self.slot_paths) + list(self.extra_paths)

        def root_obj_ids(self) -> dict:
            """Paint root -> object id (read by PRINT_KEEP). Manipulated objects and physics slots are measured on the
            stage · visual extras come from their placement record."""
            ids = {p: AR.SC.prim_obj_id(self.stage, p) for p in list(self.obj_paths) + list(self.slot_paths)}
            ids.update({r["prim"]: r.get("obj_id") for r in (self.extras_placed or []) if r.get("prim")})
            return ids

        def spec_for(self, T: dict, ep: dict, plan: dict) -> tuple[dict, dict]:
            clk = time.perf_counter
            info = {}
            if self.rows is None:  # first random episode (orig): the scene is still look D
                info["allow"] = self.measure(T, plan)
            if self.clip != T["clip"]:
                self.bind(T)
            if getattr(self, "swept_for", None) != T["clip"]:
                t = clk()
                self.build_swept()
                self.swept_for, info["swept_s"] = T["clip"], round(clk() - t, 2)
            rq = plan.get("redraw")
            if rq is not None:  # brightness gate redraw (record redraw): the original card is not applied
                spec, more = self.spare(T, ep, plan, int(rq["k"]))
                return spec, dict(info, **more)
            spec = self.rows[ep["row"]][ep["col"]]
            if spec["name"] != ep.get("look_base", ep["look"]):
                raise SystemExit(
                    f"{MARK} spec name {spec['name']} != episode look "
                    f"{ep.get('look_base', ep['look'])} (row {ep['row']} · column {ep['col']})"
                )
            spec = self.painted(spec, ep)
            t = clk()
            ok, placed, dropped = self.apply_spec(spec)
            info.update(apply_s=round(clk() - t, 2), placed=[d["key"] for d in placed], dropped=dropped, ok=ok)
            return spec, info

    LookRun.AS = AR.AS
    return LookRun


def save_allow(lr, plan: dict, meta: dict) -> Path:
    """Head session: writes the allow LookRun measured to a file (same place, overwrite by rename)."""
    from dexengine.record import episodes as EPI

    args = {k: int(v) for k, v in plan["deck"].items()}
    rec = {
        "traj": EPI.allow_traj(plan),
        "key": plan["look_key"],
        "args": args,
        "allow": lr.allow,
        "scene_info": lr.scene_info,
        "measure": meta,
    }
    lr.out0.mkdir(parents=True, exist_ok=True)
    p, tmp = lr.out0 / ALLOW_FILE, lr.out0 / (ALLOW_FILE + ".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, default=float))
    tmp.replace(p)
    print(f"[RENDER_CLIP] {ALLOW_MARK} from=measure → {p}", flush=True)
    return p


def adopt(lr, plan: dict) -> dict:
    """Drawing session: reads the head session's allow and builds the card rows (0 measure renders; a measure after
    this dies). A missing file, or an original, key or deck other than the plan, dies."""
    from dexengine.record import episodes as EPI

    p = lr.out0 / ALLOW_FILE
    if not p.is_file():
        raise InputMissingError(f"[RENDER_CLIP] {ALLOW_MARK} head session allow file missing {p}")
    rec = json.loads(p.read_text())
    args = {k: int(v) for k, v in plan["deck"].items()}
    want = {"traj": EPI.allow_traj(plan), "key": plan["look_key"], "args": args}
    got = {k: rec.get(k) for k in want}
    if got != want:
        raise SystemExit(f"[RENDER_CLIP] {ALLOW_MARK} allow file {got} != plan {want}: {p}")
    lr.allow, lr.scene_info, lr.no_measure = rec["allow"], rec["scene_info"], True
    lr.rows = lr.AS.sample_set(lr.allow, plan["look_key"], **args)
    print(f"[RENDER_CLIP] {ALLOW_MARK} from=file {p} · measure renders 0", flush=True)
    return dict(rec.get("measure") or {}, adopted=str(p))
