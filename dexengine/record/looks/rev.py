"""Look revision on top of look D: the brightness gate and the metal-0 prop. Values live in LOOK_REV.

Probe = head left eye: BT.601 Y mean (0-255) and the clipped-highlight ratio (Y >= clip_hi_y) of the frame at grasp
onset + probe_after_grasp_s.
  look D    d_gate = post: no pre-probe; exposure = canonical x 2^d_ev, measured on the drawn probe frame. Below floor
            the episode goes on the re-render list (look.exposure.rerender: the next exposure is the d_ev_step multiple
            that reaches floor at the measured gain ev_gain per stop, capped at d_ev_max). Re-renders run in a new
            session.
  random    exposure = canonical (set again per episode), measured on the drawn probe frame (r_post). Outside
            [floor, ceiling] the episode goes on the re-render list (look.rerender: spare deck cards from 0,
            random_tries cards; the line's distractors and placement seeds stay; LookRun.spare; new session).
  No grasp (object never moves) = middle frame of the trajectory (no_grasp_probe).
  The re-render list is consumed by redraw_after below (new session; settle_gate reads the plan's redraw field).
  metal-0 props: the material of metal0_props is rebound to OmniPBR with base colour, normal, ORM textures and tint
            kept and metallic 0 (metallic_texture_influence 0, roughness = ORM.g); the cupboard's mirror-like texels
            otherwise reflect the room as haze.
Marker: [LOOK_FLOOR] D_exposure|random ... once per drawn episode (gate_problems counts them).
Strings in measured / exposure / rerender land in episode lines and keep their exact text.
"""

from __future__ import annotations

import json
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record.looks import look_d as LOOK
from dexengine.record.looks import look_settle as LS

LOOK_REV = {
    "name": "D4-20261021",
    "d_gate": "post",
    "ev_gain": 1.275,
    "d_ev": 1.0,
    "d_ev_step": 0.25,
    "d_ev_max": 2.0,
    "floor": 85.0,
    "ceiling": 190.0,
    "clip_hi_y": 250,
    "clip_hi_max": 0.001,
    "probe_cam": "egoL",
    "probe_after_grasp_s": 0.6,
    "no_grasp_probe": "mid",
    "random_tries": 8,
    "metal0_props": ("SM_Cupboard.usd",),
}
FLOOR_MARK = "[LOOK_FLOOR]"
EXPOSURE_KEY = "/rtx/post/tonemap/exposureTime"


def metal0_prop(stage, root: str) -> str:
    """Rebind one prop's material (UE4 MI_*, one Material, one Shader) to OmniPBR: base colour, normal and ORM textures
    and the Color tint kept, metallic 0, roughness = ORM.g. Dies unless there is exactly one material and shader with
    resolvable textures. Returns a one-line summary."""
    from pxr import Gf, Sdf, Usd, UsdShade

    prim = stage.GetPrimAtPath(root)
    mats = [p for p in Usd.PrimRange(prim) if p.IsA(UsdShade.Material)]
    if len(mats) != 1:
        raise SystemExit(f"{LOOK.MARK} metal-0 prop {root}: {len(mats)} Materials — need 1")
    shs = [UsdShade.Shader(c) for c in mats[0].GetChildren() if c.GetTypeName() == "Shader"]
    if len(shs) != 1:
        raise SystemExit(f"{LOOK.MARK} metal-0 prop {root}: {len(shs)} Shaders")

    def asset(n: str) -> str:
        i = shs[0].GetInput(n)
        v = i.Get() if i else None
        if v is None or not v.resolvedPath:
            raise SystemExit(f"{LOOK.MARK} metal-0 prop {root}: cannot resolve texture {n} ({v})")
        return v.resolvedPath

    bc, nm, orm = asset("BaseColor"), asset("Normal"), asset("AO_R_Rough_G_Metallic_B_")
    c = shs[0].GetInput("Color").Get() if shs[0].GetInput("Color") else None
    tint = (c[0], c[1], c[2]) if c is not None else (1.0, 1.0, 1.0)
    mpath = Sdf.Path(f"/World/AugMaterials/look_metal0_{Path(root).name}")
    mat = UsdShade.Material.Define(stage, mpath)
    s = UsdShade.Shader.Define(stage, mpath.AppendChild("Shader"))
    s.SetSourceAsset("OmniPBR.mdl", "mdl")
    s.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

    def tex(n: str, path: str, raw: bool) -> None:
        i = s.CreateInput(n, Sdf.ValueTypeNames.Asset)
        i.Set(Sdf.AssetPath(str(path)))
        i.GetAttr().SetMetadata("colorSpace", "raw" if raw else "sRGB")

    tex("diffuse_texture", bc, False)
    s.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
    tex("normalmap_texture", nm, True)
    s.CreateInput("enable_ORM_texture", Sdf.ValueTypeNames.Bool).Set(True)
    tex("ORM_texture", orm, True)
    s.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    s.CreateInput("metallic_texture_influence", Sdf.ValueTypeNames.Float).Set(0.0)
    s.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
    o = s.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(o)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(o)
    mat.CreateVolumeOutput("mdl").ConnectToSource(o)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return f"{Path(bc).name} × {tuple(round(x, 3) for x in tint)} · ORM {Path(orm).name} · metallic 0"


def exposure_s(ev: float) -> float:
    """Canonical exposure x 2^ev (s)."""
    return LOOK.TONEMAP[EXPOSURE_KEY] * 2.0 ** float(ev)


def set_exposure(ev: float) -> float:
    """Set the exposure to canonical x 2^ev (checked by readback). Returns the value (s)."""
    v = exposure_s(ev)
    LOOK._set_many({EXPOSURE_KEY: v})
    return v


def exposure_now() -> float:
    import carb

    return float(carb.settings.get_settings().get(EXPOSURE_KEY))


def probe_frame(grasp: dict, n_frames: int, fps: float) -> tuple[int, str]:
    """Probe frame index and its rule: grasp onset + probe_after_grasp_s, or the middle frame without a grasp."""
    if n_frames <= 0:
        raise SystemExit(f"{LOOK.MARK} probe: {n_frames} frames")
    g = (grasp or {}).get("onset_s")
    if g is None:
        if LOOK_REV["no_grasp_probe"] != "mid":
            raise SystemExit(f"{LOOK.MARK} probe: unknown no_grasp_probe {LOOK_REV['no_grasp_probe']!r}")
        return n_frames // 2, "no grasp → middle"
    return min(
        int(round((float(g) + LOOK_REV["probe_after_grasp_s"]) * fps)), n_frames - 1
    ), f"grasp {g:.2f} s + {LOOK_REV['probe_after_grasp_s']} s"


def probe_luma(rgb) -> tuple[float, float]:
    """RGB uint8 -> (BT.601 Y mean 0-255, clipped-highlight ratio Y >= clip_hi_y); same coefficients as cv2 YCrCb."""
    import numpy as np

    x = np.asarray(rgb)[..., :3].astype(np.float32)
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return float(y.mean()), float((y >= LOOK_REV["clip_hi_y"] - 0.5).mean())


def measured(luma: float, clip_hi: float, how: str) -> dict:
    """The episode line's look.measured field."""
    return {
        "luma_mean": round(luma, 2),
        "unit": "BT.601 Y 8 bit",
        "probe": f"{LOOK_REV['probe_cam']} · {how}",
        "floor": LOOK_REV["floor"],
        "ceiling": LOOK_REV["ceiling"],
        "clip_hi": round(clip_hi, 5),
        "rev": LOOK_REV["name"],
    }


def metal0_apply(stage, props) -> None:
    """Make metal0_props among the props placed by look_d.apply (/World/CollectLook/props/p<i>) metallic 0."""
    for name in LOOK_REV["metal0_props"]:
        hit = [i for i, p in enumerate(props) if Path(p[0]).name == name]
        if len(hit) != 1:
            raise SystemExit(f"{LOOK.MARK} metal-0 prop {name} found {len(hit)} times — need one in PROPS")
        print(
            f"{LOOK.MARK} metal-0 prop {name} ← {metal0_prop(stage, f'/World/CollectLook/props/p{hit[0]}')} · "
            f"rev {LOOK_REV['name']}",
            flush=True,
        )


def gate_problems(text: str) -> list[str]:
    """Session log -> a problem when the number of brightness-gate markers != drawn episodes (the render's
    frame-0 progress line render.writer.FRAME_HEAD 0/; a reference render draws 0 episodes and has 0 gates)."""
    from dexengine.record.render import writer as CW  # noqa: PLC0415

    bad = []
    n_ep = text.count(f"{CW.FRAME_HEAD}0/")
    n_gate = sum(1 for ln in text.splitlines() if ln.startswith((f"{FLOOR_MARK} D_exposure", f"{FLOOR_MARK} random")))
    if n_gate != n_ep:
        bad.append(f"brightness gate markers {FLOOR_MARK} {n_gate} != drawn episodes {n_ep} (rev {LOOK_REV['name']})")
    return bad


def d_post(rgb, how: str, ev: float | None = None) -> dict:
    """Look D post brightness on the drawn probe frame (exposure +ev stops, default d_ev). Below floor it adds
    rerender (next exposure; none at d_ev_max). Returns the look field."""
    import math

    R = LOOK_REV
    y, hi = probe_luma(rgb)
    y = round(y, 2)  # decide on the recorded value (measured.luma_mean), the one dist() reads
    ev = R["d_ev"] if ev is None else float(ev)
    low, steps = y < R["floor"], int(round((ev - R["d_ev"]) / R["d_ev_step"]))
    nxt = None
    if low and ev < R["d_ev_max"]:
        need = math.log(R["floor"] / max(y, 1e-3)) / math.log(R["ev_gain"])
        nxt = min(R["d_ev_max"], ev + math.ceil(need / R["d_ev_step"] - 1e-9) * R["d_ev_step"])
    # stop, why and how land in the episode line
    stop = (
        "post measurement · at or above the floor"
        if not low
        else f"post measurement · below the floor → re-render list (next +{nxt} stops)"
    )
    print(
        f"{FLOOR_MARK} D_exposure ev +{ev:g} stops · step {steps} · exposure {exposure_s(ev):.4f} s · luma {y:.1f} · "
        f"clipped {hi:.4f} · {stop} · {how} · rev {R['name']}",
        flush=True,
    )
    rr = {"next_ev": nxt, "why": f"luma {y:.1f} < {R['floor']}"} if low else None
    return {
        "exposure": {
            "ev": ev,
            "steps": steps,
            "exposure_s": round(exposure_s(ev), 6),
            "luma": round(y, 2),
            "clip_hi": round(hi, 5),
            "capped": low,
            "stop": stop,
            "trail": [(ev, round(y, 2), round(hi, 5))],
            "mode": "post",
            "rerender": rr,
        },
        "measured": measured(y, hi, how),
        "floor_redraws": 0,
    }


def r_post(rgb, how: str) -> dict:
    """Random-look post brightness on the probe frame drawn at canonical exposure. Outside [floor, ceiling] it adds
    rerender (spare deck cards from 0, LookRun.spare, new session). Returns the look field."""
    R = LOOK_REV
    y, hi = probe_luma(rgb)
    y = round(y, 2)  # decide on the recorded value, so the re-render list and dist() agree at the ±0.005 edge
    out = not (R["floor"] <= y <= R["ceiling"])
    stop = (
        "post measurement · in range" if not out else "post measurement · out of range → re-render list (spare card 0)"
    )
    print(
        f"{FLOOR_MARK} random luma {y:.1f} · clipped {hi:.4f} · redraws 0 · {stop} · {how} · rev {R['name']}",
        flush=True,
    )
    return {
        "measured": measured(y, hi, how),
        "floor_redraws": 0,
        "rerender": {"next_spare": 0, "why": f"luma {y:.1f} ∉ [{R['floor']}, {R['ceiling']}]"} if out else None,
    }


def settle_gate(rig, T: dict, extra: dict | None, gate, fps: float, tm: dict) -> tuple[dict, dict, object]:
    """Before the first frame: lighting settle (LS.settle, time in tm.settle), then the brightness gate (post, no
    pre-probe) only sets the exposure (gate None = look D +d_ev, 'random' = canonical). tap(i, cap) measures the drawn
    probe frame (d_post | r_post, filling extra.look in place, else a look_gate field).
    Returns (look_settle, extra, tap)."""
    import time

    t = time.perf_counter()
    look_settle = LS.settle(rig)
    tm["settle"] += time.perf_counter() - t
    steps = T["steps"]
    fr, how = probe_frame(T["grasp"], len(steps), fps)
    how = f"frame {fr}/{len(steps)} · {how}"
    if extra is not None and isinstance(extra.get("look"), dict):
        extra = dict(extra, look=dict(extra["look"]))
        g = extra["look"]
    else:
        extra = dict(extra or {}, look_gate={})
        g = extra["look_gate"]
    if LOOK_REV["d_gate"] != "post" or gate not in (None, "random"):
        raise SystemExit(
            f"{LOOK.MARK} brightness gate d_gate {LOOK_REV['d_gate']!r} · gate {gate!r} — post and None|random only"
        )
    rq = g.get("redraw") or {}  # re-render (looks.redraw): next look-D exposure
    ev = float(rq["d_ev"]) if gate is None and rq.get("kind") == "d_ev" else LOOK_REV["d_ev"]
    set_exposure(ev if gate is None else 0.0)

    def tap(i: int, cap: dict) -> None:
        if i == fr:
            g.update(d_post(cap["egoL"], how, ev) if gate is None else r_post(cap["egoL"], how), gate_s=0.0)

    return look_settle, extra, tap


# ---- re-render of listed episodes (new sessions) ---------------------------------------------------------------
# An episode whose render_meta is listed (random look.rerender, look D look.exposure.rerender) is drawn again in a new
# session (state changes inside one session leave ghosts in the RTX cache):
#   random  spare cards k = 0, 1, ... (LOOK_REV.random_tries; plan redraw kind=spare; LookRun.spare: room and
#           lighting axes only)
#   look D  next exposure = the list's next_ev (plan redraw kind=d_ev; d_post clears next_ev at d_ev_max)
# Stops when in range; otherwise keeps the attempt closest to the range (ties: the earlier one). Attempts not kept move
# to <out>/_redraw/<traj>/<look>/round<n>/try<j> (never deleted, not stored; n = redraw round of that slot).
# Markers: '[LOOK_REDRAW] try ...' per redraw session, '[LOOK_REDRAW] decision ...' per listed episode.
REDRAW_MARK = "[LOOK_REDRAW]"
REDRAW_DIR = "_redraw"
REDRAW_KINDS = ("spare", "d_ev")


def _look(meta: dict) -> dict:
    lk = meta.get("look")
    if not isinstance(lk, dict) or "name" not in lk:
        raise InputMissingError(f"{REDRAW_MARK} render_meta has no look field (brightness gate rev {LOOK_REV['name']})")
    return lk


def ask(meta: dict) -> dict | None:
    """render_meta -> the next re-render request {kind, value}, or None (not listed, or look D at its exposure cap)."""
    lk = _look(meta)
    if lk["name"] == "D":
        rr = (lk.get("exposure") or {}).get("rerender")
        return {"kind": "d_ev", "d_ev": float(rr["next_ev"])} if rr and rr.get("next_ev") is not None else None
    rr = lk.get("rerender")
    return {"kind": "spare", "k": int(rr["next_spare"])} if rr else None


def dist(meta: dict) -> float:
    """Distance to the brightness range (0 = inside): look D only has the floor (d_post), random looks [floor, ceiling]
    (r_post)."""
    lk, R = _look(meta), LOOK_REV
    m = lk.get("measured")
    if not isinstance(m, dict) or m.get("luma_mean") is None:
        raise InputMissingError(f"{REDRAW_MARK} no look.measured — the brightness gate did not measure this episode")
    y = float(m["luma_mean"])
    return max(0.0, R["floor"] - y) if lk["name"] == "D" else max(0.0, R["floor"] - y, y - R["ceiling"])


def redraw_plan_problems(plan: dict) -> list[str]:
    """The session plan's redraw field: a fixed session, one episode, and a kind that matches the look."""
    rq = plan.get("redraw")
    if rq is None:
        return []
    eps = plan.get("episodes") or []
    bad = []
    if plan.get("per_look") != "fixed" or len(eps) != 1:
        bad.append(f"redraw needs a fixed session with one episode ({plan.get('per_look')!r} · {len(eps)})")
    elif rq.get("kind") not in REDRAW_KINDS or (rq["kind"] == "d_ev") != (eps[0]["look"] == "D"):
        bad.append(f"redraw kind {rq.get('kind')!r} <-> look {eps[0]['look']!r} (look D = d_ev, random = spare)")
    elif rq["kind"] == "spare" and not 0 <= int(rq["k"]) < int(LOOK_REV["random_tries"]):
        bad.append(f"spare card {rq['k']} outside 0–{LOOK_REV['random_tries'] - 1}")
    elif rq["kind"] == "d_ev" and not LOOK_REV["d_ev"] < float(rq["d_ev"]) <= LOOK_REV["d_ev_max"]:
        bad.append(f"exposure +{rq['d_ev']} stops outside ({LOOK_REV['d_ev']}, {LOOK_REV['d_ev_max']}]")
    return bad


def _park(out: Path, ep: dict, rnd: Path, j: int) -> Path:
    src, dst = out / ep["traj"] / ep["look"], rnd / f"try{j}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f"{REDRAW_MARK} target {dst} already exists (an earlier run's attempt; not deleted)")
    src.rename(dst)
    return dst


def redraw_after(run, out: Path, pp: dict, tag: str, g: dict) -> dict:
    """After one drawn session (pp = one-episode plan, tag, its summary g): re-render while the episode is listed.
    run(tag, plan) runs one session and returns its summary. Returns the kept attempt's summary plus the redraw record
    (g unchanged when not listed)."""
    eps = g.get("episodes") or []
    if len(eps) != 1:
        return g  # nothing drawn (original off screen): nothing to measure
    ep = dict(pp["episodes"][0], traj=eps[0]["traj"], look=eps[0]["look"])
    d = out / ep["traj"] / ep["look"]
    meta = json.loads((d / "render_meta.json").read_text())
    rq = ask(meta)
    if rq is None:
        return g
    tries = [{"j": 0, "req": None, "luma": _look(meta)["measured"]["luma_mean"], "dist": dist(meta), "tag": tag}]
    par = out / REDRAW_DIR / ep["traj"] / ep["look"]
    rnd = par / f"round{len(list(par.glob('round*')))}"
    sums = [g]
    while rq is not None and tries[-1]["dist"] > 0:
        j = len(tries)
        _park(out, ep, rnd, j - 1)
        t = f"{tag}.r{j}"
        sums.append(run(t, dict(pp, per_look="fixed", redraw=dict(rq, attempt=j))))
        meta = json.loads((d / "render_meta.json").read_text())
        tries.append({"j": j, "req": rq, "luma": _look(meta)["measured"]["luma_mean"], "dist": dist(meta), "tag": t})
        print(
            f"{REDRAW_MARK} try {j} {ep['traj']}/{ep['look']} · {rq['kind']} {rq.get('k', rq.get('d_ev'))} · "
            f"luma {tries[-1]['luma']} · {'in range' if tries[-1]['dist'] == 0 else 'out of range'}",
            flush=True,
        )
        if rq["kind"] == "spare":
            rq = {"kind": "spare", "k": j} if j < int(LOOK_REV["random_tries"]) else None
        else:
            rq = ask(meta)
    best = min(range(len(tries)), key=lambda k: (tries[k]["dist"], k))
    last = len(tries) - 1
    if best != last:
        _park(out, ep, rnd, last)
        (rnd / f"try{best}").rename(d)
    # why lands in the redraw record
    why = (
        "in range"
        if tries[best]["dist"] == 0
        else f"out of range to the end → closest (distance {tries[best]['dist']:.1f})"
    )
    print(f"{REDRAW_MARK} decision {ep['traj']}/{ep['look']} · attempts {last} · kept try {best} · {why}", flush=True)
    rec = {
        "traj": ep["traj"],
        "look": ep["look"],
        "tries": tries,
        "kept": best,
        "why": why,
        "rule": LOOK_REV["name"],
        "at": str(rnd),
        "exhausted": tries[-1]["dist"] > 0,
    }
    return dict(sums[best], redraw=[rec], redraw_sessions=last)


def redraw_problems(out: Path, got: dict) -> list[str]:
    """Listed episodes left over: every episode of summary got whose render_meta is listed needs a redraw record that
    is exhausted, and every attempt of a record needs its session log (session.<tag>.log)."""
    recs = {(r["traj"], r["look"]): r for r in got.get("redraw") or []}
    bad = []
    for e in got.get("episodes") or []:
        lk = _look(json.loads((out / e["traj"] / e["look"] / "render_meta.json").read_text()))
        listed = (lk.get("exposure") or {}).get("rerender") if lk["name"] == "D" else lk.get("rerender")
        r = recs.get((e["traj"], e["look"]))
        if listed and r is None:
            bad.append(
                f"{REDRAW_MARK} {e['traj']}/{e['look']} listed for re-render ({listed.get('why')}) but no redraw record"
            )
        elif listed and not r["exhausted"]:
            bad.append(f"{REDRAW_MARK} {e['traj']}/{e['look']} still listed after {len(r['tries']) - 1} attempts")
    for r in recs.values():  # each attempt's session log (earlier rounds are not counted)
        miss = [t["tag"] for t in r["tries"][1:] if not (out / f"session.{t['tag']}.log").is_file()]
        if miss:
            bad.append(f"{REDRAW_MARK} {r['traj']}/{r['look']} attempt session logs missing {miss[:3]}")
    return bad
