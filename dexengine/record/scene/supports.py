"""Record render, look D item 6: visible support discs plus a pillar down to the supporting surface (visual only;
render and reference sessions only).
The floating discs of the training and judge envs stay physics-only (coords.disc_visible: invisible); the disc table,
physics and judging are unchanged.
On desk scenes the env already builds the stands as visible pillars (from the desk top); their bottom touches the
supporting surface, so no pillar is added here.

Discs: the one reader of the disc table is object_pin.support_disc_rows (reset event params); only the clip's real
  discs (on) get MakeVisible and a material (look_d.MDL[SUPPORT_MDL], local mirror). Check (measured vs stated): a
  placed disc centre must equal the table centre (top - thickness/2) plus the world shift (render.body world_shift)
  within SUPPORT_POS_TOL_M, or the session dies.
Pillars: a cylinder from the disc bottom down to the supporting surface (radius = disc radius x PILLAR_R_FRAC, clamped
  to [PILLAR_R_MIN, PILLAR_R_MAX]). Surface = the desk top (render.rig.Rig.desk_local) when the scene has a desk and the
  disc centre is inside the desk top, else the floor (env z 0). No collision or rigid-body API.
Marker ``[COLLECT_SUPPORT] supports shown n/K · pillars n …`` (0/0 for a scene without discs); session.launch requires
it in the session log.
"""

from __future__ import annotations

from pathlib import Path

from dexengine.core.errors import InputMissingError

MARK = "[COLLECT_SUPPORT]"
SHOWN = f"{MARK} supports shown "  # head of the support marker line (printed by show, required by log_problems)
SUPPORT_POS_TOL_M = 2e-3
PILLAR_R_FRAC, PILLAR_R_MIN, PILLAR_R_MAX = 0.18, 0.012, 0.04
ROOT = "/World/CollectLook/supports"


def log_problems(text: str) -> list[str]:
    return [] if SHOWN in text else [f"support marker '{SHOWN}' missing"]


def ground_z(x: float, y: float, desk: dict | None) -> tuple[float, str]:
    """(env frame) pillar bottom z and its name: the desk when the scene has a desk and the point is inside the desk
    top, else the floor (0)."""
    on_desk = desk and not desk.get("aerial_desk_dz") and desk["near_x"] <= x <= desk["far_x"]
    if on_desk and abs(y - desk["cy"]) <= desk["half_y"]:
        return float(desk["top_z"]), "desk"
    return 0.0, "floor"


def show(raw_env, desk: dict | None, shift_z: float = 0.0) -> dict:
    """Call after the caller has written the first state; one-clip sessions only. desk = Rig.desk() (passed in the env
    frame) · shift_z = world shift (m)."""
    import torch

    import omni.usd
    from pxr import Gf, UsdGeom

    from dexengine.record.looks import look_d as LOOK
    from dexengine.record.looks import usd_bind as UB
    from dexengine.rl.tasks.scene import object_pin as OP

    if raw_env.num_envs != 1:
        raise SystemExit(f"{MARK} supports: {raw_env.num_envs} envs (one-clip sessions only)")
    eids = torch.tensor([0], device=raw_env.device)
    rows = OP.support_disc_rows(raw_env, eids, "right")  # on · x · y · top · r do not depend on the side (own does)
    if rows is None:
        print(f"{SHOWN}0/0 · pillars 0 · scene without discs", flush=True)
        return {"visible": 0, "slots": 0, "pillars": 0}
    st, stage = OP.support_disc_state(raw_env), omni.usd.get_context().get_stage()
    origin = raw_env.scene.env_origins[0].detach().cpu().numpy().astype(float)
    mdl = LOOK.local(LOOK.MDL[LOOK.SUPPORT_MDL])
    UsdGeom.Xform.Define(stage, ROOT)
    shown = []
    for k, name in enumerate(st["names"]):
        if not bool(rows["on"][0, k]):
            continue
        hk = float(rows["h"][0, k])  # pillar height: disc thickness · stand height (from desk top)
        want = [float(rows["x"][0, k]), float(rows["y"][0, k]), float(rows["top"][0, k]) - 0.5 * hk + shift_z]
        got = (raw_env.scene[name].data.root_pos_w[0].detach().cpu().numpy().astype(float) - origin).tolist()
        err = max(abs(a - b) for a, b in zip(got, want, strict=True))
        if err > SUPPORT_POS_TOL_M:
            raise SystemExit(f"{MARK} disc {name} pose {got} != table centre + world shift {want} (error {err:.4f} m)")
        path = raw_env.scene[name].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        path = path.replace("env_.*", "env_0")
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise InputMissingError(f"{MARK} disc prim {path} missing")
        UsdGeom.Imageable(prim).MakeVisible()
        r = float(rows["r"][0, k])
        bottom = got[2] - 0.5 * hk
        gz, gname = ground_z(got[0], got[1], desk)
        pil = None
        if bottom - gz > 1e-3:
            pr = min(PILLAR_R_MAX, max(PILLAR_R_MIN, PILLAR_R_FRAC * r))
            cyl = UsdGeom.Cylinder.Define(stage, f"{ROOT}/pillar_{k}")
            cyl.CreateRadiusAttr(pr)
            cyl.CreateHeightAttr(bottom - gz)
            cyl.CreateAxisAttr("Z")
            UsdGeom.Xformable(cyl).AddTranslateOp().Set(
                Gf.Vec3d(got[0] + origin[0], got[1] + origin[1], (bottom + gz) / 2 + origin[2])
            )
            pil = {"radius_m": round(pr, 4), "height_m": round(bottom - gz, 4), "ground": gname}
        for pp in [path] + ([f"{ROOT}/pillar_{k}"] if pil else []):
            if not UB.bind_mdl(stage, pp, f"look_support_{Path(pp).name}", mdl, project=True):
                raise SystemExit(f"{MARK} {pp} material binding failed")
        shown.append(
            {"slot": name, "r_m": round(r, 4), "center": [round(v, 4) for v in got], "pos_err_m": err, "pillar": pil}
        )
    n_p = sum(1 for s in shown if s["pillar"])
    pil = [(s["pillar"]["ground"], s["pillar"]["height_m"]) for s in shown if s["pillar"]]
    print(
        f"{SHOWN}{len(shown)}/{len(st['names'])} · pillars {n_p} {pil} · material {Path(mdl).name} · "
        f"max pose error {max([s['pos_err_m'] for s in shown], default=0):.1e} m",
        flush=True,
    )
    return {
        "visible": len(shown),
        "slots": len(st["names"]),
        "pillars": n_p,
        "material": Path(mdl).name,
        "discs": shown,
    }
