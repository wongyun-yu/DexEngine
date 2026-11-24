"""The --probe-steps branch of eval_record.py (the variant smoke): no checkpoint, no policy — zero residual action for
K control steps on eval batch 0, then probe.json (each active clip's object motion from the reset · its support disks
as posed in PhysX vs the motion's table · the PhysX object masses). Ends the process (os._exit(0)).

Runs only inside the FlashDexRetarget checkout: eval_record.py imports it by its bare name (this folder is
sys.path[0] when eval_record.py runs by path) after the Isaac AppLauncher has started, so torch and hydra.utils are
imported here only after that."""

from __future__ import annotations

import json
import os
import sys

import torch
from hydra.utils import instantiate


def _probe_rows(
    mids, act_m, names, lib, sides, pos, k6, mass, dnames, tf, origins, support_park_position, sd, radii, rows
):
    """Per active clip: each side object pose at reset and after the probe steps, each support disc vs the expected."""
    for m in sorted(set(mids[act_m].tolist())):
        e = int(((mids == m) & act_m).nonzero()[0, 0])
        name = names[m]
        st = int(lib.length_starts[m])
        r = {"env": e, "sides": {}}
        for j, side in enumerate(sides):
            p = pos[e, :, j].double()
            ref = lib.obj_trans[side][st : st + k6 + 1].double().cpu()  # frames 0..6 (env-local)
            r["sides"][side] = {
                "reset_minus_ref0_mm": round(float((p[0] - ref[0]).norm()) * 1e3, 3),
                "move_0p1s_mm": round(float((p[k6] - p[0]).norm()) * 1e3, 3),
                "dz_0p1s_mm": round(float(p[k6, 2] - p[0, 2]) * 1e3, 3),
                "max_move_all_mm": round(float((p - p[0]).norm(dim=-1).max()) * 1e3, 3),
                "ref_move_0p1s_mm": round(float((ref[k6] - ref[0]).norm()) * 1e3, 3),
                "mass_kg": {n: round(float(v[e]), 6) for n, v in mass[side].items()},
            }
        want = lib.support_disks[m].cpu().double() if lib.support_disks.numel() else torch.zeros(0, 5)
        disks = []
        for k, row in enumerate(dnames):
            for jr, n in enumerate(row):
                x = tf[n][e, :3].double() - origins[e].double()
                park = torch.tensor(support_park_position(sd, k, jr), dtype=torch.float64)
                if (x - park).norm() < 1e-4:
                    continue
                w = want[k] if k < len(want) else None
                if w is None or float(w[3]) <= 0:
                    disks.append({"k": k, "bucket": jr, "posed_at": [round(float(v), 4) for v in x], "expected": None})
                    continue
                exp = torch.tensor([float(w[0]), float(w[1]), float(w[2]) - float(sd.height) / 2], dtype=torch.float64)
                jb = min(sum(rr < float(w[3]) + float(sd.radius_margin) - 1e-6 for rr in radii), len(radii) - 1)
                disks.append(
                    {
                        "k": k,
                        "bucket": jr,
                        "bucket_expected": jb,
                        "radius_data": round(float(w[3]), 4),
                        "radius_disk": radii[jr],
                        "z_top_data": round(float(w[2]), 4),
                        "err_mm": round(float((x - exp).norm()) * 1e3, 3),
                    }
                )
        n_want = int((want[:, 3] > 0).sum()) if len(want) else 0
        r["disks"] = disks
        r["disks_expected"] = n_want
        rows[name] = r


def _say_rows(rows, say):
    """One PROBE line per clip."""
    for name, r in rows.items():
        say(
            f"PROBE {name} · "
            + " · ".join(
                f"{s} move {v['move_0p1s_mm']:.1f} mm dz {v['dz_0p1s_mm']:+.1f} mm (ref {v['ref_move_0p1s_mm']:.1f}) "
                f"reset-ref0 {v['reset_minus_ref0_mm']:.1f} mm mass "
                f"{'/'.join(f'{x:.3f}' for x in v['mass_kg'].values())} kg"
                for s, v in r["sides"].items()
            )
            + f" · disks posed {len(r['disks'])}/{r['disks_expected']} "
            f"err max {max([d.get('err_mm') or 0 for d in r['disks']] or [0]):.2f} mm "
            f"bucket ok {all(d.get('bucket') == d.get('bucket_expected') for d in r['disks'])}"
        )


def probe(a, cfg, eval_env, OUT, ov, motion, var_name, var_sha, say) -> None:
    """Zero-action probe of eval batch 0 → <OUT>/probe.json and one PROBE line per clip; exits the process."""
    raw = eval_env.unwrapped
    cmd = raw.command_manager.get_term(cfg.eval.command_name)
    lib = cmd.motion_lib
    names = [f.rsplit("#", 1)[-1] for f in lib.motion_files]
    sides = list(cmd._side_list)
    cbs = [instantiate(c) for c in cfg.eval.callbacks if "rollout_record" not in str(c.get("_target_", ""))]
    setup_cb = next(c for c in cbs if hasattr(c, "set_batch"))
    setup_cb.set_batch(0, raw.eval_k, raw.eval_num_motions)
    for c in cbs:
        if hasattr(c, "on_eval_setup"):
            c.on_eval_setup(raw)
    with torch.no_grad():
        eval_env.reset()
        o = raw.scene.env_origins[:, None, :]
        pos = [(cmd.sim_obj_trans_w - o).float().cpu().clone()]
        act = torch.zeros(raw.num_envs, raw.action_manager.total_action_dim, device=cfg.device)
        for _ in range(a.probe_steps):
            eval_env.step(act)
            pos.append((cmd.sim_obj_trans_w - o).float().cpu().clone())
    pos = torch.stack(pos, 1)  # (E, K+1, S, 3) env-local · entry 0 = the reset state
    act_m = setup_cb.active_mask.cpu()
    mids = cmd.motion_ids.cpu()
    sd = cfg.commands.object.support_disks
    dnames = list(cmd.cfg.object.support_disks.entity_names or [])
    tf = {n: raw.scene[n]._obj.root_physx_view.get_transforms().cpu() for row in dnames for n in row}  # (E, 7) world
    origins = raw.scene.env_origins.cpu()
    from envs.scene._support_disk_setup import support_park_position

    radii = [float(r) for r in sd.radii]
    mass = {
        s: {n: raw.scene[n]._obj.root_physx_view.get_masses().reshape(-1).cpu() for n in cmd.obj_entity_names(s)}
        for s in sides
    }
    k6 = min(6, a.probe_steps)  # 6 control steps = 0.1 s at 60 Hz
    rows = {}
    _probe_rows(
        mids, act_m, names, lib, sides, pos, k6, mass, dnames, tf, origins, support_park_position, sd, radii, rows
    )
    out = {
        "probe_steps": a.probe_steps,
        "window_steps": k6,
        "variant": var_name,
        "variant_sha256": var_sha,
        "overrides": [o.replace(motion, "<motion.pt>") for o in ov],
        "num_envs": int(raw.num_envs),
        "clips": rows,
    }
    with open(OUT / "probe.json", "w") as fh:
        json.dump(out, fh, indent=1)
    _say_rows(rows, say)
    say(f"PROBE_DONE {len(rows)} clips → {OUT / 'probe.json'}")
    sys.stdout.flush()
    os._exit(0)
