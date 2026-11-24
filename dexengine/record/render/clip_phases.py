"""The per-clip render loop (render_clip, called by session.entry --stage render_clip) and its head phases: the augment
trajectory choice (choose_phase), the original's camera-only load (orig_camera) and field-of-view check
(orig_fov_phase). The plan contract and the fixed per-session state live in render.clip_session. Part of the render
fingerprint (record.cache.RENDER_CODE_FILES)."""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record.looks import look_run as LRN
from dexengine.record.looks import rev as LREV
from dexengine.record.render.clip_session import (
    HEAD_PHASES,
    LOAD_METHOD,
    MARK,
    check_view,
    load_plan,
    restore,
    traj_row,
)


def _head_phase(plan, S, loaded, summ, raw_env, LR, T):
    """Session head: trajectory choice, the fixed-session original calibration, or the original off-screen rule."""
    from dexengine.record.render import episode as CR

    if plan.get("choose"):
        plan["episodes"], LR = choose_phase(S, plan, loaded, summ, raw_env)
        summ["episodes_plan"] = plan["episodes"]
        if LR is not None:
            LRN.save_allow(LR, plan, summ["looks"])
    # augment drawing session: original for calibrate only
    elif plan.get("per_look") == "fixed" and plan["round"] != "orig":
        _ev, _lr = orig_camera(S, plan, loaded, summ, raw_env, measure=False)
        summ["episodes_plan"] = plan["episodes"]
    elif plan["round"] == "orig":  # original off-screen rule (canonical): not visible = render skipped
        tm0 = dict.fromkeys(CR.TM_KEYS, 0.0)
        T = orig_fov_phase(S, plan, loaded, summ, tm0)
        summ["trajs"].append(traj_row(T, tm0, plan["trajs"][0].get("role")))
        summ["episodes_plan"] = plan["episodes"]
    return LR, T


def _draw_episodes(todo, skip, T, summ, raw_env, S, loaded, traj_of, LR, out, plan, rnd, boot_s, t_all, prev, record):
    """Draw the session's episodes in turn (reset and load only when the trajectory changes)."""
    from dexengine.record.render import episode as CR

    for i, ep in ((i, e) for i, e in enumerate(todo) if i not in skip):
        tm = dict.fromkeys(CR.TM_KEYS, 0.0)
        if T is None or T["clip"] != ep["traj"]:
            if T is not None:
                summ["prim_oracle"].append(restore(raw_env, T, S.prims0))
            T = CR.load_traj(S, loaded.index(ep["traj"]), ep["traj"], traj_of[ep["traj"]]["traj"], tm)
            summ["trajs"].append(traj_row(T, tm, traj_of[ep["traj"]].get("role")))
        spec, info, gate = None, None, None
        if ep["look"] != "D":
            LREV.set_exposure(0.0)  # random = canonical exposure
            LR = LR or LRN.look_run_cls()(S, out / "_look")
            spec, info, rnd = *LR.spec_for(T, ep, plan), True
            if "allow" in info:
                summ["looks"] = info.pop("allow")

            gate = "random"
        elif rnd:
            raise InputMissingError(
                f"{MARK} episode {i} look D comes after a random look: there is no way back to look D"
            )
        extra = {
            "look": dict(
                name=ep["look"],
                spec=spec,
                seed_key=plan.get("look_key"),
                apply=info,
                **({"redraw": plan["redraw"]} if plan.get("redraw") else {}),
            ),
            "episode_index": ep.get("index", i),
            "round": plan["round"],
        }
        cur = CR.render_episode(
            S, T, out / ep["traj"] / ep["look"], tm, extra, boot_s, t_all, prev[0] if prev else None, gate=gate
        )  # joins the previous episode's write inside
        if prev is not None:
            record(*prev, "overlap depth 1")
        prev = (cur, i, ep, info)
    return T, prev


def _loaded_trees(plan, clip_names):
    """The loaded trajectories must be exactly the plan's (as tree names); random looks need the asset root."""
    from dexengine.core import paths as P

    names = [t["name"] for t in plan["trajs"]]
    loaded = [P.tree_name(c) for c in clip_names]  # tree names, whatever the loader's name shape
    if sorted(loaded) != sorted(names):
        raise SystemExit(
            f"{MARK} the session loaded clips {sorted(clip_names)} != plan trajectories {sorted(names)}: "
            "the list override failed"
        )
    if plan["random_per_traj"] > 0:
        from dexengine.record.looks.augvis import scene as SC

        SC.assets_root()  # check the random-look asset root (AUGVIS_ASSETS, no default) before the first render
    return loaded


def render_clip(
    wrapped, policy_fn, traj_mgr, num_clips, clip_names, raw_env, checkpoint_path, out_dir, *_a, args_cli=None, **_kw
):
    """One session = one clip: the plan's episodes in turn (reset and load only when the trajectory changes). Contract
    in the module doc."""
    from dexengine.record.render import episode as CR
    from dexengine.record.session import persist_kit as PK

    boot_s, clk = PK.boot_s(), time.perf_counter
    t_all = clk()
    plan = load_plan(os.environ["COLLECT_PLAN"])
    loaded = _loaded_trees(plan, clip_names)
    traj_of = {t["name"]: t for t in plan["trajs"]}
    out = Path(plan["out"])
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / plan.get("part_json", "render_clip.json")  # partial summary (a new session per look)
    summ = {
        "version": 1,
        "clip": plan["clip"],
        "out": str(out),
        "plan": os.environ["COLLECT_PLAN"],
        "load": {
            "method": LOAD_METHOD,
            "sequence_dir": getattr(args_cli, "sequence_dir", None),
            "seq_list": os.environ.get("DEXENGINE_SEQ_LIST"),
            "clip_names": list(clip_names),
        },
        "episodes": [],
        "trajs": [],
        "prim_oracle": [],
        "camera_off": None,
        "looks": None,
    }
    S = CR.Session(wrapped, traj_mgr, raw_env, args_cli, checkpoint_path)
    T, LR, prev, rnd = None, None, None, False
    LR, T = _head_phase(plan, S, loaded, summ, raw_env, LR, T)

    # summary after the write join (plan order, FIFO depth 1)
    def record(p, i: int, ep: dict, info, mode: str) -> None:
        r = p.finalize(mode)
        summ["episodes"].append(dict(index=i, traj=ep["traj"], look=ep["look"], look_apply=info, **r))
        summ["camera_off"] = S.off
        jpath.write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=float))

    ph = plan.get("per_look")
    if ph == "measure" and T is not None and any(e["look"] != "D" for e in plan["episodes"]):  # orig head: allow
        LR = LRN.look_run_cls()(S, out / "_look")
        summ["looks"] = LR.measure(T, plan)
        LRN.save_allow(LR, plan, summ["looks"])
    todo, skip = PK.resume(jpath, summ, [] if ph in HEAD_PHASES else plan["episodes"])  # worker retry
    if any(e["look"] != "D" for e in todo):  # random look in a drawing session = head session allow file only
        if ph != "fixed":
            raise SystemExit(f"{MARK} random look drawn in a per_look={ph!r} session: fixed only")
        LR = LRN.look_run_cls()(S, out / "_look")
        summ["looks"] = LRN.adopt(LR, plan)
    print(f"{MARK} this session draws {len(todo)}/{len(plan['episodes'])} · per_look {ph}", flush=True)
    T, prev = _draw_episodes(
        todo, skip, T, summ, raw_env, S, loaded, traj_of, LR, out, plan, rnd, boot_s, t_all, prev, record
    )
    if prev is not None:
        record(*prev, "overlap depth 1 · last")  # join the last write before the DONE line and the session end
    if T is not None:
        summ["prim_oracle"].append(restore(raw_env, T, S.prims0))
    n, m = sum(1 for e in summ["episodes"] if e["ok"]), len(todo)
    summ.update(
        n=n,
        m=m,
        ok=n == m,
        total_s=round(clk() - t_all, 1),
        maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
    )
    jpath.write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=float))
    print(f"{MARK} DONE episodes={n}/{m} · {out}", flush=True)
    PK.finish_render(n == m, wrapped)


def _probe_candidates(trajs, cands, need, S, loaded, plan, ev):
    """Probe the variant trajectories in rank order until enough strictly visible candidates are found."""
    from dexengine.record import episodes as EPI
    from dexengine.record.render import episode as CR

    for r, t in enumerate(trajs[1:], 1):
        if sum(c["vis_strict"] for c in cands[1:]) >= need:
            break
        ev = CR.probe_view(S, loaded.index(t["name"]), t["name"], t["traj"])
        cands.append(
            {
                "name": t["name"],
                "role": "aug",
                "rank": r,
                "grasp_visible": EPI.kin_visible(plan, ev["grasp_visible"]),
                "vis_strict": EPI.kin_visible(plan, ev["vis_strict"]),
                "vis_in_frame": ev["vis_in_frame"],
                "vis_ladder": ev.get("vis_ladder"),
                "ego_view": ev,
            }
        )
    return ev


def choose_phase(S, plan: dict, loaded: list[str], summ: dict, raw_env) -> list[dict]:
    """Trajectory choice (record.episodes.choose): load the original to fix the camera (load_traj -> calibrate) and
    remove it, then measure the candidate variants in turn (render.episode.probe_view, no render). Stops when n
    variants are visible (the original is not chosen). Returns the chosen trajectories' episode plan; summ['choice']
    gets every candidate's ego_view and the choice."""
    from dexengine.record import episodes as EPI
    from dexengine.record.render import fov as FOW  # numpy: after AppLauncher only ([PRE_APP])
    from dexengine.record.render import framing as CF

    trajs, n = plan["trajs"], int(plan["choose"]["n"])
    t0 = time.perf_counter()
    orig = trajs[0]
    ev, LR = orig_camera(S, plan, loaded, summ, raw_env, measure=True)
    # owner of the original off-screen verdict (render.fov); None = old path
    own = plan["choose"].get("orig_fov_owner")
    # kinematic = no choice (measured = ego_view)
    gv = EPI.kin_visible(plan, FOW.orig_fov_decide(ev["grasp_visible"], own))
    if own is not None:
        print(
            f"{FOW.ORIG_FOV_OWNER_MARK} aug · owner = off screen (orig round verdict, spec row 0) · this session "
            "measured "
            + ("visible -> differs, the owner wins (row 0 chosen)" if ev["grasp_visible"] else "not visible (same)"),
            flush=True,
        )
    cands = [
        {
            "name": orig["name"],
            "role": "orig",
            "rank": 0,
            "grasp_visible": gv,
            "vis_strict": ev["vis_strict"],
            "vis_in_frame": ev["vis_in_frame"],
            "ego_view": ev,
        }
    ]
    need = n + (0 if EPI.orig_kept(gv) else 1)  # original off screen = up to the row-0 variant (ORIG_FOV_RULE)
    ev = _probe_candidates(trajs, cands, need, S, loaded, plan, ev)
    row0, chosen, meta = EPI.choose_rows(cands, n)
    if own is not None:  # episode meta traj_choice.orig_fov keeps the owner and the measured value
        meta["orig_fov"] = dict(meta["orig_fov"], owner=own, measured=ev["grasp_visible"])
    meta.update(
        ego_view={c["name"]: c["ego_view"] for c in cands},
        unprobed=[t["name"] for t in trajs[len(cands) :]],
        choose_s=round(time.perf_counter() - t0, 1),
    )
    summ["choice"] = meta
    of = meta["orig_fov"]
    seen = "visible" if meta["orig_visible"] else "not visible"
    print(
        f"{MARK} trajectory choice {EPI.CHOOSE_RULE} · original at grasp {seen} · "
        f"measured candidates {len(cands) - 1}/{len(trajs) - 1} · chosen {chosen} · visible {meta['visible_chosen']}/"
        f"{len(chosen)} · {meta['choose_s']} s",
        flush=True,
    )
    print(
        f"{EPI.ORIG_FOV_MARK} aug · "
        + (
            "original visible -> rows 1-3 (row 0 = orig round)"
            if of["orig_kept"]
            else f"original not visible -> row 0 = {row0} ({of['row0_fill']} · {EPI.FILL_RULE}) · rows 1- {chosen}"
        )
        + f" · visible trajectories {of['visible_rows']}/{of['rows']} · trigger "
        f"{CF.ORIG_OUT_RULE['name']} · variant visible {CF.VIS_RULE['name']}",
        flush=True,
    )
    return EPI.aug_episodes(orig["name"], row0, chosen, int(plan["deck"]["n_rows"])), LR


def orig_camera(S, plan: dict, loaded: list[str], summ: dict, raw_env, measure: bool) -> tuple[dict, object]:
    """Augment session head, shared by the choice session (choose_phase) and the later sessions of a new session per
    look (per_look=fixed): load the original to fix the camera (load_traj -> calibrate; later trajectories check this
    head angle with check_view); with a random look, measure allow on the original look D (LookRun.measure), then
    remove the original (restore, scene prim oracle). The original is not drawn here, so its idle hand is not
    measured (load_traj camera_only). Returns (original ego_view, LookRun | None)."""
    from dexengine.record.render import episode as CR

    orig = plan["trajs"][0]
    tm = dict.fromkeys(CR.TM_KEYS, 0.0)
    T = CR.load_traj(S, loaded.index(orig["name"]), orig["name"], orig["traj"], tm, camera_only=True)  # discarded
    # same measuring path as candidates and render
    ev = CR.ego_view_meta(S.cal, check_view(S.rig, T["groups"]), T["grasp"])
    LR = None
    # allow from the original (before removal, look D): head sessions only
    if measure and plan["random_per_traj"] > 0:
        LR = LRN.look_run_cls()(S, Path(plan["out"]) / "_look")
        summ["looks"] = LR.measure(T, plan)
    summ["prim_oracle"].append(restore(raw_env, T, S.prims0))
    return ev, LR


def orig_fov_phase(S, plan: dict, loaded: list[str], summ: dict, tm: dict) -> dict:
    """Original off-screen rule of the orig round (record.episodes.ORIG_FOV_RULE): load the original to fix the
    camera (load_traj -> calibrate) and measure grasp visibility (render.framing.VIS_RULE). Not visible: empty
    plan.episodes (render skipped; the augment round fills rows 0-3). Record summ['orig_fov'] · marker [ORIG_FOV].
    Returns the loaded trajectory context (when visible the render loop uses it)."""
    from dexengine.record import episodes as EPI
    from dexengine.record.render import episode as CR
    from dexengine.record.render import fov as FOV
    from dexengine.record.render import framing as CF

    orig = plan["trajs"][0]
    T = CR.load_traj(S, loaded.index(orig["name"]), orig["name"], orig["traj"], tm)
    # first trajectory = calibrate framing's after (= render meta)
    ev = CR.ego_view_meta(S.cal, None, T["grasp"])
    # kinematic = never skipped (measured = ego_view)
    kept = EPI.kin_visible(plan, EPI.orig_kept(ev["grasp_visible"]))
    summ["orig_fov"] = dict(
        rule=EPI.ORIG_FOV_RULE,
        orig_kept=kept,
        **({"kin": EPI.KIN_RULE} if plan.get("kin") else {}),
        trigger=dict(CF.ORIG_OUT_RULE),
        ego_view=ev,
        deferred_to="aug rows 0-3" if not kept else None,
        view=None if kept else FOV.view_record(S.cal, T, orig["name"]),
    )  # cell choice input
    if not kept:
        plan["episodes"] = []
    print(
        f"{EPI.ORIG_FOV_MARK} orig · "
        + (
            "original visible -> original drawn"
            if kept
            else "original not visible -> render skipped · the augment round fills rows 0-3"
        )
        + f" · grasp in frame {ev.get('grasp_in_frame')} · trigger {CF.ORIG_OUT_RULE['name']}",
        flush=True,
    )
    return T
