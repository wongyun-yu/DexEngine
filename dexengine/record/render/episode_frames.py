"""Frame loop and render_meta of one episode render (callers: render.episode.render_episode). The constants of the
render (FPS, FIRST_WARMUP, FRAME_RENDERS) and ego_view_meta stay owned by render.episode; these functions import them
when called (render.episode imports this module at load)."""

from __future__ import annotations

import os

from dexengine.record.render import writer as CW


def capture_frames(steps, clk, write, rig, heads, tm, tap, W, tally, per, cal, hv, hv_sides, first_cap):
    """Frame loop of one episode: per step write the state, capture (first FIRST_WARMUP renders, later FRAME_RENDERS),
    feed the background writer and the visibility / head-view tallies. Returns the first capture time."""
    from dexengine.record.render.episode import FIRST_WARMUP, FRAME_RENDERS  # this module's caller (loaded)

    for i, st in enumerate(steps):
        c = clk()
        write(st)
        cap = rig.capture(FIRST_WARMUP if i == 0 else FRAME_RENDERS)
        heads.append(rig.head_readback())
        if i == 0:
            first_cap = clk() - c
        tm["capture"] += clk() - c
        tap(i, cap)
        # mp4 x3 and depth conversion on the background thread (waits when the queue is full)
        W.add(cap)
        c = clk()
        eye = rig.pose("collect_ego_left")
        tally.add({g: per[g][st] for g in per}, eye[0], eye[1], cap["depth_m"], cal["zed"])
        tm["visible"] += clk() - c
        if hv is not None:
            c = clk()
            hv.add({s: v["pts"][st] for s, v in hv_sides.items()}, eye[0], eye[1], cal["zed"])
            tm["headview"] += clk() - c
        if i % 60 == 0:
            print(f"{CW.FRAME_HEAD}{i}/{len(steps)} (step {st})", flush=True)
    return first_cap


def episode_meta(T, look_settle, steps, cal, rig, heads, vis_meta, S, extra):
    """render_meta of one episode (look, camera, head, body, idle hand, visibility, views, tree, hand USD, checkpoint,
    extra fields)."""
    from dexengine.record.render import framing as CF
    from dexengine.record.render.episode import (  # this module's caller (loaded)
        FIRST_WARMUP,
        FPS,
        FRAME_RENDERS,
        ego_view_meta,
    )
    from dexengine.record.store import meta as HU  # numpy: after AppLauncher only ([PRE_APP])

    meta = dict(
        render_preset=T["look"]["render_preset"],
        render_look=T["look"],
        look_settle=look_settle,
        clip=T["clip"],
        traj=T["traj"],
        n_states=int(T["st"].shape[0]),
        step_dt=T["step_dt"],
        fps=FPS,
        frame_step=steps,
        counts=None,
        depth_frames=None,
        depth_file=None,  # filled after the write join
        first_warmup=FIRST_WARMUP,
        frame_renders=FRAME_RENDERS,
        camera=cal,
        head=rig.head_meta(heads),
        body=T["body"],
        idle_hand=T["idle"],
        visibility=vis_meta,
        extras_visual=T["extras"],
        camera_off=S.off,
        camera_view_check=T["view"],
        ego_view=ego_view_meta(cal, T["view"], T["grasp"]),
        exo_view=CF.exo_view(cal, T["groups"], rig.view_ref["P"]),
        tree=os.environ.get("COLLECT_TREE", ""),
        **{HU.HAND_USD_KEY: HU.hand_usd_meta()},
        checkpoint=S.ckpt,
        **(extra or {}),
        **({"place_marker": T["marker"]} if "marker" in T else {}),
    )
    return meta
