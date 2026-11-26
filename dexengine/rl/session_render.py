"""The render seam of the trainer: the one point where a record session attaches the function that draws.

The trainer draws nothing itself. With --record_session the built env goes to render() — from the eval task (the
kinematic replay dump of the record render, the pose of a clip) or from the frozen-render branch of the judge entry
(DEXENGINE_FROZEN_RENDER, the E5 closed-loop sessions). A record session (dexengine.record.session.entry: stages
render · render_clip · render_clips · plugin stages; experiments.sim_eval.session) replaces `render` right after this
module is imported (post-import hook, marker `[COLLECT_SESSION] hook: session_render.render → …`) with its own
function, which applies the canonical look (dexengine.record.looks.look_d.apply) and writes through the frame guard.
Called without that hook, render() dies: robot videos come only from scripts/stage4_record.py.

Signature of an attached function: (wrapped, policy_fn, traj_mgr, num_clips, clip_names, raw_env, checkpoint_path,
out_dir, *, args_cli, create_env).
"""

from __future__ import annotations

MARK = "[SESSION_RENDER]"
#: the post-import hook target (module attribute a record session replaces)
HOOK_TARGET = "session_render.render"


def render(*_a, **_kw):
    """Placeholder a record session replaces; reaching it means no record session attached a render function."""
    raise SystemExit(
        f"{MARK} no render function attached — the RL trainer draws no frames; robot videos come only from the record "
        "render (scripts/stage4_record.py), which attaches its function to this seam"
    )


def run(wrapped, policy_fn, traj_mgr, num_clips, clip_names, raw_env, checkpoint_path, out_dir, args_cli, create_env):
    """Hand the built env to the attached render function (looked up at call time, after the hook)."""
    import sys

    fn = sys.modules[__name__].render
    print(f"{MARK} → {getattr(fn, '__qualname__', fn)} · clips {num_clips} · out {out_dir}", flush=True)
    return fn(
        wrapped,
        policy_fn,
        traj_mgr,
        num_clips,
        clip_names,
        raw_env,
        checkpoint_path,
        out_dir,
        args_cli=args_cli,
        create_env=create_env,
    )
