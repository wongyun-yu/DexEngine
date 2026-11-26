"""The trainer process's Kit — the one owner of the rule "the entry decides the kit" (dexengine/rl/cli.py: every
train · judge · render · replay process comes through it).

Rule (training always without rendering, as fast as possible): an RL training or judging process runs
on the plain `headless.kit`, with no RTX renderer, and draws nothing. Robot frames come only from the record render
(scripts/stage4_record.py): its session replays a launcher dump of a render or replay process (the rendering kit, the
same robot plate and scene) and attaches its render function to the trainer's seam (dexengine.rl.session_render).
Which process is which is the entry the launcher names (dexengine/rl/launch/argv.Call.entry), never a switch a caller
sets.

  ENTRY_ENV  the entry of this process, set by the launcher for every trainer process (required — no default):
             train · judge  = KIT_PLAIN: no camera may reach the scene (head_camera leaves before the env is built,
                              training_scene); a render request dies (--record_session · --pose_render_out ·
                              --enable_cameras · DEXENGINE_FROZEN_RENDER=1)
             render         = physics rollouts of a checkpoint under the judge's frozen conditions handed to a record
                              session (DEXENGINE_FROZEN_RENDER=1: the E5 closed-loop evaluation dumps) — rendering kit
             replay         = the eval task: the record render's kinematic replay (--record_session) or the pose still
                              (--pose_render_out) — rendering kit
             A render entry must render (DEXENGINE_FROZEN_RENDER=1, or the eval task): a render process that draws
             nothing is a mislabelled launch and dies.
  REMOVED    DEXENGINE_TRAIN_NO_RENDER (the old opt-in switch of the plain kit): given at all = error — the entry owns
             the kit now, a caller cannot choose it.
  MARK       the line a plain-kit process prints before Kit starts — a run log without it did not use the plain kit
             (require_mark: the record stage 1 and the aug physics run prove it on their session logs)
  bake_kit   a robot USD bake process (urdf_to_usd convert · contact_report_bake) draws nothing either: it runs on
             the plain kit too (BAKE_MARK). The full app kit those tools took without --headless
             (isaaclab.python.kit) segfaults on some cluster nodes, where the plain kit bakes; the plain-kit bake
             gives the same robot bodies, byte for byte, as the full-kit bake.
Standard library only (the entry imports it before Kit; dexengine.rl.preapp watches for numpy/torch).
"""

from __future__ import annotations

from dexengine.core.errors import InputMissingError

ENTRY_ENV = "DEXENGINE_ENTRY"
KIT_PLAIN = ("train", "judge")
KIT_RENDER = ("render", "replay")
ENTRIES = KIT_PLAIN + KIT_RENDER
REMOVED = ("DEXENGINE_TRAIN_NO_RENDER",)
MARK = "[TRAIN_KIT] headless"
BAKE_MARK = "[BAKE_KIT] headless"
RENDER_MARK = "[TRAIN_KIT] rendering"
FROZEN_RENDER = "DEXENGINE_FROZEN_RENDER"


def entry_of(environ) -> str:
    """The entry this process runs as (ENTRY_ENV) — ValueError when it is missing or not one of ENTRIES, or when a
    removed switch is given."""
    gone = [k for k in REMOVED if k in environ]
    if gone:
        raise ValueError(f"{gone} removed: the entry decides the kit ({ENTRY_ENV} — dexengine/rl/app.py)")
    e = environ.get(ENTRY_ENV, "")
    if e not in ENTRIES:
        raise ValueError(
            f"{ENTRY_ENV}={e!r} — one of {ENTRIES} is required (set by the launcher, dexengine.rl.launch; "
            "the trainer has no default entry)"
        )
    return e


def plain_kit(environ, *, task: str, record_session: bool, pose_render: bool, enable_cameras: bool) -> bool:
    """True = this process runs on the plain kit (module doc). ValueError (the entry turns it into parser.error)
    for a render request in a train · judge process, or a render process that renders nothing."""
    e = entry_of(environ)
    renders = [
        n
        for n, on in (
            ("--record_session", record_session),
            ("--pose_render_out", pose_render),
            ("--enable_cameras", enable_cameras),
            (f"{FROZEN_RENDER}=1", environ.get(FROZEN_RENDER, "0") == "1"),
        )
        if on
    ]
    if e in KIT_PLAIN:
        if renders:
            raise ValueError(
                f"{ENTRY_ENV}={e} runs on the plain kit and never renders — {renders}; robot frames come only from the "
                "record render (scripts/stage4_record.py)"
            )
        return True
    if not renders and task != "eval":
        raise ValueError(f"{ENTRY_ENV}={e} is a render process but nothing renders ({FROZEN_RENDER}=1 missing)")
    return False


def bake_kit(args, tool: str) -> None:
    """Set an AppLauncher namespace of a robot USD bake tool to the plain kit (module doc, bake_kit): headless=True
    (isaaclab.python.headless.kit, no RTX renderer) and no cameras — a bake renders nothing; ``--enable_cameras``
    given dies. Marker BAKE_MARK with the tool name."""
    if getattr(args, "enable_cameras", False):
        raise ValueError(f"{tool}: a bake process never renders — --enable_cameras given (app.py bake_kit)")
    args.headless = True
    print(f"{BAKE_MARK} — {tool} · plain kit (isaaclab.python.headless.kit)", flush=True)


def context_line(entry: str, plain: bool, num_envs) -> str:
    """The [RENDER_CTX] marker of the process: entry · kit · env count (a render's env count is its entry's —
    dexengine.rl.launch.modes.render_ne / RENDER_NE)."""
    return (
        f"[RENDER_CTX] {'off' if plain else 'on'} · entry={entry} · kit={'plain' if plain else 'rendering'} · "
        f"NE={num_envs}"
    )


def training_scene(cfg, entry: str) -> None:
    """The scene of a plain-kit process = the render scene minus what only a renderer can hold: the head camera (a
    CameraCfg left in the env cfg hangs the plain kit while resolving it). Called by the entry before the env is
    built; marker [TRAIN_SCENE]."""
    if getattr(cfg.scene, "head_camera", None) is not None:
        cfg.scene.head_camera = None
        print(f"[TRAIN_SCENE] head_camera removed (plain kit · entry={entry})", flush=True)


def require_mark(log) -> None:
    """A plain-kit process proves it took: its log carries MARK — else InputMissingError."""
    from pathlib import Path

    if MARK not in Path(log).read_text(errors="replace"):
        raise InputMissingError(f"the {MARK} marker is missing — not the plain headless kit: {log}")
