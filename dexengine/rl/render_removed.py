"""Inputs of the removed RL-side renderers — one owner list (a removed input dies, it is never ignored).

The RL trainer draws no frames. Every robot video comes from the record render (scripts/stage4_record.py,
dexengine.record.*), which applies the one canonical look (dexengine.record.looks.look_d.apply) and writes through the
frame guard (dexengine.record.looks.look_guard). The per-clip policy video, the four-view kinematic replay, the tiled
trajectory replay (DEXENGINE_TRAJ_RENDER), the gym viewport recorder (--video), the RSI still frames and the old render
look's knobs are gone; given anyway, each dies with POINTER.

  ENV / ENV_PREFIX  environment names: checked by the launcher (dexengine.rl.launch.request) and by the trainer before
                    Kit (dexengine/rl/cli.py) — a dump replayed by a record session starts the trainer directly
  ARGS              trainer flags: registered by dexengine.rl.args so that a removed flag dies with POINTER rather than
                    argparse's bare "unrecognized arguments"
Standard library only.
"""

from __future__ import annotations

import argparse

MARK = "[RENDER_REMOVED]"
POINTER = (
    "the RL trainer draws no frames — robot videos come only from the record render: scripts/stage4_record.py "
    "(dexengine.record.*, one canonical look: dexengine.record.looks.look_d)"
)
#: removed environment inputs -> what they were
ENV = {
    "DEXENGINE_TRAJ_RENDER": "tiled replay render of a recorded trajectory file (rl.render.replay)",
    "TRAJ_RENDER_OUT": "video folder of DEXENGINE_TRAJ_RENDER",
    "DEXENGINE_TRAJ_RENDER_SLOTS": "slot count of the tiled replay render",
    "DEXENGINE_RENDER_VIEWS": "view list of the per-clip video cameras (a drawing scene has ego + front only)",
    "DEXENGINE_LOG_EGO_PITCH": "ego pitch log of the old render look (look_d.apply always logs it)",
    "DEXENGINE_DOME_YAW": "dome rotation of the old render look",
    "DEXENGINE_VIDEO_QUALITY": "imageio quality of the per-clip policy video",
    "DEXENGINE_VIDEO_CRF": "CRF of the per-clip and replay video encoders",
    "DEXENGINE_VIDEO_PRESET": "x264 preset of the per-clip and replay video encoders",
    "RENDER_EGO_FOCAL_MM": "A/B ego focal length of the old render look (record.looks.scene_env owns the value)",
    "RENDER_STILL_FRAMES": "RSI still PNG frames of the kinematic replay video",
    "RENDER_STILL_WARMUP": "RTX warm-up renders of the RSI still frames",
    "RENDER_FRONT_ONLY": "front-only per-clip policy video",
    "MARKERS": "reference markers of the launcher's render and replay videos",
    "VIDEOS_ONLY_CLIPS": "clip filter of the launcher's bimanual render video",
}
#: every name under these prefixes is a removed knob of the old render look (a prefix rule, not a name table)
ENV_PREFIX = ("DEXENGINE_AUG_",)
#: removed trainer flags -> what they were
ARGS = {
    "--render_per_clip_videos": "per-clip policy / replay video (the seam flag is --record_session, launcher dumps)",
    "--video": "gym viewport recorder (RecordVideo)",
    "--video_dir": "folder of --video",
    "--render_markers": "wrist/object frame markers of the per-clip video",
    "--viz_mano_fingertips": "MANO fingertip overlay of the per-clip video",
    "--videos_max_attempts": "per-clip video rollout budget",
    "--videos_max_episode_steps": "per-clip video episode cap",
    "--videos_fps": "per-clip video frame rate",
    "--videos_warmup_attempts": "per-clip video warm-up rollouts",
    "--videos_select_best_success": "per-clip video best-success selection",
    "--videos_rollouts_per_clip": "per-clip video rollouts",
    "--videos_perclip_eval": "per-clip video retry budget from an eval",
    "--videos_only_clips": "per-clip video clip filter",
    "--videos_allfail_retry_batches": "per-clip video retry batches",
    "--videos_rank_metric": "per-clip video ranking",
}


def given_env(environ) -> list[str]:
    """Removed environment inputs that `environ` sets (a non-empty value other than '0')."""
    return sorted(
        k for k, v in environ.items() if (k in ENV or k.startswith(ENV_PREFIX)) and str(v).strip() not in ("", "0")
    )


def env_error(environ) -> str:
    """'' when no removed input is set, else the error text (names, what they were, POINTER)."""
    got = given_env(environ)
    if not got:
        return ""
    what = " · ".join(f"{k} ({ENV.get(k, 'knob of the old render look')})" for k in got)
    return f"{MARK} removed input {what} — {POINTER}"


class _Removed(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{MARK} {option_string} removed ({ARGS[option_string]}) — {POINTER}")


def add_removed_args(parser) -> None:
    """Register every ARGS flag so that giving it dies with POINTER (any value form)."""
    for flag in ARGS:
        parser.add_argument(flag, nargs="?", action=_Removed, help=argparse.SUPPRESS)
