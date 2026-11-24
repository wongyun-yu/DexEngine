"""Env-cfg assembly helpers: the entry the launchers, eval and render import (`env_setup.build`).

Everything here mutates or inspects an Isaac Lab env cfg / raw env passed in as a
parameter — no module-level Isaac imports (isaaclab imports stay function-level), so this
module is import-safe right after AppLauncher. args_cli is always taken as a parameter.

The helpers live in the topic modules of this package (objects, supports, extra_slots, sides, objtrack, contact_pilot,
robot_gains, object_rewards, ...); callers import them from there. This module keeps the few symbols other
programs read from this file by path (AERIAL_DISC_HEIGHT, _apply_shared_object).
"""

from __future__ import annotations

import importlib
import os
import sys

import gymnasium as gym

from dexengine.rl.env_setup.sides import (
    manager_groups,
)

# Read by name from this file by the baseline prep scripts (experiments/baselines); used by supports._build_disc_slots.
AERIAL_DISC_HEIGHT = 0.01  # disc thickness (m): the centre sits at z − h/2 so the top face is exactly at disc.z


# ======================================================================
# Shared helpers
# ======================================================================


class _TeeLogger:
    """Duplicate stdout to both console and a log file (filters progress bars)."""

    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._file = open(filepath, "a")  # noqa: SIM115  — stdout tee: the handle lives as long as the process
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        if "\r" not in data and "it/s]" not in data:
            self._file.write(data)
            self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def isatty(self):
        return False


def resolve_env_cfg(gym_id: str):
    """Dynamically resolve the env config class from a registered gym ID."""
    spec = gym.spec(gym_id)
    entry = spec.kwargs["env_cfg_entry_point"]
    mod_path, cls_name = entry.rsplit(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def _apply_shared_object(cfg, args_cli, tag):
    """SHARED OBJECT: one physical object for both hands -> ONE prim in the scene.

    Gated by ``args_cli.shared_object`` (set by rl/cli.py --sequence_dir from each tree's task_info objects[]);
    when it is off
    this function returns before touching anything = the two-prim bimanual build, byte-identical.

    What it does when ON (all in this ONE place — the env cfg module stays byte-identical):
      1. ``cfg.scene.object_left = None`` — InteractiveScene skips None entities, so no ObjectLeft prim is spawned.
         (Two prims of the same mesh teleported to the same pose = fully interpenetrating duplicates = explosion.)
      2. Object-filtered LEFT contact sensors (``left_*_obj_s``: prim_path ``{ENV_REGEX_NS}/ObjectLeft/base_link``)
         re-point to ``ObjectRight/base_link`` — the left fingertips now report force against the one object.
      3. Every manager term whose params carry ``object_key='object_left'`` / ``SceneEntityCfg('object_left')``
         (rewards, observations, terminations, curriculum) re-points to ``object_right`` — the left reward/obs
         terms read the same prim, which IS the data's meaning (both hands hold that object).
      4. LEFT object EVENTS are DROPPED, not re-pointed: ``reset_object_left`` (the right reset already places the
         prim from the shared qpos block), ``gravity_ramp_left`` (would apply the gravity force a second time = 2g),
         ``friction_decay_left`` / ``physics_dr_*_left`` (would double-randomise the same body). Any remaining
         event bound to ``object_left`` is an error.
      5. ``cfg.shared_object = True`` — the statement runtime code reads (collision_authoring: no ho/oo cross-pair
         exclusions since the left hand's object IS objR; collision_bodies.scene_object_key: left -> object_right).
      6. Post-sweep oracle: any residual 'object_left'/'ObjectLeft' string in scene fields or term params RAISES.

    MUST be called at the VERY END of train_cfg_overrides / _eval_cfg_overrides (after every runtime term
    registration, incl. contact-pilot LEFT terms, obs_v2, keypoint) — a term registered after the sweep would
    KeyError on the absent 'object_left' at env build (same rule as apply_left_uni).
    """
    if not int(getattr(args_cli, "shared_object", 0) or 0):
        return
    if not getattr(args_cli, "bimanual", 0):
        raise ValueError(f"[{tag}] --shared_object 1 requires --bimanual 1 (a uni build has one object already).")
    if getattr(args_cli, "left_uni", 0):
        raise ValueError(f"[{tag}] --shared_object 1 is mutually exclusive with --left_uni.")
    _L, _R = "object_left", "object_right"
    _LP, _RP = "{ENV_REGEX_NS}/ObjectLeft/base_link", "{ENV_REGEX_NS}/ObjectRight/base_link"
    report = {"sensors": [], "terms": [], "events_dropped": [], "scene_left": None}

    _rebind_left_to_shared(cfg, tag, _L, _LP, _R, _RP, report)

    print(
        f"[{tag}] [shared_object] applied: scene.object_left={report['scene_left']} -> None (ONE prim ObjectRight) · "
        f"sensors->ObjectRight/base_link {len(report['sensors'])} {report['sensors']} · "
        f"terms->object_right {len(report['terms'])} {report['terms']} · "
        f"events dropped {len(report['events_dropped'])} {report['events_dropped']}",
        flush=True,
    )


def _rebind_left_to_shared(cfg, tag, _L, _LP, _R, _RP, report):
    """Drop the left object, rebind every left sensor / term / event to the shared object and refuse leftovers."""
    _binds_left = _drop_left_object_prims(cfg, _L, _LP, _RP, report)

    for _gname, _grp in manager_groups(cfg):
        for _nm in list(vars(_grp).keys()):
            _term = getattr(_grp, _nm, None)
            _p = getattr(_term, "params", None)
            if not _binds_left(_p):
                continue
            if _gname == "events":
                setattr(_grp, _nm, None)  # 4) drop — the right-side event already acts on the one prim
                report["events_dropped"].append(_nm)
                continue
            for _k, _v in list(_p.items()):  # 3) re-point
                if _v == _L:
                    _p[_k] = _R
                elif getattr(_v, "name", None) == _L:
                    _v.name = _R
            report["terms"].append(f"{_gname}.{_nm}")

    _refuse_left_leftovers(cfg, tag, _L, _LP)


def _drop_left_object_prims(cfg, _L, _LP, _RP, report):
    """Drop the left object prim, repoint the sensors filtered on it, return the predicate for terms bound to it."""
    # 1) the left prim
    if getattr(cfg.scene, _L, None) is not None:
        report["scene_left"] = getattr(getattr(cfg.scene, _L), "prim_path", "?")
        cfg.scene.object_left = None

    # 2) scene sensors filtered on the left object
    for _nm in list(vars(cfg.scene).keys()):
        _ent = getattr(cfg.scene, _nm, None)
        if _ent is None or not hasattr(_ent, "prim_path"):
            continue
        _hit = False
        if getattr(_ent, "prim_path", None) == _LP:
            _ent.prim_path = _RP
            _hit = True
        _fpp = getattr(_ent, "filter_prim_paths_expr", None)
        if isinstance(_fpp, (list, tuple)) and any(x == _LP for x in _fpp):
            _ent.filter_prim_paths_expr = [(_RP if x == _LP else x) for x in _fpp]
            _hit = True
        if _hit:
            report["sensors"].append(_nm)

    # 3)+4) manager terms
    def _binds_left(_p):
        if not isinstance(_p, dict):
            return False
        return any(_v == _L or getattr(_v, "name", None) == _L for _v in _p.values())

    return _binds_left


def _refuse_left_leftovers(cfg, tag, _L, _LP):
    """Mark the cfg shared and refuse anything that still names the left object."""
    # 5) the statement
    cfg.shared_object = True

    # 6) oracle — nothing may still name the left object
    def _mentions_left(_x, _depth=0):
        if _depth > 6:
            return False
        if isinstance(_x, str):
            return _x in (_L, _LP) or "ObjectLeft" in _x
        if isinstance(_x, dict):
            return any(_mentions_left(v, _depth + 1) for v in _x.values())
        if isinstance(_x, (list, tuple)):
            return any(_mentions_left(v, _depth + 1) for v in _x)
        if getattr(_x, "name", None) == _L:
            return True
        return False

    _left_over = []
    for _nm in list(vars(cfg.scene).keys()):
        _ent = getattr(cfg.scene, _nm, None)
        if _ent is not None and (
            _mentions_left(getattr(_ent, "prim_path", None))
            or _mentions_left(getattr(_ent, "filter_prim_paths_expr", None))
        ):
            _left_over.append(f"scene.{_nm}")
    for _gname, _grp in manager_groups(cfg):
        for _nm in list(vars(_grp).keys()):
            _term = getattr(_grp, _nm, None)
            if _term is not None and _mentions_left(getattr(_term, "params", None)):
                _left_over.append(f"{_gname}.{_nm}")
    if _left_over:
        raise ValueError(f"[{tag}] SHARED OBJECT: entries still bound to the LEFT object after the sweep: {_left_over}")

    # Contact obs / reward / termination moved to apply_contact_pilot_overrides
    # (NEW per-finger annotation + object-filtered force). See that function.


# ======================================================================
# Helper: object-keypoint reward/diag term registration
# ======================================================================
