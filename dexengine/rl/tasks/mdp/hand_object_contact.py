"""PhysX filtered contact force between hand links and an object: single owner of the judge PARK gate's 'parked hand
touch' (E) measure (env window `hand_object_contact_force`). When to read is owned by the judge side (`rl.judge.park`):
once at the cut, for envs parked at the cut step (policy off, frame-0 target, object pinned). This module only reads.

What it measures: the PhysX contact force between the object body (sensor; the object spawn turns on the contact report
API, `env_setup.build.build_object_spawn_cfg`) and the hand links (filters). One object body × L links = a force matrix
(sensor, L, 3): per pair the sum of contact impulses / physics dt. Without an active contact it is exactly 0 (also
inside
the contact offset when apart), so 'touch' = penetration or pressing.

Hand links = robot bodies matching `HandSpec.contact_body_regex_right` (right hand) and its mirror
(`dexengine.core.hands.mirror_right_to_left`, left hand). A spec rule, not a name table, so it applies to a hand never
seen. With the split articulation (`split_artic.enabled()`) the left hand lives in `robot_left` (the same rule as the
scene cfg's left sensor namespace). Why this direction (object = sensor, links = filters): the filter side needs no
contact report API, so the measure is not bound to the link set the bake gave the API (fingertips, hand multi-body
sensors) and sees touches by non-fingertip links too.

No view order is assumed: the `env_<k>` of the sensor and filter paths is read to check that the same envs pair up, and
a mismatch fails. Views are built once on the first read (after play; the physics simulation view must exist), one per
object asset (a shared object once).

Liveness check (measure vs measure, used by `rl.judge.park`): the contact state of the existing fingertip sensors
(`rewards.contact_base.OBJ_SENSOR_NAMES`, own fingertips vs own object) at the same step comes back too. Fingertips are
a subset of the hand links, so an env whose fingertip touches must touch in this measure as well. When no comparison is
possible (no sensor, or the fingertip sensors' filter is not this object) `tip` is None and `tip_note` says why.

The collection recorder (`dexengine/record/rollout/recorder.py`) reads the same views: `link_forces(obj)` = the force
vector the object receives per hand link (env, link, 3) (sum over all object bodies, column order `link_names()`). The
scene cfg's contact sensors are not added to or changed because the rollout must be the judge's; these views are read
only and do not change the physics. The PhysX force matrix is **normal force only** (contact normal impulse / physics
dt, no friction), signed as 'the force the filter (link) applies to the sensor (object)' (the fingertip sensors'
`force_matrix_w` is the opposite: the force the object applies to the fingertip).
"""

from __future__ import annotations

import re
from typing import NamedTuple

import torch

_ENV = re.compile(r"/env_(\d+)/")
MAX_CONTACT_DATA_PER_PRIM = (
    4  # the contact buffer size of Isaac Lab's ContactSensorCfg default (only the force matrix is read)
)


class StartContact(NamedTuple):
    force: dict  # hand ("right"|"left") -> (n_envs,) max force [N] between that hand's links and this side's object
    tip: torch.Tensor | None  # (n_envs,) bool: own fingertip sensor contact > 0; None when no comparison is possible
    links: dict  # hand -> number of filter links
    tip_note: str  # why tip is None ("" = compared)


def env_index(paths) -> list[int]:
    out = []
    for p in paths:
        m = _ENV.search(str(p))
        if m is None:
            raise RuntimeError(f"[HAND_OBJ_CONTACT] no env number in the path: {p}")
        out.append(int(m.group(1)))
    return out


def env_glob(paths) -> str:
    """Paths with one prim per env -> one PhysX glob (`/env_<k>/` -> `/env_*/`). More than one fails."""
    globs = {_ENV.sub("/env_*/", str(p), count=1) for p in paths}
    if len(globs) != 1:
        raise RuntimeError(f"[HAND_OBJ_CONTACT] not the same prim per env — {len(globs)} globs: {sorted(globs)[:3]}")
    return globs.pop()


def nested(seq, rows: int) -> list[list[str]]:
    """PhysX path list -> `rows` rows of equal length. Rows stay as they are, a flat list is cut in order; a shape that
    does not fit fails."""
    seq = list(seq)
    if len(seq) == rows and all(isinstance(r, (list, tuple)) for r in seq):
        out = [list(r) for r in seq]
    elif rows > 0 and seq and all(isinstance(r, str) for r in seq) and len(seq) % rows == 0:
        k = len(seq) // rows
        out = [seq[i * k : (i + 1) * k] for i in range(rows)]
    else:
        raise RuntimeError(f"[HAND_OBJ_CONTACT] unknown path list shape: length {len(seq)} · rows {rows}")
    if len({len(r) for r in out}) > 1:
        raise RuntimeError("[HAND_OBJ_CONTACT] rows have different path counts")
    return out


def check_pairs(sensor_paths, filter_paths, n_filters: int) -> list[int]:
    """Per sensor, env number -> list. Checks that all filters of sensor i are in the same env (no view order
    assumed)."""
    s_env = env_index(sensor_paths)
    rows = nested(filter_paths, len(s_env))
    for i, row in enumerate(rows):
        if len(row) != n_filters or any(e != s_env[i] for e in env_index(row)):
            raise RuntimeError(
                f"[HAND_OBJ_CONTACT] sensor {sensor_paths[i]}'s filters pair with another env: {row[:3]}"
            )
    return s_env


def hand_rules(hand_name: str) -> dict[str, str]:
    """hand -> body name regex (spec rule: right hand = contact_body_regex_right, left hand = its mirror)."""
    from dexengine.core.hands import HAND_REGISTRY, mirror_right_to_left

    spec = HAND_REGISTRY.get(hand_name)
    if spec is None:
        raise RuntimeError(f"[HAND_OBJ_CONTACT] hand {hand_name!r} is not in HAND_REGISTRY")
    rx = spec.contact_body_regex_right
    return {"right": rx, "left": mirror_right_to_left(rx)}


def link_globs(names, link_paths, n_envs: int, rx: str) -> list[tuple[str, str]]:
    """[(link name, glob)] for bodies whose name matches rx. Paths are found by leaf name per env (no order assumed)."""
    rows = nested(link_paths, n_envs)
    by_env = []
    for r in rows:
        leaf = {p.rsplit("/", 1)[-1]: p for p in r}
        if len(leaf) != len(r):
            raise RuntimeError("[HAND_OBJ_CONTACT] articulation link leaf names collide")
        by_env.append(leaf)
    out = []
    for n in names:
        if re.fullmatch(rx, n):
            if any(n not in m for m in by_env):
                raise RuntimeError(f"[HAND_OBJ_CONTACT] the link path of body {n} is not in every env")
            out.append((n, env_glob([m[n] for m in by_env])))
    return out


def object_globs(obj) -> list[str]:
    """Body globs of an object asset: one for a rigid body, one per link for an articulation. Any other view fails."""
    v = obj.root_physx_view
    if hasattr(v, "link_paths"):
        rows = nested(v.link_paths, int(v.count))
        return [env_glob([r[j] for r in rows]) for j in range(len(rows[0]))]
    if hasattr(v, "prim_paths"):
        return [env_glob(list(v.prim_paths))]
    raise RuntimeError(f"[HAND_OBJ_CONTACT] cannot read the body paths of object view {type(v).__name__}")


def per_env_link_force(m: torch.Tensor, s_env: torch.Tensor, n_env: int) -> torch.Tensor:
    """(sensor, filter, 3) force matrix -> (n_env, filter, 3): scattered by the sensor rows' env numbers (s_env), with
    the object bodies (articulation links) of one env summed (no view order assumed)."""
    out = torch.zeros((n_env, m.shape[1], 3), dtype=m.dtype, device=m.device)
    return out.index_add_(0, s_env.to(m.device), m)


class HandObjectContact:
    """Meter of one env. `read(side)` -> StartContact · `link_forces(obj)` -> per-link forces. Views are built on the
    first read."""

    def __init__(self, env):
        self.env = env
        self.views: dict = {}  # id(object) -> (view, sensor env number tensor, hand -> column list)
        self.links: dict = {}  # hand -> [(name, glob)]
        self.tips: dict = {}  # side -> (sensor list | None, reason)

    def _robot(self, hand: str):
        from dexengine.rl.tasks.scene import split_artic

        key = split_artic.LEFT_ENTITY if hand == "left" and split_artic.enabled() else "robot"
        return self.env.scene[key]

    def _build_links(self) -> None:
        from dexengine.core.arms import active_hand_name

        rules = hand_rules(active_hand_name())
        for hand, rx in rules.items():
            art = self._robot(hand)
            self.links[hand] = link_globs(
                list(art.body_names), art.root_physx_view.link_paths, int(self.env.num_envs), rx
            )
        if not any(self.links.values()):
            raise RuntimeError(f"[HAND_OBJ_CONTACT] no robot body matches the hand link rules {rules}")

    def _view(self, obj):
        key = id(obj)
        if key in self.views:
            return self.views[key]
        if not self.links:
            self._build_links()
        from isaacsim.core.simulation_manager import SimulationManager

        flt = [(h, g) for h, lst in self.links.items() for _, g in lst]
        sens = object_globs(obj)
        pats = [g for _, g in flt]
        n_env = int(self.env.num_envs)
        view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
            sens if len(sens) > 1 else sens[0],
            filter_patterns=[pats] * len(sens) if len(sens) > 1 else pats,
            max_contact_data_count=MAX_CONTACT_DATA_PER_PRIM * len(sens) * n_env,
        )
        if int(view.sensor_count) != len(sens) * n_env or int(view.filter_count) != len(pats):
            raise RuntimeError(
                f"[HAND_OBJ_CONTACT] view size: sensors {view.sensor_count} ≠ {len(sens)}×{n_env} or "
                f"filters {view.filter_count} ≠ {len(pats)}"
            )
        s_env = check_pairs(list(view.sensor_paths), view.filter_paths, len(pats))
        cols = {h: [i for i, (hh, _) in enumerate(flt) if hh == h] for h in self.links}
        self.views[key] = (view, torch.tensor(s_env, dtype=torch.long, device=self.env.device), cols)
        return self.views[key]

    def _tips(self, side: str, obj):
        if side in self.tips:
            return self.tips[side]
        from dexengine.rl.tasks.mdp.rewards.contact_base import OBJ_SENSOR_NAMES

        names = OBJ_SENSOR_NAMES[side]
        keys = set(self.env.scene.keys())
        if any(n not in keys for n in names):
            self.tips[side] = (None, "no fingertip sensor")
            return self.tips[side]
        sens = [self.env.scene[n] for n in names]
        want = set(object_globs(obj))
        got = {str(e).replace(".*", "*") for s in sens for e in s.cfg.filter_prim_paths_expr}
        if not got <= want:
            self.tips[side] = (None, "the fingertip sensors' filter is not this object")
            return self.tips[side]
        tip_names = {str(s.cfg.prim_path).rsplit("/", 1)[-1] for s in sens}
        own = {n for n, _ in self.links.get(side, [])}
        if (
            not tip_names <= own
        ):  # the spec rule does not cover the fingertips: the check would raise false alarms (a rule defect)
            raise RuntimeError(
                f"[HAND_OBJ_CONTACT] fingertip bodies {sorted(tip_names - own)} are outside the {side} hand link rule"
            )
        self.tips[side] = (sens, "")
        return self.tips[side]

    def link_names(self) -> list[str]:
        """Filter column order of the view = hand link names (hand rule order: right hand, left hand)."""
        if not self.links:
            self._build_links()
        return [n for lst in self.links.values() for n, _ in lst]

    def link_forces(self, obj) -> torch.Tensor:
        """(n_envs, hand links, 3): contact normal force each hand link applies to the object (sum over all bodies) [N ·
        world]; columns = link_names()."""
        view, s_env, _cols = self._view(obj)
        m = view.get_contact_force_matrix(float(self.env.physics_dt))
        return per_env_link_force(m, s_env, int(self.env.num_envs))

    def read(self, side: str, obj) -> StartContact:
        view, s_env, cols = self._view(obj)
        if not self.links.get(
            side
        ):  # without own hand links the measure is always 0: fail so the gate never runs the old way silently
            raise RuntimeError(
                f"[HAND_OBJ_CONTACT] no body matches the {side} hand link rule — "
                "the start touch of that side's object cannot be measured"
            )
        m = view.get_contact_force_matrix(float(self.env.physics_dt)).norm(dim=-1)  # (sensor, filter)
        n_env = int(self.env.num_envs)
        force = {}
        for hand, c in cols.items():
            per = m[:, c].amax(dim=1) if c else torch.zeros(m.shape[0], device=m.device)
            force[hand] = torch.zeros(n_env, device=m.device).scatter_reduce(0, s_env, per, "amax")
        sens, note = self._tips(side, obj)
        tip = None
        if sens is not None:
            tip = torch.zeros(n_env, dtype=torch.bool, device=m.device)
            for s in sens:
                tip |= s.data.force_matrix_w.reshape(n_env, -1, 3).norm(dim=-1).amax(dim=1) > 0.0
        return StartContact(force, tip, {h: len(v) for h, v in self.links.items()}, note)


def instance(env) -> HandObjectContact:
    """One meter per env (the judge and the collection recorder share the views)."""
    hoc = env.__dict__.get("_hand_obj_contact")
    if hoc is None:
        hoc = env.__dict__["_hand_obj_contact"] = HandObjectContact(env)
    return hoc


def measure(env, side: str) -> StartContact:
    """Body of the env window `hand_object_contact_force`: the reading for side `side`'s object
    (`held_object_assets`)."""
    hoc = instance(env)
    obj = next((o for s, o in env.held_object_assets() if s == side), None)
    if obj is None:
        raise RuntimeError(f"[HAND_OBJ_CONTACT] side {side} holds no object (held_object_assets)")
    return hoc.read(side, obj)
