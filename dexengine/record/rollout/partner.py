"""Record: contact partner decomposition (always on; the old switch DEXENGINE_COLLECT_FORCE_PARTNER is retired).

Only read-only PhysX contact views are added. Neither physics nor the scene cfg (contact sensors included) changes: the
same approach as the force spike probe. The kind names, signs and time axis are owned by rollout.recorder (channels)
and store.force (store keys); this module does the views and the arithmetic only.
  (1) force_partner          hand sensor (same sensors and bodies as force_total) x partner class: the normal force
                             the link received from that class
  (2) force_object_partner   object (sum over bodies) x partner class: the normal force the object received from that
                             class
  (3) contact_sep            object x hand link (same links and order as force_object): minimum contact point
                             separation [m] (negative = penetration, no contact = NaN)
Partner class rule (read from the stage; no name table):
  filters = every rigid body (RigidBodyAPI) under env_0 + static colliders (CollisionAPI) outside rigid bodies +
  static colliders under /World outside the envs (ground etc.).
  classes, on top of the collision truth table's prim classification
  (dexengine.rl.tasks.scene.collision_check._classify_env: robot links RH/RA/LH/LA/OTHER, DESK, DISC, SUP):
    (1) robot bodies -> relative to the sensor side own_side (own hand, own arm) · other_side (other hand, arm) ·
        torso (the rest of the robot)
    (2) robot bodies -> hand_right · hand_left (same hand link rule as force_object: hand_object_contact.links) ·
        robot_other
    common: DESK -> desk · DISC -> support_disc · SUP -> support_object · outside the envs -> world ·
    other rigid bodies -> the scene asset key the body belongs to (objects = object_right etc.); not an asset -> the
    first prim name under the env
Completeness oracles (measured vs measured): (1) Σ classes = force_total (same link, scene sensor net) · (2) Σ classes
= the same view's object net. Measured over the whole rollout; beyond the threshold rollout.recorder finalize dies
before writing. A partner missing from the classes is caught here.
"""

from __future__ import annotations

import re

# contact data buffer slots per (sensor row, filter) pair; overflow dies (no silent truncation)
PER_PAIR = 16
# |Σ classes - total| <= ABS + REL x |total| (float32, summation order over dozens of filters)
ORACLE_ABS_N = 1e-2
ORACLE_REL = 1e-3
WORLD = "world"
STATIC = {"DESK": "desk", "DISC": "support_disc", "SUP": "support_object"}
ROBOT_GROUPS = ("RH", "RA", "LH", "LA", "OTHER")
SIDE_GROUPS = {"right": ("RH", "RA"), "left": ("LH", "LA")}
_ENV = re.compile(r"/env_(\d+)/")


# ── pure arithmetic (CPU testable) ───────────────────────────────────────────────────────
def link_class(desc: dict, side: str) -> str:
    """(1) filter description -> partner class relative to the hand link (sensor side)."""
    if desc["kind"] == "robot":
        g = desc["group"]
        if g in SIDE_GROUPS[side]:
            return "own_side"
        if g in SIDE_GROUPS["left" if side == "right" else "right"]:
            return "other_side"
        return "torso"
    return desc["label"]


def object_class(desc: dict, hand_links: dict) -> str:
    """(2) filter description -> partner class relative to the object (hand links = the force_object link rule)."""
    if desc["kind"] == "robot":
        for side, names in hand_links.items():
            if desc["leaf"] in names:
                return f"hand_{side}"
        return "robot_other"
    return desc["label"]


def onehot(labels: list[str]):
    """Class per filter -> (sorted class names, [F, C] 0/1 matrix)."""
    import torch

    names = sorted(set(labels))
    m = torch.zeros(len(labels), len(names))
    for j, lab in enumerate(labels):
        m[j, names.index(lab)] = 1.0
    return names, m


def class_sum(mat, oh):
    """[R, F, 3] force matrix · [F, C] -> [R, C, 3] (sum per class)."""
    import torch

    return torch.einsum("rfk,fc->rck", mat, oh.to(mat.device, mat.dtype))


def min_sep(sep, cnt, st, rows: int, n_filt: int):
    """separation, count and start of get_contact_data -> [rows, n_filt] minimum separation per pair (no contact =
    +inf)."""
    import torch

    cnt = cnt.reshape(-1).long()
    st = st.reshape(-1).long()
    out = torch.full((cnt.numel(),), float("inf"), device=cnt.device)
    tot = int(cnt.sum())
    if tot > 0:
        rid = torch.repeat_interleave(torch.arange(cnt.numel(), device=cnt.device), cnt)
        bs = cnt.cumsum(0) - cnt
        dl = torch.arange(rid.numel(), device=cnt.device) - bs.repeat_interleave(cnt)
        out = out.scatter_reduce(0, rid, sep.reshape(-1)[st[rid] + dl].float().to(cnt.device), "amin")
    return out.reshape(rows, n_filt)


def oracle_excess(parts, total):
    """[..., C, 3] per class · [..., 3] total -> (max difference, max excess over the threshold, max total |F|,
    touched sample count)."""
    d = (parts.sum(-2) - total).norm(dim=-1)
    ref = total.norm(dim=-1)
    return d.amax(), (d - (ORACLE_ABS_N + ORACLE_REL * ref)).amax(), ref.amax(), (ref > 0).sum()


# ── stage -> filter descriptions ──────────────────────────────────────────────────
def _env_glob(path: str) -> str:
    if not _ENV.search(path):
        raise RuntimeError(f"[COLLECT_REC] partner: not an env path {path}")
    return _ENV.sub("/env_*/", path, count=1)


def stage_filters(scene) -> list[dict]:
    """Filter descriptions [{glob, path0, kind(robot|other), group|label, leaf}]; rule in the module doc."""
    from pxr import Usd, UsdPhysics

    from dexengine.rl.tasks.scene import collision_check as CA

    stage = scene.stage
    env0 = scene.env_prim_paths[0]
    cls = CA._classify_env(stage, scene, env0)  # the collision truth table's classification (no second one)
    tops = {}
    for tbl in (scene.articulations, scene.rigid_objects):
        for key, ent in tbl.items():
            tops.setdefault(str(ent.cfg.prim_path).rstrip("/").rsplit("/", 1)[-1], key)
    rigid = [
        p.GetPath().pathString for p in Usd.PrimRange(stage.GetPrimAtPath(env0)) if p.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    rset = set(rigid)

    def under(path, s):
        q = path.rsplit("/", 1)[0]
        while q and q != "/":
            if q in s:
                return True
            q = q.rsplit("/", 1)[0]
        return False

    static = [
        p.GetPath().pathString
        for p in Usd.PrimRange(stage.GetPrimAtPath(env0))
        if p.HasAPI(UsdPhysics.CollisionAPI)
        and p.GetPath().pathString not in rset
        and not under(p.GetPath().pathString, rset)
    ]
    out = []
    for path in rigid + static:
        lab = CA._lookup(cls, path)
        leaf = path.rsplit("/", 1)[-1]
        top = path[len(env0) + 1 :].split("/", 1)[0]
        if lab in ROBOT_GROUPS:
            out.append({"glob": _env_glob(path), "path0": path, "kind": "robot", "group": lab, "leaf": leaf})
        else:
            label = STATIC.get(lab) or tops.get(top) or top
            out.append({"glob": _env_glob(path), "path0": path, "kind": "other", "label": label, "leaf": leaf})
    it = iter(Usd.PrimRange(stage.GetPrimAtPath("/World")))
    for p in it:
        s = p.GetPath().pathString
        if s == "/World/envs":
            it.PruneChildren()
            continue
        if p.HasAPI(UsdPhysics.CollisionAPI) and not p.HasAPI(UsdPhysics.RigidBodyAPI):
            out.append({"glob": s, "path0": s, "kind": "other", "label": WORLD, "leaf": s.rsplit("/", 1)[-1]})
    return out


def _rows(view, n_filt: int) -> tuple[list[int], list[str]]:
    """(env index, leaf name) per sensor row: checks the filters are in the same env (or global prims outside the envs)
    without assuming the view order."""
    from dexengine.rl.tasks.mdp import hand_object_contact as HOC

    s_paths = list(view.sensor_paths)
    s_env = HOC.env_index(s_paths)
    rows = HOC.nested(view.filter_paths, len(s_env))
    for i, row in enumerate(rows):
        if len(row) != n_filt:
            raise RuntimeError(f"[COLLECT_REC] partner: sensor {s_paths[i]} filters {len(row)} != {n_filt}")
        for p in row:
            m = _ENV.search(str(p))
            if m is not None and int(m.group(1)) != s_env[i]:
                raise RuntimeError(f"[COLLECT_REC] partner: filter {p} of sensor {s_paths[i]} is in another env")
    return s_env, [p.rsplit("/", 1)[-1] for p in s_paths]


class PartnerReader:
    """One view bundle. The classes (channel width) are decided from the stage at construction; the PhysX views are made
    at the first refresh (after play: the physics simulation view must exist; same rule as hand_object_contact).
    refresh() reads once per policy step into a cache; channel reads slice the cache."""

    def __init__(self, env, sensors: list, objects: list, hoc):
        """sensors = [(sensor key, ContactSensor)] (the same as force_total) · objects = [(object key, asset)] (the same
        as force_object)."""
        from dexengine.rl.tasks.mdp import hand_object_contact as HOC

        self.env, self.n, self.dt = env, int(env.num_envs), float(env.physics_dt)
        self.filt = stage_filters(env.scene)
        hand_links = {s: {nm for nm, _g in lst} for s, lst in hoc.links.items()}
        self.link_names = hoc.link_names()
        self.links = []
        for key, s in sensors:
            names = list(s.body_names)
            by_leaf = {}
            for d in self.filt:
                if d["kind"] == "robot" and d["leaf"] in names:
                    by_leaf.setdefault(d["leaf"], []).append(d)
            if sorted(by_leaf) != sorted(names) or any(len(v) != 1 for v in by_leaf.values()):
                raise RuntimeError(
                    f"[COLLECT_REC] partner: sensor {key} bodies {names} not found one each among the "
                    f"robot rigid bodies"
                )
            groups = {by_leaf[nm][0]["group"] for nm in names}
            side = next((sd for sd, g in SIDE_GROUPS.items() if groups <= set(g)), None)
            if side is None:
                raise RuntimeError(
                    f"[COLLECT_REC] partner: side of sensor {key} bodies cannot be decided {sorted(groups)}"
                )
            cls, oh = onehot([link_class(d, side) for d in self.filt])
            self.links.append(
                {
                    "key": key,
                    "names": names,
                    "side": side,
                    "classes": cls,
                    "oh": oh,
                    "globs": [by_leaf[nm][0]["glob"] for nm in names],
                }
            )
        self.objs = []
        for key, obj in objects:
            own = sorted(set(HOC.object_globs(obj)))
            of = [d for d in self.filt if d["glob"] not in own]
            if len(of) != len(self.filt) - len(own):
                raise RuntimeError(f"[COLLECT_REC] partner: object {key} bodies {own} are not stage rigid bodies")
            cls, oh = onehot([object_class(d, hand_links) for d in of])
            idx = []
            for nm in self.link_names:
                hit = [j for j, d in enumerate(of) if d["kind"] == "robot" and d["leaf"] == nm]
                if len(hit) != 1:
                    raise RuntimeError(f"[COLLECT_REC] partner: hand link {nm} has {len(hit)} filters (must be 1)")
                idx.append(hit[0])
            self.objs.append({"key": key, "own": own, "filt": of, "classes": cls, "oh": oh, "link_idx": idx})
        self.views = False
        self.cache = {}
        self.ora = None

    def _ensure_views(self) -> None:
        import torch

        from isaacsim.core.simulation_manager import SimulationManager

        if self.views:
            return
        sv = SimulationManager.get_physics_sim_view()
        dev = self.env.device
        fg = [d["glob"] for d in self.filt]
        for d in self.links:
            view = sv.create_rigid_contact_view(
                d["globs"], filter_patterns=[fg] * len(d["globs"]), max_contact_data_count=0
            )
            s_env, leaves = _rows(view, len(fg))
            if len(s_env) != len(d["names"]) * self.n:
                raise RuntimeError(
                    f"[COLLECT_REC] partner: sensor {d['key']} rows {len(s_env)} != {len(d['names'])}×{self.n}"
                )
            d.update(
                view=view,
                env=torch.tensor(s_env, device=dev),
                oh=d["oh"].to(dev),
                body=torch.tensor([d["names"].index(x) for x in leaves], device=dev),
            )
        for o in self.objs:
            sens, of = o["own"], [d["glob"] for d in o["filt"]]
            rows = len(sens) * self.n
            cap = PER_PAIR * len(of) * rows
            view = sv.create_rigid_contact_view(
                sens if len(sens) > 1 else sens[0],
                filter_patterns=[of] * len(sens) if len(sens) > 1 else of,
                max_contact_data_count=cap,
            )
            s_env, _l = _rows(view, len(of))
            if len(s_env) != rows:
                raise RuntimeError(f"[COLLECT_REC] partner: object {o['key']} rows {len(s_env)} != {rows}")
            o.update(
                view=view,
                rows=rows,
                n_filt=len(of),
                cap=cap,
                env=torch.tensor(s_env, device=dev),
                oh=o["oh"].to(dev),
                link_idx_t=torch.tensor(o["link_idx"], device=dev),
            )
        self.views = True

    def describe(self) -> str:
        return (
            f"filters {len(self.filt)} · hand sensors "
            + " · ".join(f"{d['key']}({d['side']}) classes {d['classes']}" for d in self.links)
            + " · objects "
            + " · ".join(f"{o['key']} classes {o['classes']}" for o in self.objs)
        )

    def refresh(self) -> None:
        import torch

        self._ensure_views()
        n, dt, c = self.n, self.dt, {}
        for d in self.links:
            m = class_sum(d["view"].get_contact_force_matrix(dt).reshape(len(d["env"]), len(self.filt), 3), d["oh"])
            out = torch.zeros(n, len(d["names"]), len(d["classes"]), 3, device=m.device, dtype=m.dtype)
            out[d["env"], d["body"]] = m
            c[("link", d["key"])] = out
        for o in self.objs:
            v, R, F = o["view"], o["rows"], o["n_filt"]
            m = class_sum(v.get_contact_force_matrix(dt).reshape(R, F, 3), o["oh"])
            c[("obj", o["key"])] = torch.zeros(n, len(o["classes"]), 3, device=m.device, dtype=m.dtype).index_add_(
                0, o["env"], m
            )
            net = v.get_net_contact_forces(dt).reshape(R, 3)
            c[("net", o["key"])] = torch.zeros(n, 3, device=net.device, dtype=net.dtype).index_add_(0, o["env"], net)
            _f, _p, _nrm, sep, cnt, st = v.get_contact_data(dt)
            if int(cnt.sum()) > o["cap"]:
                raise RuntimeError(
                    f"[COLLECT_REC] partner: object {o['key']} contacts {int(cnt.sum())} > buffer {o['cap']}"
                )
            sm = min_sep(sep, cnt, st, R, F)
            se = torch.full((n, F), float("inf"), device=sm.device).scatter_reduce(
                0, o["env"].to(sm.device)[:, None].expand(R, F), sm, "amin"
            )
            sel = se[:, o["link_idx_t"].to(sm.device)]
            c[("sep", o["key"])] = torch.where(torch.isfinite(sel), sel, torch.full_like(sel, float("nan")))
        self.cache = c

    def check(self, totals: dict) -> None:
        """Oracle accumulation: totals = {sensor key: [n, B·3] force_total values (same _row)}."""
        import torch

        if self.ora is None:
            self.ora = {"link": torch.zeros(4), "obj": torch.zeros(4)}
            for k in self.ora:
                self.ora[k][1] = -float("inf")
        for d in self.links:
            parts = self.cache[("link", d["key"])]
            self._acc("link", oracle_excess(parts, totals[d["key"]].reshape(self.n, len(d["names"]), 3)))
        for o in self.objs:
            self._acc("obj", oracle_excess(self.cache[("obj", o["key"])], self.cache[("net", o["key"])]))

    def _acc(self, k, vals) -> None:
        d, ex, ref, nc = (float(x) for x in vals)
        a = self.ora[k]
        a[0], a[1], a[2] = max(float(a[0]), d), max(float(a[1]), ex), max(float(a[2]), ref)
        a[3] += nc

    def verdict(self) -> tuple[list[str], list[str]]:
        """For finalize: the oracle lines; a mismatch raises RuntimeError (dies before writing)."""
        lines, bad = [], []
        for k, what in (
            ("link", "Σ partner classes = force_total (same link)"),
            ("obj", "Σ partner classes = object net"),
        ):
            d, ex, ref, nc = (float(x) for x in self.ora[k]) if self.ora else (0.0, 0.0, 0.0, 0.0)
            lines.append(
                f"[COLLECT_REC] partner oracle {what} · touched samples {int(nc)} · max difference "
                f"{d:.4g} N · max |F| {ref:.4g} N · over threshold {max(ex, 0.0):.4g} N"
            )
            if nc <= 0:
                bad.append(f"{what}: 0 touched samples, unproven")
            elif ex > 0:
                bad.append(f"{what}: difference {d:.4g} N > threshold (a partner class is missing)")
        return lines, bad
