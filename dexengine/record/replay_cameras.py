"""Camera pool of the replay render: no camera per env (tile cost ∝ camera count).

The scene builds every session clip (env = clip) at once, and M free camera slots x V views move to the canonical
camera place of the env each slot draws now. A slot moves to the next clip when its clip ends (longest first, the
slot that frees first), so one session boot leaves almost no idle tiles from clip length differences.

The canonical camera cfgs (focal length, aperture, offset, parent prim) are those create_env built, set aside by
configure_scene (take_specs; their values are still owned by that cfg). Camera place = the parent's world pose ∘ the
parent-relative offset:
  when the parent is an articulation link (or a fixed prim under one), the link pose (after the PhysX kinematic
  update) every frame,
  otherwise a fixed place relative to the env prim, only when the slot changes env.
Marker: [REPLAY_POOL] ….
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# total camera tile pixels of a session (MP): 1024² x 64 tiles ≈ 20 GB (stage-1 bench 4 GB + 0.25 GB/tile)
TILE_MPIX_BUDGET = 64.0
SLOTS_ENV = "DEXENGINE_TRAJ_RENDER_SLOTS"  # explicit slot count (otherwise derived from TILE_MPIX_BUDGET)
POOL_ROOT = "/World/ReplayCams"


@dataclass
class CamSpec:
    view: str
    prim_path: str  # canonical camera prim path template ({ENV_REGEX_NS}/…)
    spawn: object  # copy of the PinholeCameraCfg
    pos: tuple
    rot: tuple
    convention: str
    width: int
    height: int


SPECS: list[CamSpec] = []


def take_specs(cfg, view_cameras, remove: bool) -> list[str]:  # noqa: V103 — used by dexengine/rl/render/replay.py
    """Called by configure_scene: take the canonical camera cfgs as SPECS. remove = drop the per-env camera sensors from
    the scene cfg (replay render: only the pool draws). The check (check) is kept and drawn beside the pool for
    comparison."""
    import copy

    from isaaclab.sensors import CameraCfg

    SPECS.clear()
    for view, attr in view_cameras:
        c = getattr(cfg.scene, attr, None)
        if c is None:
            continue
        if type(c) is not CameraCfg or c.spawn is None:
            raise RuntimeError(
                f"[REPLAY_POOL] {attr} is not a CameraCfg with a spawn ({type(c).__name__}): no rule to move it"
            )
        SPECS.append(
            CamSpec(
                view,
                c.prim_path,
                copy.deepcopy(c.spawn),
                tuple(c.offset.pos),
                tuple(c.offset.rot),
                c.offset.convention,
                int(c.width),
                int(c.height),
            )
        )
        if remove:
            setattr(cfg.scene, attr, None)
    if not SPECS:
        raise RuntimeError("[REPLAY_POOL] replay mode without cameras: --enable_cameras · --render_per_clip_videos")
    return [f"{s.view}({s.width}x{s.height})" for s in SPECS]


def n_slots(n_views: int, width: int, height: int, n_clips: int) -> int:  # noqa: V103 — used by dexengine/rl/render/replay.py
    """Slot count: an explicit SLOTS_ENV wins. Otherwise the tile pixel budget / (views x pixels), at most the clip
    count."""
    raw = os.environ.get(SLOTS_ENV, "").strip()
    m = int(raw) if raw else int(TILE_MPIX_BUDGET * 1e6 // (n_views * width * height))
    if m < 1:
        raise RuntimeError(f"[REPLAY_POOL] slots {m} < 1: {SLOTS_ENV}={raw!r} · budget {TILE_MPIX_BUDGET} MP")
    return max(1, min(m, n_clips))


def schedule(lengths: list[int], m: int) -> list[list[tuple[int, int]]]:  # noqa: V103 — used by dexengine/rl/render/replay.py
    """Rounds by length: clips grouped m at a time in descending length; every round all slots move to new clips
    together (round length = the longest clip of the group; cutting the sorted list minimises the sum of round
    lengths). Returns [slot][(clip index, start frame)].
    Why rounds (not moving each slot separately like LPT): a moved slot must draw more until the RTX temporal state
    (DLSS) converges (right after a move 29.6 dB -> 37.7 after 32 more draws, 37.9 after 64 = the noise floor between
    tiles), and the tiled render draws every slot at once, so that cost grows with the number of move events; with
    rounds, events = number of rounds."""
    order = sorted(range(len(lengths)), key=lambda j: (-lengths[j], j))
    out: list[list[tuple[int, int]]] = [[] for _ in range(m)]
    t0 = 0
    for r in range(0, len(order), m):
        grp = order[r : r + m]
        for s, j in enumerate(grp):
            out[s].append((j, t0))
        t0 += lengths[grp[0]]
    return out


class CameraPool:  # noqa: V102 — used by dexengine/rl/render/replay.py
    """One TiledCamera per view (M slots; prims outside the envs {POOL_ROOT}/{view}_{m}/Cam)."""

    def __init__(self, raw_env, m: int):
        import torch

        import omni.usd
        from pxr import UsdGeom

        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        from isaaclab.utils.math import convert_camera_frame_orientation_convention as conv

        self.env, self.m, self.dev = raw_env, m, raw_env.device
        stage = omni.usd.get_context().get_stage()
        self.cams, self.anchor = [], []
        for s in SPECS:
            for k in range(m):
                UsdGeom.Xform.Define(stage, f"{POOL_ROOT}/{s.view}_{k}")
            cam = TiledCamera(
                TiledCameraCfg(
                    prim_path=f"{POOL_ROOT}/{s.view}_.*/Cam",
                    spawn=s.spawn,
                    offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
                    width=s.width,
                    height=s.height,
                    data_types=["rgb"],
                    update_period=0.0,
                )
            )
            cam._initialize_impl()  # the timeline already runs: do not wait for the PLAY callback
            cam._is_initialized = True
            q_off = conv(torch.tensor(s.rot, dtype=torch.float32).unsqueeze(0), origin=s.convention, target="opengl")[0]
            self.cams.append((s.view, cam))
            self.anchor.append(self._anchor_of(stage, s, torch.tensor(s.pos, dtype=torch.float32), q_off))
        desc = " · ".join(f"{v}→{a[3]}" for (v, _c), a in zip(self.cams, self.anchor, strict=True))
        print(
            f"[REPLAY_POOL] slots {m} x views {len(self.cams)} = tiles {m * len(self.cams)} · places {desc}", flush=True
        )

    def _anchor_of(self, stage, s: CamSpec, pos, q_off):
        """Walk up from the canonical camera parent (env_0) to an articulation link: when found ("body", articulation,
        link index, description, rel), else relative to the env prim ("env", None, -1, description, rel). rel = base
        -> camera (position, wxyz)."""
        import torch

        from pxr import Gf, UsdGeom

        from isaaclab.utils.math import quat_apply, quat_mul

        env0 = "/World/envs/env_0"
        parent = s.prim_path.replace("{ENV_REGEX_NS}", env0).rsplit("/", 1)[0]
        xc = UsdGeom.XformCache()
        for name, art in self.env.scene.articulations.items():
            # after the scene is built the cfg paths are resolved as regexes (/World/envs/env_.*/…): both to env_0
            root = art.cfg.prim_path.replace("{ENV_REGEX_NS}", env0).replace("/World/envs/env_.*", env0)
            if not parent.startswith(root + "/"):
                continue
            p = parent
            while p != root and p.rsplit("/", 1)[1] not in art.body_names:
                p = p.rsplit("/", 1)[0]
            if p == root:
                raise RuntimeError(f"[REPLAY_POOL] {s.view}: no {name} link above parent {parent}")
            base, kind, bi = p, "body", art.body_names.index(p.rsplit("/", 1)[1])
            break
        else:
            base, kind, art, bi = env0, "env", None, -1
        # USD matrices use the row-vector convention (world point = local point · M): parent of the base frame =
        # M_parent · M_base^-1
        rel = (
            xc.GetLocalToWorldTransform(stage.GetPrimAtPath(parent))
            * xc.GetLocalToWorldTransform(stage.GetPrimAtPath(base)).GetInverse()
        )
        rel = Gf.Matrix4d(rel).RemoveScaleShear()
        t, q = rel.ExtractTranslation(), rel.ExtractRotationQuat()
        rp = torch.tensor([t[0], t[1], t[2]], dtype=torch.float32)
        rq = torch.tensor([q.GetReal(), *q.GetImaginary()], dtype=torch.float32)
        cp = rp + quat_apply(rq.unsqueeze(0), pos.unsqueeze(0))[0]
        cq = quat_mul(rq.unsqueeze(0), q_off.unsqueeze(0))[0]
        what = f"{base.rsplit('/', 1)[1]}({'link · every frame' if kind == 'body' else 'env fixed'})"
        return (kind, art, bi, what, cp.to(self.dev), cq.to(self.dev))

    def _poses(self, i: int, env_ids):
        import torch

        from isaaclab.utils.math import convert_quat, quat_apply, quat_mul

        kind, art, bi, _w, cp, cq = self.anchor[i]
        n = len(env_ids)
        if kind == "env":
            bp = self.env.scene.env_origins[env_ids]
            bq = torch.tensor([1.0, 0, 0, 0], device=self.dev).expand(n, 4)
        else:
            tf = art.root_physx_view.get_link_transforms()[env_ids, bi]
            bp, bq = tf[:, :3], convert_quat(tf[:, 3:7], to="wxyz")
        return bp + quat_apply(bq, cp.expand(n, 3)), quat_mul(bq, cq.expand(n, 4))

    def place(self, slot_ids, env_ids, which: str) -> None:
        """Slot -> env place. which = "static" (env-fixed views, when the slot changes env) | "dynamic" (link-based
        views, every frame: after the state is written, a PhysX kinematic update, then the link pose is read)."""
        idx = [i for i, a in enumerate(self.anchor) if (a[0] == "body") == (which == "dynamic")]
        if not idx or not len(slot_ids):
            return
        if which == "dynamic":
            self.env.sim.physics_sim_view.update_articulations_kinematic()
        for i in idx:
            p, q = self._poses(i, env_ids)
            self.cams[i][1].set_world_poses(p, q, env_ids=slot_ids, convention="opengl")

    def render(self) -> None:
        self.env.sim.render()
        for _v, c in self.cams:
            c.update(0.0, force_recompute=True)
