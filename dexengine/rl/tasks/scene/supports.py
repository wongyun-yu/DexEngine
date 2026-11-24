"""Owner of the support object fixed-slot reset (rule D42): the table, the asset identity guard (spawned asset = reset
clip asset), the write call, and the spawn shape guard of the support disc slots (`check_disc_spawn`; the disc write
itself is object_pin.reset_discs).
The one writer (`object_pin.write_slots`) and the slot name prefix (`object_pin.SUPPORT_OBJ_PREFIX`) are owned by
object_pin; this module reads them (only the reset moved here, to keep object_pin small)."""

from __future__ import annotations

import torch

from dexengine.rl.tasks.scene.object_pin import SUPPORT_OBJ_PREFIX, write_slots

DISC_ON_Z = (
    -5.0
)  # real disc = top z above this value (the same threshold as object_pin.support_disc_rows' on and reset_discs)


def reset_supports(env, env_ids: torch.Tensor) -> None:
    """Reset (`events.reset_support_objects`): write the support object slots of env_ids at their clip's first-frame
    scene pose (empty slots at the hidden pose); kinematic, so they stay until the end of the episode (no lifetime).
    Table = event params `poses` (C, K, 7) · asset identity `assets` (C, K) = that clip slot's [obj_id, USD] (None when
    absent) (writer `env_setup.supports._build_support_slots`).
    Slot assets were bound at spawn to clip i % C for env i; when the reset clip's support asset differs from the
    spawned one (identity comparison, `support_asset_ok`) another object would stand, so it fails. Clips using the same
    object (the original and its placement variants; a render session pins clip k on one env) differ in pose only. No
    identity statement fails. Slot names must be every support_obj_* in the scene."""
    if "_support_obj" not in env.__dict__:
        prm = env.event_manager.get_term_cfg("reset_support_objects").params
        t = torch.tensor(prm["poses"], dtype=torch.float32, device=env.device)
        tm = env.trajectory_manager
        scene = sorted(str(k) for k in env.scene.keys() if str(k).startswith(SUPPORT_OBJ_PREFIX))  # noqa: SIM118  — InteractiveScene has no __contains__
        if t.ndim != 3 or t.shape[-1] != 7 or t.shape[0] != tm.num_clips or scene != sorted(prm["names"]):
            raise RuntimeError(
                f"[SUPPORT] support table {tuple(t.shape)} · clips {tm.num_clips} · slots {prm['names']} ≠ scene "
                f"{scene} — (clip, slot, 7) · order contract broken"
            )
        assets = prm.get("assets")
        if assets is None or len(assets) != t.shape[0] or any(len(r) != t.shape[1] for r in assets):
            raise RuntimeError(
                "[SUPPORT] no support asset identity statement (params assets (clip, slot)) or wrong shape — "
                "the spawned assets cannot be checked against the reset clip"
            )
        env._support_obj = (list(prm["names"]), t, support_asset_ok(assets).to(env.device))
    names, t, ok = env._support_obj
    ci = env.trajectory_manager.clip_ids[env_ids]
    bad = ~ok[env_ids % t.shape[0], ci]
    if bool(bad.any()):
        raise RuntimeError(
            f"[SUPPORT] support assets of env->clip {ci[bad][:4].tolist()} ≠ spawned assets (clip i % "
            f"{t.shape[0]} · env {env_ids[bad][:4].tolist()}) — the support asset is another clip's object"
        )
    write_slots(env, env_ids, names, t[ci])


def support_asset_ok(assets: list) -> torch.Tensor:
    """(C, C) bool: [spawn clip a, reset clip b] = for every slot, b has no support asset (it writes the hidden pose) or
    the same identity ([obj_id, USD]) as the asset a spawned. False when a's slot is an empty place (a hidden stand-in
    of
    another clip's body) and b uses that slot."""
    n = len(assets)
    return torch.tensor(
        [
            [all(nb is None or na == nb for na, nb in zip(assets[a], assets[b], strict=True)) for b in range(n)]
            for a in range(n)
        ],
        dtype=torch.bool,
    )


DISC_MIN_R = 0.001  # spawn radius floor (the same value as max(r, 0.001) in env_setup.supports._build_disc_slots)
SHOWN, HIDDEN = (
    "inherited",
    "invisible",
)  # UsdGeom.Imageable visibility tokens (the values MakeVisible · MakeInvisible write)


def disc_geom(st: dict, ci: torch.Tensor) -> tuple:
    """Disc shapes (n, K) of clip ci: radius (floor DISC_MIN_R) · column height · visible (a real disc and the session's
    visibility statement true)."""
    t = st["table"][ci]
    vis = (t[..., 2] > DISC_ON_Z) & bool(st.get("visible"))
    return t[..., 3].clamp(min=DISC_MIN_R), st["h"].to(t)[ci], vis


def _mesh(env, st: dict, i: int, k: int):
    leaf = env.scene[st["names"][k]].cfg.prim_path.rsplit("/", 1)[1]
    root = env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[i]}/{leaf}")
    return root, env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[i]}/{leaf}/geometry/mesh")


def _read(env, st: dict, i: int, k: int) -> tuple:
    root, mesh = _mesh(env, st, i, k)
    return (
        float(mesh.GetAttribute("radius").Get()),
        float(mesh.GetAttribute("height").Get()),
        root.GetAttribute("visibility").Get() != HIDDEN,
    )


def fit_disc_geometry(env, st: dict, env_ids: torch.Tensor, ci: torch.Tensor) -> int:
    """Discs at each clip's own size within a session: disc prims spawn as clip i % C's cylinders for env i. When the
    reset clip differs from the clip of the disc now standing in that env (a render session with one env, clip k
    pinned), that env's disc prim radius · height (USD Cylinder radius · height) · visibility are rewritten from the
    reset clip's stamp; training and stage 1 (reset clip = i % C) never write. Guard: on the first rewrite the spawn
    shapes of every env are checked once against clip i % C's stamp, and written envs are read back against the reset
    clip's stamp; a mismatch fails. Paths with nothing to rewrite (training, stage 1) never read the USD.
    A rewrite without the visibility statement (params visible) fails. The render writes without physics, so the
    collision shape stays. A rewrite prints one `[<tag>][discs] disc size fit` line. -> the number of rewritten envs."""
    C, K = st["table"].shape[:2]
    if "cur" not in st:  # clip of the disc now standing = spawn assignment i % C
        st["cur"] = torch.arange(env.num_envs) % C
    ids, cs = env_ids.tolist(), ci.tolist()
    todo = [(i, c) for i, c in zip(ids, cs, strict=True) if int(st["cur"][i]) != c]
    if not todo:
        return 0
    if not st.get("spawn_checked"):  # once, on the first rewrite: every env's spawn shape = clip i % C's stamp
        r, h, _v = disc_geom(st, torch.arange(env.num_envs) % C)
        for i in range(env.num_envs):
            for k in range(K):
                got = _read(env, st, i, k)
                if abs(got[0] - float(r[i, k])) > 1e-6 or abs(got[1] - float(h[i, k])) > 1e-6:
                    raise RuntimeError(
                        f"[{st['tag']}][discs] env {i} slot {k} spawned disc r {got[0]:.4f} · h "
                        f"{got[1]:.4f} ≠ clip {i % C} stamp r {float(r[i, k]):.4f} · h "
                        f"{float(h[i, k]):.4f} — the spawn assignment i % C contract is broken"
                    )
        st["spawn_checked"] = True
    if st.get("visible") is None:
        raise RuntimeError(
            f"[{st['tag']}][discs] no disc visibility statement (params visible) — cannot rewrite at the "
            "reset clip's size"
        )
    r, h, v = disc_geom(st, torch.tensor([c for _, c in todo]))
    for j, (i, c) in enumerate(todo):
        for k in range(K):
            root, mesh = _mesh(env, st, i, k)
            mesh.GetAttribute("radius").Set(float(r[j, k]))
            mesh.GetAttribute("height").Set(float(h[j, k]))
            root.GetAttribute("visibility").Set(SHOWN if bool(v[j, k]) else HIDDEN)
            got = _read(env, st, i, k)
            if abs(got[0] - float(r[j, k])) > 1e-6 or abs(got[1] - float(h[j, k])) > 1e-6 or got[2] != bool(v[j, k]):
                raise RuntimeError(
                    f"[{st['tag']}][discs] env {i} slot {k} disc after reset (r {got[0]:.4f} · h "
                    f"{got[1]:.4f} · visible {got[2]}) ≠ clip {c} stamp (r {float(r[j, k]):.4f} · h "
                    f"{float(h[j, k]):.4f} · visible {bool(v[j, k])}) — a disc of another size would stand"
                )
        st["cur"][i] = c
    print(
        f"[{st['tag']}][discs] disc size fit env {[i for i, _ in todo][:4]} -> clip {[c for _, c in todo][:4]} · r "
        f"{[round(float(x), 4) for x in r[0]]} · read-back equal",
        flush=True,
    )
    return len(todo)
