"""Desk-clearance sidecar of the reach map, measured with the judge's desk ruler (desk component only).

Why: the map stores clearance as the z of link frame **origins** (`build.py`, `ca_a = _pa[..., 2]`), while the
judge measures the deepest penetration of the collider **surface** (capsule radius · hull) into the desk footprint
(`collision.desk_only_each`). The two are different quantities: a forearm origin can be above the top while its
surface dips below it, so ordering by the origin ruler disagrees with the judge.

What: the map already stores the joint values vq of every (cell, orientation atom, valley). They are expanded by FK
and measured with the judge's own function (`desk_clearance`); the result is written as a sidecar file per z layer.
The original shards are not touched.

  clearance (robot-local z) = desk_rel − desk penetration,   desk_rel = ztop − base_z

Readers (the judge's inclusion and ordering maps) test each dz layer against `desk_bar(desk_rel, dz, clearance)`:
lowering the robot by dz raises the desk by dz in robot coordinates, so the desk depth shifts by exactly dz. The
torso is deliberately left out: torso overlap moves with the robot and does not depend on dz, and mixing it into the
same scalar would shift the torso condition by dz and make the filter stricter than the judge. The torso is checked
by the judge's search scalar and exact guard.

Identity: each sidecar carries two header fields (SIDECAR_KEYS) — the ruler (`collision.ruler_id`, a syntax hash of
the code that defines the desk ruler) and the arm collider digest (`sim_collider.measured_digest`) — and two source
fields (SOURCE_KEYS): the map it was measured on (`store.pool_id`, which carries the map's wrist-input record) and the
sha256 of the shard whose joint values it expanded. The reader (`check_sidecar`, called by `store.load_pool`) stops
when any differs from the current one, a header is missing or a layer is missing, so an old sidecar never attaches
to a rebuilt map or shard. A layer measured before the source fields is read only when the map is a locked input
(`store.lock_state`). An existing layer is skipped only when its header and source equal the current ones.

Usage:
  clearance.py --pool-root <pool_root> --arm rby1 --hand sharpa_wave [--sides right,left] [--worker i/N]
  → <pool>/<combo>_<side>/clrcap_z{k:03d}.npz  (vclr_arm_cap: (Pn,A,K) float16 · desk_rel · ruler · collider_digest)
"""

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError

#: Sidecar header — the ruler and the arm collider shape this clearance was measured with (written by rebake_side,
#: read by check_sidecar).
SIDECAR_KEYS = ("ruler", "collider_digest")
#: Per-layer source — the map id and the source shard's sha256 (first 16 hex) the layer was measured from (written by
#: rebake_side, read by sidecar_problem; module doc 'Identity').
SOURCE_KEYS = ("pool_id", "shard_sha")
#: Clearance (m) of the judge's loose inclusion map — the default of the judge's `--arm-margin-cap`. The sidecar
#: is one frame on the map's grid poses, so it can read shallower than the judge's chord and polished poses; it is
#: opened by that much (it must stay looser than −collision.DESK_SURF_PEN_MM; the judge checks this).
LOOSE_MARGIN_M = -0.10


def desk_clearance(SCL, ctx: dict, q) -> np.ndarray:
    """Sidecar value (robot-local z) = desk_rel − desk component (collision.desk_only_each); no torso."""
    return (float(ctx["ztop"]) - float(ctx["base"][2])) - SCL.desk_only_each(ctx, q)


def desk_bar(desk_rel: float, dz_m: float, margin: float) -> float:
    """The value a sidecar entry must exceed at dz layer `dz_m` (robot height offset). Lowering the robot by dz
    raises the desk by dz in robot coordinates (only the desk depth moves by dz)."""
    return (desk_rel - dz_m) + margin


def sidecar_identity(arm: str) -> dict:
    """The ruler and arm shape the judge uses now (owners: collision · sim_collider)."""
    from dexengine.placement.collision import core as SCL  # noqa: PLC0415
    from dexengine.placement.collision import sim_collider as SC  # noqa: PLC0415

    return {"ruler": SCL.ruler_id(), "collider_digest": SC.measured_digest(arm)}


TOP_TAG = "[CLRCAP_TOP]"


def check_desk_top(pool_dir, desk_top) -> None:
    """The map clearances (arm sidecar · the hand clearance vclr_hand of the shards) were measured on the
    **tabletop** desk top (`coords.desk_top_for(None)`). Stop when the reader's desk top (that dataset's
    `coords.desk_top_for`) differs — there is no clearance for that top, and the tabletop one is not substituted.
    Prints one `[CLRCAP_TOP]` line."""
    from dexengine.placement.coords import core as VC  # noqa: PLC0415

    measured = tuple(float(v) for v in VC.desk_top_for(None))
    want = tuple(float(v) for v in desk_top)
    if want != measured:
        raise InputMissingError(
            f"{TOP_TAG} ✗ {pool_dir}: this dataset's desk top (x_off, half_x, half_y) = {want} != "
            f"the top the map clearances were measured on {measured} (tabletop) — there is no arm "
            "sidecar (clrcap) or hand clearance (vclr_hand) for that top"
        )
    print(
        f"{TOP_TAG} {Path(pool_dir).name} desk top (x_off, half_x, half_y) = {measured} — same as the map's", flush=True
    )


def sidecar_path(pool_dir: Path, k: int) -> Path:
    """Sidecar file of z layer k."""
    return Path(pool_dir) / f"clrcap_z{k:03d}.npz"


def layer_source(pool_dir: Path, k: int, pid: str) -> dict:
    """The source a layer k sidecar must carry now: the map id `pid` (store.pool_id) and its shard's sha256."""
    with open(Path(pool_dir) / f"shard_z{k:03d}.npz", "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    return {"pool_id": pid, "shard_sha": sha}


def _layer_problems(pool_dir: Path, arm: str) -> tuple:
    """(map id, layer count, present layers, layers with another header, with another source, without a source)."""
    from dexengine.placement.reach_map import store as _PM  # noqa: PLC0415 — store imports this module

    pool_dir = Path(pool_dir)
    nz = len(np.load(pool_dir / "meta.npz", allow_pickle=True)["axes_z"])
    want, pid = sidecar_identity(arm), _PM.pool_id(pool_dir)
    have = [k for k in range(nz) if sidecar_path(pool_dir, k).is_file()]
    bad, other, unrec = [], [], []
    for k in have:
        with np.load(sidecar_path(pool_dir, k)) as z:
            got = {key: (str(z[key]) if key in z.files else None) for key in SIDECAR_KEYS + SOURCE_KEYS}
        if {key: got[key] for key in SIDECAR_KEYS} != want:
            bad.append((k, {key: got[key] for key in SIDECAR_KEYS}))
        src = {key: got[key] for key in SOURCE_KEYS}
        if all(v is None for v in src.values()):
            unrec.append(k)
        elif src != layer_source(pool_dir, k, pid):
            other.append((k, src))
    return pid, nz, have, bad, other, unrec


def sidecar_problem(pool_dir: Path, arm: str) -> str | None:
    """Is the sidecar of one map side usable? None = every layer (count = len(meta axes_z)) exists, its header equals
    the current ruler and shape and its source equals this map and shard (a layer without a source only on a locked
    map); otherwise a one-line reason (layer count · layers with another header · another source · no source)."""
    return _reason(pool_dir, arm, _layer_problems(pool_dir, arm))


def _reason(pool_dir: Path, arm: str, parts: tuple) -> str | None:
    """sidecar_problem's verdict on `_layer_problems` output."""
    from dexengine.placement.reach_map import store as _PM  # noqa: PLC0415 — store imports this module

    pid, nz, have, bad, other, unrec = parts
    unrec_bad = unrec if unrec and not _PM.lock_state(pool_dir) else []
    if len(have) == nz and not bad and not other and not unrec_bad:
        return None
    return (
        f"sidecar layers {len(have)}/{nz} · layers with another ruler/shape {len(bad)} (e.g. {bad[:1]}) · "
        f"measured on another map or shard {len(other)} (e.g. {other[:1]}) · without a source on a map that is not "
        f"a locked input {len(unrec_bad)} — now {sidecar_identity(arm)} · map {pid}"
    )


def check_sidecar(pool_dir: Path, arm: str) -> None:
    """Stop when sidecar_problem reports one (an old clearance is never substituted). Prints one `[CLRCAP_ID]`
    line; `scripts/build_reach_map.sh` matches its `[CLRCAP_ID] <map side> ` prefix."""
    parts = _layer_problems(pool_dir, arm)
    why = _reason(pool_dir, arm, parts)
    if why:
        raise SystemExit(
            f"[CLRCAP_ID] ✗ {pool_dir}: {why}. Rebuild: clearance.py --pool-root "
            f"{Path(pool_dir).parent} --arm {arm} --hand <hand> (only layers with another header or source)"
        )
    want = sidecar_identity(arm)
    pid, unrec = parts[0], parts[-1]
    print(
        f"[CLRCAP_ID] {Path(pool_dir).name} ruler={want['ruler']} collider={want['collider_digest']} pool={pid}"
        f"{f' source=unrecorded on {len(unrec)} layers (locked input)' if unrec else ''}",
        flush=True,
    )


def rebake_side(SCL, pool_dir: Path, arm: str, hand: str, side: str, layers, chunk: int, force: bool) -> int:
    """Write the sidecar of the given z layers of one map side; returns the number of layers written."""
    ctx = SCL.build_arm_ctx(arm, hand, side)
    desk_rel = float(ctx["ztop"]) - float(ctx["base"][2])
    from dexengine.placement.reach_map import store as _PM  # noqa: PLC0415 — store imports this module

    ident, pid = sidecar_identity(arm), _PM.pool_id(pool_dir)
    done = 0
    for k in layers:
        out = sidecar_path(pool_dir, k)
        src = pool_dir / f"shard_z{k:03d}.npz"
        if out.exists() and not force and src.exists():
            with np.load(out) as old:
                got = {key: (str(old[key]) if key in old.files else None) for key in SIDECAR_KEYS + SOURCE_KEYS}
            if got == {**ident, **layer_source(pool_dir, k, pid)}:
                print(f"[clrcap] {side} z{k:03d} same ruler/shape/source — skipped", flush=True)
                continue
        if not src.exists():
            # A meta layer without a shard means a broken map — die instead of skipping (a skip would leave old
            # sidecars on the other layers behind an rc 0). Completion is checked by check_sidecar.
            raise InputMissingError(
                f"[clrcap] ✗ {pool_dir.name} z{k:03d}: shard missing ({src}) — broken map (meta layer ↔ shard)"
            )
        t0 = time.time()
        vq = np.load(src)["vq"]  # (Pn, A, K, n)
        Pn, A, K, nj = vq.shape
        q = vq.reshape(-1, nj).astype(np.float64)
        # Invalid poses (NaN) are not measured and keep the map's invalid sentinel (−10); outside the footprint
        # (penetration −1e9) is exempt (clipped to +10).
        bad = ~np.isfinite(vq.reshape(-1, nj).astype(np.float32)).all(1)
        clr = np.full(len(q), -10.0, np.float32)
        ok = np.flatnonzero(~bad)
        for c0 in range(0, len(ok), chunk):
            ii = ok[c0 : c0 + chunk]
            clr[ii] = desk_clearance(SCL, ctx, q[ii])
        clr = np.clip(clr, -10.0, 10.0)
        tmp = out.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp,
            vclr_arm_cap=clr.reshape(Pn, A, K).astype(np.float16),
            desk_rel=np.float64(desk_rel),
            **{key: np.str_(ident[key]) for key in SIDECAR_KEYS},
            **{key: np.str_(v) for key, v in layer_source(pool_dir, k, pid).items()},
        )
        os.replace(tmp, out)
        done += 1
        print(
            f"[clrcap] {side} z{k:03d}: {len(q)} poses {time.time() - t0:.1f}s "
            f"→ {out.name} [CLRCAP_ID] ruler={ident['ruler']} collider={ident['collider_digest']} pool={pid}",
            flush=True,
        )
    return done


def main() -> int:
    """Write the sidecar layers of this worker for both sides of one combo's map."""
    from dexengine.core import paths as _P  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    # Map root owner = paths.pool_root (env DEXENGINE_POOL_ROOT first).
    ap.add_argument("--pool-root", default=str(_P.pool_root()))
    # No arm/hand defaults: a default hand would build another combo's sidecar when the argument is forgotten.
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--sides", default="right,left")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--worker", default="0/1", help="i/N — split the layers across parallel workers")
    ap.add_argument("--force", action="store_true", help="rebuild even when the header is already current")
    a = ap.parse_args()

    from dexengine.placement.collision import core as SCL  # noqa: PLC0415

    wi, wn = (int(x) for x in a.worker.split("/"))
    combo = f"{a.arm}_{a.hand}"
    t0 = time.time()
    tot = 0
    from dexengine.placement import arm_ik as AIP  # noqa: PLC0415
    from dexengine.placement.reach_map import (
        store as _PM,  # noqa: PLC0415 — map identity check (store imports this module)
    )

    for side in a.sides.split(","):
        pd = Path(a.pool_root) / f"{combo}_{side}"
        _combo = AIP.resolve_combo(a.arm, a.hand, side)
        nz = int(
            len(
                _PM.verify(
                    pd,
                    arm=a.arm,
                    hand=a.hand,
                    side=side,
                    base_xyz=_combo[3],
                    palm_off=AIP.palm_frame_offset(a.hand),
                    wrist_dir=_combo[4],
                    who="clrcap",
                )["axes_z"]
            )
        )
        layers = list(range(nz))[wi::wn]
        print(f"[clrcap] {pd.name}: layers {len(layers)}/{nz} (worker {wi}/{wn}) · desk component only", flush=True)
        tot += rebake_side(SCL, pd, a.arm, a.hand, side, layers, a.chunk, a.force)
    print(f"[clrcap] done {tot} layers · {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
