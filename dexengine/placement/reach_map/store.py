#!/usr/bin/env python3
"""Reach map store — load one side of the map (`load_pool`) and check its identity (`verify`).

A map side (`<pool_root>/<combo>_<side>/`) holds the coarse maps (pos_err · rot_err · clearance), the valley maps
(vpe · vre · vclr_arm · vclr_hand · joint values vq) and the desk-clearance sidecar (`clrcap_z*.npz`, see
`clearance`). The map is a function of (arm, hand, side, robot base, palm-frame offset): the builder turns wrist
targets into `P + R·palm_off` before solving. Every reader therefore checks the meta fields `arm · hand · side ·
base_xyz · palm_off` against the values it uses now and dies on any missing or different field; a passing check
prints `[POOL_ID]`. The expected values come from the caller (`arm_ik.resolve_combo` base, `arm_ik.palm_frame_offset`).

Inputs: the grid bounds and orientation atoms are a function of the wrist-cache set the builder samples
(`build._sample_wrists` over `arm_ik.resolve_combo`'s wrist_dir), so a map built from an older set is another map
even when every identity field agrees. The builder writes the set's content digest (`wrist_inputs` · INPUTS_KEY) into
the meta; every reader recomputes it from the wrist_dir the builder would read now and dies when it differs or is
unrecorded (a map built before the field). One exception, named by the caller: a map listed in the setting
`DEXENGINE_POOL_LOCK` (`<map dir>@<12 hex of sha256(meta.npz)>`, comma list · `pool_id`) is a **locked input** — used
as is, with a loud `[POOL] locked=<id> stale_inputs=yes|no` line; a lock that names the dir with another meta dies.
The map id a stage used is `pool_id`; the judge writes it per side into its plan lines, the bake carries it into the
tree's task_info provenance (PROV_KEY) and the coarse path compares it with the map it verified (`check_recorded`).

CLI (identity check, rc 0 = match, 3 = mismatch):
  store.py --pool-dir <map side> --arm <arm> --hand <hand> --side right|left
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

from dexengine.core import config as _CFG
from dexengine.core import paths as _P
from dexengine.placement.reach_map import clearance as CLR

#: Identity fields every map meta must carry — the names the builder (`build.py`, np.savez) writes.
IDENTITY_FIELDS = ("arm", "hand", "side", "base_xyz", "palm_off")
#: Meta fields of the builder's input — the wrist-cache set's content digest (`wrist_inputs`) and its file count.
#: The builder reads every len // WRIST_SAMPLES-th wrist cache of the sorted set (`wrist_files`).
WRIST_SAMPLES = 300
INPUTS_KEY, INPUTS_N_KEY = "wrist_inputs", "wrist_inputs_n"
#: The one named setting that admits a map whose inputs are stale or unrecorded (module doc 'Inputs').
LOCK_ENV = "DEXENGINE_POOL_LOCK"
LOCK_TAG = "[POOL]"
#: task_info provenance key of the map ids the tree was placed with ({side: pool_id}) — written by the bake from the
#: judge's plan line, read by the coarse path (`check_recorded`).
PROV_KEY = "pool"
_LOCK_ITEM = re.compile(r"([^@,\s]+)@([0-9a-f]{12})")


class PoolIdentityError(RuntimeError):
    """Map identity mismatch. A RuntimeError (not SystemExit) so the stage runner reports it as an exception."""


class WristSetMissing(PoolIdentityError):
    """The wrist-cache set a map's inputs are computed from is absent (no `*.npz` in the builder's wrist_dir)."""


def wrist_files(wrist_dir) -> tuple[list, list]:
    """(every wrist cache, the ones the builder reads): the sorted `*.npz` of wrist_dir and every
    len // WRIST_SAMPLES-th of them — the one selection rule (build._sample_wrists reads this)."""
    files = sorted(Path(wrist_dir).glob("*.npz"))
    return files, files[:: max(1, len(files) // WRIST_SAMPLES)]


def wrist_inputs(wrist_dir) -> tuple[str, int]:
    """(content digest, file count) of the map's input: sha256 over the sorted names of every wrist cache (they fix
    which files are read) and the lines `<name>\\t<sha256 of its bytes>` of the files the builder reads
    (`wrist_files`). A file the builder does not read changes no map; any added, removed or renamed file, or a changed
    read file, does. A missing or empty set dies — a map has no inputs to compare with."""
    files, picked = wrist_files(wrist_dir)
    if not files:
        raise WristSetMissing(f"{LOCK_TAG} no wrist cache files in {wrist_dir} — the reach map's input set is empty")
    h = hashlib.sha256()
    h.update("".join(f"{f.name}\n" for f in files).encode())
    for f in picked:
        h.update(f"{f.name}\t{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest(), len(files)


def pool_id(pdir) -> str:
    """`<map dir name>@<12 hex of sha256(meta.npz)>` — the id a lock names and a tree records."""
    pdir = Path(pdir)
    return f"{pdir.name}@{hashlib.sha256((pdir / 'meta.npz').read_bytes()).hexdigest()[:12]}"


def locked_ids() -> dict:
    """The setting DEXENGINE_POOL_LOCK as {map dir name: meta sha12}. An entry not of the form `<dir>@<12 hex>`, or a
    dir named twice with two shas, dies (an unread lock would be a silent fallback)."""
    out = {}
    for item in (x.strip() for x in _CFG.env(LOCK_ENV).split(",")):
        if not item:
            continue
        m = _LOCK_ITEM.fullmatch(item)
        if m is None:
            raise PoolIdentityError(
                f"{LOCK_TAG} {LOCK_ENV} entry {item!r} is not <map dir>@<12 hex of sha256(meta.npz)>"
            )
        if out.get(m[1], m[2]) != m[2]:
            raise PoolIdentityError(f"{LOCK_TAG} {LOCK_ENV} names {m[1]} twice with two metas ({out[m[1]]} · {m[2]})")
        out[m[1]] = m[2]
    return out


def lock_state(pdir) -> bool:
    """Is this map side a locked input? A lock that names its dir with another meta dies."""
    pdir = Path(pdir)
    want = locked_ids().get(pdir.name)
    if want is None:
        return False
    pid = pool_id(pdir)
    if pid.rsplit("@", 1)[1] != want:
        raise PoolIdentityError(
            f"{LOCK_TAG} {LOCK_ENV} locks {pdir.name}@{want} but the map at {pdir} is {pid} — "
            "the lock names another map"
        )
    return True


def _check_inputs(pdir: Path, meta, wrist_dir, who: str) -> str:
    """The map's recorded inputs against the set now (module doc 'Inputs'); returns the [POOL_ID] tail. A locked map
    passes even without a wrist set (stale_inputs=unknown): the lock is an explicit choice checked against the meta."""
    locked = lock_state(pdir)
    try:
        now, n = wrist_inputs(wrist_dir)
    except WristSetMissing:
        if not locked:
            raise
        now, n = None, 0
    have = str(meta[INPUTS_KEY]) if INPUTS_KEY in meta.files else None
    pid = pool_id(pdir)
    if locked:
        state = "unknown (no wrist set)" if now is None else ("no" if have == now else "yes")
        line = (
            f"{LOCK_TAG} locked={pid} stale_inputs={state} map_inputs={(have or 'unrecorded')[:12]} "
            f"now={(now or 'none')[:12]} ({n} files · {wrist_dir}) who={who}"
        )
        print(line, flush=True)
        print(line, file=sys.stderr, flush=True)
        return f"id={pid} LOCKED"
    if have != now:
        why = "unrecorded (built before the input fingerprint)" if have is None else f"{have[:12]} != now {now[:12]}"
        raise PoolIdentityError(
            f"[{who}] reach map inputs {why}: {pdir} was not built from the wrist-cache set {wrist_dir} ({n} files) — "
            f"its grid bounds and orientation atoms are another set's.\n  Rebuild it (`bash scripts/build_reach_map.sh "
            f"<arm> <hand>`, the old map moved aside) or name it as a locked input: {LOCK_ENV}={pid}"
        )
    return f"inputs={now[:12]} id={pid} OK"


def record_into(ti: dict, ids) -> None:
    """Write the judge's map ids ({side: pool_id}, from the plan line via adopted.json) into a tree's task_info
    provenance (PROV_KEY) — the bake's write; None (an old plan without ids) writes nothing."""
    if ids is not None:
        ti["provenance"] = {**(ti.get("provenance") or {}), PROV_KEY: {str(k): str(v) for k, v in dict(ids).items()}}


def recorded(ti: dict, side: str):
    """The map id a tree's task_info records for `side` (record_into), or None."""
    return ((ti.get("provenance") or {}).get(PROV_KEY) or {}).get(side)


def check_recorded(pdir, recorded, who: str) -> None:
    """The map id a tree records (task_info provenance PROV_KEY, this side) against the map verified now: a different
    id dies; no record is admitted only for a locked input (a tree placed before the record), with a [POOL] line."""
    pid = pool_id(pdir)
    if recorded is None:
        if not lock_state(pdir):
            raise PoolIdentityError(
                f"[{who}] the tree records no reach map id (task_info provenance.{PROV_KEY}) and {pid} is not a "
                f"locked input ({LOCK_ENV})"
            )
        print(f"{LOCK_TAG} locked={pid} tree_record=unrecorded who={who}", flush=True)
        return
    if str(recorded) != pid:
        raise PoolIdentityError(
            f"[{who}] the tree was placed with reach map {recorded} but the map now is {pid} ({pdir}) — another map"
        )


def verify(pdir, *, arm: str, hand: str, side: str, base_xyz, palm_off, wrist_dir, who: str):
    """Check the identity of `pdir/meta.npz` and its inputs against `wrist_dir` (the set the builder would read
    now: arm_ik.resolve_combo's wrist_dir) and return the meta; raise PoolIdentityError on any mismatch."""
    pdir = Path(pdir)
    mp = pdir / "meta.npz"
    if not mp.is_file():
        raise PoolIdentityError(f"[{who}] reach map meta missing: {mp}")
    meta = np.load(mp, allow_pickle=True)
    miss = [k for k in IDENTITY_FIELDS if k not in meta.files]
    if miss:
        raise PoolIdentityError(
            f"[{who}] reach map identity fields missing {miss}: {pdir}\n"
            f"  A map whose identity cannot be checked is not used. "
            f"Rebuild it with `bash scripts/build_reach_map.sh <arm> <hand>` (move the old map aside)."
        )
    got = {"arm": str(meta["arm"]), "hand": str(meta["hand"]), "side": str(meta["side"])}
    want = {"arm": str(arm), "hand": str(hand), "side": str(side)}
    try:  # a spec directory name is the same hand (paths.hand_id is idempotent)
        got["hand"], want["hand"] = _P.hand_id(got["hand"]), _P.hand_id(want["hand"])
    except KeyError as e:
        raise PoolIdentityError(f"[{who}] reach map hand name not in the registry: {e} ({pdir})") from None
    bad = [f"{k}: map={got[k]!r} != now={want[k]!r}" for k in ("arm", "hand", "side") if got[k] != want[k]]
    pb, wb = np.asarray(meta["base_xyz"], float), np.asarray(base_xyz, float)
    if pb.shape != wb.shape or not np.allclose(pb, wb, atol=1e-6):
        bad.append(
            f"base_xyz: map={pb.tolist()} != ArmSpec={wb.tolist()} — "
            f"the canonical robot base comes from `source configs/placement/canon_env.sh`"
        )
    po, wo = np.asarray(meta["palm_off"], float).reshape(-1), np.asarray(palm_off, float).reshape(-1)
    if po.shape != wo.shape or not np.allclose(po, wo, atol=1e-9):
        bad.append(
            f"palm_off: map={po.tolist()} != HandSpec.palm_frame_offset_pos={wo.tolist()} — the map was built "
            f"with another palm frame; move it aside and rebuild"
        )
    if bad:
        raise PoolIdentityError(f"[{who}] reach map identity mismatch ({pdir}):\n  " + "\n  ".join(bad))
    tail = _check_inputs(pdir, meta, wrist_dir, who)
    print(
        f"[POOL_ID] {who} {pdir.name} arm={got['arm']} hand={got['hand']} side={got['side']} "
        f"base={pb.round(4).tolist()} palm_off={po.round(5).tolist()} {tail}",
        flush=True,
    )
    return meta


def load_pool(d: Path, arm: str, *, desk_top):
    """One map side: coarse maps, valley maps and the desk-clearance sidecar (VCAC — the judge's desk-surface
    ruler, desk component only).

    A map with valley maps requires the sidecar; `clearance.check_sidecar` first checks its layer count, ruler and
    arm shape against the current ones. The map's own vclr_arm is the z of link **origins**, more optimistic than
    the judge's surface ruler, so it must not be used for ordering. `desk_top` = the reader's dataset desk top
    (coords.desk_top_for); `clearance.check_desk_top` stops when it differs from the top the clearances were
    measured on."""
    CLR.check_desk_top(d, desk_top)
    meta = np.load(d / "meta.npz", allow_pickle=True)
    ax, ay, az = meta["axes_x"], meta["axes_y"], meta["axes_z"]
    atoms = meta["atoms_xyzw"].astype(np.float64)
    nx, ny, nz, A = len(ax), len(ay), len(az), len(atoms)
    sh0 = np.load(d / "shard_z000.npz")
    has_v = "vpe" in sh0.files
    K = sh0["vpe"].shape[-1] if has_v else 0
    PE = np.empty((nx, ny, nz, A), np.float32)
    RD = np.empty((nx, ny, nz, A), np.float32)
    CL = np.empty((nx, ny, nz, A), np.float32)
    out = {
        "atoms": atoms,
        "lo": np.array([ax[0], ay[0], az[0]]),
        "step": float(meta["pos_step"]),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "A": A,
        "K": K,
    }
    if has_v:
        CLR.check_sidecar(d, arm)
        VPE = np.empty((nx, ny, nz, A, K), np.float16)
        VRE = np.empty((nx, ny, nz, A, K), np.float16)
        VCA = np.empty((nx, ny, nz, A, K), np.float16)
        VCH = np.empty((nx, ny, nz, A, K), np.float16)
        VCAC = np.empty((nx, ny, nz, A, K), np.float16)
    for k in range(nz):
        sh = np.load(d / f"shard_z{k:03d}.npz")
        PE[:, :, k] = sh["pos_err"].astype(np.float32).reshape(nx, ny, A)
        RD[:, :, k] = sh["rot_err"].astype(np.float32).reshape(nx, ny, A)
        CL[:, :, k] = sh["clearance"].astype(np.float32).reshape(nx, ny, A)
        if has_v:
            VPE[:, :, k] = sh["vpe"].reshape(nx, ny, A, K)
            VRE[:, :, k] = sh["vre"].reshape(nx, ny, A, K)
            VCA[:, :, k] = sh["vclr_arm"].reshape(nx, ny, A, K)
            VCH[:, :, k] = sh["vclr_hand"].reshape(nx, ny, A, K)
            VCAC[:, :, k] = np.load(CLR.sidecar_path(d, k))["vclr_arm_cap"].reshape(nx, ny, A, K)
    out.update(PE=PE, RD=RD, CL=CL)
    if has_v:
        out.update(VPE=VPE, VRE=VRE, VCA=VCA, VCH=VCH, VCAC=VCAC)
    return out


def main() -> int:
    """Shell entry: check one map side against the owners' current values (`arm_ik.resolve_combo` base ·
    `arm_ik.palm_frame_offset`). `scripts/build_reach_map.sh` calls it for an existing map before skipping it
    and after building, so a map built with an old palm frame cannot stay in the canonical place."""
    import argparse

    ap = argparse.ArgumentParser(description="Reach map identity check (rc 0 = match)")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--side", required=True, choices=("right", "left"))
    a = ap.parse_args()
    from dexengine.placement import arm_ik as aip  # noqa: PLC0415 — after the parse (--help stays light)

    combo = aip.resolve_combo(a.arm, a.hand, a.side)
    try:
        verify(
            a.pool_dir,
            arm=a.arm,
            hand=a.hand,
            side=a.side,
            base_xyz=combo[3],
            palm_off=aip.palm_frame_offset(a.hand),
            wrist_dir=combo[4],
            who="pool_meta",
        )
    except PoolIdentityError as e:
        print(str(e), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
