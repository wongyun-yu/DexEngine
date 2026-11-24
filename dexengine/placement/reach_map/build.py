#!/usr/bin/env python3
"""Reach map builder — for every position cell × orientation atom of one arm side, the batched DLS IK solutions,
clustered into at most K valleys (IK branches) by joint distance.

The map is robot-local (robot base frame) and covers the real-data wrist distribution widened by the placement
search box. Per (cell, atom) it stores:
  pos_err / rot_err  wrist error of the representative solution (valley 0) — "does it reach"
  margin             smallest normalised slack to the joint limits
  clearance          lowest z of the arm links above the tabletop in that pose (base frame)
  q                  representative joint values
and per valley k < K the joint values vq, errors vpe / vre, margin vmg and the lowest arm-link (vclr_arm) and
hand-link (vclr_hand) heights.

Desk test: only links above the tabletop (base-frame x beyond the desk's robot-side edge minus 4 cm) get the height
check; links in the front gap between robot and desk cannot pierce the top and are exempt (stored as +10). The test
looks at x only, so it holds for any robot y motion or dz. Arm links (elbow…forearm) must stay above the desk, the
wrist and hand links may go lower (contact is normal); the thresholds are applied by the readers.
Orientation atoms are drawn by farthest-point sampling from real-data wrist orientations plus 45° yaw copies (to cover
the placement yaw grid), not uniformly over SO(3).

Usage (one worker per z-layer share):
  CUDA_VISIBLE_DEVICES=0 python dexengine/placement/reach_map/build.py \
      --arm rby1 --hand sharpa_wave --side right \
      --worker 0 --n-workers 8 --out <pool_root>/rby1_sharpa_wave_right
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from dexengine.placement.reach_map import store as _PM


def fps_quats(Q: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Farthest-point sampling (FPS) over quaternions — distance = 1-|dot| (double cover ignored)."""
    rng = np.random.default_rng(seed)
    n = len(Q)
    sel = [int(rng.integers(n))]
    d = 1.0 - np.abs(Q @ Q[sel[0]])
    for _ in range(1, k):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, 1.0 - np.abs(Q @ Q[i]))
    return Q[sel]


def build_parser():
    """CLI of one builder worker. The canonical grid values (pos-step, atoms, seeds, valleys) are passed by
    `scripts/build_reach_map.sh`; the defaults here are not the canonical ones."""
    ap = argparse.ArgumentParser()
    # No arm/hand defaults: a default hand would build another combo's map when the argument is forgotten.
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--side", default="right")
    ap.add_argument("--pos-step", type=float, default=0.03)
    ap.add_argument("--n-atoms", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument(
        "--lam", type=float, default=1e-2, help="DLS damping (1e-4 makes the solve collapse near singular poses)"
    )
    ap.add_argument("--n-valleys", type=int, default=4)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    return ap


def _solve_seeds(Tm, c0, a, ik, torch, gen, DEV, DT, sel_arm, EDGE_RF, sel_hand):
    """IK from S seeds per cell (seed 0 = joint mid) and the lowest arm / hand link heights over the tabletop."""
    Tb = Tm[c0 : c0 + a.chunk]
    B, S, K = Tb.shape[0], a.seeds, a.n_valleys
    q0 = ik.lo + torch.rand(B, S, ik.n, generator=gen, device=DEV, dtype=DT) * ik.rng_
    q0[:, 0, :] = ik.mid.unsqueeze(0)
    qa, pea, reda, mga = ik._dls(Tb.repeat_interleave(S, 0), a.iters, a.lam, 0.0, q0.reshape(B * S, ik.n))
    fk = ik.chain.forward_kinematics(qa, end_only=False)
    _tf = list(fk.values())
    # only links above the tabletop (x > EDGE_RF) get the height check; front-gap links are exempt (+10)
    _pa = torch.stack([_tf[i].get_matrix()[:, :3, 3] for i in sel_arm], 1)
    ca_a = torch.where(_pa[..., 0] > EDGE_RF, _pa[..., 2], torch.full_like(_pa[..., 2], 10.0)).amin(1)
    _ph = torch.stack([_tf[i].get_matrix()[:, :3, 3] for i in sel_hand], 1)
    ca_h = torch.where(_ph[..., 0] > EDGE_RF, _ph[..., 2], torch.full_like(_ph[..., 2], 10.0)).amin(1)
    qa = qa.view(B, S, ik.n)
    pea, reda = pea.view(B, S), reda.view(B, S)
    mga = mga.view(B, S)
    ca_a, ca_h = ca_a.view(B, S), ca_h.view(B, S)
    return B, S, K, qa, pea, reda, mga, ca_a, ca_h


def _cluster_valleys(pea, reda, torch, mga, B, K, ik, DT, DEV, S, qa, ca_a, ca_h):
    """Batched valley clustering: scan seeds by descending margin; a new valley needs ≥ 25° max joint difference from
    every kept one (at most K)."""
    valid = (pea < 0.005) & (reda < 5.0)
    # valley clustering (batched): scan in descending margin order and accept a
    # new valley when the max joint difference from every existing valley is ≥ 25°
    # (at most K)
    order = torch.argsort(torch.where(valid, mga, torch.full_like(mga, -1e9)), dim=1, descending=True)
    qn = torch.full((B, K, ik.n), float("nan"), dtype=DT, device=DEV)
    pen = torch.full((B, K), float("inf"), dtype=DT, device=DEV)
    ren = torch.full((B, K), float("inf"), dtype=DT, device=DEV)
    mgn_ = torch.full((B, K), -1.0, dtype=DT, device=DEV)
    can = torch.full((B, K), -10.0, dtype=DT, device=DEV)
    chn = torch.full((B, K), -10.0, dtype=DT, device=DEV)
    cnt = torch.zeros(B, dtype=torch.long, device=DEV)
    THR = 25.0 * 3.14159 / 180.0
    for si in range(S):
        j = order[:, si]
        ar = torch.arange(B, device=DEV)
        vj = valid[ar, j]
        qj = qa[ar, j]
        dmin = torch.full((B,), 1e9, dtype=DT, device=DEV)
        for k2 in range(K):
            d = (qj - qn[:, k2]).abs().amax(1)
            d = torch.where(torch.isnan(d), torch.full_like(d, 1e9), d)
            dmin = torch.minimum(dmin, d)
        new = vj & (dmin > THR) & (cnt < K)
        slot = cnt.clamp(max=K - 1)
        idx = torch.nonzero(new).squeeze(-1)
        if len(idx):
            qn[idx, slot[idx]] = qj[idx]
            pen[idx, slot[idx]] = pea[ar, j][idx]
            ren[idx, slot[idx]] = reda[ar, j][idx]
            mgn_[idx, slot[idx]] = mga[ar, j][idx]
            can[idx, slot[idx]] = ca_a[ar, j][idx]
            chn[idx, slot[idx]] = ca_h[ar, j][idx]
            cnt[idx] += 1
    return qn, pen, ren, mgn_, can, chn, cnt


def _store_chunk(
    c0,
    q,
    qn,
    vq_o,
    a0,
    pen,
    vpe_o,
    ren,
    vre_o,
    mgn_,
    vmg_o,
    can,
    vca_o,
    chn,
    vch_o,
    pe,
    pe_o,
    red,
    re_o,
    mg,
    mg_o,
    cl,
    cl_o,
    q_o,
):
    """Write this chunk's valleys and representative fields into the layer arrays (float16)."""
    sl = slice(c0, c0 + len(q))
    vq_o[sl, a0] = qn.float().cpu().numpy().astype(np.float16)
    vpe_o[sl, a0] = pen.float().cpu().numpy().astype(np.float16)
    vre_o[sl, a0] = ren.float().cpu().numpy().astype(np.float16)
    vmg_o[sl, a0] = mgn_.float().cpu().numpy().astype(np.float16)
    vca_o[sl, a0] = can.float().cpu().numpy().astype(np.float16)
    vch_o[sl, a0] = chn.float().cpu().numpy().astype(np.float16)
    pe_o[sl, a0] = pe.float().cpu().numpy().astype(np.float16)
    re_o[sl, a0] = red.float().cpu().numpy().astype(np.float16)
    mg_o[sl, a0] = mg.float().cpu().numpy().astype(np.float16)
    cl_o[sl, a0] = cl.float().cpu().numpy().astype(np.float16)
    q_o[sl, a0] = q.float().cpu().numpy().astype(np.float16)


def _save_shard(shard, k, pe_o, re_o, mg_o, cl_o, q_o, vq_o, vpe_o, vre_o, vmg_o, vca_o, vch_o):
    """Write the layer shard."""
    np.savez_compressed(
        shard,
        z_index=k,
        pos_err=pe_o,
        rot_err=re_o,
        margin=mg_o,
        clearance=cl_o,
        q=q_o,
        vq=vq_o,
        vpe=vpe_o,
        vre=vre_o,
        vmg=vmg_o,
        vclr_arm=vca_o,
        vclr_hand=vch_o,
    )


def _representative(qn, pen, ren, mgn_, torch, cnt, can, B, DEV, pea, qa, reda, mga):
    """Per-cell summary = valley 0 (largest margin); cells without a valid valley take the best seed."""
    # the representative (scalar summary) is valley 0 (largest margin)
    q, pe, red, mg = qn[:, 0], pen[:, 0], ren[:, 0], mgn_[:, 0]
    cl = torch.where(cnt > 0, can.amax(1), torch.full_like(can[:, 0], -10.0))
    # cells with no valid solution fill the representative fields from the best seed
    nov = cnt == 0
    if nov.any():
        ar = torch.arange(B, device=DEV)
        jb = pea.argmin(1)
        q = torch.where(nov.unsqueeze(1), qa[ar, jb], q)
        pe = torch.where(nov, pea[ar, jb], pe)
        red = torch.where(nov, reda[ar, jb], red)
        mg = torch.where(nov, mga[ar, jb], mg)
    return q, pe, red, mg, cl


def _build_pool_layers(torch, DEV, a, R_atoms, DT, nz, out, axes, ik, sel_arm, EDGE_RF, sel_hand):
    """Solve and save this worker's z layers (`shard_z{k:03d}.npz`); an existing shard is skipped (resume)."""
    gen = torch.Generator(device=DEV)
    gen.manual_seed(a.worker + 1)
    R_t = torch.tensor(R_atoms, dtype=DT, device=DEV)  # [A,3,3]

    # z-layers are split across workers (deterministic — a rerun resumes where it stopped)
    my_layers = [k for k in range(nz) if k % a.n_workers == a.worker]
    t0 = time.time()
    for li, k in enumerate(my_layers):
        shard = out / f"shard_z{k:03d}.npz"
        if shard.exists():
            continue
        # target matrices of this layer (pos × atom)
        gx, gy = np.meshgrid(axes[0], axes[1], indexing="ij")
        pos = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, axes[2][k])], 1)  # [P,3]
        Pn, A = len(pos), a.n_atoms
        pe_o = np.zeros((Pn, A), np.float16)
        re_o = np.zeros((Pn, A), np.float16)
        mg_o = np.zeros((Pn, A), np.float16)
        cl_o = np.zeros((Pn, A), np.float16)
        q_o = np.zeros((Pn, A, ik.n), np.float16)
        K = a.n_valleys
        vq_o = np.zeros((Pn, A, K, ik.n), np.float16)
        vpe_o = np.zeros((Pn, A, K), np.float16)
        vre_o = np.zeros((Pn, A, K), np.float16)
        vmg_o = np.zeros((Pn, A, K), np.float16)
        vca_o = np.zeros((Pn, A, K), np.float16)
        vch_o = np.zeros((Pn, A, K), np.float16)
        pos_t = torch.tensor(pos, dtype=DT, device=DEV)
        for a0 in range(A):
            Tm = torch.eye(4, dtype=DT, device=DEV).repeat(Pn, 1, 1)
            Tm[:, :3, :3] = R_t[a0]
            Tm[:, :3, 3] = pos_t
            for c0 in range(0, Pn, a.chunk):
                B, S, K, qa, pea, reda, mga, ca_a, ca_h = _solve_seeds(
                    Tm, c0, a, ik, torch, gen, DEV, DT, sel_arm, EDGE_RF, sel_hand
                )
                qn, pen, ren, mgn_, can, chn, cnt = _cluster_valleys(
                    pea, reda, torch, mga, B, K, ik, DT, DEV, S, qa, ca_a, ca_h
                )
                q, pe, red, mg, cl = _representative(qn, pen, ren, mgn_, torch, cnt, can, B, DEV, pea, qa, reda, mga)
                _store_chunk(
                    c0,
                    q,
                    qn,
                    vq_o,
                    a0,
                    pen,
                    vpe_o,
                    ren,
                    vre_o,
                    mgn_,
                    vmg_o,
                    can,
                    vca_o,
                    chn,
                    vch_o,
                    pe,
                    pe_o,
                    red,
                    re_o,
                    mg,
                    mg_o,
                    cl,
                    cl_o,
                    q_o,
                )
        _save_shard(shard, k, pe_o, re_o, mg_o, cl_o, q_o, vq_o, vpe_o, vre_o, vmg_o, vca_o, vch_o)
        el = time.time() - t0
        print(
            f"[pool_v5] w{a.worker} layer {li + 1}/{len(my_layers)} (z={axes[2][k]:+.2f}) "
            f"saved · elapsed {el / 60:.1f} min",
            flush=True,
        )
    print(f"[pool_v5] w{a.worker} DONE ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


def _link_groups(a, _names):
    """Links that get the arm and the hand height checks (readers set the thresholds): the arm's links by its ArmSpec
    reach_clearance_links (elbow to forearm = arm · wrist = hand), and the hand links whose name holds
    ``<side>_hand`` or ``palm``. A class left empty dies (the spec does not describe this chain)."""
    from dexengine.core.arms import ARM_REGISTRY  # noqa: PLC0415

    spec = ARM_REGISTRY[a.arm]
    index = {
        spec.body_name(a.side, k): cls for cls, ks in spec.reach_clearance_links.items() for k in ks
    }  # arm link name → class

    def _cls(nm: str):
        """'arm' (elbow~forearm, strict penetration check) / 'hand' (wrist and below, contact allowed) / None"""
        if nm in index:
            return index[nm]
        if f"{a.side}_hand" in nm or "palm" in nm:
            return "hand"
        return None

    sel_arm = [i for i, nm in enumerate(_names) if _cls(nm) == "arm"]
    sel_hand = [i for i, nm in enumerate(_names) if _cls(nm) == "hand"]
    if not sel_arm or not sel_hand:
        raise SystemExit(
            f"[pool_v5] {a.arm}/{a.side}: arm links {len(sel_arm)} · hand links {len(sel_hand)} on the chain {_names} "
            "— ArmSpec.reach_clearance_links names none of them"
        )
    print(f"[pool_v5] arm links {len(sel_arm)} · wrist and below {len(sel_hand)}", flush=True)
    return sel_arm, sel_hand


def _sample_wrists(wrist_dir, Rotation, rz, palm_off, DESK, base_xyz):
    """About 300 wrist caches (store.wrist_files): palm positions in the robot base frame and their rotations."""
    # ── real-data wrist distribution → grid bounds + orientation atoms ──────────────
    files = _PM.wrist_files(wrist_dir)[1]  # the one selection rule (store.wrist_inputs digests the same files)
    P_list, Q_list = [], []
    for f in files:
        z = np.load(f)
        P, Q = z["positions"], z["quaternions"]
        if P.ndim == 3:
            P, Q = P[0], Q[0]
        P_list.append(P[::5])
        Q_list.append(Q[::5])
    P = np.concatenate(P_list).astype(np.float64)
    Q = np.concatenate(Q_list).astype(np.float64)
    rot = Rotation.from_quat(np.stack([Q[:, 1], Q[:, 2], Q[:, 3], Q[:, 0]], 1))
    p_base = rz.apply(P + rot.apply(palm_off)) + np.array([0, 0, DESK]) - np.asarray(base_xyz)
    return rot, p_base


def _orientation_atoms(rz, rot, Rotation, a):
    """Orientation atoms: FPS over the wrist rotations and their 45° yaw copies."""
    # orientation atoms: real-data orientations × 45° yaw copies (the placement yaw grid) → FPS
    q_base = (rz * rot).as_quat()  # xyzw
    q_base /= np.linalg.norm(q_base, axis=1, keepdims=True)
    sub = q_base[:: max(1, len(q_base) // 8000)]
    aug = [sub]
    aug.extend(
        (Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_quat(sub)).as_quat()
        for yaw in (45, 90, 135, 180, -45, -90, -135)
    )
    atoms = fps_quats(np.concatenate(aug), a.n_atoms)
    R_atoms = Rotation.from_quat(atoms).as_matrix()  # [A,3,3]
    return atoms, R_atoms


def _inputs_guard(out: Path, inputs: tuple) -> None:
    """A resumed build keeps the shards already in `out` (`_build_pool_layers`), so it must not resume over shards
    drawn from another wrist-cache set: a meta whose recorded inputs differ (or are unrecorded) next to an existing
    shard dies — the old map is moved aside first. A fresh folder, or a meta without shards, passes."""
    mp = out / "meta.npz"
    if not mp.is_file() or not any(out.glob("shard_z*.npz")):
        return
    with np.load(mp, allow_pickle=True) as m:
        have = str(m[_PM.INPUTS_KEY]) if _PM.INPUTS_KEY in m.files else None
    if have != inputs[0]:
        raise SystemExit(
            f"[pool_v5] {out} holds shards built from another wrist-cache set (meta inputs "
            f"{(have or 'unrecorded')[:12]} · now {inputs[0][:12]}) — move the old map aside, then build"
        )


def _write_meta(a, out, axes, atoms, base_xyz, DESK, palm_off, inputs):
    """Worker 0 writes meta.npz (axes, atoms, settings, identity, the wrist-input digest — store.INPUTS_KEY)."""
    if a.worker == 0:
        np.savez(
            out / "meta.npz",
            axes_x=axes[0],
            axes_y=axes[1],
            axes_z=axes[2],
            atoms_xyzw=atoms,
            pos_step=a.pos_step,
            base_xyz=np.asarray(base_xyz),
            desk_height=DESK,
            palm_off=np.asarray(palm_off),
            arm=a.arm,
            hand=a.hand,
            side=a.side,
            **{_PM.INPUTS_KEY: np.str_(inputs[0]), _PM.INPUTS_N_KEY: np.int64(inputs[1])},
        )


def main() -> int:
    """One builder worker: write meta.npz (worker 0) and this worker's z-layer shards."""
    ap = build_parser()
    a = ap.parse_args()

    import torch
    from scipy.spatial.transform import Rotation

    # torch before pinocchio — the order this entry has always imported them in
    # isort: split
    from dexengine.placement import arm_ik as AIP
    from dexengine.placement.reach_map import batch_ik as OPG
    from dexengine.robots import urdf_load

    urdf, ee, joints, base_xyz, wrist_dir = AIP.resolve_combo(a.arm, a.hand, a.side)
    # The x threshold of the "above the tabletop" test = the desk's robot-side edge (tabletop desk, owner
    # coords.desk_bounds_scene) relative to the arm base, 4 cm inside. The map is measured at the tabletop desk;
    # a dataset with another desk top needs its own clearances (clearance.check_desk_top stops the loader).
    from dexengine.placement.coords import core as _VC  # noqa: PLC0415

    EDGE_RF = _VC.desk_bounds_scene(0.0, margin=0.0)[0][0] - float(np.asarray(base_xyz)[0]) - 0.04
    print(f"[pool_v5] base={np.round(np.asarray(base_xyz), 3).tolist()} tabletop edge x>{EDGE_RF:.3f}", flush=True)
    palm_off = AIP.palm_frame_offset(a.hand)
    DESK = AIP.DESK_HEIGHT
    model = urdf_load.build_model(str(urdf))
    ik = OPG.ArmIK(a.side, model, urdf, ee, joints)
    DEV, DT = OPG.DEV, OPG.DT
    rz = Rotation.from_euler("z", -90, degrees=True)

    # clearance looks only at the distal links that can pass over the desk (elbow and below,
    # plus the hand). Torso and shoulder-column links are always low, so putting them into
    # the min makes the condition cut away everything.
    with torch.no_grad():
        _names = list(ik.chain.forward_kinematics(ik.mid.unsqueeze(0), end_only=False).keys())
    sel_arm, sel_hand = _link_groups(a, _names)

    rot, p_base = _sample_wrists(wrist_dir, Rotation, rz, palm_off, DESK, base_xyz)
    # widened around the wrist cloud: ±0.45 m in x and y, 0.18 m down and 0.46 m up
    lo = np.percentile(p_base, 0.5, axis=0) - np.array([0.45, 0.45, 0.18])
    hi = np.percentile(p_base, 99.5, axis=0) + np.array([0.45, 0.45, 0.46])  # dz (robot lowered) headroom +28cm
    axes = [np.arange(lo[i], hi[i] + 1e-9, a.pos_step) for i in range(3)]
    nx, ny, nz = (len(ax) for ax in axes)

    atoms, R_atoms = _orientation_atoms(rz, rot, Rotation, a)

    total = nx * ny * nz * a.n_atoms
    print(
        f"[pool_v5] grid {nx}x{ny}x{nz}={nx * ny * nz:,} cells × atoms {a.n_atoms}"
        f" = {total:,} entries (worker {a.worker}/{a.n_workers})",
        flush=True,
    )

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = _PM.wrist_inputs(wrist_dir)  # the set _sample_wrists read (module doc of store: 'Inputs')
    _inputs_guard(out, inputs)
    print(f"[pool_v5] wrist inputs {inputs[0][:12]} ({inputs[1]} files · {wrist_dir})", flush=True)
    _write_meta(a, out, axes, atoms, base_xyz, DESK, palm_off, inputs)

    return _build_pool_layers(torch, DEV, a, R_atoms, DT, nz, out, axes, ik, sel_arm, EDGE_RF, sel_hand)


if __name__ == "__main__":
    sys.exit(main())
