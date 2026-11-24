"""SPIDER output → our scene: constants, the qpos map of SPIDER's MuJoCo scene and the loader of one SPIDER rollout in
our frame."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

MARK = "[SPIDER]"


FREQ_REF = 50.0  # reference frame rate (trajectory_kinematic `frequency` must equal this)


SIM_DT = 0.01  # SPIDER mjwp sim dt (config.yaml sim_dt) — asserted from the time array


CMD_FILE = "spider_cmd.npz"


ARM_MODES = ("a6",)  # --spider-arm values; anything else dies


# ── (a) load + resample
# ────────────────────────────────────────────────────────────────────────────────────────────────
def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _joint_names_by_qadr(model) -> dict:
    import mujoco  # noqa: PLC0415

    out = {}
    for j in range(model.njnt):
        out[int(model.jnt_qposadr[j])] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """wxyz Hamilton product, a (T,4) ⊗ b (4,)."""
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def _quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """rotate v (3,) by unit wxyz quats q (T,4) → (T,3)."""
    w, u = q[:, :1], q[:, 1:]
    t = 2.0 * np.cross(u, np.broadcast_to(v, u.shape))
    return v[None, :] + w * t + np.cross(u, t)


def spider_map(clip_dir: Path, layout_meta: dict, ms, nq_b: int, qb0: np.ndarray) -> dict:
    """SPIDER qpos columns → our source (base) columns — the per-clip map written by the input step
    (prep_inputs.py's layout step · task_info['spider_layout']). Unchanged clips (layout_meta
    changed=false
    ·
    task_info without the key) = identity. Every source column is covered exactly once by: a SPIDER column (qpos_src ≥
    0),
    a welded static extra (frame-0 pose of our base), an attached child (its parent's SPIDER pose ∘ the XML body
    offset),
    or a frozen hinge (angle_rad). SPIDER columns with qpos_src = −1 must be exactly the empty slots (7 each)."""
    import mujoco  # noqa: PLC0415

    name = Path(clip_dir).name
    lm = (layout_meta.get("clips") or {}).get(name)
    if lm is None:
        raise SystemExit(f"{MARK} layout_meta lacks {name}")
    ti = json.loads((Path(clip_dir) / "task_info.json").read_text())
    sl = ti.get("spider_layout")
    nq_s = int(ms.nq)
    if sl is None:
        if lm.get("changed") is not False or nq_s != nq_b:
            raise SystemExit(
                f"{MARK} {name}: no spider_layout but layout_meta changed={lm.get('changed')} · nq {nq_s}/{nq_b}"
            )
        return {"src": np.arange(nq_b), "static": {}, "attached": {}, "hinge": {}, "empty": [], "changed": False}
    if lm.get("changed") is not True or int(sl["src_nq"]) != nq_b or len(sl["qpos_src"]) != nq_s:
        raise SystemExit(
            f"{MARK} {name}: spider_layout ↔ layout_meta/base mismatch (changed {lm.get('changed')} · src_nq "
            f"{sl['src_nq']}/{nq_b} · qpos_src {len(sl['qpos_src'])}/{nq_s})"
        )
    src = np.asarray(sl["qpos_src"], np.int64)
    if (src < 0).sum() != 7 * len(sl.get("empty_slots") or []):
        raise SystemExit(f"{MARK} {name}: empty columns {(src < 0).sum()} ≠ 7 × empty slots {sl.get('empty_slots')}")
    cover = [int(c) for c in src if c >= 0]
    lay_src = {b["name"]: b for b in sl["src_qpos_layout"]}
    static = {k: int(v["src_qpos"]) for k, v in (sl.get("static") or {}).items()}
    hinge = {k: (int(v["src_qpos"]), float(v["angle_rad"])) for k, v in (sl.get("hinge_frozen") or {}).items()}
    attached = {}
    for k, v in (sl.get("attached") or {}).items():
        bid = mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_BODY, k)
        par = mujoco.mj_id2name(ms, mujoco.mjtObj.mjOBJ_BODY, int(ms.body_parentid[bid])) if bid >= 0 else None
        if bid < 0 or par != v["to"] or v["to"] not in lay_src:
            raise SystemExit(f"{MARK} {name}: attached body {k} — SPIDER body {bid} parent {par} ≠ {v['to']}")
        attached[k] = (
            int(v["src_qpos"]),
            int(lay_src[v["to"]]["start"]),
            np.array(ms.body_pos[bid], np.float64),
            np.array(ms.body_quat[bid], np.float64),
        )
    cover += [c + j for c in static.values() for j in range(7)] + [
        a[0] + j for a in attached.values() for j in range(7)
    ]
    cover += [h[0] for h in hinge.values()]
    if sorted(cover) != list(range(nq_b)):
        dup = sorted({c for c in cover if cover.count(c) > 1})[:4]
        miss = sorted(set(range(nq_b)) - set(cover))[:6]
        raise SystemExit(f"{MARK} {name}: source column cover is not a bijection (duplicates {dup} · missing {miss})")
    for k, (c, ang) in hinge.items():  # oracle: frozen angle = our frame-0 angle
        if abs(qb0[c] - ang) > 1e-6:
            raise SystemExit(f"{MARK} {name}: fixed hinge {k} {ang} ≠ our frame 0 {qb0[c]}")
    return {
        "src": src,
        "static": static,
        "attached": attached,
        "hinge": hinge,
        "empty": list(sl.get("empty_slots") or []),
        "changed": True,
    }


def map_to_src(Qs: np.ndarray, mp: dict, qb0: np.ndarray, nq_b: int) -> np.ndarray:
    T = len(Qs)
    out = np.full((T, nq_b), np.nan)
    j = np.where(mp["src"] >= 0)[0]
    out[:, mp["src"][j]] = Qs[:, j]
    for c in mp["static"].values():
        out[:, c : c + 7] = qb0[None, c : c + 7]
    for c, ang in mp["hinge"].values():
        out[:, c] = ang
    for c, cp, bp, bq in mp["attached"].values():
        pp, pq = out[:, cp : cp + 3], out[:, cp + 3 : cp + 7]
        pq = pq / np.linalg.norm(pq, axis=1, keepdims=True)
        out[:, c : c + 3] = pp + _quat_rot(pq, bp)
        out[:, c + 3 : c + 7] = _quat_mul(pq, bq)
    if np.isnan(out).any():
        raise SystemExit(f"{MARK} source column fill failed (NaN left)")
    return out


def _read_mjwp(clip_dir, base_clip_dir, f, fin, fb):
    """Read the SPIDER rollout (qpos, ctrl, time) and check lengths and the SIM_DT time grid."""
    for p in (f, fin, fb, clip_dir / "scene.xml", base_clip_dir / "scene.xml", clip_dir / "task_info.json"):
        if not p.is_file():
            raise SystemExit(f"{MARK} missing {p}")
    z = np.load(f)
    for k in ("qpos", "ctrl", "time"):
        if k not in z.files:
            raise SystemExit(f"{MARK} {f}: key {k} missing")
    nq = int(z["qpos"].shape[-1])
    q = np.asarray(z["qpos"], np.float64).reshape(-1, nq)
    u = np.asarray(z["ctrl"], np.float64).reshape(-1, z["ctrl"].shape[-1])
    t = np.asarray(z["time"], np.float64).reshape(-1)
    if not (len(q) == len(u) == len(t)):
        raise SystemExit(f"{MARK} {f}: qpos {len(q)} · ctrl {len(u)} · time {len(t)} lengths differ")
    dt = np.diff(t)
    if len(dt) and np.abs(dt - SIM_DT).max() > 1e-9:
        raise SystemExit(f"{MARK} {f}: time steps {np.unique(np.round(dt, 9))[:4]} ≠ SIM_DT {SIM_DT}")
    if abs(t[0] - SIM_DT) > 1e-9:
        raise SystemExit(
            f"{MARK} {f}: time[0]={t[0]} — not the assumed form where state i is after i+1 steps (time[0]=SIM_DT)"
        )
    return nq, q, u, t


def _check_reference(mujoco, clip_dir, base_clip_dir, fin, fb, nq, layout_meta):
    """Check that SPIDER ran from our base reference (frequency, nq, mapped columns, joint names); returns the map."""
    zi, zb = np.load(fin), np.load(fb)
    if float(zi["frequency"]) != FREQ_REF or float(zb["frequency"]) != FREQ_REF:
        raise SystemExit(f"{MARK} reference frequency {float(zi['frequency'])} · {float(zb['frequency'])} ≠ {FREQ_REF}")
    qi, qb = np.asarray(zi["qpos"], np.float64), np.asarray(zb["qpos"], np.float64)
    ms = mujoco.MjModel.from_xml_path(os.path.realpath(clip_dir / "scene.xml"))  # realpath: relative mesh dirs
    mb = mujoco.MjModel.from_xml_path(os.path.realpath(base_clip_dir / "scene.xml"))
    nq_b = int(qb.shape[1])
    if ms.nq != nq or qi.shape[1] != nq or mb.nq != nq_b:
        raise SystemExit(
            f"{MARK} nq: SPIDER output {nq} · SPIDER model {ms.nq} · SPIDER input {qi.shape[1]} · base {nq_b}/{mb.nq}"
        )
    mp = spider_map(clip_dir, layout_meta, ms, nq_b, qb[0])
    jm = np.where(mp["src"] >= 0)[0]
    if len(qi) != len(qb) or not np.array_equal(qi[:, jm], qb[:, mp["src"][jm]]):
        d = np.abs(qi[:, jm] - qb[:, mp["src"][jm]]).max() if len(qi) == len(qb) else f"length {len(qi)} ≠ {len(qb)}"
        raise SystemExit(
            f"{MARK} SPIDER input {fin} ≠ our base {fb} (mapped column difference {d}) — SPIDER ran from a "
            "different reference"
        )
    ns, nb = _joint_names_by_qadr(ms), _joint_names_by_qadr(mb)
    bad = [
        (a, ns[a], nb.get(int(mp["src"][a]))) for a in ns if mp["src"][a] >= 0 and ns[a] != nb.get(int(mp["src"][a]))
    ]
    if bad:
        raise SystemExit(f"{MARK} joint names SPIDER ≠ base (mapped columns): {bad[:4]}")
    return qb, qi, ms, nq_b, mp


def _resample(f, t, q, qi, qb, nq):
    """Pick the sim state at every 50 Hz frame time; frame 0 = SPIDER input frame 0."""
    T = len(qb)
    # frame k at time k/50 = flat index k/(FREQ_REF·SIM_DT) − 1 (= 2k − 1) — checked on the time array
    idx = np.zeros(T, np.int64)
    for k in range(1, T):
        i = int(round(k / (FREQ_REF * SIM_DT))) - 1
        if i >= len(t) or abs(t[i] - k / FREQ_REF) > 1e-9:
            raise SystemExit(
                f"{MARK} {f}: no sim state for frame {k}(t={k / FREQ_REF:.2f}) (len {len(t)}) — SPIDER output is short"
            )
        idx[k] = i
    Qs = np.empty((T, nq))
    Qs[0] = qi[0]
    Qs[1:] = q[idx[1:]]
    return T, idx, Qs


def _actuator_cols(mujoco, ms, u, mp):
    """SPIDER qpos column of each actuator; every actuator must be a joint position servo with a source column."""
    if u.shape[1] != ms.nu:
        raise SystemExit(f"{MARK} ctrl width {u.shape[1]} ≠ model nu {ms.nu}")
    cols = []
    for a in range(ms.nu):
        if int(ms.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise SystemExit(f"{MARK} actuator {a} does not drive a joint")
        g, b = ms.actuator_gainprm[a], ms.actuator_biasprm[a]
        if not (abs(b[0]) < 1e-12 and abs(g[0] + b[1]) < 1e-9 * max(1.0, abs(g[0]))):
            raise SystemExit(
                f"{MARK} actuator {a} is not a position servo (gain {g[0]} bias {b[:3]}) — ctrl ≠ joint target"
            )
        cols.append(int(ms.jnt_qposadr[int(ms.actuator_trnid[a, 0])]))
    if len(set(cols)) != len(cols) or any(mp["src"][c] < 0 for c in cols):
        raise SystemExit(f"{MARK} actuator columns overlap or lack a source column")
    return cols


def _check_attached(clip_dir, mp, qs, qb):
    """Oracle: every attached child at frame 0 equals our base frame-0 pose."""
    for k, (c, _cp, _bp, _bq) in mp["attached"].items():  # oracle: attached child at frame 0 = our frame-0 pose
        dp = np.abs(qs[0, c : c + 3] - qb[0, c : c + 3]).max()
        dq = 1 - abs(float(np.dot(qs[0, c + 3 : c + 7], qb[0, c + 3 : c + 7] / np.linalg.norm(qb[0, c + 3 : c + 7]))))
        if dp > 1e-4 or dq > 1e-6:
            raise SystemExit(
                f"{MARK} {clip_dir.name}: attached body {k} frame 0 ≠ our base (position {dp:.2e} m · 1−|q·q| {dq:.1e})"
            )


def load_spider(clip_dir: Path, base_clip_dir: Path, layout_meta: dict) -> dict:
    """SPIDER output of one clip → 50 Hz state (T, nq) + commanded (T, nq) in OUR source (base) layout, read through
    the per-clip column map (spider_map). Dies on any mismatch."""
    import mujoco  # noqa: PLC0415

    clip_dir, base_clip_dir = Path(clip_dir), Path(base_clip_dir)
    f = clip_dir / "0/trajectory_mjwp.npz"
    fin = clip_dir / "0/trajectory_kinematic.npz"  # SPIDER's own input (its frame 0 = SPIDER's initial state)
    fb = base_clip_dir / "0/trajectory_kinematic.npz"  # our base reference
    nq, q, u, t = _read_mjwp(clip_dir, base_clip_dir, f, fin, fb)
    qb, qi, ms, nq_b, mp = _check_reference(mujoco, clip_dir, base_clip_dir, fin, fb, nq, layout_meta)
    T, idx, Qs = _resample(f, t, q, qi, qb, nq)
    cols = _actuator_cols(mujoco, ms, u, mp)
    Qc = Qs.copy()
    Qc[1:, cols] = u[idx[1:]]  # actuator a → SPIDER qpos column cols[a]
    qs, cmd = map_to_src(Qs, mp, qb[0], nq_b), map_to_src(Qc, mp, qb[0], nq_b)
    _check_attached(clip_dir, mp, qs, qb)
    src_cols = sorted(int(mp["src"][c]) for c in cols)
    meta = {
        "mjwp": str(f),
        "mjwp_sha256": _sha(f),
        "input": str(fin),
        "input_sha256": _sha(fin),
        "base": str(fb),
        "base_sha256": _sha(fb),
        "T": T,
        "sim_steps": len(t),
        "layout_changed": mp["changed"],
        "static": sorted(mp["static"]),
        "attached": sorted(mp["attached"]),
        "hinge_frozen": {k: v[1] for k, v in mp["hinge"].items()},
        "empty_slots": mp["empty"],
        "rule": "frame0 = SPIDER init = input ref[0]; frame k>=1 = flat sim index with time == k/50 (= 2k-1); cmd row "
        "k = ctrl at that same index (actuator -> SPIDER joint column via SPIDER scene.xml); SPIDER columns -> "
        "our source columns via task_info.spider_layout.qpos_src (static = our frame 0 · attached = parent "
        "SPIDER pose o XML offset · hinge_frozen = angle_rad)",
        "actuated_src_cols": src_cols,
    }
    print(
        f"{MARK} [SPIDER_SRC] {clip_dir.name} T {T} · sim {len(t)} steps · nq SPIDER {nq} → source {nq_b} · map "
        f"{'changed' if mp['changed'] else 'identity'} (static {sorted(mp['static'])} · attached "
        f"{sorted(mp['attached'])} · fixed hinges "
        f"{sorted(mp['hinge'])} · empty slots {mp['empty']}) · actuators {len(cols)} columns · input = our base (sha "
        f"{meta['base_sha256'][:12]})",
        flush=True,
    )
    return {"q": qs, "cmd": cmd, "cols": src_cols, "meta": meta}
