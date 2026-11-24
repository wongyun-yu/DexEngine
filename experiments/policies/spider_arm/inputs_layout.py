"""SPIDER input step `layout` (SPIDER's qpos layout: robot, free right object, free left object), run by SPIDER's
python.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from experiments.policies.spider_arm.inputs_common import die, out_of, read_clips, sha
from experiments.policies.spider_arm.inputs_layout_xml import ATTACH_MAX_M, body_name, rewrite_xml_layout

# ------------------------------------------------------------------------------------------------------------ layout
LAYOUT_STATIC = ("receiver_welded", "touched_welded", "support_welded")  # start's per-clip extra records


POSE_TOL_M, POSE_TOL_RAD = 1e-6, 1e-5  # frame-0 world pose of every body, before vs after the layout step


def _dump_layout_meta(lf, meta, mjv):
    with open(lf + ".tmp", "w") as fh:
        json.dump(
            {
                "rule": "SPIDER's scene layout (cmd_layout)",
                "pose_tol": [POSE_TOL_M, POSE_TOL_RAD],
                "attach_max_m": ATTACH_MAX_M,
                "mujoco": mjv,
                "clips": meta,
            },
            fh,
            indent=1,
        )
    os.replace(lf + ".tmp", lf)


def _load_clip_models(clip, d, sm):
    """Paths, start extras, the source model and its reference; dies unless actuator i drives qpos i."""
    import mujoco
    import numpy as np

    p, npz, tif = f"{d}/scene.xml", f"{d}/0/trajectory_kinematic.npz", f"{d}/task_info.json"
    extras = sm[clip]["pairs"]["scene.xml"]["extras"]
    m0 = mujoco.MjModel.from_xml_path(p)
    z = {k: np.asarray(v) for k, v in np.load(npz).items()}
    q, v = z["qpos"], z["qvel"]
    if q.shape[1] != m0.nq or v.shape[1] != m0.nv:
        die(f"{clip}: reference qpos/qvel widths {q.shape[1]}/{v.shape[1]} != scene nq/nv {m0.nq}/{m0.nv}")
    nu = m0.nu
    for i in range(nu):
        j = int(m0.actuator_trnid[i, 0])
        if (
            int(m0.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT)
            or int(m0.jnt_qposadr[j]) != i
            or int(m0.jnt_dofadr[j]) != i
        ):
            die(f"{clip}: actuator {i} does not drive qpos {i} — SPIDER reads the robot as qpos[:nu]")
    return p, npz, tif, extras, m0, z, q, v, nu


def _classify_joints(clip, m0, nu, extras, slots):
    """Every joint after the robot: a main object in a slot, a welded extra, a moving extra or a frozen hinge."""
    import mujoco

    J = mujoco.mjtJoint
    slot_src, static, attach, hinge = {}, [], {}, {}
    for j in range(m0.njnt):
        a = int(m0.jnt_qposadr[j])
        if a < nu:
            continue
        b, t = int(m0.jnt_bodyid[j]), int(m0.jnt_type[j])
        n = body_name(m0, b)
        par = body_name(m0, int(m0.body_parentid[b]))
        if t == J.mjJNT_FREE and n in slots:
            slot_src[n] = (a, int(m0.jnt_dofadr[j]))
        elif t == J.mjJNT_FREE and extras.get(n) in LAYOUT_STATIC:
            static.append((n, a))
        elif t == J.mjJNT_FREE and extras.get(n) == "moving_free":
            attach[n] = a
        elif t == J.mjJNT_HINGE and n.endswith("_top") and par == n[:-4] and par in slots:
            hinge[n] = (mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j), a, float(m0.qpos0[a]))
        else:
            die(f"{clip}: joint on body {n} (type {t}, extra kind {extras.get(n)}) has no SPIDER-layout rule")
    if not slot_src:
        die(f"{clip}: no main object in a SPIDER slot ({slots})")
    return slot_src, static, attach, hinge


def _find_ghosts(m0, slots, tif):
    """A jointless <side>_object body that task_info does not list = the scene generator's leftover (Isaac spawns
    task_info's objects only); it is removed with its pairs (one E5 clip: a static copy of its extra near the
    origin)."""
    import mujoco

    ti_names = {o["name"] for o in json.loads(Path(tif).read_text())["objects"]}
    return [
        s
        for s in slots
        if mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, s) >= 0
        and s not in ti_names
        and int(m0.body_jntnum[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, s)]) == 0
    ]


def _remap(x, src, fill):
    import numpy as np

    y = np.empty((x.shape[0], len(src)), dtype=x.dtype)
    for k, c in enumerate(src):
        y[:, k] = x[:, c] if c >= 0 else fill[k]
    return y


def _remap_reference_columns(q, v, nu, slots, slot_src):
    """New reference columns: robot, then 7 qpos / 6 qvel per slot (an empty slot = origin, identity, constant)."""
    import numpy as np

    qsrc, vsrc = list(range(nu)), list(range(nu))
    for s in slots:
        if s in slot_src:
            a, dv = slot_src[s]
            qsrc += list(range(a, a + 7))
            vsrc += list(range(dv, dv + 6))
        else:
            qsrc += [-1] * 7
            vsrc += [-1] * 6
    fq = {}
    for s_i, s in enumerate(slots):
        if s not in slot_src:
            base = nu + 7 * s_i
            for k, val in enumerate((0, 0, 0, 1, 0, 0, 0)):
                fq[base + k] = val
    Q = _remap(q, qsrc, {k: fq.get(k, np.nan) for k in range(len(qsrc))})
    V = _remap(v, vsrc, dict.fromkeys(range(len(vsrc)), 0.0))
    return qsrc, vsrc, fq, Q, V


def _actuator_names(m):
    import mujoco

    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]


def _check_frame_poses(clip, m0, m1, q, Q, ghosts, changed):
    """Frame-0 world pose of every body (dies on a move) and masses; returns the last-frame deviations."""
    import mujoco
    import numpy as np

    T = q.shape[0]
    dev = {}
    for fr in (0, T - 1):
        d0, d1 = mujoco.MjData(m0), mujoco.MjData(m1)
        d0.qpos[:], d1.qpos[:] = q[fr], Q[fr]
        mujoco.mj_kinematics(m0, d0)
        mujoco.mj_kinematics(m1, d1)
        for b in range(1, m0.nbody):
            if body_name(m0, b) in ghosts:
                continue
            b1 = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, body_name(m0, b))
            if b1 < 0:
                die(f"{clip}: body {body_name(m0, b)} lost")
            dp = float(np.linalg.norm(d0.xpos[b] - d1.xpos[b1]))
            da = float(2 * np.arccos(min(1.0, abs(float(np.dot(d0.xquat[b], d1.xquat[b1]))))))
            if fr == 0 and (dp > POSE_TOL_M or da > POSE_TOL_RAD):
                die(f"{clip}: body {body_name(m0, b)} moved at frame 0 by {dp:.2e} m / {da:.2e} rad")
            if fr == T - 1 and (dp > 1e-6 or da > 1e-5):
                dev[body_name(m0, b)] = [round(dp, 4), round(float(np.degrees(da)), 2)]
        if fr == 0 and changed:
            for b in range(1, m0.nbody):
                if body_name(m0, b) in ghosts:
                    continue
                b1 = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, body_name(m0, b))
                if float(m0.body_mass[b]) != float(m1.body_mass[b1]):
                    die(f"{clip}: body {body_name(m0, b)} mass changed")
    return dev


def _check_layout(clip, m0, m1, q, Q, nu, slots, ghosts, changed, info, static):
    """Checks: SPIDER's layout, actuators, frame-0 world pose of every body, masses, pairs, welds; returns the
    last-frame deviations."""
    import mujoco
    import numpy as np

    J = mujoco.mjtJoint
    nexp_q, nexp_v = nu + 7 * len(slots), nu + 6 * len(slots)
    if (m1.nq, m1.nv, m1.nu) != (nexp_q, nexp_v, nu):
        die(f"{clip}: layout nq/nv/nu {m1.nq}/{m1.nv}/{m1.nu} != {nexp_q}/{nexp_v}/{nu}")
    tail = [
        (body_name(m1, int(m1.jnt_bodyid[j])), int(m1.jnt_type[j]))
        for j in range(m1.njnt)
        if int(m1.jnt_qposadr[j]) >= nu
    ]
    if tail != [(s, int(J.mjJNT_FREE)) for s in slots]:
        die(f"{clip}: joints after the robot {tail} != free {slots}")
    if _actuator_names(m1) != _actuator_names(m0):
        die(f"{clip}: actuators changed")
    if np.isnan(Q).any():
        die(f"{clip}: unfilled reference column")
    dev = _check_frame_poses(clip, m0, m1, q, Q, ghosts, changed)
    if changed and m1.npair != (
        m0.npair - info["pairs_static_removed"] - sum(x["pairs"] for x in info["ghosts_removed"].values())
    ):
        die(f"{clip}: pairs {m0.npair} -> {m1.npair}")
    if changed and m1.neq != m0.neq - len(static):
        die(f"{clip}: equality {m0.neq} -> {m1.neq}")
    return dev


def _check_frozen_links(clip, dev, info):
    """Frozen links at the last frame vs the reference (record): where the layout step's rigid choices differ."""
    info["last_frame_dev_m_deg"] = {
        k: x for k, x in dev.items() if k in info["hinge_frozen"] or k in info["attached"] or k in info["static"]
    }
    unexplained = {
        k: x
        for k, x in dev.items()
        if k not in info["last_frame_dev_m_deg"] and not k.startswith(("ref_object_", "ref_hand_"))
    }
    if unexplained:
        die(f"{clip}: bodies off the reference at the last frame that the layout did not freeze: {unexplained}")


def _layout_task_info(ti, nu, slots, slot_src, info, m1, qsrc, vsrc):
    """task_info.json in the new layout: qpos_layout, nq, objects, and 'spider_layout' = the column map back."""
    src_layout, src_nq = ti["qpos_layout"], ti["nq"]
    lay = [e for e in src_layout if e["kind"] == "hand"]
    for s_i, s in enumerate(slots):
        st0 = nu + 7 * s_i
        if s in slot_src:
            e = dict(next(x for x in src_layout if x["name"] == s), start=st0)
        else:
            e = {"name": s, "start": st0, "width": 7, "kind": "spider_empty", "side": s.split("_")[0]}
        lay.append(e)
    for o in ti["objects"]:
        n = o["name"]
        if n in slot_src:
            o["qpos_start"] = nu + 7 * slots.index(n)
        elif n in info["static"]:
            o.update(qpos_start=None, qpos_width=0, spider_layout="static at frame-0 pose")
        elif n in info["attached"]:
            o.update(qpos_start=None, qpos_width=0, spider_layout=f"fixed child of {info['attached'][n]['to']}")
        if o.get("articulation") and f"{n}_top" in info["hinge_frozen"]:
            o["articulation"] = dict(
                o["articulation"],
                qpos_index=None,
                spider_frozen_rad=info["hinge_frozen"][f"{n}_top"]["angle_rad"],
            )
    ti.update(
        qpos_layout=lay,
        nq=int(m1.nq),
        spider_layout={
            "tool": "tools/e5spider/prep_e5_inputs.py layout",
            "src_nq": src_nq,
            "src_qpos_layout": src_layout,
            "qpos_src": qsrc,
            "qvel_src": vsrc,
            "empty_slot_qpos": [0, 0, 0, 1, 0, 0, 0],
            **info,
        },
    )


def _write_layout_files(clip, d, p, npz, tif, z, q, Q, V, qsrc, vsrc, fq, m0, m1, nu, slots, slot_src, info):
    """Write the new reference and task_info, then replace scene.xml, the reference symlink and task_info.json."""
    import numpy as np

    T = q.shape[0]
    tgt = os.path.realpath(npz)
    h0 = sha(tgt)
    for k, x in z.items():
        if (
            k not in ("qpos", "qvel", "qpos_rollout")
            and x.ndim == 2
            and x.shape[1] in (m0.nq, m0.nv)
            and x.shape[0] == T
            and x.shape[1] > 1
        ):
            die(f"{clip}: npz key {k} has qpos/qvel width — no remap rule")
    zn = dict(z, qpos=Q, qvel=V)
    if "qpos_rollout" in z:
        if z["qpos_rollout"].shape != q.shape:
            die(f"{clip}: qpos_rollout shape {z['qpos_rollout'].shape}")
        zn["qpos_rollout"] = _remap(z["qpos_rollout"], qsrc, {k: fq.get(k, np.nan) for k in range(len(qsrc))})
    np.savez(f"{d}/0/_layout_ref.npz", **zn)
    ti = json.loads(Path(tif).read_text())
    _layout_task_info(ti, nu, slots, slot_src, info, m1, qsrc, vsrc)
    with open(f"{d}/_layout_task_info.json", "w") as fh:
        json.dump(ti, fh, indent=1)
    os.replace(f"{d}/_layout_scene.xml", p)
    os.replace(f"{d}/0/_layout_ref.npz", npz)  # replaces the symlink, not its target
    os.replace(f"{d}/_layout_task_info.json", tif)  # same
    if sha(tgt) != h0:
        die(f"{clip}: the source reference changed")


def _layout_one(r, d, sm):
    """One clip into SPIDER's layout (see cmd_layout). Returns (its layout_meta entry, the source nq)."""
    clip = r["clip"]
    p, npz, tif, extras, m0, z, q, v, nu = _load_clip_models(clip, d, sm)
    sides = ["right", "left"] if r["embodiment"] == "bimanual" else [r["embodiment"]]
    slots = [f"{s}_object" for s in sides]
    slot_src, static, attach, hinge = _classify_joints(clip, m0, nu, extras, slots)
    ghosts = _find_ghosts(m0, slots, tif)
    present = [s for s in slots if s in slot_src]
    if [slot_src[s][0] for s in present] != sorted(slot_src[s][0] for s in present):
        die(f"{clip}: slot objects out of SPIDER's order (right before left)")
    empty = [s for s in slots if s not in slot_src]
    info = {
        "static": {},
        "attached": {},
        "hinge_frozen": {},
        "empty_slots": empty,
        "pairs_static_removed": 0,
        "ghosts_removed": {},
    }
    changed = bool(static or attach or hinge or empty or ghosts)
    if changed:
        m1 = rewrite_xml_layout(
            clip, d, p, m0, q, extras, slot_src, present, static, attach, hinge, ghosts, empty, info
        )
    else:
        m1 = m0
    qsrc, vsrc, fq, Q, V = _remap_reference_columns(q, v, nu, slots, slot_src)
    dev = _check_layout(clip, m0, m1, q, Q, nu, slots, ghosts, changed, info, static)
    _check_frozen_links(clip, dev, info)
    if changed:
        _write_layout_files(clip, d, p, npz, tif, z, q, Q, V, qsrc, vsrc, fq, m0, m1, nu, slots, slot_src, info)
    if os.path.exists(f"{d}/scene_eq.xml"):
        os.remove(f"{d}/scene_eq.xml")
    return dict(info, changed=changed, nq=int(m1.nq), nv=int(m1.nv), nu=nu), m0.nq


def cmd_layout(clips_tsv, out):
    """SPIDER's scene layout, read where SPIDER and the scorer read it (io.load_data: ctrl = qpos[:, :-nq_obj];
    mjwp._diff_qpos: the last 14 (bimanual) / 7 (one hand) qpos are the free right and left objects; eval/score.py:
    same):
    qpos = robot (nu, actuator i on qpos i) + one free right object (+ one free left object when bimanual), nothing
    else.
    Upstream builds every scene that way (generate_xml.py), with an empty body when a side has no object. Our E5 scenes
    differ in three ways; each is turned into SPIDER's layout without moving any body at frame 0 (checked in MuJoCo):
      - a welded extra (start: receiver / touched / support) -> a static body at its frame-0 pose (its weld removed);
      - a free 'moving' extra -> a fixed child of the main object it stays within ATTACH_MAX_M of on every frame
        (Isaac moves it along the reference; the one E5 case travels with the left object, 0.5 cm apart);
      - an articulated object's hinge (<main>_top) -> fixed at its frame-0 angle (upstream's ARCTIC path drops the
        articulation too; we keep both links);
      - a missing side of a bimanual clip -> SPIDER's empty-object body (inputs_layout_xml._empty_slot_body).
      - a jointless <side>_object body that task_info does not list (the scene generator's leftover; Isaac spawns
        task_info's objects only) -> removed with its pairs, before the empty-object body takes its name.
    The reference (0/trajectory_kinematic.npz, a symlink into the data root) is replaced by a real file with the
    columns in that layout (the empty slot = origin, identity, constant; welded extras' columns dropped); task_info.json
    becomes a real copy with qpos_layout / nq / objects in the new layout and 'spider_layout' = the column map back.
    scene_eq.xml is removed from every clip: SPIDER reads it only at num_dyn > 1, run_e5.sh requires num_dyn 1, and it
    would keep the old layout. Clips already in SPIDER's layout keep their scene.xml and symlinks unchanged."""
    import mujoco

    rows = read_clips(clips_tsv)
    sm = json.loads(Path(f"{out}/start_meta.json").read_text())
    lf = f"{out}/layout_meta.json"
    meta = json.loads(Path(lf).read_text())["clips"] if os.path.exists(lf) else {}
    done = 0
    for r in rows:
        clip = r["clip"]
        d = out_of(out, r)
        if clip in meta:  # files are replaced only after every check, then meta is written
            print(f"[E5_SPIDER_IN] layout {clip} already applied (layout_meta.json) — skipped", flush=True)
            done += 1
            continue
        info, nq0 = _layout_one(r, d, sm)
        changed, empty = info["changed"], info["empty_slots"]
        meta[clip] = info
        _dump_layout_meta(lf, meta, mujoco.__version__)  # per clip: a rerun after a death skips nothing silently
        done += 1
        print(
            f"[E5_SPIDER_IN] layout {clip} {'CHANGED' if changed else 'as-is'} nq {nq0}->{info['nq']} "
            f"static={list(info['static'])} attached={list(info['attached'])} hinge={list(info['hinge_frozen'])} "
            f"empty={empty} ghosts={info['ghosts_removed']} last_dev={info['last_frame_dev_m_deg']}",
            flush=True,
        )
    print(f"[E5_SPIDER_IN] layout DONE {done}/{len(rows)} clips · changed={sum(x['changed'] for x in meta.values())}")
    sys.exit(0 if done == len(rows) else 1)
