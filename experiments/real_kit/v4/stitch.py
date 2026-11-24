"""v4: join approach rows and rollout rows, per-variant check values (table) and the stitched trajectory file (render
and post-gate input).

Input <rd> = a `run roll` folder: run/rollouts.npz (session, layout 2), approach/approach_b<batch>.npz (phys.py),
post.json (the v2 post gate — RSI rollouts) and <clip>/plan/plan.json (schedule check values).
Join: warm-up mode = the variant's first env approach rows[start:N] of the warm-up batch + the real-batch RSI rollout
(row 0 = RSI; join residual = approach end ↔ RSI); every mode = that rollout's (batch, env) approach rows[start:N] +
that rollout (row 0 = the approach end as is — no teleport).
Output <rd>/stitched/run/rollouts.npz (layout 2 unchanged — approach rows in front of every kept rollout, the failure
store fb as is), <rd>/v4_metrics.json (variant rows). Marker [AUG_V4_STITCH], last line DONE.
  python -m experiments.real_kit.v4.stitch --rd <roll folder> --plan <plan.json> --shadow S --dataset ds --emb emb
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import types
from pathlib import Path

import numpy as np

from dexengine.core import paths as P
from experiments.real_kit.v4 import rules as V4

MARK = "[AUG_V4_STITCH]"
JOIN_WIN = 15  # frame window around the join (slow-frame count)
SLOW_MPS = 0.01  # wrist < 1 cm/s
TOUCH_N = 1.0  # contact force threshold (N) — hand ↔ desk and objects during the approach


def joint_cols(ents: list) -> dict[str, int]:
    out = {}
    for e in ents:
        if e["kind"] == "artic":
            for i, n in enumerate(e["joint_names"]):
                out[n] = e["lo"] + 7 + i
    return out


def obj_join(end: np.ndarray, row: np.ndarray, objs: list) -> tuple[float, float]:
    """Object residual between the approach end state and one rollout row — max over objects (mm, °)."""
    mm, deg = [], []
    for ob in objs:
        lo = ob["lo"]
        mm.append(1000 * float(np.linalg.norm(end[lo : lo + 3] - row[lo : lo + 3])))
        dq = abs(float(np.dot(end[lo + 3 : lo + 7], row[lo + 3 : lo + 7])))
        deg.append(float(np.degrees(2 * np.arccos(min(1.0, dq)))))
    return round(max(mm), 2), round(max(deg), 2)


def app_touch(chans: list, pre: np.ndarray, K: int, k_ent: dict) -> dict:
    """Contacts during the approach (hand sensors ↔ every partner class — force_partner columns, N): max force before
    the straight entry (pre) and during it (ent — the plan's k_ent; for the idle hand everything is pre), first contact
    frame (negative, relative to the join). Verdict: pre = 0 N everywhere; entry = only its own object, up to the v2 RSI
    reference; desk, supports and other objects 0 N (the summary compares with the reference)."""
    touch = {}
    for ch in chans:
        if ch["kind"] != "force_partner":
            continue
        nm = ch["name"].split("/", 1)[1]
        ke = k_ent.get(nm.split("_hand_contact", 1)[0], K)
        f = np.linalg.norm(pre[:, ch["lo"] : ch["hi"]].reshape(len(pre), -1, 3), axis=2).max(1)
        hit = np.nonzero(f > TOUCH_N)[0]
        touch[nm] = {
            "max_N": round(float(f.max()), 2),
            "pre_N": round(float(f[:ke].max()) if ke else 0.0, 2),
            "ent_N": round(float(f[ke:].max()) if ke < len(f) else 0.0, 2),
            "k_ent": ke,
            "first_rel": (int(hit[0]) - K if hit.size else None),
        }
    return touch


def clip_first_touch(chans: list, st) -> int | None:
    """The clip's (policy's) first hand ↔ object contact frame — flags clips that touch right after the join · no
    rollout = None.
    """
    ft = []
    for ch in chans if st is not None else []:
        if ch["kind"] == "force_partner" and ".object" in ch["name"]:
            hit = np.nonzero(np.abs(st[:, ch["lo"] : ch["hi"]]).max(1) > TOUCH_N)[0]
            if hit.size:
                ft.append(int(hit[0]))
    return min(ft) if ft else None


def app_guard(cx, name: str, act: list, pre: np.ndarray, arm: dict) -> dict:
    """Judge guard (canonical function) on the recorded approach poses (both arms, fingers) · hands (palm, finger
    shells) ↔
    desk box (the collision desk term measures arm links only — scene.hand_desk_m, both hands)."""
    from dexengine.placement.collision import core as SCL
    from dexengine.placement.collision import hand_surface as HS
    from dexengine.placement.collision import support_surface as SSF
    from dexengine.placement.coords import core as VC
    from experiments.real_kit.v4 import scene as SCN

    a, jc, ctx = cx.a, cx.jc, cx.ctx
    combo = "_".join(V4.COMBO)
    tn = name if name.endswith("_pl_rt") else name + "_pl_rt"
    tree = a.shadow / P.data_layout.render("trees", a.dataset, combo=combo, emb=a.emb, tree=tn)
    fq = {s: pre[:, [jc[n] for n in HS.finger_names(V4.COMBO[1], s)]].astype(np.float64) for s in ("right", "left")}
    z_off = float(
        np.load(
            a.shadow
            / P.data_layout.render("precompute", a.dataset, combo=combo)
            / f"arm_ik_poses_{act[0]}_cont25"
            / f"{tree.name}.npz"
        )["base_z_off"]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        ok, why = SCL.pen_cross_guard(
            ctx,
            {s: pre[:, arm[s]].astype(np.float64) for s in ("right", "left")},
            z_off=z_off,
            desk_xy=VC.desk_bounds_scene_for(str(tree), margin=0.0),
            fingers=fq,
            supports=SSF.scene_supports(str(tree)),
        )
    dxy = VC.desk_bounds_scene_for(str(tree), margin=0.0)
    return {
        "app_guard_ok": bool(ok),
        "app_guard_why": why,
        "app_hand_desk_mm": {
            s: round(1000 * SCN.hand_desk_m(ctx[s], s, pre[:, arm[s]].astype(np.float64), [fq[s]], z_off, dxy), 1)
            for s in ("right", "left")
        },
    }


def arm_speed(states: np.ndarray, cols: dict, sides: list, dt: float) -> np.ndarray:
    """Active arm joint speed norm (rad/s between frames) — with two hands the larger."""
    return np.max(
        np.stack([np.linalg.norm(np.diff(states[:, cols[s]].astype(np.float64), axis=0), axis=1) / dt for s in sides]),
        0,
    )


def low_run(w: np.ndarray, thr: float, at: tuple[int, int] | None = None) -> int:
    """The longest run of consecutive frames with w < thr — with at=(i0, i1) only runs touching [i0, i1]."""
    best, i = 0, 0
    while i < len(w):
        if w[i] < thr:
            j = i
            while j < len(w) and w[j] < thr:
                j += 1
            if at is None or (j - 1 >= at[0] and i <= at[1]):
                best = max(best, j - i)
            i = j
        else:
            i += 1
    return best


def join_gate(cx, name: str, stitched: np.ndarray, K: int, act: list, arm: dict) -> dict:
    """Join stop gate (rules.JOIN_LOW_*): the frames of the active arm's slow run touching the join (below approach peak
    × ratio, containing the last approach step K-1 or the first policy step K, within the join ± window) ≤ the same
    value containing the v2 policy's first step (the --v2-ref RSI rollout, rank 0) + JOIN_LOW_TOL_STEPS. Slow runs away
    from the join (the policy's own slowing, also in v2) are only recorded (join_low_any_s). Without a v2 reference
    (v2ref 0 successes → no stored trajectory) the bound is only v2 ≥ 0: True when the v4 slow run ≤ the tolerance
    (true for
    any reference), else None (undecidable)."""
    dt = cx.dt
    W = int(round(V4.JOIN_LOW_WIN_S / dt))
    w = arm_speed(stitched, arm, act, dt)
    thr = V4.JOIN_LOW_FRAC * float(w[:K].max())
    a = max(0, K - W)
    win = w[a : K + W]
    v4_low = low_run(win, thr, at=(K - 1 - a, K - a))
    ref = cx.v2ref.get(name) if cx.v2ref is not None else None
    w2 = None if ref is None else arm_speed(ref[: W + 1], arm, act, dt)
    v2_low = None if w2 is None else low_run(w2, thr, at=(0, 0))
    return {
        "join_low_s": round(v4_low * dt, 2),
        "v2_low_s": None if v2_low is None else round(v2_low * dt, 2),
        "join_low_any_s": round(low_run(win, thr) * dt, 2),
        "v2_low_any_s": None if w2 is None else round(low_run(w2, thr) * dt, 2),
        "join_low_thr": round(thr, 3),
        "join_gate": (
            bool(v4_low <= v2_low + V4.JOIN_LOW_TOL_STEPS)
            if v2_low is not None
            else (True if v4_low <= V4.JOIN_LOW_TOL_STEPS else None)
        ),
    }


def variant_row(cx, name, pre, end, st, b, e, L, A, s0) -> dict:
    """One variant row (chosen trajectory = rank 0; st None = no stored rollout → approach rows only). cx =
    load_inputs' context.
    """
    from dexengine.placement.collision.geom import arm_link_transforms

    plan, mode, dt, jc, ctx = cx.plan, cx.mode, cx.dt, cx.jc, cx.ctx
    p = plan.get(name) or plan.get(name + "_pl_rt")
    K = len(pre)
    act = list(p["active"])
    stitched = pre if st is None else np.concatenate([pre, st], 0)
    rec = {
        "name": name,
        "mode": mode,
        "batch": b,
        "env": e,
        "K": K,
        "T_app_s": round(K * dt, 2),
        "L_roll": L,
        "len_v4": None if st is None else len(stitched),
        "plan_sides": p["sides"],
        "plan_guard": p["guard_ok"],
        "plan_guard_why": p["guard_why"],
        "policy_kept": st is not None,
    }
    arm = {s: [jc[f"{s}_arm_{i}"] for i in range(7)] for s in ("right", "left")}
    fing = {s: [i for n, i in jc.items() if n.startswith(s + "_") and "_arm_" not in n] for s in ("right", "left")}
    # wrist (palm frame) speed — FK of the recorded joints (IK frame)
    v = {}
    for s in act:
        T = arm_link_transforms(ctx[s], stitched[:, arm[s]].astype(np.float64))[:, -1] @ ctx[s]["ee_local"]
        v[s] = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) / dt
    vv = np.max(np.stack([v[s] for s in act]), 0)
    w0, w1 = max(0, K - JOIN_WIN), min(len(vv), K + JOIN_WIN)
    p90 = max(p["sides"][s]["clip_p90"] for s in act)
    rec.update(
        phys_peak_over_p90=round(float(vv[:K].max()) / p90, 2),
        join_min_speed=round(float(vv[w0:w1].min()), 4),
        join_slow_frames=int((vv[w0:w1] < SLOW_MPS).sum()),
        join_window=[w0, w1],
        join_slow_pre=int((vv[w0:K] < SLOW_MPS).sum()),  # approach side (min-jerk slowing tail)
        join_slow_post=int((vv[K:w1] < SLOW_MPS).sum()),  # policy side (the same RSI rest start as v2)
        phys_max_dq_frame=round(float(np.abs(np.diff(stitched[: K + 1][:, sum(arm.values(), [])], axis=0)).max()), 4),
    )
    if st is not None:
        rec.update(join_gate(cx, name, stitched, K, act, arm))
    if st is not None:
        # join residual: approach end state ↔ rollout row 0 (warm-up = RSI, every = the same state)
        aj, fj = sum(arm.values(), []), sum(fing.values(), [])
        rec["join_arm_rad"] = round(float(np.abs(end[aj] - st[0][aj]).max()), 4)
        rec["join_fing_rad"] = round(float(np.abs(end[fj] - st[0][fj]).max()), 4)
        rec["join_obj_mm"], rec["join_obj_deg"] = obj_join(end, st[0], cx.objs)
        # RSI objects are placed at the reference pose (speed 0) and settle in the first physics step (as in v2) — also
        # measured against row 1
        rec["join_obj_mm_row1"], rec["join_obj_deg_row1"] = obj_join(end, st[1], cx.objs)
    # approach end ↔ RSI residual (measured by the session, speeds included)
    rec["rsi_residual"] = {k[4:]: round(float(A[k][e]), 4) for k in A if k.startswith("res_")}
    # object motion during the approach (max; reference = the object after settling in the first physics step — row 1
    # when the start row is right after the reset (0))
    r0 = 1 if s0 == 0 else 0
    disp, rot = [], []
    for ob in cx.objs:
        lo = ob["lo"]
        disp.append(1000 * float(np.linalg.norm(pre[r0:, lo : lo + 3] - pre[r0, lo : lo + 3], axis=1).max()))
        dq = np.abs((pre[r0:, lo + 3 : lo + 7] * pre[r0, lo + 3 : lo + 7]).sum(1)).clip(max=1.0)
        rot.append(float(np.degrees(2 * np.arccos(dq)).max()))
    rec["app_obj_disp_mm"], rec["app_obj_rot_deg"] = round(max(disp), 2), round(max(rot), 2)
    k_ent = {s: int(v["k_ent"]) for s, v in ((p.get("scene") or {}).get("sides") or {}).items()}
    rec["app_touch"] = app_touch(cx.chans, pre, K, k_ent)
    rec["clip_first_touch"] = clip_first_touch(cx.chans, st)
    rec.update(app_guard(cx, name, act, pre, arm))
    rec["post_ok_rsi"] = cx.post_ok.get(name, cx.post_ok.get(name + "_pl_rt"))
    # t=0 state (all robot joints) — must have zero spread across episodes
    rec["t0_robot"] = [round(float(x), 6) for x in stitched[0][sorted(jc.values())]]
    return rec


def load_inputs(a) -> tuple:
    """roll folder, schedule, post gate, approach records → (context cx, rollout arrays Z, meta, clip names, approach
    records {batch: arrays}).
    """
    from dexengine.placement.collision import core as SCL

    with contextlib.redirect_stdout(io.StringIO()):
        ctx = {s: SCL.build_arm_ctx(*V4.COMBO, s) for s in ("right", "left")}
    z = np.load(a.rd / "run/rollouts.npz", allow_pickle=False)
    Z = {k: z[k] for k in z.files}
    meta = json.loads(str(Z["meta"]))
    ents = json.loads(str(Z["entities"]))
    chans = meta["env"]["collect"]["channels"]
    names = [str(x) for x in Z["clip_names"]]
    plan = json.loads(a.plan.read_text())
    post = json.loads((a.rd / "post.json").read_text())
    okr = {
        str(r["name"]): V4.ok_rank(post["kept"], str(r["name"])) for r in post["kept"]
    }  # variant → first passing rank
    post_ok = {n_: r_ is not None for n_, r_ in okr.items()}
    app = {
        int(p.stem.split("_b")[1]): dict(np.load(p, allow_pickle=False))
        for p in (a.rd / "approach").glob("approach_b*.npz")
    }
    mode = str(next(iter(app.values()))["mode"])
    v2ref = None
    if a.v2_ref is not None:  # v2 policy reference (RSI rollout — the warm-up round) · variant → rank-0 state
        zr = np.load(a.v2_ref / "run/rollouts.npz", allow_pickle=False)
        if json.loads(str(zr["entities"])) != ents:
            raise SystemExit(f"{MARK} --v2-ref {a.v2_ref} column layout differs from this rollout")
        nr = [str(x) for x in zr["clip_names"]]
        v2ref = {}
        for j in range(len(zr["kept_offset"])):
            if int(zr["kept_rank"][j]) == 0:
                o, L = int(zr["kept_offset"][j]), int(zr["kept_length"][j])
                v2ref[nr[int(zr["kept_clip"][j])]] = zr["states"][o : o + L]
    cx = types.SimpleNamespace(
        a=a,
        ctx=ctx,
        plan=plan,
        mode=mode,
        dt=float(meta["step_dt"]),
        jc=joint_cols(ents),
        objs=[e for e in ents if e["kind"] == "rigid"],
        chans=chans,
        post_ok=post_ok,
        okr=okr,
        v2ref=v2ref,
    )
    return cx, Z, meta, names, app


def stitch_rows(cx, Z: dict, names: list, app: dict) -> tuple[list, list, list, list]:
    """Per kept rollout the approach rows + rollout rows; rank 0 is the variant row; a variant without a stored rollout
    gets approach rows only (its first env of batch 0 — no policy values: join, first contact, length) → (variant
    rows,
    joined
    states, kept starts, kept lengths)."""
    mode = cx.mode
    rows_out, new_states, offs, lens = [], [], [], []
    for j in range(len(Z["kept_offset"])):
        o, L = int(Z["kept_offset"][j]), int(Z["kept_length"][j])
        st = Z["states"][o : o + L]
        c = int(Z["kept_clip"][j])
        name = names[c]
        er = Z["ep_table"][int(Z["kept_row"][j])]  # (clip, batch, env, warm, valid, succ, L)
        b, e = int(er[1]), int(er[2])
        A = app[0] if mode == "warmup" else app[b]
        keys = [str(k) for k in A["keys"]]
        if mode == "warmup":
            e = keys.index(name if name in keys else name + "_pl_rt")
        elif keys[e] not in (name, name + "_pl_rt"):
            raise SystemExit(f"{MARK} approach record b{b} env {e} clip {keys[e]} ≠ rollout {name}")
        N, s0 = A["rows"].shape[0] - 1, int(A["start"][e])
        pre = A["rows"][s0:N, e].astype(np.float32)
        end = A["rows"][N, e]
        stitched = np.concatenate([pre, st], 0)
        new_states.append(stitched)
        offs.append(sum(lens))
        lens.append(len(stitched))
        want = cx.okr.get(
            name, cx.okr.get(name + "_pl_rt")
        )  # variant row = the first passing rank (rank 0 when none — a failed variant)
        if int(Z["kept_rank"][j]) != (0 if want is None else want):
            continue
        rows_out.append(variant_row(cx, name, pre, end, st, b, e, int(L), A, s0))
    done_n = {r["name"] for r in rows_out} | {r["name"] + "_pl_rt" for r in rows_out}
    A0 = app[0]
    k0 = [str(k) for k in A0["keys"]]
    for key in sorted(cx.plan):
        if cx.plan[key].get("fail") or key in done_n or key not in k0:
            continue
        e0, N0 = k0.index(key), A0["rows"].shape[0] - 1
        s00 = int(A0["start"][e0])
        rows_out.append(
            variant_row(
                cx, key, A0["rows"][s00:N0, e0].astype(np.float32), A0["rows"][N0, e0], None, 0, e0, None, A0, s00
            )
        )
    return rows_out, new_states, offs, lens


def write_outputs(
    a, Z: dict, meta: dict, mode: str, rows_out: list, new_states: list, offs: list, lens: list
) -> tuple[Path, float]:
    """Layout 2 unchanged — only kept is replaced, the failure store (fb) follows as is; a copy of the run files and
    v4_metrics.json → (location, t=0 difference)."""
    fb_states, fb_off = [], []
    base = sum(lens)
    for k in range(len(Z["fb_offset"])):
        o, L = int(Z["fb_offset"][k]), int(Z["fb_length"][k])
        fb_off.append(base + sum(len(x) for x in fb_states))
        fb_states.append(Z["states"][o : o + L])
    out = a.rd / "stitched" / "run"
    out.mkdir(parents=True, exist_ok=True)
    meta["v4_stitch"] = {
        "mode": mode,
        "approach_frames": {r["name"]: r["K"] for r in rows_out},
        "owner": "experiments/real_kit/v4/stitch.py",
    }
    Z2 = dict(
        Z,
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
        states=np.concatenate(new_states + fb_states, 0).astype(np.float32),
        kept_offset=np.array(offs, np.int64),
        kept_length=np.array(lens, np.int64),
        fb_offset=np.array(fb_off, np.int64),
    )
    np.savez(out / "rollouts.npz", **Z2)
    for f in ("clips.txt", "variants.json", "diag.json", "run.json"):
        src = a.rd / "run" / f
        if src.is_file():
            (out / f).write_bytes(src.read_bytes())
    t0 = np.array([r["t0_robot"] for r in rows_out])
    var0 = float(np.abs(t0 - t0[0]).max()) if len(t0) else float("nan")
    (a.rd / "v4_metrics.json").write_text(
        json.dumps({"mode": mode, "t0_max_abs_diff": var0, "rows": rows_out}, indent=1, ensure_ascii=False)
    )
    return out, var0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rd", required=True, type=Path)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--shadow", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument(
        "--v2-ref", type=Path, default=None, help="v2 reference of the join-stop gate (RSI rollout round folder)"
    )
    a = ap.parse_args()
    cx, Z, meta, names, app = load_inputs(a)
    rows_out, new_states, offs, lens = stitch_rows(cx, Z, names, app)
    out, var0 = write_outputs(a, Z, meta, cx.mode, rows_out, new_states, offs, lens)
    for r in rows_out:
        print(
            f"{MARK} {r['name']} K={r['K']} peak/p90 {r['phys_peak_over_p90']} · join min {r['join_min_speed']} "
            "m/s · "
            f"slow {r['join_slow_frames']} (approach {r['join_slow_pre']} · policy {r['join_slow_post']}) · "
            f"join low speed {r.get('join_low_s')} s (v2 {r.get('v2_low_s')} · gate {r.get('join_gate')}) · "
            f"join residual arm {r.get('join_arm_rad')} fingers "
            f"{r.get('join_fing_rad')} object "
            f"{r.get('join_obj_mm')} mm/{r.get('join_obj_deg')}° (row 1 {r.get('join_obj_mm_row1')}/"
            f"{r.get('join_obj_deg_row1')}) · kept {r['policy_kept']}"
            " · approach object "
            f"displacement {r['app_obj_disp_mm']} mm · hand↔desk {r['app_hand_desk_mm']} mm · guard "
            f"{'pass' if r['app_guard_ok'] else r['app_guard_why']} · post (RSI) {r['post_ok_rsi']}",
            flush=True,
        )
    print(f"{MARK} DONE variants {len(rows_out)} · t=0 robot state max diff {var0:.2e} · {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
