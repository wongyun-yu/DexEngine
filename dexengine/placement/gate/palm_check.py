#!/usr/bin/env python3
"""Post-bake palm-frame oracle: on baked trees, compare the rig fingertips with the training robot's (merged URDF)
fingertips in the same world frame.

Why: the judge and the bake build the arm IK target as "rig palm FK body position + R · palm_frame_offset_pos" and put
it on the merged URDF's palm frame (urdf_palm_frame). If the two bodies are not the same point and that translation is
missing from the spec, the training robot's hand is shifted as a whole from the reference (rig FK, object placement),
while the gate, which measures tracking with the same target formula, sees 0. This oracle does not use the target
formula: it runs FK of the merged URDF from the tree qpos (fingers) and the IK npz (7 arm joints) and subtracts the
same links of the tree's body_positions (rig FK). Measured: with the palm translation missing, DG-5F-S fingertips are
off by 15.00 mm; a correct hand by 0.00–0.01 mm.

Mode (called by the stage-2 driver after the bake):
    --clips <baked_ok.txt> --tree-root <TROOT> --robots-dir <W2> --dataset <ds> --assets-root … [--sample 3]
    Only this bake's OK clips (locations from paths.bake_outputs); --sample trees spread over the name order are
    measured (an undecidable tree is replaced by the next one, see 3 below). FAIL -> rc 1.

What is measured — the definition error only:
    delta(rig FK tip - merged URDF FK tip) = definition error (wrong target formula / spec) + arm tracking error (the IK
    residual the judge allowed). Mixing the two would spread a wrist ROTATION tracking error (10–54°, position <= 0.4
    mm) into 12–45 mm at the fingertips and read as a definition error. So:
      1 only active hands are measured — the idle hand (collision.idle_sides_of_tree, task_info idle_sides) is left out;
      2 only frames where the arm followed its target — the IK npz's wrist tracking errors (wrist_pos_error_all,
        wrist_rot_error_deg_all, recomputed by the bake chain after cross repair and retiming) <= --trk-pos-mm and
        --trk-rot-deg, --frames of them spread evenly. They are not recomputed here (the oracle does not use the target
        formula). A definition error stays even on frames tracked perfectly, so it is caught;
      3 a side with fewer tracked frames than --min-frames is undecidable (never a silent pass). The tool moves to the
        next tree of the list until --sample trees have every active side decidable (at most --max-trees).

Verdict: per side and fingertip, |delta| p50 <= --tol-mm (default 1 mm) is OK. Any BAD -> FAIL (rc 1). No fingertip
measured -> UNDECIDABLE (rc 3); the fingertips of the decidable side of a partly decidable tree count. Tree / IK
generation mismatch or missing input (incl. no tracking-error arrays) -> rc 2. The `[PALM_ORACLE] PASS|FAIL|UNDECIDABLE`
line is the run's marker; without it the gate did not run.

Interpreter: Isaac venv (pinocchio, mujoco), on CPU (`CUDA_VISIBLE_DEVICES=`). Read-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# isort: split
from dexengine.core.errors import InputMissingError, input_missing
from dexengine.core.paths import bake_outputs, read_clip_list  # owner of the bake output locations and the clip list
from dexengine.placement.collision import cross_sets as CC
from dexengine.placement.collision import idle_arm as IA  # (owner of the idle side: idle_sides_of_tree)
from dexengine.placement.coords import core as VC

#: The tracking-error arrays of the IK npz (written by the arm IK and recomputed by cross_repair). Missing = an input
#: defect (rc 2).
TRK_POS_KEY, TRK_ROT_KEY = "wrist_pos_error_all", "wrist_rot_error_deg_all"


def tracked_frames(ik_npz: Path, T: int, pos_mm: float, rot_deg: float) -> np.ndarray:
    """Frame indices whose wrist tracking errors in the IK npz are within the thresholds. Missing arrays or a length
    different from the tree's -> SystemExit (rc 2 at the caller)."""
    z = np.load(ik_npz, allow_pickle=True)
    miss = [k for k in (TRK_POS_KEY, TRK_ROT_KEY) if k not in z.files]
    if miss:
        raise InputMissingError(
            f"[palm_oracle] ✗ {ik_npz.name}: no tracking-error arrays {miss} — "
            f"cannot pick the frames where the arm followed its target (no guessing)"
        )
    pe = np.asarray(z[TRK_POS_KEY], float).reshape(-1) * 1000.0
    re_ = np.asarray(z[TRK_ROT_KEY], float).reshape(-1)
    if len(pe) != T or len(re_) != T:
        raise SystemExit(
            f"[palm_oracle] ✗ {ik_npz.name}: tracking-error length {len(pe)}/{len(re_)} != tree {T} — "
            f"not the same generation"
        )
    return np.where((pe <= pos_mm) & (re_ <= rot_deg))[0]


def scene_z_off(tree: Path) -> tuple[float, str]:
    """The height offset at which the env places this tree — the same reader as the training loader
    (coords.scene_z_offset -> aerial_support -> read_aerial_support, i.e. qpos_layout.read -> aerial['z_off']).

    The z_off of an aerial scene is stamped by the bake into `task_info.aerial_support` right after placement, and the
    IK and the training loader use that stamp. Recomputing it on a baked tree would measure the whole hand as shifted
    vertically after cross repair (objects moved with the hand) and retiming (the first rest interval's middle frame
    moves). A desk scene uses the desk_z constant. An aerial tree without a stamp is another bake generation and dies.
    Returns (z_off, source label)."""
    try:
        z = float(VC.scene_z_offset(tree))
    except ValueError as e:
        raise SystemExit(f"[palm_oracle] ✗ {tree.name}: {e}") from e
    return z, ("stamp (aerial_support)" if VC.scene_mode(tree) == "aerial" else "desk (desk_z)")


def measure(tree: Path, ik_right: Path, ik_left: Path, assets_root: str, a) -> tuple[int, int, list[str], str]:
    """One tree -> (bad fingertips, measured fingertips, undecidable sides, summary). Input defects -> SystemExit."""
    L = CC.HandLayout(tree, quiet=True)
    z_off, z_src = scene_z_off(tree)
    print(f"[ZOFF] {tree.name}: {z_off:.4f} ({z_src})", flush=True)
    model = CC.CrossSetModel(L, tree_dir=tree, assets_root=assets_root, robust_zero=True, desk_h=z_off)
    ik = {"right": CC.load_ik(ik_right), "left": CC.load_ik(ik_left)}
    ikp = {"right": ik_right, "left": ik_left}
    q = np.load(tree / "0/trajectory_kinematic.npz")["qpos"]
    bp = np.load(tree / "0/body_positions.npz", allow_pickle=True)
    T = len(q)
    if any(len(ik[s][1]) != T for s in ik) or len(bp["right_hand_body_pos"]) != T:
        raise SystemExit(
            f"[palm_oracle] ✗ {tree.name}: length mismatch tree {T} · ik {[len(ik[s][1]) for s in ik]} · "
            f"body {len(bp['right_hand_body_pos'])} — not the same generation"
        )
    base = np.asarray(ik["right"][2], float)
    idle = set(IA.idle_sides_of_tree(tree))
    active = [s for s in L.hands if s not in idle]
    if not active:
        raise InputMissingError(
            f"[palm_oracle] ✗ {tree.name}: no active hand (hands={sorted(L.hands)} idle={sorted(idle)})"
        )
    bad, n, worst = 0, 0, 0.0
    undecided: list[str] = []
    lines = []
    for side in active:
        trk = tracked_frames(ikp[side], T, a.trk_pos_mm, a.trk_rot_deg)
        if len(trk) < a.min_frames:
            undecided.append(side)
            lines.append(
                f"  --- {side:5} undecidable — tracked frames {len(trk)}/{T} < {a.min_frames} "
                f"(|Δp| <= {a.trk_pos_mm} mm · |Δθ| <= {a.trk_rot_deg}°)"
            )
            continue
        idx = trk[np.unique(np.linspace(0, len(trk) - 1, min(a.frames, len(trk))).astype(int))]
        names = [str(x) for x in bp[f"{side}_hand_body_names"]]
        P = np.asarray(bp[f"{side}_hand_body_pos"], float)
        for tip in L.tips:
            link = tip.replace("right_", f"{side}_", 1)
            fid = model.model.getFrameId(link)
            if link not in names or fid >= model.model.nframes:
                raise InputMissingError(
                    f"[palm_oracle] ✗ {tree.name}: {side} {link} is not in body_positions / the merged URDF"
                )
            bi = names.index(link)
            dw, dl = [], []
            for t in idx:
                fq, _ = L.hand_blocks(q[t])
                qq = model.build_q({s: (ik[s][0], ik[s][1][t]) for s in ("right", "left")}, fq)
                model.pin.framesForwardKinematics(model.model, model.data, qq)
                M = model.data.oMf[fid]
                d = (np.asarray(M.translation) + base) - (CC.RZ_NEG90 @ P[t, bi] + np.array([0.0, 0.0, model.desk_h]))
                dw.append(d)
                dl.append(np.asarray(M.rotation).T @ d)
            e = np.linalg.norm(np.array(dw), axis=1) * 1000
            ok = float(np.median(e)) <= a.tol_mm
            bad += 0 if ok else 1
            n += 1
            worst = max(worst, float(np.median(e)))
            loc = np.round(np.mean(dl, axis=0) * 1000, 2).tolist()
            lines.append(
                f"  {'OK ' if ok else 'BAD'} {side:5} {link:34} |Δ| p50 {np.median(e):7.2f} max {e.max():7.2f} mm · "
                f"frames {len(idx)} (tracked {len(trk)}/{T}) · body-local mean {loc}"
            )
    rest = f" · idle {sorted(idle)}" if idle else ""
    und = f" · undecidable {undecided}" if undecided else ""
    head = f"{tree.name} · hand {L.hand} · active {active}{rest} · tips {n - bad}/{n} · worst p50 {worst:.2f} mm{und}"
    return bad, n, undecided, head + "\n" + "\n".join(lines)


def verdict_of(tot_bad: int, tot_n: int) -> tuple[str, int]:
    """(verdict, rc) from the measured fingertip count (tot_n) and the BAD count only."""
    if tot_bad:
        return "FAIL", 1
    if tot_n == 0:
        return "UNDECIDABLE", 3
    return "PASS", 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", required=True, help="bake-OK list (baked_ok.txt)")
    ap.add_argument("--tree-root", required=True, help="baked tree folder (stage 2's TROOT)")
    ap.add_argument(
        "--dataset", required=True, help="dataset of the list — clip name normalisation (paths.read_clip_list)"
    )
    ap.add_argument("--robots-dir", required=True, help="placement and arm IK root (stage 2's W2)")
    ap.add_argument("--sample", type=int, default=3, help="trees to measure (spread over the name order)")
    ap.add_argument("--assets-root", required=True)
    ap.add_argument("--tol-mm", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=40, help="frames per side, spread over the tracked frames")
    ap.add_argument(
        "--trk-pos-mm", type=float, default=0.25, help="tracked-frame threshold — wrist position error (mm)"
    )
    ap.add_argument(
        "--trk-rot-deg",
        type=float,
        default=0.1,
        help="tracked-frame threshold — wrist rotation error (°). 0.1° x a 0.2 m tip lever ≈ 0.35 mm",
    )
    ap.add_argument("--min-frames", type=int, default=10, help="tracked frames a side needs to be decidable")
    ap.add_argument("--max-trees", type=int, default=12, help="cap on the trees tried while looking for decidable ones")
    a = ap.parse_args()
    listed = read_clip_list(a.clips, dataset=a.dataset, label="palm clips", quiet=True)
    outs = sorted((bake_outputs(c, a.tree_root, a.robots_dir) for c in listed), key=lambda o: o["tree"].name)
    outs = [o for o in outs if (o["tree"] / "0/trajectory_kinematic.npz").is_file()]
    if not outs:
        print(input_missing(f"[palm_oracle] ✗ no tree in the bake-OK list: {a.clips}"), flush=True)
        return 2
    first = sorted(set(np.linspace(0, len(outs) - 1, min(a.sample, len(outs))).astype(int).tolist()))
    order = first + [i for i in range(len(outs)) if i not in first]  # the spread sample first, then the rest by name
    jobs = [(outs[i]["tree"], outs[i]["ik_right"], outs[i]["ik_left"]) for i in order[: max(a.max_trees, a.sample)]]
    want = a.sample
    tot_bad, tot_n, judged, undec, tried = 0, 0, 0, 0, 0
    und_sides: dict[str, int] = {}
    for tree, ikr, ikl in jobs:
        if judged >= want:
            break
        for p in (ikr, ikl):
            if not p.is_file():
                print(input_missing(f"[palm_oracle] ✗ {tree.name}: no IK npz {p}"), flush=True)
                return 2
        b, n, und, txt = measure(tree, ikr, ikl, a.assets_root, a)
        tried += 1
        tot_bad += b
        tot_n += n
        if und:
            undec += 1
            for s in und:
                und_sides[s] = und_sides.get(s, 0) + 1
        else:
            judged += 1
        print(txt, flush=True)
    short = f" · sample short {judged}/{want}" if 0 < judged < want else ""
    short += f" · undecidable sides {dict(sorted(und_sides.items()))}" if und_sides else ""
    verdict, rc = verdict_of(tot_bad, tot_n)
    print(
        f"[PALM_ORACLE] {verdict} — decided trees {judged} (tried {tried} · undecidable {undec}){short} · "
        f"tips {tot_n - tot_bad}/{tot_n} · "
        f"threshold {a.tol_mm} mm · tracked frames |Δp|<={a.trk_pos_mm} mm·|Δθ|<={a.trk_rot_deg}°·>={a.min_frames} "
        f"(rig FK <-> merged URDF FK, active hands, no target formula)",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
