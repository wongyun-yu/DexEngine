#!/usr/bin/env python3
"""Hand-touched receivers: how far the enabled collision pushes the main object off its reference (clearing distance)
-> hold list · mark list.

Why: a hand-touched static spawn that receives the main object is kind receiver (`extra_slots.hand_receiver`,
collision with the main object on). When the main object of the reference trajectory passes inside that object's
collision shape (bulb into lamp, placing into a bowl), physics pushes the main object out. When the pushed distance
exceeds the object tracking tolerance, tracking ends whatever the policy does; such clips leave the RL chain until the
problem is solved (held).

What it measures (kinematic, read only, the same formula for unseen trees):
  pairs           = the clip's (hand-touched receiver, fixed at the frame-0 pose) x (main object, reference
                    trajectory). Receivers the hands do not touch and main object pairs are not this command's targets
                    (already on, left as they are).
  collision shape = the shape the simulation collides (`collider_shape`: the receiver row's shape — its visual
                    triangles where S-true-shape is on, else its cooked convex parts — and the main object's cooked
                    convex parts). When unreadable the pair is held as 'unmeasured' and rc is 1 (never passed
                    silently).
  overlap depth   = `collider_shape.pair_depth` (both ways · every FRAME_STEP frames + first and last frame).
  clearing dist.  = the shortest distance along 14 world directions (±x ±y ±z · 8 diagonals) in STEP_M steps that brings
                    the overlap depth within CLEAR_TOL_M. Frames measured = the deepest overlap frame and the last
                    frame; the larger is the pair's value, the largest pair the clip's. Not cleared within MAX_M =
                    'over the cap' (held).
Thresholds (no second number; the owners are read):
  hold = clearing distance > tolerance. Tolerance = the canonical object tracking position threshold = the default of
         `--obj_pos_thresh_const` in `rl_args.build_parser()`.
  mark = MARK_FROM_M < clearing distance <= tolerance; not held. Names only, to look at separately in the results (one
         file).

Outputs (`--out-dir D`, one hand, several datasets):
  D/<id>_<dataset>_<hand>.txt  hold list, the form the RL queue reads (one clip name per line · '#'
                                  = description). Written even with 0 targets (the old list is emptied; an empty
                                  list = 0 holds).
  D/<id>_mark_<hand>.tsv           mark list (dataset · axis · clip · receiver · main object · clearing distance mm).
  D/recv_clearance_<hand>.tsv      one line per pair (overlap depth · frame · clearing distance · band
                                   hold|mark|ok|unmeasured).
Moving D's hold lists to the queue root `issue_hold/` is the caller's step (this command does not write the queue).
Markers: one `[RECV_CLEAR] tolerance …` line (the value read and its owner) · per dataset `[RECV_CLEAR] <hand>
<dataset>: …` · end line PASS|FAIL.

Usage: <retarget venv> dexengine/record/augment/recv_clearance.py --hand H --datasets a,b --out-dir D [--jobs 4]
       python3 dexengine/record/augment/recv_clearance.py --selftest      (synthetic boxes · no data · negative tests)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from dexengine.core import stdio
from dexengine.core.errors import InputMissingError
from dexengine.data.objects import collider_shape as CS

TAG = "[RECV_CLEAR]"
# id at the head of the hold list file names (<id>_<dataset>_<hand>.txt: a queue reads hold lists by a numeric id)
HOLD_ID = "38"
FRAME_STEP = 3  # frame stride of the overlap depth (first and last always measured)
STEP_M = 0.002  # clearing distance step
# clearing distance search cap; must exceed the tolerance (otherwise holds cannot be told apart, dies)
MAX_M = 0.10
CLEAR_TOL_M = 0.001  # overlaps within this depth count as touching (cooked part slack)
MARK_FROM_M = 0.03  # lower end of the mark band; not a hold threshold (a viewing range only, the 3 cm band)
BANDS = ("hold", "mark", "ok", "unmeasured")
_DIRS = np.array(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    + [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)],
    float,
)
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)
_PIECES: dict = {}


def tolerance() -> float:
    """Object tracking position tolerance (m): owner = the `--obj_pos_thresh_const` default of
    `rl_args.build_parser()`. <= 0 (left to the cfg schedule = not a constant) or >= the search cap dies (the rule
    cannot apply)."""
    from dexengine.rl import args as rl_args  # noqa: PLC0415  (pure argparse)

    v = rl_args.build_parser().get_default("obj_pos_thresh_const")
    if not isinstance(v, float) or not 0.0 < v < MAX_M:
        raise SystemExit(
            f"{TAG} FAIL tolerance {v!r}: 0 < value < search cap {MAX_M} m is needed to tell holds apart "
            "(owner rl_args --obj_pos_thresh_const)"
        )
    return v


def band(clear_m: float | None, tol: float) -> str:
    """Clearing distance -> band. None = not cleared up to the cap (held)."""
    if clear_m is None or clear_m > tol:
        return "hold"
    return "mark" if clear_m > MARK_FROM_M else "ok"


def obj_body(ds: str, o: dict, row: dict | None = None):
    """The object's collision shape as physics uses it (`collider_shape.body`, body frame): the receiver row's shape,
    or a main object's cooked convex parts (`row` None). Rigid bodies only (one link)."""
    shape = CS.row_shape(ds, row or {})  # no row: a moving main object (collider_shape: parts)
    key = (ds, str(o["obj_id"]), shape)
    if key not in _PIECES:
        if o.get("articulation"):
            raise ValueError("articulated: each link has its own pose, not a target of this measurement")
        with stdio.capture():
            _PIECES[key] = CS.body(ds, str(o["obj_id"]), shape)
    return _PIECES[key]


def _rot(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class Pair:
    """Overlap of fixed object A (frame-0 pose) x moving object B (reference trajectory), on their rulers
    (collider_shape bodies): signed depth sd(t, shift) (minus the pair depth) · clearing distance clear(t)."""

    def __init__(self, q: np.ndarray, a_start: int, b_start: int, A, B):
        self.q, self.sb, self.A, self.B = q, int(b_start), A, B
        self.Ra, self.pa = _rot(q[0, a_start + 3 : a_start + 7]), q[0, a_start : a_start + 3].astype(float)

    def sd(self, t: int, shift=(0.0, 0.0, 0.0)) -> float:
        Rb, pb = _rot(self.q[t, self.sb + 3 : self.sb + 7]), self.q[t, self.sb : self.sb + 3].astype(float)
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = self.Ra.T @ Rb, self.Ra.T @ (pb + np.asarray(shift, float) - self.pa)  # B in A
        return -CS.pair_depth(self.A, self.B, M)

    def clear(self, t: int) -> float | None:
        """Clearing distance (m) at frame t: 0 = no overlap · None = not cleared up to MAX_M."""
        if self.sd(t) > -CLEAR_TOL_M:
            return 0.0
        best = None
        for d in _DIRS:
            for s in np.arange(STEP_M, (best if best is not None else MAX_M) + STEP_M / 2, STEP_M):
                if best is not None and s >= best:
                    break
                if self.sd(t, d * s) > -CLEAR_TOL_M:
                    best = float(s)
                    break
        return best


def measure(q: np.ndarray, a_start: int, b_start: int, A, B) -> dict:
    """One pair (rulers A fixed, B moving) -> {sd_min_m, frame, clear_m (None = over the cap), clear_end_m}."""
    p, last = Pair(q, a_start, b_start, A, B), len(q) - 1
    fr = sorted(set(range(0, len(q), FRAME_STEP)) | {0, last})
    s = np.array([p.sd(t) for t in fr])
    k = int(s.argmin())
    c_min, c_end = p.clear(fr[k]), p.clear(last)
    worst = None if c_min is None or c_end is None else max(c_min, c_end)
    return {"sd_min_m": float(s[k]), "frame": int(fr[k]), "clear_m": worst, "clear_end_m": c_end}


def one_tree(tree: str) -> list[dict]:
    """One tree -> one row per pair ([] without pairs). Unmeasured = the row's `err`."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    with open(os.path.join(tree, "task_info.json"), encoding="utf-8") as f:
        ti = json.load(f)
    q = np.load(os.path.join(tree, "0", "trajectory_kinematic.npz"))["qpos"]
    ds = str(ti.get("dataset_name") or "")
    objs = {str(o["name"]): o for o in ti.get("objects") or []}
    out = []
    for r in (r for r in XS.clip_rows(tree, ti, q, False) if XS.hand_receiver(r)):
        for m in (o for o in objs.values() if o.get("role") == "main" and o.get("qpos_start") is not None):
            row = {
                "tree": os.path.basename(tree),
                "slot": r["name"],
                "slot_obj": str(r["obj_id"]),
                "main": str(m["name"]),
                "main_obj": str(m["obj_id"]),
                "recv_by": r.get("recv_by"),
            }
            try:
                row.update(
                    measure(
                        q, int(r["qpos_start"]), int(m["qpos_start"]), obj_body(ds, objs[r["name"]], r), obj_body(ds, m)
                    )
                )
            except Exception as e:  # noqa: BLE001  (unmeasured stays as a row; the caller counts it: hold + rc 1)
                row["err"] = f"{type(e).__name__}: {str(e)[:160]}"
            out.append(row)
    return out


def _work(tree: str) -> tuple:
    try:
        return tree, one_tree(tree), ""
    except Exception as e:  # noqa: BLE001  (the whole tree unreadable: the caller counts it, rc 1)
        return tree, [], f"{type(e).__name__}: {str(e)[:160]}"


def _mm(v) -> str:
    return "" if v is None else f"{v * 1e3:.1f}"


def _ledger_sources(hand: str) -> list[dict]:
    """[{axis, dataset, ...}] per folded ledger of the hand (the dataset axis of each ledger file)."""
    from dexengine.core import paths as P
    from dexengine.data import rules as RR

    files = RR.ledger_files("*", hand, root=P.main_root())
    if not files:
        raise SystemExit(f"[manifest] {hand}: no folded ledger — the stage-1 and judge chain builds it first")
    out = []
    for f in files:
        ax = RR.ledger_axis(f)
        ds, emb, _v = RR.ledger_axis_parts(ax)
        out.append(
            {
                "axis": ax,
                "dataset": ds,
                "emb": emb,
                "variant": RR.ledger_axis_variant(ax),
                "combo": RR.combo(hand, RR.ledger_axis_variant(ax)),
                "source": f"ledger:{f}",
            }
        )
    return out


def _measure(a, want, src, tol, marks, table, bad):
    """Measure every source in a process pool sized by capacity: table rows, marker lines and the bad count."""
    import multiprocessing as mp  # noqa: PLC0415

    from dexengine.core import capacity  # noqa: PLC0415
    from dexengine.core import paths as P  # noqa: PLC0415

    step = "dexengine.record.augment.recv_clearance"  # capacity.NEEDS key (this module)
    with capacity.sized(step, len(src), a.jobs) as cap, mp.Pool(cap.workers, capacity.worker_init, cap.initargs) as p:
        for ds in sorted(want):
            hold, n_tree, n_pair, n_band = set(), 0, 0, dict.fromkeys(BANDS, 0)
            for s in (s for s in src if s["dataset"] == ds):
                base = str(P.layer_path("trees", ds, combo=s["combo"], emb=s["emb"]))
                trees = [
                    os.path.join(base, n)
                    for n in (sorted(os.listdir(base)) if os.path.isdir(base) else [])
                    if os.path.isfile(os.path.join(base, n, "task_info.json"))
                ]
                n_tree += len(trees)
                for tree, rows, err in p.imap(_work, trees, chunksize=8):
                    clip = P.clip_name(os.path.basename(tree))
                    if err:
                        bad += 1
                        print(f"{TAG} ✗ {ds} {s['axis']} {clip}: tree unreadable: {err}", flush=True)
                    for r in rows:
                        b = "unmeasured" if "err" in r else band(r["clear_m"], tol)
                        n_pair, n_band[b] = n_pair + 1, n_band[b] + 1
                        if b in ("hold", "unmeasured"):
                            hold.add(clip)
                        if b == "mark":
                            marks.append((ds, s["axis"], clip, r["slot_obj"], r["main_obj"], _mm(r["clear_m"])))
                        cm = "" if "err" in r else "over the cap" if r["clear_m"] is None else _mm(r["clear_m"])
                        table.append(
                            (
                                ds,
                                s["axis"],
                                clip,
                                r["slot"],
                                r["slot_obj"],
                                r["main"],
                                r["main_obj"],
                                _mm(r.get("sd_min_m")),
                                str(r.get("frame", "")),
                                cm,
                                _mm(r.get("clear_end_m")),
                                b,
                                r.get("err", ""),
                            )
                        )
            bad += n_band["unmeasured"]
            f = os.path.join(a.out_dir, f"{HOLD_ID}_{ds}_{a.hand}.txt")
            _write(
                f,
                [
                    f"# hold list {HOLD_ID}: clearing distance with the hand-touched receiver on > tolerance "
                    f"{tol * 100:g} cm (or unmeasured) · {a.hand} · {ds} · {len(hold)} clips · generator "
                    "dexengine/record/augment/recv_clearance.py"
                ]
                + sorted(hold),
            )
            print(
                f"{TAG} {a.hand} {ds}: trees {n_tree} · pairs {n_pair} · hold pairs {n_band['hold']} · mark pairs "
                f"{n_band['mark']} · within tolerance {n_band['ok']} · unmeasured {n_band['unmeasured']} -> held "
                f"clips {len(hold)} → {f}",
                flush=True,
            )
    return bad


def run(a) -> int:

    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data import rules as RR  # noqa: PLC0415

    tol = tolerance()
    print(
        f"{TAG} tolerance {tol * 100:g} cm (owner rl_args --obj_pos_thresh_const default) · hold = clearing "
        f"distance > {tol * 100:g} cm · mark = over {MARK_FROM_M * 100:g} cm – up to {tol * 100:g} cm · targets = "
        f"hand-touched receiver x main object (extra_slots.hand_receiver) · tree {P.main_root()}",
        flush=True,
    )
    want = RR.datasets_scope(a.datasets)
    src = [s for s in _ledger_sources(a.hand) if s["dataset"] in want]
    if off := sorted(set(want) - {s["dataset"] for s in src}):
        raise InputMissingError(f"{TAG} FAIL {a.hand}: datasets {off} have no folded ledger axis: no trees to measure")
    os.makedirs(a.out_dir, exist_ok=True)
    table, marks, bad = [], [], 0
    bad = _measure(a, want, src, tol, marks, table, bad)
    _write(
        os.path.join(a.out_dir, f"{HOLD_ID}_mark_{a.hand}.tsv"),
        ["\t".join(("dataset", "axis", "clip", "receiver_obj", "main_obj", "clear_mm"))]
        + ["\t".join(x) for x in sorted(set(marks))],
    )
    _write(
        os.path.join(a.out_dir, f"recv_clearance_{a.hand}.tsv"),
        [
            "\t".join(
                (
                    "dataset",
                    "axis",
                    "clip",
                    "slot",
                    "receiver_obj",
                    "main",
                    "main_obj",
                    "sd_min_mm",
                    "frame",
                    "clear_mm",
                    "clear_end_mm",
                    "band",
                    "err",
                )
            )
        ]
        + ["\t".join(x) for x in sorted(table)],
    )
    print(
        f"{TAG} {'FAIL' if bad else 'PASS'} {a.hand}: pairs {len(table)} · mark clips {len({m[:3] for m in marks})} "
        f"· unmeasured {bad} (held) -> {a.out_dir}",
        flush=True,
    )
    return 1 if bad else 0


def _write(path: str, lines: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def selftest() -> int:
    """Synthetic boxes (receiver with 10 cm sides · main object with 2 cm sides): planting a depth splits the bands,
    removing it clears them."""
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'} {name}")

    def box(h: float) -> list:
        g = np.array([-h, h])
        return CS.Parts([np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)])

    tol = tolerance()
    A, B = box(0.05), box(0.01)

    def clip(depth: float, T: int = 7) -> np.ndarray:
        """The main object goes `depth` below the receiver top (+z 0.05) and comes out upward at the end (the middle
        frame is the deepest)."""
        q = np.zeros((T, 14), np.float32)
        q[:, 3] = q[:, 10] = 1.0
        z = 0.05 + 0.01 - depth * (1.0 - np.abs(np.linspace(-1.0, 1.0, T)))
        q[:, 9] = z + 0.02 * (np.linspace(-1.0, 1.0, T) > 0.9)
        return q

    def got(depth: float) -> tuple:
        m = measure(clip(depth), 0, 7, A, B)
        return m["clear_m"], band(m["clear_m"], tol)

    check(
        f"tolerance = owner value (rl_args default {tol} m) · between 0 and the search cap {MAX_M} m", 0.0 < tol < MAX_M
    )
    c, b = got(tol + 0.012)
    check(
        f"1.2 cm deeper than the tolerance -> hold (clearing distance {c})", b == "hold" and c is not None and c > tol
    )
    c, b = got(tol - 0.008)
    check(
        f"negative: 0.8 cm shallower than the tolerance -> not held · marked (clearing distance {c})",
        b == "mark" and MARK_FROM_M < c <= tol,
    )
    c, b = got(0.012)
    check(f"negative: only 1.2 cm in -> band ok (clearing distance {c})", b == "ok" and 0.0 < c <= MARK_FROM_M)
    c, b = got(-0.004)
    check("negative: no touch -> clearing distance 0 · band ok", (c, b) == (0.0, "ok"))
    check(
        "band edge: a value equal to the tolerance is not held (only > holds) · not cleared (None) is held",
        band(tol, tol) == "mark" and band(None, tol) == "hold" and band(MARK_FROM_M, tol) == "ok",
    )
    end = clip(0.02)
    end[-1, 9] = 0.05 + 0.01 - 0.045  # deep only at the last frame: is the last frame measured?
    m = measure(end, 0, 7, A, B)
    check(
        "ending while deep at the last frame makes that the pair's value",
        m["clear_end_m"] is not None and m["clear_m"] == m["clear_end_m"] and m["clear_m"] > 0.03,
    )
    print(f"{TAG} selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hand")
    ap.add_argument("--datasets", help="comma list: every tree of those datasets' folded ledger axes is measured")
    ap.add_argument("--out-dir")
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.hand and a.datasets and a.out_dir):
        ap.error("--hand · --datasets · --out-dir are required (no defaults)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
