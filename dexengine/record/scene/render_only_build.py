#!/usr/bin/env python3
"""Generator of the record's render-only scene object list (original capture objects outside the tree): single owner
of what goes into the list (shape, reading and moving into the tree are owned by scene.render_only).

RL puts only main, auxiliary and extra objects into the scene; the record render puts **every** object of the original
scene as a visual mesh, as long as it does not overlap the robot trajectory. Extras in the tree's objects[] are handled
by scene.extras already; this file lists the original objects **not** in the tree. The tree stays as it is (the
training and judging env do not read this list, so RL is unaffected).

Membership (outside the tree or not), one path for every dataset: the raw scene measurement of readers/raw as it is
  (module window, measure, scene_classify.reduce_window, tree_scene(ti, obj_key)). Original objects with samples
  inside the window [w0, w1) whose raw id is not in the tree.
  Clips = the union over hands of the base layer clips (paths.HAND_IDS). Base tree = the base tree of the canonical axis
  hand (data.rules.AXIS_HAND) · clips missing for the axis hand (per-hand windows: `_t<k>` pieces only that hand
  has) use the base tree of the first hand by name that has the clip (the record's base_tree says which); base frames
  do not depend on the hand (measured 0.0000 mm). Counting only the axis hand would leave those pieces out of the
  index, and the reader read_clip would die with 'clip not in the index'. A root without the axis hand (for example a
  root that holds wuji2 only) is stated by the caller (--root-hands wuji2, checked against the base layers); then
  every clip follows that same rule, the run says so once (MARK line) and the windows of every clip are checked
  across hands. Without the statement a missing axis-hand base dies as before.
Record entry: the record builds a missing index with generate (scene.render_only_entry owns when and with which hands).
Pose: the raw -> base transform has its own owner per dataset (POSE_SOURCES, the module's render_only_tracks). A
  dataset with objects outside the tree but no transform owner makes that clip an error (never 0 silently; rc 1).
  Record = the 7-field pose of base row 0 (clip f0) · max move move_m and rotation rot_deg over the window relative to
  f0 (measurement only; the static thresholds belong to scene.extras). The placement (xy, yaw) is rigid and does not
  change move or rotation sizes. When the f0 row is outside the object's raw frame range: reason no_pose (no
  extrapolation).
Asset, same object id and same mesh rule: asset folders (assets_objects layer) whose dataset obj_key is that rid, whose
  visual.obj has the same vertex and face counts as the original mesh with vertex and face triangle position
  differences <= MESH_TOL_M, and that have a training USD and surface samples (code and tolerance:
  scene.render_only_assets). With several, the first by name (all the same mesh; the choice and the candidates are
  recorded). None: reason no_asset (rc 1).
Writing: clip files -> index (merged). --plan-out = tsv of write targets (path, dataset, clip, exists) · --plan-only
  = no writing.
Marker [RENDER_ONLY]: per dataset an end line `[RENDER_ONLY] end <ds> clips n · clips with outside objects · objects
m · … · rc r`.

Usage: DEXENGINE_MAIN_ROOT=<data root> OMP_NUM_THREADS=1 nice -n 19 <retarget venv python> -m
        dexengine.record.scene.render_only_build --dataset <ds>[,<ds>…] [--clips a,b] [--out <root>] [--plan-out <tsv>]
        [--plan-only] [--jobs N] [--root-hands <hand>[,<hand>…]]
      (CPU only, one thread. No --out = write to the canonical layer · --out = the same shape under that root (dry
      run))
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dexengine.record.scene import render_only as RL  # home of the anchor rule (anchor_starts)
from dexengine.record.scene.render_only import (
    INDEX,
    MARK,
    NAME_PREFIX,
    REASONS,
    VERSION,
    RenderOnlyError,
    clip_file,
    list_dir,
    oracle_module,
)
from dexengine.record.scene.render_only_assets import resolve_asset

#: owners of the raw -> base transform (dataset -> module, function render_only_tracks(seq, sk, ti, kin, rids)), read
#: through pose_source. A dataset missing from the table that has objects outside the tree dies: the table says 'where
#: the transform is known', not 'where objects exist'.
POSE_SOURCES = {"oakink2": "dexengine.data.oakink2.raw_base"}
EMBS = ("bimanual", "right", "left")


def pose_source(ds: str):
    """The raw -> base transform of dataset ds: POSE_SOURCES[ds].render_only_tracks. A module that does not import (a
    name from before the package layout) or has no such function dies naming the table entry."""
    import importlib

    name = POSE_SOURCES[ds]
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise RenderOnlyError(f"POSE_SOURCES[{ds!r}] = {name!r} does not import ({e})") from e
    fn = getattr(mod, "render_only_tracks", None)
    if not callable(fn):
        raise RenderOnlyError(f"POSE_SOURCES[{ds!r}] = {name!r} has no render_only_tracks")
    return fn


def axis_hand() -> str:
    from dexengine.data import rules as RR

    return str(RR.AXIS_HAND)


def hand_trees(ds: str, h: str) -> dict:
    """{clip: base tree} of hand h over the embs; the same clip in two embs of one hand dies."""
    from dexengine.core import paths as P

    hand, mine = P.hand_spec_dir(h), {}
    for emb in EMBS:
        d = P.layer_path("base", ds, hand=hand, emb=emb)
        if not d.is_dir():
            continue
        for t in sorted(d.iterdir()):
            if (t / "task_info.json").is_file():
                if t.name in mine:
                    raise RenderOnlyError(f"{ds}/{t.name}: {h} has two base trees ({mine[t.name]} · {t})")
                mine[t.name] = t
    return mine


def _axis_check(ds: str, ax: str, mine: dict, root_hands: tuple) -> None:
    """The axis hand's base: required unless the caller stated a root without it (--root-hands; said once)."""
    from dexengine.record.scene.render_only import MARK

    if mine:
        return
    if ax in root_hands:
        raise RenderOnlyError(f"{ds}: axis hand {ax} is stated in --root-hands but has no base tree")
    if not root_hands:
        raise RenderOnlyError(f"{ds}: axis hand {ax} has no base tree (a root without it: --root-hands)")
    print(
        f"{MARK} {ds}: root hands {','.join(sorted(root_hands))} (stated) · axis hand {ax} not among them — every "
        f"clip takes the base tree of the first hand by name that has it (base frames do not depend on the hand)",
        flush=True,
    )


def base_trees(ds: str, root_hands: tuple = ()) -> dict:
    """{clip: base tree}: the union of the base layers over hands (module doc 'Membership'). Axis hand first, then the
    other hands by name; the tree of the first hand that has the clip. Dies (never picks silently): the same clip in
    two embs of one hand · no axis-hand base · a clip missing for the axis hand held by two or more hands with
    different windows (task_info segment_range).
    root_hands (--root-hands, the caller's statement of the hands this root holds; empty = not stated): a stated set
    without the axis hand makes every clip 'missing for the axis hand' (said once, MARK line); the statement is checked
    against the layers measured here (a stated hand without a base tree, or a base tree of an unstated hand, dies)."""
    from dexengine.core import paths as P

    ax = axis_hand()
    unknown = sorted(set(root_hands) - set(P.HAND_IDS))
    if unknown:
        raise RenderOnlyError(f"{ds}: --root-hands {unknown} are not hands (paths.HAND_IDS)")
    out: dict = {}
    axis_clips: set = set()
    have = set()
    for h in (ax, *sorted(x for x in P.HAND_IDS if x != ax)):
        mine = hand_trees(ds, h)
        have |= {h} if mine else set()
        if h == ax:
            _axis_check(ds, P.hand_spec_dir(h), mine, root_hands)
            axis_clips = set(mine)
        for c, t in mine.items():
            if c not in out:
                out[c] = t
            elif c not in axis_clips and _window(out[c]) != _window(t):
                raise RenderOnlyError(
                    f"{ds}/{c}: the window of a clip missing for the axis hand differs between hands "
                    f"{out[c]} {_window(out[c])} · {t} {_window(t)}"
                )
    if root_hands and have != set(root_hands):
        raise RenderOnlyError(
            f"{ds}: --root-hands {sorted(root_hands)} but the hands with a base tree here are {sorted(have)}"
        )
    return out


def _window(tree: Path):
    return json.loads((Path(tree) / "task_info.json").read_text(encoding="utf-8")).get("segment_range")


# ── generation ────────────────────────────────────────────────────────────────
def _motion(pos, wxyz, have) -> tuple[float, float]:
    import numpy as np

    p, q = np.asarray(pos, float)[have], np.asarray(wxyz, float)[have]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    rot = np.degrees(2 * np.arccos(np.clip(np.abs((q * q[0]).sum(1)), 0, 1)))
    return float(np.linalg.norm(p - p[0], axis=1).max()), float(rot.max())


def seq_clips(ds: str, clips: dict) -> dict:
    """{seq key: [(clip, base tree)]}: each original sequence is read once."""
    M = oracle_module(ds)
    out: dict = {}
    for c, t in clips.items():
        out.setdefault(M.seq_key(c), []).append((c, t))
    return out


def _anchor_rows(tree, ti, rec):
    """Frame-0 poses of the tree objects that are articulation/object roots (anchors)."""
    import numpy as np

    kin = np.load(Path(tree) / "0" / "trajectory_kinematic.npz")["qpos"]
    roots = RL.anchor_starts(tree, ti, kin.shape[1])  # articulation root poses included
    for o in ti.get("objects") or []:
        if str(o["name"]) in roots:
            s0 = roots[str(o["name"])]
            rec["anchors"].append(
                {
                    "name": str(o["name"]),
                    "obj_id": str(o["obj_id"]),
                    "pose": [float(v) for v in kin[0, s0 : s0 + 7]],
                }
            )
    return kin


def _render_only_rows(rids, ds, rec, seq, sk, ti, kin, M):
    """One row per object outside the tree: frame-0 pose, motion, asset match (or the reason it has none)."""
    if rids:
        if ds not in POSE_SOURCES:
            raise RenderOnlyError(
                f"{len(rids)} objects outside the tree but {ds} has no raw -> base transform owner (POSE_SOURCES)"
            )
        tracks, rec["oracle"] = pose_source(ds)(seq, sk, ti, kin, rids)
        for rid in rids:
            tr = tracks[rid]
            row = {
                "name": NAME_PREFIX + str(rid).lower().replace("@", "_"),
                "rid": str(rid),
                "obj_id": None,
                "reason": "",
                "mesh": tr["mesh"],
                "pose": None,
                "move_m": None,
                "rot_deg": None,
            }
            rec["objects"].append(row)
            if tr["pos"] is None or not bool(tr["have"][0]):
                row["reason"] = "no_pose"
                continue
            row["pose"] = [float(v) for v in (*tr["pos"][0], *tr["wxyz"][0])]
            row["move_m"], row["rot_deg"] = (round(v, 5) for v in _motion(tr["pos"], tr["wxyz"], tr["have"]))
            a = resolve_asset(ds, str(rid), tr["mesh"], M.obj_key)
            row.update(asset_matches=a["matches"], asset_candidates=a["candidates"])
            if a["obj_id"] is None:
                row["reason"] = "no_asset"
            else:
                row["obj_id"] = a["obj_id"]


def _gen_clip(tree, M, raw, seq, clip, stages, ds, out):
    """One clip record of the sequence (window, measured objects outside the tree, anchors); a failing clip becomes an
    error row."""
    from dexengine.core import paths as P
    from dexengine.data.readers.raw import common as SC
    from dexengine.data.readers.raw import scene_classify as SCL

    try:
        ti = json.loads((Path(tree) / "task_info.json").read_text(encoding="utf-8"))
        sk, w0, w1, act, decl, how = M.window(raw, seq, clip, ti)
        if sk not in stages:
            stages[sk] = M.measure(raw, seq, sk)
        st = stages[sk]
        w1 = SC.TP.tree_window_end(st, w0, w1, ti)
        meas = SCL.reduce_window(st, w0, w1, act, decl)
        scene = SCL.tree_scene(ti, M.obj_key)
        rids = sorted(r for r, m in meas.items() if r not in scene and m["n_samples"] > 0)
        rec = {
            "version": VERSION,
            "dataset": ds,
            "clip": clip,
            "axis_hand": axis_hand(),
            "base_tree": str(Path(tree).relative_to(P.main_root())),
            "window": [int(w0), int(w1)],
            "how": how,
            "oracle": None,
            "objects": [],
            "anchors": [],
        }
        kin = _anchor_rows(tree, ti, rec)
        _render_only_rows(rids, ds, rec, seq, sk, ti, kin, M)
        out.append(rec)
    except Exception as e:  # noqa: BLE001 — a failing clip is counted and keeps its first line
        out.append({"clip": clip, "error": f"{type(e).__name__}:{str(e)[:160]}"})


def gen_seq(ds: str, key: str, items: list) -> list[dict]:
    """Clips of one original sequence -> list of clip records (dict). A failing clip goes into that record's error field
    (counted, rc 1)."""

    from dexengine.core import paths as P

    M = oracle_module(ds)
    M._MANO.setdefault("dir", str(P.mano_model_dir()))  # the same MANO place as the audit run_seq (owner paths)
    raw = M.Raw(P.layer_path("raw", ds))
    try:
        seq = M.load_seq(raw, key)
    except Exception as e:  # noqa: BLE001 — one failing sequence counts as errors of its clips
        return [{"clip": c, "error": f"NO_RAW:{type(e).__name__}:{str(e)[:120]}"} for c, _ in items]
    stages, out = {}, []
    for clip, tree in items:
        _gen_clip(tree, M, raw, seq, clip, stages, ds, out)
    return out


def plan_rows(ds: str, recs: list[dict], root: str | None) -> list[tuple]:
    """Write targets (path, dataset, clip, exists, canonical place, exists in canonical): index + clip files with
    objects > 0. In a dry run (--out) the canonical place field is 'what a move would overwrite' (in a canonical run
    both fields are the same)."""

    def row(clip: str, f: Path, g: Path) -> tuple:
        return (str(f), ds, clip, int(f.exists()), str(g), int(g.exists()))

    rows = [row("-", list_dir(ds, root) / INDEX, list_dir(ds) / INDEX)]
    rows.extend(
        row(r["clip"], clip_file(ds, r["clip"], root), clip_file(ds, r["clip"]))
        for r in recs
        if "error" not in r and r["objects"]
    )
    return rows


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def write(ds: str, recs: list[dict], root: str | None) -> int:
    """Clip files -> index (merged into the existing index; the same clip takes the new value). Returns the number of
    clip files written."""
    n = 0
    for r in recs:
        if "error" not in r and r["objects"]:
            _write_json(clip_file(ds, r["clip"], root), r)
            n += 1
    ip = list_dir(ds, root) / INDEX
    idx = (
        json.loads(ip.read_text(encoding="utf-8"))
        if ip.is_file()
        else {"version": VERSION, "dataset": ds, "axis_hand": axis_hand(), "clips": {}}
    )
    if idx.get("version") != VERSION or idx.get("axis_hand") != axis_hand():
        raise RenderOnlyError(
            f"{ip}: version {idx.get('version')} · axis hand {idx.get('axis_hand')} != {VERSION} · "
            f"{axis_hand()}: an old index is not merged into (remove it and rerun everything)"
        )
    for r in recs:
        if "error" not in r:
            idx["clips"][r["clip"]] = len(r["objects"])
    _write_json(ip, idx)
    return n


def summary_line(ds: str, recs: list[dict], rc: int) -> str:
    ok = [r for r in recs if "error" not in r]
    objs = [o for r in ok for o in r["objects"]]
    by = {k: sum(o["reason"] == k for o in objs) for k in REASONS}
    mesh = len({o["rid"] for o in objs})
    return (
        f"{MARK} end {ds} clips {len(recs)} · clips with outside objects {sum(1 for r in ok if r['objects'])} · "
        f"outside objects {len(objs)} · unique {mesh} · no_pose {by['no_pose']} · no_asset {by['no_asset']} · "
        f"errors "
        f"{len(recs) - len(ok)} · rc {rc}"
    )


def generate(ds: str, hands: tuple = (), clips: str = "", jobs: int | None = None) -> list[dict]:
    """Records of every clip of ds (base_trees(ds, hands)), or of the clips subset (comma separated); one gen_seq per
    original sequence. A failing clip is an error record, printed here (counted by run_rc)."""
    trees = base_trees(ds, hands)
    if clips:
        from dexengine.core import paths as P

        want = P.read_clip_list([c for c in clips.split(",") if c], dataset=ds, label="--clips", quiet=True)
        miss = [c for c in want if c not in trees]
        if miss:
            raise RenderOnlyError(f"{ds}: clips without a base tree {miss[:5]} (total {len(miss)})")
        trees = {c: trees[c] for c in want}
    work = [(ds, k, v) for k, v in seq_clips(ds, trees).items()]
    recs: list[dict] = []
    from dexengine.core import capacity

    with capacity.sized("dexengine.record.scene.render_only_build", len(work), jobs) as cap:
        if cap.workers > 1:
            from multiprocessing import Pool

            with Pool(cap.workers, capacity.worker_init, cap.initargs) as p:
                for r in p.starmap(gen_seq, work):
                    recs += r
        else:
            for j in work:
                recs += gen_seq(*j)
    if sorted(r["clip"] for r in recs) != sorted(trees):
        raise RenderOnlyError(f"{ds}: records {len(recs)} != clips {len(trees)}")
    for r in recs:
        if "error" in r:
            print(f"{MARK} error {ds}/{r['clip']}: {r['error']}", flush=True)
    return recs


def run_rc(recs: list[dict]) -> int:
    """1 when a clip failed or an object has no asset (reason no_asset), else 0."""
    n_asset = sum(o["reason"] == "no_asset" for r in recs if "error" not in r for o in r["objects"])
    return 1 if (any("error" in r for r in recs) or n_asset) else 0


# ── reading (consumer scene.extras) ───────────────────────────────────────────────
def main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="generator of the record's render-only scene objects (original objects outside the tree)"
    )
    ap.add_argument("--dataset", required=True, help="dataset (several, comma separated)")
    ap.add_argument(
        "--clips", default="", help="clip subset (comma separated; default = union of the base trees over hands)"
    )
    ap.add_argument(
        "--out", default=None, help="write the same shape under this root (dry run; default = canonical layer)"
    )
    ap.add_argument("--plan-out", default="", help="tsv of write targets (path, dataset, clip, exists)")
    ap.add_argument(
        "--root-hands",
        default="",
        help="hands this root holds (comma separated; a one-hand root without the axis hand states it here — checked "
        "against the base layers; default = not stated, the axis hand is required)",
    )
    ap.add_argument("--plan-only", action="store_true", help="compute and write the plan only (nothing in the layer)")
    ap.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="sequence parallelism (default: capacity.plan — cap 1, the entry shell's one-thread rule)",
    )
    a = ap.parse_args()
    plan, lines, rc_all = [], [], 0
    for ds in [x for x in a.dataset.split(",") if x]:
        recs = generate(ds, tuple(x for x in a.root_hands.split(",") if x), a.clips, a.jobs)
        plan += plan_rows(ds, recs, a.out)
        rc = run_rc(recs)
        if not a.plan_only:
            write(ds, recs, a.out)
        lines.append(summary_line(ds, recs, rc))
        # per dataset right away (earlier lines survive a later death) · once more at the end
        print(lines[-1], flush=True)
        rc_all = max(rc_all, rc)
    if a.plan_out:
        with open(a.plan_out, "w", encoding="utf-8") as fh:
            fh.write("path\tdataset\tclip\texists\tcanon_path\tcanon_exists\n")
            for row in plan:
                fh.write("\t".join(str(x) for x in row) + "\n")
        print(
            f"{MARK} plan {a.plan_out} lines {len(plan)} · exists {sum(r[3] for r in plan)} · exists in canonical "
            f"{sum(r[5] for r in plan)}",
            flush=True,
        )
    for ln in lines:
        print(ln, flush=True)
    sys.stdout.flush()
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
