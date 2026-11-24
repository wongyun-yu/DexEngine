"""Placement augment: reads and checks the judged cell bundle (judge mode D region dump, mode L list judgement;
called by augment.place_gen). Judged-cell contract
(v2): bundle folder = region/, cells/, judged/, ikc/ of paths.run_path("run_aug_cells", …).
External API: fov_cells_done, read by name (getattr) by a deployment's dashboard.

Checks (a mismatch dies):
  · exactly one done line per chunk jsonl, the end line (policy.parse_done, contract v2.1), whose numbers = (clips,
    cells, passes) of that chunk's cell rows (done excluded) · one chunk = one run (one judge_commit; judge_fp is per
    clip, checked by load below)
  · a clip's row set (union of chunks) {(k, cell)} = {(order, cell)} of the cells json · no k overlap · ok ⇔ stage
    pass
  · judge identity (policy.JUDGE_ID_KEYS): region dump = cells json = every row
  · a refusal holds that clip only (Refused.kind; place_gen records it; chain.after_store) · the bundle's judge
    commits (bundle_commits) = the bundle part of the augment cache key
  · θ and d of a row = the same cell's values in the region dump · canon = the baked tree placement
    (coords.tree_placement) · canonical dz x step = robot_z_off of the placement file (variants do not change the robot
    height) · first row of the cells json = the region's canonical cell · seed = clip_seed · rule = CELL_RULE · when the
    bundle has fov/fov_<clip>.json (fov_path), CELL_RULE_FOV ∧ list fov.source_sha256 = that material
Arm trajectory: the judge q of passing cells (ikc/) is only a **seed of the bake step** (stage_dp --reuse-npz); the
  variant arm cache and time axis are built by the same path as the bake (augment.bake: stage_dp -> cross repair ->
  retiming). The seed length is owned by the stage_dp reuse rule.
The gate speed rule of variants lives in augment.speed_gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from dexengine.core.errors import input_missing
from dexengine.record.augment import policy as AP

MARK = "[AUG_JUDGED]"
FOV_DIR = "fov"  # off-screen clip material in the bundle (cells_pick.put_fov writes it; choice and load read it)
#: extra-cell sub-bundle (cells_ext: only clips short of judge-passed cells get the next cells of the same rank
#: judged): a bundle of the same form <bundle>/ext (region · fov copy (regular files, lane package) · cells =
#: [canonical cell] + extra cells · judged · ikc · RUN_DONE.json{clips, offset · exhausted}). Stage n is the ext
#: inside stage n-1 (<bundle>/ext/ext …); offset is the number of non-canonical cells of the list one stage up.
#: load merges the original rows + extra rows (k + offset; the extra canonical row k = 0 is dropped) per stage
#: recursively; (k) or (cell) overlaps die. exhausted = clips that used up the region's remaining cells at that stage
#: (the choice returns short; no deeper stage).
EXT = "ext"
#: same-bundle off-screen extra cells (cells_ext --fov, relaxed ladder included): a one-stage sub-bundle <bundle>/fovx.
#: Same form as ext: fov/ = copy of that clip's off-screen material · cells = [canonical cell] + cells of the
#: off-screen rank (CELL_RULE_FOV) that do not overlap the original or ext chain cells · RUN_DONE.json{clips, offset,
#: exhausted, native}. offset = number of non-canonical cells of the original + ext chain list; load merges it last,
#: after those two (bundle top only). fovx clips drop out of later new ext stages (k overlap; cells_ext). Home of the
#: end decision = fov_cells_done.
FOVX = "fovx"
STAGES = ("volume", "gate", "full", "mesh", "desk", "guard", "pass")
ROW_KEYS = ("clip", "k", "cell", "theta_abs_deg", "d_abs", "ok", "stage", "reason", "judge_commit") + AP.JUDGE_ID_KEYS


class Refused(SystemExit):
    """Judged cell bundle refusal: kind = policy.HOLD_* (a reason that holds that clip only)."""

    def __init__(self, msg: str, kind: str):
        super().__init__(msg)
        self.kind = kind


def die(msg: str, kind: str = AP.HOLD_INPUT, missing: bool = False):
    # the message is data: place_gen stores str(e) in refused_<clip>.json, which chain.hold_why copies into counts.json
    # held[].why and skipped (old Korean texts -> the site migration table). missing = an input is absent: the text
    # carries core.errors.INPUT_MISSING, which labels an escaped refusal rc 12 (record.driver_spec.bundle_rc)
    text = f"{MARK} {msg}"
    raise Refused(input_missing(text) if missing else text, kind)


def refused_path(gen_dir: Path, clip: str) -> Path:
    """Judge input refusal record of this clip (written when generation dies in load; holds the clip only; read by
    augment.stage hold_why)."""
    return Path(gen_dir) / f"refused_{clip}.json"


def _obj(v):
    """npz field -> Python value (0-d array, JSON text, dict)."""
    v = v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v
    if isinstance(v, (bytes, str)):
        try:
            return json.loads(v)
        except ValueError:
            return str(v)
    return v


def fov_path(bundle: Path, clip: str) -> Path:
    return Path(bundle) / FOV_DIR / f"fov_{clip}.json"


def ext_bundle(bundle: Path) -> Path:
    return Path(bundle) / EXT


def fovx_bundle(bundle: Path) -> Path:
    return Path(bundle) / FOVX


def fovx_done(bundle: Path) -> dict | None:
    rd = fovx_bundle(bundle) / "RUN_DONE.json"
    return json.loads(rd.read_text()) if rd.is_file() else None


def fovx_offset(bundle: Path, clip: str) -> int | None:
    """k offset of the clip's off-screen extra cells (non-canonical cells of the original + ext chain list); None when
    fovx lacks the clip."""
    d = fovx_done(bundle)
    off = None if d is None else d["offset"].get(clip)
    return None if off is None else int(off)


def held_of(bundle: Path) -> dict:
    """Generation holds {clip: reason}: RUN_DONE held (gen hold). One reader. No held field = a bundle that ended
    before that field existed (a mismatched clip killed the bundle) -> 0 holds."""
    rd = Path(bundle) / "RUN_DONE.json"
    if not rd.is_file():
        die(f"{rd} missing — the generation hold of an unfinished bundle is unknown", missing=True)
    return dict(json.loads(rd.read_text()).get("held") or {})


def fov_cells_done(bundle: Path, clip: str) -> bool:  # noqa: V103 — public API
    """Has the same-bundle extra-cell judgement of an off-screen clip ended (augment hand-off gate; the only call of
    augment targets). True = the original bundle already uses the off-screen rule (fov/ material) · clips of the fovx
    RUN_DONE (judged) · exhausted (0 new cells) · native."""
    if fov_path(bundle, clip).is_file():
        return True
    d = fovx_done(bundle)
    return d is not None and clip in {*d["clips"], *d.get("exhausted", []), *d.get("native", [])}


def ext_offset(bundle: Path, clip: str) -> int | None:
    """k offset of the clip's extra cells (non-canonical cells of the original list); None when the extra-cell bundle
    lacks the clip."""
    rd = ext_bundle(bundle) / "RUN_DONE.json"
    if not rd.is_file():
        return None
    off = json.loads(rd.read_text())["offset"].get(clip)
    return None if off is None else int(off)


def ext_k(bundle: Path, clip: str, k: int) -> tuple[Path, int]:
    """Merged k -> (the bundle that gave the row (stage, fovx), that bundle's k): finds the bake seed (ikc) place.
    Recursive per stage; fovx last."""
    foff = fovx_offset(bundle, clip)
    if foff is not None and int(k) > foff:
        return fovx_bundle(bundle), int(k) - foff
    off = ext_offset(bundle, clip)
    if off is not None and int(k) > off:
        return ext_k(ext_bundle(bundle), clip, int(k) - off)
    return Path(bundle), int(k)


def ext_levels(bundle: Path, clip: str) -> list[Path]:  # noqa: V103 — used by a deployment's cell jobs
    """Extra-cell stage bundles holding the clip (shallowest first; only those with RUN_DONE whose clips include the
    clip; stops at a gap)."""
    out, b = [], Path(bundle)
    while (rd := ext_bundle(b) / "RUN_DONE.json").is_file() and clip in json.loads(rd.read_text())["clips"]:
        b = ext_bundle(b)
        out.append(b)
    return out


def exhausted(bundle: Path, clip: str) -> bool:
    """At which stage the clip used up its judge region cells (RUN_DONE exhausted; clips that ended with 0 cells are
    outside clips)."""
    b = Path(bundle)
    while (rd := ext_bundle(b) / "RUN_DONE.json").is_file():
        d = json.loads(rd.read_text())
        if clip in d.get("exhausted", []):
            return True
        if clip not in d["clips"]:
            return False
        b = ext_bundle(b)
    return False


JUDGE_CAP, REGION_EXHAUSTED = "judge cap", "region exhausted"  # home of the judge end reasons (data: meta judge_end)


def judge_end(bundle: Path, clip: str) -> str:
    return REGION_EXHAUSTED if exhausted(bundle, clip) else JUDGE_CAP


_EMPTY_SAID: set = set()


def empty_stage(done: dict, where: Path) -> bool:
    """Empty stage: with RUN_DONE clips [] (all targets exhausted the region, or native so 0 cells built) no judge
    chunks is normal (not read; marker [EMPTY_STAGE] once per place). With clips it is not an empty stage (0 chunks
    makes read_chunks die). add is not looked at: an ext stage's add is the per-clip cell budget (64), not the count
    built. The one meaning of empty stage (the same stage fov_cells_done takes as 'ended')."""
    if done["clips"]:
        return False
    if str(where) not in _EMPTY_SAID:
        _EMPTY_SAID.add(str(where))
        print(
            f"{MARK} [EMPTY_STAGE] {where} — clips 0 · exhausted {len(done.get('exhausted', []))} · "
            f"native {len(done.get('native', []))} · chunks not read",
            file=sys.stderr,
            flush=True,
        )
    return True


def pass_counts(bundle: Path, ext: bool = True, done: set | None = None) -> dict:
    """Clip -> number of judge-passed non-canonical cells (original chunks + with ext the extra rows of the extra-cell
    stages, recursively)."""
    out: dict = {}
    for r in read_chunks(Path(bundle) / "judged", done):
        if int(r["k"]) != 0 and r["ok"]:
            out[r["clip"]] = out.get(r["clip"], 0) + 1
    rd = ext_bundle(bundle) / "RUN_DONE.json"
    if ext and rd.is_file() and not empty_stage(d := json.loads(rd.read_text()), ext_bundle(bundle)):
        clips = set(d["clips"])
        for c, n in pass_counts(ext_bundle(bundle), ext=True).items():
            if c in clips:
                out[c] = out.get(c, 0) + n
    fd = fovx_done(bundle) if ext else None
    # same-bundle off-screen extra cells (RUN_DONE clips)
    if fd is not None and not empty_stage(fd, fovx_bundle(bundle)):
        for c, n in pass_counts(fovx_bundle(bundle), ext=False, done=set(fd["clips"])).items():
            if c in set(fd["clips"]):
                out[c] = out.get(c, 0) + n
    return out


def merge_ext(main: dict, ext: dict, off: int, n_main_cells: int, clip: str) -> list[dict]:
    """Original rows + extra rows (k + off): the extra canonical row (k = 0) must equal the original canonical cell and
    is dropped · off = number of original non-canonical cells · (k) or (cell) overlaps die (a cell never becomes a
    candidate twice)."""
    if off != n_main_cells - 1:
        die(f"{clip}: added-cell offset {off} ≠ non-canonical cell count of the original list {n_main_cells - 1}")
    m0 = [r for r in main["rows"] if int(r["k"]) == 0]
    e0 = [r for r in ext["rows"] if int(r["k"]) == 0]
    if len(m0) != 1 or len(e0) != 1 or _cell(m0[0]["cell"]) != _cell(e0[0]["cell"]):
        die(f"{clip}: the canonical cell of the added-cell bundle differs from the original canonical cell")
    add = [dict(r, k=int(r["k"]) + off, ext_k=int(r["k"])) for r in ext["rows"] if int(r["k"]) != 0]
    rows = main["rows"] + add
    ks = [int(r["k"]) for r in rows]
    cells = [_cell(r["cell"]) for r in rows]
    if len(set(ks)) != len(ks) or len(set(cells)) != len(cells):
        die(
            f"{clip}: original + added cells overlap — k overlap {len(ks) - len(set(ks))} · "
            f"cell overlap {len(cells) - len(set(cells))}"
        )
    for key in ("canon", "judge_fp", "judge_value_fp", "seed", "n_region"):
        if main[key] != ext[key]:
            die(f"{clip}: added-cell bundle {key} ≠ original")
    return sorted(rows, key=lambda r: int(r["k"]))


def dense_path(bundle: Path, clip: str) -> Path:
    """Dense placement mark (policy.CELL_RULE_DENSE): when present, the clip's list follows the dense rule."""
    return Path(bundle) / "dense" / f"dense_{clip}.json"


def rule_of(bundle: Path, clip: str) -> tuple[str, str | None]:
    """The clip's cell choice rule and material sha (decided by whether the bundle has a fov file; choice and load read
    it alike). With a dense mark (dense_path), the dense rule only when AUG_DENSE_CELLS=1, else it dies."""
    dp = dense_path(bundle, clip)
    if dp.is_file():
        import os

        if os.environ.get(AP.DENSE_ENV) != "1":
            die(
                f"{clip}: dense placement mark {dp} is present but {AP.DENSE_ENV}=1 "
                "is not set — not a canonical collect bundle"
            )
        print(f"[AUG_DENSE] {clip} cell rule {AP.CELL_RULE_DENSE} · mark {dp}", flush=True)
        return AP.CELL_RULE_DENSE, None
    f = fov_path(bundle, clip)
    return (AP.CELL_RULE_FOV, json.loads(f.read_text())["source_sha256"]) if f.is_file() else (AP.CELL_RULE, None)


def read_chunks(judged: Path, done: set | None = None) -> list[dict]:
    """Every judged/augcells_*__*.jsonl: end line DONE per chunk, number check and row field check -> union of rows."""
    files = sorted(Path(judged).glob("augcells_*__*.jsonl"))
    if not files:
        die(f"judge chunks 0 — {judged}")
    files = split(files, done, judged)
    rows = []
    for f in files:
        recs = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        at = [i for i, r in enumerate(recs) if isinstance(r, dict) and "done" in r]
        if at != [len(recs) - 1]:
            die(f"{f.name}: {len(at)} done rows · at {at[:3]} — must be one final row (chunk unfinished · overlap)")
        c, n_cells, n_pass = AP.parse_done(recs[-1])
        rs = recs[:-1]
        for r in rs:
            miss = [k for k in ROW_KEYS if k not in r]
            if miss:
                die(f"{f.name}: {miss} missing from the row", missing=True)
            if r["stage"] not in STAGES or bool(r["ok"]) != (r["stage"] == "pass"):
                die(f"{f.name}: {r['clip']} k={r['k']} ok={r['ok']} stage={r['stage']!r} — ok must equal pass")
        if len(cm := {r["judge_commit"] for r in rs}) > 1:
            die(f"{f.name}: {len(cm)} commits in one chunk — chunk = one run")
        got = (len({r["clip"] for r in rs}), len(rs), sum(bool(r["ok"]) for r in rs))
        if got != (c, n_cells, n_pass):
            die(f"{f.name}: DONE (clips {c} · cells {n_cells} · passed {n_pass}) ≠ rows {got}")
        rows += rs
    return rows


def _cell(c) -> tuple:
    return int(c[0]), int(c[1]), int(c[2]), float(c[3])


def bundle_commits(bundle: Path) -> str:
    """The bundle part of the augment cache key: the judge commits of its region dumps (sorted, comma-joined)."""
    commits = set()
    for f in sorted((Path(bundle) / "region").glob("region_*.npz")):
        z = np.load(f, allow_pickle=True)
        if "judge_commit" not in z.files:
            die(f"{f}: judge_commit missing", missing=True)
        commits.add(str(_obj(z["judge_commit"])))
    return ",".join(sorted(commits))


def _check_pivot_z(R, cn, clip, src):
    """Region pivot = canonical pivot; region robot z offset = the placement file's (grid tolerance)."""
    if max(abs(R["pivot"][0] - cn["pivot"][0]), abs(R["pivot"][1] - cn["pivot"][1])) > 1e-9:
        die(f"{clip}: region pivot {R['pivot']} ≠ canon pivot {cn['pivot']}")
    tol = AP.CANON_MATCH_FRAC * float(R["grid"]["step"])
    if abs(R["robot_z_off"] - float(src.adopted["robot_z_off"])) > tol:
        die(
            f"{clip}: region robot_z_off {R['robot_z_off']} ≠ placement "
            f"file {src.adopted['robot_z_off']} (canonical dz layer)"
        )


def _check_row_values(cells, R, rows, clip):
    """Every judge row's θ · d equals the same cell of the region dump (float64 ==)."""
    by = {
        c: (float(t), (float(d[0]), float(d[1]))) for c, t, d in zip(cells, R["theta_abs_deg"], R["d_abs"], strict=True)
    }
    for r in rows:
        t, d = by[_cell(r["cell"])]
        if (float(r["theta_abs_deg"]), float(r["d_abs"][0]), float(r["d_abs"][1])) != (t, d[0], d[1]):
            die(
                f"{clip} k={r['k']}: row θ · d differs from the same cell of "
                "the region dump (float64 == — contract v2.1 · no rounding)"
            )


def load(bundle: Path, clip: str, dataset: str, src, done: set | None = None) -> dict:
    """Judge result of one clip; returned only after every check (module doc) passes."""
    from dexengine.placement.coords import core as coords
    from dexengine.record.augment.judged_region import read_region

    bundle = Path(bundle)
    R = read_region(bundle / "region" / f"region_{clip}.npz")
    cj_p = bundle / "cells" / f"cells_{clip}.json"
    if not cj_p.is_file():
        die(f"cell list missing {cj_p}", missing=True)
    cj = json.loads(cj_p.read_text())
    rows = [r for r in read_chunks(bundle / "judged", done) if r["clip"] == clip]
    ks = [int(r["k"]) for r in rows]
    if len(set(ks)) != len(ks):
        die(f"{clip}: judge rows k overlap {len(ks) - len(set(ks))}")
    want = {(i, _cell(c)) for i, c in enumerate(cj["cells"])}
    got = {(int(r["k"]), _cell(r["cell"])) for r in rows}
    if got != want:
        die(f"{clip}: judge rows {len(got)} ≠ list {len(want)} — absent {len(want - got)} · extra {len(got - want)}")
    for key in AP.JUDGE_ID_KEYS:
        vals = {R[key]} | {str(r[key]) for r in rows} | ({str(cj[key])} if key in cj else set())
        if len(vals) != 1:
            die(f"{clip}: {key} disagrees {sorted(vals)[:3]} (region dump · list · judge rows)")
    if "judge_fp" not in cj:
        die(f"{clip}: judge_fp missing from the cell list", missing=True)
    seed = AP.clip_seed(dataset, clip)
    rule, sha = rule_of(bundle, clip)
    if cj.get("seed") != seed or cj.get("rule") != rule or (cj.get("fov") or {}).get("source_sha256") != sha:
        die(f"{clip}: list seed {cj.get('seed')} · rule {cj.get('rule')} ≠ clip_seed {seed} · {rule} (inputs {sha})")
    cells = [(int(c[0]), int(c[1]), int(c[2]), float(z)) for c, z in zip(R["cells"], R["dz"], strict=True)]
    i0 = AP.canon_index(list(R["theta_abs_deg"]), [tuple(d) for d in R["d_abs"]], list(R["dz"]), R["canon"], R["grid"])
    if _cell(cj["cells"][0]) != cells[i0]:
        die(f"{clip}: first list row {cj['cells'][0]} ≠ the region's canonical cell {cells[i0]}")
    tp = coords.tree_placement(src.ti)
    cn = R["canon"]
    if (
        tp is None
        or max(
            abs(tp[0] - cn["dx"]),
            abs(tp[1] - cn["dy"]),
            abs(tp[2] - cn["yaw_deg"]),
            abs(tp[3][0] - cn["pivot"][0]),
            abs(tp[3][1] - cn["pivot"][1]),
        )
        > 1e-9
    ):
        die(
            f"{clip}: region canon {cn} ≠ baked tree placement {tp} — "
            "the judge and the tree are of different generations"
        )
    _check_pivot_z(R, cn, clip, src)
    _check_row_values(cells, R, rows, clip)
    out = {
        "rows": sorted(rows, key=lambda r: int(r["k"])),
        "canon": cn,
        "scale": AP.scale_of(R["grid"]),
        "seed": seed,
        "n_region": len(cells),
        "judge_fp": R["judge_fp"],
    }
    off = ext_offset(bundle, clip)
    # extra cells (securing of the acceptance ladder): merge the sub-bundle that passed the same checks
    if off is not None:
        ext = load(ext_bundle(bundle), clip, dataset, src)
        out = dict(out, rows=merge_ext(out, ext, off, len(cj["cells"]), clip), n_ext=len(ext["rows"]) - 1)
    foff = fovx_offset(bundle, clip)
    # same-bundle off-screen extra cells: last, after merging the original + ext chain (offset = non-canonical cells of
    #   that sum)
    if foff is not None:
        fx = load(fovx_bundle(bundle), clip, dataset, src, set(fovx_done(bundle)["clips"]))
        out = dict(out, rows=merge_ext(out, fx, foff, len(out["rows"]), clip), n_fovx=len(fx["rows"]) - 1)
    return out


_SAID: set = set()


def chunk_clips(f: Path) -> tuple[set, bool]:
    """(clips, has a done line) of a chunk; read even when the end line is cut mid-write (cut rows are skipped)."""
    cs, fin = set(), False
    for ln in Path(f).read_text().splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("clip") is not None:
            cs.add(str(r["clip"]))
        fin = fin or (isinstance(r, dict) and "done" in r)
    return cs, fin


def split(files: list[Path], done_clips: set | None, where="") -> list[Path]:
    """-> chunks to read by the completion rule (mixed chunk = clips inside and outside RUN_DONE in one chunk dies).
    done_clips None = all (old rule). Pending = chunks holding only clips outside RUN_DONE (being written; even with
    done written, RUN_DONE not yet): outside the count, marker [PENDING_CHUNK] (once per place)."""
    if done_clips is None:
        return list(files)
    keep, pend, mixed = [], [], []
    for f in files:
        cs, _fin = chunk_clips(f)
        if cs & done_clips and cs - done_clips:
            mixed.append(f.name)
        elif cs & done_clips:
            keep.append(f)
        else:
            pend.append(f.name)
    if pend and str(where) not in _SAID:
        _SAID.add(str(where))
        print(
            f"[AUG_JUDGED] [PENDING_CHUNK] {where} judge pending {len(pend)} · e.g. {pend[0]}",
            file=sys.stderr,
            flush=True,
        )
    if mixed:
        raise SystemExit(f"[AUG_JUDGED] ✗ {mixed[0]}: clips inside and outside RUN_DONE mixed in one chunk — {where}")
    return keep
