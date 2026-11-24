#!/usr/bin/env python3
"""**Input fingerprint of the hand-independent contact sidecars: single owner.**

Why. `precompute/<ds>/contact_mano_{all,base}` is a hand-independent layer, but it is built from **one hand's base**
object poses (`contact/sidecar.py --hand`). If inputs went unrecorded and consumers (cases, roles, bakes) compared only
the length T, sidecars built from old poses would silently remain after object poses changed (e.g. object GT gap
interpolation), and a new hand's cases, D23 and bake contact would run on the old poses.

What. The sidecar records an input fingerprint and readers compare it with **the fingerprint recomputed from their own
base**.
  exact (byte-stable fields; must be equal as is)
    code     generation code version (CODE_VERSION) · palm coefficient α for all-objects (value owner = the measured
        file)
    objects  object slot names · obj_id · articulated or not · content hash of the surface samples (meshes layer)
    mano     hash of that side's MANO keypoint source arrays · window (segment.start + frame_map.offset) · T · yaw ·
             (h2o) desk plane record
  pose (object pose witness: (T, object, 8) = position 3 · quaternion wxyz 4 · joint value 1 (NaN without))
    recorded **as an array, not a hash**, and compared with a tolerance. Measured (60 two-hand base samples per
    dataset): even within a generation the two hands' object poses are not byte-identical; dexcanvas · hocap · hrdexdb ·
    oakink2 carry 1e-16 noise, and arctic has **the opposite quaternion sign on every frame** (q ↔ −q, the same
    rotation). A raw hash gives false 'generation differs' on both, and a quantised hash flips rarely at rounding
    boundaries, hence the sign-free · POSE_ATOL comparison. Generation differences (0.02–0.22 m before/after object GT
    gap interpolation) exceed this threshold by more than four orders of magnitude.
  input_id  hash of exact + the raw pose bytes; **bit-identical when recomputed from the same hand's base**, so cases
            record their own inputs with it and roles (D23) compare that short name (not used across hands, for the
                reason above).
  prov      building hand · that base's adapter time and version (not used in verdicts; for reporting which base is
      newest)

Verdict: fresh (equal) · stale (differs, with a reason) · unknown (no fingerprint = version before fingerprints) ·
missing (no file).
Dependencies: numpy · hashlib · json · paths (path accessor), importable from system python3, the Isaac venv and the
retarget venv. (dexengine.data.contact.common pulls dexengine.record.viz.mano_render → torch, which bake and status
tools cannot load, so the version
(CODE_VERSION) is owned here; contact.sidecar_compute.VERSION reads this value.)

Conflict rule for the layer shared by two hands (live_disagreement · disagreement_kind, applied by the producer
contact.sidecar before writing):
  layer generation = version of the producer that built the base (`provenance.adapter_version`) + versions of the base
  steps that change the layer's expected objects and poses (GEN_RULES[layer] in task_info.rules), values the tree
  states itself (generation_of). disagreement_kind alone classifies:
  · same_gen    same generation, different content → **conflict** (rc 4); mismatches between new bases are defects, with
    no overwrite path.
  · gen_change  different generation and this hand is newer → **generation replacement** (default; marker
                [SIDECAR_SUPERSEDE] old → new generation). Consumers of the old-generation hand fail the fingerprint
                    check
                until they rebuild their base.
  · gen_behind  different generation and that hand is newer → conflict (an old generation never overwrites a new one).
  · gen_unknown either generation unknown · that hand's base unreadable · version order undecidable → conflict
                (unclassifiable counts as a defect).
  Which side is newer = **version order** (newer_by_generation), not build time. For each part where the two
  generations differ (producer version · GEN_RULES versions · coordinate version), the side equal to that part's
  **current version** (current_part: the owner constant of the step writing it) is newer. Steps that change base in
  place, such as 2b, do not change the base build time, so timing would read new-version trees as old. If exactly one
  side is current on every differing part, that side is newer; otherwise (both old · mixed · parts without an owner)
      incomparable.
  Premise: one dataset's stage 1 runs hand by hand (CPU exclusive). Two hands building the same dataset at once
  would read a half-built base (after 2, before 2b) as another generation and fail with rc 4; nothing is overwritten
      silently.

Usage (check, no writes):
  python3 dexengine/data/contact/fingerprint.py --scan --dataset hot3d --hand dg5f_s [--layer <layer>|both] [--out
      f.tsv]
  … --scan … --gen-check   compares every (clip, side) with other hands' live bases and counts same-generation
                           mismatches · unclassifiable (rc 1) and generation replacements (differences from
                               other-generation
                           bases, recorded only). Marker [SIDECAR_GEN].
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dexengine.core import paths as P  # noqa: E402  — owner of the path conventions

#: MANO palm proxy coefficient. **The measured file owns the value** ; no number is
#: written here.

_LIVE_CACHE: dict = {}


from dexengine.data.contact.fingerprint_check import (  # noqa: E402,F401  (re-exported)
    POSE_ATOL,
    _diff_detail,
    check,
    compare,
    pose_close,
    read,
    regen_hint,
)
from dexengine.data.contact.fingerprint_core import (  # noqa: E402,F401  (re-exported)
    _REF_CACHE,
    _SURF_CACHE,
    CODE_VERSION,
    FP_KEYS,
    FP_SCHEMA,
    GEN_BEHIND,
    GEN_CHANGE,
    GEN_DEFECT,
    GEN_FRAME_KEY,
    GEN_LABEL,
    GEN_LEVER_KEY,
    GEN_PRESENCE_KEYS,
    GEN_REF_KEY,
    GEN_RULES,
    GEN_STATIC_KEY,
    GEN_UNKNOWN,
    LAYER_ALL,
    LAYER_BASE,
    LAYERS,
    SAME_GEN,
    SidecarFingerprintError,
    TreeInputs,
    _exact_json,
    _lever_part,
    _rule_version,
    _sha_arrays,
    _static_part,
    base_time_of,
    current_part,
    disagreement_kind,
    dl,
    ds_ref,
    gen_parts,
    generation_of,
    input_id,
    layer_of,
    newer_by_generation,
    objects_for,
    palm_alpha,
    payload,
    pose_witness,
    sidecar_is_own,
    sidecar_path,
    sides_for,
    surface_path,
    surface_sha,
)


def live_hands(root: Path, ds: str) -> list[str]:
    """Hands (spec directories) with a **live** base for the dataset: base layer directories counted, only those whose
    base ledger (`INTAKE/base_<spec>.json`) is complete (the same gate producers apply to their own hand; a half-built
    base does not block others' sidecars). The list is a measurement, not a declaration."""
    key = (str(root), ds)
    if key not in _LIVE_CACHE:
        broot = Path(root) / "data" / ds / "base"
        out = []
        for h in sorted(p.name for p in broot.iterdir() if p.is_dir()) if broot.is_dir() else []:
            led = Path(root) / dl.render("intake", ds) / f"base_{h}.json"
            try:
                if json.loads(led.read_text()).get("complete"):
                    out.append(h)
            except (OSError, ValueError):
                continue
        _LIVE_CACHE[key] = out
    return _LIVE_CACHE[key]


_REG_CACHE: dict = {}


def piece_current(root: Path, ds: str, emb: str, clip: str, ti: dict) -> str:
    """Whether a D36 piece (`_tN`) tree is **that piece of the current registry**: "" if so, else the reason. "" when it
    is not piece-shaped or that axis has no registry (nothing to compare). Pieces the hand has not re-sliced yet after
    the registry changed windows or dropped pieces (the job of stage-1 reslice-stale · purge-unregistered) are not live
    bases; like a half-built base (live_hands) they do not block others' shared sidecars. That hand's consumers fail
    the fingerprint check until their pieces are re-sliced."""
    if not re.fullmatch(r".+_t\d+", clip):
        return ""
    key = (str(root), ds, emb)
    if key not in _REG_CACHE:
        f = P.units_registry(ds, emb, root)
        reg = json.loads(f.read_text()) if f.is_file() else None
        _REG_CACHE[key] = (
            None
            if reg is None
            else {u["name"]: [u["t0"], u["t1"]] for us in (reg.get("parents") or {}).values() for u in us}
        )
    win = _REG_CACHE[key]
    if win is None:
        return ""
    if clip not in win:
        return "piece not in the current registry"
    got = ti.get("segment_range")
    return "" if got == win[clip] else f"window {got} ≠ current registry {win[clip]}"


def other_live_bases(
    root: Path, ds: str, emb: str, clip: str, spec_dir: str, skipped: list | None = None
) -> list[tuple[str, Path]]:
    """Live base trees of **other hands** with the same (dataset, emb, clip). Pieces differing from the current registry
    are left out and recorded in `skipped` as (hand, reason) (piece_current)."""
    out = []
    for h in live_hands(root, ds):
        if h == spec_dir:
            continue
        if clip in dl.sidecar_own_clips(
            root, ds, h
        ):  # that hand reads this clip from a per-hand location and does not expect the shared file
            continue
        t = Path(root) / dl.render("base", ds, hand=h, emb=emb, clip=clip)
        if (t / "task_info.json").is_file() and (t / "0/trajectory_kinematic.npz").is_file():
            why = piece_current(root, ds, emb, clip, json.loads((t / "task_info.json").read_text()))
            if why:
                (skipped if skipped is not None else []).append({"other": h, "why": why})
                continue
            out.append((h, t))
    return out


def live_disagreement(
    root: Path,
    ds: str,
    emb: str,
    clip: str,
    layer: str,
    side: str,
    mine: dict,
    spec_dir: str,
    skipped: list | None = None,
) -> list[dict]:
    """Whether the content built from this hand's base differs from what **another hand's live base** expects (a
    conflict on the shared layer).

    No conflict when that hand does not expect this (clip, side) file (no objects to hold on that side).
    newer = which side is newer (version order newer_by_generation, not time): mine · other · same · unknown.
    kind  = disagreement_kind(this hand's generation, that hand's generation, newer): same_gen · gen_change · gen_behind
    · gen_unknown.
    """
    out = []
    mine_gen = mine["prov"]["gen"]
    for h, t in other_live_bases(root, ds, emb, clip, spec_dir, skipped):
        try:
            other = TreeInputs.load(root, ds, t, h)
            if side not in sides_for(other.ti, layer):
                continue
            exp_o = other.expected(layer, side, own=False)
        except Exception as e:  # noqa: BLE001 — an incomparable case is recorded as a conflict (never passed silently)
            out.append(
                {
                    "other": h,
                    "other_base_time": "",
                    "newer": "unknown",
                    "other_gen": "",
                    "mine_gen": mine_gen,
                    "kind": GEN_UNKNOWN,
                    "why": f"cannot read the other hand's base: {type(e).__name__}: {e}"[:200],
                }
            )
            continue
        v, why = compare(exp_o, mine)
        if v == "fresh":
            continue
        og = exp_o["prov"]["gen"]
        newer, order = newer_by_generation(mine_gen, og, spec_dir, h, ds_ref(root, ds))
        out.append(
            {
                "other": h,
                "other_base_time": exp_o["prov"]["base_time"],
                "why": why,
                "newer": newer,
                "order": order,
                "other_gen": og,
                "mine_gen": mine_gen,
                "kind": disagreement_kind(mine_gen, og, newer),
            }
        )
    return out


# Check (no writes).
def _scan_tree(job: dict) -> list[dict]:
    root, ds, tree, spec, layers = Path(job["root"]), job["ds"], Path(job["tree"]), job["spec"], job["layers"]
    head = {"emb": tree.parent.name, "clip": tree.name}
    try:
        tin = TreeInputs.load(root, ds, tree, spec)
    except Exception as e:  # noqa: BLE001
        return [{**head, "layer": "-", "side": "-", "verdict": "error", "why": f"{type(e).__name__}: {e}"[:200]}]
    rows = []
    for layer in layers:
        try:
            sides = sides_for(tin.ti, layer)
        except ValueError as e:
            rows.append({**head, "layer": layer, "side": "-", "verdict": "error", "why": str(e)[:200]})
            continue
        for side in sides:
            f = sidecar_path(root, ds, layer, tree.name, side, spec)
            row = {
                **head,
                "layer": layer,
                "side": side,
                "base_time": base_time_of(tin.ti),
                "built_from": "",
                "sidecar_base_time": "",
            }
            try:
                exp = tin.expected(layer, side)
                if not f.is_file():
                    row.update(verdict="missing", why="")
                else:
                    st = read(f)
                    v, why = compare(st, exp)
                    row.update(verdict=v, why=why)
                    if st:
                        row.update(built_from=st["prov"]["built_from"], sidecar_base_time=st["prov"]["base_time"])
                if job.get("gen_check") or (job.get("conflicts") and row["verdict"] != "fresh"):
                    # would rebuilding clash with another hand's live base (the same function the producer applies
                    # before writing).
                    # gen_check measures every row including fresh ones; the question is whether same-generation hands
                    # expect the same content.
                    dis = live_disagreement(root, ds, tree.parent.name, tree.name, layer, side, exp, spec)
                    row.update(
                        conflict="; ".join(f"{d['other']}:{d['newer']}:{d['why'][:120]}" for d in dis),
                        gen=[d["kind"] for d in dis],
                    )
            except Exception as e:  # noqa: BLE001
                row.update(verdict="error", why=f"{type(e).__name__}: {e}"[:200])
            rows.append(row)
    return rows


def scan(
    root: Path,
    ds: str,
    spec_dir: str,
    layers=LAYERS,
    workers: int | None = None,
    clips: set | None = None,
    conflicts: bool = False,
    gen_check: bool = False,
) -> list[dict]:
    """Verdict per layer and side against that hand's base. Returns rows (verdict ∈ fresh·stale·unknown·missing·error).
    With conflicts=True, every non-fresh row is also checked for 'would rebuilding clash with another hand's live base'
    (no writes)."""
    jobs = []
    for emb in dl.EMBODIMENTS:
        d = Path(root) / dl.render("base", ds, hand=spec_dir, emb=emb)
        if not d.is_dir():
            continue
        for tp in sorted(p for p in d.iterdir() if (p / "task_info.json").is_file()):
            if clips is not None and tp.name not in clips:
                continue
            jobs.append(
                {
                    "root": str(root),
                    "ds": ds,
                    "tree": str(tp),
                    "spec": spec_dir,
                    "layers": tuple(layers),
                    "conflicts": conflicts,
                    "gen_check": gen_check,
                }
            )
    from multiprocessing import Pool  # noqa: PLC0415

    from dexengine.core import capacity  # noqa: PLC0415

    rows: list[dict] = []
    if len(jobs) < 64:  # a small scan (the stage status asks per layer): a pool costs more than it saves
        for j in jobs:
            rows += _scan_tree(j)
        return rows
    step = "dexengine.data.contact.fingerprint"  # capacity.NEEDS key (this module)
    with capacity.sized(step, len(jobs), workers) as cap, Pool(cap.workers, capacity.worker_init, cap.initargs) as pool:
        for r in pool.imap(_scan_tree, jobs, chunksize=16):  # job order (not finish order)
            rows += r
    return rows


def summarize(rows: list[dict], layer: str) -> dict:
    """Verdict counts of one layer. total = expected (clip, side) count (tree-level errors count as 1)."""
    c = {"fresh": 0, "stale": 0, "unknown": 0, "missing": 0, "error": 0}
    for r in rows:
        if r["layer"] in (layer, "-"):
            c[r["verdict"]] = c.get(r["verdict"], 0) + 1
    c["total"] = sum(c.values())
    return c


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scan", action="store_true", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hand", required=True, help="whose base to compare against (no default)")
    ap.add_argument(
        "--layer",
        choices=(*LAYERS, "both"),
        default="both",  # values = layer names
        help="layer to measure (default both)",
    )
    ap.add_argument("--root", default="", help="data root (default paths.main_root)")
    ap.add_argument("--workers", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--out", default="", help="per-row TSV")
    ap.add_argument(
        "--conflicts", action="store_true", help="also measure two-hand conflicts (on regeneration) for non-fresh rows"
    )
    ap.add_argument(
        "--gen-check",
        action="store_true",
        help="compare every row with other hands' live bases; rc 1 on same-generation mismatches or unclassifiable "
        "rows (generation replacements are only recorded). The rc of this mode ignores fingerprint freshness",
    )
    a = ap.parse_args()
    spec = P.hand_spec_dir(a.hand)
    root = Path(a.root) if a.root else P.main_root()
    layers = LAYERS if a.layer == "both" else (a.layer,)
    rows = scan(root, a.dataset, spec, layers, a.workers, conflicts=a.conflicts, gen_check=a.gen_check)
    if a.out:
        cols = (
            "emb",
            "clip",
            "layer",
            "side",
            "verdict",
            "why",
            "base_time",
            "built_from",
            "sidecar_base_time",
            "conflict",
            "gen",
        )
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in sorted(rows, key=lambda r: (r["layer"], r["emb"], r["clip"], r["side"])):
                fh.write("\t".join(str(r.get(k, "")) for k in cols) + "\n")
    rc = 0
    for layer in layers:
        s = summarize(rows, layer)
        counts = " ".join(f"{k}={v}" for k, v in s.items())
        if a.conflicts:
            counts += f" conflict={sum(1 for r in rows if r['layer'] == layer and r.get('conflict'))}"
        print(f"[SIDECAR_FP] ds={a.dataset} hand={spec} layer={layer} {counts}", flush=True)
        if not a.gen_check:
            rc = rc or int(s["fresh"] != s["total"])
            continue
        g = {
            k: sum(1 for r in rows if r["layer"] == layer and k in r.get("gen", ()))
            for k in (SAME_GEN, GEN_UNKNOWN, GEN_CHANGE, GEN_BEHIND)
        }
        print(
            f"[SIDECAR_GEN] ds={a.dataset} hand={spec} layer={layer} same_gen_mismatch={g[SAME_GEN]} "
            f"unclassified={g[GEN_UNKNOWN]} error={s['error']} · generation differences (recorded) this hand "
            f"newer {g[GEN_CHANGE]} · "
            f"other hand newer {g[GEN_BEHIND]}",
            flush=True,
        )
        rc = rc or int(any(g[k] > 0 for k in GEN_DEFECT) or s["error"] > 0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
