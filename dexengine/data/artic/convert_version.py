"""Version of the D1-g conversion (oakink2.artic_convert); a dependency-free module.

Used by oakink2.artic_convert (tree record rules.D1g.version) and data.stages (the stage-1 2b column counts markers of
this version). The converter pulls in numpy and mujoco, which the stage table cannot import, so only the version lives
here. Trees of another version are not rewritten by the converter and fail (merged trees no longer hold the source part
poses); rerun from 2b after regenerating base (2).
"""

import functools
import os
from pathlib import Path

#: Canonicalisation: the committed asset spec's statement as is · joint values = dexengine.data.artic.ranges.reexpress
#: (window = spec lo·hi) ·
#: pairing = oakink2.artic_fit.part_clusters (kind = per-object joint spec oakink2.part_kinds; motion only checks it and
#: a
#: mismatch fails · rigid joints = merged mesh assets oakink2.fixed_merge, assignment and oracle share one statistic) ·
#: an attached part moving outside its band is excluded with that kind's reason word (oakink2.part_kinds.ATTACH_MOVED) ·
#: lever band = tip displacement · share pushed by the jointed part (oakink2.part_kinds.LEVER_TIP_M ·
#: LEVER_DRIVE_SHARE);
#: inside it the lever is a fixed attachment at its rest pose, and the FK oracle measures levers by tip displacement too
#: ·
#: scenes with two or more merged instances: at most one moved instance becomes articulated, the rest are rigid joints
#: at
#: their recorded poses (oakink2.multi_instance) · the joint of a tree whose jointed part is frozen = the published
#: joint
#: of the part pair, independent of prefixes (oakink2.spec_lookup) · supplement parts are accepted only when sampled on
#: the body's time grid with the same interpolation (raw_sample), otherwise a raw_time error · slides (drawers) are
#: checked
#: directly with the slide model (fit_slide), band = rail play (oakink2.part_kinds.SLIDE_PLAY_*; kind check, canonical
#: comparison and FK oracle read the one oakink2.artic_fit.band) · joint values are clamped to the spec asset's physical
#: range (dexengine.data.artic.limits: mesh penetration limits ± scan error band, decidable sides only) and written to
#: rules.D1g.range_clamp (after the FK oracle, joint-value column only).
D1G_VERSION = "D1g-kinds-6"
#: Bootstrap (--per-tree): each tree's own first frame = joint 0. The 2b column does not count it (compared including
#: quotes)
D1G_PER_TREE = D1G_VERSION + "+per_tree"

#: Finished conversion states (merged now · dry run · already at this version)
DONE = ("converted", "would_convert", "skip_done")
#: Generator name written to the exclusion ledger (data.rules.retarget_excluded_record)
EXCLUDE_SRC = "D1g"
#: Description head of inherited exclusion lines; the ledger and the converter audit rows use the same form (stored
#: text)
INHERIT_HEAD = "inherited(parent {parent})"

#: **Static articulated object = one rigid body**: when the jointed part moved as one body with the base in this clip
#: (pairing one_cluster) and the clip range (max − min) of joint values measured with the spec joint is at or below the
#: threshold, the base and all parts become one rigid body at their recorded poses instead of a joint (rigid joint,
#: oakink2.fixed_merge, the same path as fixed_only). Independent of role (handled · received · touched); roles are
#: decided by steps 3 · 4 after 2b. Above the threshold (an articulation that actually moved) it stays articulated.
#: Units = hinge ° (degrees) · slide m. Marker STATIC_ARTIC_TAG (the converter prints static_mark per tree).
STATIC_ARTIC_MAX = {"hinge": 2.0, "slide": 0.005}
STATIC_ARTIC_TAG = "[ARTIC_STATIC_RIGID]"


def static_artic(kind: str, lo: float, hi: float) -> tuple[bool, str]:
    """Static articulation rule: joint value min `lo` · max `hi` (hinge rad · slide m, the values stored in the tree) →
    (static articulated object?, range text '<value><unit>'). A kind outside the table (STATIC_ARTIC_MAX) or a negative
    range (lo > hi) fails."""
    import math  # noqa: PLC0415 — this module has no dependencies at import time

    if kind not in STATIC_ARTIC_MAX:
        raise ValueError(
            f"{STATIC_ARTIC_TAG} joint kind {kind!r} is outside the threshold table {sorted(STATIC_ARTIC_MAX)}"
        )
    rng = float(hi) - float(lo)
    if not rng >= 0.0:
        raise ValueError(f"{STATIC_ARTIC_TAG} joint value range is negative or not a number: lo={lo} hi={hi}")
    val, unit = (math.degrees(rng), "deg") if kind == "hinge" else (rng, "m")
    return val <= STATIC_ARTIC_MAX[kind], (f"{val:.3f}{unit}" if kind == "hinge" else f"{val * 100:.3f}cm")


def piece_verdict(parent: str, word: str, why: str = "") -> tuple[str, str] | None:
    """Verdict for a piece cut from parent `parent`, which D1g excluded (reason word `word` · description `why`). **The
    one owner of the inheritance rule** (the piece writer segment.build.sync_piece_exclusions and the converter
    oakink2.artic_convert both call it; if only one side passed it on, lines would vanish when 2b reruns). A verdict on
    the object itself gives (word, description) = the inherited exclusion; an interval-dependent one gives None =
    measure
    again in the piece window (the converter measures the piece). Word classes are owned by oakink2.part_kinds
    (PIECE_INHERIT; undecided words fail there)."""
    from dexengine.data.oakink2 import (
        part_kinds as PK,  # noqa: PLC0415 — owner of the reason words (this module has no dependencies at import time)
    )

    if not PK.piece_inherits(word):
        return None
    return word, INHERIT_HEAD.format(parent=parent) + (f" · {why}" if why else "")


def pre_row(tree: str, ti: dict, ledger: dict | None = None) -> dict | None:
    """Rows of the converter (oakink2.artic_convert.convert_tree, 2b) that are **decided without measuring**: already at
    this version (skip_done) · old version (error: merged trees hold no source part poses) · inherited exclusion (the
    piece's parent was excluded by a verdict on the object itself: piece_verdict; rerunning 2b keeps the line the piece
    writer passed on) · a piece that should inherit but has a conversion record (error: ledger and tree disagree).
    Otherwise None (measure; interval-dependent verdicts too, in the piece window).
    ledger = D1g exclusion ledger {clip: (word, description)}; empty reads the ledger once at the start of the run (a
    parent verdict changed within a run is picked up by the next run). A decided row with a lever-pin request
    (lever_for)
    fails, except a tree already at this version whose record holds as many lever pins as requested (a tree built with
    lever pins)."""
    name, par = os.path.basename(tree), ti.get("parent_clip")
    word, why = (_ledger() if ledger is None else ledger).get(par, (None, "")) if par else (None, "")
    v = piece_verdict(par, word, why) if word else None
    done = EXCLUDE_SRC in (ti.get("rules") or {})
    lv = lever_for(ti) if (done or v is not None) else frozenset()
    if lv and (
        v is not None or lever_recorded(ti) < len(lv)
    ):  # lever-pin request on a tree that is not measured (not a new base)
        raise ValueError(
            f"{LEVER_RIGID_TAG} {name}: lever-pin request {sorted(lv)} on a tree that is not measured — an "
            "inherited-exclusion piece or an old static-articulation skip state (between two passes); "
            "rerun after regenerating base (2)"
        )
    if done and v is None:
        got = ti["rules"][EXCLUDE_SRC].get("version")
        return {
            "clip": name,
            "status": "skip_done"
            if got == D1G_VERSION
            else f"error:old conversion version {got} — rerun after regenerating base (2) (merged trees hold no "
            "source part poses)",
        }
    if v is None:
        return None
    if done:
        return {
            "clip": name,
            "status": f"error:inherit_clash:{par}",
            "why": f"parent {par} excluded as {word} (passed on), yet the piece has rules.{EXCLUDE_SRC}; cut it again",
        }
    return {"clip": name, "status": f"excluded:{v[0]}", "exclude": v[0], "why": v[1], "inherited": par}


@functools.cache
def _ledger() -> dict:
    """D1g exclusion ledger {clip: (word, description)}, read through data.rules (late; no dependencies at import
    time)."""
    from dexengine.data import rules as RR  # noqa: PLC0415

    full = RR.retarget_excluded("oakink2")
    return {c: (w, full[c].partition(" — ")[2]) for c, w in RR.retarget_excluded_by("oakink2", EXCLUDE_SRC).items()}


def tally(rows: list[dict], n_trees: int, n_no_parts: int, limited: bool = False) -> dict:
    """Fate count of one converter run: converted (DONE) · **explicit exclusion** (the row's `exclude` = the reason word
    of
    the spec verdict, set where the verdict is made) · **error** (everything else: exceptions, broken input, skips
    without
    a reason) · not applicable (no parts). When the four do not sum to the input tree count ok=False (--limit is not
    checked). rc = 0 (trees > 0 · errors 0 · sum matches), else 3; exclusions do not raise rc. Reason words are not
    split
    here (this path stays when words change), but they must come from the closed vocabulary
    (oakink2.part_kinds.EXCLUDE_WORDS): a word outside it fails (before reaching the ledger)."""
    from dexengine.data.oakink2 import (
        part_kinds as PK,  # noqa: PLC0415 — owner of the exclusion reason words (this module has no dependencies at import time)
    )

    excl = [r for r in rows if r.get("exclude")]
    for r in excl:
        PK.exclude_label(r["exclude"])
    done = [r for r in rows if not r.get("exclude") and r["status"] in DONE]
    errs = [r for r in rows if not r.get("exclude") and r["status"] not in DONE]
    by: dict = {}
    for r in excl:
        by[r["exclude"]] = by.get(r["exclude"], 0) + 1
    ok = limited or len(done) + len(excl) + len(errs) + n_no_parts == n_trees
    return {
        "done": len(done),
        "excluded": excl,
        "errors": errs,
        "by_reason": dict(sorted(by.items())),
        "ok": ok,
        "rc": 0 if (n_trees and not errs and ok) else 3,
    }


def static_mark(clip: str, dec: dict, status: str) -> str:
    """Static-articulation marker line: whether conversion decision `dec` (field static_artic) is one rigid body
    (rigid-joint
    asset) or stays articulated because the cluster is outside the rigid-joint band (reason in skip)."""
    st = dec["static_artic"]
    who = dec["fixed"]["anchor"]["name"] if "fixed" in dec else dec["a_o"]["obj_id"]
    to = f"stays articulated: {st['skip']}" if "skip" in st else f"rigid joint {dec['asset']}"
    if st.get("lever_rigid"):  # one rigid body that also fixes the out-of-band parts at their recorded poses
        to = f"rigid joint {dec['asset']} · out-of-band parts fixed {','.join(st['lever_rigid'])}({st['band_out']})"
    tag = LEVER_RIGID_TAG if st.get("lever_rigid") else STATIC_ARTIC_TAG
    return f"{tag} {clip} {who} kind={st['kind']} range={st['range']} → {to} ({status})"


#: Lever pins (rule D1-g, case "static articulated object whose attached part is outside the fixed-merge band", e.g. a
#: microwave door at rest with a swinging lever): when such an object is on the judge's static-object path, it is built
#: as one rigid body that also fixes the out-of-band part at its recorded pose (the clip median of the fixed-merge
#: record). The trees it applies to are listed in a tracked table (hand-independent: rows of every hand are merged).
LEVER_RIGID_TAG = "[ARTIC_LEVER_RIGID]"
#: The lever-pin table and the columns read from it.
LEVER_LIST_FILE = Path(__file__).resolve().parents[3] / "configs" / "datasets" / "oakink2_lever_rigid.tsv"
LEVER_LIST_COLS = ("hand", "dataset", "clip", "object")


@functools.cache
def _lever_list() -> dict:
    """{(dataset, clip): object names} from the lever-pin table (hand column ignored, rows merged). A missing table or
    a row with an empty column is fatal."""
    f = LEVER_LIST_FILE
    if not f.is_file():
        raise ValueError(f"{LEVER_RIGID_TAG} lever-pin table {f} is missing")
    rows = f.read_text(encoding="utf-8").splitlines()
    head = rows[0].split("\t") if rows else []
    if miss := [k for k in LEVER_LIST_COLS if k not in head]:
        raise ValueError(f"{LEVER_RIGID_TAG} {f}: header lacks columns {miss}")
    out: dict = {}
    for ln in rows[1:]:
        if not ln.strip():
            continue
        c = dict(zip(head, ln.split("\t"), strict=False))
        if not all(c.get(k) for k in LEVER_LIST_COLS):
            raise ValueError(f"{LEVER_RIGID_TAG} {f}: empty column in row {ln!r}")
        out.setdefault((c["dataset"], c["clip"]), set()).add(c["object"])
    return {k: frozenset(v) for k, v in out.items()}


def lever_for(ti: dict) -> frozenset:
    """Object names of this tree (ti['clip_id']) to build as lever-pinned rigid bodies; empty when not listed."""
    clip = str(ti.get("clip_id") or "")
    return _lever_list().get((str(ti.get("dataset_name") or ""), clip), frozenset())


def lever_recorded(ti: dict) -> int:
    """Number of decisions in this tree's D1g record that also fixed out-of-band parts (static_artic.lever_rigid): was
    the tree built with lever pins."""
    rec = (ti.get("rules") or {}).get(EXCLUDE_SRC) or {}
    return sum(1 for r in [rec, *(rec.get("multi") or [])] if (r.get("static_artic") or {}).get("lever_rigid"))


def lever_used(clip: str, lever: frozenset, plans: list) -> None:
    """Whether the converter used every lever-pin request `lever` in this tree's decisions `plans`; an unused name fails
    (base was not rebuilt or the object is not a static-articulation skip)."""
    if miss := lever - {p["fixed"]["anchor"]["name"] for p in plans if p.get("static_artic", {}).get("lever_rigid")}:
        raise ValueError(
            f"{LEVER_RIGID_TAG} {clip}: requests {sorted(miss)} unused — not a static-articulation skip "
            "object of the new base"
        )
