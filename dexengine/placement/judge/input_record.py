#!/usr/bin/env python3
"""Judge input record (one line per clip) — what the judge saw, so a rule change re-judges only the clips it touches.

Clips the judge rejected have no tree, so the judge writes the inputs it used; a later comparison
(dexengine.placement.train_ready) recomputes the same functions on the current base and rules and re-judges only the
clips whose record differs.

Record = nine fields (`jfp3|key=value|…`, no tab or newline — it fits one ledger cell):
  rs        rules registry version `datasets.RULESET_VERSION`
  rules     condition hash per stage (`j:<h>,b:<h>,g:<h>`, keys = STAGES). Judge (j) = the record flags of the judge
            configuration (judge.config.FINGERPRINT_FLAGS minus exhaust-max) · dz layers · judge-stage registry
            thresholds · search origin (key `center_scene` only for datasets where it is on) · arm-flip rule (key
            `arm_flip` only when it differs from the default). Bake (b) · gate (g) = that stage's registry thresholds.
            Thresholds come from `datasets.resolve`, only those that apply to this dataset and scene (key = the value
            owner's symbol); stage tokens map to hashes by pipeline position (input_fields.STAGE_POSITION).
  budget    search budget `--exhaust-max` (before the one-arm x2). Monotone: a stored value ≥ the current one counts
            as the same (a larger budget's rejection is a rejection under a smaller one).
  collider  arm collider shape digest (`sim_collider.measured_digest`), plus the capsule proxy identity when the
            search filter uses it (`arm_proxy.judge_identity`) and the desk ruler.
  setup     setup values (scene : z_off : support discs x,y,z,r,kind … ~SUP~ support objects … ~D40~ exclusions) —
            the bake stamp's functions (coords.scene_z_offset · compute_desk_support / compute_aerial_support) on the
            **base tree**.
  case      hash of case · roles · active sides (collision.idle_arm.active_idle_for_tree, the judge's function) ·
            object roles
  contact   contact input hash — per expected side `dexengine.data.contact.fingerprint.input_id(...)` + file presence
  base      hash of the base trajectory `0/trajectory_kinematic.npz` qpos
  objcol    hash of the collision shapes of the main and support objects (`dexengine.data.objects.hull.collision_parts`)
The comparison looks only at the stages a row has passed (`decided_stages`, state groups owned by ledger_states).
At judge time a field that cannot be measured kills the judge (`clip_parts(strict=True)` · `_run_field`). As the
comparison's current value, a run field that cannot be measured dies; a clip field gives that row 'unevaluable'.
Why the base tree: at comparison time the f0 shadow may not exist, and f0 is a deterministic function of the base.

Modules: input_fields (vocabulary) · input_run (run fields) · input_clip (clip fields) · this module (encode, parse,
compare, CLI `--canon-conditions <dataset> <scene> <arm>`).
"""

from __future__ import annotations

import json
import math
import os

from dexengine.placement import ledger_states as _LS
from dexengine.placement.judge.input_clip import (  # noqa: F401
    _base,
    _case,
    _contact,
    _objcol,
    _objcol_digests,
    _objcol_rows,
    _setup,
    base_part,
    base_tree,
    clip_parts,
    inputs,
)
from dexengine.placement.judge.input_fields import (  # noqa: F401
    _CANON,
    ARM_FLIP_FLAG,
    CLIP_FIELDS,
    COLUMN,
    D40_SEP,
    DESK_READ,
    ERR,
    FIELDS,
    FILE_PREFIX,
    JUDGE_STAGE,
    MONOTONE_FIELDS,
    RUN_FIELDS,
    SCHEMA,
    SETUP_ROUND,
    STAGE_POSITION,
    STAGES,
    SUP_SEP,
    _ar,
    _coords,
    _csp,
    _h,
    _idle_arm,
    _objhull,
    _ql,
    _rr,
    _sf,
)
from dexengine.placement.judge.input_run import (  # noqa: F401
    _norm,
    _rules_body,
    _rules_field,
    _run_field,
    canon_conditions,
    canon_flags,
    canon_run_flags,
    collider,
    conditions,
    judge_conditions,
    stage_of,
    stage_thresholds,
)


# ── encoding · comparison ─────────────────────────────────────────────────────────────────────────────────────────
def encode(parts: dict, cond: dict) -> str:
    v = {**cond, **parts}
    bad = [k for k in FIELDS if k not in v]
    if bad:
        raise KeyError(f"judge input record: missing fields {bad}")
    for k in FIELDS:
        if any(c in str(v[k]) for c in "|\t\n"):
            raise ValueError(f"judge input record: separator in field {k}: {v[k]!r}")
    return "|".join([SCHEMA] + [f"{k}={v[k]}" for k in FIELDS])


def fingerprint(ds: str, emb: str, hand: str, clip: str, scene: str, cond: dict, active_mode: str = "auto") -> str:
    """The one function the judge and the comparison call — run fields (cond) + clip fields (clip_parts)."""
    return encode(clip_parts(ds, emb, hand, clip, scene, active_mode), cond)


def read_files(roots, key=str) -> dict:
    """Read every `<FILE_PREFIX><combo>.tsv` under the judge roots into {clip: record}, oldest file first, so a newer
    judge's record wins. key = the caller's clip-name normalisation (join key)."""
    import glob as _glob

    out: dict = {}
    for R in roots:
        for f in sorted(_glob.glob(str(R) + f"/**/{FILE_PREFIX}*.tsv", recursive=True), key=os.path.getmtime):
            with open(f) as fh:
                for ln in fh:
                    p = ln.rstrip("\n").split("\t")
                    if len(p) >= 2 and p[0]:
                        out[key(p[0])] = p[1]
    return out


def no_fingerprint(fp: str | None) -> bool:
    """The one test for 'no record' (None or empty): a segment that inherited its placement without a judge.
    setup_height raises NoSetupHeight exactly then; the gate counts it and goes on."""
    return not fp


class NoSetupHeight(ValueError):
    """The clip has no judge input record (no_fingerprint — a declared case the caller counts). A record that exists
    but cannot be read is UnreadableFingerprint, not this."""


class UnreadableFingerprint(ValueError):
    """The record exists but its height cannot be read (other SCHEMA · missing field · empty / error / broken setup).
    A defect, not 'none' (not a subclass of NoSetupHeight, so a handler of that cannot swallow it)."""


def setup_height(fp: str | None) -> float:
    """Scene height the setup field records (`scene_z_offset` — world z = capture z + this value): the height the
    judge solved the robot at, which the bake stamp must carry. No record → NoSetupHeight; unreadable →
    UnreadableFingerprint."""
    if no_fingerprint(fp):
        raise NoSetupHeight("no judge input record")
    d = parse(fp)
    parts = d["setup"].split(":") if d is not None else []
    try:
        z = float(parts[1])
    except (IndexError, ValueError):
        z = math.nan
    if not math.isfinite(z):
        raise UnreadableFingerprint(
            f"cannot read the setup height of the judge input record (not schema {SCHEMA} · "
            f"missing field · empty / error / broken setup): {str(fp)[:120]!r}"
        )
    return z


def height_violation(tree, fp: str | None) -> str | None:
    """Height contract: the stamped height of a baked aerial tree (aerial_support.z_off) equals the record's height
    (the robot was solved at it; otherwise the hand rig and the robot are offset vertically). Tolerance = one unit of
    the record's rounding. Returns the reason (data, kept as written) or None."""
    zj = setup_height(fp)
    zs = float(_coords().read_aerial_support(tree)["z_off"])
    if abs(zs - zj) <= 10.0**-SETUP_ROUND:
        return None
    return f"height:judge{zj:.4f}≠stamp{zs:.4f}({1000 * (zs - zj):+.1f}mm)"


def parse(s: str | None) -> dict | None:
    """Record text → {field: value}. None for no record or another schema (= unknown)."""
    if not s:
        return None
    p = str(s).strip().split("|")
    if p[0] != SCHEMA:
        return None
    out = {}
    for t in p[1:]:
        k, sep, v = t.partition("=")
        if sep:
            out[k] = v
    return out if all(k in out for k in FIELDS) else None


def err_fields(s: str | None) -> list[str]:
    """The `error:` fields of one record [field=value …] ('the judge could not prove its input'). [] for no record
    or another schema (that row is 'unknown')."""
    return [f"{k}={v}" for k, v in (parse(s) or {}).items() if v.startswith(ERR)]


def decided_stages(status: str) -> tuple:
    """Stages (STAGES keys) a row of this state has passed — state groups owned by ledger_states. Rows the judge
    dropped (PLACEMENT_FAILED) up to the judge; other states (GUARD_FAIL · adopted without tree) up to the gate.
    An unknown state dies."""
    if status in _LS.PLACEMENT_FAILED:
        return (JUDGE_STAGE,)
    if status in _LS.STATES:
        return STAGES
    raise ValueError(f"judge input record: unknown state {status!r} — ledger_states.STATES {_LS.STATES}")


def rules_stages(v: str) -> dict:
    """Rules field `j:<h>,b:<h>,g:<h>` → {stage: hash}. Stage keys other than STAGES raise ValueError."""
    out = dict(t.split(":", 1) for t in v.split(","))
    if sorted(out) != sorted(STAGES):
        raise ValueError(f"rules field stages {sorted(out)} ≠ {list(STAGES)}: {v!r}")
    return out


def _setup_diff(a: str, b: str) -> bool:
    """Do two setup fields differ — coords.setup_statement_diff (1 mm · kind · exclusions · support objects). Disc =
    x,y,z,r,kind (5). Records from before the disc kind (4 values) differ; records with the retired lifetime values
    (7) are read up to the kind. Any other count is outside the format and dies."""

    def split(s):
        s, _, ex = s.partition(D40_SEP)
        s, _, sp = s.partition(SUP_SEP)
        sups = []
        for t in filter(None, sp.split(";")):
            head, _, pose = t.partition("@")
            obj, main, holds = head.split(">")
            sups.append({"obj": obj, "main": main, "holds": holds == "1", "pose": [float(v) for v in pose.split(",")]})
        scene, z, discs = s.split(":", 2)
        ds_ = []
        for d in filter(None, discs.split(";")):
            t = d.split(",")
            if len(t) not in (4, 5, 7):
                raise ValueError(f"setup field disc {d!r}: {len(t)} values — outside x,y,z,r[,kind]")
            rec = {"x": float(t[0]), "y": float(t[1]), "z": float(t[2]), "r": float(t[3])}
            if len(t) > 4:
                rec["kind"] = t[4]
            ds_.append(rec)
        exl = [dict(zip(("obj", "why"), e.rsplit("/", 1), strict=False)) for e in filter(None, ex.split(","))]
        return scene, float(z), ds_, exl, sups

    sa, za, da, ea, pa = split(a)
    sb, zb, db, eb, pb = split(b)
    return sa != sb or bool(
        _coords().setup_statement_diff(
            da, db, za, zb, excluded_stamped=ea, excluded_current=eb, supports_stamped=pa, supports_current=pb
        )
    )


def diff(stored: str | None, current: str, status: str) -> tuple[str, list]:
    """(verdict, differing fields) — verdict ∈ same · stale · unknown (no stored record / other schema) ·
    unevaluable (a current clip field cannot be measured). The current side comes first: what cannot be measured now
    would hit the same wall when re-judged, so the row is counted, not listed. A current **run** field with an error
    is the same for every row, so it raises ValueError (canon_conditions should have died first). The rules field
    compares only the stages the row has passed (decided_stages) and reports others as `rules.<key>`."""
    stages = decided_stages(status)
    cur = parse(current)
    if cur is None:
        raise ValueError(f"current record is outside the format: {current!r}")
    bad_now = [k for k in FIELDS if cur[k].startswith(ERR)]
    if dead := [k for k in bad_now if k in RUN_FIELDS]:
        raise ValueError(f"current record run fields {dead} cannot be measured — no comparison here: {current!r}")
    if bad_now:
        return "unevaluable", bad_now
    st = parse(stored)
    if st is None:
        return "unknown", []
    out = []
    for k in FIELDS:
        a, b = st[k], cur[k]
        if a == b:
            continue
        if a.startswith(ERR):
            out.append(k)
        elif k in MONOTONE_FIELDS:
            if int(a) < int(b):
                out.append(k)
        elif k == "rules":
            try:
                ra, rb = rules_stages(a), rules_stages(b)
                out += [f"rules.{s_}" for s_ in stages if ra[s_] != rb[s_]]
            except ValueError:
                out.append(k)
        elif k == "setup":
            try:
                if _setup_diff(a, b):
                    out.append(k)
            except ValueError:
                out.append(k)
        else:
            out.append(k)
    return ("stale" if out else "same"), out


#: Fields the rules registry leaves in the record — when the version stamp (rs) differs, their differences are read
#: by the registry's version-change predicate (datasets.ruleset_row_reason).
RULESET_FIELDS = ("rs", "rules")


def diff_ruleset(stored: str | None, current: str, status: str, row: tuple[str, str, str, str]) -> tuple[str, list]:
    """diff plus the registry version-change predicate — `row` = (dataset, emb, hand, clip). With a different stamp,
    rs · rules differences count as the same when ruleset_row_reason is None (the change does not reach this clip);
    the other fields are compared as usual. Within one version, rules differences that the owner-move record
    (judge.owner_moves.equivalent_stages — same values, threshold symbols moved) explains count as the same."""
    verdict, comps = diff(stored, current, status)
    ruled = [k for k in comps if k.split(".")[0] in RULESET_FIELDS]
    if verdict != "stale" or not ruled:
        return verdict, comps
    old = (parse(stored) or {}).get("rs")
    if old == (parse(current) or {}).get("rs"):
        st, cur = parse(stored), parse(current)
        moved = [k.split(".", 1)[1] for k in comps if k.startswith("rules.")]
        if not moved or cur["setup"].startswith(ERR):
            return verdict, comps
        from dexengine.placement.judge import owner_moves as _om  # noqa: PLC0415 — owner-move record

        same = _om.equivalent_stages(row[0], cur["setup"].split(":", 1)[0], st["rules"], cur["rules"], moved)
        rest = [k for k in comps if not (k.startswith("rules.") and k.split(".", 1)[1] in same)]
        return ("stale" if rest else "same"), rest
    if _rr().ruleset_row_reason(*row[:4], old) is not None:
        return verdict, comps
    rest = [k for k in comps if k not in ruled]
    return ("stale" if rest else "same"), rest


def main(argv=None) -> int:
    """Print the comparison's current run fields (canon_conditions) — the reproduce command of a failed comparison."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--canon-conditions",
        nargs=3,
        required=True,
        metavar=("<dataset>", "<scene>", "<arm>"),
        help="the comparison's current run fields (canon_conditions) as one JSON line",
    )
    print(json.dumps(canon_conditions(*ap.parse_args(argv).canon_conditions), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
