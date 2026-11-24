"""D23: apply object roles (main/background, side, shared, slots) to base from the **hand–object assignment record**
(`rules.ASSIGN`). The assignment rules (who carries what, assisting hand, fixed roles) live in
dexengine/data/cases/hand_object_assign.py, the record is written by the case stage (cases.taxonomy, stage 1 step 3).
This file has no rules of its own; it only applies them.

Application (D23)
-----------------
  main object  = rules.ASSIGN.per_side[s].main: the one object the assignment left for each active hand (case.active).
                 None (inactive hand, hand dropped as assisting) means that hand has no main object (idle pose).
  shared (OS)  = the two hands' main objects have the same **name** (shared_object · side=both). In a part group where
                 the hands hold different parts, the assignment has a shared unit (ASSIGN.shared) but there are two
                 objects, and each hand's own part is its main object.
  support      = a still object under a main object that holds it up (role=support, side = side of the supported main
                 object, both if two). Decided by coords.support_of alone (first-frame geometry, independent of holds;
                 a property of the scene). Undecidable cases (several candidates, moving support, flipped, beyond the
                 containment limit) get no role and only a reason in rules.D23.support (that side takes the disc path).
                 The env spawns support objects fixed in their first-frame pose; the bake stamp supports states it.
  background   = objects that are neither a main object nor a support (role=extra, side=None). Not spawned; not a
                 reward or termination target.
  tag          = ASSIGN.obj_tags (carry · hold_static · static_support · background); information, does not change
                 roles or slots.
  retarget keys = `<side>_object_mesh_dir` · `_convex_dir` = retarget folder of that side's main object (None without
      one).
D24 input (lowest z of the main object's collision-hull trajectory) is measured by
dexengine.data.desk_penetration.measure once the main object is set; marker rules.D24.
Marker: task_info.rules.D23 = {"version", "assign", "changed", "before", "support"}: applied version · assignment
version
      read · change history (audit) · support decisions (side → support · why · candidates).
Main objects and active hands are stated only in rules.ASSIGN.
A missing assignment record, or one from another version or contact source, fails (StaleCase: run case stage 3 first).

Required follow-up: objects[] assignments change, so regenerate the canonical contact sidecar with
`contact/sidecar.py --dataset <ds>` (otherwise contact_derive in the bake chain fails with FAIL_ASSIGN).
Target = the **base layer** (data/<ds>/base/<hand>/<emb>/<clip>/task_info.json). Baked trees (trees/) are copies of
base,
so clips whose roles changed need rejudging and rebaking; this script writes the changed-clip list (--out-changed) for
the rejudge queue.

Usage:
  python3 dexengine/data/object_roles.py --datasets hocap,grab,h2o,hot3d,oakink2,dexcanvas,hrdexdb,arctic   # dry-run
      (dry-run report = <name>.dryrun.json; it does not overwrite the report of a write run)
  python3 dexengine/data/object_roles.py --datasets ... --write --out-changed <changed clips>.tsv
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import glob
import json
import os
import time

import numpy as np

from dexengine.data import rules as RR  # owner of the hand axis

HAND = RR.hand_dir()
from dexengine.core import paths as _P  # noqa: E402 — owner of layer paths (layer_path)
from dexengine.core.paths import (
    main_root as _data_root,  # noqa: E402 — owner of the data root (env DEXENGINE_MAIN_ROOT)
)
from dexengine.core.paths import qpos_layout as _QL  # noqa: E402 — owner of the side vocabulary (qpos_layout.SIDES)
from dexengine.data.contact import fingerprint as SF  # noqa: E402 — owner of the sidecar inputs the case read
from dexengine.data.contact.source import (
    CONTACT_SOURCE,  # noqa: E402 — D35: the roles' contact source = the cases' contact source
)
from dexengine.data.object_roles_report import exit_code, print_table, report_path  # noqa: E402
from dexengine.data.tree_io import (
    write_json_if_changed,  # noqa: E402 — writes base task_info only when the content changed (keeps the f0 cache fingerprint)
)

VERSION = (
    "D23-ASSIGN-2"  # version applying roles from the hand–object assignment record (rules.ASSIGN) · 2: support role
)
from dexengine.data import (
    desk_penetration as D24,  # noqa: E402 — owner of the D24 input measurement and version (rules.D24 marker)
)
from dexengine.data.cases import (
    assign_version as AR,  # noqa: E402 — owner of the assignment version (one shared field, hand_version)
)
from dexengine.data.cases.assign_version import (
    ASSIGN_KEY,  # noqa: E402 — key of the assignment record (dependency-free owner)
)

D24_VERSION = D24.D24_VERSION
SIDES = _QL.SIDES


class StaleCase(ValueError):
    """Case or assignment is missing or was made with **another version or contact source**; roles on top of it would
    disagree with the case. Not swallowed by the census."""


def from_assign(ti: dict, hand: str = HAND) -> dict:
    """task_info → {"active", "main": {active side: object name|None}, "roles": {object: (role, side)}, "tags",
    "assign"}:
    pure function moving the assignment record into roles (testable). No case → ValueError (no_case); an assignment
    record not on that hand's version (dexengine.data.cases.assign_version.hand_version) → StaleCase. "assign" = version
    of the applied record (rules.D23.assign marker)."""
    case = ti.get("case") or {}
    if "active" not in case:
        raise ValueError("task_info.case.active missing — run cases/taxonomy.py --write first (no silent both)")
    want = AR.hand_version(hand)
    a = (ti.get("rules") or {}).get(ASSIGN_KEY)
    if not isinstance(a, dict) or a.get("version") != want:
        raise StaleCase(
            f"rules.{ASSIGN_KEY} version {(a or {}).get('version') if isinstance(a, dict) else a!r} ≠ "
            f"{want}({hand}) — run cases/taxonomy.py --write first"
        )
    if a.get("contact_source") != CONTACT_SOURCE:
        raise StaleCase(
            f"rules.{ASSIGN_KEY}.contact_source={a.get('contact_source')!r} ≠ current {CONTACT_SOURCE!r} — "
            "run cases/taxonomy.py --write first (D35)"
        )
    hands = [h for h in (ti.get("hands") or []) if h in SIDES] or [
        s for s in SIDES if any(e.get("kind") == "hand" and e.get("side") == s for e in ti.get("qpos_layout") or [])
    ]
    per = a.get("per_side") or {}
    active = [s for s in SIDES if s in hands and (per.get(s) or {}).get("active")]
    if active != [s for s in SIDES if s in case["active"] and s in hands]:
        raise StaleCase(
            f"assignment active hands {active} ≠ case active hands {case['active']} — two records of the same "
            "step disagree (rerun 3)"
        )
    objs = ti.get("objects") or []
    main = {s: per[s].get("main") for s in active}
    lost = sorted({n for n in main.values() if n} - {o["name"] for o in objs})
    if lost:
        raise StaleCase(
            f"assignment main objects {lost} not in objects[] — the assignment is older than this base (rerun 3)"
        )
    users = collections.defaultdict(list)
    for s, n in main.items():
        if n:
            users[n].append(s)
    roles = {}
    for o in objs:
        u = users.get(o["name"], [])
        roles[o["name"]] = ("main", "both" if len(u) == 2 else u[0]) if u else ("extra", None)
    return {"active": active, "main": main, "roles": roles, "tags": a.get("obj_tags") or {}, "assign": want}


class SupportUndecidable(RuntimeError):
    """The support decision cannot be measured (mesh or trajectory unreadable); counted as a failure, not as no case
    (ValueError)."""


def support_roles(ti: dict, dec: dict, tree_dir: str) -> dict:
    """Add support roles to dec (from_assign): coords.support_of per main object (independent of holds, first-frame
    geometry). One support object → that object becomes (support, side of the supported main object, both if two).
    Undecidable cases leave only a reason in dec["support"]."""
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — owner of the support decision (placement layer)

    mains = {n: dec["roles"][n][1] for n in dict.fromkeys(v for v in dec["main"].values() if v)}
    if not mains:
        return {**dec, "support": {}}
    view = {
        **ti,
        "objects": [
            {**o, "role": dec["roles"][o["name"]][0], "side": dec["roles"][o["name"]][1]}
            for o in ti.get("objects") or []
        ],
    }
    npz = glob.glob(os.path.join(tree_dir, "0", "trajectory_kinematic.npz"))
    try:
        with np.load(npz[0]) as z:
            q = z["qpos"]
        by = {o["name"]: o for o in view["objects"]}
        rec = {n: C.support_of(tree_dir, view, q, by[n]) for n in mains}
    except (IndexError, OSError, KeyError, ValueError) as e:
        raise SupportUndecidable(f"cannot measure the support decision: {type(e).__name__}: {e}"[:300]) from e
    roles = dict(dec["roles"])
    for n, r in rec.items():
        if r["support"]:
            prev = roles[r["support"]][1] if roles[r["support"]][0] == "support" else None
            roles[r["support"]] = ("support", mains[n] if prev in (None, mains[n]) else "both")
    return {
        **dec,
        "roles": roles,
        "support": {n: {k: r[k] for k in ("support", "why", "candidates")} for n, r in rec.items()},
    }


#: Fields apply rewrites **per tree** (parents and D36 pieces, each from its own assignment); a piece overwrites the
#: values it got from its parent with its own contact. The D36 piece invariant
#: (dexengine.data.segment.invariant.piece_stale_reason: piece copy = parent PIECE_COPY) reads this declaration from
#: source to mask them (dexengine.data.segment.invariant.RESTATED_DECL). It must equal the fields apply writes: fewer
#: would make pieces with different roles stale every time (re-split, rejudge); more would hide real parent changes.
ROLE_TOP = (
    "shared_object",
    "right_object_slot",
    "left_object_slot",
    "right_object_mesh_dir",
    "left_object_mesh_dir",
    "right_object_convex_dir",
    "left_object_convex_dir",
)
ROLE_OBJECT = ("role", "side", "tag")  # per objects[] element
ROLE_LAYOUT = ("role", "side")  # per object element of qpos_layout[] (kind == "object"); not the side of hand elements


def apply(ti: dict, dec: dict, tree_dir: str) -> tuple[dict, bool]:
    """Align roles, slots, sharing, layout side, tags and retarget keys with the assignment and measure the D24 input.
    True when roles changed."""
    before = {o["name"]: (o.get("role"), o.get("side")) for o in ti.get("objects") or []}
    changed = any(before[n] != dec["roles"][n] for n in before)
    for o in ti.get("objects") or []:
        o["role"], o["side"] = dec["roles"][o["name"]]
        o["tag"] = dec["tags"].get(o["name"])
    for e in ti.get("qpos_layout") or []:
        if e.get("kind") == "object" and e.get("name") in dec["roles"]:
            e["role"], e["side"] = dec["roles"][e["name"]]
    for s in SIDES:
        ti.pop(f"{s}_object_slot", None)
    for s, n in dec["main"].items():
        if n:
            ti[f"{s}_object_slot"] = next(o["slot"] for o in ti["objects"] if o["name"] == n)
    named = [n for n in dec["main"].values() if n]
    ti["shared_object"] = (
        len(named) == 2 and named[0] == named[1]
    )  # the two hands' main objects are the **same object** (by name)
    for s in SIDES:  # retarget keys = folder of that side's main object (both representations agree)
        m = next((o for o in ti.get("objects") or [] if o["name"] == dec["main"].get(s)), None)
        ti[f"{s}_object_mesh_dir"] = m["spider_mesh_dir"] if m else None
        ti[f"{s}_object_convex_dir"] = m["spider_convex_dir"] if m else None
    ti.setdefault("rules", {})["D24"] = {"version": D24_VERSION, "basis": D24.measure(ti, tree_dir, set(named))}
    prev = (ti.get("rules") or {}).get("D23") or {}
    # A rerun (idempotent) keeps the first application's record (before/changed) as an audit trail.
    ti["rules"]["D23"] = {
        "version": VERSION,
        "assign": dec["assign"],
        "changed": bool(changed or prev.get("changed")),
        "before": ({n: list(v) for n, v in before.items()} if changed else prev.get("before")),
        "support": dec.get("support", {}),
    }
    return ti, changed


def check_case_input(ds: str, tree_dir: str, ti: dict) -> None:
    """Whether the sidecar inputs the case read (case.contact_input, input_id per side) equal **this base's** inputs
    now.

    A case is only written after checking the sidecar against this base (cases.taxonomy). When the record is missing (it
    may come from an old-pose sidecar) or differs from the value recomputed from this base (base, MANO or code version
    changed after the case), no roles are built on that case: StaleCase.
    """
    case = ti.get("case") or {}
    if "active" not in case:  # no case or UNCLASSIFIED; owned by the no_case path of from_assign()
        return
    rec = case.get("contact_input") or {}
    if rec.get("schema") != SF.FP_SCHEMA:
        raise StaleCase(
            f"case.contact_input version {rec.get('schema')!r} ≠ {SF.FP_SCHEMA} — case from before sidecar "
            "fingerprints. "
            "run cases/taxonomy.py --write first"
        )
    npz = glob.glob(os.path.join(tree_dir, "*", "trajectory_kinematic.npz")) or glob.glob(
        os.path.join(tree_dir, "trajectory_kinematic.npz")
    )
    with np.load(npz[0]) as z:
        kin = z["qpos"]
    tin = SF.TreeInputs(_data_root(), ds, os.path.basename(tree_dir), ti, kin, HAND)  # data root (not the code tree)
    for side in SF.sides_for(ti, SF.LAYER_ALL):
        try:
            now = SF.input_id(tin.expected(SF.LAYER_ALL, side))
        except (
            KeyError,
            ValueError,
            OSError,
        ) as e:  # inputs that cannot be measured leave no ground to trust the case either
            raise StaleCase(
                f"cannot measure the {side} sidecar inputs on this base: {type(e).__name__}: {e}"[:300]
            ) from e
        was = (rec.get("sides") or {}).get(side)
        if was != now:
            raise StaleCase(
                f"{side} sidecar inputs the case read {was} ≠ current base {now} — base, MANO or code version "
                "changed after the case. "
                "run contact.sidecar --all-objects → cases/taxonomy.py --write first"
            )


#: Per (dataset, emb) counters of one roles run, in report order.
ROW_KEYS = (
    "candidates",
    "trees",
    "changed",
    "main_to_extra",
    "extra_to_main",
    "shared_flip",
    "idle_hand",
    "active_no_main",
    "stale_case",
    "no_case",
    "case_missing",
    "written",
    "same",
    "support",
    "support_fail",
)


@dataclasses.dataclass
class RunTally:
    """Trees one roles run collects across datasets. stale: roles not built on a stale case (any -> rc 2, no silent
    partial success). missing_case: **no case at all**, the case stage (3) did not run on them (an upstream gap; unlike
    UNCLASSIFIED, a case without active that legitimately yields no roles); any -> rc 3. changed: (dataset, emb, clip)
    whose roles changed."""

    stale: list[str] = dataclasses.field(default_factory=list)
    missing_case: list[str] = dataclasses.field(default_factory=list)
    changed: list[tuple[str, str, str]] = dataclasses.field(default_factory=list)


def run(datasets: list[str], write: bool, out_changed: str | None) -> tuple[list[dict], RunTally]:
    rows = []
    tally = RunTally()
    for ds in datasets:
        for emb in ("bimanual", "right", "left"):
            root = str(_P.layer_path("base", ds, hand=HAND, emb=emb))  # owner of layer paths (data root)
            if not os.path.isdir(root):
                continue
            c = collections.Counter()
            for ti_path in sorted(glob.glob(os.path.join(root, "*", "task_info.json"))):
                _role_tree(ds, emb, ti_path, write, c, tally)
            rows.append(
                {
                    "dataset": ds,
                    "emb": emb,
                    **{k: c[k] for k in ROW_KEYS},
                    "support_undecided": {
                        k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("support_undecided:")
                    },
                }
            )
    if out_changed:
        with open(out_changed, "w") as fh:
            fh.write("dataset\temb\tclip\n" + "".join(f"{d}\t{e}\t{c}\n" for d, e, c in tally.changed))
    return rows, tally


def _role_tree(ds: str, emb: str, ti_path: str, write: bool, c: collections.Counter, tally: RunTally) -> None:
    """One base tree: decide its roles from the assignment record, count what changed in `c`, write when asked."""
    with open(ti_path) as fh:
        ti = json.load(fh)
    c["candidates"] += 1
    if not ti.get("case"):
        c["case_missing"] += 1
        tally.missing_case.append(f"{ds}/{emb}/{os.path.basename(os.path.dirname(ti_path))}")
        return
    try:
        if "case" in ti:
            check_case_input(ds, os.path.dirname(ti_path), ti)
        dec = support_roles(ti, from_assign(ti), os.path.dirname(ti_path))
    except SupportUndecidable as e:
        c["support_fail"] += 1
        tally.stale.append(f"{ds}/{emb}/{os.path.basename(os.path.dirname(ti_path))}: {e}")
        return
    except StaleCase as e:
        c["stale_case"] += 1
        tally.stale.append(f"{ds}/{emb}/{os.path.basename(os.path.dirname(ti_path))}: {e}")
        return
    except ValueError:
        c["no_case"] += 1
        return
    before = {o["name"]: (o.get("role"), o.get("side")) for o in ti.get("objects") or []}
    c["trees"] += 1
    diff = [n for n in before if before[n] != dec["roles"][n]]
    if diff:
        _count_role_changes(c, before, dec, diff)
        tally.changed.append((ds, emb, os.path.basename(os.path.dirname(ti_path))))
    if any(s not in dec["active"] for s in (ti.get("hands") or SIDES)):
        c["idle_hand"] += 1
    c["support"] += sum(1 for r in dec["support"].values() if r["support"])
    for r in dec["support"].values():
        if r["why"]:
            c[f"support_undecided:{r['why']}"] += 1
    if any(dec["main"][s] is None for s in dec["active"]):
        c["active_no_main"] += 1  # active hand dropped as assisting
    if write:
        ti2, _ = apply(ti, dec, os.path.dirname(ti_path))
        # Write only when the content changed: unconditional writes would change the mtime of every tree in the
        # dataset and invalidate the whole stage 1 f0 cache (fingerprint = size + mtime). Owned by tree_io.
        c["written" if write_json_if_changed(ti_path, ti2) else "same"] += 1


def _count_role_changes(c: collections.Counter, before: dict, dec: dict, diff: list[str]) -> None:
    c["changed"] += 1
    if any(before[n][0] == "main" and dec["roles"][n][0] == "extra" for n in diff):
        c["main_to_extra"] += 1
    if any(before[n][0] == "extra" and dec["roles"][n][0] == "main" for n in diff):
        c["extra_to_main"] += 1
    if any((before[n][1] == "both") != (dec["roles"][n][1] == "both") for n in diff):
        c["shared_flip"] += 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="D23 contact-based object roles — dry-run by default, --write updates base task_info"
    )
    ap.add_argument("--datasets", required=True, help="comma list")
    ap.add_argument("--write", action="store_true")
    ap.add_argument(
        "--out-changed", default=None, help="TSV of (dataset, emb, clip) whose roles changed — rejudge queue input"
    )
    a = ap.parse_args()
    dss = [d.strip() for d in a.datasets.split(",") if d.strip()]
    rows, tally = run(dss, a.write, a.out_changed)
    out_p, why = report_path(dss, a.write)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as fh:
        # The `hand` field: outputs without it stay silent when hands mix (the guard needs the field).
        json.dump(
            {
                "generated": time.strftime("%Y-%m-%d %H:%M"),
                "version": VERSION,
                "hand": HAND,
                "datasets": sorted(dss),
                "write": a.write,
                "rows": rows,
            },
            fh,
            ensure_ascii=False,
            indent=1,
        )
    print(f"[roles] report → {out_p} ({why}) hand={HAND}", flush=True)
    print_table(rows, a.write)
    return exit_code(rows, tally)


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.object_roles")
    raise SystemExit(main())
