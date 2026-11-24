#!/usr/bin/env python3
"""Extra-slot count K of a hand x dataset form: a statement counted once over all trees, checked by every job.

  statement  write(hand, arm, ds): over every hand x dataset tree of the data root (each DL.EMBODIMENTS branch,
             paths.trees_dir) count the clip's slots (data.objects.extra_slots.slot_rows) -> statement file
             path(hand, arm, ds) {K, count per clip, sources}. It is written before jobs run, over the full trees: a job
             tree holds only its own clips, so counting in jobs would give each piece its own K.
  read       dataset_k(hand, arm, ds): the statement's K (missing = error), checked by recounting every clip of the
             job tree (independent oracle: a different count or a clip outside the statement means the statement is
             stale). The source carries STATED_KEY.
  object_joints J: 1 when any clip of the same scan has an articulated object (task_info objects[].articulation, one
             joint), else 0. dataset_j(hand, arm, ds) = the statement's J (missing = error); an articulated object in
             the job tree with J = 0 is an error.
  stated(hand, arm, ds): the statement file as written (it also carries the form object names and the object column
             prefix for readers that cannot import store.form).
Own slots (own): a record that ran several clips (or augmented variants) in one scene carries the union of their slots
  (extra_obj_*, support_disc_*) and parks the slots of other clips below the floor. Dropping the parked slots' object
  and partner-force features makes the episode match a single-clip record of the same clip.
  A clip's own slots = receiving-object slots listed in the record meta["extra_slots"][clip] (slot numbers);
  support discs whose first-frame top z > DISC_ON_Z (same threshold as rl.tasks.scene.object_pin.support_disc_rows;
  owner store.scene_layout.DISC_ON_Z; the env always builds a fixed number of disc slots, so a single-clip record can
  also have parked discs); receiving-object stand-in slots parked at SLOT_HIDDEN_Z are parked. Dropped = parked slots'
  object pose (+ joint) features, force features whose partner is a parked slot (force_partner.<link>.<slot>), the
  slot's own force features, and that partner's column of force_object_partner; with no standing disc the partner
  'support_disc' goes too (likewise 'support_object' with no standing receiving object).
  A receiving-object slot feature without meta extra_slots is an error (the clip's slots are unknown).
Markers: [COLLECT_K], [OWN_SLOTS], [STMT_CACHE].
Usage: python -m dexengine.record.store.slots write|show --hand H --arm A --dataset D
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError, input_missing

MARK = "[COLLECT_K]"
STATED_KEY = "stated"  # info.collect.form.k_source field: K comes from the statement (dataset_k)
OWN_MARK = "[OWN_SLOTS]"
EXTRA, DISC, DISC_PARTNER = "extra_obj_", "support_disc_", "support_disc"
SUP_PARTNER = "support_object"  # partner name of receiving-object slots (rollout.partner STATIC["SUP"])
OBJ, OBJ_JOINT = "observation.object_pose.", "observation.object_joint_pos."


def die(msg: str):
    raise SystemExit(f"{MARK} {msg}")


def artic_joints(ti: dict, where: str) -> int:
    """Max joints per object of a tree's task_info: an articulation has one joint (a joint name and a type from
    augment.scene.JOINT_TYPES); any other shape is an error."""
    from dexengine.record.augment import scene as AA

    n = 0
    for o in ti.get("objects") or []:
        a = o.get("articulation")
        if not a:
            continue
        if not isinstance(a.get("joint"), str) or a.get("type") not in AA.JOINT_TYPES:
            die(
                f"{where}: object {o.get('name')} articulation fields {sorted(a)} · type {a.get('type')!r} — must be "
                f"one joint (joint name, type in {AA.JOINT_TYPES})"
            )
        n = 1
    return n


def tree_roots(hand: str, arm: str, ds: str) -> tuple[list[tuple[str, Path]], list[str]]:
    """([(emb, tree root)] that exist, combos tried) for this hand x dataset under the data root."""
    from dexengine.core import paths as P
    from dexengine.data import rules as RR

    found, combos = [], []
    for emb in P.data_layout.EMBODIMENTS:
        try:
            # tree combo suffix = the dataset axis variant (arctic = _artic)
            combo = P.combo_id(arm, hand, RR.ledger_axis_variant(RR.ledger_key(ds, emb)))
        except KeyError:  # no such (dataset, emb) axis, so no trees (all absent dies below)
            continue
        combos.append(combo)
        root = P.trees_dir(combo, ds, emb)
        if root.is_dir():
            found.append((emb, root))
    return found, combos


# Scan cache: the statement check and the store do not recount the full trees in every job. Key = tree fingerprint
# (per file: relative path, size, mtime_ns, ctime_ns, inode; a copy that keeps mtime cannot restore ctime/inode)
# + sha of the counting code (SCAN_CODE module sources) + (hand, arm, dataset). Only the scan result is cached; the
# comparison with the statement runs every time. The cache file sits next to the statement file.
STMT_CACHE_MARK = "[STMT_CACHE]"
SCAN_CODE = ("dexengine.record.store.slots", "dexengine.data.objects.extra_slots", "dexengine.record.augment.scene")
_SCANNED: dict[str, tuple] = {}


def tree_fingerprint(roots: list[tuple[str, Path]]) -> str:
    import hashlib

    h = hashlib.sha256()
    for emb, root in roots:
        for dp, dns, fns in os.walk(root):
            dns.sort()
            for fn in sorted(fns):
                st = os.stat(os.path.join(dp, fn))
                rel = os.path.relpath(os.path.join(dp, fn), root)
                h.update(f"{emb}|{rel}|{st.st_size}|{st.st_mtime_ns}|{st.st_ctime_ns}|{st.st_ino}\n".encode())
    return h.hexdigest()


def code_sha() -> str:
    import hashlib
    import importlib

    h = hashlib.sha256()
    for m in SCAN_CODE:
        h.update(Path(importlib.import_module(m).__file__).read_bytes())
    return h.hexdigest()


def cache_path(hand: str, arm: str, ds: str) -> Path:
    p = path(hand, arm, ds)
    return p.with_name(f"stmt_scan_cache__{arm}_{hand}__{ds}.json")


def scan(hand: str, arm: str, ds: str) -> tuple[dict[str, int], list[str], dict[str, int]]:
    """(slots per clip, tree roots, joints per clip), cached on disk and once per process."""
    import hashlib
    import time as _t

    t0 = _t.perf_counter()
    found, _c = tree_roots(hand, arm, ds)
    key = hashlib.sha256("|".join((tree_fingerprint(found), code_sha(), hand, arm, ds)).encode()).hexdigest()
    if key in _SCANNED:
        return _SCANNED[key]
    cp = cache_path(hand, arm, ds)
    try:
        doc = json.loads(cp.read_text()) if cp.is_file() else {}
    except (OSError, ValueError):
        doc = {}
    if doc.get("key") == key:
        out = (doc["counts"], doc["roots"], doc["arts"])
        print(
            f"{STMT_CACHE_MARK} hit {key[:16]} · {hand} × {ds} · clips {len(out[0])} · {_t.perf_counter() - t0:.1f} s",
            flush=True,
        )
    else:
        arts: dict[str, int] = {}
        counts, roots = count(hand, arm, ds, arts)
        out = (counts, roots, arts)
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_name(f".{cp.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"key": key, "counts": counts, "roots": roots, "arts": arts}, sort_keys=True))
            os.replace(tmp, cp)
        except OSError as e:  # the values were just counted; the next run counts again
            print(f"{STMT_CACHE_MARK} ⚠ cache write failed {cp}: {e}", flush=True)
        print(
            f"{STMT_CACHE_MARK} miss {key[:16]} · {hand} × {ds} · clips {len(counts)} · "
            f"counted in {_t.perf_counter() - t0:.1f} s",
            flush=True,
        )
    _SCANNED[key] = out
    return out


def count(hand: str, arm: str, ds: str, arts: dict | None = None) -> tuple[dict[str, int], list[str]]:
    """({emb/tree name: slot count}, tree roots counted) over every tree of this hand x dataset.
    With arts, also fills {emb/tree name: joint count (artic_joints)} in the same scan."""
    from dexengine.core import paths as P
    from dexengine.data.objects import extra_slots as XS

    counts, roots = {}, []
    found, combos = tree_roots(hand, arm, ds)
    for emb, root in found:
        roots.append(str(root))
        for t in sorted(p for p in root.iterdir() if (p / "task_info.json").is_file()):
            ti = json.loads((t / "task_info.json").read_text())
            try:
                rows, _ = XS.slot_rows(str(t), ti, strict=False)
            except Exception as e:  # noqa: BLE001 — an uncountable tree is an error
                die(input_missing(f"cannot count K — {emb}/{t.name}: {type(e).__name__}: {e}"))
            counts[f"{emb}/{t.name}"] = len(rows)
            if arts is not None:
                arts[f"{emb}/{t.name}"] = artic_joints(ti, f"{emb}/{t.name}")
    if not counts:
        die(
            input_missing(
                f"no trees to count K — 0 folders for {sorted(set(combos))} × {ds} in {list(P.data_layout.EMBODIMENTS)}"
            )
        )
    return counts, roots


def path(hand: str, arm: str, ds: str) -> Path:
    """Statement file: next to the final collect folder (data_layout 'collect'), inside the data root."""
    from dexengine.core import paths as P

    return (
        P.main_root()
        / Path(P.data_layout.render("collect", ds, combo=f"{arm}_{hand}")).parent
        / f"stated_k__{arm}_{hand}.json"
    )


def write(hand: str, arm: str, ds: str) -> dict:
    """Write the statement file (also the form object names and object column prefix, owner store.form)."""
    from dexengine.data.objects import extra_slots as XS
    from dexengine.record.store import form as FM

    arts: dict[str, int] = {}
    counts, roots = count(hand, arm, ds, arts)
    K = max(counts.values())
    doc = dict(
        hand=hand,
        arm=arm,
        dataset=ds,
        K=K,
        rule=f"extra_slots.slot_rows max slot count ({XS.RULE})",
        trees=roots,
        clips=len(counts),
        at_k=sorted(c for c, n in counts.items() if n == K)[:3],
        counts=counts,
        objects=FM.objects(K),
        obj_prefix=FM.OBJ,
        **{FM.JOINTS_KEY: max(arts.values(), default=0)},
        artic_clips=sum(1 for v in arts.values() if v),
    )
    p = path(hand, arm, ds)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Per-process temp name: several writers may write the same statement at once; the content is deterministic.
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True))
    os.replace(tmp, p)
    print(
        f"{MARK} statement written {hand} × {ds} K={K} · J={doc[FM.JOINTS_KEY]} (articulated clips "
        f"{doc['artic_clips']}) · clips {len(counts)} → {p}",
        flush=True,
    )
    return doc


def stated(hand: str, arm: str, ds: str) -> dict:
    """The statement file (missing or inconsistent is an error)."""
    p = path(hand, arm, ds)
    if not p.is_file():
        die(
            input_missing(
                f"no K statement {p} — write it over the full trees: `python -m dexengine.record.store.slots "
                f"write --hand {hand} --arm {arm} --dataset {ds}`"
            )
        )
    doc = json.loads(p.read_text())
    if (
        (doc.get("hand"), doc.get("arm"), doc.get("dataset")) != (hand, arm, ds)
        or not doc.get("counts")
        or doc.get("K") != max(doc["counts"].values())
        or not doc.get("obj_prefix")
        or len(doc.get("objects") or []) <= doc["K"]
    ):
        die(
            f"K statement {p}: axis or K inconsistent (hand · arm · dataset {doc.get('hand')} · {doc.get('arm')} · "
            f"{doc.get('dataset')} · K {doc.get('K')})"
        )
    return doc


def source(doc: dict) -> dict:
    """info.collect.form.k_source: the statement's sources + STATED_KEY."""
    return dict(rule=doc["rule"], trees=doc["trees"], clips=doc["clips"], at_k=doc["at_k"], **{STATED_KEY: True})


def dataset_j(hand: str, arm: str, ds: str) -> int:
    """The form's articulated-object width J from the statement, checked against the job trees."""
    from dexengine.record.store import form as FM

    doc = stated(hand, arm, ds)
    if FM.JOINTS_KEY not in doc:
        die(
            input_missing(
                f"statement {path(hand, arm, ds).name} has no {FM.JOINTS_KEY} (written before form version "
                f"{FM.FORM_VERSION}) — write it again"
            )
        )
    _c, _r, arts = scan(hand, arm, ds)
    J = int(doc[FM.JOINTS_KEY])
    if bad := sorted(c for c, n in arts.items() if n > J):
        die(
            f"job trees have {len(bad)} clips with articulated objects (e.g. {bad[:2]}) but the statement says "
            f"J={J} — stale statement, write it again"
        )
    n_art = sum(1 for v in arts.values() if v)
    print(
        f"{MARK} statement J={J} ← {path(hand, arm, ds).name} · articulated clips in job trees {n_art} (check passed)",
        flush=True,
    )
    return J


def stmt_gate(hand: str, cells: list[dict], main) -> list[tuple]:
    """Read the statements (K, J) the store will read before rendering, so a stale statement fails before GPU time
    is spent. Axis = (hand, cell env DEXENGINE_ARM, DEXENGINE_DATASET); data root = main (DEXENGINE_MAIN_ROOT)."""
    os.environ.update(DEXENGINE_MAIN_ROOT=str(main))
    axes = set()
    for cell in cells:
        env = dict(t.split("=", 1) for t in cell["env"] if "=" in t)
        if not env.get("DEXENGINE_ARM") or not env.get("DEXENGINE_DATASET"):
            die(
                input_missing(f"statement gate: cell env lacks DEXENGINE_ARM / DEXENGINE_DATASET — {cell.get('label')}")
            )
        axes.add((hand, env["DEXENGINE_ARM"], env["DEXENGINE_DATASET"]))
    for ax in sorted(axes):
        dataset_k(*ax)
        dataset_j(*ax)
    print(f"{MARK} [STMT_GATE] statements valid (before render) {sorted(axes)}", flush=True)
    return sorted(axes)


def dataset_k(hand: str, arm: str, ds: str) -> tuple[int, dict]:
    """The form K from the statement; every job-tree clip's recount must equal the statement."""
    doc = stated(hand, arm, ds)
    local, _roots, _arts = scan(hand, arm, ds)
    bad = sorted((c, n, doc["counts"].get(c)) for c, n in local.items() if doc["counts"].get(c) != n)
    if bad:
        die(
            input_missing(
                f"{len(bad)} job-tree clips' slot counts != statement: {bad[:3]} (None = not in the "
                "statement) — stale statement, write it again"
            )
        )
    print(
        f"{MARK} statement K={doc['K']} ← {path(hand, arm, ds).name} · job-tree clips {len(local)} ⊆ statement "
        f"{doc['clips']} (check passed)",
        flush=True,
    )
    return int(doc["K"]), dict(source(doc), job_clips=len(local))


def parked(cols: dict, meta: dict, clip: str) -> set[str]:
    """Names of the slots not belonging to this clip (parked): receiving-object slots and support discs."""
    from dexengine.placement.coords import core as CO
    from dexengine.record.store import form as FM
    from dexengine.record.store.scene_layout import DISC_ON_Z

    names = {k[len(OBJ) :] for k in cols if k.startswith(OBJ)}
    extras = {n for n in names if n.startswith(EXTRA)}
    if extras and clip not in (meta.get("extra_slots") or {}):
        raise InputMissingError(
            f"{OWN_MARK} ✗ {clip}: the record meta extra_slots has no entry for this clip — the "
            f"owner of slots {sorted(extras)} is unknown"
        )
    own_s = {f"{EXTRA}{int(r['slot'])}" for r in (meta.get("extra_slots") or {}).get(clip, [])}
    discs = {n for n in names if n.startswith(DISC)}
    stand = {n for n in discs if float(np.asarray(cols[OBJ + n])[0][2]) > DISC_ON_Z}
    sups = {n for n in names if n.startswith(FM.support_obj_prefix())}
    hid = {n for n in sups if abs(float(np.asarray(cols[OBJ + n])[0][2]) - CO.SLOT_HIDDEN_Z) <= FM.STAND_IN_TOL}
    return (
        (extras - own_s)
        | (discs - stand)
        | ({DISC_PARTNER} if discs and not stand else set())
        | hid
        | ({SUP_PARTNER} if sups and sups <= hid else set())
    )


def own(cols: dict, names: dict, forces: dict, meta: dict, clip: str, chans: list) -> tuple:
    """Copies of (cols, names, forces, record channel rows) without the parked slots' features.

    Channel rows whose partner or owner is a parked slot are dropped; force_object_partner rows keep the remaining
    partners (names, width; hi = lo + width, which differs from the record; the store union ignores lo/hi)."""
    gone = parked(cols, meta, clip)
    drop = {
        k
        for k in cols
        if k[len(OBJ) :] in gone and k.startswith(OBJ) or k[len(OBJ_JOINT) :] in gone and k.startswith(OBJ_JOINT)
    }
    cols = {k: v for k, v in cols.items() if k not in drop}
    names = {k: v for k, v in names.items() if k not in drop}
    out, nf = {}, 0
    for k, (blk, nm) in forces.items():
        kind, rest = k.split(".", 1)
        if rest.rsplit(".", 1)[-1] in gone or rest in gone:  # partner is a parked slot, or the slot itself
            nf += 1
            continue
        if kind == "force_object_partner" and any(n in gone for n in nm):
            keep = [i for i, n in enumerate(nm) if n not in gone]
            blk, nm, nf = blk[:, :, keep], [nm[i] for i in keep], nf + 1
        out[k] = (blk, nm)
    keep_ch = []
    for c in chans:
        tail = c["name"].split("/", 1)[-1]
        if c.get("partner") in gone or tail in gone or tail.rsplit(".", 1)[-1] in gone:
            continue
        if c["kind"] == "force_object_partner" and any(n in gone for n in c["names"]):
            nm = [n for n in c["names"] if n not in gone]
            w = int(c["width"]) // len(c["names"]) * len(nm)
            c = dict(c, names=nm, width=w, hi=int(c["lo"]) + w)
        keep_ch.append(c)
    print(
        f"{OWN_MARK} {clip} · parked slots {sorted(gone)} · dropped object features {len(drop)} · force features "
        f"{nf} · channel rows {len(chans) - len(keep_ch)}",
        flush=True,
    )
    return cols, names, out, keep_ch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("write", "show"))
    for k in ("--hand", "--arm", "--dataset"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "write":
        write(a.hand, a.arm, a.dataset)
    else:
        d = stated(a.hand, a.arm, a.dataset)
        print(f"{MARK} {a.hand} × {a.dataset} K={d['K']} · clips {d['clips']} · at_k {d['at_k']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
