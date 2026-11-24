#!/usr/bin/env python3
"""Training-scene comparison — the one owner of "is the scene a checkpoint was trained in the scene this code builds
for the clip now?". Extra object slots (dexengine/data/objects/extra_slots.py) change the training scene; a run
states its scene in the run ledger <run folder>/extra_slots.txt (LEDGER — written by dexengine.rl.env_setup.build).

  ledger forms  no file     trained with no extra slot (the feature off, or a run older than it)
                value line  `1` only — a run of the first slot rule: its spawn list is unknown here (a case this rule
                            does not have: the comparison dies)
                full        `1` · `rule<TAB>rule version` · `clip<TAB>name<TAB>signature` per clip (signature below
                            · name = the clip's **tree folder name**, paths.tree_name — the env names clips by
                            sequence folder; the owner ledger's clip name has no suffix, so the CLI asks by tree name)
                            · `collide<TAB>kind<TAB>partners` per slot kind (partners — the collision table row the
                            authoring applied) · `authored<TAB>…` (the authoring's measured counts of that run — a
                            record; counts follow the env count, so entries do not compare them)
                            · `drop<TAB>name<TAB>object:reason,…` per clip that has any — the grazed still extras
                            the place rule (dexengine/placement/scene/place_rule.py) kept out of that clip's training
                            scene
                            (drop_text). A record, not a second comparison: the signature already
                            differs when the spawn list does. The collection render reads these names and draws
                            none of them — it does not compute the rule again.
  same scene    no file: the clip stands no slot that stands a collider now (`hits` — the kinds whose collision table
                row turns a partner ON: the robot or the main objects, collision_presets.table_has_collider; a slot
                that collides with nothing the policy moves is the same physics as no slot). full: the clip's
                signature in the ledger equals its signature now AND the collision partners of the kinds the clip
                stands are the ledger's (collide_changed — a table cell switched since the run is another physics
                for the clips that stand that kind, and only for them; a clip the ledger does not hold was not
                trained by that run — changed).
  readers       dexengine.rl.env_setup.build.build_extra_slots (every entry that loads a checkpoint: a changed clip
                dies there)
                · the CLI below (the owner checkpoints' table: a clip keeps
                its old checkpoint only when its content AND its training scene are the same)
  CLI           <retarget venv> -m dexengine.rl.launch.train_scene --hand H --owner <rl_clip_owner.tsv> --datasets a,b
                --out <tsv> [--jobs 4]   every canon clip of those datasets — and every held clip whose row still
                names its checkpoint (an issue list holds it: the scene handoff
                is declared for it before the list lets it go) — axis · clip · scene (same | changed | no_tree) ·
                spawn_key · why. Marker [TRAIN_SCENE] per dataset; rc 0 (the table is the result).
                A canon row whose axis is not an axis of the hand now (a dataset rebuilt without it — HOT3D's
                aerial axes) has no tree to compare: no_tree, why `no axis` (split_axes — a dataset with owner
                rows and no axis at all now is a wrong input and dies).
The module top is standard library only (the queue tools read TABLE rows with it); the CLI imports the slot rule
and the campaign manifest sources (dexengine.rl.manifest).
"""

from __future__ import annotations

import hashlib
import os

from dexengine.core.errors import InputMissingError

TAG = "[TRAIN_SCENE]"
LEDGER = "extra_slots.txt"
SAME, CHANGED, NO_TREE = "same", "changed", "no_tree"
HEAD = ("axis", "clip", "scene", "spawn_key", "why")
ROBOT = ("hand", "arm")


def ledger_of(ckpt: str) -> str:
    """Run ledger of a checkpoint: <run folder>/LEDGER (run folder = the parent of checkpoints/)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))), LEDGER)


def slots_value(ckpt: str) -> str:
    """Extra-slot value "0"|"1" the checkpoint trained with; no ledger = "0" (old checkpoints trained with it off)."""
    rec = read_ledger(ledger_of(ckpt))
    v = "0" if rec is None else rec["value"]
    if v not in ("0", "1"):
        raise ValueError(f"{TAG} ledger {ledger_of(ckpt)} = {v!r} — only 0|1")
    return v


def read_ledger(path: str) -> dict | None:
    """The run ledger → {value, rule (None = value line only), clips {clip: signature}} · None = no file. A line
    this reader does not know raises ValueError."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    out = {
        "value": lines[0].strip() if lines else "",
        "rule": None,
        "clips": {},
        "collide": {},
        "authored": None,
        "drops": {},
    }
    for ln in lines[1:]:
        p = ln.split("\t")
        if p[0] == "rule" and len(p) == 2:
            out["rule"] = p[1]
        elif p[0] in ("clip", "collide") and len(p) == 3:
            out[p[0] + ("s" if p[0] == "clip" else "")][p[1]] = p[2]
        elif p[0] == "authored" and len(p) == 2:
            out["authored"] = p[1]
        elif p[0] == "drop" and len(p) == 3:
            out["drops"][p[1]] = p[2]
        elif ln.strip():
            raise ValueError(f"ledger {path}: unknown line {ln!r} — only rule, clip, collide, authored and drop lines")
    return out


def drop_text(pairs: list) -> str:
    """The `drop` line's third field — [(object name, reason)] the place rule kept out of a clip's training scene →
    `name:reason,name:reason` (sorted by name). An object name or a reason holding `:` · `,` · a tab raises."""
    for n, w in pairs:
        if any(ch in f"{n}{w}" for ch in ":,\t\n"):
            raise ValueError(f"object kept out of the spawn {n!r} or its reason {w!r} holds a separator")
    return ",".join(f"{n}:{w}" for n, w in sorted(pairs))


def run_ledger(ckpt_path: str) -> str:
    """The ledger path of a checkpoint (<run folder>/checkpoints/<file> → <run folder>/LEDGER)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path))), LEDGER)


def signature(slots: list, spawn_key: str) -> str:
    """One clip's spawn-list signature — `<spawn_key>:<sha1[:12] of the name · object · kind list>` (slots = the slot
    rows in slot order · spawn_key = extra_slots.spawn_key(slots), the per-kind counts)."""
    body = "|".join(f"{r['name']}={r['obj_id']}={r['kind']}" for r in slots)
    return f"{spawn_key}:{hashlib.sha1(body.encode()).hexdigest()[:12]}"


def partners(table: dict) -> str:
    """One collision table row as the ledger writes it — `hand ON · arm ON · objects OFF` (keys sorted by name)."""
    return " · ".join(f"{p} {table[p]}" for p in sorted(table))


def sig_kinds(sig: str, kinds: tuple) -> set:
    """The slot kinds a signature counts at least once (its spawn_key part `<n>/<n>/<n>` is in `kinds` order)."""
    n = sig.split(":", 1)[0].split("/")
    if len(n) != len(kinds):
        raise ValueError(f"signature {sig!r} has {len(n)} kinds != {kinds}")
    return {k for k, v in zip(kinds, n, strict=True) if int(v)}


def collide_changed(led: dict, clips: list, now_sigs: dict, now_collide: dict, kinds: tuple) -> list:
    """The slot kinds whose collision partners differ between a full ledger and now — only the kinds the compared
    clips stand (then or now). A full ledger without a kind's `collide` line counts as different."""
    used = {k for c in clips for s in (led["clips"][c], now_sigs[c]) for k in sig_kinds(s, kinds)}
    # The scene-rule kinds (spawn_key counts them in a base bucket): compared when both sides state them. One side only
    # needs no comparison: the signature hash names each slot's kind, so a clip whose slots changed to or from a
    # scene-rule kind already differs in its signature.
    from dexengine.data.objects.extra_slots import SCOPED_KINDS

    used |= {k for k in SCOPED_KINDS if k in led["collide"] and k in now_collide}
    return sorted(k for k in used if led["collide"].get(k) != now_collide.get(k))


def robot_kinds(tables: dict) -> set:
    """The slot kinds that collide with the robot — tables = {kind: {hand · arm · objects: ON|OFF}} (the collision
    owner's rows, collision_presets.extra_collide — read, never restated here)."""
    return {k for k, t in tables.items() if any(t[p] == "ON" for p in ROBOT)}


def changed(led: dict | None, clip: str, kinds: list, sig: str, hits: set, now_collide: dict, all_kinds: tuple) -> str:
    """'' = the clip's training scene under the ledger `led` is the scene now · else the reason (module doc: same
    scene). kinds = the kinds of the slots the rule stands for the clip now · sig = its signature now · hits = the
    kinds that stand a collider now · now_collide = {kind: partners(row)} now · all_kinds = the rule's kind order."""
    if led is None:
        bad = sorted(k for k in kinds if k in hits)
        return (
            f"trained without extra slots, now {len(bad)} slots collide with the robot or main objects "
            f"({'/'.join(sorted(set(bad)))})"
            if bad
            else ""
        )
    if led["rule"] is None:
        raise ValueError(
            f"{clip}: an old value-only ledger — the run's spawn list is unknown (this rule has no such case)"
        )
    from dexengine.data.objects.extra_slots import SCOPED_KINDS

    # scene-rule kinds: only the ones this clip stands now
    now_collide = {k: v for k, v in now_collide.items() if k not in SCOPED_KINDS or k in kinds}
    if clip not in led["clips"]:
        return f"clip not in the ledger (a clip the rule {led['rule']} run did not train)"
    if led["clips"][clip] != sig:
        return f"spawn list {led['clips'][clip]} -> {sig} (ledger rule {led['rule']})"
    off = collide_changed(led, [clip], {clip: sig}, now_collide, all_kinds)
    return (
        (
            "collision partners "
            + " · ".join(f"{k} [{led['collide'].get(k, 'no line')}] -> [{now_collide.get(k)}]" for k in off)
            + f" (ledger rule {led['rule']})"
        )
        if off
        else ""
    )


def read_table(path: str) -> dict:
    """The CLI's table → {(axis, clip): scene}. A row twice or a scene outside the three raises ValueError."""
    out: dict = {}
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    if not lines or tuple(lines[0].split("\t")) != HEAD:
        raise ValueError(f"{path}: header is not {HEAD}")
    for i, ln in enumerate(lines[1:], 2):
        c = ln.split("\t")
        if len(c) != len(HEAD) or c[2] not in (SAME, CHANGED, NO_TREE) or (c[0], c[1]) in out:
            raise ValueError(f"{path}: line {i} — {len(c)} cells, scene {c[2] if len(c) > 2 else None!r}, or a repeat")
        out[(c[0], c[1])] = c[2]
    return out


def clip_now(tree: str) -> tuple[list, str, str] | None:
    """(slot kinds, signature, spawn_key) of the slots the rule stands for the tree now (extra_slots.clip_rows —
    slot order = qpos order, as the env builds them) · None = no tree there. Imports the slot rule (numpy)."""
    import json

    from dexengine.data.objects import extra_slots as XS

    f = os.path.join(tree, "task_info.json")
    if not os.path.isfile(f):
        return None
    with open(f, encoding="utf-8") as fh:
        ti = json.load(fh)
    slots = sorted(
        (r for r in XS.clip_rows(tree, ti, None, False) if r["cls"] in XS.SLOTTED), key=lambda r: r["qpos_start"]
    )
    key = XS.spawn_key(slots)
    return [r["kind"] for r in slots], signature(slots, key), key


def measured(r: dict) -> bool:
    """An owner-ledger row the CLI compares (module doc · CLI): canon, or held with the checkpoint that owned it."""
    return r["state"] == "canon" or (r["state"] == "held" and r.get("ckpt_rel") not in (None, "", "-"))


def split_axes(rows: list[dict], src: dict) -> tuple[list[dict], list[dict]]:
    """Owner rows → (rows whose axis is an axis of the hand now, rows whose axis is not). The second kind is a
    dataset rebuilt without that axis: its clips have no tree in the axis the checkpoint was trained in, so they
    are NO_TREE rows (never a carry — the label dies on a carry candidate without a verdict). A dataset that has
    owner rows and no axis at all now is not a rebuild but a wrong input (nothing to compare with) → dies."""
    gone = [r for r in rows if r["axis"] not in src]
    if off := sorted({r["dataset"] for r in gone} - {s["dataset"] for s in src.values()}):
        raise InputMissingError(
            f"{TAG} FAIL datasets {off} of the owner ledger have no axis in the folded ledger now "
            "— no current generation to compare with"
        )
    return [r for r in rows if r["axis"] in src], gone


def _now_state(cp, XS, ds: str) -> tuple:
    """(kinds with a collider, partners per kind, base kinds) of the current tables for dataset `ds`."""
    tables = {k: cp.extra_collide(k, ds=ds) for k in XS.KINDS + XS.SCOPED_KINDS}  # scene-rule kinds: clips with one
    return (
        {k for k, t in tables.items() if cp.table_has_collider(t)},
        {k: partners(t) for k, t in tables.items()},
        tuple(XS.KINDS),
    )


def _work(r: dict, s: dict, root, now: tuple, P) -> tuple:
    """One canon row of the owner ledger (+ its axis' source `s`) → _one's argument. The clip travels by its tree
    folder name (P = paths: tree_name) — the name a run ledger holds it under (module doc: full); the owner ledger's
    own name is kept for the row only."""
    trees = P.layer_path("trees", s["dataset"], combo=s["combo"], emb=s["emb"])
    return r["axis"], r["clip"], P.tree_name(r["clip"]), str(trees), str(root / r["ckpt_rel"]), now


def _one(a: tuple) -> tuple:
    """One canon clip (CLI worker · argument = _work) → a HEAD row. `name` is the tree folder AND the ledger key —
    one value, so the two cannot name the clip differently."""
    axis, clip, name, trees, ckpt, (hits, now_collide, all_kinds) = a
    tree = os.path.join(trees, name)
    now = clip_now(tree)
    if now is None:
        return axis, clip, NO_TREE, "", f"no tree {tree}"
    why = changed(read_ledger(run_ledger(ckpt)), name, now[0], now[1], hits, now_collide, all_kinds)
    return axis, clip, CHANGED if why else SAME, now[2], why


def main(argv=None) -> int:
    import argparse
    import collections
    import csv
    import multiprocessing as mp

    from dexengine.core import capacity
    from dexengine.core import paths as P
    from dexengine.data.objects import extra_slots as XS
    from dexengine.data.objects import scene_rules as SC
    from dexengine.rl import manifest as CM

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hand", required=True)
    ap.add_argument(
        "--owner", required=True, help="owner ledger rl_clip_owner.tsv (the run ledgers of its canon rows are read)"
    )
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=None, help="pool size (default: capacity.plan)")
    a = ap.parse_args(argv)
    cp = P.load_registry("collision_presets")  # the collision table's owner, read through the registry window
    want = set(a.datasets.split(","))
    state = {ds: _now_state(cp, XS, ds) for ds in want}  # the robot cells of the still kinds depend on the dataset
    src = {s["axis"]: s for s in CM.sources(a.hand)}
    with open(a.owner, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if measured(r) and r["dataset"] in want]
    live, gone = split_axes(rows, src)
    root = P.main_root()
    work = [_work(r, src[r["axis"]], root, state[r["dataset"]], P) for r in live]
    step = "dexengine.rl.launch.train_scene"  # capacity.NEEDS key (this module)
    with capacity.sized(step, len(work), a.jobs) as cap, mp.Pool(cap.workers, capacity.worker_init, cap.initargs) as p:
        res = p.map(_one, work, chunksize=16)
    res = sorted(
        res
        + [
            (
                r["axis"],
                r["clip"],
                NO_TREE,
                "",
                f"no axis {r['axis']} — not in the folded ledger of the current generation",
            )
            for r in gone
        ]
    )
    n_gone = collections.Counter(r["dataset"] for r in gone)
    n_held = collections.Counter(r["dataset"] for r in rows if r["state"] == "held")
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\t".join(HEAD) + "\n" + "".join("\t".join(x) + "\n" for x in res))
    os.replace(tmp, a.out)
    ds_of = {r["axis"]: r["dataset"] for r in rows}
    n = collections.Counter((ds_of[x[0]], x[2]) for x in res)
    for ds in sorted(want):
        print(
            f"{TAG} {a.hand} {ds}: owner clips {sum(v for (d, _k), v in n.items() if d == ds)} (held with a "
            f"checkpoint {n_held[ds]}) · training scene same "
            f"{n[(ds, SAME)]} · changed {n[(ds, CHANGED)]} · no tree {n[(ds, NO_TREE)]} (no axis {n_gone[ds]} — "
            f"{sorted({r['axis'] for r in gone if r['dataset'] == ds})}) · rule {XS.RULE} · kinds that stand a "
            f"collider (hit the robot or main objects) "
            f"{sorted(k for k in state[ds][0] if k in XS.KINDS or k in SC.kinds_on(ds))} -> {a.out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
