"""Per-bundle axis facts of the campaign manifest: the bundle class, the generation tag, the tree facts (scene kinds,
counts), the merge key, the ledger sha256 and one bundle's axis record (bundle_axis). Read by dexengine.rl.manifest
(build · verify)."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from dexengine.core import datasets as RR
from dexengine.core import paths as P
from dexengine.placement import train_ready_util as TRU
from dexengine.placement.train_set import select as ST
from dexengine.rl import manifest_rules as BR

#: A pool below this many clips is flagged `small` (it still gets its own bundle).
SMALL_POOL = 30


#: The scene departures a bundle counts (the canary grouping).
SCENE_KINDS = ("disc", "extra", "support")


def klass(emb: str, g: dict) -> str:
    """Run-group class — a rule on the group (no dataset names): articulated > aerial > one-hand > shared > idle."""
    if g["artic"] != "rigid":
        return "artic"
    if g["scene"] != "table":
        return "aerial"
    if emb != "bimanual":
        return "uni"
    if int(g["shared"]):
        return "shared"
    return "idle" if g["idle"] != "none" else "base"


def gtag(emb: str, g: dict) -> str:
    """Short readable tag of a run group inside one axis (injective on RUN_AXES): uniR · both · idleL · -sh ·
    -artRhLs (side · joint kind initial) · -aerial."""
    if emb != "bimanual":
        head = f"uni{emb[0].upper()}"
    else:
        head = "both" if g["idle"] == "none" else f"idle{g['idle'][0].upper()}"
    parts = [head] + (["sh"] if int(g["shared"]) else [])
    parts += (
        []
        if g["artic"] == "rigid"
        else ["art" + "".join(s[0].upper() + k[0] for s, k in (t.split(":") for t in g["artic"].split(",")))]
    )
    return "-".join(parts + ([] if g["scene"] == "table" else [g["scene"]]))


def bundle_datasets(b: dict) -> dict[str, int]:
    """{dataset: primary clips of that dataset} of one bundle (cell_files.bundle_datasets — the cell rule)."""
    from dexengine.rl import cell_files

    return cell_files.bundle_datasets(b)


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scene_kinds(ti: dict) -> tuple:
    """The SCENE_KINDS one tree carries (task_info only): background objects (case.n_extra > 0) · desk support discs
    (desk_support.discs) · support objects (objects[] role support), in the recorded order (extra, disc, support)."""
    got = {
        "extra": int((ti.get("case") or {}).get("n_extra") or 0) > 0,
        "disc": bool((ti.get("desk_support") or {}).get("discs")),
        "support": any(o.get("role") == "support" for o in ti.get("objects") or []),
    }
    return tuple(k for k in ("extra", "disc", "support") if got[k])


def tree_facts(root: Path, clips: list[str]) -> tuple[str, dict[str, tuple]]:
    """(tree fingerprint = manifest_rules.axis_fp over (clip, sha256(tree task_info.json)), {clip: SCENE_KINDS})."""
    pairs, kinds = [], {}
    for c in sorted(clips):
        raw = (root / P.tree_name(c) / "task_info.json").read_bytes()
        pairs.append((c, hashlib.sha256(raw).hexdigest()))
        kinds[c] = scene_kinds(json.loads(raw))
    return BR.axis_fp(pairs), kinds


def merge_key(emb: str, variant: str, g: dict) -> str:
    """The run key without its tree root: the embodiment, the combo variant and the run group."""
    return f"emb={emb}|variant={variant or '-'}|{TRU.run_group_key(g)}"


def bundle_axis(
    src: dict, target: list[str], hand: str, n: int, block: int, lists: Path, gen: str | None = None
) -> tuple[list[dict], str]:
    """One axis' bundles (the selector's run pools + packing) as manifest rows, and the axis tree fingerprint; list
    files (<id>.txt — the run's clips, DEXENGINE_SEQ_LIST) and tables (<id>.tsv — clip_table) are written to `lists`."""
    if not target:
        return [], ""
    pools = ST.run_pools(target, src["combo"], src["dataset"], src["emb"])
    hd = RR.hand_dir(hand)
    print(ST.obj_key_marker(src["dataset"], src["emb"], hd), flush=True)
    packed = ST.partition(
        pools, lambda c: ST.target_object(c, src["dataset"], src["emb"], hd), ST.key_shape(src["dataset"]), n, block
    )
    root = P.trees_dir(src["combo"], src["dataset"], src["emb"])
    tree_fp, kinds = tree_facts(root, target)
    group_of = {rk: TRU.run_group(P.qpos_layout.read(str(root / P.tree_name(clips[0])))) for rk, clips in pools.items()}
    seen, ids = collections.Counter(), []
    for pb in packed:
        tag = gtag(src["emb"], group_of[pb["run_key"]])
        seen[tag] += 1
        ids.append(f"{src['axis']}.{tag if gen is None else f'{tag}.{gen}'}.{seen[tag]:02d}")
    rows = []
    for bid, pb in zip(ids, packed, strict=True):
        rk, clips = pb["run_key"], pb["clips"]
        g = group_of[rk]
        row = {
            "id": bid,
            "axis": src["axis"],
            "dataset": src["dataset"],
            "emb": src["emb"],
            "variant": src["variant"],
            "combo": src["combo"],
            "tree_root": str(root.relative_to(P.main_root())),
            "side": "both" if src["emb"] == "bimanual" else src["emb"],
            "run_key": rk,
            "run_group": g,
            "klass": klass(src["emb"], g),
            "pool": len(pools[rk]),
            "small": len(pools[rk]) < SMALL_POOL,
            "merge_key": merge_key(src["emb"], src["variant"], g),
            "n": len(clips),
            "n_primary": len(pb["prim"]),
            "objects": pb["objects"],
            "families": pb["families"],
            "scene_kinds": {k: sum(k in kinds[c] for c in clips) for k in SCENE_KINDS},
            "list": f"lists/{bid}.txt",
            "table": f"lists/{bid}.tsv",
            "clips": clips,
            "dups": {c: ids[o] for c, o in pb["dups"].items()},
            "keys": pb["keys"],
        }
        (lists / f"{bid}.txt").write_text("".join(c + "\n" for c in clips), encoding="utf-8")
        (lists / f"{bid}.tsv").write_text(BR.clip_table(row), encoding="utf-8")
        rows.append(row)
    return rows, tree_fp
