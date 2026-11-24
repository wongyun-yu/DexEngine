"""Rules of the RL training entry (dexengine/rl/cli.py --sequence_dir) about what one run trains on: the roster
(`resolve_roster` — from the hand's folded ledger, or a named frozen roster), the tree rig check (`tree_rig_check`)
and the run key (`run_key` — one run = one key; the training-set selector builds bundles with the same function).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dexengine.core import paths as _P
from dexengine.core.hashing import sha256_file
from dexengine.data import rules as _RR
from dexengine.placement import final_ledger as FL


def _adopted(path: str) -> list[str]:
    """Adopted clips of one folded ledger (file order)."""
    return [r["clip"] for r in FL.read_final_rows(path) if r["status"] == "ADOPTED"]


# ── Roster — the owner is the hand's folded ledger (datasets.ledger_files); no separate roster file. Only frozen
# generations without a ledger folder use a named source of data_layout.NAMED_ROSTERS, and only when the caller names
# it (never picked from the combo).


def ledger_roster(hand: str, dataset: str, emb: str, variant: str = "", root=None) -> tuple[Path, list[str]]:
    """The one folded ledger of hand `hand` on axis ledger_key(dataset, emb, variant) and its adopted clips.
    ValueError unless there is exactly one."""
    key = _RR.ledger_key(dataset, emb, variant)
    files = _RR.ledger_files(key, hand, root=root)
    if len(files) != 1:
        raise ValueError(
            f"hand {hand} · axis {key}: {len(files)} folded ledgers (place "
            f"{_RR.ledger_file(key, hand, root=root)}) — no roster. For a frozen-generation combo name "
            f"its named source (DEXENGINE_ROSTER, registered: "
            f"{', '.join(sorted(_P.data_layout.NAMED_ROSTERS))})"
        )
    return files[0], _adopted(str(files[0]))


def named_roster(name: str) -> tuple[list[Path], list[str], dict]:
    """Named roster source (data_layout.NAMED_ROSTERS[name]) — (list files, clip union in file order, registry
    entry). ValueError when the file count or unique clip count differs from the registry or a clip is in two
    files."""
    spec = _P.data_layout.NAMED_ROSTERS.get(name)
    if spec is None:
        raise ValueError(
            f"unknown roster source {name!r} — registered: {', '.join(sorted(_P.data_layout.NAMED_ROSTERS))}"
        )
    d = _P.layer_path("views", spec["dataset"])
    files = sorted(d.glob(spec["files"]))
    clips = [ln.strip() for f in files for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(files) != spec["n_files"] or len(clips) != spec["n_clips"] or len(set(clips)) != len(clips):
        raise ValueError(
            f"roster source {name}: {d}/{spec['files']} files {len(files)} (registered "
            f"{spec['n_files']}) · clips {len(clips)} (unique {len(set(clips))}, registered "
            f"{spec['n_clips']}) — differs from the registry"
        )
    return files, clips, spec


def _outside(names, allowed, what: str) -> None:
    bad = sorted(set(names) - set(allowed))
    if bad:
        raise ValueError(f"{what} {len(bad)}: {', '.join(bad[:10])}{' …' if len(bad) > 10 else ''}")


def _roster_source(source, combo, dataset, emb, hand, variant, root):
    """Clips of the roster source — a named roster (its combo/dataset/emb must match) or the hand's ledger — with its
    label and ledger file hashes."""
    if source:
        files, clips, spec = named_roster(source)
        got = (combo, dataset, emb)
        want = (spec["combo"], spec["dataset"], spec["emb"])
        if got != want:
            raise ValueError(f"roster source {source} is for (combo, dataset, emb) = {want} — this run is {got}")
        src, ledgers = f"named:{source}", [(str(f), sha256_file(f)) for f in files]
    else:
        lp, clips = ledger_roster(hand, dataset, emb, variant, root=root)
        src, ledgers = f"ledger:{lp}", [(str(lp), sha256_file(lp))]
    return clips, src, ledgers


def _train_trees(seq_list, have, uni, src, is_view):
    """The trees to train and the mode that chose them (all · list · view · folder); every entry must exist on both
    sides."""
    if seq_list == "all":
        mode, train = "opt-out=all", sorted(have)
    elif seq_list:
        want = [
            _P.tree_name(ln.strip()) for ln in Path(seq_list).read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        if len(want) != len(set(want)):
            raise ValueError(f"list {seq_list} has {len(want) - len(set(want))} duplicate entries")
        _outside(want, uni, f"entries of list {seq_list} outside the roster source ({src}) —")
        _outside(want, have, f"entries of list {seq_list} missing from the sequence folder —")
        mode, train = f"list:{seq_list}", sorted(want)
    elif is_view:
        _outside(have, uni, f"view trees outside the roster source ({src}) —")
        mode, train = "view", sorted(have)
    else:
        _outside(uni, have, f"clips of the roster source ({src}) without a tree in the tree root —")
        mode, train = "folder", sorted(have & uni)
    return mode, train


def resolve_roster(
    *,
    combo: str,
    hand: str,
    dataset: str,
    emb: str,
    variant: str,
    trees: list[str],
    is_view: bool,
    seq_dir,
    seq_list: str = "",
    source: str = "",
    root=None,
) -> dict:
    """The trees to train, from the roster owner. `trees` = tree names (with suffix) in the sequence folder
    `seq_dir`. Whether the tree root (for a view, the root its entries point to) is a frozen generation's preserved
    root (paths.named_roster_of) must agree with the source — frozen trees run only from their named source, a named
    source only from its preserved root.

    Universe = ADOPTED of the hand's folded ledger, or `source` (a named source; combo, dataset and emb must match
    its registry entry).
      seq_list=<file>  every entry ∈ universe and present in the folder (else ValueError) → train = the list
      seq_list="all"   explicit opt-out — the whole folder (marker opt-out=all)
      view (is_view)   every view tree ∈ universe (a view is an intended list — it dies, never filters) → the view
      tree root        train = universe ∩ folder (trees outside the universe drop out, e.g. gate rejects); a
                       universe clip missing from the folder is a ValueError
    Returns a record dict (source · ledgers[(path, sha256)] · axis · universe · trees · mode · train · excluded ·
    sha256 · marker)."""
    axis = _RR.ledger_key(dataset, emb, variant)
    frozen = _P.named_roster_of(Path(os.path.realpath(Path(seq_dir) / trees[0])).parent) if trees else ""
    if frozen != source:
        raise ValueError(
            f"tree root = {f'the preserved root of frozen generation {frozen}' if frozen else 'a layer root'}"
            f" · roster source = {source or 'folded ledger'} — frozen trees run only with "
            "DEXENGINE_ROSTER=<that name>, a named source only from its preserved root (paths.roster_root)"
        )
    clips, src, ledgers = _roster_source(source, combo, dataset, emb, hand, variant, root)
    uni = {_P.tree_name(c) for c in clips}
    have = set(trees)
    mode, train = _train_trees(seq_list, have, uni, src, is_view)
    if not train:
        raise ValueError(f"0 trees to train after the roster (source {src} · mode {mode})")
    excluded = sorted(have - set(train))
    body = "".join(f"{_P.clip_name(t)}\n" for t in train)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    marker = (
        f"[ROSTER] hand={hand} axis={axis} source={src} universe={len(uni)} trees={len(have)} mode={mode} "
        f"→ train {len(train)} (trees left out of the folder {len(excluded)}"
        f"{': ' + ', '.join(excluded[:5]) if excluded else ''}{' …' if len(excluded) > 5 else ''}) sha256={sha}"
    )
    return {
        "source": src,
        "ledgers": ledgers,
        "axis": axis,
        "hand": hand,
        "combo": combo,
        "universe": len(uni),
        "trees": len(have),
        "mode": mode,
        "train": train,
        "excluded": excluded,
        "sha256": sha,
        "marker": marker,
        "body": body,
    }


def write_roster_record(rec: dict, path) -> None:
    """The roster record a run writes (key=value head lines + `---` + one clip name per line) — atomic
    (tmp → replace; the tmp name is per process, so two sessions on one experiment name do not clash)."""
    p = Path(path)
    head = [f"source={rec['source']}", f"hand={rec['hand']} combo={rec['combo']} axis={rec['axis']}"]
    head += [f"ledger={lp} sha256={h}" for lp, h in rec["ledgers"]]
    head += [
        f"mode={rec['mode']}",
        f"universe={rec['universe']} trees={rec['trees']} excluded={len(rec['excluded'])}",
        f"n={len(rec['train'])}",
        f"sha256={rec['sha256']}",
        "---",
    ]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    tmp.write_text("\n".join(head) + "\n" + rec["body"], encoding="utf-8")
    os.replace(tmp, p)


#: The rig identity key a tree states (written by the base adapter into the base task_info, carried by the bake).
TREE_RIG_KEY = "hand_asset_sha256_xml3"


def tree_rig_check(*, hand: str, seq_dir, trees: list[str], roster_source: str) -> str:
    """The training trees' rig identity must equal the run hand's registered rig (paths.hand_rig_sha —
    RETARGET_HAND_ASSETS[<spec dir>].sha256_xml3).

    Why: rewards (dexengine.rl.tasks.mdp.rewards.contact fingertip points) and observations read the CURRENT HandSpec
    sites, while a tree's qpos was solved with the rig that built it; after a rig change (e.g. a moved fingertip site)
    old trees would mix two generations in one run without failing anywhere. Rules (independent of the hand name):
      · key present and different from the registry   → ValueError (mixed generations · another hand's tree)
      · key missing (frozen generation before stamps)   → accepted only when the caller named a named roster source
                                                          (`named:` — data_layout.NAMED_ROSTERS); ValueError for a
                                                          ledger roster
      · hand or rig not registered                       → ValueError
    Returns one marker line `[TREE_RIG] …`. Tree names are the names inside the sequence folder (`trees`)."""
    try:
        spec = _P.hand_spec_dir(hand)
        reg = _P.hand_rig_sha(hand)
    except KeyError as e:
        raise ValueError(f"no registered rig for hand {hand!r} — {e}") from None
    if not trees:
        raise ValueError("0 trees to check")
    bad, unstamped = [], []
    for t in trees:
        p = Path(seq_dir) / t / "task_info.json"
        try:
            v = json.loads(p.read_text(encoding="utf-8")).get(TREE_RIG_KEY)
        except (OSError, ValueError) as e:
            raise ValueError(f"{p}: cannot read task_info — {type(e).__name__}: {e}") from None
        if v is None:
            unstamped.append(t)
        elif v != reg:
            bad.append((t, str(v)))
    if bad:
        seen = sorted({v[:12] for _, v in bad})
        raise ValueError(
            f"rig identity {seen} of {len(bad)}/{len(trees)} trees ≠ registered rig {reg[:12]} of hand "
            f"{hand} (first tree {bad[0][0]}) — built with another rig generation. Run this hand's "
            f"current trees, or reproduce the old generation from its code tree"
        )
    if unstamped and not roster_source.startswith("named:"):
        raise ValueError(
            f"{len(unstamped)}/{len(trees)} trees have no rig identity ({TREE_RIG_KEY}) (first tree "
            f"{unstamped[0]}) — roster source {roster_source} is not a frozen-generation roster. "
            f"Unstamped frozen generations run only with a named roster source "
            f"(DEXENGINE_ROSTER, registered: {', '.join(sorted(_P.data_layout.NAMED_ROSTERS))})"
        )
    return (
        f"[TREE_RIG] hand={hand} spec={spec} registry={reg[:12]} trees={len(trees)} "
        f"match={len(trees) - len(unstamped)} unstamped={len(unstamped)}"
        f"{f' (frozen generation · {roster_source})' if unstamped else ''}"
    )


#: The training entry's run axes, in key order — dexengine/rl/cli.py refuses one run whose clips differ on any of
#: them (shared object · articulated main object per side · aerial scene · idle side).
RUN_AXES = ("idle", "shared", "artic", "scene")


def run_group(lay) -> dict:
    """One tree's values on RUN_AXES, read from its qpos_layout.Layout. artic = `<side>:<joint kind>` per
    articulated main object (a run spawns one ArticulationCfg kind per slot — artic_parts.require_one_kind) ·
    "rigid" when none."""
    arts = {s: lay.main_articulation(s) for s in _P.qpos_layout.SIDES}
    return {
        "idle": ",".join(lay.idle_sides()) or "none",
        "shared": int(bool(lay.shared_object)),
        "artic": ",".join(f"{s}:{a['type']}" for s, a in arts.items() if a) or "rigid",
        "scene": lay.scene,
    }


def run_group_key(g: dict) -> str:
    """`idle=…|shared=…|artic=…|scene=…` — the printable key of a run_group (one run = one key)."""
    return "|".join(f"{k}={g[k]}" for k in RUN_AXES)


def run_key(axis: str, g: dict) -> str:
    """One run = one key — what may share one training run.

    key = the tree root (ledger axis `axis` = ledger_key(dataset, emb, variant)) + the run group on RUN_AXES. The
    training-set selector (train_set/select.py) pools on it and the training entry (dexengine/rl/cli.py) requires
    exactly one value per run; both call this function. The root part is there because the entry takes one tree
    root per run (--sequence_dir)."""
    return f"root={axis}|{run_group_key(g)}"


def one_run(axis: str, clips: list[str], groups: list[dict]) -> tuple[str, dict]:
    """The one run key of `clips` (with their run_group `groups`) — ValueError naming every key with its count and
    examples when there is more than one (the entry dies on it; the selector never builds such a list)."""
    if len(clips) != len(groups) or not clips:
        raise ValueError(f"run key: clips {len(clips)} · groups {len(groups)} — empty or different lengths")
    by: dict[str, list[str]] = {}
    for c, g in zip(clips, groups, strict=True):
        by.setdefault(run_key(axis, g), []).append(c)
    if len(by) != 1:
        desc = " · ".join(f"{k} {len(v)} (e.g. {v[:2]})" for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])))
        raise ValueError(f"{len(by)} run keys — one run = one key (train_ready_util.run_key): {desc}")
    key = next(iter(by))
    return key, groups[0]


def run_flags(g: dict, side: str, bimanual: bool) -> dict:
    """The entry's run statements, derived from ONE run group.

    idle_side ("" = both active) · shared · art_active = the articulated flag of the `side` object slot · art_left =
    the left slot's (bimanual) · aerial. An idle side's slot carries the ACTIVE side's object as a placeholder, so
    its articulated flag is the active side's. A one-hand run reads only the `side` slot and the scene."""
    art = {t.split(":")[0] for t in g["artic"].split(",")} - {"rigid"}  # sides only — the kind is the key's
    aerial = g["scene"] == "aerial"
    if not bimanual:
        return {"idle_side": "", "shared": False, "art_active": side in art, "art_left": False, "aerial": aerial}
    idle = "" if g["idle"] == "none" else g["idle"]
    other = {"right": "left", "left": "right"}
    slot_active = side if idle != side else other[side]
    slot_left = "left" if idle != "left" else "right"
    return {
        "idle_side": idle,
        "shared": bool(int(g["shared"])),
        "art_active": slot_active in art,
        "art_left": slot_left in art,
        "aerial": aerial,
    }
