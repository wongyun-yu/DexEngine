"""The `--what` answers of the command line, grouped by topic."""

from __future__ import annotations

import json
import sys

from dexengine.core.locations.hands import (
    combo_for,
    hand_id,
    hand_spec_dir,
)
from dexengine.core.locations.layers import (
    default_src_layer,
    entry_layer_roots,
    gate_table,
    layer_path,
    meta_dir,
    ref_base_dir,
    roster_root,
    taco_source_root,
)
from dexengine.core.locations.modules import data_layout
from dexengine.core.locations.names import PLACEMENT_SFX, TREE_SFX, clip_name, ik_suffix, tree_name
from dexengine.core.locations.roots import (
    f0_root,
    main_root,
    mano_model_dir,
    output_root,
    owner_root,
    pool_root,
    reread_stage_root,
    retarget_stage_root,
)
from dexengine.core.locations.runs import (
    canonical_clip,
    scope_takes,
)
from dexengine.core.locations.trees import ClipPaths, canon_count, sync_dirs

WHAT = {
    "clip": lambda P: P.clip,
    "tree": lambda P: P.tree,
    "placement": lambda P: P.placement,
    "ik_right": lambda P: P.ik("right"),
    "ik_left": lambda P: P.ik("left"),
    "wrist_right": lambda P: P.wrist("right"),
    "wrist_left": lambda P: P.wrist("left"),
    "f0_right": lambda P: P.f0_wrist("right"),
    "f0_left": lambda P: P.f0_wrist("left"),
    "tree_qk": lambda P: P.tree_qk,
    "tree_dir": lambda P: P.tree_dir,
    "contact_right": lambda P: P.contact("right"),
    "contact_left": lambda P: P.contact("left"),
}


def need(a, *names: str) -> bool:
    """A `--what` branch that consumes dataset/emb calls this first. There is no default: a missing
    axis prints which flag to give and the branch returns rc 2 — it never falls back to V1 (taco·bimanual)."""
    miss = [n for n in names if not getattr(a, n)]
    if miss:
        print(
            f"[paths --what {a.what}{' --clip' if a.clip else ''}] {' '.join('--' + n for n in miss)} must be given"
            f" (no default) — datasets {sorted(data_layout.DATASETS)} · emb {list(data_layout.EMBODIMENTS)}",
            file=sys.stderr,
        )
        return False
    return True


def _what_roster_root(a):
    """--what roster-root: the frozen generation root of --roster."""
    try:
        print(roster_root(a.roster, a.layer))
        return 0
    except KeyError as e:
        print(f"[paths --what roster-root] {e}", file=sys.stderr)
        return 2


def _what_compare_root(a):
    """--what compare-root: the data root of a compare generation."""
    try:
        print(data_layout.compare_root(str(main_root()), a.gen))
        return 0
    except KeyError as e:
        print(f"[paths --what compare-root] {e}", file=sys.stderr)
        return 2


def _what_gate_table(a):
    """--what gate-table: the gate table of (--root, --combo, --tag)."""
    if not (a.root and a.combo and a.tag):
        print(
            "[paths --what gate-table] needs all of --root <gate meta root> --combo <combo> --tag <chunk tag>",
            file=sys.stderr,
        )
        return 2
    print(gate_table(a.root, a.combo, a.tag))
    return 0


def what_roots(a, ap) -> int | None:
    """`--what` branch ①: suffixes and roots (sfx · main_root · output_root · pool_root · taco-source-root ·
    roster-root · f0_root · sandbox-layers · retarget-stage · gate-table). rc when handled."""
    # The exit that keeps the shell from **reassembling the suffix** (invariant a).
    #   TREE_SFX=$(python3 dexengine/core/paths.py --what sfx)   # call once, then reuse
    _SFX = {"sfx": TREE_SFX, "placement_sfx": PLACEMENT_SFX, "ik_dir_sfx": ik_suffix()}
    if a.what in _SFX:
        print(_SFX[a.what])
        return 0
    # Canonical roots — the exit that keeps the shell from **re-typing a different default**, as
    # in `${DEXENGINE_MAIN_ROOT:-$WT/../<main checkout>}`. The env-precedence rule is owned by the
    # functions above.
    _ROOTS = {
        "main_root": main_root,
        "pool_root": pool_root,
        "output_root": output_root,
        "taco-source-root": taco_source_root,  # TACO source tree (lift input · reference material)
        "owner-root": owner_root,
    }  # clip-owner ledger root (layout file owner_root, else data root)
    if a.what in _ROOTS:
        print(_ROOTS[a.what]())
        return 0
    if a.what == "roster-root":  # frozen generation root (--roster · --layer base|trees)
        return _what_roster_root(a)
    if a.what == "compare-root":  # data root of a compare generation (next to the canonical root)
        return _what_compare_root(a)
    if a.what == "data-generation":  # generation of this data root (canon, or its record JSON)
        g = data_layout.data_generation(str(main_root()))
        print("canon" if g is None else json.dumps(g, ensure_ascii=False, sort_keys=True))
        return 0
    if a.what == "sandbox-layers":  # written layers a sandbox run moves — markers/checks read it
        print(" ".join(data_layout.SANDBOX_LAYERS))
        return 0
    if a.what == "f0_root":  # per-dataset root — the axis must be given
        if not need(a, "dataset"):
            return 2
        print(f0_root(a.dataset))
        return 0
    if a.what == "retarget-stage":
        print(retarget_stage_root(a.hand))
        return 0
    if a.what == "reread-stage":
        if not need(a, "dataset"):
            return 2
        print(reread_stage_root(a.dataset))
        return 0
    if a.what == "mano-model-dir":
        print(mano_model_dir())
        return 0
    if a.what == "gate-table":
        return _what_gate_table(a)
    return None


def what_axis(a, ap) -> int | None:
    """`--what` branch ②: the dataset axis (src-layer · ref-base · canon). rc when handled, else None."""
    # Stage 2 source layer and reference universe — the exit that keeps the shell
    # (stage2_ground.sh) from re-typing the "taco means retarget" rule.
    if a.what == "src-layer":
        if not need(a, "dataset"):
            return 2
        print(a.src_layer or default_src_layer(a.dataset))
        return 0
    if a.what == "ref-base":
        if not need(a, "dataset", "emb"):
            return 2
        try:
            print(ref_base_dir(a.dataset, a.hand, a.emb, a.src_layer or None))
        except (KeyError, ValueError) as e:
            print(f"[paths] {e}", file=sys.stderr)
            return 2
        return 0
    if a.what == "scope-takes":
        # Range list → takes (scope_takes). Without --emb: the embs with rows. With --emb: input folder names.
        if not need(a, "dataset", "scope") or (a.emb and not need(a, "input_dir")):
            return 2
        try:
            out = scope_takes(a.scope, a.dataset, a.emb or None, a.input_dir or None)
        except (OSError, ValueError) as e:
            print(f"[RERETARGET_SCOPE] ✗ {e}", file=sys.stderr)
            return 2
        print("\n".join(out))
        return 0
    if a.what == "canon":
        # Names from stdin, one line at a time, into a **joinable form**. Before the shell runs
        # comm/join it pushes both streams through here (invariant c). One process does it all.
        # The shell window is `scripts/_clip_canon.sh` (`canon <dataset>`).
        # The dataset is **required** here: a default (taco) would reach shell wrappers that pass none,
        # and they would join with taco's legacy table.
        if not a.dataset or a.dataset not in data_layout.DATASETS:
            print(
                f"[paths --what canon] give --dataset <dataset> (no default) — got "
                f"{a.dataset or '(none)'}, registered {sorted(data_layout.DATASETS)}",
                file=sys.stderr,
            )
            return 2
        for ln, line in enumerate(sys.stdin, 1):
            t = line.rstrip("\n")
            if not t:
                continue
            try:
                print(canonical_clip(t, a.dataset))
            except ValueError as e:
                # Nothing is let through silently. It names the offending line and stops.
                print(f"[paths --what canon] line {ln}: {e}", file=sys.stderr)
                return 2
        return 0
    return None


def _what_entry_layers(a):
    """--what entry-layers: 'layer<TAB>path' lines and an '#EOF<TAB>count' line."""
    # layer roots the training entry needs (ENTRY_LAYER_ROOTS) — "layer<TAB>path" lines + "#EOF<TAB>line count"
    # (so a truncated output never reads as a pass — a reader compares the counted lines).
    if not need(a, "dataset", "emb", "combo"):
        return 2
    try:
        rows = entry_layer_roots(a.dataset, a.combo, a.emb)
    except (KeyError, ValueError) as e:
        print(f"[paths] {e}", file=sys.stderr)
        return 2
    print("".join(f"{ly}\t{p}\n" for ly, p in rows) + f"#EOF\t{len(rows)}")
    return 0


def _what_layer_paths(a, ap):
    """--what layout|layer: one layer path, or every layer of the dataset."""
    if not need(a, "dataset"):
        return 2
    try:
        # Combo axis: a given one wins, else it is **derived from the hand** (combo_for). If that fails too the
        # axis is left out — a layer that needs a combo dies there with KeyError (no silent default combo).
        _combo = a.combo or ""
        if not _combo and a.hand:
            try:
                _combo = combo_for(a.hand, a.arm)
            except ValueError:
                _combo = ""
        # emb goes into the axes **only when given** — there is no default, so for layers that need emb
        # (retarget · base · trees …) render refuses the missing axis with KeyError (rc 2), and layers without emb
        # (usd · assets_objects …) run without it. The hand axis likewise: hand-independent layers (mano · raw ·
        # INTAKE …) run without a hand, hand layers get KeyError from render (rc 2). A given hand, id or spec
        # directory, becomes the directory; an unregistered one is a KeyError.
        axes = {"hand": hand_spec_dir(hand_id(a.hand))} if a.hand else {}
        if a.emb:
            axes["emb"] = a.emb
        if _combo:
            axes["combo"] = _combo
        if a.clip:
            axes["clip"] = clip_name(a.clip)
            axes["tree"] = tree_name(a.clip)
        if a.what == "layer":
            if not a.layer:
                ap.error("--what layer needs --layer")
            print(layer_path(a.layer, a.dataset, **axes))
            return 0
        for layer in data_layout.LAYERS:
            try:
                print(f"{layer}\t{layer_path(layer, a.dataset, **axes)}")
            except (KeyError, ValueError) as e:  # noqa: PERF203 — each layer prints its own "not applicable" line
                print(f"{layer}\t(not applicable: {str(e).splitlines()[0][:70]})")
        return 0
    except (KeyError, ValueError) as e:
        print(f"[paths] {e}", file=sys.stderr)
        return 2


def what_layout(a, ap) -> int | None:
    """`--what` branch ⑤: layer tables (datasets · layout · layer). rc when handled, else None."""
    # ── Canonical layout window: keeps the shell from joining data/<ds>/… by hand ──
    #   python3 dexengine/core/paths.py --what layer --layer retarget --dataset show3d --hand wuji --emb bimanual
    #   python3 dexengine/core/paths.py --what layout --dataset taco       # table of where each layer really stores
    if a.what == "datasets":
        print(" ".join(f"{k}:{v.status}" for k, v in sorted(data_layout.DATASETS.items())))
        return 0
    if a.what == "embodiments":  # every emb axis value (space separated — shell loops)
        print(*data_layout.EMBODIMENTS)
        return 0
    if a.what == "entry-layers":
        return _what_entry_layers(a)
    if a.what in ("layout", "layer"):
        return _what_layer_paths(a, ap)
    return None


def _what_clip(a, ap):
    """--what <kind> | all for one clip (needs --clip, --dataset and --emb)."""
    if not a.clip:
        ap.error("--clip is required (except for dirs/counts/meta-dir)")
    if not need(a, "dataset", "emb"):
        return 2
    P = ClipPaths(a.clip, a.combo, a.hand, a.dataset, a.emb)
    if a.what == "all":
        for k in sorted(WHAT):
            print(f"{k}\t{WHAT[k](P)}")
        return 0
    if a.what not in WHAT:
        ap.error(f"invalid --what value: {a.what}")
    print(WHAT[a.what](P))
    return 0


def what_combo(a, ap) -> int:
    """`--what` branch ⑥: the rest, which needs a combo (dirs · counts · meta-dir · all · WHAT of --clip). The
    last branch — always rc."""
    # These branches **require** a combo. When empty it is derived from the hand; if that fails too, it dies.
    if a.what in ("dirs", "counts", "meta-dir", "all") or a.clip:
        if not a.combo:
            try:
                a.combo = combo_for(a.hand, a.arm) if a.hand else ""
            except ValueError as e:
                print(f"[paths] {e}", file=sys.stderr)
                return 2
        if not a.combo:
            print(
                f"[paths] --what {a.what} needs --combo (or a --hand it can be derived from) — "
                f"there is no default combo",
                file=sys.stderr,
            )
            return 2

    if a.what == "dirs":
        for sub, pat in sync_dirs(a.combo):
            print(f"{sub}\t{pat}")
        return 0
    if a.what == "counts":
        if not need(a, "dataset"):
            return 2
        for sub, n in canon_count(a.combo, a.dataset):
            print(f"{sub}\t{n}")
        return 0
    if a.what == "meta-dir":  # combo ledger folder (bake failure reasons …) — not the
        if not need(a, "dataset"):  #   training roster (that is the per-hand folded ledger)
            return 2
        print(meta_dir(a.combo, a.dataset))
        return 0

    return _what_clip(a, ap)
