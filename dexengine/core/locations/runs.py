"""Run layers (per-hand RL and data runs): run_path / run_glob, RL bundle and chunk ids, clip lists, the raw
fields of a clip id and the stage banner."""

from __future__ import annotations

import os
import pathlib
import re
from pathlib import Path

from dexengine.core.locations.hands import HAND_IDS, HAND_SPEC_DIR
from dexengine.core.locations.layers import layer_path
from dexengine.core.locations.modules import data_layout
from dexengine.core.locations.names import (
    FIELD_SEP,
    TAKE_SEP,
    WT,
    clip_id_from_raw,
    is_clip_id,
    known_suffixes,
    make_clip_id,
    parent_clip,
)
from dexengine.core.locations.roots import run_root


def raw_from_clip_id(clip_id: str, dataset: str = "taco", raw_root=None) -> tuple[str, str]:
    """clip_id → the **MANO raw location** `(triplet directory name, take)`. The inverse of
    `clip_id_from_raw`.

    Normalisation (N4: non-alphanumeric→`-`) is lossy, so it cannot be undone from the string alone
    ("pour-in-some" ← "pour in some"). So nothing is undone: instead the raw directory is scanned
    for **the entry that produces the same clip_id in the forward direction** — one rule
    (`clip_id_from_raw`) serves both directions, so it fits new names too and no table is kept.
    """
    cid = canonical_clip(clip_id, dataset)
    if dataset != "taco":
        raise ValueError(f"no raw reverse-lookup adapter for dataset: {dataset}")
    root = pathlib.Path(raw_root) if raw_root else layer_path("raw", "taco") / "Hand_Poses"
    if not root.is_dir():
        raise FileNotFoundError(f"TACO raw Hand_Poses missing: {root}")
    task, _, take = cid.partition(TAKE_SEP)
    hits = []
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith("("):
            continue
        try:
            if clip_id_from_raw(d.name, take, dataset) == cid:
                hits.append(d.name)
        except ValueError:
            continue
    if len(hits) != 1:
        raise KeyError(
            f"raw reverse lookup failed: {clip_id!r} → candidates {hits} (exactly one must exist under Hand_Poses)"
        )
    if not (root / hits[0] / take).is_dir():
        raise KeyError(f"raw take directory missing: {root / hits[0] / take}")
    return hits[0], take


# ── Run layers (per-hand directory scheme) — the table is dexengine.core.layout.RUN_LAYERS; this is the window ──
def run_path(layer: str, hand: str = "", *, root=None, **axes: str) -> Path:
    """**The one assembler** of run output paths. A pure function without I/O — shells call it often.

    · A hand layer's `hand` accepts only a **registered hand id**. A spec directory (`wuji`) is a KeyError —
      the run and doc layers have one axis, the hand id (one axis, one meaning: accepting both names would give the
      same hand two roots).
    · Hand-agnostic layers (`exp_shared`·`exp_v2`) take no `hand` (dexengine.core.layout.render_run refuses it).
    · `root` = the output tree (default run_root() — this code tree WT for a canonical data root · the data root
      itself for a generation or folder root, roots.run_root_of). Pass it only to **read** a canonical checkout's
      output from a worktree (same meaning as ledger_files(root=)).
    The owner-mark check before a write is not done here — a write guard outside the package does it."""
    if layer in data_layout.RUN_LAYERS and hand not in HAND_SPEC_DIR:
        raise KeyError(
            f"run layer {layer}: hand {hand!r} is not a registered hand id (registered: {list(HAND_IDS)}). "
            "Pass the hand id, not the spec directory — there is no default"
        )
    _check_run_axes(layer, axes)
    rel = data_layout.render_run(layer, hand, **axes)
    return Path(root if root is not None else run_root()) / rel


def run_glob(layer: str, hand: str = "*", *, root=None, **axes: str) -> str:
    """A **read-only glob** of one run layer — given axes are filled, missing placeholders become `*` (hand `*` = every
    hand). Write places are built only by run_path (never by the glob). A given axis value is one path segment (no
    empty value · `/` · `..` — the same rule as render_run).
    e.g. run_glob("run_collect_work") → <run_root>/overfit/hands/*/collect/work/*__*"""
    tpl = data_layout.RUN_SHARED[layer][0] if layer in data_layout.RUN_SHARED else data_layout.RUN_LAYERS[layer]
    names = set(re.findall(r"\{([a-z]+)\}", tpl))
    vals = dict(axes, **({"hand": hand} if "hand" in names else {}))
    if set(axes) - names:
        raise KeyError(f"{layer}: axes not in this layer {sorted(set(axes) - names)} (template {tpl})")
    for k, v in vals.items():
        if not v or "/" in v or v in (".", ".."):
            raise ValueError(f"{layer}: axis {k}={v!r} — must be one path segment")
    return str(Path(root if root is not None else run_root()) / tpl.format(**{n: vals.get(n, "*") for n in names}))


#: Shape (rule) of RL campaign and bundle ids — one path segment starting with a letter or digit (segment names
#: starting with `_` are reserved by the layer).
RL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
#: Separator between run-name fields (dexengine.rl.campaign.run_name = V2C_<campaign>__<hand>_<bundle>_s42). A
#: campaign id may not contain it or '.', so one campaign's run name + "_" can never be the prefix of another
#: campaign's run name (finalisation treats <RUN>_* as that run's files).
RL_CAMPAIGN_SEP = "__"


def _check_run_axes(layer: str, axes: dict) -> None:
    """Rules for run-layer axis values: the `dataset` axis takes only registered dataset names (DATASETS — the
    mixed-bundle segment name is refused too); `campaign`·`bundle` of the RL layers (`run_rl_*`) follow RL_ID_RE.
    A violation dies (no new segment is built from an unknown name)."""
    ds = axes.get("dataset")
    if ds is not None and ds not in data_layout.DATASETS:
        raise KeyError(
            f"run layer {layer}: dataset {ds!r} is not registered"
            f" (dexengine.core.layout.DATASETS {sorted(data_layout.DATASETS)})"
            + (" — rl_bundle_dir picks the mixed bundle's segment by rule" if ds == data_layout.RL_MIXED else "")
        )
    if layer.startswith("run_rl_") or layer == "run_collect_work":  # collect piece key = RL bundle key
        for k in ("campaign", "bundle"):
            v = axes.get(k)
            if v is not None and not RL_ID_RE.fullmatch(v):
                raise ValueError(f"run layer {layer}: {k}={v!r} — outside the shape {RL_ID_RE.pattern}")
        if axes.get("campaign") is not None:
            check_rl_campaign(axes["campaign"])


def check_rl_campaign(c: str) -> str:
    """The RL campaign id rule — matches RL_ID_RE and holds no RL_CAMPAIGN_SEP·'.' (the first field of a run name,
    dexengine.rl.campaign.run_name). A violation is a ValueError. Layer segments and run names both read this one
    function."""
    if not RL_ID_RE.fullmatch(c) or RL_CAMPAIGN_SEP in c or "." in c:
        raise ValueError(
            f"campaign id {c!r} — must match {RL_ID_RE.pattern} and hold no '{RL_CAMPAIGN_SEP}'·'.' "
            "(it is the first field of a run name; "
            "with one inside it could become the prefix of another campaign's run name)"
        )
    return c


def rl_bundle_dir(hand: str, campaign: str, datasets: dict[str, int], bundle: str, *, root=None) -> Path:
    """**The one selection rule** for an RL campaign bundle segment: a single member dataset → `run_rl_bundle`
    (<campaign>/<dataset>/<bundle>), two or more → `run_rl_mixed` (<campaign>/<RL_MIXED>/<bundle>). `datasets` =
    {dataset: clip count of that dataset} (owner `dexengine.rl.cell_files.bundle_datasets`). An empty
    composition, a non-positive count or an unknown dataset (all checked for mixed too) dies."""
    if not datasets or any(not isinstance(n, int) or n <= 0 for n in datasets.values()):
        raise ValueError(f"RL bundle {bundle}: dataset composition {datasets!r} — empty or a non-positive count")
    for ds in datasets:
        _check_run_axes("run_rl_bundle", {"dataset": ds})
    if len(datasets) == 1:
        return run_path(
            "run_rl_bundle", hand, root=root, campaign=campaign, dataset=next(iter(datasets)), bundle=bundle
        )
    return run_path("run_rl_mixed", hand, root=root, campaign=campaign, bundle=bundle)


def rl_cell_of(path, root=None) -> tuple[str, str, dict] | None:
    """(hand, layer, axes) when the path **is an RL bundle segment itself** (`run_rl_bundle` · `run_rl_mixed`), else
    None. The rule is read from the layer templates (placeholder = one segment) and accepted only when run_path,
    rebuilt from the found axes, gives the same path — an unknown hand or dataset or an id outside the shape is
    refused by run_path, so None (a rule, not a name table). The judge sidecar checks RLJUDGE_OUT_DIR with it."""
    base = Path(root if root is not None else WT).resolve()
    p = Path(os.path.abspath(path))
    try:
        rel = p.relative_to(base)
    except ValueError:
        try:
            rel = p.resolve().relative_to(base)  # a symlinked tree (the same two tries as hand_of_run_path)
        except ValueError:
            return None
    for layer in ("run_rl_bundle", "run_rl_mixed"):
        tpl = data_layout.RUN_LAYERS[layer].split("/")
        if len(tpl) != len(rel.parts):
            continue
        axes, fit = {}, True
        for t, v in zip(tpl, rel.parts, strict=True):
            m = _RUN_PH_ONE.fullmatch(t)
            if m:
                axes[m.group(1)] = v
            elif t != v:
                fit = False
                break
        if not fit:
            continue
        hand = axes.pop("hand")
        try:
            if run_path(layer, hand, root=base, **axes) == base / rel:
                return hand, layer, axes
        except (KeyError, ValueError):
            return None
    return None


_RUN_PH_ONE = re.compile(r"\{([a-z]+)\}")


def hand_of_run_path(path, root=None) -> str | None:
    """Path → the name of the hand (or migration segment) whose hand root it belongs to; None outside the hand
    roots. Judged **by structure only** (`overfit/hands/<h>/…` · `records/hands/<h>/…`) — no name table.
    The root itself (`overfit/hands`) or anything above it is None (it belongs to no single hand)."""
    base = Path(root if root is not None else WT).resolve()
    p = Path(os.path.abspath(path))
    try:
        rel = p.relative_to(base)
    except ValueError:
        try:
            rel = p.resolve().relative_to(base)
        except ValueError:
            return None
    for key in data_layout.RUN_HAND_ROOTS:
        head = data_layout.RUN_LAYERS[key].split("{", 1)[0].rstrip("/").split("/")
        parts = rel.parts
        if len(parts) > len(head) and list(parts[: len(head)]) == head:
            return parts[len(head)]
    return None


# ── Normalisation for joining (invariant c) ─────────────────────────────────
def canonical_clip(name: str, dataset: str = "taco") -> str:
    """**The function everything must pass through before joining by name** (one
    normal form).

    (a) strips the known suffixes and (b) applies the name rule — an old (pre-contract) name comes out as the
    same clip_id the retired name table gave (the rule covers every entry it had). It is idempotent.
    A name that is already a contract name comes back unchanged.
    """
    n = os.path.basename(str(name)).strip()
    # Strip the extension first. Neither a contract name nor a legacy name contains `.`, so
    # cutting at the first dot removes `.npz`, `.adopted.json` and `.pre_f0r.npz` at once (§2.1).
    n = n.split(".", 1)[0]
    for sfx in known_suffixes():
        if sfx and n.endswith(sfx):
            n = n[: -len(sfx)]
            break
    if is_clip_id(n):
        return n
    # Not contract grammar → **apply the rule**. A lookup table would only know yesterday's names — the
    # spaces in an input never seen (new MANO, another dataset) would go straight into the path.
    if TAKE_SEP in n:
        task, take = n.split(TAKE_SEP, 1)
        return make_clip_id(task.split(FIELD_SEP), take.split(FIELD_SEP))
    raise ValueError(
        f"not a clip name (no take separator {TAKE_SEP!r}): {name!r} → {n!r}. "
        "To build one from MANO raw, use clip_id_from_raw()."
    )


def scope_takes(scope_file, dataset: str, emb: str | None = None, input_dir=None) -> list[str]:
    """A range list (one `<ds>\t<emb>/<clip>` per line — the judge-scope format) → the **takes** to retarget.

    Retargeting runs per take, so every clip folds to its parent (parent_clip ∘ canonical_clip).
    emb None → the sorted embs that have rows for `dataset`. Otherwise the folder names **as they are in
    input_dir** (what dexengine.retarget.batch iterates), sorted. It stops (ValueError) on a malformed line, no row for
    the dataset, two input names folding onto one clip_id, or a listed take that input_dir lacks."""
    rows: dict[str, set[str]] = {}
    for i, line in enumerate(pathlib.Path(scope_file).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        p = line.split("\t")
        if len(p) != 2 or "/" not in p[1]:
            raise ValueError(f"{scope_file}:{i} is not of the shape '<ds>\\t<emb>/<clip>': {line!r}")
        if p[0] == dataset:
            e, c = p[1].split("/", 1)
            rows.setdefault(e, set()).add(parent_clip(canonical_clip(c, dataset)))
    if not rows:
        raise ValueError(f"the scope file has no {dataset} row: {scope_file}")
    if emb is None:
        return sorted(rows)
    have: dict[str, str] = {}
    for n in os.listdir(input_dir):
        k = canonical_clip(n, dataset)
        if k in have:
            raise ValueError(f"{dataset}/{emb}: two input names fold onto the same clip_id {k}: {have[k]} · {n}")
        have[k] = n
    want = rows.get(emb, set())
    absent = sorted(want - set(have))
    if absent:
        raise ValueError(
            f"{dataset}/{emb}: {len(absent)} listed takes missing from the input ({input_dir}): {absent[:5]}"
        )
    return sorted(have[k] for k in want)


def read_clip_list(spec, *, dataset: str = "taco", label: str = "clips", quiet: bool = False) -> list[str]:
    """**The only entry point by which a clip list enters the pipeline** (one
    normal form).

    Why there must be exactly one function: with several entry points, one of them will certainly
    skip the normalisation — an old name containing spaces then goes straight into a path, and the
    missing files look like 'clip failures'.

    Accepts: a file path (1 name per line) · a single name · a list of names. Whichever it is, what
    comes back is **a list of contract-grammar clip_ids** (deduplicated, input order kept).

    Nothing is changed silently:
      · one line is printed saying how many were changed (omitted with `quiet=True`).
      · if two different inputs fold onto the same clip_id it **stops** — there is no way to know
        which one was meant, so it does not fall back.
    """
    if spec is None:
        raw: list[str] = []
    elif isinstance(spec, (list, tuple)):
        raw = [str(x).strip() for x in spec if str(x).strip()]
    else:
        text = str(spec)
        pth = pathlib.Path(text)
        if pth.exists() and pth.is_file():
            raw = [
                ln.strip()
                for ln in pth.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")
            ]
        elif "/" in text or text.endswith((".txt", ".tsv")):
            # A string that looks like a path but is not a file means **the file is missing** —
            # interpreting it as a clip name produces a nonsense error such as
            # "this is not a clip name '…/universe.txt'".
            raise FileNotFoundError(f"[{label}] list file missing: {text}")
        else:
            raw = [text.strip()] if text.strip() else []

    out: list[str] = []
    seen: dict[str, str] = {}
    changed = 0
    for r in raw:
        c = canonical_clip(r, dataset)
        changed += c != r
        if seen.get(c, r) != r:
            raise SystemExit(
                f"[{label}] ★input conflict: '{seen[c]}' and '{r}' normalise to the same clip_id "
                f"'{c}' — fix the input list (two names, one clip)"
            )
        seen.setdefault(c, r)
        out.append(c)
    out = list(dict.fromkeys(out))
    if raw and not quiet:
        print(f"[{label}] names normalised {changed}/{len(raw)} · unique clips {len(out)}", flush=True)
    return out


# ── Stage start banner (shows an upstream hole within 10 seconds) ──
def stage_banner(stage: str, reads=(), writes=(), inputs=None, probe=None, require: bool = False) -> int:
    """At the start of a stage, print the **read and write roots** and count **input
    completeness**.

    Why: an upstream hole (e.g. 2,316 wrist caches for 2,317 clips) must be visible within 10 seconds of
    the start, not hours later as a single failure line. Which root is read and where it is written has
    to be printed then too, so that "we looked at a different tree" is noticed at once.

    inputs  = the list of names this stage needs
    probe   = a function taking one name and returning True if it is present
    require = True stops immediately when something is missing (rc 2), False warns and continues
    Returns: the number missing (0 if there is no probe)
    """
    print(f"[{stage}] start", flush=True)
    for r in reads:
        print(f"[{stage}]   read {r}", flush=True)
    for w in writes:
        print(f"[{stage}]   write {w}", flush=True)
    if not (inputs and probe):
        return 0
    missing = [n for n in inputs if not probe(n)]
    print(
        f"[{stage}]   input completeness {len(inputs) - len(missing)}/{len(inputs)}"
        + (
            f" · **missing {len(missing)}**: " + ", ".join(map(str, missing[:5])) + (" …" if len(missing) > 5 else "")
            if missing
            else " · complete"
        ),
        flush=True,
    )
    if missing and require:
        raise SystemExit(2)
    return len(missing)
