"""Run layers: per-hand run outputs (ledgers, chunks, logs, RL campaigns, record work areas) and hand-free run
outputs, rendered with `render_run()`."""

from __future__ import annotations

import os
import re

from dexengine.core.errors import InputMissingError

# ══ Run layers — the hand id splits the top level (per-hand directory scheme) ═════════════════════════════════════
# Why: if run outputs (chunks · audit json · campaign folders) mix without a hand axis, one hand deletes or
# overwrites another hand's outputs.
# Rule: run outputs live only under `overfit/hands/<hand>/…` and `records/hands/<hand>/…`. This table alone
# builds the paths (`render_run`); shell and Python get them only from `dexengine.core.paths.run_path()` / the paths
# CLI `--what run-path`. A path string in any other file is blocked by the canon check. A new hand gets all these
# paths from its one hand-registry row.
# Axes: hand      hand **id** (not the spec directory — runs and docs are per hand as people name it). Checked by
#                 dexengine.core.paths.run_path (this module cannot read the hand registry — stdlib only).
#       campaign  campaign id = the campaign's **tag prefix** (TAGP, e.g. G1v · G1ds). Unique per campaign.
#       tag       chunk tag (`<TAGP>_<axis>_<nn>`) — the naming rule is unchanged (fold, stage 2 and the judge trend
#                 read it with regexes).
#       tool      name of the tool that writes the log (reretarget · cases.taxonomy …).
#       dataset   dataset of an RL campaign bundle (a DATASETS name — checked by dexengine.core.paths
#                 rl_bundle_dir / run_path).
#       bundle    RL campaign bundle id (manifest `<ledger axis>.<group label>.<number>`).
#
#: The segment where an RL bundle mixing clips of several datasets lives (instead of a dataset segment). It starts
#: with `_` so it never collides with a dataset name — giving this value as the dataset axis dies in
#: dexengine.core.paths (the one rule that picks the segment is `rl_bundle_dir`).
RL_MIXED = "_mixed"
RUN_LAYERS: dict[str, str] = {
    "run_root": "overfit/hands/{hand}",  # hand root (the owner mark HAND_OWNER lives here)
    "run_campaigns": "overfit/hands/{hand}/campaigns",
    "run_campaign": "overfit/hands/{hand}/campaigns/{campaign}",  # ledgers · queues · logs
    "run_chunks": "overfit/hands/{hand}/chunks",
    "run_chunk": "overfit/hands/{hand}/chunks/{tag}",  # judge · bake chunk
    # remote **attempt place** — a new folder per launch
    "run_chunk_try": "overfit/hands/{hand}/chunks/{tag}/{attempt}",
    # transfer lists <tag>.{txt,txt.canon,judge.txt,f0_failed.txt,paths}
    "run_lists": "overfit/hands/{hand}/lists",
    "run_ledger": "overfit/hands/{hand}/ledger",  # final ledger status_final_<axis>.tsv
    # hand judge carry-over declaration (readers: fold · dexengine.placement.train_ready)
    # — if present, that hand is provisional (hand_provisional)
    "run_hand_carry": "overfit/hands/{hand}/ledger/hand_carry.tsv",
    "run_harvest": "overfit/hands/{hand}/harvest",  # harvest inbox (where remote copies unpack)
    # Rebake backup — where bake_drive really copies the tree and both IK sides when it rebakes a clip of the training
    # list (DEXENGINE_BAKE_BACKUP_DIR). Outside the chunk folder (the harvest deletes chunk folders) · a new folder
    # per attempt (ledger attempt — never overwrites the restore place of an earlier attempt).
    "run_bake_backups": "overfit/hands/{hand}/bake_backup",
    "run_bake_backup": "overfit/hands/{hand}/bake_backup/{tag}/{attempt}",
    "run_logs": "overfit/hands/{hand}/logs",
    "run_log": "overfit/hands/{hand}/logs/{tool}",  # per-tool logs
    "run_render": "overfit/hands/{hand}/render_queue",  # hand page render queue state
    "run_regress": "overfit/hands/{hand}/regress",  # canon-50 regression — previous run kept
    "exp_root": "records/hands/{hand}",  # hand docs root (owner mark HAND_OWNER)
    "exp_audit": "records/hands/{hand}/audit",  # per-hand audit json
    "exp_table": "records/hands/{hand}/tables",  # per-hand tables
    "exp_page": "records/hands/{hand}/page",  # hand page
    "exp_register": "records/hands/{hand}/register",  # hand registration figures and tables
    # RL inputs and stats (pilot lists lists/<set>/ · pair comparisons stats/) — outside the bundle builder
    "run_rl": "overfit/hands/{hand}/rl",
    # RL campaign outputs (runner dexengine/rl/campaign.py · the judge sidecar · render · collection read and write
    # here). Campaign segment = manifest copy · canary results · per-dataset index.
    # Bundle segment = launch record · rc/pid · judge tables/rc · moved training logs and kept checkpoints.
    "run_rl_campaigns": "overfit/hands/{hand}/rl_campaigns",
    "run_rl_campaign": "overfit/hands/{hand}/rl_campaigns/{campaign}",
    # one-dataset bundle · mixed-dataset bundle · index (dataset → bundle · clip count, written by a generator)
    "run_rl_bundle": "overfit/hands/{hand}/rl_campaigns/{campaign}/{dataset}/{bundle}",
    "run_rl_mixed": "overfit/hands/{hand}/rl_campaigns/{campaign}/" + RL_MIXED + "/{bundle}",
    "run_rl_index": "overfit/hands/{hand}/rl_campaigns/{campaign}/by_dataset.tsv",
    # placement-augmentation judge cells (judge mode D region dumps · mode L list judging → region/ · cells/ ·
    # judged/ · ikc/)
    "run_aug_cells": "overfit/hands/{hand}/aug_cells/{campaign}/{bundle}",
    # judge collection (written by `python -m dexengine.rl.cell_tools collect`): one row per bundle ·
    # one per dataset (a mixed bundle has its own row) · one per unique clip (main bundle · duplicate bundles · the best
    # value among bundles that trained the clip — never count a duplicate twice)
    "run_rl_results": "overfit/hands/{hand}/rl_campaigns/{campaign}/results.tsv",
    "run_rl_results_ds": "overfit/hands/{hand}/rl_campaigns/{campaign}/results_by_dataset.tsv",
    "run_rl_results_clips": "overfit/hands/{hand}/rl_campaigns/{campaign}/results_by_clip.tsv",
    # cluster copy prune ledger after harvest (the prune step — rows for *.pt the bundle does not keep · a
    # bundle-end row)
    "run_rl_volume_prune": "overfit/hands/{hand}/rl_campaigns/{campaign}/volume_prune.tsv",
    # Operating place of the public data collection driver (dexengine/record/driver.py) — per run label:
    # progress.json (the one file a monitor reads) · bundle state (bundles/<campaign>__<bundle>/state) · logs ·
    # pidfile · ledger copy. Final data goes to the data layer collect (hand × one dataset).
    "run_collect": "overfit/hands/{hand}/collect/{tag}",
    # Collect work place (temporary — emptied when collection ends): one bundle piece = one collect.py --out (stage1 ·
    # stage2c · seq1c · dataset/ · aug/ · preview/ · cleanup.json). Keyed by bundle (not label) — the first and
    # second pass continue the same piece cache. It is a fixed segment under the same parent as the label segment
    # (run_collect), so render_run refuses a label equal to that segment name (_run_overlap).
    "run_collect_work": "overfit/hands/{hand}/collect/work/{campaign}__{bundle}",
}
#: Env a job sets to give run_collect_work its own root — the work place has exactly one owner job. Local sim jobs
#: share the data root, so they pass the job tree; remote hosts, nodes and clusters
#: do not (their data root is the job tree). The one reader is the record piece work_root (two jobs of one bundle on
#: one place would rewrite stage 1 keys and the cleanup would delete the other job's media) — read the value through
#: collect_work_root_env() (env reads have one home; other modules do not read the env by attribute name).
COLLECT_WORK_ROOT_ENV = "COLLECT_WORK_ROOT"


def collect_work_root_env() -> str:
    """Value of COLLECT_WORK_ROOT_ENV ('' if unset) — interpretation (directory check · default = data root) is the
    record piece work_root."""
    return os.environ.get(COLLECT_WORK_ROOT_ENV, "")


#: Hand roots — where the owner mark lives. An outside write / delete / move guard decides ownership from these two.
RUN_HAND_ROOTS: tuple[str, ...] = ("run_root", "exp_root")
#: Owner mark file name (right under each hand root). Content = `key=value` lines (hand · spec · combo · created · by).
HAND_OWNER_FILE = "HAND_OWNER"

#: Run outputs declared **hand-free** — no hand axis (`hand_free`). The second field of each value owns the writer.
#:   exp_shared   hand-free outputs (MANO · objects · cross-hand comparisons · pipeline lists · layout audits). A file
#:                without a hand name is a hand-free declaration — anything whose value differs per hand belongs in
#:                exp_audit / exp_table instead.
#:   run_rl_loops operating files of the RL campaign launch loop (dexengine/cluster/scheduler.py) — under <campaign>/:
#:                pidfile · status (status.json) · event ledger. One campaign spans several hands, so no hand axis.
#:                No training results here (they go to the bundle segment — hand root).
#:   exp_v2       shared V2 dashboard. No hand axis, but **only the axis hand (AXIS_HAND)** writes it
#:                (writer="axis") — the guard kills a run of another hand that tries to write this layer. Moving the
#:                folder (hands/<AXIS_HAND>/…) is on hold: 423 tracked files and published URLs depend on it.
RUN_SHARED: dict[str, tuple[str, str]] = {
    "exp_shared": ("records", "any"),
    "exp_v2": ("records/v2_dashboard", "axis"),
    "run_rl_loops": ("overfit/rl_loops", "any"),
    # text track state — per track track.json · track_state.json · rounds/ · final/ · ledger.json. Clip text is
    # hand-free, so no hand axis. Per-dataset outputs go to the data layer collect_text.
    "run_collect_text": ("overfit/collect_text/{track}", "any"),
}

#: The **work places** of RL campaign runs — the scripts/stage3_rl.sh contract (where logs and checkpoints are written
#: while training and judging run). No hand axis, and not a run layer: `run_literal_roots` does not read this table
#: (scripts/stage3_rl.sh and dozens of tools use these paths as an older contract, so the canon path check does not
#: cover them). Python campaign code reads this table only, and the move guard accepts
#: sources only **inside** these two — once judging ends, the campaign finalize step moves them from here into the
#: bundle segment. Trainer checkpoint area (relative to the run tree — dexengine.core.paths.run_root, the data root for
#: a release or copied data root): run folders <RUN>/checkpoints · resume parts <RUN>_R* · judge aliases. The one owner
#: of the name — the trainer (rl.train.env_build), the launcher (rl.launch.records.run_dir), the judge sidecar and the
#: skrl run directory reach it through paths.work_area("work_ckpt") (`paths --what work-area --area work_ckpt`;
#: `--what run-ckpt-root` prints this name).
RUN_CKPT_ROOT = "logs/skrl/dexengine"
RUN_WORK_AREAS: dict[str, str] = {
    "work_log": "overfit/logs/refbase",  # training / judge logs · training manifest <RUN>_cmd.txt
    "work_ckpt": RUN_CKPT_ROOT,
}
#: Default folder of the launcher's render and replay videos and of the trainer's own video outputs, relative to the
#: run tree (paths.run_out). Not a work area: finalize never imports from it.
RUN_RENDER_OUT = "results/taco/videos"

_RUN_PH = re.compile(r"\{([a-z]+)\}")


#: **Provisional hand mark** — outputs of a hand with a judge carry-over declaration (`run_hand_carry`) are
#: non-canonical (a reduced version that carried over another generation's verdicts:
#: provisional · not for release · expires when that hand is judged in full). The mark text and its reading have one
#: owner, here — writers of outputs (ledger fold · clip fate · RL bundles · checkpoints · collection) read
#: hand_provisional and carry the same text (no copies). Release and export guards read this function.
PROVISIONAL_MARK = "provisional · non-canonical · do not publish"
#: **Mark file name** placed in a provisional hand's output folders — content = the statement hand_provisional returns
#: (hand · from · mark · expiry · declaration place). When the hand becomes canonical (the declaration is removed) the
#: writer deletes this file too. Collect finals keep it under `meta/` (a final's top level allows LeRobot-style names
#: only — writer dexengine/record/store/scene_layout.py).
PROVISIONAL_FILE = "PROVISIONAL.json"
#: Canonical-confirmation key in the carry-over declaration header — value = the decision (who · when · why). If
#: present the hand is not provisional (hand_provisional).
CANON_KEY = "canon"


def hand_provisional(hand: str, root: str) -> dict | None:
    """Is a hand provisional (has a carry-over declaration)? None if not; else {hand, from, mark, expiry, file}.
    If the declaration header holds a canonical confirmation (CANON_KEY — who · when · why), the carried verdicts are
    that hand's canon → None (the declaration stays as a provenance record).
    Without a canonical confirmation, a missing mark (PROVISIONAL_MARK), expiry or source dies (so an unmarked reduced
    version never flows on as if canonical)."""
    p = os.path.join(root, render_run("run_hand_carry", hand))
    if not os.path.isfile(p):
        return None
    head: dict[str, str] = {}
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.startswith("#"):
                break
            k, _, v = ln[1:].strip().partition("=")
            head[k.strip()] = v.strip()
    if head.get(CANON_KEY):
        if not head.get("from") or head.get("mark") or head.get("expiry"):
            raise SystemExit(
                f"[PROVISIONAL] ✗ {p}: the canonical confirmation ({CANON_KEY}) declaration has no source "
                f"(from) or still carries the provisional mark · expiry (read {head}) — a canonical declaration keeps "
                f"`from` and drops `mark` and `expiry`"
            )
        return None
    if head.get("mark") != PROVISIONAL_MARK or not head.get("expiry") or not head.get("from"):
        raise InputMissingError(
            f"[PROVISIONAL] ✗ {p}: the carry-over declaration header lacks mark="
            f"{PROVISIONAL_MARK!r} · expiry · from (read {head}) — an unmarked reduced version is "
            "not used"
        )
    return {"hand": hand, "from": head["from"], "mark": PROVISIONAL_MARK, "expiry": head["expiry"], "file": p}


def render_run(layer: str, hand: str = "", **axes: str) -> str:
    """Repository-relative path of one run layer. A hand layer requires `hand`; a hand-free layer (RUN_SHARED) does
    not accept one.

    · An unknown layer or a missing axis raises KeyError (no default is invented).
    · An axis value with `/`, `..` or an empty string raises ValueError — one axis eating two path segments would
      leak into another hand's root.
    Hand ids are checked by dexengine.core.paths.run_path.

    >>> render_run("run_chunk", "wuji", tag="G1v_taco_00")
    'overfit/hands/wuji/chunks/G1v_taco_00'
    >>> render_run("exp_shared")
    'records'
    """
    if layer in RUN_SHARED:
        if hand:
            raise ValueError(
                f"{layer} is a hand-free layer — do not pass a hand ({hand!r}) (writer = {RUN_SHARED[layer][1]})"
            )
        tpl = RUN_SHARED[layer][0]
    elif layer in RUN_LAYERS:
        tpl = RUN_LAYERS[layer]
        if not hand:
            raise KeyError(f"{layer}: a hand layer — hand is required (no default)")
        if hand.startswith("_"):
            raise ValueError(f"{hand!r} is not a hand id")
    else:
        raise KeyError(f"unknown run layer {layer!r} — registered: {sorted(RUN_LAYERS) + sorted(RUN_SHARED)}")
    vals = dict(axes, hand=hand)
    for k, v in vals.items():
        if k == "hand" and layer in RUN_SHARED:
            continue
        if not isinstance(v, str) or not v or "/" in v or v in (".", "..") or "\x00" in v:
            raise ValueError(
                f"{layer}: axis {k}={v!r} — no empty value, path separator or '..' (one axis = one path segment)"
            )
    need = set(_RUN_PH.findall(tpl))
    missing = need - {k for k, v in vals.items() if v}
    if missing:
        raise KeyError(f"{layer}: axes {sorted(missing)} are required (template {tpl})")
    extra = set(axes) - need
    if extra:
        raise KeyError(f"{layer}: axes {sorted(extra)} are not in this layer (template {tpl}) — not dropped silently")
    out = tpl.format(**{k: v for k, v in vals.items() if k in need})
    if layer in RUN_LAYERS and (hit := _run_overlap(layer, hand, out)):
        raise ValueError(
            f"{layer}: {out} is a fixed segment of the other layer {hit} — an axis value equals that "
            f"segment name (two layers on one place)"
        )
    return out


def _run_head(tpl: str, hand: str) -> str:
    """Fixed head of a hand-layer template — the hand filled in, up to the segment holding the first remaining
    placeholder (all of it if none)."""
    t = tpl.replace("{hand}", hand)
    return t if "{" not in t else t.split("{", 1)[0].rsplit("/", 1)[0]


def _run_overlap(layer: str, hand: str, out: str) -> str:
    """The name of another hand layer when the rendered place `out` falls inside that layer's fixed segment (its head)
    while this layer's head is outside it — an axis value equal to a sibling layer's fixed segment name, so two layers
    share one place (e.g. a run_collect label = run_collect_work's fixed segment). Overlap under the same head
    (run_chunk ⊂ run_chunk_try's head) is designed nesting, not this."""
    own = _run_head(RUN_LAYERS[layer], hand)
    for k, t in RUN_LAYERS.items():
        h = _run_head(t, hand)
        if k != layer and (out == h or out.startswith(h + "/")) and not (own == h or own.startswith(h + "/")):
            return k
    return ""
