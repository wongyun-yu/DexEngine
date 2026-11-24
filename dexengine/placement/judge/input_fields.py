"""Vocabulary of the judge input record (input_record): field names, stage keys, separators, the lazy module getters
and the per-process cache. A leaf module — input_run, input_clip and input_record import it."""

from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "jfp3"  # 3: objcol field · supports (~SUP~) in the setup field
#: Ledger column that carries the record.
COLUMN = "judge_input"
#: Judge output file prefix: <out>/<FILE_PREFIX><combo>.tsv (clip \t record).
FILE_PREFIX = "judge_input_"
#: Field order of the record.
FIELDS = ("rs", "rules", "budget", "collider", "setup", "case", "contact", "base", "objcol")
#: Run-level fields (measured once per run).
RUN_FIELDS = ("rs", "rules", "budget", "collider")
#: Judge flag of the arm-flip rule (its default is owned by arm_flip.JUDGE_GATE_DEFAULT).
ARM_FLIP_FLAG = "--arm-flip"
#: Clip-level fields (cacheable by the input files' identity).
CLIP_FIELDS = ("setup", "case", "contact", "base", "objcol")
#: Desk fields of the base tree's `support` that the judge and the bake read as values (qpos_layout.desk_z_of →
#: coords.scene_z_offset). Not a record field: train_ready compares adopted trees with the current base on these.
DESK_READ = ("desk_z",)

#: Stage keys inside the rules field, in the order a clip passes them (judge → bake → gate). The bake and gate
#: thresholds also change judge results (GUARD_FAIL), so each gets its own hash.
STAGES = ("j", "b", "g")
JUDGE_STAGE = STAGES[0]
#: Pipeline position of each stage token of the rules registry (a rule's stage statement split on `·`) → the stage
#: hashes the rule enters. Classified by position, not by name:
#:   1. a stage before the judge that builds judge input no other field hashes → j. The f0 shadow runs before the judge
#:      and both judge and bake read it → j and b.
#:   2. judge · bake (stamp, entry, post gate) · final guard · gate → that stage.
#:   3. earlier stages other fields already carry (base, readers → base field; cases → case field; first-pass
#:      candidates → re-filtered by dexengine.placement.train_ready) and stages that do not change judge results → none.
#: Matching (stage_of): the whole token, else the head of a parenthesised tail `<head>(…)`, else it dies. No prefix
#: matching. A new stage name in the registry must get its position here (the keys are the registry's stage tokens).
STAGE_POSITION = {
    "bake(f0)": ("j", "b"),
    "judge": ("j",),
    "bake": ("b",),
    "bake entry": ("b",),
    "post-bake gate": ("b",),
    "final guard": ("b",),
    "gate": ("g",),
    "base": (),
    "readers": (),
    "source readers": (),
    "quality(4b)": (),
    "cases": (),
    "candidates(first pass)": (),
    "env": (),
    "visualization": (),
    "trainable": (),
    "training set selection": (),
    "fold": (),
    "assets": (),
    "hand registration": (),
    # stages after the judge (bundle cutting, RL evaluation) — they do not change judge results
    "bundling": (),
    "RL eval": (),
    # placement filter = rules inside the judge that only remove candidate cells (the place rule): not in the record.
    # Adopted rows are re-measured on the baked tree (dexengine.placement.train_ready --stale-out); a rejection proved
    # by exhaustion stays a rejection with fewer cells. Putting it in would make every desk-scene row stale.
    "placement filter": (),
}
_PAREN_TAIL = re.compile(r"(.+?)\(.*\)")
#: Separator of the excluded-disc list inside the setup field (disjoint from `|`, `=`, `;`, `,`, `:`).
D40_SEP = "~D40~"
#: Separator of the support-object list inside the setup field (after the discs, before the exclusions).
SUP_SEP = "~SUP~"
#: Rounding of setup values (m) — 1/10 of the comparison tolerance (coords.SETUP_STALE_TOL_M = 1 mm).
SETUP_ROUND = 4
#: Monotone fields: a stored value ≥ the current value counts as the same.
MONOTONE_FIELDS = ("budget",)
ERR = "error:"

#: Per-process cache (run fields, rules, object digests).
_CANON: dict = {}


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()[:12]


# ── lazy module getters (the training entry imports the record before the simulator starts) ─────────────────────
def _coords():
    from dexengine.placement.coords import core as C  # noqa: PLC0415

    return C


def _idle_arm():
    from dexengine.placement.collision import idle_arm as IA  # noqa: PLC0415 — owner of active_idle_for_tree

    return IA


def _ql():
    from dexengine.core import qpos_layout  # noqa: PLC0415

    return qpos_layout


def _rr():
    from dexengine.data import rules  # noqa: PLC0415 — the rules registry

    return rules


def _csp():
    from dexengine.placement import search_policy  # noqa: PLC0415 — dz layers and search origin

    return search_policy


def _ar():
    from dexengine.data.cases import assign_version  # noqa: PLC0415 — key of the assignment record

    return assign_version


def _sf():
    return _coords()._sf_mod()  # sidecar fingerprint owner — the instance coords uses (shared cache)


def _objhull():
    from dexengine.data.objects import hull as OH  # noqa: PLC0415 — object collision shapes (one module, one cache)

    return OH
