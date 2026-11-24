"""The stage-1 step table: what each step runs and how its progress is counted.

Each row of ``STAGES`` is ``(name, cell, interpreter slot, command)``. The name's first token (``2``, ``2b`` ...) is
the step id. The command is a ``Run`` (the argv this package drives) or a shell line printed as guidance. The
interpreter slot names the interpreter: ``retarget`` = the retargeting venv (``paths.retarget_py``), ``""`` = a shell
entry point that picks its own.

Cells (how a step's progress is counted):

- ``_COVER``: parent takes retargeted with the current MANO frame rule / parent takes the base layer needs (step 1 is
  counted in takes, every later step in segments).
- ``_KEYED + text``: trees whose ``task_info.json`` contains ``text`` (the step's own marker). With ``Run.every`` every
  base tree must carry it (cell x/y); otherwise any tree counts (cell = count, 0 = empty).
- ``_ASSIGN_KEYED``: like ``_KEYED`` with the hand's case-assignment version as the text.
- ``_INPUT_FP + module``: trees whose input fingerprint equals the current inputs / all trees; the judgement is the
  producer module's ``freshness(ds, spec_dir)`` (and ``tree_fresh(ds, spec_dir, tree)`` per tree).
- ``_SIDECAR_FP + layer``: (clip, side) contact sidecars whose fingerprint matches this hand's base / expected.
- ``_F0_JUDGED``: clips the frame-0 pre-judge has settled / clips of the base layer.
- ``_POOL`` / ``_UNI`` / ``_TREES``: IK pool sides, one-hand scene templates, baked trees (per combo, not per dataset).

``Run`` fields: ``axis`` ("" once per dataset, "emb" per embodiment, "emb_side" per embodiment and arm side);
``argv`` with slots ``{ds} {hand} {emb} {side}`` and path tokens ``@base @wrist @clips @f0`` (resolved by
``paths.layer_path`` and the f0 axis); ``rule`` = a rule id whose dataset scope is the step's scope (outside it the step
is "not applicable", shown as ``-``); ``scene`` = prefix the dataset's judge scene (``DEXENGINE_SCENE``) for steps that
read the desk; ``reuse`` = a per-tree reuse command sent instead of ``argv`` when the caller says the rerun has the same
identity; ``every`` = see ``_KEYED``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

from dexengine.data import rules as RR
from dexengine.data.artic.convert_version import D1G_VERSION as _D1G_VERSION
from dexengine.data.cases import assign_version as _AR
from dexengine.data.desk_penetration import D24_VERSION as _D24_VERSION

REPO = Path(__file__).resolve().parents[3]

_COVER = "@parent-take-coverage"
_KEYED = "@task-info-contains:"
_ASSIGN_SLOT = "{assign_version_of_hand}"
_ASSIGN_KEYED = _KEYED + _ASSIGN_SLOT
_TREES = "@trees-layer"
_POOL = "@ikpool"
_UNI = "@uni-template"
_SIDECAR_FP = "@sidecar-fp:"
_INPUT_FP = "@input-fp:"
_F0_JUDGED = "@f0-judged"

Run = namedtuple("Run", "axis argv rule scene reuse every", defaults=(None, False, (), False))

STAGES = (
    (
        "1 retarget",
        _COVER,
        "",
        "python scripts/stage1_retarget.py --arm <arm> --hand <hand> --dataset taco --emb bimanual"
        " --bases <list>   # TACO · other datasets: python3 -m dexengine.retarget.reretarget <ds>",
    ),
    # The base layer is current when every tree records the current adapter version (base_layer.stale.freshness).
    (
        "2 base layer",
        _INPUT_FP + "dexengine.data.base_adapter",
        "retarget",
        Run(
            "",
            ("dexengine/data/base_adapter.py", "--dataset", "{ds}", "--hand", "{hand}"),
            reuse=("dexengine/data/base_adapter.py", "--dataset", "{ds}", "--hand", "{hand}", "--skip-existing"),
        ),
    ),
    # 2b and 2c are dataset-specific base-layer rules. They run after the base layer and before cases (3): D1-g changes
    # the object list (a part pair becomes one articulated object) and 3, 4, 4b and 7 read that list. Both are
    # idempotent. 2b first builds or checks the articulation spec (artic.spec_stage.ensure) so that more hands never
    # move the joint ranges; it uses a GPU (first device of DEXENGINE_F0_GPUS) when assets must be baked.
    (
        "2b articulated D1-g",
        _KEYED + json.dumps(_D1G_VERSION),
        "retarget",
        Run("emb", ("dexengine/data/oakink2/artic_convert.py", "--emb", "{emb}", "--write"), rule="D1-g"),
    ),
    (
        "2c joint value D25",
        _KEYED + '"D25"',
        "retarget",
        Run("emb", ("dexengine/data/artic/arctic_align.py", "--emb", "{emb}", "--write"), rule="D25"),
    ),
    # 2d: the all-object contact sidecar read by cases (3), checked against this hand's base and rebuilt when it
    # differs. The layer is hand independent but built from one hand's object poses.
    (
        "2d contact sidecar (all objects)",
        _SIDECAR_FP + "contact_all",
        "retarget",
        Run("", ("dexengine/data/contact/sidecar.py", "--dataset", "{ds}", "--hand", "{hand}", "--all-objects")),
    ),
    # 2e: wrist tilt repair of references whose wrist housing enters the desk (placement.repair.wrist_desk_stage).
    # After 2d (it reads contact_all), before 3 (cases, roles, quality, wrist cache and f0 read the repaired hand).
    (
        "2e wrist tilt",
        _INPUT_FP + "dexengine.placement.repair.wrist_desk_stage",
        "retarget",
        Run(
            "",
            ("dexengine/placement/repair/wrist_desk_stage.py", "--dataset", "{ds}", "--hand", "{hand}", "--write"),
            rule="WRIST_SLACK",
            scene=True,
        ),
    ),
    # 3 and 4 write into the base trees; they count by their own marker on every tree (the base adapter rewrites a
    # tree folder whole and the long-clip split drops `case`/`rules`, so a rerun upstream clears the markers).
    (
        "3 cases",
        _ASSIGN_KEYED,
        "retarget",
        Run("", ("dexengine/data/cases/taxonomy.py", "--dataset", "{ds}", "--write"), every=True),
    ),
    (
        "4 roles D23",
        _KEYED + json.dumps(_D24_VERSION),
        "retarget",
        Run("", ("dexengine/data/object_roles.py", "--datasets", "{ds}", "--write"), every=True),
    ),
    # 4b: quality flags and the training-candidate flag (filtering = dexengine.data.filtering). After roles (it reads
    # the main object), before f0 (its input key reads the task_info written here). Counted by input fingerprint.
    (
        "4b quality D12-c",
        _INPUT_FP + "dexengine.data.filtering",
        "retarget",
        Run("", ("dexengine/data/filtering.py", "--dataset", "{ds}", "--hand", "{hand}")),
    ),
    # 5: the main-object contact sidecar (contact_base) read by the bake and D26. After roles (4).
    (
        "5 contact sidecar (main object)",
        _SIDECAR_FP + "contact_base",
        "retarget",
        Run("", ("dexengine/data/contact/sidecar.py", "--dataset", "{ds}", "--hand", "{hand}")),
    ),
    # 5b: the support gap of aerial-scene trees, measured once and stamped (the candidate filter only reads the stamp).
    (
        "5b support gap",
        _INPUT_FP + "dexengine.data.support.gap",
        "retarget",
        Run("", ("dexengine/data/support/gap.py", "--dataset", "{ds}", "--hand", "{hand}", "--write")),
    ),
    # 6: the wrist pose cache, once per arm side, from each tree's own body_positions. `--skip_fresh` rewrites only
    # caches whose value changed (unchanged bytes keep the f0 key).
    (
        "6 wrist cache",
        _INPUT_FP + "dexengine.placement.fk.wrist_poses",
        "retarget",
        Run(
            "emb_side",
            (
                "dexengine/placement/fk/wrist_poses.py",
                "--data_dir",
                "@base",
                "--output_dir",
                "@wrist",
                "--side",
                "{side}",
                "--skip_fresh",
            ),
        ),
    ),
    # 7: the frame-0 pre-judge. Its clip list is counted from the base layer (one name per line; names contain
    # spaces). Write location and scene come from the same function as its cell (rules.f0_axis).
    (
        "7 f0 shadow",
        _F0_JUDGED,
        "retarget",
        Run(
            "emb",
            (
                "dexengine/retarget/f0/prejudge.py",
                "--hand",
                "{hand}",
                "--dataset",
                "{ds}",
                "--emb",
                "{emb}",
                "--clips",
                "@clips",
                "--f0-root",
                "@f0",
            ),
            scene=True,
        ),
    ),
    # 8 and 8b are per combo (arm + hand), the same for every dataset (marked †).
    (
        "8 IK pool†",
        _POOL,
        "",
        "bash scripts/build_reach_map.sh <arm> <hand> [GPU] [workers]   # ~94 MB per side · once per combo · "
        "when the pool"
        " exists and only the clearance sidecar's ruler/shape differs: python3 "
        "dexengine/placement/reach_map/clearance.py"
        " --arm <arm> --hand <hand> --force",
    ),
    (
        "8b one-hand scene templates†",
        _UNI,
        "",
        "python3 dexengine/placement/uni_layout.py --extract-for <hand>   # once per hand · after the "
        "stage-1 base layer",
    ),
    (
        "9 trees (judge · bake)",
        _TREES,
        "",
        "bash scripts/stage2_ground.sh --arm <arm> --hand <hand> --dataset <ds> --emb <emb> --clips <list> --fold "
        "$(python3 -m dexengine.data.rules --launch-flags <ds>)",
    ),
)

#: Interpreter slot -> `paths --what` value.
_PY_SLOT = {"retarget": "retarget-py"}

#: Path tokens as the printed guidance shows them (a person can paste the command).
_DOC_TOKEN = {
    "@base": "$(python3 dexengine/core/paths.py --what layer --dataset <ds> --hand <hand> --emb <emb> --layer base)",
    "@wrist": "$(python3 dexengine/core/paths.py --what layer --dataset <ds> --hand <hand> --emb <emb>"
    " --layer wrist_base)/wrist_poses_<side>",
    "@clips": "<base-layer clip list file -- written by --emit-run>",
    "@f0": "$(python3 -m dexengine.data.rules --f0-axis <ds> | cut -f2,3 --output-delimiter=/)",
}
_DOC_SCENE = "DEXENGINE_SCENE=$(python3 -m dexengine.data.rules --f0-axis <ds> | cut -f1) "
_DOC_AXIS = {"": "", "emb": "       # once per emb", "emb_side": "       # once per emb x side (right, left)"}


def step_id(stage) -> str:
    """The step id of a STAGES row (first token of its name)."""
    return stage[0].split()[0]


def py(what: str) -> str:
    """The interpreter `paths --what <what>` names; an empty answer is fatal (the command would be malformed)."""
    r = subprocess.run(
        [sys.executable, str(REPO / "dexengine/core/paths.py"), "--what", what],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit(
            f"[hand_pipeline] paths.py --what {what} failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout).strip().splitlines()[-1:] or ['(no output)']}"
        )
    return r.stdout.strip()


def cell_needle(pat: str, hand_dir: str) -> bytes:
    """The bytes a `_KEYED` cell looks for; the assignment-version slot becomes that hand's version."""
    needle = pat[len(_KEYED) :]
    return (json.dumps(_AR.hand_version(hand_dir)) if needle == _ASSIGN_SLOT else needle).encode()


def rule_scope(cmd) -> tuple:
    """The datasets a step applies to (empty = all). A rule id missing from the registry is fatal."""
    rid = getattr(cmd, "rule", None)
    if not rid:
        return ()
    hit = [r for r in RR.RULES if r.id == rid]
    if len(hit) != 1:
        raise SystemExit(f"[hand_pipeline] step rule {rid!r} appears {len(hit)} times in the rule registry")
    return tuple(hit[0].datasets)


def applies(cmd, ds: str) -> bool:
    """Does the step apply to dataset `ds` (its rule's scope)?"""
    sc = rule_scope(cmd)
    return not sc or ds in sc


def every_tree(pat: str, cmd) -> bool:
    """Must the step's marker be on every base tree (``Run.every``)? Only `_KEYED` cells may say so."""
    ev = bool(getattr(cmd, "every", False))
    if ev and not pat.startswith(_KEYED):
        raise SystemExit(f"[hand_pipeline] Run.every is only for `{_KEYED}` cells -- cell {pat!r}")
    return ev


def tree_unit(pat: str, cmd) -> bool:
    """Is the step's cell counted per base tree (input-fingerprint cells and every-tree marker cells)?"""
    return pat.startswith(_INPUT_FP) or every_tree(pat, cmd)


def embs(ds: str) -> tuple[str, ...]:
    """The dataset's embodiments (rule-registry profile); an unknown dataset raises KeyError."""
    return RR.DATASET_PROFILE[ds].embs


def emb_sides(emb: str) -> tuple[str, ...]:
    """The arm sides an embodiment uses: bimanual = both, right/left = that side."""
    return ("right", "left") if emb == "bimanual" else (emb,)


def cmd_of(stage) -> str:
    """A command line for the step that can be pasted into a shell (generated from the Run argv)."""
    slot, cmd = stage[2], stage[3]
    if isinstance(cmd, Run):
        body = " ".join(_DOC_TOKEN.get(a, a) for a in cmd.argv)
        for k, v in (("{ds}", "<ds>"), ("{hand}", "<hand>"), ("{emb}", "<emb>"), ("{side}", "<side>")):
            body = body.replace(k, v)
        env = _DOC_SCENE if cmd.scene else ""
        return env + (f"{py(_PY_SLOT[slot])} {body}" if slot else body) + _DOC_AXIS[cmd.axis]
    if not slot:
        return cmd
    return f"{py(_PY_SLOT[slot])} {cmd}"
